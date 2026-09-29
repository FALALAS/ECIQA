"""Portable construction, strict checkpoint loading, and paired-image scoring."""

import contextlib
import hashlib
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from .contract import MODEL_ID, validate_config, validate_payload
from .data import PairDataset
from .model import ExperimentModel, QMapEncoder
from .text_encoder import TextEncoder
from .vendor.clip_model import VPTCLIP
from .vendor.restormer_arch import Restormer


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _depth(state, prefix):
    indices = {int(key[len(prefix):].split(".")[0]) for key in state if key.startswith(prefix)}
    if not indices or indices != set(range(max(indices) + 1)):
        raise ValueError(f"Missing or non-contiguous transformer blocks: {prefix}")
    return len(indices)


class _CheckpointTokens:
    """Use the fixed, already-tokenized templates stored with the weights."""

    def __init__(self, quality_tokens):
        if tuple(quality_tokens.shape) != (3, 2, 77) or quality_tokens.dtype != torch.int64:
            raise ValueError("quality_tokens must be int64 [3, 2, 77]")
        if (quality_tokens < 0).any() or (quality_tokens >= 49408).any():
            raise ValueError("quality_tokens contains an out-of-vocabulary token")
        self.lookup = {}
        for i, prefix in enumerate(("underexposure ", "overexposure ", "")):
            for j, quality in enumerate(("high quality", "low quality")):
                self.lookup[f"X X X X {prefix}{quality}."] = quality_tokens[i, j:j + 1].cpu().clone()

    def tokenize(self, text):
        return self.lookup[text].clone()


def construct_model(state, config, qmap_metadata=None):
    """Construct the released model and require an exact tensor-key/shape match.

    This never downloads backbones or substitutes an initialization for absent
    parameters. The full trained state includes frozen CLIP representations.
    """
    validate_config(config)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected a complete tensor state dictionary")
    required_shapes = {
        "visual.base.conv1.weight": (768, 3, 16, 16),
        "visual.base.positional_embedding": (197, 768),
        "visual.base.proj": (768, 512),
        "visual.base.prompt_embeddings": (1, 100, 768),
        "visual.base.deep_prompt_embeddings": (11, 100, 768),
        "text_encoder.positional_embedding": (77, 512),
        "text_encoder.token_embedding.weight": (49408, 512),
        "text_encoder.text_projection": (512, 512),
        "ctx": (4, 512),
        "relation.0.weight": (128, 256),
        "fusion_projection.weight": (512, 128),
    }
    for key, shape in required_shapes.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"Expected tensor {key} with shape {shape}")
    if any(key.startswith("global_head.") for key in state):
        raise ValueError("This release excludes the separate cX/cY score branch")
    if _depth(state, "visual.base.transformer.resblocks.") != 12 or _depth(state, "text_encoder.transformer.resblocks.") != 12:
        raise ValueError("The released model requires twelve visual and text transformer blocks")
    clip_model = VPTCLIP(
        embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768,
        vision_patch_size=16, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
    ).float()
    student = SimpleNamespace(
        image_encoder=clip_model.visual, text_encoder=TextEncoder(clip_model),
        prompt_learner2=SimpleNamespace(ctx=torch.nn.Parameter(torch.empty_like(state["ctx"]))),
        logit_scale=clip_model.logit_scale,
    )
    if "qmap.patch_embed.proj.weight" not in state:
        raise ValueError("Missing complete qMap encoder state")
    levels = ("encoder_level1", "encoder_level2", "encoder_level3", "latent")
    dim = int(state["qmap.patch_embed.proj.weight"].shape[0])
    blocks = [_depth(state, f"qmap.{level}.") for level in levels]
    heads = [int(state[f"qmap.{level}.0.attn.temperature"].shape[0]) for level in levels]
    architecture = {"dim": dim, "num_blocks": blocks, "heads": heads,
                    "ffn_expansion_factor": 2.66, "bias": False, "LayerNorm_type": "WithBias"}
    for key, value in (qmap_metadata or {}).items():
        if key in architecture and value != architecture[key]:
            raise ValueError(f"qMap metadata disagrees with its tensors: {key}")
    body = Restormer(
        inp_channels=3, out_channels=3, dim=dim, num_blocks=blocks,
        num_refinement_blocks=blocks[0], heads=heads,
        ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
    )
    qmap = QMapEncoder(SimpleNamespace(body=body), dim)
    result = ExperimentModel(student, _CheckpointTokens(state["quality_tokens"]), qmap, config).float()
    result.load_state_dict(state, strict=True)
    result.qmap_metadata = dict(qmap_metadata or architecture)
    result.qmap_architecture = architecture
    return result


def load_model(weights, device="cpu", trainable=False):
    """Load a full ECIQA export without loading unsafe Python checkpoint objects.

    trainable=True preserves the original selective freezing: CLIP backbone
    frozen, visual/text prompts and qMap/new layers trainable. The default
    returns an evaluation model with gradients disabled.
    """
    weights_path = Path(weights).expanduser().resolve()
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device='cpu'")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    validate_payload(payload)
    if "model" not in payload:
        raise ValueError("Checkpoint is missing its complete 'model' state")
    result = construct_model(payload["model"], payload["config"], payload["qmap_metadata"])
    result = result.to(target)
    if trainable:
        result.train()
    else:
        result.eval().requires_grad_(False)
    result.checkpoint_path = str(weights_path)
    result.checkpoint_metadata = {
        key: payload[key] for key in ("epoch", "format", "format_version", "model_id", "kind")
    }
    return result


class _ExplicitPair(PairDataset):
    """Feed two files through the same preprocessing used for evaluation."""

    def __init__(self, source, candidate, config):
        validate_config(config)
        self.files = {"source": Path(source).expanduser().resolve(),
                      "candidate": Path(candidate).expanduser().resolve()}
        for path in self.files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        self.mode, self.training = config["data_mode"], False
        self.seed, self.epoch = int(config.get("seed", 2024)), 0
        self.train_tiles, self.eval_tiles = 4, 0
        self.image_size, self.long_edge = 224, int(config.get("long_edge", 448))
        self._images, self.cache_size = OrderedDict(), 8
        self.rows = [{"image_id": "explicit_pair", "relative_path": "candidate",
                      "input_relative_path": "source", "scene_id": "explicit_pair",
                      "split": "test", "elo_scene_0_100": "0"}]
        self.indices = [0]

    def _path(self, relative_path):
        return self.files[relative_path]


def make_pair_sample(source_path, candidate_path, config):
    """Prepare an aspect-preserving thumbnail plus all original-resolution tiles."""
    return _ExplicitPair(source_path, candidate_path, config)[0]


@torch.inference_mode()
def score_sample(model, sample, precision="fp32"):
    """Score a prepared pair; high-quality probability lies in [0, 1]."""
    device = next(model.parameters()).device
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision is supported here only for CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support bf16")
    transferred = {key: value.to(device) if torch.is_tensor(value) else value
                   for key, value in sample.items()}
    autocast = (contextlib.nullcontext() if precision == "fp32" else
                torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision]))
    previous_mode = model.training
    model.eval()
    try:
        with autocast:
            output = model(transferred)
    finally:
        model.train(previous_mode)
    p_high = float(output["prediction"].float().cpu())
    logits = output["quality_logits"].float().cpu().reshape(-1).tolist()
    quality_logit = float(output["quality_logit"].float().cpu())
    if not all(math.isfinite(value) for value in (p_high, quality_logit, *logits)):
        raise RuntimeError("Nonfinite model output")
    return {
        "p_high": p_high, "score_0_100": 100 * p_high,
        "quality_logit": quality_logit, "logits_high_low": logits,
        "precision": precision, "view_count": int(sample["source"].shape[0]),
        "native_tile_count": int((sample["view_type"] == 1).sum()),
        "full_native_coverage": bool(sample["full_native_coverage"]),
        "source_size": list(sample["source_size"]),
        "candidate_original_size": list(sample["candidate_original_size"]),
        "candidate_resized_to_input": bool(sample["candidate_resized_to_input"]),
    }


def predict_pair(model, source_path, candidate_path, precision="fp32"):
    """Score candidate_path conditioned on its corresponding source_path."""
    return score_sample(model, make_pair_sample(source_path, candidate_path, model.config), precision)
