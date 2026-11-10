"""ECIQA: cross-view paired patch attention without a separate CLS score path.

The module names and arithmetic match the selected training checkpoint. The
CLIP class token still participates inside its transformer, but neither cX
nor cY is concatenated into the final quality representation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .components import CandidateCrossAttention, CheckpointLayerAggregator, PatchQualityPool, QMapTokenFusion
from .contract import validate_config


def split_attention(block, sequence, key_mask=None):
    """CLIP attention with valid-patch masking and the original residual/MLP."""
    normalized = block.ln_1(sequence).transpose(0, 1)
    batch, length, width = normalized.shape
    attn = block.attn
    heads, d = attn.num_heads, width // attn.num_heads
    qkv = F.linear(normalized, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = [x.reshape(batch, length, heads, d).transpose(1, 2) for x in qkv.chunk(3, -1)]
    mask = key_mask[:, None, None, :] if key_mask is not None else None
    update = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    update = update.transpose(1, 2).reshape(batch, length, width)
    update = F.linear(update, attn.out_proj.weight, attn.out_proj.bias).transpose(0, 1)
    output = sequence + update
    return output + block.mlp(block.ln_2(output))


class VisualAdapter(nn.Module):
    """Deep visual prompts and normalized final-layer CLIP patch features."""

    def __init__(self, visual, config):
        super().__init__()
        self.base = visual
        self.layers = tuple(config["layers"])
        self.use_checkpoint = config.get("gradient_checkpointing", True)

    def position(self, h, w):
        position = self.base.positional_embedding
        old = math.isqrt(position.shape[0] - 1)
        spatial = position[1:].reshape(old, old, -1).permute(2, 0, 1).unsqueeze(0)
        if (h, w) != (old, old):
            spatial = F.interpolate(spatial.float(), (h, w), mode="bicubic", align_corners=False)
        return torch.cat((position[:1], spatial[0].permute(1, 2, 0).reshape(h * w, -1)), 0)

    def forward(self, image, patch_mask=None, context=None):
        visual = self.base
        x = visual.conv1(image)
        batch, width, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls = visual.class_embedding.to(x.dtype).expand(batch, 1, -1)
        x = visual.ln_pre(torch.cat((cls, x), 1) + self.position(h, w).to(x.dtype))
        prompts = visual.prompt_embeddings.expand(batch, -1, -1) + 0
        prompts = visual.prompt_dropout(visual.prompt_proj(prompts))
        x = torch.cat((x[:, :1], prompts, x[:, 1:]), 1).transpose(0, 1)
        key_mask = None
        if patch_mask is not None:
            prefix = torch.ones(batch, 1 + visual.num_tokens, device=x.device, dtype=torch.bool)
            key_mask = torch.cat((prefix, patch_mask.reshape(batch, h * w)), 1)
        features = []
        for i, block in enumerate(visual.transformer.resblocks):
            if i > 0:
                p = visual.deep_prompt_embeddings[i - 1].expand(batch, -1, -1) + 0
                p = visual.prompt_dropout(visual.prompt_proj(p)).transpose(0, 1)
                x = torch.cat((x[:1], p, x[1 + visual.num_tokens:]), 0)
            if self.training and self.use_checkpoint:
                x = checkpoint(lambda value, blk=block: split_attention(blk, value, key_mask),
                               x, use_reentrant=False)
            else:
                x = split_attention(block, x, key_mask)
            if i in self.layers:
                features.append(x[-h * w:].transpose(0, 1))
        encoded = visual.ln_post(visual.prompt_norm(x).transpose(0, 1)) @ visual.proj
        raw_global = encoded[:, 0]
        dense = encoded[:, -h * w:]
        return {"global": F.normalize(raw_global, dim=-1), "raw_global": raw_global,
                "dense": F.normalize(dense, dim=-1), "layers": features, "grid": (h, w)}


class QMapEncoder(nn.Module):
    """The four pretrained Restormer encoder levels; no qMap decoder."""

    def __init__(self, original, dim):
        super().__init__()
        body = original.body
        for name in ("patch_embed", "encoder_level1", "down1_2", "encoder_level2",
                     "down2_3", "encoder_level3", "down3_4", "latent"):
            setattr(self, name, getattr(body, name))
        self.widths = [dim * 2**i for i in range(4)]

    def forward(self, image):
        f1 = self.encoder_level1(self.patch_embed(image))
        f2 = self.encoder_level2(self.down1_2(f1))
        f3 = self.encoder_level3(self.down2_3(f2))
        f4 = self.latent(self.down3_4(f3))
        return [f1, f2, f3, f4]


class ExperimentModel(nn.Module):
    """The released four-tile training architecture, scoring one pair at a time."""

    def __init__(self, student, clip_module, qmap, config):
        super().__init__()
        validate_config(config)
        self.config = dict(config)
        self.direct_quality = self.cross_view_attention = True
        self.use_global_cls = False
        self.use_input = self.use_qmap = True
        self.cross_output_fusion = "residual"
        self.cross_query_chunk = int(config.get("cross_query_chunk", 2048))
        self.dim = dim = config["dim"]
        self.visual = VisualAdapter(student.image_encoder, config)
        self.text_encoder = student.text_encoder
        self.ctx = student.prompt_learner2.ctx
        self.logit_scale = student.logit_scale
        # Preserve all three template pairs for strict checkpoint compatibility.
        # Scoring uses only the exposure-free high/low quality pair at index 2.
        templates = []
        for prefix in ("underexposure ", "overexposure ", ""):
            templates.append(torch.cat([clip_module.tokenize(f"X X X X {prefix}{quality}.")
                                        for quality in ("high quality", "low quality")]))
        self.register_buffer("quality_tokens", torch.stack(templates))
        self.qmap = qmap
        self.visual_projection = nn.Linear(512, dim)
        # Keep original inactive tensors for exact strict checkpoint loading.
        self.layer_projections = nn.ModuleList([
            nn.Linear(student.image_encoder.proj.shape[0], dim) for _ in self.visual.layers])
        self.layer_aggregator = CheckpointLayerAggregator(len(self.visual.layers))
        self.qmap_projection = nn.ModuleList([nn.Conv2d(w, dim, 1) for w in qmap.widths])
        self.qmap_fusion = QMapTokenFusion(dim)
        self.cross = CandidateCrossAttention(dim, config["heads"])
        self.relation = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.pool = PatchQualityPool(dim)
        self.position_projection = nn.Linear(3, dim, bias=False)
        self.fusion_projection = nn.Linear(dim, 512)
        self.global_head = None
        self.requires_grad_(True)
        self.visual.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        for name, parameter in self.visual.named_parameters():
            if "prompt" in name:
                parameter.requires_grad_(True)
        self.layer_projections.requires_grad_(False)
        self.layer_aggregator.requires_grad_(False)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def visual_forward(self, rgb, mask):
        return self.visual((rgb - self.image_mean) / self.image_std, mask, self.ctx)

    def text_features(self):
        """Encode learned context plus high/low quality, without exposure prompts."""
        tokens = self.quality_tokens[2]
        embedded = self.text_encoder.token_embedding(tokens)
        prompts = torch.cat((embedded[:, :1], self.ctx[None].expand(2, -1, -1), embedded[:, 5:]), 1)
        return F.normalize(self.text_encoder(prompts, tokens), dim=-1)

    def local_features(self, output, mask):
        value = self.visual_projection(output["dense"])
        value = value.masked_fill(~mask[..., None], 0)
        return value.masked_fill(~mask[..., None], 0)

    def qmap_features(self, rgb, grid):
        maps = self.qmap(rgb)
        projected = [F.adaptive_avg_pool2d(layer(feature), grid)
                     for layer, feature in zip(self.qmap_projection, maps)]
        return torch.stack(projected).mean(0).flatten(2).transpose(1, 2), maps

    def _joint_cross_attention(self, candidate, input, mask, coords):
        views, patches, channels = candidate.shape
        if input.shape != candidate.shape or mask.shape != (views, patches):
            raise ValueError("joint input/candidate views and masks must align")
        if coords.shape != (views, patches, 3):
            raise ValueError("joint coordinates must contain original x, y and scale")
        queries = candidate.reshape(1, views * patches, channels)
        keys = input.reshape(1, views * patches, channels)
        valid = mask.reshape(1, views * patches)
        xy = coords[..., :2].reshape(1, views * patches, 2)
        chunk = self.cross_query_chunk or queries.shape[1]
        updates = []
        for begin in range(0, queries.shape[1], chunk):
            end = min(begin + chunk, queries.shape[1])
            updates.append(self.cross(queries[:, begin:end], keys, valid[:, begin:end], valid,
                                      xy[:, begin:end], xy))
        return torch.cat(updates, dim=1).reshape(views, patches, channels)

    def forward(self, sample):
        source, target = sample["source"], sample["candidate"]
        masks = sample["masks"].flatten(1)
        token_areas = sample["valid_fraction"].flatten(1)
        coords = sample["coords"].flatten(1, 2)
        view_chunk = self.config.get("view_chunk", 4)
        joint_source, joint_target = [], []
        for begin in range(0, target.shape[0], view_chunk):
            end = min(begin + view_chunk, target.shape[0])
            mask = masks[begin:end]
            y = self.visual_forward(target[begin:end], mask)
            x = self.visual_forward(source[begin:end], mask)
            fx, fy = self.local_features(x, mask), self.local_features(y, mask)
            qy, _ = self.qmap_features(target[begin:end], y["grid"])
            qx, _ = self.qmap_features(source[begin:end], x["grid"])
            fx = self.qmap_fusion(fx, qx, mask)
            fy = self.qmap_fusion(fy, qy, mask)
            position = self.position_projection(coords[begin:end].to(fy.dtype))
            fx, fy = fx + position, fy + position
            joint_source.append(fx)
            joint_target.append(fy)
        fx, fy = torch.cat(joint_source), torch.cat(joint_target)
        fy = self._joint_cross_attention(fy, fx, masks, coords)
        tokens = self.relation(torch.cat((fx, fy), -1))
        _, pooled, _ = self.pool(tokens, masks, token_area=token_areas)
        area = sample["valid_fraction"].sum((1, 2)).to(pooled.dtype)
        weights = area / area.sum().clamp_min(1)
        local_feature = (pooled * weights[:, None]).sum(0, keepdim=True)
        fused = F.normalize(self.fusion_projection(local_feature), dim=-1)
        quality_text = self.text_features()
        logits = (self.logit_scale.clamp(max=math.log(100)).exp() * fused @ quality_text.T).float()
        return {"prediction": logits.softmax(-1)[0, 0],
                "quality_logit": logits[0, 0] - logits[0, 1],
                "quality_logits": logits, "fused_embedding": fused,
                "aux_loss": logits.new_zeros(())}


def build_model(weights, device="cpu"):
    """Build a trainable model from the complete Stage-II initialization export."""
    from .inference import load_model
    return load_model(weights, device=device, trainable=True)
