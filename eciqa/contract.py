"""The single architecture and checkpoint schema supported by this release."""

MODEL_ID = "q60_joint_tile4_no_global_ce"
FORMAT = "eciqa"
FORMAT_VERSION = 1

ARCHITECTURE = {
    "experiment_id": MODEL_ID,
    "data_mode": "native_tiles", "image_size": 224,
    "train_tiles": 4, "eval_tiles": 0,
    "dim": 128, "heads": 4, "layers": [11],
    "use_input": True, "use_qmap": True, "qmap_trainable": True,
    "interaction": "cross", "qmap_fusion": "concat", "pool": "mean",
    "multilayer": False, "readout": "vanilla", "auxiliary": "none",
    "prompt_only": True, "exposure_prompt": False, "self_attention": False,
    "scoring_head": "fused_text_similarity", "quality_loss": "soft_ce",
    "cross_view_attention": True, "use_global_cls": False,
    "cross_output_fusion": "residual",
}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Checkpoint must contain its complete configuration")
    differences = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in ARCHITECTURE.items() if config.get(key) != expected
    }
    if differences:
        raise ValueError(f"Checkpoint is not the released ECIQA architecture: {differences}")
    unsupported = (
        "crossscore_decoder", "dynamic_prompt", "prompt_coupling", "qmap_late",
        "maniqa_tab", "maniqa_swin", "topiq", "factorized_position",
        "activation_stats", "degradation_condition", "remove_source_relation",
    )
    enabled = [key for key in unsupported if config.get(key)]
    if enabled:
        raise ValueError(f"Unsupported architecture options: {enabled}")
    if int(config.get("view_chunk", 4)) <= 0:
        raise ValueError("view_chunk must be positive")
    if int(config.get("cross_query_chunk", 2048)) < 0:
        raise ValueError("cross_query_chunk must be nonnegative")


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected an ECIQA checkpoint dictionary")
    for key, expected in (("format", FORMAT), ("format_version", FORMAT_VERSION),
                          ("model_id", MODEL_ID)):
        if payload.get(key) != expected:
            raise ValueError(f"Expected checkpoint {key}={expected!r}; got {payload.get(key)!r}")
    if payload.get("kind") not in {"inference", "stage2_initialization", "training"}:
        raise ValueError("Checkpoint kind must be inference, stage2_initialization, or training")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("Checkpoint epoch must be a nonnegative integer")
    validate_config(payload.get("config"))
    if not isinstance(payload.get("qmap_metadata"), dict):
        raise ValueError("Checkpoint qmap_metadata must be a dictionary")

