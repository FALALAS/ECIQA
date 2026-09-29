"""Raw correlations and image-level evaluation; no fitted PLCC mapping."""

from __future__ import annotations

import contextlib
import math
import warnings

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def compute_metrics(targets, predictions):
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.ndim != 1 or prediction.shape != target.shape:
        raise ValueError("targets and predictions must be equally sized 1D arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite target or prediction in evaluation")
    srcc = plcc = None
    if target.size >= 2 and np.ptp(target) > 0 and np.ptp(prediction) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            srcc = float(spearmanr(target, prediction).statistic)
            plcc = float(pearsonr(target, prediction).statistic)
        srcc = srcc if math.isfinite(srcc) else None
        plcc = plcc if math.isfinite(plcc) else None
    return {
        "count": int(target.size), "srcc": srcc, "plcc": plcc,
        "selection_score": (srcc + plcc) / 2 if srcc is not None and plcc is not None else None,
        "mae": float(np.abs(prediction - target).mean()) if target.size else None,
        "rmse": float(np.sqrt(np.square(prediction - target).mean())) if target.size else None,
        "score_scale": "0_1", "plcc_mapping": "none_raw_pearson",
    }


def grouped_metrics(rows):
    result = {"overall": compute_metrics([r["target"] for r in rows], [r["prediction"] for r in rows])}
    for exposure in ("under", "over", "unknown"):
        subset = [r for r in rows if r["exposure"] == exposure]
        result[exposure] = compute_metrics([r["target"] for r in subset], [r["prediction"] for r in subset])
    return result


def is_better(candidate, best):
    """Historical rule: mean correlation, then SRCC, then earliest epoch."""
    def key(record):
        if not record:
            return None
        srcc, plcc = record.get("srcc"), record.get("plcc")
        if srcc is None or plcc is None or not math.isfinite(srcc) or not math.isfinite(plcc):
            return None
        return ((srcc + plcc) / 2, srcc, -int(record.get("epoch", 0)))
    left, right = key(candidate), key(best)
    return left is not None and (right is None or left > right)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {name: move_to_device(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def autocast_context(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def resolve_precision(device, precision):
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device.index if device.index is not None else 0)
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("CPU evaluation/training requires fp32 or bf16")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support BF16; use --precision fp32")
    return precision


def scalar(value, name, device):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
    if value.numel() != 1:
        raise ValueError(f"{name} must be a scalar per image")
    value = value.reshape(())
    if not torch.isfinite(value).item():
        raise FloatingPointError(f"nonfinite {name}")
    return value


def direct_quality_logit(output, device):
    if "quality_logit" not in output:
        raise ValueError("soft CE requires the final fused quality_logit")
    logit = scalar(output["quality_logit"], "quality_logit", device).float()
    prediction = scalar(output["prediction"], "prediction", device).float()
    if not 0.0 <= float(prediction.detach()) <= 1.0:
        raise ValueError("prediction must be in [0, 1]")
    if not torch.isclose(prediction.detach(), logit.detach().sigmoid(), rtol=1e-5, atol=1e-6).item():
        raise ValueError("prediction must equal sigmoid(quality_logit)")
    return logit


@torch.inference_mode()
def evaluate(model, loader, device, precision="fp32", max_samples=None, should_stop=None):
    was_training = model.training
    model.eval()
    rows = []
    try:
        for samples in loader:
            for original in samples:
                if should_stop is not None and should_stop():
                    raise InterruptedError("stop requested during evaluation")
                sample = move_to_device(original, device)
                with autocast_context(device, precision):
                    output = model(sample)
                target = float(scalar(sample["label"], "label", device))
                prediction = float(scalar(output["prediction"], "prediction", device))
                rows.append({
                    "image_id": original.get("image_id", ""), "row_index": original.get("row_index", len(rows)),
                    "scene_id": original.get("scene_id", ""), "content_id": original.get("content_id", ""),
                    "exposure": {0: "under", 1: "over"}.get(int(original.get("exposure", 2)), "unknown"),
                    "target": target, "prediction": prediction,
                    "quality_logit": float(direct_quality_logit(output, device)),
                    "target_0_100": target * 100, "prediction_0_100": prediction * 100,
                    "full_native_coverage": bool(original.get("full_native_coverage", False)),
                    "candidate_resized_to_input": bool(original.get("candidate_resized_to_input", False)),
                })
                if max_samples is not None and len(rows) >= max_samples:
                    break
            if max_samples is not None and len(rows) >= max_samples:
                break
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("evaluation loader produced no images")
    return {"metrics": grouped_metrics(rows), "predictions": rows}
