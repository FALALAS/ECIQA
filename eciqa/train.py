"""Stage-II training for the released 4-tile, cross-view, no-cX/cY model.

Use --init-weights for a new run, or --resume for an epoch-boundary resume with
optimizer, cosine schedule, scaler and random states. A released epoch-15 model
can initialize further fine-tuning, but is not a Stage-I initialization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import signal
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .metrics import (
    autocast_context, direct_quality_logit, evaluate, is_better,
    move_to_device, resolve_precision, scalar,
)

DEFAULT_TRAINING = {
    "epochs": 100, "scheduler_epochs": 100, "eval_interval": 5, "seed": 2024,
    "learning_rate": 1e-5, "new_learning_rate": 1e-4, "qmap_learning_rate": 1e-5,
    "weight_decay": 1e-4, "accumulation": 8, "loader_batch_size": 1,
    "log_interval": 25, "clip_grad": 1.0,
}
PREDICTION_FIELDS = (
    "image_id", "row_index", "scene_id", "content_id", "exposure", "target",
    "prediction", "quality_logit", "target_0_100", "prediction_0_100",
    "full_native_coverage", "candidate_resized_to_input",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        options = {} if binary else {"encoding": "utf-8", "newline": ""}
        with os.fdopen(fd, "wb" if binary else "w", **options) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, payload):
    def write(handle):
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    _atomic_write(path, write)


def atomic_checkpoint(path, payload):
    _atomic_write(path, lambda handle: torch.save(payload, handle), binary=True)


def atomic_predictions(path, rows):
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write(path, write)


def capture_rng_state(generators=None):
    """Only tensors and Python primitives, compatible with weights_only=True."""
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"bit_generator": state[0], "keys": state[1].tolist(), "pos": state[2],
                  "has_gauss": state[3], "cached_gaussian": state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generators": {name: generator.get_state() for name, generator in (generators or {}).items()},
    }


def restore_rng_state(state, generators=None):
    random.setstate(state["python"])
    array = state["numpy"]
    np.random.set_state((array["bit_generator"], np.asarray(array["keys"], dtype=np.uint32),
                         array["pos"], array["has_gauss"], array["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG restoration is unavailable")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    for name, generator in (generators or {}).items():
        generator.set_state(state["generators"][name].cpu())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def dataset_options(config, manifest, project_root):
    return {
        "manifest": manifest, "project_root": project_root, "mode": config["data_mode"],
        "seed": int(config["seed"]), "train_tiles": int(config["train_tiles"]),
        "eval_tiles": int(config["eval_tiles"]), "image_size": int(config["image_size"]),
        "long_edge": int(config["long_edge"]),
    }


def dataset_metadata(rows, manifest, selection_split):
    splits, scenes, contents = {}, {}, {}
    for row in rows:
        split = row["split"]
        splits[split] = splits.get(split, 0) + 1
        scenes.setdefault(split, set()).add(row["scene_id"])
        content = row.get("content_id") or re.sub(
            r"_(?:N1(?:\.5)?|P1(?:\.5)?|low|high|under|over)$", "", row["scene_id"], flags=re.IGNORECASE,
        )
        contents.setdefault(split, set()).add(content)
    return {
        "manifest_sha256": file_sha256(manifest), "split_counts": splits,
        "train_test_exposure_overlap_count": len(scenes.get("train", set()) & scenes.get("test", set())),
        "train_test_content_overlap_count": len(contents.get("train", set()) & contents.get("test", set())),
        "content_identity_rule": "content_id_or_scene_suffix_inference",
        "selection_split": selection_split, "test_used_for_selection": selection_split == "test",
        "target": "elo_scene_0_100 / 100; scene-relative preference, not globally calibrated MOS",
    }


def select_split(rows, requested):
    available = {row["split"] for row in rows}
    if requested == "auto":
        if "val" not in available:
            raise ValueError("Manifest has no 'val' split. For the historical train/test protocol, "
                             "explicitly pass --selection-split test; this selects checkpoints on test labels.")
        return "val"
    if requested not in available:
        raise ValueError(f"Manifest has no {requested!r} split")
    return requested


def build_optimizer(model, config):
    """Exactly the original three parameter groups and AdamW defaults."""
    groups = {name: [] for name in ("pretrained", "qmap", "new")}
    names = {name: [] for name in groups}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parts = name.lower().split(".")
        if any(part in {"qmap", "qmap_encoder"} for part in parts):
            group = "qmap"
        elif "visual" in parts and any(part in {"prompt_projection", "coupling_scale", "row_position_delta", "col_position_delta"} for part in parts):
            group = "new"
        elif any(part in {"visual", "image_encoder", "text", "text_encoder", "clip_model", "ctx", "logit_scale", "prompt_learner2"} for part in parts):
            group = "pretrained"
        else:
            group = "new"
        groups[group].append(parameter)
        names[group].append(name)
    rates = {"pretrained": float(config["learning_rate"]), "qmap": float(config["qmap_learning_rate"]),
             "new": float(config["new_learning_rate"])}
    parameter_groups, metadata = [], {}
    for group, parameters in groups.items():
        metadata[group] = {"learning_rate": rates[group], "numel": sum(p.numel() for p in parameters),
                           "tensor_count": len(parameters), "parameter_names": names[group]}
        if parameters:
            parameter_groups.append({"params": parameters, "lr": rates[group], "name": group})
    if not parameter_groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(parameter_groups, weight_decay=float(config["weight_decay"])), metadata


def make_grad_scaler(device, precision):
    enabled = device.type == "cuda" and precision == "fp16"
    modern = getattr(torch.amp, "GradScaler", None)
    if modern is not None:
        return modern("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def quality_objective(output, target, config=None):
    """Soft CE for [low, high]=[1-y,y], exactly BCE(high_logit-low_logit,y)."""
    if config is not None and config.get("quality_loss", "soft_ce") != "soft_ce":
        raise ValueError("the released baseline uses soft_ce")
    target = scalar(target, "label", target.device).float()
    if not 0 <= float(target.detach()) <= 1:
        raise ValueError("soft CE targets must already be normalized to [0,1]")
    logit = direct_quality_logit(output, target.device)
    if torch.is_grad_enabled() and not logit.requires_grad:
        raise RuntimeError("quality_logit is detached from trainable parameters")
    return F.binary_cross_entropy_with_logits(logit, target)


def apply_optimizer_step(optimizer, scaler, image_count, clip_grad=1.0):
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    scaler.unscale_(optimizer)
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not parameters:
        raise RuntimeError("loss produced no trainable parameter gradients")
    for parameter in parameters:
        parameter.grad.div_(image_count)
        if not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError("nonfinite gradient before optimizer step")
    norm = nn.utils.clip_grad_norm_(parameters, clip_grad, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.item())


def train_epoch(model, loader, optimizer, device, scaler, precision="fp32", accumulation=8,
                clip_grad=1.0, max_optimizer_steps=None, should_stop=None, progress_callback=None, config=None):
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    sums = {name: 0.0 for name in ("loss", "quality_loss", "l1", "aux_loss")}
    images = pending = steps = 0
    norm = None
    limited = False
    for samples in loader:
        for original in samples:
            if should_stop is not None and should_stop():
                raise InterruptedError("stop requested during an incomplete training epoch")
            sample = move_to_device(original, device)
            target = scalar(sample["label"], "label", device).float()
            with autocast_context(device, precision):
                output = model(sample)
                prediction = scalar(output["prediction"], "prediction", device)
                auxiliary = scalar(output.get("aux_loss", 0.0), "aux_loss", device)
                l1 = (prediction.float() - target).abs()
                quality_loss = quality_objective(output, target, config)
                loss = quality_loss + auxiliary.float()
            scalar(loss, "loss", device)
            scaler.scale(loss).backward()
            for name, value in (("loss", loss), ("quality_loss", quality_loss), ("l1", l1), ("aux_loss", auxiliary)):
                sums[name] += float(value.detach())
            images += 1
            pending += 1
            if pending == accumulation:
                norm = apply_optimizer_step(optimizer, scaler, pending, clip_grad)
                pending = 0
                steps += 1
                if progress_callback:
                    progress_callback({"images": images, "optimizer_steps": steps, "loss": sums["loss"] / images})
                if max_optimizer_steps is not None and steps >= max_optimizer_steps:
                    limited = True
                    break
        if limited:
            break
    if pending:
        norm = apply_optimizer_step(optimizer, scaler, pending, clip_grad)
        steps += 1
    if not images:
        raise ValueError("training loader produced no images")
    return {**{key: value / images for key, value in sums.items()}, "quality_loss_name": "soft_ce",
            "images": images, "optimizer_steps": steps, "last_gradient_norm": norm,
            "elapsed_seconds": time.monotonic() - started, "limited_for_smoke": limited}


def validate_config(config):
    for name in ("epochs", "scheduler_epochs", "eval_interval", "accumulation", "loader_batch_size", "log_interval"):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "new_learning_rate", "qmap_learning_rate", "clip_grad"):
        if not math.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(float(config["weight_decay"])) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config.get("quality_loss") != "soft_ce" or config.get("scoring_head") != "fused_text_similarity":
        raise ValueError("the released model requires soft CE with fused text similarity")
    if config.get("auxiliary", "none") != "none":
        raise ValueError("the released baseline has no auxiliary loss")


def run(args):
    from .data import PairDataset, pair_collate
    from .inference import load_model

    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    device = torch.device(args.device)
    precision = resolve_precision(device, args.precision)
    weight_path = Path(args.resume or args.init_weights).resolve()
    payload = torch.load(weight_path, map_location="cpu", weights_only=True)
    config = {**DEFAULT_TRAINING, **payload["config"]}
    if args.config:
        requested = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        for key, value in requested.items():
            if key not in DEFAULT_TRAINING and config.get(key) != value:
                raise ValueError(f"--config cannot change the released architecture/data policy: {key}")
        config.update(requested)
    validate_config(config)
    manifest = Path(args.manifest).resolve()
    project_root = Path(args.project_root).resolve()
    seed_everything(int(config["seed"]))
    train_data = PairDataset(split="train", training=True, **dataset_options(config, manifest, project_root))
    selection_split = select_split(train_data.rows, args.selection_split)
    selection_data = PairDataset(split=selection_split, training=False, **dataset_options(config, manifest, project_root))
    data_info = dataset_metadata(train_data.rows, manifest, selection_split)
    if selection_split == "test":
        warnings.warn("Historical protocol: checkpoints are selected on test labels. These are not untouched-test results.")
    initial_hash = payload.get("initial_weight_sha256") if args.resume else file_sha256(weight_path)
    signature = config_fingerprint({"config": config, "manifest_sha256": data_info["manifest_sha256"],
                                    "selection_split": selection_split, "precision": precision,
                                    "workers": args.workers, "smoke": args.smoke,
                                    "initial_weight_sha256": initial_hash})
    output_dir = Path(args.output_dir).resolve()
    report_path = output_dir / "report.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.resume:
            raise ValueError("output directory is nonempty; use a new directory or --resume")
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("run_signature") != signature:
                raise ValueError("output directory belongs to a different training run")
            if int(previous.get("completed_epochs", 0)) > int(payload.get("epoch", 0)):
                raise ValueError("Resume checkpoint predates this output directory; use a new output directory to branch")
        else:
            raise ValueError("nonempty output directory lacks report.json; refusing to overwrite")
    if args.resume:
        required = {"optimizer", "scheduler", "scaler", "rng_state", "history", "run_signature", "initial_weight_sha256"}
        if payload.get("kind") != "training" or payload.get("epoch_complete") is not True or not required.issubset(payload):
            raise ValueError("--resume requires a complete training checkpoint, not an inference/initialization export")
        if payload["run_signature"] != signature:
            raise ValueError("resume configuration/manifest/selection/precision/workers/smoke policy differs")
    generators = {"train": torch.Generator().manual_seed(int(config["seed"])),
                  "test": torch.Generator().manual_seed(int(config["seed"]) + 1)}
    loader_options = {"batch_size": int(config["loader_batch_size"]), "num_workers": args.workers,
                      "collate_fn": pair_collate, "worker_init_fn": seed_worker,
                      "pin_memory": device.type == "cuda", "persistent_workers": False}
    if args.workers:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(train_data, shuffle=True, generator=generators["train"], **loader_options)
    selection_loader = DataLoader(selection_data, shuffle=False, generator=generators["test"], **loader_options)
    model = load_model(weight_path, device=device, trainable=True)
    model.config = config
    optimizer, groups = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["scheduler_epochs"]))
    scaler = make_grad_scaler(device, precision)
    history, best, completed_epoch = [], None, 0
    initialization_kind = payload.get("kind", "unspecified")
    if args.resume:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload["rng_state"], generators)
        history, best, completed_epoch = payload["history"], payload.get("best"), int(payload["epoch"])
        initial_hash = payload["initial_weight_sha256"]
        initialization_kind = payload.get("initialization_kind", "unspecified")
    elif int(config["seed"]) != int(payload["config"].get("seed", 2024)):
        # An explicitly changed seed must not be overwritten by the exported
        # baseline's post-initialization random stream.
        seed_everything(int(config["seed"]))
    elif payload.get("training_rng_state"):
        restore_rng_state(payload["training_rng_state"])
    elif initialization_kind == "stage2_initialization":
        warnings.warn("Initialization has no post-construction RNG state; exact historical random-stream reproduction is unavailable.")
    else:
        warnings.warn("New fine-tuning run starts from trained weights; this does not reproduce Stage II from Stage I.")
    effective_epochs = 1 if args.smoke else int(config["epochs"])
    if args.stop_after_epoch is not None and args.stop_after_epoch <= completed_epoch:
        raise ValueError("--stop-after-epoch must be later than the resumed checkpoint epoch")
    report = {
        "status": "running", "created_at": payload.get("created_at", utc_now()) if args.resume else utc_now(),
        "updated_at": utc_now(), "config": config, "run_signature": signature, "data": data_info,
        "initialization_kind": initialization_kind, "initial_weight_sha256": initial_hash,
        "requested_epochs": int(config["epochs"]), "effective_epochs": effective_epochs,
        "completed_epochs": completed_epoch, "best": best, "history": history,
        "selection_metric": "mean(srcc,plcc); tie srcc then earliest epoch",
        "selection_split": selection_split, "precision": precision, "workers": args.workers,
        "smoke": args.smoke, "parameter_groups": groups,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_count": sum(p.numel() for p in model.parameters()), "plcc_mapping": "none_raw_pearson",
        "training_objective": "image-level soft CE; [low,high]=[1-y,y]; y=elo_scene_0_100/100",
    }
    metadata = {key: payload[key] for key in ("format", "format_version", "model_id", "qmap_metadata")}
    metadata.update(kind="training", config=config, run_signature=signature,
                    initial_weight_sha256=initial_hash, initialization_kind=initialization_kind,
                    data=data_info, precision=precision, smoke=args.smoke, created_at=report["created_at"])
    del payload
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, report)
    stop = {"requested": False}
    previous_handlers = {}

    def stop_requested(_signum, _frame):
        stop["requested"] = True

    def checkpoint(epoch):
        return {**metadata, "epoch": epoch, "epoch_complete": True, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng_state": capture_rng_state(generators),
                "history": history, "best": best}

    # Epoch zero provides a recovery point even if the first epoch is interrupted.
    if not args.resume:
        atomic_checkpoint(output_dir / "last.pt", checkpoint(0))
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, stop_requested)
    try:
        for epoch in range(completed_epoch + 1, effective_epochs + 1):
            if stop["requested"]:
                raise InterruptedError("stop requested at epoch boundary")
            train_data.set_epoch(epoch)

            def progress(info):
                if info["optimizer_steps"] % int(config["log_interval"]) == 0:
                    print(json.dumps({"event": "train_progress", "epoch": epoch, **info}), flush=True)

            train_result = train_epoch(
                model, train_loader, optimizer, device, scaler, precision, int(config["accumulation"]),
                float(config["clip_grad"]), 2 if args.smoke else None, lambda: stop["requested"], progress, config,
            )
            scheduler.step()
            record = {"epoch": epoch, "train": train_result, "timestamp": utc_now(),
                      "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}}
            evaluation = None
            improved = False
            if args.smoke or epoch % int(config["eval_interval"]) == 0 or epoch == effective_epochs:
                evaluation = evaluate(model, selection_loader, device, precision, 8 if args.smoke else None,
                                      should_stop=lambda: stop["requested"])
                current = {**evaluation["metrics"]["overall"], "epoch": epoch}
                record.update(evaluation=current, by_exposure={key: value for key, value in evaluation["metrics"].items() if key != "overall"})
                improved = is_better(current, best)
                if improved:
                    best = current
                atomic_predictions(output_dir / "last_predictions.csv", evaluation["predictions"])
                atomic_json(output_dir / "last_metrics.json", evaluation["metrics"])
            history.append(record)
            state = checkpoint(epoch)
            if improved:
                # Best and periodic checkpoints also support an exact resume.
                atomic_checkpoint(output_dir / "best.pt", state)
                atomic_predictions(output_dir / "best_predictions.csv", evaluation["predictions"])
            if evaluation is not None:
                atomic_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", state)
            atomic_checkpoint(output_dir / "last.pt", state)
            del state
            completed_epoch = epoch
            report.update(updated_at=utc_now(), completed_epochs=epoch, best=best, history=history)
            atomic_json(report_path, report)
            print(json.dumps({"event": "epoch_complete", **record}), flush=True)
            if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
                report["status"] = "paused_at_epoch_boundary"
                break
        else:
            report["status"] = "smoke_completed" if args.smoke else "completed"
    except (InterruptedError, KeyboardInterrupt) as error:
        report.update(status="interrupted", error=str(error))
        # A partially updated epoch must never overwrite the complete last.pt.
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        report.update(updated_at=utc_now(), completed_epochs=completed_epoch)
        atomic_json(report_path, report)
    print(json.dumps({"event": report["status"], "report": str(report_path), "best": best}), flush=True)
    return 0 if report["status"] in {"completed", "smoke_completed", "paused_at_epoch_boundary"} else 130


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-weights", help="Portable Stage-I initialization, or trained weights for additional fine-tuning")
    source.add_argument("--resume", help="Complete release training checkpoint, including optimizer and RNG state")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", required=True, help="Explicit root for both relative image paths in the manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", help="Optional JSON training overrides; architecture and tile policy must match the release")
    parser.add_argument("--selection-split", choices=("auto", "val", "test"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Two optimizer steps and up to eight evaluation images; not formal metrics")
    parser.add_argument("--stop-after-epoch", type=int, help="Pause after a complete epoch without changing the cosine schedule")
    args = parser.parse_args(argv)
    if args.stop_after_epoch is not None and args.stop_after_epoch <= 0:
        parser.error("--stop-after-epoch must be positive")
    return args


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
