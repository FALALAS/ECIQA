"""Evaluate paired images, reporting raw SRCC/PLCC overall and by exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .metrics import evaluate, resolve_precision
from .train import atomic_json, atomic_predictions, dataset_options, file_sha256, seed_everything, seed_worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, help="Debug subset only; reported as an incomplete evaluation")
    args = parser.parse_args(argv)
    if args.workers < 0 or args.max_samples is not None and args.max_samples <= 0:
        parser.error("workers must be nonnegative and max-samples positive")
    output_dir = Path(args.output_dir).resolve()
    for name in ("metrics.json", "predictions.csv"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to replace existing evaluation artifact: {output_dir / name}")
    from .data import PairDataset, pair_collate
    from .inference import load_model

    device = torch.device(args.device)
    precision = resolve_precision(device, args.precision)
    model = load_model(args.weights, device=device)
    config = model.config
    seed_everything(int(config["seed"]))
    dataset = PairDataset(split=args.split, training=False,
                          **dataset_options(config, args.manifest, args.project_root))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=pair_collate, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(int(config["seed"]) + 1),
                        pin_memory=device.type == "cuda", persistent_workers=False)
    result = evaluate(model, loader, device, precision, args.max_samples)
    report = {"split": args.split, "precision": precision, "expected_count": len(dataset),
              "complete": len(result["predictions"]) == len(dataset), "metrics": result["metrics"],
              "manifest_sha256": file_sha256(args.manifest), "weight_sha256": file_sha256(args.weights),
              "evaluation_tiles": config["eval_tiles"], "tile_policy": "all valid original-resolution tiles",
              "exposure_grouping": "explicit manifest exposure fields, otherwise scene suffix; unknown kept separately",
              "score_meaning": "scene-relative preference score in [0,1]; not globally calibrated MOS"}
    atomic_predictions(output_dir / "predictions.csv", result["predictions"])
    atomic_json(output_dir / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
