"""Verify local ECIQA release assets without downloading or loading PyTorch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes asset directory: {relative}")
    return path


def verify(root: Path) -> dict:
    errors = []
    checked = {}

    def check(path: Path, expected: str, size=None):
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            errors.append(f"Missing: {relative} (download the assets linked in README.md)")
            return
        actual_size = path.stat().st_size
        if size is not None and actual_size != size:
            errors.append(f"Size mismatch: {relative}")
        digest = checked[relative]["sha256"] if relative in checked else sha256(path)
        if digest != expected:
            errors.append(f"SHA-256 mismatch: {relative}")
        checked[relative] = {"path": relative, "bytes": actual_size, "sha256": digest}

    weights = json.loads((root / "weights/checksums.json").read_text(encoding="utf-8"))
    for entry in weights["files"]:
        check(local_path(root / "weights", entry["filename"]), entry["sha256"], entry["bytes"])
    print(f"Checked {len(weights['files'])} checkpoints.", flush=True)

    dataset = root / "data/ECIQA-Pref"
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    check(dataset / "manifest.csv", metadata["portable_manifest_sha256"])
    check(dataset / "checksums.sha256", metadata["image_checksum_file_sha256"])
    expected_images = {}
    for line in (dataset / "checksums.sha256").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*").replace("\\", "/")
        if relative in expected_images:
            raise ValueError(f"Duplicate checksum path: {relative}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"Invalid SHA-256 value for {relative}")
        expected_images[relative] = digest.lower()
    for i, (relative, digest) in enumerate(expected_images.items(), 1):
        check(local_path(dataset, relative), digest)
        if i % 1000 == 0:
            print(f"Checked {i}/{len(expected_images)} dataset files.", flush=True)

    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"image_id", "scene_id", "content_id", "relative_path", "input_relative_path",
                "elo_scene_0_100", "split", "exposure"}
    if not rows or not required.issubset(fields):
        raise ValueError(f"Manifest missing required fields: {sorted(required - set(fields))}")
    image_ids = [r["image_id"] for r in rows]
    if len(image_ids) != len(set(image_ids)):
        errors.append("Duplicate image IDs in manifest")
    relative_paths = [r["relative_path"].replace("\\", "/") for r in rows]
    if len(relative_paths) != len(set(relative_paths)):
        errors.append("Duplicate candidate paths in manifest")
    referenced = set(relative_paths)
    referenced.update(r["input_relative_path"].replace("\\", "/") for r in rows)
    if referenced != set(expected_images):
        errors.append("Manifest references differ from the checksummed image set")
    scenes = {}
    contents = {}
    input_paths = {}
    row_by_path = dict(zip(relative_paths, rows))
    for row in rows:
        for field in ("relative_path", "input_relative_path"):
            path = local_path(dataset, row[field])
            if not path.is_file():
                errors.append(f"Unresolved manifest image: {row[field]}")
        score = float(row["elo_scene_0_100"])
        if not math.isfinite(score) or not 0 <= score <= 100:
            errors.append(f"Invalid target: {row['image_id']}")
        if row["exposure"] not in {"under", "over"}:
            errors.append(f"Invalid exposure: {row['image_id']}")
        if scenes.setdefault(row["scene_id"], row["split"]) != row["split"]:
            errors.append(f"Exposure instance crosses splits: {row['scene_id']}")
        if input_paths.setdefault(row["scene_id"], row["input_relative_path"]) != row["input_relative_path"]:
            errors.append(f"Exposure instance has inconsistent inputs: {row['scene_id']}")
        source = row_by_path.get(row["input_relative_path"].replace("\\", "/"))
        if source is None or source["scene_id"] != row["scene_id"] or source["split"] != row["split"]:
            errors.append(f"Input/candidate scene or split mismatch: {row['image_id']}")
        contents.setdefault(row["content_id"], set()).add(row["split"])
    split_counts = dict(Counter(r["split"] for r in rows))
    for split in ("train", "test"):
        with (dataset / "splits" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
            split_rows = list(csv.DictReader(handle))
        if split_rows != [row for row in rows if row["split"] == split]:
            errors.append(f"The {split} split file differs from the full manifest")
    if split_counts != {"train": 4160, "test": 1040} or len(scenes) != 1040 or len(expected_images) != 5200:
        errors.append("Counts do not match the packaged historical 5,200-image split")
    overlap = sum({"train", "test"}.issubset(parts) for parts in contents.values())
    if overlap != 84:
        errors.append(f"Historical content-group overlap changed: {overlap}, expected 84")
    package_manifest = root / "PACKAGE_MANIFEST.json"
    if package_manifest.is_file():
        package = json.loads(package_manifest.read_text(encoding="utf-8"))
        for entry in package["files"]:
            check(local_path(root, entry["path"]), entry["sha256"], entry["bytes"])
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "weights": len(weights["files"]),
        "image_files": len(expected_images),
        "exposure_instances": len(scenes),
        "split_counts": split_counts,
        "source_content_overlap": overlap,
        "manifest_sha256": sha256(dataset / "manifest.csv"),
        "verified_bytes": sum(entry["bytes"] for entry in checked.values()),
        "errors": errors,
        "files": list(checked.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, help="Optional JSON output; contains relative paths only")
    args = parser.parse_args()
    report = verify(args.root.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
