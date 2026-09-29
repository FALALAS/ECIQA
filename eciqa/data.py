"""Aligned paired RGB inputs and original-resolution tiles for ECIQA.

Images stay in [0, 1]; the model owns encoder-specific normalization. A token
mask is true when its 16 x 16 pixel footprint intersects real image pixels.
Coordinates contain original-image normalized x/y and the resized/original
pixel scale (geometric mean for the square-resize control).
"""

from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


MODES = (
    "global224", "aspect224", "native_tiles", "multiscale_tiles",
    "resize448", "native_full",
)
PATCH_SIZE = 16


def infer_exposure(row: dict[str, str]) -> int:
    """Return under=0, over=1, neutral/unknown=2 from input metadata only."""
    for key in ("input_exposure", "exposure", "exposure_type"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"under", "underexposure", "underexposed", "low"}:
            return 0
        if value in {"over", "overexposure", "overexposed", "high"}:
            return 1
        if value in {"neutral", "normal", "well_exposed"}:
            return 2
    name = str(row.get("scene_id", ""))
    if re.search(r"_(?:N1(?:\.5)?|low|under)$", name, flags=re.IGNORECASE):
        return 0
    if re.search(r"_(?:P1(?:\.5)?|high|over)$", name, flags=re.IGNORECASE):
        return 1
    return 2


def pair_collate(samples):
    """Keep variable view counts/resolutions intact; no silent truncation."""
    return list(samples)


def _resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape[:2] == (height, width):
        return array
    return np.asarray(
        Image.fromarray(array).resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )


def _rounded_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    scale = long_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


class PairDataset(Dataset):
    def __init__(
        self, manifest, project_root, split, mode="global224", training=True,
        seed=2024, train_tiles=4, eval_tiles=0, image_size=224, long_edge=448,
    ):
        self.manifest = Path(manifest).resolve()
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.mode = mode
        self.training = bool(training)
        self.seed = int(seed)
        self.train_tiles = int(train_tiles)
        self.eval_tiles = int(eval_tiles)
        self.image_size = int(image_size)
        self.long_edge = int(long_edge)
        self.epoch = 0
        self._images = OrderedDict()
        self.cache_size = 8
        if mode not in MODES:
            raise ValueError(f"Unknown input mode: {mode}")
        if self.image_size <= 0 or self.image_size % PATCH_SIZE:
            raise ValueError("image_size must be a positive multiple of 16")
        if self.long_edge <= 0 or min(self.train_tiles, self.eval_tiles) < 0:
            raise ValueError("long_edge must be positive and tile limits nonnegative")
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        required = {
            "image_id", "relative_path", "input_relative_path", "scene_id",
            "split", "elo_scene_0_100",
        }
        if not self.rows or not required.issubset(self.rows[0]):
            raise ValueError(f"Manifest must contain rows and fields {sorted(required)}")
        if len({row["image_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate image_id values in manifest")
        self.indices = [i for i, row in enumerate(self.rows) if row["split"] == split]
        if not self.indices:
            raise ValueError(f"Manifest has no rows for split {split!r}")
        source_rows = {row["relative_path"].replace("\\", "/"): row for row in self.rows}
        scene_splits = {}
        for row in self.rows:
            old_split = scene_splits.setdefault(row["scene_id"], row["split"])
            if old_split != row["split"]:
                raise ValueError("Exposure-instance scene_id crosses manifest splits")
            label = float(row["elo_scene_0_100"])
            if not math.isfinite(label) or not 0.0 <= label <= 100.0:
                raise ValueError("elo_scene_0_100 must be finite and in [0, 100]")
            source = source_rows.get(row["input_relative_path"].replace("\\", "/"))
            if source is not None and source["split"] != row["split"]:
                raise ValueError("Input image belongs to a different split")

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _path(self, relative_path):
        path = (self.project_root / relative_path.replace("\\", "/")).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError(f"Image path escapes project_root: {relative_path}")
        return path

    def _decode(self, path):
        if path in self._images:
            self._images.move_to_end(path)
            return self._images[path]
        with Image.open(path) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8).copy()
        self._images[path] = rgb
        while len(self._images) > self.cache_size:
            self._images.popitem(last=False)
        return rgb

    def _plan_views(self, width, height, rng):
        size = self.image_size
        if self.mode == "global224":
            return [(size, size, 0, 0, size, size, 0, False)]
        if self.mode in {"resize448", "native_full"}:
            scaled_w, scaled_h = (
                _rounded_size(width, height, self.long_edge)
                if self.mode == "resize448" else (width, height)
            )
            out_w = math.ceil(scaled_w / PATCH_SIZE) * PATCH_SIZE
            out_h = math.ceil(scaled_h / PATCH_SIZE) * PATCH_SIZE
            return [(scaled_w, scaled_h, 0, 0, out_w, out_h, 0, False)]
        global_w, global_h = _rounded_size(width, height, size)
        global_view = (global_w, global_h, 0, 0, size, size, 0, True)
        if self.mode == "aspect224":
            return [global_view]
        scales = [(width, height)]
        if self.mode == "multiscale_tiles":
            medium = _rounded_size(width, height, self.long_edge)
            if medium != scales[0]:
                scales.insert(0, medium)
        locals_ = []
        for scaled_w, scaled_h in scales:
            for top in range(0, scaled_h, size):
                for left in range(0, scaled_w, size):
                    locals_.append((scaled_w, scaled_h, left, top, size, size, 1, False))
        limit = self.train_tiles if self.training else self.eval_tiles
        if limit and len(locals_) > limit:
            if self.training:
                selected = sorted(rng.choice(len(locals_), size=limit, replace=False).tolist())
            else:
                selected = np.linspace(0, len(locals_) - 1, num=limit, dtype=int).tolist()
            locals_ = [locals_[i] for i in selected]
        return [global_view, *locals_]

    def _make_view(self, source, candidate, plan, original_w, original_h, flip):
        scaled_w, scaled_h, left, top, out_w, out_h, view_type, centered = plan
        source, candidate = _resize(source, scaled_w, scaled_h), _resize(candidate, scaled_w, scaled_h)
        right, bottom = min(left + out_w, scaled_w), min(top + out_h, scaled_h)
        real_w, real_h = right - left, bottom - top
        offset_x = (out_w - real_w) // 2 if centered else 0
        offset_y = (out_h - real_h) // 2 if centered else 0
        padding = ((offset_y, out_h - real_h - offset_y), (offset_x, out_w - real_w - offset_x), (0, 0))
        pair = []
        for array in (source, candidate):
            array = np.pad(array[top:bottom, left:right], padding, mode="edge")
            pair.append(torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255.0))
        ys = np.arange(out_h // PATCH_SIZE, dtype=np.float32) * PATCH_SIZE
        xs = np.arange(out_w // PATCH_SIZE, dtype=np.float32) * PATCH_SIZE
        overlap_x = np.maximum(0, np.minimum(xs + PATCH_SIZE, offset_x + real_w) - np.maximum(xs, offset_x))
        overlap_y = np.maximum(0, np.minimum(ys + PATCH_SIZE, offset_y + real_h) - np.maximum(ys, offset_y))
        fraction = overlap_y[:, None] * overlap_x[None, :] / PATCH_SIZE**2
        grid_x, grid_y = np.meshgrid(
            np.clip((xs + PATCH_SIZE / 2 - offset_x + left) / scaled_w, 0, 1),
            np.clip((ys + PATCH_SIZE / 2 - offset_y + top) / scaled_h, 0, 1),
        )
        if flip:
            grid_x = 1 - grid_x
        scale = math.sqrt((scaled_w / original_w) * (scaled_h / original_h))
        coords = np.stack((grid_x, grid_y, np.full_like(grid_x, scale)), axis=-1)
        bounds = [left / scaled_w, top / scaled_h, right / scaled_w, bottom / scaled_h]
        if flip:
            bounds[0], bounds[2] = 1 - bounds[2], 1 - bounds[0]
        return pair, torch.from_numpy(fraction > 0), torch.from_numpy(coords), torch.from_numpy(fraction), bounds, view_type

    def __getitem__(self, index):
        row_index = self.indices[index]
        row = self.rows[row_index]
        source = self._decode(self._path(row["input_relative_path"]))
        candidate = self._decode(self._path(row["relative_path"]))
        original_h, original_w = source.shape[:2]
        candidate_h, candidate_w = candidate.shape[:2]
        aligned = (candidate_w, candidate_h) != (original_w, original_h)
        if aligned:
            candidate = _resize(candidate, original_w, original_h)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, row_index]))
        flip = self.training and bool(rng.integers(0, 2))
        if flip:
            source, candidate = source[:, ::-1], candidate[:, ::-1]
        plans = self._plan_views(original_w, original_h, rng)
        views = [self._make_view(source, candidate, plan, original_w, original_h, flip) for plan in plans]
        tile_mode = self.mode in {"native_tiles", "multiscale_tiles"}
        return {
            "source": torch.stack([item[0][0] for item in views]),
            "candidate": torch.stack([item[0][1] for item in views]),
            "masks": torch.stack([item[1] for item in views]),
            "coords": torch.stack([item[2] for item in views]),
            "valid_fraction": torch.stack([item[3] for item in views]),
            "view_bounds": torch.tensor([item[4] for item in views], dtype=torch.float32),
            "view_type": torch.tensor([item[5] for item in views], dtype=torch.long),
            "label": torch.tensor(float(row["elo_scene_0_100"]) / 100.0, dtype=torch.float32),
            "image_id": row["image_id"],
            "row_index": row_index,
            "scene_id": row["scene_id"],
            "content_id": row.get("content_id", row["scene_id"]),
            "exposure": infer_exposure(row),
            "candidate_resized_to_input": aligned,
            "source_size": (original_w, original_h),
            "candidate_original_size": (candidate_w, candidate_h),
            "horizontal_flip": flip,
            "full_native_coverage": self.mode == "native_full" or (
                tile_mode and (self.train_tiles if self.training else self.eval_tiles) == 0
            ),
        }
