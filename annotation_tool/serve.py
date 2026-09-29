"""Serve the ECIQA annotation UI and manifest-listed images on localhost."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit


TOOL_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = TOOL_DIR.parent
STATIC_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "review.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "review_app.js": "text/javascript; charset=utf-8",
    "label_core.js": "text/javascript; charset=utf-8",
    "group_csv_core.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}
IMAGE_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
    ".avif": "image/avif",
}
REQUIRED_COLUMNS = {
    "dataset", "scene_id", "method", "relative_path", "input_relative_path",
    "width", "height",
}


@dataclass(frozen=True)
class Manifest:
    records: list[dict]
    images: dict[str, Path]
    data_root: Path
    javascript: bytes


def relative_image_path(value: str) -> str:
    """Accept only portable, explicit paths below the dataset root."""
    parts = value.split("/")
    if (not value or "\\" in value or ":" in value or "\x00" in value
            or any(part in {"", ".", ".."} for part in parts)):
        raise ValueError(f"Invalid relative image path: {value!r}")
    if PurePosixPath(value).suffix.lower() not in IMAGE_TYPES:
        raise ValueError(f"Unsupported image extension: {value!r}")
    return value


def resolve_image(data_root: Path, relative_path: str) -> Path:
    path = (data_root / relative_path).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError(f"Image path escapes the dataset root: {relative_path!r}")
    if not path.is_file():
        raise ValueError(f"Missing image: {relative_path!r}")
    return path


def load_manifest(manifest_path: Path, data_root: Path) -> Manifest:
    """Convert the released CSV to the front-end contract, excluding labels."""
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_root}")
    records, images, scenes, image_ids = [], {}, {}, set()
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Manifest is missing columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) is None for key in REQUIRED_COLUMNS):
                raise ValueError(f"Malformed CSV row at line {line}")
            scene_id, dataset, method = (row[key].strip() for key in
                                         ("scene_id", "dataset", "method"))
            if not scene_id or not dataset or not method:
                raise ValueError(f"Empty dataset, scene_id or method at line {line}")
            relative = relative_image_path(row["relative_path"].strip())
            input_relative = relative_image_path(row["input_relative_path"].strip())
            if relative in images:
                raise ValueError(f"Duplicate image path at line {line}: {relative!r}")
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                if image_id in image_ids:
                    raise ValueError(f"Duplicate image_id at line {line}: {image_id!r}")
                image_ids.add(image_id)
            try:
                width, height = int(row["width"]), int(row["height"])
                if width <= 0 or height <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid image width or height at line {line}") from None
            images[relative] = resolve_image(data_root, relative)
            scene = scenes.setdefault(scene_id, {"dataset": dataset, "methods": set(),
                                                 "inputs": [], "references": set(),
                                                 "candidates": 0})
            if scene["dataset"] != dataset:
                raise ValueError(f"scene_id occurs in multiple datasets: {scene_id!r}")
            if method in scene["methods"]:
                raise ValueError(f"Duplicate method {method!r} in scene {scene_id!r}")
            scene["methods"].add(method)
            scene["references"].add(input_relative)
            item_type = "input" if method == "input" else "candidate"
            if item_type == "input":
                scene["inputs"].append(relative)
            else:
                scene["candidates"] += 1
            order = re.match(r"(\d+)(?:[_-]|\.)", PurePosixPath(relative).name)
            # Whitelist fields explicitly: preference scores and local paths
            # must never enter the browser's annotation manifest.
            records.append({
                "dataset": dataset, "scene_id": scene_id,
                "sample_id": (row.get("sample_id") or scene_id.removeprefix(dataset + "_")).strip(),
                "item_type": item_type, "method": method,
                "display_order": int(order.group(1)) if order else 0,
                "path": "/images/" + quote(relative, safe="/"),
                "width": width, "height": height,
            })
    if not records:
        raise ValueError("Manifest contains no images")
    for scene_id, scene in scenes.items():
        if len(scene["inputs"]) != 1:
            raise ValueError(f"Scene {scene_id!r} must contain exactly one method=input row")
        if scene["references"] != set(scene["inputs"]):
            raise ValueError(f"Scene {scene_id!r} has an inconsistent input_relative_path")
        if scene["candidates"] < 2:
            raise ValueError(f"Scene {scene_id!r} needs at least two candidate images")
    meta = {"record_count": len(records), "scene_count": len(scenes),
            "image_count": len(images)}
    javascript = ("window.MOS_MANIFEST_META = " + json.dumps(meta) + ";\n"
                  "window.MOS_RECORDS = " + json.dumps(records, ensure_ascii=True,
                                                       separators=(",", ":")) + ";\n")
    return Manifest(records, images, data_root, javascript.encode("utf-8"))


def create_server(manifest: Manifest, port: int = 8010,
                  tool_dir: Path = TOOL_DIR) -> ThreadingHTTPServer:
    """Expose only UI assets and the manifest's raster images; never list folders."""
    tool_dir = Path(tool_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ECIQA-Annotation/1.0"

        def do_GET(self):
            self.serve_resource(head_only=False)

        def do_HEAD(self):
            self.serve_resource(head_only=True)

        def serve_resource(self, head_only: bool):
            # No permissive CORS or non-local hostnames (including DNS rebinding).
            valid_hosts = {f"127.0.0.1:{self.server.server_port}",
                           f"localhost:{self.server.server_port}"}
            if self.headers.get("Host", "") not in valid_hosts:
                self.send_error(403, "Use the localhost URL printed by the server")
                return
            try:
                parsed = urlsplit(self.path)
                if parsed.scheme or parsed.netloc:
                    raise ValueError("Absolute URLs are not supported")
                request_path = unquote(parsed.path, errors="strict")
                if request_path == "/favicon.ico":
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if request_path == "/manifest.js":
                    self.send_content_headers("text/javascript; charset=utf-8",
                                              len(manifest.javascript))
                    if not head_only:
                        self.wfile.write(manifest.javascript)
                    return
                name = "index.html" if request_path == "/" else request_path.removeprefix("/")
                if name in STATIC_TYPES:
                    path = (tool_dir / name).resolve()
                    if not path.is_relative_to(tool_dir):
                        raise ValueError("Static path escapes the tool directory")
                    content_type = STATIC_TYPES[name]
                elif request_path.startswith("/images/"):
                    relative = relative_image_path(request_path[len("/images/"):])
                    if relative not in manifest.images:
                        raise ValueError("Image is not in the manifest")
                    # Resolve again so a replaced symlink cannot expose other files.
                    path = resolve_image(manifest.data_root, relative)
                    content_type = IMAGE_TYPES[path.suffix.lower()]
                else:
                    raise ValueError("Resource is not exposed")
                with path.open("rb") as stream:
                    self.send_content_headers(content_type, path.stat().st_size)
                    if not head_only:
                        shutil.copyfileobj(stream, self.wfile)
            except (OSError, ValueError, KeyError):
                self.send_error(404, "Not found")

        def send_content_headers(self, content_type: str, length: int):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY_DIR / "data/ECIQA-Pref/manifest.csv")
    parser.add_argument("--data-root", type=Path,
                        default=REPOSITORY_DIR / "data/ECIQA-Pref")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        missing = [name for name in STATIC_TYPES if not (TOOL_DIR / name).is_file()]
        if missing:
            raise ValueError("Missing UI files: " + ", ".join(missing))
        manifest = load_manifest(args.manifest, args.data_root)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Cannot load annotation data: {error}", file=sys.stderr)
        return 2
    try:
        server = create_server(manifest, args.port)
    except OSError as error:
        print(f"Cannot bind 127.0.0.1:{args.port}: {error}. "
              "If this port is occupied, choose another with --port.", file=sys.stderr)
        return 2
    with server:
        print(f"Loaded {len(manifest.records)} images. "
              f"Open http://127.0.0.1:{args.port}/", flush=True)
        print("Annotations are saved by the browser to your chosen local folder. "
              "Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
