"""Contract and local-serving checks; uses only Python's standard library."""

import csv
import http.client
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import unittest


SPEC = importlib.util.spec_from_file_location("annotation_serve", Path(__file__).parents[1] / "serve.py")
serve = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = serve
SPEC.loader.exec_module(serve)


class AnnotationServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.ui = self.root / "ui"
        self.ui.mkdir()
        (self.ui / "index.html").write_text("<h1>Annotate</h1>", encoding="utf-8")
        self.rows = []
        for order, method in [(0, "input"), (10, "method_a"), (20, "method_b")]:
            relative = f"scenes/example/{order:02d}_{method}.jpg"
            path = self.data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\xff\xd8sample-image\xff\xd9")
            self.rows.append({"image_id": str(order), "dataset": "Example",
                              "scene_id": "Example_scene", "method": method,
                              "relative_path": relative,
                              "input_relative_path": "scenes/example/00_input.jpg",
                              "width": "20", "height": "10",
                              "elo_scene_0_100": "89.123456", "private_path": "PRIVATE_ROOT"})
        self.csv_path = self.root / "manifest.csv"
        self.write_manifest()

    def write_manifest(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def load(self):
        return serve.load_manifest(self.csv_path, self.data)

    def test_mapping_excludes_scores_and_private_metadata(self):
        manifest = self.load()
        self.assertEqual([row["item_type"] for row in manifest.records],
                         ["input", "candidate", "candidate"])
        self.assertEqual([row["display_order"] for row in manifest.records], [0, 10, 20])
        self.assertEqual(manifest.records[0]["path"], "/images/scenes/example/00_input.jpg")
        self.assertEqual(manifest.records[0]["sample_id"], "scene")
        for private in [b"elo_scene", b"89.123456", b"PRIVATE_ROOT",
                        str(self.root).encode("utf-8")]:
            self.assertNotIn(private, manifest.javascript)

    def test_missing_image_and_invalid_scene_fail_before_serving(self):
        candidate = self.data / self.rows[-1]["relative_path"]
        candidate.unlink()
        with self.assertRaisesRegex(ValueError, "Missing image"):
            self.load()
        candidate.write_bytes(b"image")
        self.rows = self.rows[1:]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "exactly one method=input"):
            self.load()

    def test_duplicate_and_inconsistent_input_are_rejected(self):
        self.rows.append(dict(self.rows[-1]))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate image path"):
            self.load()
        self.rows.pop()
        self.rows[-1]["input_relative_path"] = self.rows[-1]["relative_path"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "inconsistent input_relative_path"):
            self.load()

    def test_http_only_exposes_whitelisted_assets_and_images(self):
        (self.data / "secret.jpg").write_bytes(b"not in manifest")
        (self.ui / "secret.txt").write_text("private", encoding="utf-8")
        server = serve.create_server(self.load(), 0, self.ui)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            self.addCleanup(connection.close)
            for path, expected in [
                ("/", 200), ("/manifest.js", 200),
                ("/images/scenes/example/00_input.jpg", 200),
                ("/images/secret.jpg", 404), ("/secret.txt", 404),
                ("/weights/model.pt", 404), ("/data/manifest.csv", 404),
                ("/images/../manifest.csv", 404),
                ("/images/%2e%2e%2fsecret.jpg", 404),
                ("/images/scenes%5cexample%5c00_input.jpg", 404),
            ]:
                connection.request("GET", path)
                response = connection.getresponse()
                self.assertEqual(response.status, expected, path)
                response.read()
            connection.request("GET", "/manifest.js", headers={"Host": "external.example"})
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
            connection.request("POST", "/annotations", body=b"write me")
            response = connection.getresponse()
            self.assertEqual(response.status, 501)
            response.read()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_manifest_cannot_reference_outside_dataset(self):
        self.rows[0]["relative_path"] = "../outside.jpg"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Invalid relative image path"):
            self.load()


if __name__ == "__main__":
    unittest.main()
