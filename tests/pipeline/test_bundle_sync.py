import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.features.horse_history import bundle_sync as bs


class _Blob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def exists(self):
        return self.name in self.store

    def reload(self):
        pass

    @property
    def md5_hash(self):
        return base64.b64encode(hashlib.md5(self.store[self.name]).digest()).decode()

    def upload_from_filename(self, path, timeout=None):
        self.store[self.name] = Path(path).read_bytes()
        self.store.setdefault("_uploads", []).append(self.name)

    def upload_from_string(self, s, content_type=None):
        self.store[self.name] = s.encode()

    def download_as_text(self):
        return self.store[self.name].decode()

    def download_to_filename(self, path):
        Path(path).write_bytes(self.store[self.name])

    def delete(self):
        del self.store[self.name]


class _Bucket:
    def __init__(self):
        self.store = {}

    def blob(self, name):
        return _Blob(self.store, name)

    def list_blobs(self, prefix=""):
        return [_Blob(self.store, n) for n in list(self.store) if n.startswith(prefix) and n != "_uploads"]


def _make_bundle(d: Path, built_at: str, payload: bytes):
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"built_at": built_at, "history_last_date": "2026-10-10", "rows": 3}))
    (d / "runs.arrow").write_bytes(payload)


class BundleSyncTest(unittest.TestCase):
    def test_publish_then_fetch_roundtrip_and_noop(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            b = _Bucket()
            _make_bundle(t / "src", "2026-10-15T00:25:59+09:00", b"abc" * 100)
            out = bs.publish(t / "src", b)
            self.assertEqual(out["version"], "20261015T002559")
            self.assertIn(f"{bs.GCS_PREFIX}/{bs.LATEST}", b.store)
            r = bs.fetch(t / "dest", b)
            self.assertTrue(r["updated"])
            self.assertEqual((t / "dest" / "runs.arrow").read_bytes(), b"abc" * 100)
            self.assertFalse(bs.fetch(t / "dest", b)["updated"])        # 同じ版なら何もしない

    def test_same_version_republish_uploads_nothing_new(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            b = _Bucket()
            _make_bundle(t / "src", "2026-10-15T00:25:59+09:00", b"x" * 50)
            bs.publish(t / "src", b)
            n = len(b.store["_uploads"])
            bs.publish(t / "src", b)
            self.assertEqual(len(b.store["_uploads"]), n)

    def test_prune_keeps_latest_versions_only(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            b = _Bucket()
            for i in range(5):
                _make_bundle(t / "src", f"2026-10-1{i}T00:00:00+09:00", bytes([i]) * 10)
                bs.publish(t / "src", b, keep=3)
            self.assertEqual(bs._versions(b), ["20261012T000000", "20261013T000000", "20261014T000000"])

    def test_fetch_detects_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            b = _Bucket()
            _make_bundle(t / "src", "2026-10-15T00:25:59+09:00", b"abc")
            bs.publish(t / "src", b)
            b.store[f"{bs.GCS_PREFIX}/20261015T002559/runs.arrow"] = b"zzz"
            with self.assertRaises(IOError):
                bs.fetch(t / "dest", b)
            self.assertFalse((t / "dest").exists())


if __name__ == "__main__":
    unittest.main()
