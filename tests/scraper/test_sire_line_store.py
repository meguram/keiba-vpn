"""系統台帳の GCS 同期（sync_with_gcs）と足し合わせ。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.scraper import sire_line_store as S


class _Backend:
    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self.blobs = dict(blobs or {})

    def list(self, prefix):
        return {k: ("", 0.0) for k in self.blobs if k.startswith(prefix)}

    def download(self, key):
        return self.blobs[key]

    def upload(self, key, data):
        self.blobs[key] = data


def _rec(line, at="2026-10-29", **kw):
    return {"name": "x", "line": line, "from": "c", "at": at, **kw}


class TestSyncWithGcs(unittest.TestCase):
    def setUp(self) -> None:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = Path(d.name) / "map.json"

    def _put(self, lines, complete_at=None):
        self.path.write_text(json.dumps({"lines": lines, "complete_at": complete_at}), encoding="utf-8")

    def test_first_sync_uploads_local(self) -> None:
        self._put({"a": _rec("サンデー系")})
        b = _Backend()
        res = S.sync_with_gcs(b, path=self.path)
        self.assertEqual(res["pushed"], 1)
        self.assertEqual(json.loads(b.blobs[S.GCS_KEY])["lines"]["a"]["line"], "サンデー系")

    def test_union_both_ways_and_line_beats_missing(self) -> None:
        self._put({"a": _rec("サンデー系"), "b": _rec(None, reason="no_line")})
        remote = {"lines": {"b": _rec("ナスルーラ系"), "c": _rec("ミスプロ系")}}
        b = _Backend({S.GCS_KEY: json.dumps(remote).encode()})
        res = S.sync_with_gcs(b, path=self.path)
        self.assertEqual(res["pulled"], 2)
        local = S.load(self.path)
        self.assertEqual(sorted(local), ["a", "b", "c"])
        self.assertEqual(local["b"]["line"], "ナスルーラ系")
        self.assertEqual(sorted(json.loads(b.blobs[S.GCS_KEY])["lines"]), ["a", "b", "c"])

    def test_complete_mark_propagates_either_way(self) -> None:
        self._put({"a": _rec("x")})
        b = _Backend({S.GCS_KEY: json.dumps({"lines": {}, "complete_at": "2026-10-29T12:00:00+09:00"}).encode()})
        S.sync_with_gcs(b, path=self.path)
        self.assertTrue(S.is_complete(self.path))
        self._put({"a": _rec("x")}, complete_at="2026-10-30T00:00:00+09:00")
        b2 = _Backend()
        S.sync_with_gcs(b2, path=self.path)
        self.assertEqual(json.loads(b2.blobs[S.GCS_KEY])["complete_at"], "2026-10-30T00:00:00+09:00")

    def test_broken_remote_is_overwritten_not_fatal(self) -> None:
        self._put({"a": _rec("x")})
        b = _Backend({S.GCS_KEY: b"not json"})
        S.sync_with_gcs(b, path=self.path)
        self.assertEqual(sorted(json.loads(b.blobs[S.GCS_KEY])["lines"]), ["a"])


if __name__ == "__main__":
    unittest.main()
