"""ハートビートの保存先。GCS の ``ops/heartbeat/latest.json``（本番）とローカルファイル（テスト・dev）。

latest.json には最新の値・受信時刻・直近の履歴・通知済みの異常の状態（alert_state）を 1 つにまとめて持つ。
更新は 1 インスタンスずつ（Cloud Run の最大 3 でも、5 分おきの書き込み 1 本 + 確認 1 本なので競合は実害が小さい。
GCS の世代一致条件つき書き込みで、競合したら読み直して 1 回だけやり直す）。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

OBJECT_NAME = "ops/heartbeat/latest.json"


class Backend(Protocol):
    def read(self) -> tuple[dict | None, Any]: ...
    def write(self, data: dict, token: Any) -> bool: ...


class FileBackend:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def read(self) -> tuple[dict | None, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")), None
        except FileNotFoundError:
            return None, None

    def write(self, data: dict, token: Any) -> bool:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        return True


class GcsBackend:
    """GCS の 1 オブジェクト。バケットは GCS_BUCKET（接頭辞なしの運用オブジェクト）。dev では作れない（gcp_guard）。"""

    def __init__(self, bucket: str | None = None, name: str = OBJECT_NAME) -> None:
        from src.config.gcp_guard import assert_gcp_allowed

        assert_gcp_allowed("ハートビートの保存（GCS）")
        self.bucket_name = bucket or os.environ.get("GCS_BUCKET", "")
        if not self.bucket_name:
            raise RuntimeError("GCS_BUCKET が未設定です")
        self.name = name
        self._bucket = None

    def _blob(self):
        if self._bucket is None:
            from google.cloud import storage

            self._bucket = storage.Client().bucket(self.bucket_name)
        return self._bucket.blob(self.name)

    def read(self) -> tuple[dict | None, Any]:
        from google.api_core.exceptions import NotFound

        blob = self._blob()
        try:
            text = blob.download_as_text()
        except NotFound:
            return None, 0
        return json.loads(text), blob.generation

    def write(self, data: dict, token: Any) -> bool:
        from google.api_core.exceptions import PreconditionFailed

        try:
            self._blob().upload_from_string(json.dumps(data, ensure_ascii=False), content_type="application/json",
                                            if_generation_match=token if token is not None else None)
            return True
        except PreconditionFailed:
            return False


def default_backend(base_dir: Path | None = None) -> Backend:
    """GCS_BUCKET があり dev でなければ GCS、それ以外はローカルファイル。"""
    from src.config.gcp_guard import gcp_forbidden

    if os.environ.get("GCS_BUCKET") and not gcp_forbidden():
        return GcsBackend()
    root = base_dir or Path(__file__).resolve().parents[2]
    return FileBackend(root / "data" / "local" / "meta" / "ops_heartbeat" / "latest.json")


def update(backend: Backend, fn) -> dict:
    """read → fn(record) → write を、競合したら 1 回だけやり直して実行する。fn は新しい record を返す。"""
    from src.ops.heartbeat_schema import new_record

    for attempt in (1, 2):
        cur, token = backend.read()
        new = fn(cur or new_record())
        if backend.write(new, token):
            return new
        if attempt == 2:
            raise RuntimeError("ハートビートの保存が競合しました")
    raise AssertionError("unreachable")
