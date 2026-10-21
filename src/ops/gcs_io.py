"""GCS の読み書きを薄く包む（運用基盤共通）。テストではメモリ実装 ``MemoryIO`` を使う。

- GCS 実装は dev 環境では作れない（``gcp_guard``）。認証は実行環境のサービスアカウント（Cloud Run / VPS）か、ホストの ``.env`` の資格情報。
- キーはバケット内のパス（先頭スラッシュなし）。``ops/…``（運用）、``state/…``（ジョブの入出力）、``models/…``（モデル配布）、``parallel/…``（並走検証）。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol


class BlobIO(Protocol):
    def get_bytes(self, key: str) -> bytes | None: ...
    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...
    def size(self, key: str) -> int | None: ...
    def delete(self, key: str) -> bool: ...
    def download_file(self, key: str, path: Path) -> bool: ...
    def upload_file(self, path: Path, key: str) -> None: ...


class MemoryIO:
    """テスト・ローカル用。"""

    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get_bytes(self, key: str) -> bytes | None:
        with self._lock:
            return self._d.get(key)

    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        with self._lock:
            self._d[key] = bytes(data)

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._d

    def list(self, prefix: str) -> list[str]:
        with self._lock:
            return sorted(k for k in self._d if k.startswith(prefix))

    def size(self, key: str) -> int | None:
        with self._lock:
            v = self._d.get(key)
            return None if v is None else len(v)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._d.pop(key, None) is not None

    def download_file(self, key: str, path: Path) -> bool:
        data = self.get_bytes(key)
        if data is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return True

    def upload_file(self, path: Path, key: str) -> None:
        self.put_bytes(key, Path(path).read_bytes())


class GcsIO:
    def __init__(self, bucket: str, client: Any | None = None) -> None:
        from src.config.gcp_guard import assert_gcp_allowed

        assert_gcp_allowed("ops GcsIO")
        if not bucket:
            raise ValueError("バケット名が空（GCS_BUCKET）")
        if client is None:
            from google.cloud import storage  # 遅延 import

            client = storage.Client()
        self.bucket_name = bucket
        self._bucket = client.bucket(bucket)

    def get_bytes(self, key: str) -> bytes | None:
        blob = self._bucket.get_blob(key)
        return None if blob is None else blob.download_as_bytes()

    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        self._bucket.blob(key).upload_from_string(data, content_type=content_type or "application/octet-stream")

    def exists(self, key: str) -> bool:
        return bool(self._bucket.blob(key).exists())

    def list(self, prefix: str) -> list[str]:
        return sorted(b.name for b in self._bucket.list_blobs(prefix=prefix))

    def size(self, key: str) -> int | None:
        blob = self._bucket.get_blob(key)
        return None if blob is None else int(blob.size or 0)

    def delete(self, key: str) -> bool:
        from google.api_core.exceptions import NotFound

        try:
            self._bucket.blob(key).delete()
            return True
        except NotFound:
            return False

    def download_file(self, key: str, path: Path) -> bool:
        from google.api_core.exceptions import NotFound

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._bucket.blob(key).download_to_filename(str(path))
            return True
        except NotFound:
            return False

    def upload_file(self, path: Path, key: str) -> None:
        self._bucket.blob(key).upload_from_filename(str(path))


def default_io() -> BlobIO:
    """``GCS_BUCKET`` のバケットを開く（Cloud Run / VPS / ホスト用）。"""
    return GcsIO(os.environ.get("GCS_BUCKET", "").strip())


def read_json(io: BlobIO, key: str) -> Any | None:
    raw = io.get_bytes(key)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def write_json(io: BlobIO, key: str, obj: Any) -> None:
    io.put_bytes(key, json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"), content_type="application/json")
