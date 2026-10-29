"""影モード（VPS の取得をホストと並走で検証する）専用の GCS 書き込みガード。

影モードの子プロセスは、本番と同じ取得コード（``HybridStorage.save`` など）を通すが、**書き込み先を ``shadow/<日付>/`` 以下に付け替え、
それ以外の書き込み・削除・コピーは例外で拒否する**。読み取りは本番のデータをそのまま読む（出馬表から馬を決める等のため）。

- ``install()`` が ``HybridStorage._get_bucket`` と ``google.cloud.storage.Client.bucket / get_bucket / lookup_bucket`` を包み、
  どの経路で取ったバケットでも ``ShadowBucket`` になる。``ShadowBlob.upload_from_*`` だけが書け、名前は必ず ``shadow/`` で始まる。
- これはコードの保証。**本物の保証は IAM**（影モードの実行役のサービスアカウントには ``shadow/`` 以下だけに書ける条件付きの権限を付ける。
  docs/operations/vps_agent.md の影モードの章）。両方で二重に守る。
- 「取得済みならスキップ」の判定（``HybridStorage.exists`` / ``exists_gcs``）だけは**影の側**を見る（影に無ければ取る）。
  本番を見ると本番に既にあるものは全部スキップされ、影が空のまま検証にならないため。``load`` などの読みは本番のまま。
- ``KEIBA_FETCH_SHADOW=1`` の子プロセスだけが ``install()`` する（``fetch_child``）。通常の取得には一切影響しない。
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
SHADOW_ROOT = "shadow"

READ_BLOB_ATTRS = frozenset({
    "download_as_bytes", "download_as_text", "download_as_string", "download_to_filename", "download_to_file",
    "exists", "reload", "name", "size", "updated", "generation", "metageneration", "metadata", "content_type", "md5_hash",
    "crc32c", "etag", "time_created", "bucket", "path", "public_url", "id", "storage_class",
})
WRITE_BLOB_METHODS = frozenset({"upload_from_string", "upload_from_file", "upload_from_filename"})
READ_BUCKET_ATTRS = frozenset({"name", "exists", "path", "id", "location", "project_number"})


_EXISTS_MODE = threading.local()


def _shadow_exists_mode() -> bool:
    return bool(getattr(_EXISTS_MODE, "on", False))


class ShadowViolation(RuntimeError):
    """影モードで許されない操作（本番への書き込み・削除など）を試みた。"""


def shadow_prefix(now: datetime | None = None, root: str = SHADOW_ROOT) -> str:
    """``shadow/YYYYMMDD/``（JST の日付）。"""
    return f"{root}/{(now or datetime.now(JST)).astimezone(JST).strftime('%Y%m%d')}/"


def assert_shadow_name(name: str, root: str = SHADOW_ROOT) -> str:
    if not name.startswith(f"{root}/"):
        raise ShadowViolation(f"影モードは {root}/ 以外に書けません: {name!r}")
    return name


class ShadowBlob:
    """読みは本番の同名オブジェクト、書きは ``shadow/<日付>/<同名>``。それ以外の操作は拒否する。"""

    def __init__(self, real_bucket: Any, name: str, prefix_fn: Callable[[], str]) -> None:
        self._real_bucket = real_bucket
        self._orig_name = name
        self._prefix_fn = prefix_fn
        self._read_blob = real_bucket.blob(name)

    @property
    def shadow_name(self) -> str:
        return assert_shadow_name(self._prefix_fn() + self._orig_name)

    def _write_blob(self) -> Any:
        return self._real_bucket.blob(self.shadow_name)

    def upload_from_string(self, *a: Any, **k: Any) -> Any:
        return self._write_blob().upload_from_string(*a, **k)

    def upload_from_file(self, *a: Any, **k: Any) -> Any:
        return self._write_blob().upload_from_file(*a, **k)

    def upload_from_filename(self, *a: Any, **k: Any) -> Any:
        return self._write_blob().upload_from_filename(*a, **k)

    def exists(self, *a: Any, **k: Any) -> bool:
        """通常は本番の有無。``exists_gcs`` 経由（スキップ判定）のときだけ影の有無を返す。"""
        if _shadow_exists_mode():
            return bool(self._write_blob().exists(*a, **k))
        return bool(self._read_blob.exists(*a, **k))

    def __getattr__(self, attr: str) -> Any:
        if attr in READ_BLOB_ATTRS:
            return getattr(self._read_blob, attr)
        raise ShadowViolation(f"影モードで許されない操作: Blob.{attr}（{self._orig_name}）")


class _ShadowListing:
    def __init__(self, real_bucket: Any, it: Any) -> None:
        self._real_bucket, self._it = real_bucket, it

    def __iter__(self) -> Iterator[Any]:
        for b in self._it:
            yield ShadowBlob(self._real_bucket, b.name, lambda: shadow_prefix())

    def __getattr__(self, attr: str) -> Any:     # prefixes / pages / num_results など読み取りの属性
        if attr.startswith("_"):
            raise AttributeError(attr)
        return getattr(self._it, attr)


class ShadowBucket:
    def __init__(self, real_bucket: Any, prefix_fn: Callable[[], str] = shadow_prefix) -> None:
        self._real = real_bucket
        self._prefix_fn = prefix_fn

    def blob(self, name: str, *a: Any, **k: Any) -> ShadowBlob:
        return ShadowBlob(self._real, name, self._prefix_fn)

    def get_blob(self, name: str, *a: Any, **k: Any) -> ShadowBlob | None:
        got = self._real.get_blob(name, *a, **k)
        return None if got is None else ShadowBlob(self._real, name, self._prefix_fn)

    def list_blobs(self, *a: Any, **k: Any) -> _ShadowListing:
        return _ShadowListing(self._real, self._real.list_blobs(*a, **k))

    def __getattr__(self, attr: str) -> Any:
        if attr in READ_BUCKET_ATTRS:
            return getattr(self._real, attr)
        raise ShadowViolation(f"影モードで許されない操作: Bucket.{attr}")


_INSTALLED = False


def wrap_bucket(bucket: Any) -> Any:
    return bucket if isinstance(bucket, ShadowBucket) else ShadowBucket(bucket)


def install() -> None:
    """影モードの子プロセスの先頭で 1 回呼ぶ。冪等。"""
    global _INSTALLED
    if _INSTALLED:
        return
    from google.cloud import storage as gcs_lib

    from src.scraper.storage import HybridStorage

    orig_get_bucket = HybridStorage._get_bucket

    def _get_bucket(self: Any) -> Any:
        return wrap_bucket(orig_get_bucket(self))

    HybridStorage._get_bucket = _get_bucket          # type: ignore[method-assign]

    orig_exists, orig_exists_gcs = HybridStorage.exists, HybridStorage.exists_gcs

    def _exists_gcs(self: Any, category: str, key: str) -> bool:
        prev = _shadow_exists_mode()
        _EXISTS_MODE.on = True
        try:
            return orig_exists_gcs(self, category, key)
        finally:
            _EXISTS_MODE.on = prev

    def _exists(self: Any, category: str, key: str) -> bool:
        if self._is_local_only(category) or self._dev_store is not None:
            return orig_exists(self, category, key)
        return self.exists_gcs(category, key)         # 本番由来の load / 一覧キャッシュは見ない

    HybridStorage.exists_gcs = _exists_gcs            # type: ignore[method-assign]
    HybridStorage.exists = _exists                    # type: ignore[method-assign]

    for name in ("bucket", "get_bucket", "lookup_bucket"):
        orig = getattr(gcs_lib.Client, name)

        def _wrapped(self: Any, *a: Any, __orig: Callable[..., Any] = orig, **k: Any) -> Any:
            b = __orig(self, *a, **k)
            return None if b is None else wrap_bucket(b)

        setattr(gcs_lib.Client, name, _wrapped)

    def _deny(self: Any, *a: Any, **k: Any) -> Any:
        raise ShadowViolation("影モードではバケットを作れません")

    gcs_lib.Client.create_bucket = _deny              # type: ignore[method-assign]
    _INSTALLED = True
