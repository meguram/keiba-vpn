"""学習済みモデルの公開と配布（バージョン付き・検証付き）。

学習は別PC（ローカル）で行い、できたモデルを ``publish_model`` でストアへ公開する。
定期実行側（VPS/GCP）は ``fetch_latest`` で現行版を取得して使う。

ストア上のレイアウト::

    <prefix>/latest.json                      # {"version": "..."}  現行版を指す
    <prefix>/<version>/manifest.json          # 版の情報と各ファイルの sha256
    <prefix>/<version>/ensemble_meta.json     # EnsembleTrainer と同じ形式
    <prefix>/<version>/<name>_model.pkl ...   # 各ベースモデルと meta_model.pkl

ストアは ``LocalModelStore``（ディレクトリ）か ``GcsModelStore``（GCS）。``open_store`` が
``gs://bucket/prefix`` と通常のパスを見分ける。

安全策:
  - 取得時に ``sha256`` を検証する（改ざん・転送欠損の検知）。
  - pickle はライブラリの版が違うと読めない／結果が変わることがあるため、公開時に版を記録し、
    取得時に major.minor の不一致を ``ModelIncompatibleError`` とする。
  - 特徴量の列名・順序は ``manifest.json`` に記録し、推論前に ``EnsemblePredictor`` が検証する。
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("pipeline.models.model_registry")

MANIFEST_NAME = "manifest.json"
LATEST_NAME = "latest.json"
TRACKED_LIBRARIES = ("scikit-learn", "lightgbm", "xgboost", "catboost", "numpy", "pandas")


class ModelIntegrityError(RuntimeError):
    """sha256 が manifest と一致しない（破損・改ざんの疑い）。"""


class ModelIncompatibleError(RuntimeError):
    """学習時とライブラリの版が合わない。"""


# ── ストア ─────────────────────────────────────────────


class ModelStore(Protocol):
    def put_file(self, rel: str, local_path: Path) -> None: ...
    def get_file(self, rel: str, dest: Path) -> None: ...
    def read_json(self, rel: str) -> dict | None: ...
    def write_json(self, rel: str, obj: dict) -> None: ...


class LocalModelStore:
    """ディレクトリをストアとして使う（開発・テスト・ネットワーク共有用）。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put_file(self, rel: str, local_path: Path) -> None:
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)

    def get_file(self, rel: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / rel, dest)

    def read_json(self, rel: str) -> dict | None:
        p = self.root / rel
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def write_json(self, rel: str, obj: dict) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)


class GcsModelStore:
    """GCS バケット（``gs://bucket/prefix``）をストアとして使う。認証は .env の ``GCS_*``。"""

    def __init__(self, bucket_name: str, prefix: str = "", *, bucket: Any = None):
        self.prefix = prefix.strip("/")
        if bucket is not None:  # テスト用の注入
            self._bucket = bucket
        else:
            from google.cloud import storage as gcs_lib

            from src.config.gcp_credentials import build_gcp_credentials

            creds = build_gcp_credentials()
            client = gcs_lib.Client(credentials=creds, project=getattr(creds, "project_id", None)) if creds else gcs_lib.Client()
            self._bucket = client.bucket(bucket_name)

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def put_file(self, rel: str, local_path: Path) -> None:
        self._bucket.blob(self._key(rel)).upload_from_filename(str(local_path))

    def get_file(self, rel: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._bucket.blob(self._key(rel)).download_to_filename(str(dest))

    def read_json(self, rel: str) -> dict | None:
        blob = self._bucket.blob(self._key(rel))
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text(encoding="utf-8"))

    def write_json(self, rel: str, obj: dict) -> None:
        self._bucket.blob(self._key(rel)).upload_from_string(
            json.dumps(obj, ensure_ascii=False, indent=2), content_type="application/json"
        )


def open_store(url: str) -> ModelStore:
    """``gs://bucket/prefix`` なら GCS、それ以外はローカルディレクトリ。"""
    if url.startswith("gs://"):
        rest = url[len("gs://"):]
        bucket, _, prefix = rest.partition("/")
        return GcsModelStore(bucket, prefix)
    return LocalModelStore(url)


# ── manifest ───────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for lib in TRACKED_LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            continue
    return out


def build_manifest(
    model_dir: str | Path,
    version: str,
    feature_names: list[str],
    *,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """``model_dir`` 内の全ファイル（manifest 自身を除く）の sha256 を記録した manifest を返す。"""
    d = Path(model_dir)
    files = {
        p.name: {"sha256": _sha256(p), "size": p.stat().st_size}
        for p in sorted(d.iterdir())
        if p.is_file() and p.name != MANIFEST_NAME
    }
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
        "libraries": library_versions(),
        "metrics": metrics or {},
        "files": files,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _major_minor(v: str) -> tuple[str, str]:
    parts = v.split(".")
    return parts[0], parts[1] if len(parts) > 1 else "0"


def check_compatibility(manifest: dict, *, strict: bool = True) -> list[str]:
    """学習時と現環境のライブラリ版を比較する。不一致のメッセージ一覧を返し、strict なら例外。"""
    current = library_versions()
    problems: list[str] = []
    for lib, trained in (manifest.get("libraries") or {}).items():
        now = current.get(lib)
        if now is None:
            problems.append(f"{lib}: 学習時 {trained} / 現環境に未インストール")
        elif _major_minor(now) != _major_minor(trained):
            problems.append(f"{lib}: 学習時 {trained} / 現環境 {now}（major.minor が不一致）")
    if problems and strict:
        raise ModelIncompatibleError("; ".join(problems))
    for msg in problems:
        logger.warning("モデルのライブラリ版不一致: %s", msg)
    return problems


# ── 公開・取得 ─────────────────────────────────────────


def publish_model(
    model_dir: str | Path,
    version: str,
    store: ModelStore,
    *,
    feature_names: list[str] | None = None,
    metrics: dict | None = None,
    set_latest: bool = True,
) -> dict:
    """``model_dir`` をストアへ ``<version>/`` として公開する（別PCの学習後に実行）。

    ``feature_names`` 省略時は ``ensemble_meta.json`` の ``feature_names`` を使う。
    """
    d = Path(model_dir)
    if feature_names is None:
        meta_path = d / "ensemble_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} が無く feature_names を特定できません")
        feature_names = json.loads(meta_path.read_text(encoding="utf-8"))["feature_names"]

    manifest = build_manifest(d, version, feature_names, metrics=metrics)
    for name in manifest["files"]:
        store.put_file(f"{version}/{name}", d / name)
    store.write_json(f"{version}/{MANIFEST_NAME}", manifest)
    if set_latest:
        set_latest_version(store, version)
    logger.info("モデルを公開: version=%s files=%d", version, len(manifest["files"]))
    return manifest


def set_latest_version(store: ModelStore, version: str) -> None:
    """``latest.json`` を書き換える（新版の採用・ロールバックはこれだけ）。"""
    if store.read_json(f"{version}/{MANIFEST_NAME}") is None:
        raise FileNotFoundError(f"version {version!r} はストアに公開されていません")
    store.write_json(LATEST_NAME, {"version": version, "updated_at": datetime.now(timezone.utc).isoformat()})


def verify_dir(model_dir: Path, manifest: dict) -> None:
    for name, info in (manifest.get("files") or {}).items():
        p = model_dir / name
        if not p.is_file():
            raise ModelIntegrityError(f"{name} がありません")
        if _sha256(p) != info["sha256"]:
            raise ModelIntegrityError(f"{name} の sha256 が manifest と一致しません")


def fetch_version(store: ModelStore, version: str, dest_root: str | Path, *, strict_versions: bool = True) -> Path:
    """指定版を ``dest_root/<version>/`` へ取得して検証する。取得済み・検証OKなら再ダウンロードしない。"""
    manifest = store.read_json(f"{version}/{MANIFEST_NAME}")
    if manifest is None:
        raise FileNotFoundError(f"version {version!r} の manifest がストアにありません")
    check_compatibility(manifest, strict=strict_versions)

    dest = Path(dest_root) / version
    if dest.is_dir():
        try:
            verify_dir(dest, manifest)
            return dest
        except ModelIntegrityError:
            logger.warning("取得済みの %s は検証に失敗したため再取得します", dest)
            shutil.rmtree(dest, ignore_errors=True)

    Path(dest_root).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=str(dest_root)))
    try:
        for name in manifest["files"]:
            store.get_file(f"{version}/{name}", tmp / name)
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        verify_dir(tmp, manifest)
        tmp.replace(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def fetch_latest(store: ModelStore, dest_root: str | Path, *, strict_versions: bool = True) -> Path | None:
    """``latest.json`` が指す版を取得して返す。公開済みの版が無ければ ``None``。"""
    latest = store.read_json(LATEST_NAME)
    if not latest or not latest.get("version"):
        return None
    return fetch_version(store, str(latest["version"]), dest_root, strict_versions=strict_versions)
