"""学習済みモデル（バンドル）の配布。ホストが GCS に上げ（publish）、Cloud Run Jobs が検証つきで取る（fetch）。

レイアウト（バケット内）::

    models/<component>/<version>/<相対パス…>      # 実体
    models/<component>/<version>/manifest.json   # 各ファイルの size / sha256 と合計
    models/<component>/latest.json                # {version, manifest_sha256, …}。最後に書く（書き込み途中の版を誰も読まない）

- ``publish`` はホストのモデリングの最後に呼ぶ。同じ内容（ツリーのハッシュが同じ）なら再アップロードしない。
- ``fetch`` は ``latest.json`` → ``manifest.json`` の sha256 を確かめ、ファイルごとに size と sha256 を検証してから、
  一時ディレクトリに作ったものを置き換える。**検証に失敗したら何も置かない**（古い版が残る）。合計サイズの上限を超えるときも取らない。
- 学習・評価の成果物は GCS → ホストの方向には流さない（外向き転送が有料のため。architecture_target.md 5.3）。
- コンポーネント名と環境変数の対応は ``config/gcp_job_map.json`` の ``models``。

CLI: ``python -m src.ops.model_publish publish --component bundle_v5 [--from DIR] | status | fetch --component X --dest DIR``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ops.gcs_io import BlobIO, read_json, write_json

PREFIX = "models"
MB = 1024 * 1024
CHUNK = 1024 * 1024


class ModelIntegrityError(RuntimeError):
    """sha256 / size の不一致、manifest の不一致。"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_files(src: Path) -> list[tuple[str, Path]]:
    """(相対パス, 実体)。単一ファイルならファイル名 1 件。"""
    src = Path(src)
    if src.is_file():
        return [(src.name, src)]
    if not src.is_dir():
        raise FileNotFoundError(f"モデルのパスがありません: {src}")
    return [(f.relative_to(src).as_posix(), f) for f in sorted(src.rglob("*")) if f.is_file() and not f.name.startswith(".")]


def build_manifest(component: str, version: str, files: list[tuple[str, Path]], *, now: datetime | None = None) -> dict[str, Any]:
    rows = [{"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)} for rel, p in files]
    return {"component": component, "version": version, "files": rows, "total_bytes": sum(r["size"] for r in rows),
            "published_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")}


def tree_digest(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["path"]):
        h.update(f"{r['path']}\0{r['size']}\0{r['sha256']}\n".encode())
    return h.hexdigest()


def _manifest_key(component: str, version: str) -> str:
    return f"{PREFIX}/{component}/{version}/manifest.json"


def publish(io: BlobIO, component: str, src: Path, *, version: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    files = tree_files(src)
    if not files:
        raise FileNotFoundError(f"空のモデルディレクトリ: {src}")
    now = now or datetime.now(timezone.utc)
    manifest = build_manifest(component, "pending", files, now=now)
    digest = tree_digest(manifest["files"])
    version = version or f"{now.strftime('%Y%m%d')}-{digest[:8]}"
    manifest["version"] = version
    existing = read_json(io, _manifest_key(component, version))
    if existing and tree_digest(existing.get("files") or []) == digest:
        uploaded = 0
    else:
        for rel, p in files:
            io.upload_file(p, f"{PREFIX}/{component}/{version}/{rel}")
        uploaded = len(files)
        write_json(io, _manifest_key(component, version), manifest)
    raw = io.get_bytes(_manifest_key(component, version)) or b""
    latest = {"component": component, "version": version, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "total_bytes": manifest["total_bytes"], "published_at": manifest["published_at"]}
    write_json(io, f"{PREFIX}/{component}/latest.json", latest)      # 最後に書く
    return {"component": component, "version": version, "files": len(files), "uploaded_files": uploaded, "total_mb": round(manifest["total_bytes"] / MB, 1)}


def resolve_version(io: BlobIO, component: str, version: str = "latest") -> dict[str, Any]:
    """latest.json（または指定版）→ manifest を検証して返す。"""
    if version == "latest":
        latest = read_json(io, f"{PREFIX}/{component}/latest.json")
        if not latest:
            raise ModelIntegrityError(f"{component}: latest.json がありません")
        version = latest["version"]
        want = latest.get("manifest_sha256")
    else:
        want = None
    raw = io.get_bytes(_manifest_key(component, version))
    if raw is None:
        raise ModelIntegrityError(f"{component}/{version}: manifest.json がありません")
    if want and hashlib.sha256(raw).hexdigest() != want:
        raise ModelIntegrityError(f"{component}/{version}: manifest の sha256 が latest.json と一致しません")
    return json.loads(raw.decode("utf-8"))


def fetch(io: BlobIO, component: str, dest_root: Path, *, version: str = "latest", max_mb: int | None = None) -> Path:
    """検証つきで ``dest_root/<component>/<version>/`` に取得して、そのパスを返す。取得済み（検証済み印あり）なら何もしない。"""
    manifest = resolve_version(io, component, version)
    version = manifest["version"]
    if max_mb and manifest["total_bytes"] > max_mb * MB:
        raise ModelIntegrityError(f"{component}/{version}: {manifest['total_bytes'] / MB:.0f}MB が上限 {max_mb}MB を超える")
    final = Path(dest_root) / component / version
    marker = final / ".verified.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("digest") == tree_digest(manifest["files"]):
                return final
        except (OSError, json.JSONDecodeError):
            pass
    Path(dest_root).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{component}.", dir=str(dest_root)))
    try:
        for r in manifest["files"]:
            dest = tmp / r["path"]
            if not io.download_file(f"{PREFIX}/{component}/{version}/{r['path']}", dest):
                raise ModelIntegrityError(f"{component}/{version}: {r['path']} がありません")
            if dest.stat().st_size != r["size"] or sha256_file(dest) != r["sha256"]:
                raise ModelIntegrityError(f"{component}/{version}: {r['path']} の size / sha256 が一致しません")
        (tmp / ".verified.json").write_text(json.dumps({"digest": tree_digest(manifest["files"]), "version": version}), encoding="utf-8")
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def env_for(doc_models: dict[str, Any], component: str, fetched: Path) -> tuple[str | None, str]:
    """(環境変数名 | None, 値)。単一ファイルのコンポーネント（job_map の ``entry`` を持つ）は、その中のファイルを指す。"""
    spec = doc_models[component]
    entry = spec.get("entry")
    return spec.get("env"), str(fetched / entry) if entry else str(fetched)


def install(doc_models: dict[str, Any], component: str, fetched: Path, root: Path) -> Path | None:
    """固定パスを読むコード向け: ``install_to``（リポジトリ相対）に取得物へのシンボリックリンクを張る。無ければ None。"""
    spec = doc_models[component]
    dest_rel = spec.get("install_to")
    if not dest_rel:
        return None
    src = Path(env_for(doc_models, component, fetched)[1])
    dest = Path(root) / dest_rel
    if ".." in Path(dest_rel).parts or dest_rel.startswith("/"):
        raise ValueError(f"install_to が不正: {dest_rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink() if (dest.is_symlink() or dest.is_file()) else shutil.rmtree(dest)
    dest.symlink_to(src)
    return dest


def status(io: BlobIO, components: list[str]) -> list[dict[str, Any]]:
    out = []
    for c in components:
        latest = read_json(io, f"{PREFIX}/{c}/latest.json")
        out.append({"component": c, "published": bool(latest), **({"version": latest["version"], "published_at": latest.get("published_at"),
                                                                    "total_mb": round(latest.get("total_bytes", 0) / MB, 1)} if latest else {})})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--component", required=True)
    p.add_argument("--from", dest="src", default=None, help="ローカルのパス（既定: job_map の local_path）")
    p.add_argument("--version", default=None)
    sub.add_parser("status")
    f = sub.add_parser("fetch")
    f.add_argument("--component", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--version", default="latest")
    ns = ap.parse_args(argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()           # ホストの cron/手動実行でも .env の GCS_*（バケット・サービスアカウント）を読む
    from src.ops import job_map
    from src.ops.gcs_io import default_io

    models = job_map.load()["models"]
    io = default_io()
    if ns.cmd == "status":
        print(json.dumps(status(io, sorted(models)), ensure_ascii=False, indent=1))
        return 0
    if ns.component not in models:
        print(f"未知のコンポーネント: {ns.component}（{sorted(models)}）", file=sys.stderr)
        return 2
    if ns.cmd == "publish":
        src = Path(ns.src or models[ns.component]["local_path"])
        print(json.dumps(publish(io, ns.component, src, version=ns.version), ensure_ascii=False))
        return 0
    path = fetch(io, ns.component, Path(ns.dest), version=ns.version)
    print(json.dumps({"component": ns.component, "path": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
