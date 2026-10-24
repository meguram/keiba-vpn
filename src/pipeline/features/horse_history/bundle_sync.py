"""サービング用バンドルを GCS 経由で小さなメモリのマシンへ配る。

  publish: ホストで feature-accumulate の後に実行。バンドルを ``chuou/models/serving_bundle/<版>/`` へ上げ、
           最後に ``LATEST.json``（どの版が最新か）を書く。古い版は ``KEEP`` 版だけ残して消す（このプレフィックス内のみ）。
  fetch:   小さなマシンで実行。``LATEST.json`` の版が手元と違うときだけ、一時ディレクトリへ取得して入れ替える。

    python -m src.pipeline.features.horse_history.bundle_sync publish
    python -m src.pipeline.features.horse_history.bundle_sync fetch [--dest DIR]

配布先にも GCS の読み取り権限（``.env`` の ``GCS_*``）が要る。版のディレクトリは書き込み後に変更しないので、取得中に新しい版が
公開されても、`LATEST.json` を読んだ時点の版を最後まで取れる。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GCS_PREFIX = "chuou/models/serving_bundle"
LATEST = "LATEST.json"
KEEP = 3
JST = timezone(timedelta(hours=9))


def _bucket():
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    if st._dev_store is not None or not st.gcs_enabled:
        raise RuntimeError("GCS に接続できない環境です（dev または GCS 未設定）")
    return st._get_bucket()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version_of(bundle_dir: Path) -> str:
    """版の名前: バンドルの built_at（JST）を YYYYMMDDTHHMMSS にしたもの。"""
    meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
    return datetime.fromisoformat(meta["built_at"]).strftime("%Y%m%dT%H%M%S")


def publish(bundle_dir: Path, bucket: Any = None, keep: int = KEEP) -> dict:
    bucket = bucket or _bucket()
    files = sorted(p for p in bundle_dir.iterdir() if p.is_file())
    if not any(p.name == "meta.json" for p in files):
        raise FileNotFoundError(f"meta.json がありません: {bundle_dir}")
    ver = version_of(bundle_dir)
    manifest = {}
    for p in files:
        blob = bucket.blob(f"{GCS_PREFIX}/{ver}/{p.name}")
        md5 = _md5(p)
        manifest[p.name] = {"size": p.stat().st_size, "md5": md5}
        if blob.exists():
            blob.reload()
            import base64

            if base64.b64decode(blob.md5_hash).hex() == md5:
                continue   # 同じ版を再実行したとき、同じ内容は上げ直さない
        blob.upload_from_filename(str(p), timeout=600)
    meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
    pointer = {"version": ver, "published_at": datetime.now(JST).isoformat(timespec="seconds"), "files": manifest,
               "history_last_date": meta.get("history_last_date"), "rows": meta.get("rows")}
    bucket.blob(f"{GCS_PREFIX}/{LATEST}").upload_from_string(json.dumps(pointer, ensure_ascii=False, indent=1), content_type="application/json")
    removed = _prune(bucket, keep)
    return {"version": ver, "files": len(files), "bytes": sum(m["size"] for m in manifest.values()), "pruned_versions": removed}


def _versions(bucket: Any) -> list[str]:
    vs = set()
    for b in bucket.list_blobs(prefix=f"{GCS_PREFIX}/"):
        parts = b.name[len(GCS_PREFIX) + 1:].split("/")
        if len(parts) == 2:
            vs.add(parts[0])
    return sorted(vs)


def _prune(bucket: Any, keep: int) -> list[str]:
    """最新 keep 版以外を消す。このプレフィックスの下の版ディレクトリだけが対象。"""
    vs = _versions(bucket)
    old = vs[:-keep] if keep > 0 else []
    for v in old:
        for b in list(bucket.list_blobs(prefix=f"{GCS_PREFIX}/{v}/")):
            b.delete()
    return old


def fetch(dest: Path, bucket: Any = None) -> dict:
    bucket = bucket or _bucket()
    blob = bucket.blob(f"{GCS_PREFIX}/{LATEST}")
    if not blob.exists():
        raise FileNotFoundError("LATEST.json がありません（まだ公開されていません）")
    pointer = json.loads(blob.download_as_text())
    ver = pointer["version"]
    cur = dest / "meta.json"
    if cur.is_file() and version_of(dest) == ver:
        return {"version": ver, "updated": False}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".bundle_", dir=str(dest.parent)))
    try:
        for name, m in pointer["files"].items():
            target = tmp / name
            bucket.blob(f"{GCS_PREFIX}/{ver}/{name}").download_to_filename(str(target))
            if target.stat().st_size != m["size"] or _md5(target) != m["md5"]:
                raise IOError(f"取得したファイルが一致しません: {name}")
        if dest.exists():
            old = dest.with_name(dest.name + ".old")
            shutil.rmtree(old, ignore_errors=True)
            dest.rename(old)
            tmp.rename(dest)
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"version": ver, "updated": True, "history_last_date": pointer.get("history_last_date")}


def main() -> int:
    from src.pipeline.features.horse_history.serving import default_bundle_dir
    from src.utils.project_env import load_project_dotenv

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("publish")
    sp.add_argument("--bundle-dir", type=Path)
    sf = sub.add_parser("fetch")
    sf.add_argument("--dest", type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    load_project_dotenv()
    if args.cmd == "publish":
        print(json.dumps(publish(args.bundle_dir or Path(default_bundle_dir())), ensure_ascii=False))
    else:
        print(json.dumps(fetch(args.dest or Path(default_bundle_dir())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
