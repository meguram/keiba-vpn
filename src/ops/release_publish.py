"""pull 型デプロイの配布側（ホスト PC で実行）。成果物を GCS に置き、最後に latest.json を書く（途中の状態は VPS が拾わない）。

  python -m src.ops.release_publish publish-ui --file data/local/releases/keiba-ui-<commit8>-<日時>.tar.gz [--dry-run]
  python -m src.ops.release_publish build-fetch [--out data/local/releases]     # 取得の実行役のコードバンドル（src/ と取得が読む config/ の数ファイル・requirements-fetch.txt）
  python -m src.ops.release_publish publish-fetch --file <fetch バンドル.tar.gz> [--dry-run]
  python -m src.ops.release_publish status [--kind ui|fetch]                     # latest.json と VPS の適用報告

latest.json の書き込み = 『prod に出す』という人の判断（昇格は常に手動）。VPS はメンテナンス窓（JST 03:00〜04:00）に取りに行く。
GCS の資格情報はホストの .env（GCS_BUCKET など）。dev PC では動かない（gcp_guard）。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "ops/releases"
KINDS = ("ui", "fetch")
_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
# 取得の実行役のバンドルに入れるもの（秘密・データ・ノートブック・学習物は入れない）
# config は取得が読むものだけ（会員名簿 members.json・費用・モデルの重みは入れない。足すときはここに明示する）
FETCH_CONFIG = ("ports.env", "settings.yaml", "entries_schedule.json", "fetch_owner.json", "anti_scraping.json",
                "maintenance.json", "gcp_job_map.json", "behavior_flags.json")
FETCH_INCLUDE = ("src", *(f"config/{n}" for n in FETCH_CONFIG), "requirements-fetch.txt", "scripts/auto_scrape.sh")
FETCH_EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints"}
FETCH_EXCLUDE_NAMES = {"anti_scraping.local.json", ".env"}
FETCH_EXCLUDE_SUFFIX = (".pyc", ".ipynb", ".parquet", ".pkl", ".joblib")


def key_of(kind: str, name: str) -> str:
    return f"{PREFIX}/{kind}/{name}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_latest(kind: str, release_id: str, sha256: str, size: int, *, commit: str = "", now: datetime | None = None) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"kind は {KINDS}")
    if not _ID.match(release_id) or not _SHA.match(sha256):
        raise ValueError("release_id または sha256 が不正")
    return {"schema": 1, "kind": kind, "release_id": release_id, "sha256": sha256, "size": int(size),
            "tarball": key_of(kind, f"{release_id}.tar.gz"), "commit": commit,
            "created_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")}


def _fetch_filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
    p = Path(ti.name)
    if FETCH_EXCLUDE_PARTS & set(p.parts) or p.name in FETCH_EXCLUDE_NAMES or p.name.startswith(".env") or p.name.endswith(FETCH_EXCLUDE_SUFFIX):
        return None
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def build_fetch_bundle(out_dir: Path, *, root: Path = ROOT, commit: str | None = None) -> Path:
    """取得の実行役のコードバンドルを作る。ID は fetch-<commit8>-<日時>（UI と同じく日時で新旧を比べる）。"""
    if commit is None:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    rid = f"fetch-{commit[:8]}-{time.strftime('%Y%m%d%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rid}.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for item in FETCH_INCLUDE:
            p = root / item
            if p.exists():
                tf.add(p, arcname=item, filter=_fetch_filter)
        meta = json.dumps({"release": rid, "commit": commit}).encode()
        ti = tarfile.TarInfo("RELEASE.json")
        ti.size = len(meta)
        tf.addfile(ti, io.BytesIO(meta))
    return path


def publish(io_: Any, kind: str, tarball: Path, *, commit: str = "", dry_run: bool = False) -> dict[str, Any]:
    """tar.gz・.sha256 を置き、直前の latest.json を previous.json に退避し、最後に latest.json を書く。"""
    rid = tarball.name[: -len(".tar.gz")]
    sha = sha256_file(tarball)
    side = Path(str(tarball) + ".sha256")
    if side.exists():
        want = side.read_text().split()[0]
        if want != sha:
            raise ValueError(f"{side.name} の sha256 が tar.gz と一致しません")
    latest = make_latest(kind, rid, sha, tarball.stat().st_size, commit=commit)
    if dry_run:
        return {"dry_run": True, "latest": latest}
    io_.upload_file(tarball, key_of(kind, tarball.name))
    io_.put_bytes(key_of(kind, tarball.name + ".sha256"), f"{sha}  {tarball.name}\n".encode(), content_type="text/plain")
    manifest = tarball.with_name(rid + ".manifest.json")
    if manifest.exists():
        io_.put_bytes(key_of(kind, manifest.name), manifest.read_bytes(), content_type="application/json")
    old = io_.get_bytes(key_of(kind, "latest.json"))
    if old:
        io_.put_bytes(key_of(kind, "previous.json"), old, content_type="application/json")
    io_.put_bytes(key_of(kind, "latest.json"), json.dumps(latest, ensure_ascii=False, indent=1).encode(), content_type="application/json")
    return {"dry_run": False, "latest": latest}


def status(io_: Any, kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("latest.json", "previous.json", "applied/vps.json"):
        raw = io_.get_bytes(key_of(kind, name))
        out[name] = json.loads(raw) if raw else None
    return out


def _gcs_io() -> Any:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from src.ops.gcs_io import GcsIO

    return GcsIO(os.environ.get("GCS_BUCKET", ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("publish-ui", "publish-fetch"):
        p = sub.add_parser(name)
        p.add_argument("--file", required=True)
        p.add_argument("--commit", default="")
        p.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("build-fetch")
    b.add_argument("--out", default=str(ROOT / "data/local/releases"))
    s = sub.add_parser("status")
    s.add_argument("--kind", choices=KINDS, default="ui")
    a = ap.parse_args(argv)
    if a.cmd == "build-fetch":
        print(build_fetch_bundle(Path(a.out)))
        return 0
    if a.cmd == "status":
        print(json.dumps(status(_gcs_io(), a.kind), ensure_ascii=False, indent=1))
        return 0
    kind = "ui" if a.cmd == "publish-ui" else "fetch"
    path = Path(a.file)
    if not path.name.endswith(".tar.gz") or not path.exists():
        print("tar.gz を指定してください", file=sys.stderr)
        return 2
    res = publish(None if a.dry_run else _gcs_io(), kind, path, commit=a.commit, dry_run=a.dry_run)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
