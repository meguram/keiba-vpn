#!/usr/bin/env python3
"""学習PC（ホスト）のローカルデータのうち、GCS に正本が無いものを GCS へバックアップする。

正本が GCS にあるスクレイピング結果（race_result など）は対象外。ローカルにしか無い派生データ
（特徴量・学習用テーブル・実験結果・血統 10 世代・UI 用テーブル・追走難度など）を、単位ごとに
tar.gz にして ``chuou/backup/host/<日付>/`` へ上げる。上書きも削除もしない（日付ごとに別の置き場）。

  python -m src.scripts.maintenance.backup_local_to_gcs --dry-run          # 対象と容量の確認のみ
  python -m src.scripts.maintenance.backup_local_to_gcs --execute          # 実行（ホストで）
  python -m src.scripts.maintenance.backup_local_to_gcs --execute --units local_features,models

方針:
  - 読み取りだけ（ローカルのファイルは変更しない）。権限で読めないものは ``sudo -n tar`` で読む。
  - 認証情報になりうるファイル（.env・cookie・鍵など）は全単位で除外する。
  - 1 つずつ直列で作って上げ、上げたら一時ファイルを消す（CPU・ディスクを使いすぎない）。nice で低優先度にする。
  - 上げた後に md5 とサイズを照合する。単位ごとに ``<unit>.done.json`` を書き、同じ日付で再実行すれば続きから。
  - SQLite（mlflow.db）は動作中に壊れた状態で写さないよう、バックアップ API で一貫した複製を作ってから上げる。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(os.environ.get("KEIBA_REPO_ROOT") or Path(__file__).resolve().parents[3])   # 別の場所に置いたコピーで実行するとき用
GCS_BACKUP_ROOT = "chuou/backup/host"

# 認証情報・ロック・一時物は全単位で除外する
COMMON_EXCLUDES = [
    "*/.env", "*/.env.*", "*.pem", "*.key", "*cookie*", "*credential*", "*secret*", "*.lock",
    "__pycache__", "*.pyc", ".ipynb_checkpoints",
]


@dataclass
class Unit:
    name: str
    paths: list[str]                       # REPO_ROOT からの相対パス
    excludes: list[str] = field(default_factory=list)
    note: str = ""


UNITS: list[Unit] = [
    Unit("page_reference", ["data/page_reference"], note="UI ポータブルバンドル（レース一覧・UI 用テーブル・メタ）"),
    Unit("calculated_data", ["data/calculated_data"], note="追走難度・成長曲線など計算済み（シンボリックリンクは辿らない）"),
    Unit("research", ["data/research"], note="血統研究の成果物（bloodline.db など）"),
    Unit("local_features", ["data/local/features"], note="特徴量ストア（base/race/horse/ped_tbl・ラベル）"),
    Unit("local_modeling", ["data/local/modeling"], excludes=["data/local/modeling/training_pipeline/scratch"],
         note="学習用テーブル・OOF・ハイパーパラメータ探索・実験結果・馬券検証（scratch は除く）"),
    Unit("local_pedigree_10gen", ["data/local/horse_pedigree_10gen"], note="10 世代血統（接ぎ木の成果物）"),
    Unit("local_backup", ["data/local/backup"], note="修復前のスナップショット"),
    Unit("local_meta", ["data/local/meta"], excludes=["data/local/meta/backfill_locks", "data/local/meta/netkeiba_global_slots"],
         note="データヘルスの台帳・再取得の進捗など（ロック類は除く）"),
    Unit("local_misc", ["data/local/horse_training", "data/local/megu_index", "data/local/ml", "data/local/page_quality",
                        "data/local/research", "data/local/image", "data/local/stallion_notes_overrides.json"], note="小さなローカルデータ"),
    Unit("knowledge_misc", ["data/knowledge", "data/meta", "data/jra_baba"], note="知識ベース・構造メタ・JRA 馬場"),
    Unit("models", ["models"], note="学習済みモデル（旧）・ペース予測・最終オッズ"),
    Unit("mlflow_runs", ["mlflow/runs"], excludes=["mlflow/runs/mlflow.db"], note="MLflow の実験記録（DB は別にバックアップ API で複製）"),
]
SQLITE_SNAPSHOTS = [("mlflow_db", "mlflow/runs/mlflow.db")]


def tar_command(unit: Unit, root: Path = REPO_ROOT, sudo: bool = True) -> list[str]:
    """単位を tar にするコマンド。存在しないパスは含めない。"""
    cmd = (["sudo", "-n"] if sudo else []) + ["tar", "-C", str(root), "--warning=no-file-changed", "-cf", "-"]
    for pat in COMMON_EXCLUDES:
        cmd += ["--exclude", pat]
    for ex in unit.excludes:
        cmd += ["--exclude", ex]
    cmd += [p for p in unit.paths if (root / p).exists()]
    return cmd


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def _bucket():
    sys.path.insert(0, str(REPO_ROOT))
    from src.scraper.storage import HybridStorage

    st = HybridStorage(base_dir=str(REPO_ROOT))
    if st._dev_store is not None or not st.gcs_enabled:
        raise RuntimeError("GCS に接続できない環境です（dev または GCS 未設定）。ホストで実行してください")
    return st._get_bucket()


def _du(unit: Unit, root: Path) -> int:
    total = 0
    for p in unit.paths:
        if (root / p).exists():
            out = subprocess.run(["sudo", "-n", "du", "-sb", str(root / p)], capture_output=True, text=True).stdout.split()
            total += int(out[0]) if out else 0
    return total


def _upload(bucket, local: Path, blob_name: str) -> dict:
    blob = bucket.blob(blob_name)
    blob.chunk_size = 32 << 20
    blob.upload_from_filename(str(local), checksum="md5", timeout=3600)
    blob.reload()
    ok = blob.size == local.stat().st_size and blob.md5_hash == _md5_b64(local)
    if not ok:
        raise RuntimeError(f"アップロード後の照合に失敗: {blob_name}")
    return {"blob": blob_name, "bytes": blob.size, "md5": blob.md5_hash}


def run_unit(bucket, unit: Unit, prefix: str, tmpdir: Path, root: Path = REPO_ROOT) -> dict:
    marker = bucket.blob(f"{prefix}/{unit.name}.done.json")
    if marker.exists():
        return {"unit": unit.name, "skipped": "done"}
    started = datetime.now(timezone.utc).isoformat()
    out = tmpdir / f"{unit.name}.tar.gz"
    cmd = tar_command(unit, root)
    with open(out, "wb") as fh:
        tar = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gz = subprocess.Popen(["nice", "-n", "19", "gzip", "-1"], stdin=tar.stdout, stdout=fh)
        tar.stdout.close()
        gz.wait()
        err = tar.stderr.read().decode(errors="replace")
        rc = tar.wait()
    if rc not in (0, 1) or gz.returncode != 0:      # tar の 1 は『読む間にファイルが変わった』（許容）
        out.unlink(missing_ok=True)
        raise RuntimeError(f"{unit.name}: tar 失敗 rc={rc} gz={gz.returncode}: {err[-400:]}")
    info = _upload(bucket, out, f"{prefix}/{unit.name}.tar.gz")
    out.unlink(missing_ok=True)
    rec = {"unit": unit.name, "paths": unit.paths, "excludes": unit.excludes, "note": unit.note, "started": started,
           "finished": datetime.now(timezone.utc).isoformat(), "tar_warnings": err.strip().splitlines()[:5], **info}
    marker.upload_from_string(json.dumps(rec, ensure_ascii=False, indent=1), content_type="application/json")
    return rec


def run_sqlite(bucket, name: str, rel: str, prefix: str, tmpdir: Path, root: Path = REPO_ROOT) -> dict:
    marker = bucket.blob(f"{prefix}/{name}.done.json")
    if marker.exists():
        return {"unit": name, "skipped": "done"}
    src = root / rel
    if not src.exists():
        return {"unit": name, "skipped": "missing"}
    snap = tmpdir / f"{name}.sqlite"
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(str(snap))
    with d:
        s.backup(d)
    s.close(); d.close()
    info = _upload(bucket, snap, f"{prefix}/{name}.sqlite")
    snap.unlink(missing_ok=True)
    rec = {"unit": name, "path": rel, **info, "finished": datetime.now(timezone.utc).isoformat()}
    marker.upload_from_string(json.dumps(rec, ensure_ascii=False, indent=1), content_type="application/json")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--units", default="", help="カンマ区切り。省略で全部")
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"), help="置き場の日付（再実行で続きから）")
    ap.add_argument("--tmpdir", default="")
    args = ap.parse_args()

    names = [n for n in args.units.split(",") if n]
    units = [u for u in UNITS if not names or u.name in names]
    prefix = f"{GCS_BACKUP_ROOT}/{args.date}"
    if args.dry_run:
        total = 0
        for u in units:
            b = _du(u, REPO_ROOT)
            total += b
            print(f"{u.name:24s} {b / 1e6:9.1f} MB  {u.note}")
        print(f"{'合計(圧縮前)':24s} {total / 1e6:9.1f} MB  → gs://<bucket>/{prefix}/")
        return 0

    bucket = _bucket()
    tmpdir = Path(args.tmpdir or tempfile.mkdtemp(prefix="keiba_backup_"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for u in units:
            print(f"[{datetime.now():%H:%M:%S}] {u.name} ...", flush=True)
            r = run_unit(bucket, u, prefix, tmpdir)
            results.append(r)
            print(f"   → {r.get('bytes', 0) / 1e6:.1f} MB {r.get('skipped', '')}", flush=True)
        if not names:
            for n, rel in SQLITE_SNAPSHOTS:
                r = run_sqlite(bucket, n, rel, prefix, tmpdir)
                results.append(r)
                print(f"   {n}: {r.get('bytes', 0) / 1e6:.1f} MB {r.get('skipped', '')}", flush=True)
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "date": args.date, "host": os.uname().nodename, "units": results}
        bucket.blob(f"{prefix}/manifest.json").upload_from_string(json.dumps(manifest, ensure_ascii=False, indent=1), content_type="application/json")
        print("manifest:", f"{prefix}/manifest.json")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
