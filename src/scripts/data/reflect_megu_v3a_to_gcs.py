"""めぐ指数 v3a の parquet を GCS の正本へ反映する（退避してから上書き。ロールバックも同じスクリプト）。

対象: gs://<bucket>/chuou/data/preprocessed/netkeiba/pc/megu_index/<年>/megu_index_flat.parquet（compute._GCS_MEGU_BLOB_TPL と同じ）
上げるもの: ホストの data/page_reference/tables/<年>/megu_index_flat.parquet（工程4-1で v3a 版に差し替え済み）

手順（既定は dry-run＝読み取りのみ。--execute で書き込む。何度実行しても同じ結果）:
  1. 年ごとに、現行のオブジェクトを退避先（既定 chuou/backup/megu_index_before_v3a_20261019/<年>/）へコピーし、md5 を照合。
     退避先に既にあれば上書きしない（最初の退避を守る）。
  2. ローカルの parquet を現行のオブジェクトへアップロードし、md5 を照合。すでに同じ md5 なら何もしない。
ロールバック: --rollback --execute で、退避先から現行のオブジェクトへ戻す（md5 照合）。

使い方（ホスト）::

  KEIBA_ENV=stg python3 -m src.scripts.data.reflect_megu_v3a_to_gcs                      # dry-run（一覧と計画）
  KEIBA_ENV=stg python3 -m src.scripts.data.reflect_megu_v3a_to_gcs --execute            # 退避→アップロード
  KEIBA_ENV=stg python3 -m src.scripts.data.reflect_megu_v3a_to_gcs --rollback --execute # 退避から戻す
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BLOB_TPL = "chuou/data/preprocessed/netkeiba/pc/megu_index/{year}/megu_index_flat.parquet"
BACKUP_TPL = "chuou/backup/megu_index_before_v3a_20261019/{year}/megu_index_flat.parquet"
TABLES_DIR = Path("data/page_reference/tables")


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def plan_year(local_md5: str, current_md5: str | None, backup_md5: str | None) -> dict:
    """1 年分の計画（純粋関数）。backup=退避する必要があるか、upload=アップロードする必要があるか。"""
    return {
        "backup": current_md5 is not None and backup_md5 is None,
        "upload": current_md5 != local_md5,
        "backup_conflict": current_md5 is not None and backup_md5 is not None and backup_md5 != current_md5 and current_md5 != local_md5,
    }


def plan_rollback(current_md5: str | None, backup_md5: str | None) -> dict:
    return {"restore": backup_md5 is not None and current_md5 != backup_md5, "missing_backup": backup_md5 is None}


def _md5_of(bucket, name: str) -> str | None:
    blob = bucket.get_blob(name)
    return None if blob is None else blob.md5_hash


def run_forward(bucket, years: list[int], tables_dir: Path, execute: bool) -> list[dict]:
    out = []
    for y in years:
        local = tables_dir / str(y) / "megu_index_flat.parquet"
        if not local.exists():
            raise SystemExit(f"{y}: ローカルの parquet が無い: {local}")
        cur_name, bak_name = BLOB_TPL.format(year=y), BACKUP_TPL.format(year=y)
        lmd5, cmd5, bmd5 = md5_b64(local), _md5_of(bucket, cur_name), _md5_of(bucket, bak_name)
        pl = plan_year(lmd5, cmd5, bmd5)
        rec = {"year": y, "local_bytes": local.stat().st_size, "current_md5": cmd5, "backup_md5": bmd5, "local_md5": lmd5, **pl, "done": []}
        if pl["backup_conflict"]:
            logger.warning("%d: 退避先に別の内容が既にある（上書きしない）。現行は未退避の可能性: 手動確認が必要", y)
        if execute:
            if pl["backup"]:
                cur = bucket.get_blob(cur_name)
                bucket.copy_blob(cur, bucket, bak_name)
                if _md5_of(bucket, bak_name) != cmd5:
                    raise SystemExit(f"{y}: 退避のコピー後の md5 が一致しない")
                rec["done"].append("backup")
            if pl["upload"]:
                if cmd5 is not None and not (bmd5 is not None or "backup" in rec["done"]):
                    raise SystemExit(f"{y}: 退避が無いままの上書きは行わない")
                blob = bucket.blob(cur_name)
                blob.chunk_size = 32 << 20
                blob.upload_from_filename(str(local), content_type="application/octet-stream", checksum="md5", timeout=3600)
                blob.reload()
                if blob.md5_hash != lmd5:
                    raise SystemExit(f"{y}: アップロード後の md5 が一致しない")
                rec["done"].append("upload")
        out.append(rec)
        logger.info("%d: %s", y, {k: rec[k] for k in ("local_bytes", "backup", "upload", "done")})
    return out


def run_rollback(bucket, years: list[int], execute: bool) -> list[dict]:
    out = []
    for y in years:
        cur_name, bak_name = BLOB_TPL.format(year=y), BACKUP_TPL.format(year=y)
        cmd5, bmd5 = _md5_of(bucket, cur_name), _md5_of(bucket, bak_name)
        pl = plan_rollback(cmd5, bmd5)
        rec = {"year": y, "current_md5": cmd5, "backup_md5": bmd5, **pl, "done": []}
        if pl["missing_backup"]:
            raise SystemExit(f"{y}: 退避が無いので戻せない（{bak_name}）")
        if execute and pl["restore"]:
            bak = bucket.get_blob(bak_name)
            bucket.copy_blob(bak, bucket, cur_name)
            if _md5_of(bucket, cur_name) != bmd5:
                raise SystemExit(f"{y}: 復元後の md5 が一致しない")
            rec["done"].append("restore")
        out.append(rec)
        logger.info("%d: %s", y, {k: rec[k] for k in ("restore", "done")})
    return out


def _bucket():
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    if getattr(st, "_dev_store", None) is not None or not st.gcs_enabled:
        raise SystemExit("GCS に接続できない環境です（dev または GCS 未設定）。ホストで実行してください")
    return st._get_bucket()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="指定しなければ dry-run（読み取りのみ）")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--years", default="2020-2026")
    ap.add_argument("--tables-dir", default=str(TABLES_DIR))
    args = ap.parse_args()
    bucket = _bucket()
    years = parse_years(args.years)
    if args.rollback:
        run_rollback(bucket, years, args.execute)
    else:
        run_forward(bucket, years, Path(args.tables_dir), args.execute)
    if not args.execute:
        logger.info("dry-run: GCS への書き込みはしていません")


if __name__ == "__main__":
    main()
