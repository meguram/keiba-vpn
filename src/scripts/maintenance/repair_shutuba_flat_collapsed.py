#!/usr/bin/env python3
"""race_shutuba_flat で 1 レース 1 頭に潰れたレースを、元の race_shutuba JSON から作り直して置き換える。

背景: 枠順確定前の出馬表は horse_number が全頭 0。以前の重複除去は (race_id, horse_number) だったので 1 行に潰れた（修正済み: horse_id キー）。
  python -m src.scripts.maintenance.repair_shutuba_flat_collapsed --year 2026            # 対象の確認（既定はドライラン）
  python -m src.scripts.maintenance.repair_shutuba_flat_collapsed --year 2026 --apply    # バックアップを取って置き換え

- 対象: flat の行数が元 JSON の頭数より少ないレース。元 JSON の頭数が flat 以下（欠けている）なら触らない。
- 置き換える前の flat は data/local/backup/shutuba_flat_collapsed/ に退避する。他のレースの行は変えない。
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger("maintenance.repair_shutuba_flat_collapsed")
BACKUP_DIR = Path("data/local/backup/shutuba_flat_collapsed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", default="2026")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.scraper import export_tables as et
    from src.scraper.run import ScraperRunner

    st = ScraperRunner(auto_login=False).storage
    if st._dev_store is not None or not st.gcs_enabled:
        raise SystemExit("GCS に接続できない環境（dev または未設定）です。KEIBA_ENV=stg で実行すること")

    path = et.OUTPUT_DIR / args.year / "race_shutuba_flat.parquet"
    old = pd.read_parquet(path)
    counts = old.groupby("race_id").size()
    suspects = [str(r) for r in counts.index if counts[r] == 1]       # 1 行のレースだけが対象候補（1 頭立てはまず無い）
    logger.info("flat %d 行 / %d レース、1 行のレース %d", len(old), len(counts), len(suspects))

    fixed: dict[str, list[dict]] = {}
    for rid in suspects:
        data = st.load("race_shutuba", rid, bypass_cache=True)
        entries = [e for e in ((data or {}).get("entries") or []) if isinstance(e, dict)]
        if len(entries) <= 1:
            logger.info("  %s: 元 JSON %d 頭（復元できない）", rid, len(entries))
            continue
        idx = {rid: {f: data.get(f) for f in et.RACE_META_FIELDS}}
        rows = et._data_to_flat_rows(data, "race_shutuba", idx)
        fixed[rid] = rows
        logger.info("  %s: 1 → %d 頭", rid, len(rows))
    by_date = pd.Series({r: str(old.loc[old["race_id"] == r, "date"].iloc[0]) for r in fixed}).value_counts().sort_index() if fixed else None
    logger.info("復元できるレース %d / 候補 %d\n%s", len(fixed), len(suspects), by_date)
    if not args.apply or not fixed:
        logger.info("ドライラン（--apply で置き換え）")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    bak = BACKUP_DIR / f"{args.year}_race_shutuba_flat_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    shutil.copy2(path, bak)
    logger.info("バックアップ: %s", bak)
    keep = old[~old["race_id"].astype(str).isin(fixed)]
    new = pd.DataFrame([r for rows in fixed.values() for r in rows])
    for c in old.columns:
        if c not in new.columns:
            new[c] = None
    merged = pd.concat([keep, new[list(old.columns) + [c for c in new.columns if c not in old.columns]]], ignore_index=True)
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)
    logger.info("置き換え完了: %d → %d 行", len(old), len(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
