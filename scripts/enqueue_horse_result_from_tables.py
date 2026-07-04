#!/usr/bin/env python3
"""
data/page_reference/tables の parquet から horse_id を収集し、
horse_result のスクレイピングジョブをキューに一括投入する。

対象期間: 2024-01-01 〜 2026-05-20
上書き: True（既存データがあっても再取得）

実行:
  cd /var/www/megu-keiba/keiba-vpn
  /opt/venv/bin/python3 scripts/enqueue_horse_result_from_tables.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq

START_DATE = "20240101"
END_DATE   = "20260520"

TABLES_DIR = ROOT / "data" / "page_reference" / "tables"

# 対象年（2024〜2026）
TARGET_YEARS = ["2024", "2025", "2026"]

# horse_id が含まれるファイル（出馬表 flat が最も網羅的）
# フォールバックで race_result_flat も使う
CANDIDATE_FILES = [
    "race_shutuba_flat.parquet",
    "race_result_flat.parquet",
]


def collect_horse_ids() -> set[str]:
    horse_ids: set[str] = set()

    for year in TARGET_YEARS:
        year_dir = TABLES_DIR / year
        if not year_dir.exists():
            print(f"  ⚠ {year_dir} が存在しません。スキップ。")
            continue

        for fname in CANDIDATE_FILES:
            fpath = year_dir / fname
            if not fpath.exists():
                continue

            try:
                schema = pq.read_schema(str(fpath))
                cols = schema.names
                read_cols = ["horse_id"]
                if "date" in cols:
                    read_cols.append("date")

                table = pq.read_table(str(fpath), columns=read_cols)
                df = table.to_pandas()

                # 日付フィルタ
                if "date" in df.columns:
                    df["date"] = df["date"].astype(str).str.replace("-", "")
                    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

                ids = df["horse_id"].dropna().astype(str).unique()
                before = len(horse_ids)
                horse_ids.update(ids)
                added = len(horse_ids) - before
                print(f"  {year}/{fname}: {len(df):,} 行 → +{added:,} 頭 (累計 {len(horse_ids):,})")

            except Exception as e:
                print(f"  ⚠ {year}/{fname} 読み込みエラー: {e}")

    return horse_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="キューへの投入を行わず、対象頭数だけ表示する")
    args = parser.parse_args()

    print(f"=== horse_id 収集 ({START_DATE} 〜 {END_DATE}) ===")
    horse_ids = collect_horse_ids()
    horse_ids_sorted = sorted(horse_ids)
    print(f"\n対象馬: {len(horse_ids_sorted):,} 頭")

    if args.dry_run:
        print("\n[DRY RUN] キューへの投入はスキップしました。")
        print("サンプル (先頭10件):", horse_ids_sorted[:10])
        return

    print("\n=== キューへ投入中（overwrite=True）===")
    from src.scraper.job_queue import ScrapeJobQueue

    queue = ScrapeJobQueue()
    # horse_result は race 経由の別名。直接 horse ジョブは horse_profile を使う
    result = queue.add_horse_jobs_bulk(
        horse_ids=horse_ids_sorted,
        tasks=["horse_profile"],
        overwrite=True,
        smart_skip=False,
    )

    created   = result.get("created",   0)
    requeued  = result.get("requeued",  0)
    duplicate = result.get("duplicate", 0)

    print(f"\n✅ 完了")
    print(f"  新規追加:      {created:,}")
    print(f"  再キュー:      {requeued:,}")
    print(f"  重複スキップ:  {duplicate:,}")
    print(f"  合計投入:      {created + requeued:,} / {len(horse_ids_sorted):,} 頭")
    print("\nスクレイピング管理ページでワーカーを起動してください。")
    print("  https://megu-ai.megu-keiba.com/scrape?tab=control")


if __name__ == "__main__":
    main()
