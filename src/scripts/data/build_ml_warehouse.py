"""ML 向けデータ倉庫を構築する。

- 2020–2025 のレース系: GCS / キャッシュ経由で Parquet を data/local/tables に用意しつつ、
  data/ml/warehouse に symlink で陳列。
- 馬系: horse_id 先頭4桁（生年コホート）ごとに SQLite シャード + catalog.sqlite3 で索引。

Usage:
  python -m src.scripts.data.build_ml_warehouse --years 2020,2021,2022,2023,2024,2025
  python -m src.scripts.data.build_ml_warehouse --layout-only
  python -m src.scripts.data.build_ml_warehouse --sqlite-only --race-linked-horses
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.scraper.export_tables import ALL_CATEGORIES, export_year
from src.scraper.storage import HybridStorage
from src.scripts.data.ml_warehouse.layout import materialize_layout
from src.scripts.data.ml_warehouse.paths import DEFAULT_YEARS, manifest_path, warehouse_root
from src.scripts.data.ml_warehouse.sqlite_builder import (
    horse_ids_from_race_results,
    rebuild_horse_sqlite,
)
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="ML 向けデータ倉庫（Parquet 陳列 + 馬 SQLite）")
    ap.add_argument(
        "--years",
        default=",".join(DEFAULT_YEARS),
        help="カンマ区切り年度 (default: 〜2025)",
    )
    ap.add_argument("--base-dir", default=".", help="リポジトリルート")
    ap.add_argument(
        "--export-parquet",
        action="store_true",
        help="各年・全カテゴリを export_tables で生成（欠損分のみ GCS 取得）",
    )
    ap.add_argument(
        "--layout-only",
        action="store_true",
        help="symlink + manifest のみ（export はしない）",
    )
    ap.add_argument(
        "--sqlite-only",
        action="store_true",
        help="SQLite 更新のみ",
    )
    ap.add_argument(
        "--no-sqlite",
        action="store_true",
        help="SQLite を更新しない",
    )
    ap.add_argument(
        "--race-linked-horses",
        action="store_true",
        help="2020–2025 race_result に出た馬に限定して horse_result / training を取り込む",
    )
    ap.add_argument(
        "--no-training",
        action="store_true",
        help="追切テーブルを更新しない",
    )
    ap.add_argument(
        "--force-symlinks",
        action="store_true",
        help="既存 symlink を置き換える",
    )
    ap.add_argument(
        "--migrate-monolith",
        action="store_true",
        help="旧 keiba_horse_ml.sqlite3 をシャードへ移行してから終了",
    )
    ap.add_argument(
        "--shards",
        default="",
        help="SQLite 更新対象シャードのみ (例: 2023,2024)。空なら全件",
    )
    args = ap.parse_args()

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    base = Path(args.base_dir).resolve()
    shard_filter = (
        {s.strip() for s in args.shards.split(",") if s.strip()}
        if args.shards.strip()
        else None
    )

    if args.migrate_monolith:
        from src.scripts.data.ml_warehouse.migrate_monolith import migrate_monolith_to_shards

        print(migrate_monolith_to_shards(base))
        return 0

    if not args.sqlite_only:
        if args.export_parquet and not args.layout_only:
            storage = HybridStorage(str(base))
            for y in years:
                logger.info("=== export_tables %s ===", y)
                export_year(
                    y,
                    categories=list(ALL_CATEGORIES),
                    storage=storage,
                )

        man = materialize_layout(base, years, force_symlinks=args.force_symlinks)
        if args.layout_only:
            print(json.dumps(man, ensure_ascii=False, indent=2))
            return 0

    if not args.no_sqlite:
        hid_filter = None
        if args.race_linked_horses:
            hid_filter = horse_ids_from_race_results(base, years)

        st = rebuild_horse_sqlite(
            base,
            years,
            horse_id_filter=hid_filter,
            include_training=not args.no_training,
            shard_filter=shard_filter,
        )
        logger.info("sqlite: %s", st)

    mp = manifest_path(base)
    warehouse_root(base).mkdir(parents=True, exist_ok=True)
    doc: dict = {}
    if mp.exists():
        try:
            doc = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
    doc.setdefault("layout_version", 1)
    doc.setdefault("warehouse_root", str(warehouse_root(base)))
    doc["sqlite"] = {
        "layout": "sharded_by_horse_id_prefix4",
        "catalog": "sqlite/catalog.sqlite3",
        "shard_glob": "sqlite/horses/{prefix4}.sqlite3",
        "shard_key": "horse_id 先頭4桁（生年コホート。GCS horse_result/{prefix4}/ と一致）",
        "tables_per_shard": {
            "horse_profile": {"pk": ["horse_id"]},
            "horse_race_history": {"pk": ["horse_id", "race_id"]},
            "horse_training_row": {"pk": ["horse_id", "training_date", "seq"]},
        },
        "catalog_tables": {
            "shard_registry": "シャード別件数・ファイルサイズ",
            "horse_lookup": "horse_id → shard",
        },
        "access": "src.scripts.data.ml_warehouse.sqlite_access",
    }
    doc["export_categories_for_ml"] = list(ALL_CATEGORIES)
    mp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"warehouse: {warehouse_root(base)}")
    print(f"manifest: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
