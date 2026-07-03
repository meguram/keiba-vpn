"""
旧 ``data/features/columns/*.parquet``（キー: race_id + horse_number）を
``race_horse_tbl``（キー: race_id + horse_id）へ移し、レジストリを更新する。

  python -m src.pipeline.migrate_legacy_feature_columns
  python -m src.pipeline.migrate_legacy_feature_columns --dry-run

前提: ``data/local/tables`` の race_shutuba に horse_id が含まれること。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.pipeline.features.feature_layout import BLOCK_RACE_HORSE_TBL, MERGE_KEYS_RACE_HORSE_TBL
from src.pipeline.features.feature_store import FeatureStore
from src.pipeline.features.id_value_policy import sanitize_netkeiba_string_id
from src.pipeline.features.race_performance import FEATURE_COLUMNS
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)


def _shutuba_horse_map(store: FeatureStore) -> pd.DataFrame:
    ys = store.available_years()
    if not ys:
        return pd.DataFrame()
    df = store.load_source("race_shutuba", years=ys, columns=["horse_id"])
    df["horse_id"] = sanitize_netkeiba_string_id(df["horse_id"])
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce")
    df = df.dropna(subset=["race_id", "horse_number", "horse_id"])
    return df[["race_id", "horse_number", "horse_id"]].drop_duplicates()


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = FeatureStore(base_dir=args.base_dir)
    legacy = store._features_dir / "columns"
    if not legacy.is_dir():
        logger.info("columns/ なし — スキップ")
        return 0

    mmap = _shutuba_horse_map(store)
    if mmap.empty:
        logger.warning("race_shutuba から馬番→horse_id マップを作れませんでした")
        return 1

    reg_path = store._registry_path
    registry: dict = {}
    if reg_path.is_file():
        registry = json.loads(reg_path.read_text(encoding="utf-8"))

    migrated = 0
    for f in sorted(legacy.glob("*.parquet")):
        stem = f.stem
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            logger.warning("読み込み失敗 %s: %s", f, e)
            continue
        if "race_id" not in df.columns or "horse_number" not in df.columns:
            continue
        # レパフォ列または旧レジストリの custom 列を対象
        meta = registry.get(stem, {})
        is_race_perf = stem in FEATURE_COLUMNS or meta.get("source") == "race_performance"
        is_custom = meta.get("source") == "custom"
        if not (is_race_perf or is_custom):
            continue

        hn = pd.to_numeric(df["horse_number"], errors="coerce")
        tmp = df.assign(horse_number=hn)
        m = tmp.merge(mmap, on=["race_id", "horse_number"], how="left")
        m = m.dropna(subset=["horse_id"])
        m = m.drop(columns=["horse_number"], errors="ignore")
        val_cols = [c for c in m.columns if c not in ("race_id", "horse_id")]
        if not val_cols:
            continue
        if args.dry_run:
            logger.info("[dry-run] %s -> %s 行", stem, len(m))
            migrated += 1
            continue
        store.save_feature_column(
            stem,
            m[["race_id", "horse_id", *val_cols]],
            table_block=BLOCK_RACE_HORSE_TBL,
            merge_keys=list(MERGE_KEYS_RACE_HORSE_TBL),
            overwrite=True,
            registry_extra={**{k: v for k, v in meta.items() if k not in ("rel_path", "table_block", "merge_keys")}, "migrated_from": "data/local/features/columns"},
        )
        f.unlink()
        migrated += 1
        logger.info("移行: %s (%d 行)", stem, len(m))

    logger.info("レパフォ等の移行: %d ファイル", migrated)

    # レジストリが新ブロックを指す場合、旧 columns/ の同名ファイルを削除
    removed = 0
    for f in sorted(legacy.glob("*.parquet")):
        stem = f.stem
        p = store._resolve_column_path(stem)
        if p is None:
            continue
        try:
            if p.resolve() == f.resolve():
                continue
        except OSError:
            continue
        if not args.dry_run:
            f.unlink()
            removed += 1
            logger.info("旧ファイル削除: %s（正は %s）", f.name, p)
    logger.info("完了: 移行=%d 旧削除=%d", migrated, removed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
