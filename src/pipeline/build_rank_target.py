"""
``race_result_flat`` から着順ラベルを抽出し、年別 Parquet に書き出す。

出力::

  data/features/target/rank_tbl/<YYYY>/rank.parquet

列: ``race_id``, ``horse_id``, ``rank``（ソースの ``finish_position`` を数値化。失敗は null）。

  python -m src.pipeline.build_rank_target
  python -m src.pipeline.build_rank_target --years 2024 2025 --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.pipeline.features.feature_layout import FEATURES_DIR, RANK_TBL_SUBDIR
from src.pipeline.features.feature_store import TABLES_DIR
from src.pipeline.features.id_value_policy import sanitize_netkeiba_string_id
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"


def _read_race_result_year(base: Path, year: str) -> pd.DataFrame:
    path = base / TABLES_DIR / year / "race_result_flat.parquet"
    if not path.is_file():
        return pd.DataFrame()
    need = ["race_id", "horse_id", "finish_position"]
    schema = pq.read_schema(path)
    missing = [c for c in need if c not in schema.names]
    if missing:
        logger.warning("%s: 列不足のためスキップ missing=%s", path, missing)
        return pd.DataFrame()
    return pq.read_table(path, columns=need).to_pandas()


def build_rank_tbl_for_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["race_id", "horse_id", "rank"])
    out = df.dropna(subset=["race_id"]).copy()
    out["horse_id"] = sanitize_netkeiba_string_id(out["horse_id"])
    out = out.dropna(subset=["horse_id"])
    out["rank"] = pd.to_numeric(out["finish_position"], errors="coerce")
    out = out[["race_id", "horse_id", "rank"]].drop_duplicates(subset=["race_id", "horse_id"], keep="last")
    out["race_id"] = out["race_id"].astype(str)
    out["horse_id"] = out["horse_id"].astype("string")
    out["rank"] = out["rank"].astype("Int64")
    return out.reset_index(drop=True)


def write_rank_target_parquets(
    *,
    base_dir: str | Path = ".",
    years: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    base = Path(base_dir)
    store_tables = base / TABLES_DIR
    out_root = base / FEATURES_DIR / RANK_TBL_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    if years is None:
        if not store_tables.is_dir():
            raise FileNotFoundError(f"ソース無し: {store_tables}")
        years = sorted(d.name for d in store_tables.iterdir() if d.is_dir() and d.name.isdigit())
    years = [str(y) for y in years]

    written: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    for y in years:
        ydf = _read_race_result_year(base, y)
        if ydf.empty:
            skipped.append({"year": y, "reason": "race_result_flat 無しまたは空"})
            continue
        rank_df = build_rank_tbl_for_year(ydf)
        if rank_df.empty:
            skipped.append({"year": y, "reason": "有効行なし"})
            continue
        ydir = out_root / y
        ydir.mkdir(parents=True, exist_ok=True)
        out_path = ydir / "rank.parquet"
        if out_path.is_file() and not overwrite:
            skipped.append({"year": y, "reason": "既存ファイルあり（--overwrite で上書き）"})
            continue
        rank_df.to_parquet(out_path, index=False)
        written[y] = str(out_path.relative_to(base))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(TABLES_DIR / "<YYYY>" / "race_result_flat.parquet"),
        "source_column_rank": "finish_position → rank (numeric, coerce null)",
        "merge_keys": ["race_id", "horse_id"],
        "output_pattern": str(FEATURES_DIR / RANK_TBL_SUBDIR / "<YYYY>" / "rank.parquet"),
        "years_requested": years,
        "files_written": written,
        "skipped": skipped,
    }
    mp = out_root / MANIFEST_NAME
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "rank_tbl: wrote %d year-files → %s manifest=%s",
        len(written),
        out_root,
        mp,
    )
    return manifest


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="着順ラベル rank_tbl を生成")
    ap.add_argument("--base-dir", type=Path, default=Path("."))
    ap.add_argument("--years", nargs="*", help="年（省略時は tables にある全年）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    try:
        write_rank_target_parquets(
            base_dir=args.base_dir,
            years=list(args.years) if args.years else None,
            overwrite=args.overwrite,
        )
    except Exception as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
