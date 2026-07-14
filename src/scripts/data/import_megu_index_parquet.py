"""
NB-04 出力 parquet を megu_index テーブル (model_version=v2) に取り込む。

例:
    KEIBA_ENV=stg python -m src.scripts.data.import_megu_index_parquet \\
        --megu-parquet notebooks/megu_index/output/nb04/megu_index.parquet \\
        --dataset-parquet notebooks/megu_index/output/nb01/megu_dataset.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.session import get_session, init_engine
from src.pipeline.megu_index.compute import MODEL_VERSION, _upsert_batch
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_upsert_frame(megu_path: Path, dataset_path: Path | None) -> pd.DataFrame:
    mi = pd.read_parquet(megu_path)
    need = {
        "race_id",
        "horse_id",
        "delta_pace_sec",
        "delta_track_sec",
        "corrected_time",
        "par_time_class_sec",
        "megu_index",
        "computation_status",
    }
    missing = need - set(mi.columns)
    if missing:
        raise ValueError(f"megu parquet missing columns: {sorted(missing)}")

    if dataset_path and dataset_path.exists():
        ds = pd.read_parquet(
            dataset_path,
            columns=["race_id", "horse_id", "finish_time_sec"],
        )
        df = mi.merge(ds, on=["race_id", "horse_id"], how="left")
    else:
        df = mi.copy()
        df["finish_time_sec"] = np.nan

    out = pd.DataFrame(
        {
            "race_id": df["race_id"].astype(str),
            "horse_id": df["horse_id"].astype(str),
            "finish_time_sec": pd.to_numeric(df.get("finish_time_sec"), errors="coerce"),
            "par_time_final": pd.to_numeric(df["par_time_class_sec"], errors="coerce"),
            "delta_pace_sec": pd.to_numeric(df["delta_pace_sec"], errors="coerce").fillna(0.0),
            "delta_track_sec": pd.to_numeric(df["delta_track_sec"], errors="coerce").fillna(0.0),
            "delta_weight_sec": 0.0,
            "delta_level_sec": 0.0,
            "adjusted_time_sec": pd.to_numeric(df["corrected_time"], errors="coerce"),
            "megu_index": pd.to_numeric(df["megu_index"], errors="coerce"),
            "field_quality": np.nan,
            "computation_status": df["computation_status"].astype(str),
            "front_split_sec": np.nan,
            "split_point_m": np.nan,
            "tsi_raw": np.nan,
        }
    )
    out.loc[out["computation_status"] != "valid", "megu_index"] = np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Import NB-04 megu_index parquet into DB")
    parser.add_argument(
        "--megu-parquet",
        type=Path,
        default=PROJECT_ROOT / "notebooks/megu_index/output/nb04/megu_index.parquet",
    )
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        default=PROJECT_ROOT / "notebooks/megu_index/output/nb01/megu_dataset.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    args = parser.parse_args()

    script_basic_config()
    df = build_upsert_frame(args.megu_parquet, args.dataset_parquet)
    logger.info("import rows=%d valid=%d", len(df), (df["computation_status"] == "valid").sum())

    init_engine()
    with get_session() as session:
        saved = _upsert_batch(session, df, batch_size=args.batch_size, model_version=args.model_version)
        session.commit()
    logger.info("UPSERT complete: %d rows (model_version=%s)", saved, args.model_version)


if __name__ == "__main__":
    main()
