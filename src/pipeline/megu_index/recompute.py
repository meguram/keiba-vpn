"""めぐ指数を全期間・同じ版で作り直し、別ディレクトリに出力する（稼働データは書き換えない）。

背景（2026-10-19）: 年ごと・バッチごとに計算の版が混ざり、勝ち馬の中央値が 2024 以前≈50 / 2025-01〜2026-04≈94 / 2026-05 以降≈55 と
段差になっていた。parquet は ``(race_id, horse_id)`` で新しい行だけ上書きするマージのため、古い版の行が残り続けた。

  KEIBA_ENV=stg python -m src.pipeline.megu_index.recompute --out-dir data/local/modeling/megu_recompute_20261019 --years 2020-2026

DB は係数（megu_regression_params）と基準タイム（megu_par_time）の SELECT のみ。DB・GCS・page_reference へは書かない。
delta_level（レベル補正）は v3 の指数に使わないため過去履歴を渡さず 0 のまま（列は互換のために残る）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.pipeline.megu_index.compute import (
    MODEL_VERSION,
    _load_cushion,
    _load_par_time,
    _load_regression_params,
    _load_result_flat,
    _save_megu_parquet_backup,
    compute_for_dataframe,
    frame_for_parquet,
    merge_cushion_tsi,
)

logger = logging.getLogger(__name__)


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


def recompute_year(year: int, params: dict[str, float], df_par: pd.DataFrame, out_dir: Path) -> dict:
    df_year = _load_result_flat(year)
    if df_year.empty:
        return {"year": year, "rows": 0, "skipped": "no_flat"}
    df_flat = merge_cushion_tsi(df_year.copy(), _load_cushion(year), params)
    df_result = compute_for_dataframe(df_flat, params, df_par, df_hist=None, df_ref=df_year)
    df_save = frame_for_parquet(df_result, df_year)
    info = _save_megu_parquet_backup(df_save, year, out_dir=out_dir, replace=True)
    return {"year": year, "rows": len(df_save), "path": info["local_path"],
            "status": df_save["megu_index_status"].value_counts().to_dict()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="めぐ指数の再計算（別ディレクトリ出力。稼働データは触らない）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--years", default="2020-2026")
    ap.add_argument("--model-version", default=MODEL_VERSION)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    from src.db.session import get_session, init_engine

    init_engine()
    with get_session() as session:  # SELECT のみ
        params = _load_regression_params(session, args.model_version)
        df_par = _load_par_time(session, args.model_version)
    if not params or df_par.empty:
        raise SystemExit("係数または基準タイムが DB に無い")
    for yr in parse_years(args.years):
        logger.info("再計算: %d", yr)
        logger.info("結果: %s", recompute_year(yr, params, df_par, out_dir))


if __name__ == "__main__":
    main()
