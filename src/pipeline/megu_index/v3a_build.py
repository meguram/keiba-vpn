"""めぐ指数 v3a の基準期間づくり・変種の評価用 CLI。

  # 1) 全年の途中フレーム（v3a を掛ける前）を pickle に保存（変種の比較を速くする）
  KEIBA_ENV=stg python -m src.pipeline.megu_index.v3a_build dump-frames --out-dir /tmp/v3a_frames --years 2020-2026
  # 2) 基準期間からモデルを作って保存（DB は係数・基準タイムの SELECT のみ。書き込まない）
  KEIBA_ENV=stg python -m src.pipeline.megu_index.v3a_build build --frames-dir /tmp/v3a_frames --ref-years 2020-2024 \
      --out-dir config/megu_v3a --beta-weight 0.1 --pace-cap 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.pipeline.megu_index.recompute import parse_years
from src.pipeline.megu_index.v3a import V3ASpec, build_model

logger = logging.getLogger(__name__)


def dump_frames(years: list[int], out_dir: Path, model_version: str | None = None) -> None:
    from src.db.session import get_session, init_engine
    from src.pipeline.megu_index.compute import (
        MODEL_VERSION, _load_cushion, _load_par_time, _load_regression_params, _load_result_flat,
        compute_for_dataframe, merge_cushion_tsi,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    init_engine()
    mv = model_version or MODEL_VERSION
    with get_session() as session:  # SELECT のみ
        params = _load_regression_params(session, mv)
        df_par = _load_par_time(session, mv)
    pd.to_pickle(params, out_dir / "params.pkl")
    for y in years:
        df_year = _load_result_flat(y)
        if df_year.empty:
            continue
        df_flat = merge_cushion_tsi(df_year.copy(), _load_cushion(y), params)
        full = compute_for_dataframe(df_flat, params, df_par, df_hist=None, df_ref=df_year, return_full=True)
        full.to_pickle(out_dir / f"{y}.pkl")
        logger.info("dump %d: %d 行", y, len(full))


def load_frames(frames_dir: Path, years: list[int]) -> pd.DataFrame:
    return pd.concat([pd.read_pickle(Path(frames_dir) / f"{y}.pkl") for y in years], ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-frames")
    d.add_argument("--out-dir", required=True)
    d.add_argument("--years", default="2020-2026")
    b = sub.add_parser("build")
    b.add_argument("--frames-dir", required=True)
    b.add_argument("--ref-years", default="2020-2024")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--beta-weight", type=float, default=V3ASpec.beta_weight)
    b.add_argument("--pace-cap", type=float, default=V3ASpec.pace_cap)
    args = ap.parse_args()
    if args.cmd == "dump-frames":
        dump_frames(parse_years(args.years), Path(args.out_dir))
        return
    frames_dir = Path(args.frames_dir)
    params = pd.read_pickle(frames_dir / "params.pkl")
    spec = V3ASpec(beta_weight=args.beta_weight, pace_cap=args.pace_cap)
    ref = load_frames(frames_dir, parse_years(args.ref_years))
    model = build_model(ref, spec, beta_pace=params.get("beta_pace", 0.0),
                        meta={"ref_years": args.ref_years, "rows": int(len(ref))})
    model.save(Path(args.out_dir))
    logger.info("saved model: %s (par cells=%d)", args.out_dir, len(model.par))


if __name__ == "__main__":
    main()
