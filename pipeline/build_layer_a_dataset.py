#!/usr/bin/env python3
"""
LayerA 学習用 Parquet を生成する CLI。

  python3 -m pipeline.build_layer_a_dataset
  python3 -m pipeline.build_layer_a_dataset --years 2022,2023,2024

出力:
  data/modeling/layer_a_train.parquet
  data/modeling/layer_a_train.meta.json
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.layer_a_dataset import (  # noqa: E402
    DEFAULT_META_JSON,
    DEFAULT_OUTPUT_PARQUET,
    write_layer_a_dataset,
)
from utils.keiba_logging import script_basic_config  # noqa: E402


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="LayerA（直前予測整合）学習母表のビルド")
    ap.add_argument(
        "--years",
        default="",
        help="カンマ区切り（省略時は data/local/tables にある全ての年）",
    )
    ap.add_argument(
        "--no-jra-filter",
        action="store_true",
        help="JRA 限定フィルタを無効化（地方・海外混在データ向け）",
    )
    ap.add_argument(
        "--no-history-stats",
        action="store_true",
        help="馬・騎手・調教師の履歴累計特徴を付けない（v1 相当のみ）",
    )
    ap.add_argument(
        "--jockey-trainer-stats",
        action="store_true",
        help="騎手・調教師の拡張統計（data/features/race_horse_tbl/<年>/jt_race_features.parquet）を結合（先に build_jockey_trainer_stats が必要）",
    )
    ap.add_argument(
        "--no-track-bias-pedigree",
        action="store_true",
        help="当日・前日馬場傾向×血統特徴を付けない（pipeline/track_bias_pedigree.py）",
    )
    ap.add_argument(
        "--track-bias-same-day-weight",
        type=float,
        default=None,
        help="馬場傾向の加重合成: 当日（該当レース前）の重み（既定 0.75）",
    )
    ap.add_argument(
        "--track-bias-prev-day-weight",
        type=float,
        default=None,
        help="馬場傾向の加重合成: 前日の重み（既定 0.25）。same+prev は正規化して使用",
    )
    ap.add_argument(
        "--track-bias-weights-json",
        type=Path,
        default=None,
        help="research.tune_track_bias_weights の出力 JSON（指定時は --track-bias-*-weight より優先）",
    )
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT_PARQUET), help="出力 Parquet")
    ap.add_argument("--meta", default=str(DEFAULT_META_JSON), help="メタ JSON")
    args = ap.parse_args()

    years = None
    if args.years.strip():
        years = [x.strip() for x in args.years.split(",") if x.strip()]

    os.chdir(ROOT)
    info = write_layer_a_dataset(
        years=years,
        jra_only=not args.no_jra_filter,
        include_history_stats=not args.no_history_stats,
        include_jockey_trainer_stats=args.jockey_trainer_stats,
        include_track_bias_pedigree=not args.no_track_bias_pedigree,
        track_bias_same_day_weight=args.track_bias_same_day_weight,
        track_bias_prev_day_weight=args.track_bias_prev_day_weight,
        track_bias_weights_json=args.track_bias_weights_json,
        output_parquet=Path(args.output),
        meta_path=Path(args.meta),
        base_dir=ROOT,
    )
    print(info)


if __name__ == "__main__":
    main()
