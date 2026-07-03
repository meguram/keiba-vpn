#!/usr/bin/env python3
"""
ペース予測モデル（前半1F / 3F ラップタイム）を学習し MLflow に登録する。

  python -m src.scripts.data.train_pace_model
  python -m src.scripts.data.train_pace_model --years 2022,2023,2024,2025
  python -m src.scripts.data.train_pace_model --max-races 500   # デバッグ用

学習後、models/pace_predictor/model_1f.txt と model_3f.txt に保存される。
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.utils.mlflow_client import init_mlflow
from src.utils.logger import get_logger

logger = get_logger("train_pace")


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ペース予測モデル学習 (MLflow)")
    parser.add_argument(
        "--years",
        default="2022,2023,2024,2025",
        help="学習対象年（カンマ区切り、デフォルト: 2022,2023,2024,2025）",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=None,
        help="デバッグ用: 処理レース数上限",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="テストデータ比率（デフォルト: 0.2）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="詳細ログを stderr に出力",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    logger.info("学習年: %s", years)

    from src.pipeline.models.pace_predictor import PacePredictorTrainer, build_training_dataset_from_parquet

    # MLflow URI を解決（サーバー不在時はローカルストアへフォールバック）
    uri = init_mlflow()
    logger.info("MLflow URI: %s", uri)

    print("データセット構築中 (parquet)...", file=sys.stderr)
    df = build_training_dataset_from_parquet(years=years)

    if df.empty:
        print("ERROR: 学習データが見つかりません", file=sys.stderr)
        return 1

    if args.max_races:
        df = df.head(args.max_races)
        logger.info("デバッグ制限: %d レースに限定", args.max_races)

    print(f"学習データ: {len(df)} レース", file=sys.stderr)

    trainer = PacePredictorTrainer(mlflow_tracking_uri=uri)
    print("モデル学習中...", file=sys.stderr)
    metrics = trainer.train(df=df, test_ratio=args.test_ratio)

    if "error" in metrics:
        print(f"ERROR: {metrics['error']}", file=sys.stderr)
        return 1

    print("\n=== 学習結果 ===", file=sys.stderr)
    print(f"1F MAE  : {metrics['mae_1f']:.3f} 秒", file=sys.stderr)
    print(f"1F RMSE : {metrics['rmse_1f']:.3f} 秒", file=sys.stderr)
    print(f"3F MAE  : {metrics['mae_3f']:.3f} 秒", file=sys.stderr)
    print(f"3F RMSE : {metrics['rmse_3f']:.3f} 秒", file=sys.stderr)
    print(f"学習行数: {metrics['n_train']}, テスト行数: {metrics['n_test']}", file=sys.stderr)
    print(f"学習時間: {metrics['training_time_sec']:.1f} 秒", file=sys.stderr)
    print("MLflow 記録 + models/pace_predictor/ に保存完了", file=sys.stderr)

    import json
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
