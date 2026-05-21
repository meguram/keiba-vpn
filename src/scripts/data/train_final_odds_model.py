#!/usr/bin/env python3
"""
最終オッズ予測モデルを学習し MLflow に登録する。

  学習: 2020-2024 全レース（race_result ベース）
  評価: 2025 年データ

例:
  python -m src.scripts.data.train_final_odds_model
  python -m src.scripts.data.train_final_odds_model --max-races 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src.pipeline.models.final_odds_progress import STATUS_PATH, load_status_file
from src.pipeline.models.final_odds_trainer import FinalOddsTrainer
from src.utils.logger import get_logger

logger = get_logger("train_final_odds")


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
    parser = argparse.ArgumentParser(description="最終オッズ予測モデル学習 (MLflow)")
    parser.add_argument(
        "--train-years",
        default="2020,2021,2022,2023,2024",
        help="学習対象年（カンマ区切り）",
    )
    parser.add_argument(
        "--eval-years",
        default="2025",
        help="評価対象年（カンマ区切り）",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=None,
        help="デバッグ用: 処理レース数上限",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="進行中ジョブのステータス JSON を表示して終了",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="詳細ログを stderr に出力",
    )
    args = parser.parse_args()

    if args.status_only:
        print(json.dumps(load_status_file(), ensure_ascii=False, indent=2))
        print(f"\nstatus file: {STATUS_PATH}", file=sys.stderr)
        return 0

    _setup_logging(args.verbose)
    train_years = tuple(y.strip() for y in args.train_years.split(",") if y.strip())
    eval_years = tuple(y.strip() for y in args.eval_years.split(",") if y.strip())

    print(
        f"進捗は stderr のバーと {STATUS_PATH} で確認できます。"
        f" 別ターミナル: python -m src.scripts.data.train_final_odds_model --status-only",
        file=sys.stderr,
    )

    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    trainer = FinalOddsTrainer(mlflow_tracking_uri=args.mlflow_uri)
    result = trainer.train(
        storage,
        train_years=train_years,
        eval_years=eval_years,
        max_races=args.max_races,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
