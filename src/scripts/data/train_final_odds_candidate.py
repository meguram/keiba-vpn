"""最終オッズ予測モデルの候補を、本番の bundle（models/final_odds_bundle.json）を上書きせずに学習する。

python -m src.scripts.data.train_final_odds_candidate --halflife 1.0
- 学習 2020〜2024、評価 2025（既定）。新しい年を重く（半減期）。bundle は models/final_odds_bundle.candidate.json に保存する。
- MLflow には学習の記録と Registry の新しいバージョンが残る（追加のみ。既存バージョンは消さない）。
- 本番へ切り替えるときは、評価値を現行と比べてから bundle のファイルを入れ替える（旧版は退避）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.ports import mlflow_tracking_uri
from src.pipeline.models import final_odds_trainer as fot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", default="2020,2021,2022,2023,2024")
    ap.add_argument("--eval-years", default="2025")
    ap.add_argument("--halflife", type=float, default=1.0)
    ap.add_argument("--candidate", default="models/final_odds_bundle.candidate.json")
    a = ap.parse_args()
    fot.LOCAL_BUNDLE_PATH = Path(a.candidate)                      # 本番の bundle を上書きしない
    res = fot.FinalOddsTrainer(mlflow_tracking_uri()).train(
        train_years=tuple(a.train_years.split(",")), eval_years=tuple(a.eval_years.split(",")), recency_halflife_years=a.halflife)
    keep = {k: v for k, v in res.items() if k != "feature_names"}
    print(json.dumps(keep, ensure_ascii=False, indent=1, default=str)[:2500])
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    raise SystemExit(main())
