#!/usr/bin/env python3
"""
企画書どおりの一次モデル選定（model_selection_primary）を実行するエントリ。

準備（当日直前 LayerA 母表）::

  python3 -m src.pipeline.build_layer_a_dataset

学習::

  python3 -m src.pipeline.run_baseline_train

`data/modeling/layer_a_train.parquet` があれば最優先で読み込む。
`dataset_split_manifest.json` が存在すれば自動で 2020-23 / 2024 / 2025 分割を使用。
MLflow が localhost:5000 で待機している前提（未起動時は記録のみ警告）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.models.trainer import ModelTrainer  # noqa: E402
from src.utils.keiba_logging import script_basic_config  # noqa: E402


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="Baseline LightGBM 学習（マニフェスト分割）")
    ap.add_argument(
        "--features-dir",
        default="data/features",
        help="特徴量 CSV ディレクトリ（FeatureStore 優先）",
    )
    ap.add_argument(
        "--no-manifest",
        action="store_true",
        help="dataset_split_manifest を使わず従来の行比率分割",
    )
    ap.add_argument(
        "--manifest",
        default="data/meta/modeling/dataset_split_manifest.json",
        help="分割マニフェスト JSON",
    )
    ap.add_argument(
        "--protocol",
        default="model_selection_primary",
        help="protocols のキー",
    )
    ap.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    args = ap.parse_args()

    trainer = ModelTrainer(mlflow_tracking_uri=args.mlflow_uri)
    metrics = trainer.train(
        features_dir=args.features_dir,
        use_split_manifest=False if args.no_manifest else None,
        split_manifest_path=args.manifest,
        split_protocol=args.protocol,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    if metrics.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
