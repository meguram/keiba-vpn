"""
パイプラインCLI

使い方:
  python -m pipeline.cli predict 20260315
  python -m pipeline.cli predict 20260315 --race-ids 202606010101,202606010102
  python -m pipeline.cli train
  python -m pipeline.cli train --features-dir data/features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def cmd_predict(args):
    from pipeline.race_day import RaceDayPipeline, PipelineConfig

    config = PipelineConfig(
        scrape_interval=args.interval,
        auto_login=not args.no_login,
        skip_existing_scrape=not args.force_scrape,
    )
    pipeline = RaceDayPipeline(config)

    race_ids = args.race_ids.split(",") if args.race_ids else None

    def progress(phase: str, detail: str):
        print(f"  [{phase}] {detail}", flush=True)

    try:
        results = pipeline.run(args.date, race_ids=race_ids, callback=progress)
    finally:
        pipeline.close()

    ok = sum(1 for r in results if not r.error)
    print(f"\n{'='*60}")
    print(f"完了: {ok}/{len(results)} レース")

    for r in results:
        if r.error:
            print(f"  ❌ {r.venue}{r.round_num}R {r.race_name}: {r.error}")
        else:
            print(f"  ✅ {r.venue}{r.round_num}R {r.race_name} ({len(r.predictions)}頭)")
            if not r.predictions.empty:
                top = r.predictions.head(3)
                for _, row in top.iterrows():
                    print(f"     {int(row['pred_rank'])}位: {row['horse_name']} (スコア {row['pred_score']:.1f})")


def cmd_train(args):
    from pipeline.trainer import ModelTrainer
    import yaml
    _cfg = {}
    try:
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        with open(cfg_path) as f:
            _cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    trainer = ModelTrainer(config=_cfg)
    metrics = trainer.train(features_dir=args.features_dir)

    if "error" in metrics:
        print(f"❌ {metrics['error']}")
        sys.exit(1)

    print(f"\n学習完了:")
    print(f"  AUC: {metrics.get('auc', 0):.4f}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Train: {metrics.get('n_train', 0)}行 / Test: {metrics.get('n_test', 0)}行")
    if metrics.get("top_features"):
        print(f"\n  重要特徴量 (Top 10):")
        for name, score in list(metrics["top_features"].items())[:10]:
            print(f"    {name}: {score:.1f}")


def main():
    parser = argparse.ArgumentParser(description="競馬予測パイプライン")
    sub = parser.add_subparsers(dest="command")

    p_pred = sub.add_parser("predict", help="指定日のレースを予測する")
    p_pred.add_argument("date", help="日付 (YYYYMMDD)")
    p_pred.add_argument("--race-ids", help="カンマ区切りのレースID")
    p_pred.add_argument("--interval", type=float, default=1.0, help="スクレイピング間隔(秒)")
    p_pred.add_argument("--no-login", action="store_true", help="ログインなし")
    p_pred.add_argument("--force-scrape", action="store_true", help="キャッシュ無視")

    p_train = sub.add_parser("train", help="モデルを学習する")
    p_train.add_argument("--features-dir", default="data/features", help="特徴量ディレクトリ")

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
