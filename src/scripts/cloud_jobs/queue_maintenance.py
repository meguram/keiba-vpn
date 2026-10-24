#!/usr/bin/env python3
"""
スクレイピングキューの定期メンテナンスを1回実行する CLI。

ストール中（running のまま放置）のジョブ回収、failed ジョブの pending への復元
（アクセス一時停止中はスキップ）、完了レコードの削除、長時間 failed のままの
ジョブの Slack 通知を行う。

元は src/api/app.py 内の daemon thread（既定 1時間ごと、環境変数
``SCRAPE_QUEUE_HOURLY_MAINTENANCE_SEC``）が呼んでいた
``src.scraper.job_queue.run_hourly_queue_maintenance`` をそのまま呼び出す薄いラッパー。
処理ロジック自体はこの CLI では実装しない。

Cloud Scheduler + Cloud Run Jobs から1時間ごとに起動する想定
（docs/operations/gcp-cloud-run-jobs.md 参照）。

Usage:
  python -m src.scripts.cloud_jobs.queue_maintenance
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    from src.scraper.job_queue import run_hourly_queue_maintenance

    result = run_hourly_queue_maintenance()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
