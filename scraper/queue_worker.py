"""
スクレイピングキュー処理ワーカー

バックグラウンドでキューを監視し、ジョブを順次処理する。
"""

import logging
import time
from pathlib import Path

from scraper.job_queue import ScrapeJobQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """キューワーカーメインループ"""
    queue = ScrapeJobQueue()
    logger.info("キューワーカー起動")

    while True:
        try:
            # キューに未処理ジョブがあるか確認
            next_job = queue.get_next_job()
            if not next_job:
                queue.requeue_stale_running_jobs(assume_lock_holder=False)
                next_job = queue.get_next_job()

            if next_job:
                logger.info("未処理ジョブ検出: %s", next_job.get("race_id"))
                queue.process_queue()
            else:
                # ジョブがない場合は15秒待機
                time.sleep(15)

        except KeyboardInterrupt:
            logger.info("キューワーカー終了")
            break
        except Exception as e:
            logger.error("ワーカーエラー: %s", e, exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
