#!/usr/bin/env python3
"""
厩舎コメント再スクレイピングスクリプト

既存のrace_resultデータからrace_idリストを取得し、
厩舎コメントのみを再スクレイピングする。
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.client import NetkeibaClient
from scraper.parsers import TrainerCommentParser
from scraper.storage import HybridStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/rescrape_comments.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="厩舎コメント再スクレイピング")
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2021", "2022", "2023", "2024", "2025", "2026"],
        help="対象年度（デフォルト: 2021-2026）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="処理件数上限（デフォルト: 0=無制限）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存データも上書き",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("厩舎コメント再スクレイピング開始")
    logger.info("対象年度: %s", ", ".join(args.years))
    logger.info("=" * 60)

    storage = HybridStorage()
    client = NetkeibaClient(auto_login=True)
    comment_parser = TrainerCommentParser()

    # race_resultから全レースIDを取得
    result_keys = storage.list_keys("race_result")
    logger.info("race_result キー数: %d", len(result_keys))

    # 年度でフィルタ
    target_keys = [k for k in result_keys if k[:4] in args.years]
    logger.info("対象レース数: %d", len(target_keys))

    if args.limit > 0:
        target_keys = target_keys[:args.limit]
        logger.info("処理上限適用: %d件", args.limit)

    processed = 0
    skipped = 0
    errors = 0
    success = 0

    for i, race_id in enumerate(target_keys, 1):
        try:
            # 既存データチェック
            if not args.force:
                existing = storage.load("race_trainer_comment", race_id)
                if existing and existing.get("entries"):
                    skipped += 1
                    if i % 100 == 0:
                        logger.info(
                            "[%d/%d] スキップ: %s (既存データあり)",
                            i, len(target_keys), race_id
                        )
                    continue

            # スクレイピング
            url = f"https://race.netkeiba.com/race/comment.html?race_id={race_id}"
            html = client.fetch(url, use_cache=False)

            if not html or len(html) < 500:
                errors += 1
                logger.warning("[%d/%d] エラー: %s (HTMLが空)", i, len(target_keys), race_id)
                continue

            # パース
            data = comment_parser.parse(html, race_id)

            if not data.get("entries"):
                errors += 1
                logger.warning(
                    "[%d/%d] エラー: %s (コメント0件)",
                    i, len(target_keys), race_id
                )
                continue

            # 保存
            storage.save("race_trainer_comment", race_id, data)
            success += 1
            processed += 1

            if i % 50 == 0:
                logger.info(
                    "[%d/%d] 成功: %s (%d件)",
                    i, len(target_keys), race_id, len(data["entries"])
                )

            # レート制限
            time.sleep(0.5)

        except Exception as e:
            errors += 1
            logger.error("[%d/%d] 例外: %s - %s", i, len(target_keys), race_id, e)
            continue

    logger.info("=" * 60)
    logger.info("処理完了")
    logger.info("  成功: %d件", success)
    logger.info("  スキップ: %d件", skipped)
    logger.info("  エラー: %d件", errors)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
