"""
``data/local/tables`` の race_result / race_shutuba から馬ID集合を作り、
``horse_training`` タスクをスクレイピングキューへ一括投入する。

ワーカー（API のキュー処理、または ``python -m scraper.queue_worker``）が処理すると、
GCS の ``horse_training`` とローカル ``data/local/horse_training/{shard4}/`` に保存される。

例::

    python3 -m research.enqueue_horse_training_all
    python3 -m research.enqueue_horse_training_all --dry-run
    # 全キューを空にしてから全馬再投入（precheck 即時 → 未所持のみ待機、ワーカキック）
    python3 -m research.enqueue_horse_training_all --replace-all
    # 全消去なしで投入だけしつつ、手元でワーカを起動したい場合
    python3 -m research.enqueue_horse_training_all --kick
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.build_horse_entity_store import collect_horse_ids_from_tables  # noqa: E402
from pipeline.feature_store import TABLES_DIR  # noqa: E402
from scraper.job_queue import ScrapeJobQueue, kick_process_queue_background  # noqa: E402
from utils.keiba_logging import script_basic_config  # noqa: E402


def main() -> None:
    script_basic_config()
    log = logging.getLogger(__name__)
    ap = argparse.ArgumentParser(description="全馬 horse_training をキュー投入")
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="プロジェクトルート（tables は base-dir/data/local/tables）",
    )
    ap.add_argument(
        "--years",
        nargs="*",
        default=None,
        help="対象年（省略で tables 配下の全年ディレクトリ）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="馬ID数だけ表示しキューには載せない",
    )
    ap.add_argument(
        "--no-smart-skip",
        action="store_true",
        help="GCS+ローカル既存があっても上書き取得寄り（ワーカの smart_skip を false）",
    )
    ap.add_argument(
        "--replace-all",
        action="store_true",
        help="先にキュー上の全ジョブ（待機・precheck 等含む）を一時停止付きで空にしてから一括投入する",
    )
    ap.add_argument(
        "--no-precheck-now",
        action="store_true",
        help="投入直後の一括 precheck（存在確認）をスキップ（次回 process_queue 冒頭で実施）",
    )
    ap.add_argument(
        "--kick",
        action="store_true",
        help="投入・precheck 後に process_queue をバックグラウンド起動（--replace-all 時は既定でキック、それを止めるには --no-kick）",
    )
    ap.add_argument(
        "--no-kick",
        action="store_true",
        help="ワーカ（process_queue）を起動しない（--replace-all の既定キックも止める）",
    )
    args = ap.parse_args()

    tables_dir = (args.base_dir / TABLES_DIR).resolve()
    horse_ids = sorted(
        collect_horse_ids_from_tables(tables_dir, list(args.years) if args.years else None)
    )
    if not horse_ids:
        raise SystemExit(f"horse_id が0件です: tables_dir={tables_dir}")

    log.info("対象馬: %d 頭（%s）", len(horse_ids), tables_dir)
    if args.dry_run:
        log.info("dry-run のためキュー投入しません")
        return

    q = ScrapeJobQueue()
    if args.replace_all:
        nrm = q.clear_all_jobs_with_transport_pause()
        log.info("replace-all: キュー全消去 %d 件", nrm)
    # smart_skip=True: 新規行は precheck 経由。一括 I/O 後、満たす行は completed、取得要は pending。
    stats = q.add_horse_jobs_bulk(
        horse_ids,
        ["horse_training"],
        priority=0,
        smart_skip=not bool(args.no_smart_skip),
    )
    log.info(
        "投入完了: created=%(created)d requeued=%(requeued)d duplicate=%(duplicate)d "
        "skipped_complete=%(skipped_already_complete)d",
        stats,
    )
    if not args.no_precheck_now:
        pc = q.run_storage_precheck_horse_now()
        log.info(
            "precheck(即時): from_precheck=%(from_precheck)d to_completed=%(to_completed)d "
            "to_pending=%(to_pending)d",
            pc,
        )
    do_kick = (bool(args.replace_all) or bool(args.kick)) and not args.no_kick
    if do_kick:
        kick_process_queue_background()
        log.info("process_queue をバックグラウンド起動（キック）")


if __name__ == "__main__":
    main()
