"""
調教キューを 10 件だけにして投入し、process_queue で消化したあと
``data/local/horse_training`` に JSON が付いているか検証する。

別プロセスがキューロックを持っている場合は ``acquire_lock`` が失敗するので、
先に queue_worker / API のキュー処理を止めること。

例::

    SCRAPE_QUEUE_PARALLEL=1 python3 -m src.research.scripts.trial_horse_training_queue_10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.features.horse_entity_layout import horse_shard4  # noqa: E402
from src.scraper.job_queue import ScrapeJobQueue  # noqa: E402
from src.utils.keiba_logging import script_basic_config  # noqa: E402

# 過去の手動確認で調教ページに行があった馬 + result_tbl 先頭付近（多めに取り、重複除去で10頭）
_TRIAL_IDS_RAW = [
    "2021105143",
    "2023106216",
    "2022103639",
    "2018105689",
    "2020104623",
    "2021100003",
    "2021100005",
    "2021100006",
    "2021100008",
    "2021100009",
    "2021100011",
    "2021100012",
]


def _unique_ten() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in _TRIAL_IDS_RAW:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 10:
            break
    return out


def main() -> int:
    script_basic_config()
    log = logging.getLogger("trial_horse_training")

    ap = argparse.ArgumentParser(description="horse_training キュー10件トライアル")
    ap.add_argument(
        "--skip-process",
        action="store_true",
        help="キュー投入のみ（process_queue は呼ばない）",
    )
    args = ap.parse_args()

    os.chdir(_ROOT)
    horse_ids = _unique_ten()
    if len(horse_ids) < 10:
        log.error("試験用馬IDが10頭に足りません: %s", horse_ids)
        return 1

    q = ScrapeJobQueue()
    cleared = q.clear_all_jobs()
    log.info("clear_all_jobs: %d 件削除", cleared)

    stats = q.add_horse_jobs_bulk(
        horse_ids,
        ["horse_training"],
        priority=0,
        smart_skip=False,
    )
    log.info("投入: %s 馬ID=%s", stats, horse_ids)

    if args.skip_process:
        log.info("--skip-process のため process_queue は実行しません")
        return 0

    if q.is_locked():
        log.error(
            "キューロックが取得できません。別の queue_worker / API を停止してから再実行してください。"
        )
        return 2

    os.environ.setdefault("SCRAPE_QUEUE_PARALLEL", "1")
    log.info("process_queue 開始（SCRAPE_QUEUE_PARALLEL=%s）", os.environ.get("SCRAPE_QUEUE_PARALLEL"))
    q.process_queue()
    log.info("process_queue 終了")

    from src.scraper.run import REPO_ROOT
    from src.scraper.horse_training_local import horse_training_local_root

    root = horse_training_local_root(REPO_ROOT)
    ok_paths: list[Path] = []
    missing: list[str] = []
    for hid in horse_ids:
        p = root / horse_shard4(hid) / f"{hid}.json"
        if p.is_file():
            ok_paths.append(p)
        else:
            missing.append(hid)

    log.info(
        "ローカル JSON: %d / %d 件 → %s",
        len(ok_paths),
        len(horse_ids),
        [x.name for x in ok_paths],
    )
    if missing:
        log.error("ローカル JSON なし: %s", missing)

    if len(ok_paths) != len(horse_ids):
        log.error("投入頭数とローカルファイル数が一致しません")
        return 3

    import json

    for p in ok_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("horse_id") != p.stem:
            log.error("horse_id 不一致: %s vs %s", p, data.get("horse_id"))
            return 4
        if not isinstance(data.get("entries"), list):
            log.error("entries が list でない: %s", p)
            return 5

    log.info("検証 OK（全頭分の JSON が正しい形式）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
