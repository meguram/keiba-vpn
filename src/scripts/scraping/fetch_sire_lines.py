"""種牡馬ごとの『○○系』（netkeiba の血統ページの表記）を取得して台帳（sire_line_map.json）に貯める。

python -m src.scripts.scraping.fetch_sire_lines                  # 未取得の種牡馬を、出走の多い順に 1 頭ずつ一定ペースで取る。途中で止めても再開できる
python -m src.scripts.scraping.fetch_sire_lines --limit 5        # 5 ページだけ

対象: horse_sire_map.parquet の父（sire_id）。各種牡馬の産駒の血統ページを 1 枚ずつ取る（父のセルに系統が付く）。
守り: 同時 1 本・一定間隔（既定 60 秒。SIRE_LINE_PACE_SEC）。netkeiba への当日の総件数（共通カウンタ＋自分の分）が上限に近いときは取らずに待つ。アクセス制限の疑い（HTTP 400/403 の連続）が出たら止める。
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time

import pandas as pd

from src.scraper import sire_line_store as store

logger = logging.getLogger("fetch_sire_lines")

PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
DAILY_STOP = int(os.environ.get("SIRE_LINE_DAILY_STOP", "2300"))     # その日の総件数がこれ以上なら取らずに待つ（上限 2,700 に余裕を残す）
PACE_SEC = float(os.environ.get("SIRE_LINE_PACE_SEC", "60"))
MAX_CONSECUTIVE_ERRORS = 5


def _my_count_path():
    """このスクリプト自身の当日の取得数（JST）。共通の日次カウンタは root の所有で追記できないので、自分の分は別に数える。"""
    from datetime import datetime, timedelta, timezone

    day = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
    return store.default_path().parent / "requests" / f"{day}.count"


def my_count_today() -> int:
    try:
        return _my_count_path().stat().st_size
    except OSError:
        return 0


def _count_mine() -> None:
    p = _my_count_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "ab") as f:
        f.write(b".")


def pick_targets(sire_map: pd.DataFrame, have: set[str]) -> list[tuple[str, str]]:
    """[(種牡馬ID, 代表の産駒の馬ID)] を、産駒の多い順に。取得済みの種牡馬は除く。"""
    m = sire_map.dropna(subset=["sire_id"]).copy()
    cnt = m.groupby("sire_id")["horse_id"].agg(["count", "max"]).sort_values("count", ascending=False)
    return [(str(sid), str(r["max"])) for sid, r in cnt.iterrows() if str(sid) not in have]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="この件数だけ取って終わる（0=全部）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from src.research.race_analysis.runs import load_sire_map
    from src.scraper.client import NetkeibaClient
    from src.scraper.request_counter import count_today

    sm = load_sire_map()
    if sm is None:
        logger.error("horse_sire_map.parquet がありません")
        return 1
    have = set(store.load())      # 表記なしで残したものも取得済みとして扱う
    todo = pick_targets(sm, have)
    logger.info("対象 %d 頭 / 取得済み %d 頭", len(todo), len(have))
    client = NetkeibaClient(interval=2.0, cache_dir=None, auto_login=False)
    done = errors = 0
    stopped_early = False
    for sire_id, horse_id in todo:
        if args.limit and done >= args.limit:
            stopped_early = True
            break
        while count_today() + my_count_today() >= DAILY_STOP:
            logger.info("今日の総件数が %d に達しているので待機（10 分後に再確認）", DAILY_STOP)
            time.sleep(600)
        try:
            _count_mine()
            html = client.fetch(PED_URL.format(horse_id=horse_id))
            store.record_html(html, horse_id)
            got = sire_id in store.labels()
            logger.info("%s (産駒 %s): 系統 %s", sire_id, horse_id, store.labels().get(sire_id, "表記なし"))
            if not got:      # 表記が無い種牡馬は台帳に『空』で残して、何度も取りに行かない
                store.mark_missing(sire_id, horse_id)
            errors, done = 0, done + 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.warning("取得失敗 %s: %s (%d 連続)", horse_id, exc, errors)
            if errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error("失敗が続くので止めます（アクセス制限の可能性）")
                return 2
        time.sleep(PACE_SEC * random.uniform(0.9, 1.1))
    if not stopped_early:
        store.set_complete()      # 全部取り終えた印。これが付くまで、めぐ分析の『系統』は出さない
        logger.info("全種牡馬を取り終えました（完了の印を付けた）")
    logger.info("終了: 今回 %d 頭", done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
