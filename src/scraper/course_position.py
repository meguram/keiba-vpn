"""開催日ごとの「芝のコース区分（A・B・C・D＝柵の位置）」と、開催回・日目・芝/ダートの馬場状態を、netkeiba のレース一覧から集める。

取得元（ブラウザの race_list.js と同じ XHR。認証不要の HTML 断片。netkeiba_top_race_list.py と同じ経路）:
  1. GET .../top/race_list_get_date_list.html?kaisai_date=YYYYMMDD&encoding=UTF-8   … 近くの 4 日分の開催日と、その current_group
  2. GET .../top/race_list_sub.html?kaisai_date=YYYYMMDD&current_group=...         … 会場ごとのブロック
     <p class="RaceList_DataTitle"><small>4回</small> 東京 <small>2日目</small></p> / <span class="Shiba">芝(A)：良</span> / <span class="Da">ダ：良</span>
1 の応答には 4 日分の group が入るので、同じ group の日は 1 の取得を省く（1 日あたり約 1.3 リクエスト）。

保存: data/page_reference/course_position/course_position_<年>.json（git 管理外のローカルデータ）
  {"version": 1, "dates": {"2019-03-09": [{"venue": "中山", "kai": 1, "day": 1, "turf_course": "A", "turf_condition": "良", "dirt_condition": "良"}, ...]}}
  turf_course は A・B・C・D など（括弧の中の文字。無ければ None）。

取得の決まり（2026-11-01 に bot 制限を受けた経緯。docs/operations/backfill_2016.md。他の補填取得 backfill_horse_training_past と同じ作法）:
  * 既存の NetkeibaClient のスロットル（2.2〜4.0 秒＋バースト休止）に従う。認証は使わない（auto_login=False）。
  * 開始前と 1 日ごとに、停止の印（backfill_guard.halted）・アクセス保留（scrape_access_pause）・AccessGuard の trip を確かめ、
    他の取得が動いている間は待つ（wait_ok）。同時に 2 本は走らない（ロック）。
  * アクセス制限の疑い（HTTP 400/403/429）は AccessGuard で即停止し、再開しない（人が確認して再実行）。
  * 取得済みの日は飛ばす（途中から再開できる）。今日より前の日だけを対象にする（当日・未来の馬場状態を固定しない）。
  * 応答が確かにその日のものかを、払戻一覧のリンク（kaisai_date・kaisai_id の回・日目）で確かめる。突き合わない・解析が空・読み切れない
    ブロックがある日は保存せず、数件続いたら止める（書式の変更・制限ページ・group の食い違いの疑い）。

使い方:
    python -m src.scraper.course_position --from 2016 --to 2024 [--max 300]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import time
from datetime import date as _date
from pathlib import Path
from typing import Any, Callable, Iterable

from bs4 import BeautifulSoup

logger = logging.getLogger("scraper.course_position")

REPO = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("KEIBA_COURSE_POSITION_DIR") or REPO / "data/page_reference/course_position")
RACE_LISTS = REPO / "data/page_reference/race_lists"
BASE = "https://race.netkeiba.com/top"
JRA_VENUES = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
MAX_BAD_STREAK = 5            # 保存できない日（空・突き合わず・読み切れず・group なし）が続いたら止める
MAX_WAIT_SEC = 6 * 3600       # 他の取得を待つ上限
_TITLE = re.compile(r"(\d+)\s*回\s*(\S+?)\s*(\d+)\s*日目")
_SHIBA = re.compile(r"芝\s*(?:[（(]\s*([^)）]*?)\s*[)）])?\s*[：:]\s*(\S+)")
_DA = re.compile(r"ダ\s*(?:[（(]\s*([^)）]*?)\s*[)）])?\s*[：:]\s*(\S+)")
_PAYBACK = re.compile(r"kaisai_id=(\d{10})&(?:amp;)?kaisai_date=(\d{8})")


def parse_date_list(html: str) -> dict[str, str]:
    """race_list_get_date_list の HTML から {YYYYMMDD: current_group}。"""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, str] = {}
    for li in soup.select("li[date]"):
        g = li.get("group")
        if li.get("date") and g:
            out[str(li["date"])] = str(g)
    return out


def parse_sub(html: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """race_list_sub の HTML から、JRA の会場ごとの 開催回・日目・芝のコース区分・馬場状態と、検証用の情報。
    検証用: blocks = 会場ブロックの数、unparsed = 読み切れなかった JRA ブロックの数、links = 払戻一覧のリンクの (kaisai_id, kaisai_date)。
    海外・地方などの JRA 以外の会場は読み飛ばす。"""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, Any]] = []
    unparsed = 0
    blocks = soup.select("dl.RaceList_DataList")
    for dl in blocks:
        title = dl.select_one(".RaceList_DataTitle")
        txt = re.sub(r"\s+", " ", title.get_text(" ", strip=True)) if title else ""
        m = _TITLE.search(txt)
        if not m:
            if any(v in txt for v in JRA_VENUES):
                unparsed += 1                          # JRA の会場なのに「回・日目」が読めない（表記の変化の疑い）
            continue
        if m.group(2) not in JRA_VENUES:
            continue
        shiba = dl.select_one(".Shiba")
        da = dl.select_one(".Da")
        ms = _SHIBA.search(shiba.get_text(strip=True)) if shiba else None
        md = _DA.search(da.get_text(strip=True)) if da else None
        if shiba is not None and ms is None:
            unparsed += 1                              # 芝の欄はあるが読めない
            continue
        out.append({"venue": m.group(2), "kai": int(m.group(1)), "day": int(m.group(3)),
                    "turf_course": (ms.group(1) or None) if ms else None, "turf_condition": ms.group(2) if ms else None,
                    "dirt_condition": md.group(2) if md else None})
    links = sorted(set(_PAYBACK.findall(html)))
    return out, {"blocks": len(blocks), "unparsed": unparsed, "links": links}


def verify(ds: str, recs: list[dict[str, Any]], info: dict[str, Any]) -> str | None:
    """応答が ds の日のものか確かめる。問題があればその理由、無ければ None。"""
    if not recs:
        return "解析結果が空"
    if info["unparsed"]:
        return f"読み切れないブロックが {info['unparsed']} 件"
    if not info["links"]:
        return "払戻一覧のリンクが無く、日付を確かめられない"
    if any(d != ds for _, d in info["links"]):
        return f"別の日の応答（リンクの日付 {sorted({d for _, d in info['links']})}）"
    ids = {i for i, _ in info["links"]}
    for r in recs:                                    # kaisai_id = 年(4) + 会場(2) + 回(2) + 日(2)
        if f"{ds[:4]}{JRA_VENUES[r['venue']]}{r['kai']:02d}{r['day']:02d}" not in ids:
            return f"開催回・日目がリンクと合わない（{r['venue']} {r['kai']}回{r['day']}日目）"
    return None


def store_path(year: int) -> Path:
    return STORE / f"course_position_{year}.json"


def load_store(year: int) -> dict[str, Any]:
    try:
        d = json.loads(store_path(year).read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("dates"), dict):
            return d
    except (OSError, ValueError):
        pass
    return {"version": 1, "dates": {}}


def save_store(year: int, d: dict[str, Any]) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    p = store_path(year)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def race_dates(year_from: int, year_to: int, root: Path = RACE_LISTS, before: str | None = None) -> list[str]:
    """レースのある日（手元の race_lists から）。YYYYMMDD の昇順。before（YYYYMMDD）より前の日だけ。"""
    out = []
    for f in sorted(root.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            ds = str(d.get("date") or f.stem)
            if not re.fullmatch(r"\d{8}", ds) or not d.get("races"):
                continue
        except (OSError, ValueError, AttributeError):
            continue
        if year_from <= int(ds[:4]) <= year_to and (before is None or ds < before):
            out.append(ds)
    return sorted(set(out))


def collect(client: Any, dates: Iterable[str], max_dates: int | None = None, check: Callable[[], str | None] | None = None) -> dict[str, Any]:
    """dates の各日を取得して保存する。取得済みの日は飛ばす。check は各日の前に呼び、理由（文字列）を返したらそこで止める。
    戻り値は件数の内訳（aborted に止めた理由）。保存できない日が MAX_BAD_STREAK 件続いたら止める。"""
    stats: dict[str, Any] = {"fetched": 0, "skipped": 0, "bad": 0, "aborted": None}
    groups: dict[str, str] = {}
    stores: dict[int, dict[str, Any]] = {}
    dirty: set[int] = set()
    streak = 0
    try:
        for ds in dates:
            y = int(ds[:4])
            st = stores.setdefault(y, load_store(y))
            key = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
            if key in st["dates"]:
                stats["skipped"] += 1
                continue
            if max_dates is not None and stats["fetched"] + stats["bad"] >= max_dates:
                break
            reason = check() if check else None
            if reason:
                stats["aborted"] = reason
                break
            if ds not in groups:
                html = client.fetch(f"{BASE}/race_list_get_date_list.html?kaisai_date={ds}&encoding=UTF-8", use_cache=False, encoding="UTF-8")
                groups.update(parse_date_list(html))
            g = groups.get(ds)
            problem = None
            recs: list[dict[str, Any]] = []
            if not g:
                problem = "開催タブに group が無い"
            else:
                sub = client.fetch(f"{BASE}/race_list_sub.html?kaisai_date={ds}&current_group={g}", use_cache=False, encoding="UTF-8")
                recs, info = parse_sub(sub)
                problem = verify(ds, recs, info)
            if problem:
                stats["bad"] += 1
                streak += 1
                logger.warning("保存しない %s: %s", ds, problem)
                if streak >= MAX_BAD_STREAK:
                    stats["aborted"] = f"保存できない日が {streak} 件続いた（直近: {ds} {problem}）。制限ページ・書式の変更・group の食い違いの疑い"
                    break
                continue
            streak = 0
            st["dates"][key] = recs
            dirty.add(y)
            stats["fetched"] += 1
            if stats["fetched"] % 20 == 0:
                for yy in list(dirty):
                    save_store(yy, stores[yy])
                dirty.clear()
                logger.info("%s まで %s", ds, stats)
    finally:
        for yy in dirty:                          # アクセス制限などで途中で止まっても、取れた分は残す
            save_store(yy, stores[yy])
    return stats


def _lock() -> Any:
    STORE.mkdir(parents=True, exist_ok=True)
    f = open(STORE / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit("別の取得が動いている（ロック中）。同時に 2 本は走らせない")
    return f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="y0", type=int, default=2016)
    ap.add_argument("--to", dest="y1", type=int, default=_date.today().year)
    ap.add_argument("--max", type=int, default=None, help="1 回の実行で取る日数の上限")
    ap.add_argument("--dry-run", action="store_true", help="対象数と所要時間の見積もりだけ")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    today = _date.today().strftime("%Y%m%d")
    dates = race_dates(a.y0, a.y1, before=today)
    todo = [d for d in dates if f"{d[:4]}-{d[4:6]}-{d[6:8]}" not in load_store(int(d[:4]))["dates"]]
    n = len(todo) if a.max is None else min(len(todo), a.max)
    logger.info("対象 %d 日（%d〜%d、今日より前）／ 未取得 %d 日 ／ 今回 %d 日（見積もり %.1f 時間: 約 1.3 リクエスト/日 × 約 3 秒）", len(dates), a.y0, a.y1, len(todo), n, n * 1.3 * 3 / 3600)
    if a.dry_run:
        return 0
    from src.scraper import backfill_guard as bg
    from src.scraper.access_guard import AccessGuard
    from src.scraper.client import NetkeibaClient
    from src.scraper.scrape_access_pause import read_access_pause
    from src.scripts.scraping.backfill_horse_training_recent import wait_ok

    lock = _lock()
    guard = AccessGuard(statuses={400, 403, 429}, watch_transport_errors=False)
    guard.__enter__()

    def check() -> str | None:
        halt, pause = bg.halted(), read_access_pause()
        if halt or pause.get("active") or guard.tripped is not None:
            return f"アクセス保留／中断印あり: {(halt or {}).get('reason') or pause.get('reason') or guard.tripped}"
        if not wait_ok(MAX_WAIT_SEC):
            return "他の取得が終わらず待ちきれなかった"
        return None

    try:
        stats = collect(NetkeibaClient(auto_login=False), todo, a.max, check)
    finally:
        guard.__exit__(None, None, None)
        lock.close()
    print(json.dumps(stats, ensure_ascii=False))
    return 3 if stats["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
