"""
netkeiba レーストップ（https://race.netkeiba.com/top/?rf=navi 等）が JavaScript で取っている
レース一覧を、ブラウザと同じ XHR 先から取得する（ヘッドレス不要）。

ブラウザの race_list.js と同じ HTTP フロー:
  1. GET .../top/race_list_get_date_list.html?kaisai_date=YYYYMMDD&encoding=UTF-8
  2. 対象日の li[date] から current_group（group 属性）を取得
  3. GET .../top/race_list_sub.html?kaisai_date=...&current_group=...

フロント（自前 JS）から使う場合も、上記 2 URL を netkeiba ドメインに対して fetch すればよい
（認証不要・HTML 断片）。パースは本モジュールの parse_race_list_sub_html と同等ロジックを
サーバ側で再利用するか、DOM を渡す。

JRA 公式サイトは使わない。db.netkeiba.com/race/list/ は直近数日〜1週間ほど空になりがちなため、
ScraperRunner.scrape_race_list は「直近ウィンドウ」で本フローを優先する。

出力レコード（1 レース）のキーは db 一覧パースの上位互換:
  race_id, round, venue, race_name に加え date, entries_count, list_grade_icon（トップ固有）
"""

from __future__ import annotations

import copy
import logging
import re
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from scraper.client import NetkeibaClient

logger = logging.getLogger(__name__)

NETKEIBA_TOP_BASE = "https://race.netkeiba.com/top"

# upcoming 画面のポーリング対策（秒）
_RACE_LIST_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_RACE_LIST_CACHE_TTL_SEC = 90.0

_VENUE_CODE = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def resolve_current_group(date_list_html: str, kaisai_date: str) -> str | None:
    """race_list_get_date_list の HTML から、開催日に対応する current_group を返す。"""
    soup = BeautifulSoup(date_list_html, "html.parser")
    for li in soup.select("li[date]"):
        if li.get("date") == kaisai_date:
            g = li.get("group")
            if g:
                return str(g)
    return None


def _fetch_date_list_html(client: NetkeibaClient, query_kaisai: str) -> str:
    url = (
        f"{NETKEIBA_TOP_BASE}/race_list_get_date_list.html"
        f"?kaisai_date={query_kaisai}&encoding=UTF-8"
    )
    return client.fetch(url, use_cache=False, encoding="UTF-8")


def find_current_group_for_target_date(
    client: NetkeibaClient,
    target_yyyymmdd: str,
    *,
    max_hops: int = 40,
) -> str | None:
    """
    開催日 target がタブに載るまで prev/next の kaisai_date を BFS で辿り、
    その日の li に付いた group（= race_list_sub の current_group）を返す。

    単発リクエストでは週ウィンドウ外の日付が ul に出ず group が取れないことがあるため。
    """
    q: deque[str] = deque([target_yyyymmdd])
    visited: set[str] = set()
    hops = 0

    while q and hops < max_hops:
        kd = q.popleft()
        if kd in visited:
            continue
        visited.add(kd)
        hops += 1

        html = _fetch_date_list_html(client, kd)
        group = resolve_current_group(html, target_yyyymmdd)
        if group:
            logger.debug(
                "netkeiba top: group for %s found via query_kaisai=%s (%d hops)",
                target_yyyymmdd,
                kd,
                hops,
            )
            return group

        soup = BeautifulSoup(html, "html.parser")
        for sid in ("prevBtn", "nextBtn"):
            el = soup.select_one(f"#{sid}[date]")
            if el and el.get("date"):
                nxt = str(el["date"]).strip()
                if nxt and nxt not in visited:
                    q.append(nxt)

    logger.warning(
        "netkeiba top: current_group 取得失敗 target=%s (%d hops)",
        target_yyyymmdd,
        hops,
    )
    return None


def parse_race_list_sub_html(html: str, date_compact: str) -> list[dict[str, Any]]:
    """race_list_sub.html をパース（レース名・頭数は一覧 DOM から）。"""
    soup = BeautifulSoup(html, "html.parser")
    races: list[dict[str, Any]] = []
    seen: set[str] = set()

    for li in soup.select("li.RaceList_DataItem"):
        a = li.select_one('a[href*="race_id="]')
        if not a or not a.get("href"):
            continue
        m = re.search(r"race_id=(\d{12})", a["href"])
        if not m:
            continue
        rid = m.group(1)
        if rid in seen:
            continue
        seen.add(rid)

        venue_code = rid[4:6] if len(rid) == 12 else ""
        if venue_code not in _VENUE_CODE:
            continue

        rnd = int(rid[10:12]) if len(rid) == 12 else 0
        title_el = li.select_one(".RaceList_ItemTitle .ItemTitle")
        race_name = title_el.get_text(strip=True) if title_el else f"{rnd}R"

        entries_count = 0
        num_el = li.select_one(".RaceList_Itemnumber")
        if num_el:
            tm = re.search(r"(\d+)\s*頭", num_el.get_text())
            if tm:
                entries_count = int(tm.group(1))

        list_grade_icon = None
        grade_el = li.select_one("[class*='Icon_GradeType']")
        if grade_el:
            for c in grade_el.get("class") or []:
                if isinstance(c, str) and c.startswith("Icon_GradeType") and c != "Icon_GradeType":
                    list_grade_icon = c
                    break

        races.append({
            "race_id": rid,
            "round": rnd,
            "venue": _VENUE_CODE[venue_code],
            "race_name": race_name,
            "date": date_compact,
            "entries_count": entries_count,
            "list_grade_icon": list_grade_icon,
        })

    races.sort(key=lambda x: (x["venue"], x["round"]))
    return races


def fetch_races_for_kaisai_date(
    client: NetkeibaClient,
    kaisai_date: str,
    *,
    use_cache: bool = True,
    return_sub_html: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """
    kaisai_date: YYYYMMDD（https://race.netkeiba.com/top/?kaisai_date= と同じ）

    開催がない日・タブが空の日は []。
    use_cache: True のとき短時間同一日付の再取得を抑える（API ポーリング用）。
    return_sub_html: True のとき (races, race_list_sub.html の生 HTML) を返す。
        アーカイブ用。True 時はキャッシュヒットを使わず常に再取得する。
    """
    now = time.time()
    if (
        not return_sub_html
        and use_cache
        and kaisai_date in _RACE_LIST_CACHE
    ):
        ts, cached = _RACE_LIST_CACHE[kaisai_date]
        if now - ts < _RACE_LIST_CACHE_TTL_SEC:
            return copy.deepcopy(cached)

    group = find_current_group_for_target_date(client, kaisai_date)
    if not group:
        logger.info(
            "netkeiba top: %s に対応する開催タブなし（開催なしまたは一覧範囲外）",
            kaisai_date,
        )
        _RACE_LIST_CACHE[kaisai_date] = (now, [])
        if return_sub_html:
            return [], ""
        return []

    url2 = (
        f"{NETKEIBA_TOP_BASE}/race_list_sub.html"
        f"?kaisai_date={kaisai_date}&current_group={group}"
    )
    sub_html = client.fetch(url2, use_cache=False, encoding="UTF-8")
    races = parse_race_list_sub_html(sub_html, kaisai_date)
    logger.info("netkeiba top: %s → %d レース (group=%s)", kaisai_date, len(races), group)

    _RACE_LIST_CACHE[kaisai_date] = (now, copy.deepcopy(races))
    if return_sub_html:
        return races, sub_html
    return races


def kaisai_date_in_top_priority_window(
    date_compact: str,
    *,
    days_past: int = 7,
    days_future: int = 14,
) -> bool:
    """db 一覧が空になりやすい「直近」を判定（JST）。"""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    try:
        rd = datetime.strptime(date_compact, "%Y%m%d").date()
    except ValueError:
        return False
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    lo = today - timedelta(days=days_past)
    hi = today + timedelta(days=days_future)
    return lo <= rd <= hi


def invalidate_race_list_cache(kaisai_date: str | None = None) -> None:
    """キャッシュ削除。kaisai_date None なら全消し。"""
    if kaisai_date is None:
        _RACE_LIST_CACHE.clear()
    else:
        _RACE_LIST_CACHE.pop(kaisai_date, None)


def is_plausible_race_day_races(races: list) -> bool:
    """
    race_lists の件数が「その日の一覧としてあり得るか」を返す。

    会場あたり 12R が基本だが、中止・移設などで 12 の倍数でない日は普通にあるため
    倍数条件は使わない。極端に少ない件数だけ旧テスト用の断片取得などとみなして弾く。
    """
    n = len(races)
    if n == 0:
        return True
    if n < 4:
        return False
    # 同一日の JRA 中央競馬は多くて 3 場×12R=36。37 件以上は異常とみなす。
    if n > 36:
        return False
    return True
