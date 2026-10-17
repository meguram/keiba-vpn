"""馬の戦績ページ（/horse/result/{id}/）の 備考・厩舎ｺﾒﾝﾄ・ｽﾀｰﾄ指数・追走指数・上がり指数 を、新カテゴリ ``horse_race_remarks`` に保存する。

既存の ``horse_result`` には触らない（上書きしない）。1 頭 = 1 オブジェクトで、``race_history`` の各走を ``race_id`` で持つ。
保存前の検証（parse_audit 的）: 表のヘッダーに必須の列が無い、または行があるのに 1 行も読めないページは保存しない（status=unreadable）。
アクセス制限の疑い（AccessRestrictionDetected）や連続エラーでは中断する。再開できる（保存済みの馬はスキップ）。
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Any, Iterable

logger = logging.getLogger("scraper.horse_race_remarks")

CATEGORY = "horse_race_remarks"
FIELDS = ("start_index", "chase_index", "closing_index", "remarks", "stable_comment")
REQUIRED_HEADERS = ("備考", "ｽﾀｰﾄ指数")
RESULT_URL = "https://db.netkeiba.com/horse/result/{horse_id}/"
_DATE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})$")


def to_record(horse_id: str, history: list[dict[str, Any]], fetched_at: str) -> dict[str, Any]:
    """HorseParser の race_history（列名ベースで読んだ新しい列を含む）から保存用レコードを作る。race_id の無い行は捨てる。"""
    rows = []
    for r in history:
        rid = str(r.get("race_id") or "").strip()
        if not rid or not any(k in r for k in FIELDS):
            continue
        m = _DATE.match(str(r.get("date") or ""))
        row: dict[str, Any] = {"race_id": rid, "date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else str(r.get("date") or ""),
                               "horse_number": r.get("horse_number"), "finish_position": r.get("finish_position")}
        for k in FIELDS:
            if k in r:
                row[k] = r[k]
        rows.append(row)
    return {"horse_id": horse_id, "fetched_at": fetched_at, "source": "netkeiba_horse_result", "race_history": rows}


def audit_html(html: str) -> str:
    """'ok' | 'empty'（戦績の表が無い＝未出走など）| 'unreadable'（表はあるが必須の列が無い）。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    t = soup.select_one("table.db_h_race_results") or soup.select_one("table[class*='race_results']")
    if t is None:
        return "empty"
    first = t.select_one("tr")
    heads = [re.sub(r"\s+", "", th.get_text()) for th in first.select("th")] if first else []
    if not all(any(h.startswith(req) for h in heads) for req in REQUIRED_HEADERS):
        return "unreadable"
    return "ok"


def fetch_one(runner: Any, horse_id: str, now_iso: str) -> dict[str, Any]:
    """1 頭分（戦績ページ 1 リクエスト）を取得して保存する。戻り値: {"status": saved|empty|unreadable, "n_rows": 件数}。"""
    html = runner.client.fetch(RESULT_URL.format(horse_id=horse_id), use_cache=False)
    st = audit_html(html)
    if st != "ok":
        logger.warning("戦績ページ %s: %s（保存しない）", horse_id, st)
        return {"status": st, "n_rows": 0}
    from bs4 import BeautifulSoup

    history = runner.horse_parser._parse_race_history(BeautifulSoup(html, "html.parser"))
    rec = to_record(horse_id, history, now_iso)
    if not rec["race_history"]:
        logger.warning("戦績ページ %s: 表はあるが 1 行も読めない（保存しない）", horse_id)
        return {"status": "unreadable", "n_rows": 0}
    ok = runner.storage.save(CATEGORY, horse_id, rec)
    return {"status": "saved" if ok else "error", "n_rows": len(rec["race_history"])}


def run(runner: Any, horse_ids: Iterable[str], *, max_consecutive_errors: int = 5, limit: int = 0, wait_idle: Any = None) -> dict[str, Any]:
    """順に取得する。保存済みの馬はスキップ（再開用）。アクセス制限の疑い・連続エラーで中断。"""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.scraper.access_guard import AccessRestrictionDetected

    lo = float(os.environ.get("HORSE_REMARKS_MIN_SEC", "0.5"))
    hi = float(os.environ.get("HORSE_REMARKS_MAX_SEC", "1.5"))
    ids = list(horse_ids)
    stats: dict[str, Any] = {"total": len(ids), "saved": 0, "skipped_existing": 0, "empty": 0, "unreadable": 0, "error": 0, "aborted": "", "rows": 0,
                             "rows_with_remarks": 0, "rows_with_start_index": 0, "slow_start_mentions": 0}
    consecutive = 0
    done = 0
    for i, hid in enumerate(ids, 1):
        if runner.storage.exists(CATEGORY, hid):
            stats["skipped_existing"] += 1
            continue
        if limit and done >= limit:
            break
        if wait_idle is not None:
            wait_idle()
        try:
            r = fetch_one(runner, hid, datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"))
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            done += 1
            consecutive = 0 if r["status"] in ("saved", "empty") else consecutive + (1 if r["status"] == "unreadable" else 0)
            if r["status"] == "saved":
                rec = runner.storage.load(CATEGORY, hid, bypass_cache=True) or {}
                for x in rec.get("race_history", []):
                    stats["rows"] += 1
                    stats["rows_with_start_index"] += x.get("start_index") is not None
                    rm = str(x.get("remarks") or "")
                    stats["rows_with_remarks"] += bool(rm)
                    stats["slow_start_mentions"] += "出遅" in rm
        except AccessRestrictionDetected as e:
            stats["aborted"] = f"アクセス制限の疑い: {e}"
            logger.error("中断: %s", stats["aborted"])
            break
        except Exception as e:  # noqa: BLE001
            stats["error"] += 1
            consecutive += 1
            logger.error("戦績ページの取得に失敗 %s: %s", hid, e)
            msg = str(e)
            if any(w in msg for w in ("403", "429", "ログイン", "login")):
                stats["aborted"] = f"ブロック/ログイン切れの疑い: {msg[:120]}"
                break
        if consecutive >= max_consecutive_errors:
            stats["aborted"] = f"連続 {consecutive} 回の失敗/読めないページ"
            logger.error("中断: %s", stats["aborted"])
            break
        if i % 100 == 0:
            logger.info("  戦績 %d/%d: %s", i, len(ids), {k: v for k, v in stats.items() if k != "total"})
        time.sleep(random.uniform(lo, hi))
    return stats
