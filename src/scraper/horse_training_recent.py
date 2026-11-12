"""出走馬の調教ページ（最新 1 ページ）を取り直して、保存済みの記録に合わせる（前日夕方 raceday-eve 用）。

馬別の調教ページは、次走に向けた調教（最終追い切りまで）を前日に載せる。1 ページ = 最新 10 走分なので、全ページを取らずに
最新 1 ページだけ取り、同じ日に走ったレースの行を入れ替え、それ以前の行は保存済みの記録をそのまま残す。
読めなかった（0 件）ページは保存しない（既存の記録を空で上書きしない）。キーは horse_id。
保存済みの記録が無い馬（新馬・一度も取っていない馬）は、そこから土台を作る。2 ページ以上ある馬は 1 ページ目だけだと
古い調教が欠けるので、全ページ取り直し（scrape_horse_training(force=True)）で土台を作り、以後は最新 1 ページの追記に戻る。
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any

logger = logging.getLogger("scraper.horse_training_recent")

_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def race_date_of(entry: dict[str, Any]) -> str:
    m = _DATE.search(str(entry.get("race_info", "")))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def merge_recent(existing: dict[str, Any] | None, fresh: dict[str, Any]) -> dict[str, Any]:
    """fresh = 最新 1 ページのパース結果。同じ race の日付の行は fresh で入れ替え、それ以外の既存の行は残す（新しい順を保つ）。"""
    new_entries = list(fresh.get("entries") or [])
    new_dates = {d for d in (race_date_of(e) for e in new_entries) if d}
    old_entries = [e for e in ((existing or {}).get("entries") or []) if race_date_of(e) not in new_dates]
    merged = new_entries + old_entries
    # 既存の記録の付帯情報（打ち切りの印 truncated_before など）は引き継ぐ。_meta は保存のたびに付け直される
    keep = {k: v for k, v in (existing or {}).items() if k not in ("entries", "_meta")}
    return {
        **keep,
        "horse_id": fresh.get("horse_id") or (existing or {}).get("horse_id"),
        "total_items": max(int(fresh.get("total_items") or 0), int((existing or {}).get("total_items") or 0)),
        "pages_fetched": max(int((existing or {}).get("pages_fetched") or 0), 1),
        "entries": merged,
    }


def _has_local_record(runner: Any, horse_id: str) -> bool:
    """手元に調教の行がある記録があるか（GCS が一時的に読めず load が None を返したときに、全ページの取り直しへ流れないため）。"""
    from src.scraper.horse_training_local import horse_training_json_path

    try:
        path = horse_training_json_path(runner.storage._base_dir, horse_id)
        return bool(json.loads(path.read_text(encoding="utf-8")).get("entries"))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def refresh_one(runner: Any, horse_id: str) -> dict[str, Any]:
    """1 頭分を取り直して保存する。戻り値: {"status": "saved"|"unreadable"|"empty", "n_new": 件数}。"""
    url = runner.HORSE_TRAINING_URL.format(horse_id=horse_id)
    html = runner.client.fetch(url, use_cache=False)
    fresh = runner.training_parser.parse(html, horse_id=horse_id)
    if not fresh.get("entries"):
        # ページに調教の表があるのに 0 件 = パーサが表記の変化を読めていない。既存の記録を空で上書きしない
        status = "unreadable" if "race_table_01" in html else "empty"
        logger.warning("調教ページ %s: %s（保存しない）", horse_id, status)
        return {"status": status, "n_new": 0}
    runner.archive.save("horse_training", f"{horse_id}_recent", html)
    existing = runner.storage.load("horse_training", horse_id)
    if not (existing or {}).get("entries") and int(fresh.get("max_page") or 1) > 1 and not _has_local_record(runner, horse_id):
        # 記録が無い馬で 2 ページ以上ある: 最新 1 ページだけの土台にしない。全ページ取って保存する（保存は scrape_horse_training の中）
        logger.info("調教ページ %s: 記録が無いので全 %s ページで土台を作る", horse_id, fresh.get("max_page"))
        full = runner.scrape_horse_training(horse_id, force=True)
        # n_new はこの経路では土台の全件数（scrape_horse_training は 2 ページ目以降の失敗でも取れた分を保存する）
        return {"status": "saved" if (full or {}).get("entries") else "empty",
                "n_new": len((full or {}).get("entries") or []), "full": True}
    record = merge_recent(existing, fresh)
    runner._persist_horse_training(horse_id, record)
    return {"status": "saved", "n_new": len(fresh["entries"])}


def refresh_for_horses(runner: Any, horse_ids: list[str]) -> dict[str, Any]:
    """出走馬の調教ページを順に取り直す。1 頭の失敗で止めない。アクセス制限の疑い（AccessRestrictionDetected）が出たら中断。"""
    from src.scraper.access_guard import AccessRestrictionDetected

    lo = float(os.environ.get("HORSE_TRAINING_RECENT_MIN_SEC", "0.5"))
    hi = float(os.environ.get("HORSE_TRAINING_RECENT_MAX_SEC", "1.5"))
    stats: dict[str, Any] = {"total": len(horse_ids), "saved": 0, "unreadable": 0, "empty": 0, "error": 0, "aborted": False}
    for i, hid in enumerate(horse_ids, 1):
        try:
            r = refresh_one(runner, hid)
            stats[r["status"]] += 1
        except AccessRestrictionDetected as e:
            logger.error("アクセス制限の疑いで調教の取り直しを中断: %s", e)
            stats["aborted"] = True
            break
        except Exception as e:  # noqa: BLE001
            stats["error"] += 1
            logger.error("調教ページの取り直しに失敗 %s: %s", hid, e)
        if i % 50 == 0:
            logger.info("  調教ページ %d/%d: %s", i, len(horse_ids), {k: v for k, v in stats.items() if k != "total"})
        time.sleep(random.uniform(lo, hi))
    return stats


def horse_ids_for_races(storage: Any, race_ids: list[str]) -> list[str]:
    """レースの出馬表から horse_id を集める（重複なし）。"""
    seen: dict[str, None] = {}
    for rid in race_ids:
        card = storage.load("race_shutuba", rid) or {}
        for e in card.get("entries") or []:
            hid = str(e.get("horse_id") or "").strip()
            if hid:
                seen[hid] = None
    return list(seen)
