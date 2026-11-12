"""出走馬の調教ページ（最新 1 ページ）を取り直して、保存済みの記録に合わせる（前日夕方 raceday-eve 用）。

馬別の調教ページは、次走に向けた調教（最終追い切りまで）を前日に載せる。1 ページ = 最新 10 走分なので、全ページを取らずに
最新 1 ページだけ取り、同じ日に走ったレースの行を入れ替え、それ以前の行は保存済みの記録をそのまま残す。
読めなかった（0 件）ページは保存しない（既存の記録を空で上書きしない）。キーは horse_id。
保存済みの記録が無い馬（新馬・一度も取っていない馬）は、そこから土台を作る。2 ページ以上ある馬は 1 ページ目だけだと
古い調教が欠けるので、全ページ取り直し（scrape_horse_training(force=True)）で土台を作り、以後は最新 1 ページの追記に戻る。
土台の取得が途中で止まった記録（incomplete の印）は、次の取り直しのときに全ページで作り直す（印は追記では消えない）。
作り直しでは既存の行を消さずにまとめ、MAX_REBUILD_TRIES 回続けて途中で止まったら印を外して追記だけに戻る。
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

MAX_REBUILD_TRIES = 3   # 全ページの作り直しが続けて途中で止まったら、それ以上は取り直さない（取得回数を増やし続けない）
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


def _entry_key(e: dict[str, Any]) -> tuple:
    return (e.get("date"), e.get("course"), e.get("time_raw"))


def _union(full: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    """全ページ取り直しの結果に、既存の記録にしか無い行（取り直しで取れなかった古いページの行）を足す。
    重なりは merge_recent と同じく race の日付で判定する（日付の無い行だけは行の中身で判定）。"""
    entries = list(full.get("entries") or [])
    dates = {d for d in (race_date_of(e) for e in entries) if d}
    seen = {_entry_key(e) for e in entries}
    extra = [e for e in ((existing or {}).get("entries") or [])
             if ((d := race_date_of(e)) and d not in dates) or (not d and _entry_key(e) not in seen)]
    keep = {k: v for k, v in (existing or {}).items() if k not in ("entries", "_meta", "incomplete", "rebuild_tries")}
    return {**keep, **{k: v for k, v in full.items() if k != "_meta"}, "entries": list(full.get("entries") or []) + extra}


def _mark_incomplete(record: dict[str, Any], existing: dict[str, Any] | None, horse_id: str) -> dict[str, Any]:
    """作り直しが途中で止まった: 回数を数え、上限に達したら印を外す。"""
    tries = int((existing or {}).get("rebuild_tries") or 0) + 1
    if tries >= MAX_REBUILD_TRIES:
        logger.warning("調教ページ %s: 全ページの作り直しが %d 回続けて途中で止まったので、以後は追記だけにする", horse_id, tries)
        record.pop("incomplete", None)
        record.pop("rebuild_tries", None)
    else:
        record["incomplete"], record["rebuild_tries"] = True, tries
    return record


def _rebuild(runner: Any, horse_id: str, existing: dict[str, Any] | None) -> dict[str, Any] | None:
    """全ページを取り直し、既存の行とまとめて保存する。1 ページ目から取れなければ None（呼び出し側が追記で保存する）。"""
    from src.scraper.access_guard import AccessRestrictionDetected

    try:
        full = runner.scrape_horse_training(horse_id, force=True, persist=False)   # 保存は既存とまとめて 1 回だけ
    except AccessRestrictionDetected:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("調教ページ %s: 全ページの取り直しに失敗（最新 1 ページの追記で保存する）: %s", horse_id, exc)
        return None
    if not (full or {}).get("entries"):
        return None
    record = _union(full, existing)
    if full.get("incomplete"):
        record = _mark_incomplete(record, existing, horse_id)
    else:   # 全ページ取れた: 途中の印も、過去分の打ち切りの印も外す
        for k in ("incomplete", "rebuild_tries", "truncated_before"):
            record.pop(k, None)
    runner._persist_horse_training(horse_id, record)
    return record


def _load_local(runner: Any, horse_id: str) -> dict[str, Any] | None:
    """手元の記録（GCS が一時的に読めず load が None を返したときの代わり。記録を最新 1 ページで上書きして消さないため）。"""
    from src.scraper.horse_training_local import horse_training_json_path

    try:
        path = horse_training_json_path(runner.storage._base_dir, horse_id)
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and d.get("entries") else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


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
    existing = runner.storage.load("horse_training", horse_id) or _load_local(runner, horse_id)
    no_base = not (existing or {}).get("entries") and int(fresh.get("max_page") or 1) > 1
    rebuild = no_base or bool((existing or {}).get("incomplete"))
    if rebuild:
        # 記録が無い馬で 2 ページ以上ある、または前回の土台が途中で止まった: 最新 1 ページの追記にせず、全ページで作り直す
        logger.info("調教ページ %s: %s全 %s ページで土台を作る", horse_id,
                    "記録が無いので" if no_base else "前回の土台が途中までなので", fresh.get("max_page"))
        rec = _rebuild(runner, horse_id, existing)
        if rec is not None:
            # n_new はこの経路では土台の全件数
            return {"status": "saved", "n_new": len(rec["entries"]), "full": True}
    record = merge_recent(existing, fresh)
    if rebuild:   # 作り直せなかった: 最新ページを追記で保存し、次の取り直しで全ページを取る
        record = _mark_incomplete(record, existing, horse_id)
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
