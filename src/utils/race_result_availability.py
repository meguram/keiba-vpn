"""
レース結果データ（確定・速報）の有無をストレージから判定する。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any


def _blob_has_entries(data: dict | None) -> bool:
    if not data or not isinstance(data, dict):
        return False
    entries = data.get("entries")
    if isinstance(entries, list) and len(entries) > 0:
        return True
    # 速報・一部形式は entries 以外に着順が載る場合あり
    if data.get("results") or data.get("horses"):
        return True
    return bool(data.get("race_id") and (data.get("finish_time") or data.get("lap_times")))


def race_result_status(storage, race_id: str) -> dict[str, Any]:
    """
    単一レースの結果データ有無。

    Returns:
        has_confirmed, has_flash, viewable, kind ('confirmed' | 'flash' | None)
    """
    has_confirmed = False
    has_flash = False

    if storage.exists("race_result", race_id):
        blob = storage.load("race_result", race_id)
        if _blob_has_entries(blob):
            has_confirmed = True

    if not has_confirmed and storage.exists("race_result_on_time", race_id):
        blob = storage.load("race_result_on_time", race_id)
        if _blob_has_entries(blob):
            has_flash = True

    viewable = has_confirmed or has_flash
    kind = "confirmed" if has_confirmed else ("flash" if has_flash else None)
    return {
        "race_id": race_id,
        "has_confirmed": has_confirmed,
        "has_flash": has_flash,
        "viewable": viewable,
        "kind": kind,
        "result_view_url": f"/race/{race_id}" if viewable else None,
    }


def batch_race_result_status(
    storage,
    race_ids: list[str],
    *,
    date: str = "",
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    """
    複数レースの結果有無をまとめて返す。

    ``date``（YYYYMMDD）があれば、その年の list_keys で高速化してから exists で補完。
    """
    ids = [str(r).strip() for r in race_ids if r]
    if not ids:
        return {}

    year = (date or ids[0])[:4] if len(date or ids[0]) >= 4 else ""
    confirmed_keys: set[str] = set()
    flash_keys: set[str] = set()
    if year:
        try:
            confirmed_keys = set(storage.list_keys("race_result", year))
            flash_keys = set(storage.list_keys("race_result_on_time", year))
        except Exception:
            pass

    def _check(rid: str) -> tuple[str, dict[str, Any]]:
        if rid in confirmed_keys:
            return rid, {
                "race_id": rid,
                "has_confirmed": True,
                "has_flash": rid in flash_keys,
                "viewable": True,
                "kind": "confirmed",
                "result_view_url": f"/race/{rid}",
            }
        if rid in flash_keys:
            return rid, {
                "race_id": rid,
                "has_confirmed": False,
                "has_flash": True,
                "viewable": True,
                "kind": "flash",
                "result_view_url": f"/race/{rid}",
            }
        return rid, race_result_status(storage, rid)

    out: dict[str, dict[str, Any]] = {}
    workers = min(max_workers, max(len(ids), 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rid, st in pool.map(_check, ids):
            out[rid] = st
    return out
