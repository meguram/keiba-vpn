"""レースの分析結果（暫定出馬表 + 各分析モジュールの結果）を組み立てる。"""

from __future__ import annotations

import logging
from typing import Any

from src.research.race_analysis.base import AnalysisModule, ModuleOutput, RaceContext
from src.research.race_analysis.registry import MODULES

logger = logging.getLogger(__name__)

_ENTRY_SOURCES = (("race_shutuba", "entries"), ("race_result", "results"))
_ENTRY_FIELDS = ("bracket_number", "horse_name", "horse_id", "sex_age", "jockey_name", "jockey_id", "jockey_weight", "trainer_name", "trainer_id")


def _load_card(storage: Any, race_id: str) -> tuple[dict, list[dict]]:
    """出馬表（無ければ結果）のカードと出走馬。馬番が未確定（特別登録など）の馬も含める。"""
    for cat, key in _ENTRY_SOURCES:
        try:
            card = storage.load(cat, race_id) or {}
        except Exception:
            logger.exception("%s の読み込みに失敗: %s", cat, race_id)
            card = {}
        rows = [r for r in (card.get(key) or card.get("entries") or []) if isinstance(r, dict) and (r.get("horse_name") or r.get("horse_id"))]
        if rows:
            return card, rows
    return {}, []


def _normalize_entries(rows: list[dict]) -> tuple[list[dict[str, Any]], bool]:
    nums = [int(r.get("horse_number") or 0) for r in rows]
    provisional = len({n for n in nums if n > 0}) != len(rows)     # 馬番が欠けている・重複している → 確定前
    out = []
    for i, r in enumerate(rows):
        e: dict[str, Any] = {k: (r.get(k) or None) for k in _ENTRY_FIELDS}
        e["horse_number"] = None if provisional else int(r["horse_number"])
        e["key"] = str(r.get("horse_id") or f"idx{i}")
        out.append(e)
    if not provisional:
        out.sort(key=lambda e: e["horse_number"])
    return out, provisional


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def analyze_race(storage: Any, race_id: str, modules: tuple[AnalysisModule, ...] = MODULES) -> dict[str, Any]:
    card, rows = _load_card(storage, race_id)
    entries, provisional = _normalize_entries(rows)
    race = {
        "race_id": race_id,
        "race_name": card.get("race_name"),
        "date": card.get("date"),
        "venue": card.get("venue"),
        "round": card.get("round"),
        "surface": card.get("surface"),
        "distance": _int_or_none(card.get("distance")),
        "grade": card.get("grade"),
        "start_time": card.get("start_time"),
    }
    payload: dict[str, Any] = {"race": race, "entries": entries, "provisional_entries": provisional, "modules": []}
    if not entries:
        payload["status"] = "no_entries"
        payload["message"] = "このレースの出馬表はまだ取得できていません。"
        return payload

    ctx = RaceContext(race_id=race_id, date=race["date"], venue=race["venue"], surface=race["surface"], distance=race["distance"], entries=entries)
    for m in modules:
        meta: dict[str, Any] = {"id": m.id, "label": m.label, "description": m.description}
        try:
            if not m.applicable(ctx):
                meta["status"] = "not_applicable"
                payload["modules"].append(meta)
                continue
            out: ModuleOutput = m.compute(ctx)
        except Exception:
            logger.exception("分析モジュール %s の計算に失敗: %s", m.id, race_id)
            meta["status"] = "error"
            payload["modules"].append(meta)
            continue
        meta.update(
            status="ok",
            columns=[c.__dict__ for c in out.columns],
            groups=[g.__dict__ for g in out.groups],
            notes=out.notes,
            data_until=out.data_until,
        )
        payload["modules"].append(meta)
        for e in entries:
            e.setdefault("analysis", {})[m.id] = out.cells.get(e["key"], {})
    payload["status"] = "ok"
    return payload
