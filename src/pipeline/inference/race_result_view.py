"""画面（週間予測のレース詳細）に出すレース結果。確定結果（race_result）があればそれ、無ければ速報（race_result_on_time。発走 +15 分に取得）。

返す内容は着順表と払戻表だけ（予測との突き合わせは画面側でする）。結果が無い・まだ着順が付いていないときは ``available: False``。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PAYOFF_ORDER = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "3連複", "三連単", "3連単"]
_LABEL = {"三連複": "3連複", "三連単": "3連単"}


def _to_int(x: Any) -> int | None:
    try:
        return int(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_float(x: Any) -> float | None:
    try:
        v = float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _payoff_rows(payoff: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(payoff, dict):
        return out
    keys = sorted(payoff, key=lambda k: _PAYOFF_ORDER.index(k) if k in _PAYOFF_ORDER else 99)
    for jp in keys:
        val = payoff[jp]
        rows = val if isinstance(val, list) else [val]
        items = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            pay = _to_int(r.get("payout"))
            nums = str(r.get("numbers", "")).strip()
            if not nums or not pay:
                continue
            items.append({"numbers": nums, "payout": pay, "popularity": _to_int(r.get("popularity"))})
        if items:
            out.append({"bet_label": _LABEL.get(str(jp), str(jp)), "rows": items})
    return out


def _entries(doc: dict) -> list[dict[str, Any]]:
    rows = []
    for e in doc.get("entries") or []:
        if not isinstance(e, dict):
            continue
        pos = _to_int(e.get("finish_position"))
        if not pos or pos <= 0:
            continue
        rows.append({
            "pos": pos, "horse_number": _to_int(e.get("horse_number")), "bracket_number": _to_int(e.get("bracket_number")),
            "horse_name": e.get("horse_name") or "", "horse_id": e.get("horse_id") or "", "sex_age": e.get("sex_age") or "",
            "jockey_name": e.get("jockey_name") or "", "finish_time": e.get("finish_time") or "", "margin": e.get("margin") or "",
            "passing_order": e.get("passing_order") or "", "last_3f": _to_float(e.get("last_3f")),
            "odds": _to_float(e.get("odds")), "popularity": _to_int(e.get("popularity")),
            "weight": _to_int(e.get("weight")), "weight_change": e.get("weight_change"),
        })
    rows.sort(key=lambda r: (r["pos"], r["horse_number"] or 0))
    return rows


def build_result_view(storage: Any, race_id: str) -> dict[str, Any]:
    for cat, status in (("race_result", "final"), ("race_result_on_time", "provisional")):
        try:
            doc = storage.load(cat, race_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s の読み込み失敗 %s: %s", cat, race_id, exc)
            continue
        if not isinstance(doc, dict):
            continue
        finish = _entries(doc)
        if not finish:
            continue
        payoff = _payoff_rows(doc.get("payoff"))
        if not payoff:
            try:
                pdoc = storage.load(f"{cat}_payoff", race_id)
            except Exception:  # noqa: BLE001
                pdoc = None
            if isinstance(pdoc, dict):
                payoff = _payoff_rows(pdoc.get("payoff"))
        meta = doc.get("_meta") if isinstance(doc.get("_meta"), dict) else {}
        return {
            "available": True, "race_id": race_id, "status": status, "official": bool(payoff),
            "race_name": doc.get("race_name") or "", "date": doc.get("date") or "", "venue": doc.get("venue") or "",
            "surface": doc.get("surface") or "", "distance": doc.get("distance"), "weather": doc.get("weather") or "",
            "track_condition": doc.get("track_condition") or "", "start_time": doc.get("start_time") or "",
            "field_size": doc.get("field_size"), "finish": finish, "payoff": payoff,
            "fetched_at": meta.get("scraped_at_jst") or "",
        }
    return {"available": False, "race_id": race_id}
