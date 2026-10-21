"""出馬表の公示の段階（特別登録／出走馬決定／出馬表確定／馬体重発表済み）と最終更新を API に付ける。

収集側（src/scraper/entries_sync.py）が data/local/meta/entries_state/<race_id>.json に残す状態を読む。
状態が無いレースは、保存済みの出馬表（race_shutuba）の中身から段階だけ判定する（更新時刻は取得時刻があれば使う）。
読み取りだけで、ネットワークには触れない。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from src.scraper import entries_phase as EP
from src.scraper import entries_sync as ES

_PUBLIC_KEYS = ("phase", "label", "note", "updated_at", "checked_at", "n_entries", "n_numbered", "n_jockey", "n_weight", "first_seen_at")


def _race_date_of(race_id: str, card: dict | None) -> date | None:
    d = EP.parse_date((card or {}).get("date"))
    return d


def entries_status(race_id: str, storage: Any = None, *, now: datetime | None = None) -> dict[str, Any] | None:
    """レース 1 件の出馬表の状態。出馬表が無い（まだ公開されていない）ときは phase=none。"""
    now = (now or datetime.now(EP.JST)).astimezone(EP.JST)
    st = ES.load_state(race_id)
    if st and st.get("phase"):
        out = {k: st.get(k) for k in _PUBLIC_KEYS}
        out["source"] = "state"
        ch = [c for c in (st.get("changes") or []) if c.get("type") in ("removed", "added", "jockey_change")]
        out["recent_changes"] = ch[-5:]
        return _with_race_state(out, st.get("date"), now)
    card = None
    if storage is not None:
        try:
            card = storage.load("race_shutuba", race_id)
        except Exception:  # noqa: BLE001
            card = None
    if not card:
        return {"phase": EP.PHASE_NONE, "label": EP.PHASE_LABELS[EP.PHASE_NONE], "note": EP.PHASE_NOTES[EP.PHASE_NONE],
                "updated_at": None, "checked_at": None, "n_entries": 0, "source": "none", "recent_changes": []}
    rd = _race_date_of(race_id, card)
    cls = EP.classify_entries(card, rd, now)
    at = ((card.get("_meta") or {}).get("scraped_at_jst") or (card.get("_meta") or {}).get("saved_at")) if isinstance(card.get("_meta"), dict) else None
    out = {k: cls.get(k) for k in ("phase", "label", "note", "n_entries", "n_numbered", "n_jockey", "n_weight")}
    out.update(updated_at=at, checked_at=at, source="card", recent_changes=[])
    return _with_race_state(out, rd.isoformat() if rd else None, now)


def _with_race_state(out: dict[str, Any], race_date: str | None, now: datetime) -> dict[str, Any]:
    rd = EP.parse_date(race_date)
    out["race_date"] = rd.isoformat() if rd else None
    out["is_future"] = bool(rd and rd >= now.date())
    return out


def attach_entries_status(resp: dict[str, Any], race_id: str, storage: Any = None) -> dict[str, Any]:
    """予測 API の応答に entries_status を足す（後方互換: 未知のキーとして付くだけ）。失敗しても応答は返す。"""
    try:
        if not isinstance(resp, dict):
            return resp
        return {**resp, "entries_status": entries_status(race_id, storage)}
    except Exception:  # noqa: BLE001
        return resp


def race_list_fields(race_ids: list[str]) -> dict[str, dict[str, Any]]:
    """レース一覧の各行に付ける軽量な状態（状態ファイルだけを読む。無いレースは含めない）。"""
    out: dict[str, dict[str, Any]] = {}
    for rid in race_ids:
        st = ES.load_state(rid)
        if st and st.get("phase"):
            out[rid] = {"entries_phase": st["phase"], "entries_label": st.get("label"), "entries_updated_at": st.get("updated_at")}
    return out
