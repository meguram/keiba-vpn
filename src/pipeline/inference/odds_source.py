"""馬券推奨が使うオッズの出どころ（実オッズ / 予測オッズ）を決める。

- 実オッズ: ``race_odds`` / ``race_pair_odds``。トップが最新、``snapshots`` に以前の phase（t60/t15）が入る。
  phase を持たない旧形式は「最新」として扱う。
- 予測オッズ: 最終オッズ予測のキャッシュ（単勝・複勝のみ。2連系・3連複は存在しない）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

LATEST = "latest"
SINGLE_ONLY_BET_TYPES = frozenset({"tansho", "fukusho"})


def _flatten(doc: dict, snap: dict | None) -> dict:
    """snapshots を除いたフラットな文書にする。snap があればその phase の内容で置き換える。"""
    base = {k: v for k, v in doc.items() if k != "snapshots"}
    if snap is not None:
        base.update(snap)
    return base


def pick_snapshot(doc: dict | None, phase: str = LATEST) -> dict | None:
    """指定 phase（latest / t60 / t15）のフラットな文書を返す。無ければ None。"""
    if not isinstance(doc, dict) or not doc:
        return None
    phase = (phase or LATEST).strip().lower()
    if phase == LATEST:
        return _flatten(doc, None)
    if doc.get("phase") == phase:
        return _flatten(doc, None)
    for snap in reversed(doc.get("snapshots") or []):
        if isinstance(snap, dict) and snap.get("phase") == phase:
            return _flatten(doc, snap)
    return None


def list_phases(doc: dict | None) -> list[dict[str, Any]]:
    """文書から取れる phase 一覧 [{"phase", "as_of"}]（最新が先頭）。"""
    if not isinstance(doc, dict) or not doc:
        return []
    out = [{"phase": doc.get("phase") or LATEST, "as_of": doc.get("snapshot_at")}]
    for snap in reversed(doc.get("snapshots") or []):
        if isinstance(snap, dict) and snap.get("phase"):
            out.append({"phase": snap["phase"], "as_of": snap.get("snapshot_at")})
    return out


def predicted_single_odds(storage, race_id: str) -> dict | None:
    """最終オッズ予測キャッシュを race_odds 互換の ``{"entries": [...]}`` に変換する。"""
    try:
        from src.pipeline.inference.final_odds_service import load_cached_response

        cached = load_cached_response(storage, race_id)
    except Exception as exc:
        logger.debug("予測オッズ読み込み失敗 %s: %s", race_id, exc)
        return None
    entries = []
    for e in (cached or {}).get("entries") or []:
        hn = e.get("horse_number")
        win = e.get("predicted_win_odds")
        if not hn or not win:
            continue
        entries.append({
            "horse_number": int(hn),
            "win_odds": float(win),
            "place_odds_min": float(e.get("predicted_place_odds_min") or 0.0),
            "place_odds_max": float(e.get("predicted_place_odds_max") or 0.0),
        })
    return {"entries": entries} if entries else None


def _load(storage, category: str, race_id: str) -> dict | None:
    try:
        doc = storage.load(category, race_id)
    except Exception as exc:
        logger.warning("%s/%s 読み込み失敗: %s", category, race_id, exc)
        return None
    return doc if isinstance(doc, dict) and doc else None


def _has_single(doc: dict | None) -> bool:
    return bool(doc and doc.get("entries"))


def resolve_odds(storage, race_id: str, mode: str = "auto", phase: str = LATEST) -> dict[str, Any]:
    """推奨に使うオッズを決める。

    mode: auto=実オッズ（単複があれば）→無ければ予測 / actual=実オッズのみ / predicted=予測オッズのみ
    戻り値 source は "actual" | "predicted" | None（使えるオッズ無し）。
    predicted では 2連系・3連複を出せないため ``allowed_bet_types`` を単勝・複勝に絞る。
    """
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "actual", "predicted"):
        mode = "auto"
    phase = (phase or LATEST).strip().lower()

    single_doc = _load(storage, "race_odds", race_id)
    pair_doc = _load(storage, "race_pair_odds", race_id)
    available = list_phases(single_doc) or list_phases(pair_doc)

    result: dict[str, Any] = {
        "single_odds": None,
        "pair_odds": {},
        "source": None,
        "phase": None,
        "as_of": None,
        "available": available,
        "allowed_bet_types": None,
        "note": "",
    }

    notes: list[str] = []
    if mode in ("auto", "actual"):
        snap = pick_snapshot(single_doc, phase)
        if _has_single(snap):
            pair_snap = pick_snapshot(pair_doc, phase) or {}
            result.update(
                single_odds=snap,
                pair_odds=pair_snap,
                source="actual",
                phase=snap.get("phase") or phase,
                as_of=snap.get("snapshot_at"),
            )
            if not pair_snap:
                notes.append("2連系・3連複の実オッズは未取得（単勝・複勝のみ評価）")
            result["note"] = "。".join(notes)
            return result
        if single_doc is None:
            notes.append("実オッズ（単勝・複勝）が未取得")
        else:
            notes.append(f"phase={phase} の実オッズがありません")
        if mode == "actual":
            result["note"] = "。".join(notes)
            return result

    pred = predicted_single_odds(storage, race_id)
    if pred is not None:
        result.update(
            single_odds=pred,
            pair_odds={},
            source="predicted",
            phase="predicted",
            allowed_bet_types=set(SINGLE_ONLY_BET_TYPES),
        )
        notes.append("予測オッズ基準の暫定推奨（単勝・複勝のみ）")
    else:
        notes.append("予測オッズも未計算")
    result["note"] = "。".join(notes)
    return result
