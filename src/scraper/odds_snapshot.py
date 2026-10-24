"""オッズの時点別スナップショット（T-60 / T-15 など）を 1 文書にまとめるためのマージ処理。

契約（race_odds / race_pair_odds 共通）:
  - トップレベルが「最新」。phase / snapshot_at(JST ISO) を付ける。
  - snapshots は“最新より前の別 phase”の履歴。各 phase 最大 1 件（同一 phase は上書き）。
  - 既存文書が phase 無しの旧形式なら履歴には入れない。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

ODDS_PAYLOAD_KEYS: tuple[str, ...] = ("entries",)
PAIR_ODDS_PAYLOAD_KEYS: tuple[str, ...] = ("umaren", "wide", "umatan", "sanrenpuku")
TRIFECTA_PAYLOAD_KEYS: tuple[str, ...] = ("trifecta", "n_head", "n_combos", "expected_combos")


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def merge_snapshot(
    existing: dict[str, Any] | None,
    new: dict[str, Any],
    phase: str,
    payload_keys: Iterable[str],
    *,
    snapshot_at: str | None = None,
) -> dict[str, Any]:
    """``new`` に phase/snapshot_at/snapshots を付けた新しい文書を返す（引数は破壊しない）。"""
    keys = tuple(payload_keys)
    out = dict(new)
    out["phase"] = phase
    out["snapshot_at"] = snapshot_at or now_jst_iso()

    history: dict[str, dict[str, Any]] = {}
    if isinstance(existing, dict):
        for s in existing.get("snapshots") or []:
            if isinstance(s, dict) and s.get("phase"):
                history[str(s["phase"])] = s
        prev_phase = existing.get("phase")
        if prev_phase:
            snap = {"phase": prev_phase, "snapshot_at": existing.get("snapshot_at")}
            for k in keys:
                if k in existing:
                    snap[k] = existing[k]
            history[str(prev_phase)] = snap
    history.pop(phase, None)

    out.pop("snapshots", None)
    if history:
        out["snapshots"] = sorted(history.values(), key=lambda s: str(s.get("snapshot_at") or ""))
    return out
