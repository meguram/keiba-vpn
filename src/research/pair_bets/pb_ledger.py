"""pair-bets の台帳の入口（研究の CLI からだけ呼ぶ。tests/ops/test_data_ledger.py の _GUARD_ENTRY に登録）。

段 1 は着順・払戻を読まない。読むのは確定オッズ・AI の予測（2022〜24）と、組のオッズ（2026-10-24 以降・1 号の条件 8）だけ。
系統は R0-12 のとおり rules_council に数える。
"""
from __future__ import annotations

import os

FAMILY = "rules_council"
SESSION = "pair-bets"


def open_token(*, tool: str, version: str, period: tuple[str, str], purpose: str, reads: tuple[str, ...], reason: str):
    from src.ops import data_ledger
    if any(w in r for r in reads for w in ("着順", "払戻")):
        raise SystemExit("段 1 は着順・払戻を読まない")
    os.environ.setdefault("KEIBA_SESSION", SESSION)
    return data_ledger.guard(tool=tool, family=FAMILY, version=version, period=period, purpose=purpose,
                             reads=reads, reason=reason, session=SESSION)
