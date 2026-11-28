"""day-pedigree の台帳の入口（研究の CLI からだけ呼ぶ。tests/ops/test_data_ledger.py の _GUARD_ENTRY に登録）。

開発の行（2016〜24）だけを select で読む。2025・26 年は血統の研究で使い切ったので、この族では読まない。
"""
from __future__ import annotations

import os

FAMILY = "day_pedigree"
SESSION = "day-pedigree"


def open_token(period: tuple[str, str], purpose: str, version: str, reason: str):
    from src.ops import data_ledger
    assert period[1] <= "2024-12-31", "2025・26 年はこの族では読まない"
    os.environ.setdefault("KEIBA_SESSION", SESSION)
    return data_ledger.guard(tool="day_pedigree", family=FAMILY, version=version, period=period,
                             purpose=purpose, reason=reason, session=SESSION)
