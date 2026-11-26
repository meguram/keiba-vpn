"""track-bias の台帳の入口（研究の CLI からだけ呼ぶ。tests/ops/test_data_ledger.py の _GUARD_ENTRY に登録）。

build-all（全期間の名札）は eval_only（採否・選び直しに使わない）。calibrate（k と校正を決める）は 2022〜24 を select で読む。
"""
from __future__ import annotations

import os

FAMILY = "track_bias"
SESSION = "track-bias"


def open_token(period: tuple[str, str], version: str, reason: str, purpose: str = "eval_only"):
    from src.ops import data_ledger
    os.environ.setdefault("KEIBA_SESSION", SESSION)
    return data_ledger.guard(tool="track_bias", family=FAMILY, version=version, period=period,
                             purpose=purpose, reason=reason, session=SESSION)
