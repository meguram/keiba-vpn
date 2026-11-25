"""pedigree-lab の台帳の入口（研究の CLI からだけ呼ぶ。tests/ops/test_data_ledger.py の _GUARD_ENTRY に登録）。"""
from __future__ import annotations

import os


def open_token(period: tuple[str, str], purpose: str, version: str, reason: str):
    from src.ops import data_ledger
    from src.research.pedigree_lab.data import FAMILY
    os.environ.setdefault("KEIBA_SESSION", "pedigree-lab")
    return data_ledger.guard(tool="pedigree_lab.data", family=FAMILY, version=version, period=period,
                             purpose=purpose, reason=reason, session="pedigree-lab")
