#!/usr/bin/env python3
"""CI: 2025 年 test・2026 年の結果を読むらしい評価の道具のうち、見た回数の台帳（src.ops.data_ledger）を呼ばないものを一覧にする。

今は一覧を出すだけ（終了コード 0）。全担当の道具に台帳を組み込んだ後に `--strict` で失敗にする（計画 progress/ledger_plan_20261118.md §6）。
一覧は手掛かりで、確実ではない（名前の決まりに合わない読み出しは見つからない）。道具の外で見たら `python -m src.ops.data_ledger declare` で申告する。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRS = ("src/scripts/data", "src/research")
# 2025 年 test・2026 年の評価・1 回だけの印に使われている名前
PATTERN = re.compile(r"test2025|TEST2025|_2026_rules|blind2026|apply2026|EVALUATED|prereg|frozen_hash|final_eval_v6")
GUARD = re.compile(r"from src\.ops import data_ledger|import src\.ops\.data_ledger|from src\.ops\.data_ledger import")


def scan() -> tuple[list[str], list[str]]:
    hit, ok = [], []
    for d in DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            t = p.read_text(encoding="utf-8", errors="ignore")
            if not PATTERN.search(t):
                continue
            (ok if GUARD.search(t) else hit).append(str(p.relative_to(ROOT)))
    return hit, ok


def main(argv: list[str]) -> int:
    hit, ok = scan()
    print(f"台帳を呼ぶ評価の道具: {len(ok)} 本・呼ばない（候補）: {len(hit)} 本")
    for h in hit:
        print(f"  - {h}")
    if "--strict" in argv and hit:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
