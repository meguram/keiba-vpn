#!/usr/bin/env python3
"""race_result の last_3f・通過順・馬体重・オッズ・人気が壊れたレースを、race_result_on_time の値で直す。

壊れた原因は結果ページのパーサーが会員限定列（通過・上り）を取り違えたこと（修正済み）。
既定はドライラン。--apply で保存する。保存前の JSON は --backup-dir に残す（元に戻せる）。

  python -m src.scripts.data.repair_race_result_from_on_time --years 2023 2025          # ドライラン
  python -m src.scripts.data.repair_race_result_from_on_time --years 2023 2025 --apply
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.utils.project_env import load_project_dotenv

_PO = re.compile(r"^\d+(-\d+)*$")


def _ok_l3f(v) -> bool:
    return isinstance(v, (int, float)) and 30 <= v <= 50


def _ok_po(v, n) -> bool:
    return isinstance(v, str) and bool(_PO.match(v)) and max(int(x) for x in v.split("-")) <= n


def _ok_wt(v) -> bool:
    return isinstance(v, (int, float)) and 300 <= v <= 700


def _ok_odds(v) -> bool:
    return isinstance(v, (int, float)) and v > 0


def _ok_pop(v, n) -> bool:
    return isinstance(v, int) and 1 <= v <= n


def repair_entries(rr: dict, ot: dict) -> dict[str, int]:
    """rr の entries を on_time の値で直す（不正な項目のみ）。直した項目数を返す。"""
    n = len(rr.get("entries") or [])
    by_no = {e.get("horse_number"): e for e in (ot.get("entries") or ot.get("results") or []) if isinstance(e, dict)}
    fixed = {"last_3f": 0, "passing_order": 0, "weight": 0, "odds": 0, "popularity": 0}
    for e in rr.get("entries") or []:
        src = by_no.get(e.get("horse_number"))
        if not src or e.get("finish_position", 0) < 1:
            continue
        if not _ok_l3f(e.get("last_3f")) and _ok_l3f(src.get("last_3f")):
            e["last_3f"] = src["last_3f"]; fixed["last_3f"] += 1
        if not _ok_po(e.get("passing_order"), n) and _ok_po(src.get("passing_order"), n):
            e["passing_order"] = src["passing_order"]; fixed["passing_order"] += 1
        if not _ok_wt(e.get("weight")) and _ok_wt(src.get("weight")):
            e["weight"] = src["weight"]; e["weight_change"] = src.get("weight_change", e.get("weight_change")); fixed["weight"] += 1
        if not _ok_odds(e.get("odds")) and _ok_odds(src.get("odds")):
            e["odds"] = src["odds"]; fixed["odds"] += 1
        if not _ok_pop(e.get("popularity"), n) and _ok_pop(src.get("popularity"), n):
            e["popularity"] = src["popularity"]; fixed["popularity"] += 1
    return fixed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="race_result を race_result_on_time の値で直す")
    ap.add_argument("--years", nargs="*", default=["2023", "2025"])
    ap.add_argument("--apply", action="store_true", help="保存する（既定はドライラン）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backup-dir", type=Path, default=Path("data/local/backup/race_result_pre_repair"))
    args = ap.parse_args(argv)

    load_project_dotenv()
    from src.pipeline.features.horse_history.common import load_runs
    from src.pipeline.features.horse_history.repair import suspicious_race_ids
    from src.scraper.storage import HybridStorage

    ids = [r for r in suspicious_race_ids(load_runs()) if r[:4] in set(args.years)]
    if args.limit:
        ids = ids[: args.limit]
    st = HybridStorage()
    totals = {"races": 0, "changed": 0, "no_on_time": 0, "last_3f": 0, "passing_order": 0, "weight": 0, "odds": 0, "popularity": 0}
    for rid in ids:
        totals["races"] += 1
        rr, ot = st.load("race_result", rid), st.load("race_result_on_time", rid)
        if not rr or not ot:
            totals["no_on_time"] += 1
            continue
        before = json.dumps(rr, ensure_ascii=False, sort_keys=True)
        fixed = repair_entries(rr, ot)
        if not any(fixed.values()):
            continue
        totals["changed"] += 1
        for k, v in fixed.items():
            totals[k] += v
        if args.apply:
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            (args.backup_dir / f"{rid}.json").write_text(before, encoding="utf-8")
            rr.setdefault("_meta", {})["repaired_from"] = "race_result_on_time"
            st.save("race_result", rid, rr)
    print(("適用" if args.apply else "ドライラン"), json.dumps(totals, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
