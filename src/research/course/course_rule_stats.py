"""めぐ分析の表示用に、凍結した全レース版の第 3 版の条件ごとの成績を数える（評価ではない。条件は選び直さない）。

使い方: python -m src.research.course.v3 rule_stats（= python -m src.research.course.course_rule_stats）
出力: <COURSE_OUT>/v3/rule_stats_v3.json と、めぐ分析が読む写し data/meta/course/rule_stats_v3.json（commit する）。
2025・2026 年の着順・払戻を読むので、見た回数の台帳に eval_only（表示用）として記録する（guard は _rule_stats_inner だけ）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

import pandas as pd

from src.research.course import run as CR
from src.research.course.v3 import (
    C3,
    EVAL_MARK3,
    EVAL_YEAR,
    OUT3,
    PREREG3,
    RULE_STATS,
    RULE_STATS_REPO,
    RULE_TOP,
    TRAIN_END,
    _frame_2026,
    _pick_values,
    load3,
    pick_shown,
    rule_list,
    rule_stats_period,
)

__all__ = ["RULE_TOP", "cmd_rule_stats"]


def cmd_rule_stats() -> None:
    """凍結した第 3 版の条件ごとに、訓練 2016〜2024・2025・2026 年の成績を数えて <COURSE_OUT>/v3/rule_stats_v3.json に書く（めぐ分析の表示用）。
    規則は変えない（sha を照合）。2026 年は 1 回だけの評価が済んだ後にだけ読む。見た回数の台帳には eval_only（表示用）として記録する。"""
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark:
        sys.exit("2026 年の 1 回だけの評価が済んでいない（封印中の 2026 年は読まない）")
    CR.wait_resources()
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval")}
    subprocess.run([sys.executable, "-m", "src.research.course.course_rule_stats", "_inner"], env=e, cwd=CR.REPO, check=True)


def _rule_stats_inner() -> None:
    from src.ops import data_ledger as DL
    from src.research.graded import catalog as CAT
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark:
        sys.exit("2026 年の 1 回だけの評価が済んでいない（封印中の 2026 年は読まない）")
    c = C3(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    names = {t.id: t.name for t in CAT.TYPES}
    with DL.guard(tool="src/research/course/course_rule_stats.py:_rule_stats_inner", family="course", version=P["sha256_cells"][:12],
                  period=("2025-01-01", "2026-11-20"), purpose="eval_only",
                  reason="めぐ分析の表示用に、凍結した第 3 版の条件ごとの成績を数える（条件は選び直さない・採否に使わない）",
                  frozen_paths=[PREREG3, OUT3 / P["cells_file"]]) as tok:
        out = {}
        d, L, tr, ev = load3()
        out["train"] = rule_stats_period(d[tr].reset_index(drop=True), L[tr].reset_index(drop=True), kept, c)
        out["2025"] = rule_stats_period(d[ev].reset_index(drop=True), L[ev].reset_index(drop=True), kept, c)
        tok.check_dates(d.loc[ev, "date"]) if "date" in d.columns else None
        del d, L
        d6, L6 = _frame_2026()
        tok.check_dates(d6["date"]) if "date" in d6.columns else None
        out["2026"] = rule_stats_period(d6, L6, kept, c)
        data_until = str(pd.to_datetime(d6["date"]).max())[:10] if "date" in d6.columns and len(d6) else None
        shown = pick_shown(out["train"], kept)
        res = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cycle": P["cycle"], "sha256_cells": P["sha256_cells"],
               "periods": {"train": [2016, TRAIN_END], "2025": [EVAL_YEAR, EVAL_YEAR], "2026": [2026, 2026]}, "data_until": data_until,
               "rule_top": RULE_TOP, "rules": rule_list(kept, names, shown), "values": {k: _pick_values(v, shown) for k, v in out.items()},
               "ledger": tok.summary(),
               "note": "凍結した第 3 版の条件を当て直して数えた値（評価ではない・表示用）。2025・2026 年は何度か見た期間"}
        RULE_STATS.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
        RULE_STATS_REPO.parent.mkdir(parents=True, exist_ok=True)
        RULE_STATS_REPO.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")   # めぐ分析が読む写し（commit する）
        tok.finish(result_path=RULE_STATS)
    print("rule_stats", RULE_STATS, {k: len(v) for k, v in out.items()}, flush=True)




if __name__ == "__main__":
    if sys.argv[1:] == ["_inner"]:
        _rule_stats_inner()
    else:
        cmd_rule_stats()
