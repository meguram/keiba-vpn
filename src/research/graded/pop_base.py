"""重賞の「好走データ予想」の比べる相手: 人気ごとの成績（G1〜G3 の全出走）を、期間（2018〜2024・2025・2026）ごとに数える。

画面では、◎ に選ばれた馬と同じ人気の馬の値（◎ の人気の構成で重みづけ）を、◎ の成績の横に並べる。
◎ の値がこれを上回らなければ、人気で選ぶのと変わらない（これまでの確かめでは、ルールの ◎ はオッズに織り込み済み）。

入力: 手元の race_result_flat・race_result_race（GCS は読まない）。出力: data/meta/graded/pop_base.json（commit する）。
2025・2026 年の着順・払戻を読むので、見た回数の台帳に eval_only（表示用）として記録する。
使い方: KEIBA_SESSION=<担当名> python -m src.research.graded.pop_base
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.graded import payout as PO
from src.research.graded.features import TABLES

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/meta/graded/pop_base.json"
POP_CAP = 13          # 13 番人気以下はまとめる
PERIODS = {"dev": (2018, 2024), "2025": (2025, 2025), "2026": (2026, 2026)}
JST = timezone(timedelta(hours=9))


def period_table(d: pd.DataFrame) -> dict:
    """d（G1〜G3 の出走。pop・fin・ret_tan・ret_fuku・pay_ok）→ 人気ごとの {n, rec, npay, tan, fuku}。"""
    pop = np.clip(pd.to_numeric(d["pop"], errors="coerce").fillna(0).astype(int).to_numpy(), 0, POP_CAP)
    fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy()
    pay = d["pay_ok"].to_numpy(bool)
    out = {}
    for p in range(1, POP_CAP + 1):
        m = pop == p
        if not m.any():
            continue
        f = fin[m]
        out[str(p)] = {"n": int(m.sum()), "rec": [int((f == 1).sum()), int((f == 2).sum()), int((f == 3).sum()), int((~(f <= 3)).sum())],
                       "npay": int(pay[m].sum()), "tan": float(d["ret_tan"].to_numpy()[m & pay].sum()),
                       "fuku": float(d["ret_fuku"].to_numpy()[m & pay].sum())}
    return out


def load_graded(years: range) -> pd.DataFrame:
    parts = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["race_id", "date", "grade", "horse_number", "finish_position", "popularity"]))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["grade"].astype(str).isin(["G1", "G2", "G3"])].copy()
    d["hn"] = pd.to_numeric(d["horse_number"], errors="coerce")
    d["fin"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d["pop"] = pd.to_numeric(d["popularity"], errors="coerce")
    d["race_id"] = d["race_id"].astype(str)
    hits, ok = PO.load_win_place(TABLES, years)
    return PO.attach_returns(d, hits, ok)


def build(d: pd.DataFrame) -> dict:
    yr = pd.to_datetime(d["date"]).dt.year.to_numpy()
    return {k: period_table(d[(yr >= lo) & (yr <= hi)]) for k, (lo, hi) in PERIODS.items()}


def main() -> None:
    from src.ops import data_ledger as DL
    with DL.guard(tool="src/research/graded/pop_base.py:main", family="graded", version="pop_base", period=("2025-01-01", "2026-11-20"),
                  purpose="eval_only", reason="めぐ分析の表示用に、重賞の人気ごとの成績（比べる相手）を数える（規則は選び直さない・採否に使わない）") as tok:
        d = load_graded(range(2018, 2027))
        tok.check_dates(d.loc[pd.to_datetime(d["date"]).dt.year >= 2025, "date"])
        res = {"made_at": datetime.now(JST).isoformat(timespec="seconds"), "periods": {k: list(v) for k, v in PERIODS.items()},
               "data_until": str(pd.to_datetime(d["date"]).max())[:10], "pop_cap": POP_CAP, "values": build(d),
               "note": "G1〜G3 の全出走の、人気（確定）ごとの [1-2-3-着外]・払戻の和。比べる相手の表示だけに使う"}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=OUT)
    print(OUT, {k: sum(v["n"] for v in t.values()) for k, t in res["values"].items()})


def same_pop(table: dict, pops: list[int]) -> dict | None:
    """◎ の人気の並び（pops）と同じ人気の馬の値: 人気ごとの率を、pops の構成で重みづけ。"""
    w = {}
    for p in pops:
        if p is None or not p or int(p) < 1:     # 人気が不明（0・欠け）の ◎ は比べない
            continue
        k = str(int(min(max(int(p), 1), POP_CAP)))
        w[k] = w.get(k, 0) + 1
    num = {"place": 0.0, "tan": 0.0, "fuku": 0.0}
    den = {"n": 0.0, "pay": 0.0}
    for k, c in w.items():
        t = table.get(k)
        if not t or not t["n"]:
            continue
        num["place"] += c * (t["rec"][0] + t["rec"][1] + t["rec"][2]) / t["n"]
        den["n"] += c
        if t["npay"]:
            num["tan"] += c * t["tan"] / t["npay"]
            num["fuku"] += c * t["fuku"] / t["npay"]
            den["pay"] += c
    if not den["n"]:
        return None
    return {"place": num["place"] / den["n"], "roi_t": num["tan"] / den["pay"] if den["pay"] else None,
            "roi_f": num["fuku"] / den["pay"] if den["pay"] else None, "n_picks": int(den["n"])}


if __name__ == "__main__":
    main()
