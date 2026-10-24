"""払戻（race_result の payoff）で、複数行の券種（複勝・ワイド・同着）の数字と金額が区切りなしで連結されたレースを検出し、
着順・オッズとの整合で分割し直した補正テーブルを作る。

2024-01〜2026-03 のレース（5,901 件）が、複勝 numbers=``341`` payout=``120110140`` のような連結形式で保存されている
（2026-07 の再取得時の保存。popularity も無い）。元の HTML は保持期間（直近 10 件）を過ぎて残っていないので、再取得せずに
  1. 金額を 3 桁または桁区切りカンマ付きのトークンに分け、馬番の組を着順上位の候補で分ける全ての解釈を出し（build_odds_payoff_tables と同じ）、
  2. 複勝は複勝オッズの幅、ワイドはワイドオッズの幅に、払戻/100 が入る解釈だけに絞る
ことで、曖昧さを減らす。結果は ``data/local/meta/payoff_fix/payoff_fix_<年>.parquet`` に追加保存する（既存の race_result は上書きしない）。

python -m src.scraper.payoff_repair --years 2024 2025 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.pipeline.betting import build_odds_payoff_tables as bo

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "data/page_reference/tables"
OUT = ROOT / "data/local/meta/payoff_fix"
MULTI = ("複勝", "ワイド")


def is_concatenated(payoff: dict | None) -> bool:
    """複数行のはずの券種（複勝・ワイド）が、1 つの辞書（popularity なし）に連結されて保存されている。"""
    if not isinstance(payoff, dict):
        return False
    for k in MULTI:
        v = payoff.get(k)
        if isinstance(v, dict) and "popularity" not in v:
            return True
    return False


def load_fix(year: int, path: Path | None = None) -> dict[str, list[dict]]:
    """補正テーブルを race_id → 補正行の一覧にする。曖昧（ambiguous）・解析不能の行を含むレースは入れない（誤った払戻を載せないため）。"""
    f = Path(path) if path else OUT / f"payoff_fix_{year}.parquet"
    if not f.exists():
        return {}
    d = pd.read_parquet(f)
    bad = set(d[d["ambiguous"].astype(bool) | (d["resolved_by"] == "unparsed")]["race_id"])
    d = d[~d["race_id"].isin(bad)]
    return {rid: g.to_dict("records") for rid, g in d.groupby("race_id")}


def apply_fix(payoff: dict, rows: list[dict] | None) -> dict:
    """連結されている券種（複勝・ワイド）だけを補正行で置き換えた払戻を返す（形は通常の複数行と同じ。人気は空）。"""
    if not rows or not isinstance(payoff, dict):
        return payoff
    out = dict(payoff)
    for jp in MULTI:
        v = payoff.get(jp)
        if not (isinstance(v, dict) and "popularity" not in v):
            continue
        rs = [r for r in rows if r["bet_type"] == bo.BET_KEYS[jp] and r.get("combo")]
        if rs:
            out[jp] = [{"numbers": r["combo"], "payout": str(int(r["payout"])), "popularity": ""} for r in rs]
    return out


def _hits(v: dict, bt: str, top3: set[int], field: set[int]) -> list[tuple[list[int], list[tuple[int, ...]]]]:
    ordered = bt in bo.ORDERED
    n = bo.ARITY[bt]
    pays = bo.split_payouts(v.get("payout", ""))
    for cands in (top3, field):
        out = []
        for p in pays:
            for r in bo.parse_rows(v.get("numbers", ""), n, len(p), cands, ordered):
                out.append((p, r))
        if out:
            return out
    return []


def _inside(payout: int, lo: float | None, hi: float | None, tol: float = 0.051) -> bool:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return True
    return lo - tol <= payout / 100 <= hi + tol


def repair_race(payoff: dict, finishers: dict[int, int], field: Iterable[int], place_odds: dict[int, tuple[float, float]],
                wide_odds: dict[str, tuple[float, float]]) -> list[dict]:
    """1 レース分の補正行（bet_type, combo, payout, n_rows, ambiguous, resolved_by）を返す。複勝・ワイドのみ。"""
    top3 = {h for h, p in finishers.items() if 1 <= p <= 3}
    field_set = set(field) | set(finishers)
    rows: list[dict] = []
    for jp in MULTI:
        v = payoff.get(jp)
        if not isinstance(v, dict):
            continue
        bt = bo.BET_KEYS[jp]
        hits = _hits(v, bt, top3, field_set)
        if not hits:
            rows.append({"bet_type": bt, "combo": None, "payout": None, "n_rows": 0, "ambiguous": False, "resolved_by": "unparsed"})
            continue
        uniq = {(tuple(p), tuple(r)) for p, r in hits}
        resolved_by = "unique" if len(uniq) == 1 else ""
        if len(uniq) > 1:
            ok = []
            for p, r in uniq:
                good = True
                for pay, row in zip(p, r):
                    if bt == "fukusho":
                        lo, hi = place_odds.get(row[0], (None, None))
                    else:
                        lo, hi = wide_odds.get("-".join(map(str, sorted(row))), (None, None))
                    if not _inside(pay, lo, hi):
                        good = False
                        break
                if good:
                    ok.append((p, r))
            if len(ok) == 1:
                uniq = {ok[0]}
                resolved_by = "odds"
            elif ok:
                uniq = set(ok)
        ambiguous = len(uniq) > 1
        p, r = sorted(uniq)[0]
        for pay, row in zip(p, r):
            rows.append({"bet_type": bt, "combo": bo.combo_str(sorted(row), False), "payout": int(pay), "n_rows": len(p),
                         "ambiguous": ambiguous, "resolved_by": resolved_by or "first"})
    return rows


def build_year(year: int) -> pd.DataFrame:
    base = TABLES / str(year)
    res = pd.read_parquet(base / "race_result_flat.parquet", columns=["race_id", "horse_number", "finish_position", "payoff"])
    def _opt(name: str, cols: list[str]) -> pd.DataFrame:     # 年によってはオッズの表が無い（その場合はオッズでの絞り込みなし）
        f = base / name
        return pd.read_parquet(f, columns=cols) if f.exists() else pd.DataFrame(columns=cols)

    odds = _opt("race_odds_flat.parquet", ["race_id", "horse_number", "place_odds_min", "place_odds_max"])
    pair = _opt("race_pair_odds_flat.parquet", ["race_id", "odds_type", "pair", "odds_min", "odds_max"])
    pair = pair[pair["odds_type"] == "wide"]
    p_odds = {rid: {int(h): (a, b) for h, a, b in zip(g["horse_number"], g["place_odds_min"], g["place_odds_max"]) if pd.notna(h)} for rid, g in odds.groupby("race_id")}
    w_odds = {rid: {bo.pair_combo(pr, "wide"): (a, b) for pr, a, b in zip(g["pair"], g["odds_min"], g["odds_max"])} for rid, g in pair.groupby("race_id")}
    out = []
    for rid, g in res.groupby("race_id", sort=False):
        raw = g["payoff"].iloc[0]
        try:
            payoff = json.loads(raw) if isinstance(raw, str) and raw else {}
        except ValueError:
            payoff = {}
        if not is_concatenated(payoff):
            continue
        fin = {int(h): int(f) for h, f in zip(g["horse_number"], g["finish_position"]) if pd.notna(h) and pd.notna(f) and f > 0}
        for r in repair_race(payoff, fin, [int(h) for h in g["horse_number"].dropna()], p_odds.get(rid, {}), w_odds.get(rid, {})):
            r["race_id"] = rid
            out.append(r)
    return pd.DataFrame(out, columns=["race_id", "bet_type", "combo", "payout", "n_rows", "ambiguous", "resolved_by"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for y in a.years:
        d = build_year(y)
        d.to_parquet(OUT / f"payoff_fix_{y}.parquet", index=False)
        races = d["race_id"].nunique()
        amb = d[d["ambiguous"]]["race_id"].nunique()
        un = d[d["resolved_by"] == "unparsed"]["race_id"].nunique()
        summary[y] = {"races_fixed": int(races), "ambiguous_races": int(amb), "unparsed_races": int(un),
                      "resolved_by": d["resolved_by"].value_counts().to_dict()}
        print(y, summary[y], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
