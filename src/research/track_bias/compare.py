"""判定の作りの候補を、同じ物差しで比べて採否を決める（2016〜24 だけ。2025・26 年は使わない）。

物差しと採否の決まり（2026-11-26、数字を見る前に固定）:
- 基準線の追加の項（baseline.EXTRA_TERMS の meet・firm）: 1 レースの値の as-of の誤差 (S − E)² の平均で比べる。
  meet は 2018〜24、firm は 2022〜24（クッション値が揃うのは 2021 年から）の窓。窓の誤差が 0.05% 以上下がった項を採る（値ごと）。
  基準線の項は「一般的な傾向」の作りなので、日の偏りの割合ではなく予測の誤差で決める（偏りを基準線に吸わせると割合は下がって見えるため）。
- 1 レースの値（前後・内外の x）の候補: 日の値の 4 つの物差しで比べる（2017〜24、レース 3 以上の日）。
  1. 本当の偏りの割合 1 − 1/s²（芝・ダートの平均）
  2. 日の中の持続: 前半（〜6R）と後半（7R〜）の z_raw の相関（各 2 レース以上）
  3. 翌日の持続: 同じ場・芝ダの 1〜2 日後の日との相関
  4. 同じ開催の中の持続: 同じ年・場・回・芝ダの、ほかの日の z_raw の平均との相関
  主を替えるのは、候補が今の主より 2〜4 の 3 つのうち 2 つ以上で高く、かつ 1 が下がらないときだけ。
  前後の候補: front_early（今の主）・front_l3f・front_combo。4 角の版は着順の結果を含むので主の候補にしない。
  内外の候補: inner_b（統括役の決定で主）・inner_b_r・inner_b_g。案 A・C は並べるだけ。

レビューの後に直した点（2026-11-26 の 2 回目の比べの後。critic の指摘。直した後に比べ直した）:
- 候補の値は、追加の項なしの同じ基準線で比べる（1 回目は候補ごとに採った項が違い、同じ物差しでなかった）。追加の項は、主を決めた後に値ごとに採否する。
- 2〜4 の「高い」は、差が 0.01 を超えたときだけ数える（それより小さい差は同じ扱い）。持続の相関は芝・ダートで別に取って平均する。
- 決まりを満たす候補が複数あるときは、1 の高い方を採る。
- 追加の項を 2 つとも採った値は、2 つを合わせた誤差も確かめ、1 つだけより悪ければ良い方の 1 つにする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.track_bias import baseline as B
from src.research.track_bias import data as D
from src.research.track_bias import day as Y

FRONT_CANDS = ("front_early", "front_l3f", "front_combo")
INNER_CANDS = ("inner_b", "inner_b_r", "inner_b_g")
EXTRA_SETS = {"meet": ("2018-01-01", "2024-12-31"), "firm": ("2022-01-01", "2024-12-31")}
MSE_GAIN = 0.0005
TIE = 0.01


def _mse(R, col, e, lo, hi):
    sel = ((R["date"] >= lo) & (R["date"] <= hi) & R[col].notna()).to_numpy()
    return float(np.mean((R[col].to_numpy()[sel] - e[sel]) ** 2))


def choose_extras(R: pd.DataFrame, ks: dict, cols) -> tuple[dict, dict]:
    """値ごとに、追加の項を 1 つずつ足して誤差が下がるかを見る（決まりは冒頭）。"""
    chosen, table = {}, {}
    for c in cols:
        base = B.expect(R, c, ks[c])
        row, take = {}, []
        for name, (lo, hi) in EXTRA_SETS.items():
            m0 = _mse(R, c, base, lo, hi)
            m1 = _mse(R, c, B.expect(R, c, ks[c], (name,)), lo, hi)
            row[name] = {"window": [lo, hi], "mse_without": m0, "mse_with": m1, "gain": 1 - m1 / m0, "adopt": bool(1 - m1 / m0 >= MSE_GAIN)}
            if row[name]["adopt"]:
                take.append(name)
        if len(take) == 2:
            lo, hi = EXTRA_SETS["firm"]                       # 両方の値がある窓
            both = _mse(R, c, B.expect(R, c, ks[c], tuple(take)), lo, hi)
            singles = {n: _mse(R, c, B.expect(R, c, ks[c], (n,)), lo, hi) for n in take}
            row["both"] = {"window": [lo, hi], "mse": both, "singles": singles}
            if both > min(singles.values()):
                take = [min(singles, key=singles.get)]
        chosen[c], table[c] = take, row
    return chosen, table


def _zr(g: pd.DataFrame, col: str) -> float:
    ok = g[col].notna() & g[f"E_{col}"].notna() & g[f"V_{col}"].notna()
    if ok.sum() < 2:
        return np.nan
    v = g.loc[ok, f"V_{col}"].sum()
    return float((g.loc[ok, col] - g.loc[ok, f"E_{col}"]).sum() / np.sqrt(v)) if v > 0 else np.nan


def persistence(R: pd.DataFrame, days: pd.DataFrame, col: str) -> dict:
    """物差し 2〜4 を芝・ダートで別に取り、平均する（数は足す）。"""
    per = {s: _persistence(R[R["surf"] == s], days[days["surf"] == s], col) for s in ("芝", "ダ")}
    out = {k: float(np.mean([per[s][k] for s in per])) for k in ("split_half", "next_day", "same_meet")}
    out.update({k: int(sum(per[s][k] for s in per)) for k in ("split_days", "next_pairs", "meet_days")})
    out["by_surf"] = per
    return out


def _persistence(R: pd.DataFrame, days: pd.DataFrame, col: str) -> dict:
    """物差し 2〜4（決まりは冒頭）。days は Y.day_raw の出力（kai を足したもの）。"""
    sel = days[(days["date"] >= Y.CAL_FROM) & (days["date"] <= Y.CAL_TO) & (days[f"n_{col}"] >= Y.MIN_RACES)].copy()
    rr = R[(R["date"] >= Y.CAL_FROM) & (R["date"] <= Y.CAL_TO)]
    h1 = rr[rr["round"] <= 6].groupby(Y.DAY_KEYS).apply(lambda g: _zr(g, col)).rename("a")
    h2 = rr[rr["round"] > 6].groupby(Y.DAY_KEYS).apply(lambda g: _zr(g, col)).rename("b")
    hh = pd.concat([h1, h2], axis=1).dropna()
    split = float(hh["a"].corr(hh["b"]))
    z = f"zr_{col}"
    s = sel.copy()
    s["dt"] = pd.to_datetime(s["date"])
    s = s.sort_values(["venue", "surf", "dt"])
    nxt = s.groupby(["venue", "surf"])[[z, "dt"]].shift(-1)
    ok = (nxt["dt"] - s["dt"]).dt.days.between(1, 2)
    nd = float(s.loc[ok, z].corr(nxt.loc[ok, z]))
    s["yr"] = s["date"].str[:4]
    m = s.dropna(subset=["kai", z])
    key = ["yr", "venue", "kai", "surf"]
    tot = m.groupby(key)[z].transform("sum")
    cnt = m.groupby(key)[z].transform("count")
    loo = (tot - m[z]) / (cnt - 1)
    okm = cnt >= 2
    meet = float(m.loc[okm, z].corr(loo[okm]))
    return {"split_half": split, "split_days": int(len(hh)), "next_day": nd, "next_pairs": int(ok.sum()), "same_meet": meet, "meet_days": int(okm.sum())}


def true_share(cal: dict, col: str) -> float:
    return float(np.mean([cal["metrics"][col][s]["true_share"] for s in ("芝", "ダ") if s in cal["metrics"][col]]))


def pick(current: str, cands, ys: dict) -> tuple[str, list[str]]:
    """主を替えるかの決まり（冒頭）。理由の文も返す。"""
    notes = []
    best = current
    for c in cands:
        if c == current:
            continue
        wins = sum(ys[c][k] > ys[current][k] + TIE for k in ("split_half", "next_day", "same_meet"))
        ok_share = ys[c]["true_share"] >= ys[current]["true_share"]
        notes.append(f"{c}: 持続の 3 つのうち {wins} つで {current} より 0.01 を超えて高い・本当の偏りの割合は {'下がらない' if ok_share else '下がる'}")
        if wins >= 2 and ok_share and (best == current or ys[c]["true_share"] > ys[best]["true_share"]):
            best = c
    return best, notes


def run() -> dict:
    h = D.horse_values(D.load_horses(range(D.FIRST_YEAR, 2025)))
    R = D.race_values(h)
    ks = {}
    for c in D.METRICS:
        ks[c], _ = B.choose_k(R, c)
    for c in D.AUX:
        ks[c] = ks["front_early"]
    # 1. 候補の値を、追加の項なしの同じ基準線で比べて主を決める
    R0 = B.add_baseline(R, D.METRICS + D.AUX, ks)
    d0 = Y.day_raw(R0)
    d0 = d0.merge(R0.groupby(Y.DAY_KEYS)["kai"].first().reset_index(), on=Y.DAY_KEYS, how="left")
    cal0 = Y.calibrate(d0)
    ys = {c: {"true_share": true_share(cal0, c), **persistence(R0, d0, c)} for c in D.METRICS}
    front, fnotes = pick("front_early", FRONT_CANDS, ys)
    inner, inotes = pick("inner_b", INNER_CANDS, ys)
    # 2. 追加の項を値ごとに採否し、採った基準線で校正する
    extras, ex_table = choose_extras(R, ks, D.METRICS)
    R = B.add_baseline(R, D.METRICS + D.AUX, ks, extras)
    days = Y.day_raw(R)
    days = days.merge(R.groupby(Y.DAY_KEYS)["kai"].first().reset_index(), on=Y.DAY_KEYS, how="left")
    cal = Y.calibrate(days)
    ys_final = {c: {"true_share": true_share(cal, c), **persistence(R, days, c)} for c in (front, inner)}
    return {"k": ks, "extras": extras, "extras_table": ex_table, "yardstick": ys, "yardstick_final": ys_final,
            "main": {"front": front, "inner": inner}, "notes": {"front": fnotes, "inner": inotes}, "cal": cal, "days": days, "R": R}
