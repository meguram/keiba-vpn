"""その日の値・校正・5 段の区分・1 行の名札。

- その日の値 D = Σ_r (S_r − E_r)、z_raw = D / √(Σ V_r)（日 = 日付 × 場 × 芝ダ）。
- 校正: 2017〜24 の日（レース 3 以上）の z_raw の平均 μ と SD s を、値（METRICS）× 芝ダ ごとに取り、z = (z_raw − μ) / s。
  2016 年は基準線の溜め（as-of の履歴が短い）なので校正に入れない。2025・26 年は校正に使わない。
- 区分の閾値（数字を見る前に決めた）: |z| < 1 中立、1 ≤ |z| < 2 有利、|z| ≥ 2 強い。レース 3 未満は判定なし。
- 区間: D の 80% 区間 = D ± 1.28 · s · √ΣV（校正の尺度で広げた区間）。
- 名札: 内外 × 前後 を 1 文に（両方中立なら「中立」）。主の値は校正の JSON の main（compare で決める。既定は前後 front_early・内外 inner_b_g）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.track_bias.data import AUX, METRICS

DAY_KEYS = ["date", "venue", "surf"]
CAL_FROM, CAL_TO = "2017-01-01", "2024-12-31"
MIN_RACES = 3
TH_WEAK, TH_STRONG = 1.0, 2.0
Z80 = 1.2816
DEFAULT_FRONT, DEFAULT_INNER = "front_early", "inner_b_g"
FRONT_MAIN = DEFAULT_FRONT      # 主の値。calibrate の比較（compare）で決まり、set_main(cal) で校正の JSON の main に合わせる
INNER_MAIN = DEFAULT_INNER      # 内外は案 B 系（2026-11-26 の統括役の決定）。変種は compare の決まりで決まる


def set_main(cal: dict) -> None:
    """校正の JSON の main（compare で決めた主）を、このモジュールの主の値にする。"""
    global FRONT_MAIN, INNER_MAIN
    m = cal.get("main") or {}
    FRONT_MAIN = m.get("front", DEFAULT_FRONT)       # main が無い JSON は既定に戻す（呼ぶ順で主が変わらない）
    INNER_MAIN = m.get("inner", DEFAULT_INNER)

FRONT_WORDS = {2: "強い前残り", 1: "前残り", 0: "前後は中立", -1: "差し有利", -2: "強い差し有利"}
INNER_WORDS = {2: "強い内有利", 1: "内有利", 0: "内外は中立", -1: "外有利", -2: "強い外有利"}
NO_JUDGE = f"判定なし（レース {MIN_RACES} 未満）"


def level(z: float) -> int:
    if z is None or not np.isfinite(z):
        return 0
    a = abs(z)
    lv = 2 if a >= TH_STRONG else (1 if a >= TH_WEAK else 0)
    return int(np.sign(z) * lv)


def label(front_lv: int, inner_lv: int, n_races: int) -> str:
    if n_races < MIN_RACES:
        return NO_JUDGE
    parts = []
    if inner_lv:
        parts.append(INNER_WORDS[inner_lv])
    if front_lv:
        parts.append(FRONT_WORDS[front_lv])
    return "・".join(parts) if parts else "中立"


def day_raw(R: pd.DataFrame, cols=METRICS + AUX, keys=DAY_KEYS) -> pd.DataFrame:
    """日ごとの D・ΣV・z_raw（cols ごと）と、レース数。"""
    t = R[keys].copy()
    for c in cols:
        ok = R[c].notna() & R[f"E_{c}"].notna() & R[f"V_{c}"].notna()
        t[f"D_{c}"] = (R[c] - R[f"E_{c}"]).where(ok)
        t[f"V_{c}"] = R[f"V_{c}"].where(ok)
        t[f"n_{c}"] = ok.astype(int)
    g = t.groupby(keys, sort=True)
    out = g.agg(**{f"{p}_{c}": (f"{p}_{c}", "sum") for c in cols for p in ("D", "V", "n")})
    out["n_races"] = g.size()
    out = out.astype({f"{p}_{c}": float for c in cols for p in ("D", "V")})
    for c in cols:
        out[f"zr_{c}"] = out[f"D_{c}"] / np.sqrt(out[f"V_{c}"].where(out[f"V_{c}"] > 0))
    return out.reset_index()


def calibrate(days: pd.DataFrame, cols=METRICS + AUX) -> dict:
    """2017〜24 の日の z_raw の平均・SD（値 × 芝ダ）と、5 段の出現率。"""
    cal: dict = {"from": CAL_FROM, "to": CAL_TO, "min_races": MIN_RACES, "thresholds": [TH_WEAK, TH_STRONG], "metrics": {}}
    for c in cols:
        cal["metrics"][c] = {}
        sel = days[(days["date"] >= CAL_FROM) & (days["date"] <= CAL_TO) & (days[f"n_{c}"] >= MIN_RACES)]
        for s, g in sel.groupby("surf"):
            z = g[f"zr_{c}"].dropna()
            mu, sd = float(z.mean()), float(z.std(ddof=1))
            zz = (z - mu) / sd
            lv = zz.map(level)
            cal["metrics"][c][s] = {
                "mu": mu, "sd": sd, "days": int(len(z)),
                # 偶然だけなら sd ≈ 1。sd > 1 の分（1 − 1/sd²）が、日の本当の偏りの分散の割合の目安
                "true_share": float(max(0.0, 1 - 1 / sd ** 2)),
                "freq": {str(k): float((lv == k).mean()) for k in (-2, -1, 0, 1, 2)},
            }
    return cal


def apply(days: pd.DataFrame, cal: dict, cols=METRICS + AUX) -> pd.DataFrame:
    """z・区分・区間・名札を足す（主の値は校正の JSON の main）。"""
    set_main(cal)
    d = days.copy()
    for c in cols:
        mu = d["surf"].map(lambda s: cal["metrics"][c].get(s, {}).get("mu", 0.0))
        sd = d["surf"].map(lambda s: cal["metrics"][c].get(s, {}).get("sd", 1.0))
        d[f"z_{c}"] = (d[f"zr_{c}"] - mu) / sd
        half = Z80 * sd * np.sqrt(d[f"V_{c}"])
        d[f"lo_{c}"] = d[f"D_{c}"] - half
        d[f"hi_{c}"] = d[f"D_{c}"] + half
        d[f"lv_{c}"] = [level(z) if n >= MIN_RACES else 0 for z, n in zip(d[f"z_{c}"], d[f"n_{c}"])]
        d[f"ok_{c}"] = d[f"n_{c}"] >= MIN_RACES             # 値ごとの判定の可否（案 B は 4 角の並びの欠けで数が減る）
    nf, ni = d[f"n_{FRONT_MAIN}"], d[f"n_{INNER_MAIN}"]
    d["front_label"] = [FRONT_WORDS[v] if n >= MIN_RACES else NO_JUDGE for v, n in zip(d[f"lv_{FRONT_MAIN}"], nf)]
    d["inner_label"] = [INNER_WORDS[v] if n >= MIN_RACES else NO_JUDGE for v, n in zip(d[f"lv_{INNER_MAIN}"], ni)]
    d["label"] = [label(f, i, min(a, b)) for f, i, a, b in zip(d[f"lv_{FRONT_MAIN}"], d[f"lv_{INNER_MAIN}"], nf, ni)]
    d["front_main"], d["inner_main"] = FRONT_MAIN, INNER_MAIN
    return d


def race_contrib(R: pd.DataFrame, days: pd.DataFrame, cal: dict, cols=None) -> pd.DataFrame:
    """レースごとの S − E と、日の z への寄与（寄与の和 + その日の −μ/s の分 = 日の z）。"""
    if cols is None:
        cols = tuple(dict.fromkeys(METRICS))
    m = R.merge(days[DAY_KEYS + [f"V_{c}" for c in cols]].rename(columns={f"V_{c}": f"dV_{c}" for c in cols}), on=DAY_KEYS, how="left")
    for c in cols:
        sd = m["surf"].map(lambda s: cal["metrics"][c].get(s, {}).get("sd", 1.0))
        m[f"SE_{c}"] = m[c] - m[f"E_{c}"]
        m[f"zc_{c}"] = m[f"SE_{c}"] / (sd * np.sqrt(m[f"dV_{c}"]))
    return m


def segments(R: pd.DataFrame, cal: dict, col: str | None = None) -> pd.DataFrame:
    """1 日の中の区切り（前半 〜6R・後半 7R〜、馬場の替わり）ごとの z（記述だけ。区分は付けない）。
    区切りの z は、その区切りのレースだけの D / √(区切りの ΣV) を日の尺度 s で割ったもの（μ は引かない）。
    分母が区切りの ΣV なので、race_contrib の寄与（分母は日の ΣV）を区切りで足した値とは一致しない。"""
    col = col or FRONT_MAIN
    rows = []
    for key, g in R.groupby(DAY_KEYS, sort=True):
        g = g.sort_values("round")
        sd = cal["metrics"][col].get(key[2], {}).get("sd", 1.0)
        segs = [("前半（〜6R）", g[g["round"] <= 6]), ("後半（7R〜）", g[g["round"] > 6])]
        conds = list(dict.fromkeys(g["track_condition"].tolist()))
        if len(conds) > 1:
            segs += [(f"馬場 {c}", g[g["track_condition"] == c]) for c in conds]
        for name, s in segs:
            ok = s[col].notna() & s[f"E_{col}"].notna() & s[f"V_{col}"].notna()
            if not ok.any():
                continue
            D = float((s[col] - s[f"E_{col}"])[ok].sum())
            V = float(s[f"V_{col}"][ok].sum())
            rows.append(dict(zip(DAY_KEYS, key), segment=name, n_races=int(ok.sum()), D=D, z=D / np.sqrt(V) / sd if V > 0 else np.nan))
    return pd.DataFrame(rows)
