"""基準線（一般的な傾向）: 1 レースの値 S の期待 E と分散 V を、as-of（その日より前のレースだけ）の区分の平均で作る。

縮小の段（経験ベイズ。m = (和 + k·上の段) / (数 + k)）:
  1. 芝ダ（surf）
  2. 芝ダ × 距離帯
  3a. 場 × 芝ダ × 距離帯（コースの癖）            → 2 へ縮める
  3b. 芝ダ × 距離帯 × クラス群 × 馬場 × 頭数群（条件の癖。場を問わない） → 2 へ縮める
  4. 場 × 芝ダ × 距離帯 × クラス群 × 馬場 × 頭数群 → 3a + 3b − 2（コースの癖に条件の癖を足したもの）へ縮める
追加の癖の項（extras。採るかは compare で決める）:
  meet: 場 × 芝ダ × 開催の何日目の帯 × 仮柵（芝の A・B・C・D）→ 場 × 芝ダ へ縮め、(meet − 場 × 芝ダ) を足す（距離を問わない開催の癖）
  firm: 芝ダ × 距離帯 × 硬さの帯（芝はクッション値〔2020 年の途中から〕、ダートはゴール前の含水率〔2018 年 7 月から〕）→ 2 へ縮め、(firm − 2) を足す
分散 V は、残り (S − E)² の as-of の平均（芝ダ × 距離帯 × 頭数群 → 芝ダ × 頭数群 へ縮める）。日の z の尺度は後で校正する。
縮小の強さ k は 2016〜24 だけで選ぶ（choose_k）。2025・26 年は選び直しに使わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K_GRID = (5, 20, 50, 100, 200, 500, 1000, 2000, 5000)
L1 = ["surf"]
L2 = ["surf", "dist_band"]
L3V = ["venue", "surf", "dist_band"]
L3C = ["surf", "dist_band", "class_g", "cond_g", "field_g"]
L4 = ["venue", "surf", "dist_band", "class_g", "cond_g", "field_g"]
VAR_FINE = ["surf", "dist_band", "field_g"]
VAR_COARSE = ["surf", "field_g"]


def asof_sums(R: pd.DataFrame, keys: list[str], col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各レースについて、同じ区分（keys）で、その日より前の日のレースの 数・和・二乗和（col が NaN のレースは数えない）。"""
    v = pd.to_numeric(R[col], errors="coerce").astype(float)
    ok = v.notna()
    t = pd.DataFrame({k: R[k] for k in keys})
    t["date"] = R["date"]
    t["n"] = ok.astype(float)
    t["s"] = v.where(ok, 0.0)
    t["s2"] = (v * v).where(ok, 0.0)
    d = t.groupby(keys + ["date"], sort=True, observed=True)[["n", "s", "s2"]].sum()
    c = d.groupby(level=list(range(len(keys))), sort=False, observed=True).cumsum() - d   # その日を除いた累計（as-of）
    m = t[keys + ["date"]].merge(c.reset_index(), on=keys + ["date"], how="left")
    return (m["n"].to_numpy(dtype=float), m["s"].to_numpy(dtype=float), m["s2"].to_numpy(dtype=float))


def _shrink(n, s, prior, k):
    return (s + k * prior) / (n + k)


EXTRA_TERMS = {"meet": (["venue", "surf", "meet_band", "rail"], "evs"), "firm": (["surf", "dist_band", "firm_band"], "e2")}
LVS = ["venue", "surf"]


def expect(R: pd.DataFrame, col: str, k: float, extras=()) -> np.ndarray:
    """as-of の縮小の期待 E（段は冒頭の説明）。"""
    n1, s1, _ = asof_sums(R, L1, col)
    e1 = np.where(n1 > 0, s1 / np.maximum(n1, 1), 0.0)
    n2, s2, _ = asof_sums(R, L2, col)
    e2 = _shrink(n2, s2, e1, k)
    n3, s3, _ = asof_sums(R, L3V, col)
    e3v = _shrink(n3, s3, e2, k)
    n3c, s3c, _ = asof_sums(R, L3C, col)
    e3c = _shrink(n3c, s3c, e2, k)
    prior = e3v + e3c - e2
    parents = {"e2": e2}
    if extras:
        nvs, svs, _ = asof_sums(R, LVS, col)
        parents["evs"] = _shrink(nvs, svs, e1, k)
    for name in extras:
        keys, par = EXTRA_TERMS[name]
        nx, sx, _ = asof_sums(R, keys, col)
        prior = prior + (_shrink(nx, sx, parents[par], k) - parents[par])
    n4, s4, _ = asof_sums(R, L4, col)
    return _shrink(n4, s4, prior, k)


def variance(R: pd.DataFrame, col: str, e: np.ndarray, k: float = 50) -> np.ndarray:
    """as-of の残り (S − E)² の平均（芝ダ × 距離帯 × 頭数群 → 芝ダ × 頭数群 へ縮める）。分子 S − E と同じ尺度。
    E も as-of なので、前の日のレースの残りだけを使う。履歴が 1 件以下の区分は NaN（日の値に入れない）。"""
    t = R[["date"] + VAR_COARSE + [c for c in VAR_FINE if c not in VAR_COARSE]].copy()
    t["_r2"] = (pd.to_numeric(R[col], errors="coerce").to_numpy(dtype=float) - e) ** 2
    nc, sc, _ = asof_sums(t, VAR_COARSE, "_r2")
    vc = np.where(nc > 1, sc / np.maximum(nc, 1), np.nan)
    nf, sf, _ = asof_sums(t, VAR_FINE, "_r2")
    v = _shrink(nf, sf, vc, k)
    return np.where(np.isfinite(v), np.maximum(v, 1e-6), np.nan)


def choose_k(R: pd.DataFrame, col: str, lo: str = "2018-01-01", hi: str = "2024-12-31", extras=()) -> tuple[int, dict]:
    """2018〜24 のレースで (S − E)² の平均が最小の k（as-of なので先の日は入らない）。"""
    sel = ((R["date"] >= lo) & (R["date"] <= hi) & R[col].notna()).to_numpy()
    mse = {}
    for k in K_GRID:
        e = expect(R, col, k, extras)
        mse[k] = float(np.mean((R[col].to_numpy()[sel] - e[sel]) ** 2))
    best = min(mse, key=mse.get)
    return int(best), mse


def add_baseline(R: pd.DataFrame, cols, ks: dict[str, int], extras: dict | None = None) -> pd.DataFrame:
    """R に E_<col>・V_<col> を足す。extras は 値 → 追加の癖の項の名前の並び。"""
    R = R.copy()
    extras = extras or {}
    for c in cols:
        e = expect(R, c, ks.get(c, 50), tuple(extras.get(c, ())))
        R[f"E_{c}"] = e
        R[f"V_{c}"] = variance(R, c, e)
    return R
