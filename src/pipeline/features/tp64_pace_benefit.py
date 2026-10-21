"""tp-64: 過去走の『展開利（pace benefit）』と、次走の期待展開利とのずれ。

展開利 = 実測の展開（ペース3段階 × 隊列3段階 = 9 セル）で、その脚質（実際に取った位置）の馬が、同じ脚質の平均より
どれだけ着順（fin_pct, 0=1着〜1=最下位）が良かったか（負号を付けて、正 = 展開に恵まれた）。
係数表は『対象レースの年より前の年』の結果だけで作る（as-of。対象年の結果は一切入らない）。

- benefit_table(runs, year): 年 year より前の走から、(セル, 脚質) ごとの縮約済みの展開利を作る
- build_features(runs, pcell): 馬ごとに直近の走の展開利・展開補正後の着順・次走とのずれを作る
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PACES = ["slow", "mid", "high"]
SHAPES = ["compact", "normal", "stretched"]
CELLS = [f"{p}|{s}" for p in PACES for s in SHAPES]
CELL_IDX = {c: i for i, c in enumerate(CELLS)}
FRONT_MAX, STALKER_MAX, MID_MAX = 0.15, 0.35, 0.65     # tp61_analysis と同じ脚質の境界
SHRINK_M = 150.0                                          # 縮約の擬似件数（セル×脚質の走数がこれくらいで半分信じる）
MIN_FIELD = 8
LUCKY = 0.03                                              # 展開利の『恵まれた／不利』のしきい値（fin_pct の単位）
GOOD = 0.30                                               # 『好走』= fin_pct がこれ以下


def style_id(early_norm) -> np.ndarray:
    """通過位置（0=先頭〜1=後方）→ 0 逃げ / 1 先行 / 2 差し / 3 追込、欠損は -1。"""
    x = np.asarray(early_norm, dtype=float)
    out = np.select([x <= FRONT_MAX, x <= STALKER_MAX, x <= MID_MAX], [0, 1, 2], default=3)
    return np.where(np.isnan(x), -1, out).astype(int)


def cell_id(pace_cls, shape_cls) -> np.ndarray:
    pa = pd.Series(pace_cls).map({p: i for i, p in enumerate(PACES)})
    sh = pd.Series(shape_cls).map({s: i for i, s in enumerate(SHAPES)})
    v = (pa * 3 + sh).to_numpy(dtype=float)
    return np.where(np.isnan(v), -1, v).astype(int)


def benefit_table(runs: pd.DataFrame, year: int, m: float = SHRINK_M) -> np.ndarray:
    """(9 セル × 4 脚質) の展開利。runs は year / field_size / cell_idx / style_idx / fin_pct 列を持つ。
    年 year より前の走だけを使う。件数ゼロのところは 0（= 平均並み）。"""
    r = runs[(runs["year"] < year) & (runs["field_size"] >= MIN_FIELD) & (runs["cell_idx"] >= 0) & (runs["style_idx"] >= 0)]
    r = r[r["fin_pct"].notna()]
    tab = np.zeros((9, 4), dtype=float)
    if len(r) == 0:
        return tab
    base = r.groupby("style_idx")["fin_pct"].mean()
    g = r.groupby(["cell_idx", "style_idx"])["fin_pct"].agg(["mean", "count"])
    for (c, s), row in g.iterrows():
        tab[int(c), int(s)] = -(row["mean"] - base[s]) * row["count"] / (row["count"] + m)
    return tab


def lookup(tab: np.ndarray, cell_idx, style_idx) -> np.ndarray:
    c, s = np.asarray(cell_idx), np.asarray(style_idx)
    ok = (c >= 0) & (s >= 0)
    out = np.full(len(c), np.nan)
    out[ok] = tab[c[ok], s[ok]]
    return out


def expected_today(tab: np.ndarray, pcell: np.ndarray, style_idx) -> np.ndarray:
    """今走の期待展開利 = Σ_セル P(セル) × tab[セル, 今走の脚質]。pcell は (行, 9)。"""
    s = np.asarray(style_idx)
    ok = (s >= 0) & ~np.isnan(pcell).any(axis=1)
    out = np.full(len(s), np.nan)
    out[ok] = (pcell[ok] * tab[:, s[ok]].T).sum(axis=1)
    return out


def _shifted(df: pd.DataFrame, col: str, k: int) -> np.ndarray:
    return df.groupby("horse_id", sort=False)[col].shift(k).to_numpy(dtype=float)


def _nanstat(arrs: list[np.ndarray], fn) -> np.ndarray:
    a = np.vstack(arrs).T
    with np.errstate(all="ignore"):
        out = fn(a, axis=1)
    return np.where(np.isnan(a).all(axis=1), np.nan, out)


FEATURE_COLS = [
    "pb_ben_1", "pb_ben_m3", "pb_ben_m5", "pb_ben_max3",
    "pb_adj_1", "pb_adj_m3", "pb_adj_best3",
    "pb_lucky_good", "pb_hard_good", "pb_lucky_bad", "pb_unlucky_bad",
    "pb_pos_m3", "pb_pos_sd5", "pb_nruns",
    "pb_exp_today", "pb_gap_1", "pb_gap_m3", "pb_exp_rel", "pb_exp_rank",
]


def build_features(runs: pd.DataFrame, pcell: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """runs: 1 行 = 1 頭 1 走（race_id, horse_id, date, year, field_size, fin_pct, early_norm, cell_idx, style_idx）。
    pcell: race_id + p0..p8（今走の事前のセル確率。無ければ pb_exp_today 系は欠損）。
    戻り値: (race_id, horse_id) をキーに FEATURE_COLS を持つ表（years の走だけ）。"""
    runs = runs.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    pc = runs[["race_id"]].merge(pcell, on="race_id", how="left")
    pmat = pc[[f"p{i}" for i in range(9)]].to_numpy(dtype=float) if len(pcell) else np.full((len(runs), 9), np.nan)
    outs = []
    for y in years:
        tab = benefit_table(runs, y)
        d = runs[runs["year"] <= y].copy()
        ben = lookup(tab, d["cell_idx"].to_numpy(), d["style_idx"].to_numpy())
        d["ben"] = ben
        d["adj"] = d["fin_pct"].to_numpy(dtype=float) + ben
        d["early"] = d["early_norm"].where(d["early_norm"] <= 1.0)
        d["good"] = (d["fin_pct"] <= GOOD).astype(float).where(d["fin_pct"].notna())
        b = [_shifted(d, "ben", k) for k in range(1, 6)]
        a = [_shifted(d, "adj", k) for k in range(1, 4)]
        e = [_shifted(d, "early", k) for k in range(1, 6)]
        fin1, good1 = _shifted(d, "fin_pct", 1), _shifted(d, "good", 1)
        f = pd.DataFrame({"race_id": d["race_id"].to_numpy(), "horse_id": d["horse_id"].to_numpy()})
        f["pb_ben_1"] = b[0]
        f["pb_ben_m3"] = _nanstat(b[:3], np.nanmean)
        f["pb_ben_m5"] = _nanstat(b, np.nanmean)
        f["pb_ben_max3"] = _nanstat(b[:3], np.nanmax)
        f["pb_adj_1"] = a[0]
        f["pb_adj_m3"] = _nanstat(a, np.nanmean)
        f["pb_adj_best3"] = _nanstat(a, np.nanmin)
        has = ~np.isnan(b[0]) & ~np.isnan(fin1)
        f["pb_lucky_good"] = np.where(has, ((fin1 <= GOOD) & (b[0] >= LUCKY)).astype(float), np.nan)
        f["pb_hard_good"] = np.where(has, ((fin1 <= GOOD) & (b[0] <= -LUCKY)).astype(float), np.nan)
        f["pb_lucky_bad"] = np.where(has, ((fin1 >= 0.6) & (b[0] >= LUCKY)).astype(float), np.nan)
        f["pb_unlucky_bad"] = np.where(has, ((fin1 >= 0.6) & (b[0] <= -LUCKY)).astype(float), np.nan)
        pos3 = _nanstat(e[:3], np.nanmean)
        f["pb_pos_m3"] = pos3
        cnt = (~np.isnan(np.vstack(e).T)).sum(axis=1)
        sd = _nanstat(e, np.nanstd)
        f["pb_pos_sd5"] = np.where(cnt >= 2, sd, np.nan)
        f["pb_nruns"] = (~np.isnan(np.vstack(b).T)).sum(axis=1).astype(float)
        sty_today = style_id(pos3)
        pm = pmat[d.index.to_numpy()] if len(d) else pmat[:0]
        exp = expected_today(tab, pm, sty_today)
        f["pb_exp_today"] = exp
        f["pb_gap_1"] = b[0] - exp
        f["pb_gap_m3"] = f["pb_ben_m3"].to_numpy() - exp
        f = f[d["year"].to_numpy() == y].reset_index(drop=True)
        g = f.groupby("race_id", sort=False)["pb_exp_today"]
        f["pb_exp_rel"] = f["pb_exp_today"] - g.transform("mean")
        f["pb_exp_rank"] = g.rank(pct=True)
        outs.append(f)
    out = pd.concat(outs, ignore_index=True)
    return out[["race_id", "horse_id"] + FEATURE_COLS]
