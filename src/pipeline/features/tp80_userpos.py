"""tp-80: ユーザー手順の位置取り予測（テン1F の加重平均 → 枠番補正 → 早い順に隊列 → 脚質区分）とペース予測の部品。

手順（ユーザー）:
1. 直近のテン1F の平均（直近重視の加重平均。重み λ^(k-1)、k=1 が最新）
2. 枠番で補正（外枠ほど遅く。係数はコース×距離帯ごとにデータで推定し、行数が少ないセルは上位へ縮約）
3. 補正後の想定テン1F が速い順に前から並べる。同じ値なら馬番が小さい（内）ほうを前にする
先読み防止: 過去走は rdate < 当該レース日 のものだけ。係数は評価年より前の年だけで当てはめる。
キーは race_id / horse_id（名前で結合しない）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FRONT_MAX, STALKER_MAX, MID_MAX = 0.15, 0.35, 0.65
K_MAX = 8


def add_race_relative(career: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """career に ten_rel（同じ rcode の SmartRC 掲載馬のテン平均との差。5 頭未満は欠損）を足す。"""
    c = career.copy()
    g = c.groupby("rcode")["ten"]
    n = g.transform("count")
    m = g.transform("mean")
    c["ten_rel"] = np.where(n >= min_n, c["ten"] - m, np.nan)
    return c


def history_matrix(entries: pd.DataFrame, career: pd.DataFrame, col: str, k: int = K_MAX) -> np.ndarray:
    """各出走の『それより前の有効な過去走の col』を新しい順に k 本（足りない分は NaN）。形 (n_entries, k)。"""
    ent_h = entries["horse_id"].astype(str).to_numpy()
    ent_d = pd.to_datetime(entries["date"]).dt.strftime("%Y%m%d").astype(int).to_numpy()
    out = np.full((len(entries), k), np.nan, dtype="float64")
    c = career[["horse_id", "rdate", col]].dropna(subset=[col]).copy()
    c["horse_id"] = c["horse_id"].astype(str)
    c["rdate"] = c["rdate"].astype(int)
    c = c.sort_values(["horse_id", "rdate"], kind="stable")
    ch = c["horse_id"].to_numpy()
    cd = c["rdate"].to_numpy()
    cv = c[col].to_numpy(dtype="float64")
    hs, hstart = np.unique(ch, return_index=True)
    hend = np.append(hstart[1:], len(ch))
    pos_of = {h: (s, e) for h, s, e in zip(hs, hstart, hend)}
    order = np.argsort(ent_h, kind="stable")
    eh_sorted = ent_h[order]
    uh, ustart = np.unique(eh_sorted, return_index=True)
    uend = np.append(ustart[1:], len(order))
    ar = np.arange(k)
    for h, s, e in zip(uh, ustart, uend):
        if h not in pos_of:
            continue
        a, b = pos_of[h]
        rd, v = cd[a:b], cv[a:b]
        ix = order[s:e]
        m = np.searchsorted(rd, ent_d[ix], side="left")          # rd < レース日 の本数（同日は含めない）
        idx = m[:, None] - 1 - ar[None, :]
        ok = idx >= 0
        out[ix] = np.where(ok, v[np.clip(idx, 0, None)], np.nan)
    return out


def weighted_mean(M: np.ndarray, lam: float, k: int | None = None) -> np.ndarray:
    """新しい順の履歴行列 M の λ^(j) 加重平均（NaN は飛ばす）。k で使う本数を絞る。全部 NaN なら NaN。"""
    if k is not None:
        M = M[:, :k]
    w = lam ** np.arange(M.shape[1])[None, :]
    ok = ~np.isnan(M)
    num = np.where(ok, M, 0.0) @ w[0]
    den = (ok * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gate_rel(horse_number: np.ndarray, field_size: np.ndarray) -> np.ndarray:
    """0=最内 … 1=最外。"""
    hn = np.asarray(horse_number, dtype="float64")
    fs = np.asarray(field_size, dtype="float64")
    return np.where(fs > 1, (hn - 1) / np.maximum(fs - 1, 1), 0.5)


def dist_band(distance: np.ndarray) -> np.ndarray:
    d = np.asarray(distance, dtype="float64")
    return np.select([d <= 1200, d <= 1600, d <= 2000, d <= 2400], [0, 1, 2, 3], 4)


def fit_gate_coef(resid: np.ndarray, g: np.ndarray, venue: np.ndarray, surface: np.ndarray, distance: np.ndarray, kappa: float = 300.0) -> dict:
    """resid（実テン − 想定テン）を gate_rel で回帰した傾き（外ほど遅い=正）を、(面,距離帯) と (場,面,距離) の 2 段で縮約して返す。"""
    d = pd.DataFrame({"r": resid, "g": g, "v": venue, "s": surface, "b": dist_band(distance), "d": distance}).dropna(subset=["r", "g"])

    def slope(x: pd.DataFrame) -> tuple[float, int]:
        gg = x["g"].to_numpy() - x["g"].mean()
        den = float((gg ** 2).sum())
        return (float((gg * (x["r"].to_numpy() - x["r"].mean())).sum() / den) if den > 0 else 0.0), len(x)

    glob, _ = slope(d)
    mid, cell = {}, {}
    for key, x in d.groupby(["s", "b"]):
        b, n = slope(x)
        mid[key] = (n * b + kappa * glob) / (n + kappa)
    for key, x in d.groupby(["v", "s", "d"]):
        b, n = slope(x)
        parent = mid.get((key[1], int(dist_band(np.array([key[2]]))[0])), glob)
        cell[key] = (n * b + kappa * parent) / (n + kappa)
    return {"global": glob, "mid": mid, "cell": cell}


def gate_adjust(coef: dict, g: np.ndarray, venue: np.ndarray, surface: np.ndarray, distance: np.ndarray) -> np.ndarray:
    out = np.empty(len(g))
    bands = dist_band(distance)
    for i in range(len(g)):
        k = (venue[i], surface[i], distance[i])
        b = coef["cell"].get(k)
        if b is None:
            b = coef["mid"].get((surface[i], int(bands[i])), coef["global"])
        out[i] = b * g[i]
    return out


def order_in_race(race_id: np.ndarray, score: np.ndarray, horse_number: np.ndarray, fill: str = "median", tiebreak: bool = True, rng_seed: int = 0) -> np.ndarray:
    """レースごとに score の小さい順に 1..N の想定順位を付ける。同値は馬番の小さい順（tiebreak=False なら乱数）。
    score が NaN の馬は fill: 'median'=その馬群の中央値扱い / 'back'=最後尾 / 'front'=先頭側。"""
    df = pd.DataFrame({"r": race_id, "s": score, "h": horse_number})
    med = df.groupby("r")["s"].transform("median")
    if fill == "median":
        s = df["s"].fillna(med)
    elif fill == "back":
        s = df["s"].fillna(df.groupby("r")["s"].transform("max") + 1.0)
    else:
        s = df["s"].fillna(df.groupby("r")["s"].transform("min") - 1.0)
    s = s.fillna(0.0)
    tb = df["h"].to_numpy(dtype="float64") if tiebreak else np.random.default_rng(rng_seed).random(len(df))
    d2 = pd.DataFrame({"r": df["r"].to_numpy(), "s": s.to_numpy(), "t": tb, "i": np.arange(len(df))})
    d2 = d2.sort_values(["r", "s", "t"], kind="mergesort")
    d2["rank"] = d2.groupby("r").cumcount() + 1
    out = np.empty(len(df), dtype="int64")
    out[d2["i"].to_numpy()] = d2["rank"].to_numpy()
    return out


def style_class(pos_norm: np.ndarray) -> np.ndarray:
    """正規化位置（0=先頭〜1=最後尾）→ 0 逃げ / 1 先行 / 2 差し / 3 追込（position_forecast と同じ境界）。"""
    p = np.asarray(pos_norm, dtype="float64")
    return np.select([p <= FRONT_MAX, p <= STALKER_MAX, p <= MID_MAX], [0, 1, 2], 3)
