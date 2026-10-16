"""コース適性の類似度: 確定したコース形状マスタ（``data/meta/course_master``）と、レース記録から測った統計プロファイルから、
コース間の類似度（カーネル）を作り、馬の過去の実績を『似たコース』から借りる特徴量を計算する。

- ``course_vectors``: コース（競馬場・面・距離）ごとの形状ベクトル。内外回りの行は混合（既定は等しい重み。新潟芝2000 だけ記録から約 52:48）。
- ``similarity_kernel``: 標準化した距離（形状 + 任意で統計プロファイル）から、面（芝 / ダート）が同じコース間のカーネル重み。
- ``horse_similarity_features``: 各馬・各レースについて、**そのレースより前の**実績だけで
    sim_aff = カーネル重みで似たコースの実績を重みづけ平均（親 = 馬の全体の平均へ縮約）、sim_n = 実質的なサンプル数、
    shape_pred / shape_se = 形状の軸（直線の長さ・坂・コーナー・スタート位置 …）への馬の適性（リッジ回帰の予測と不確かさ）。
  実績の指標は『着順の相対位置』res = 0.5 − (着順 − 1)/(頭数 − 1)（上位ほど正）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

GEOM_COLS = ["final_straight_m", "lap_length_m", "elevation_range_m", "goal_slope_rise", "first_curve_m", "n_corner_records", "distance"]
VARIANT_WEIGHTS = {("04", "芝", 2000): {"内": 0.48, "外": 0.52}}


def load_master(path: str | Path = "data/meta/course_master/jra_courses.json") -> pd.DataFrame:
    m = pd.DataFrame(json.loads(Path(path).read_text(encoding="utf-8")))
    m = m[m.get("in_use", True).fillna(True).astype(bool)] if "in_use" in m else m
    return m


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def course_vectors(master: pd.DataFrame) -> pd.DataFrame:
    """index = (venue_code, surface, distance)、列 = 形状の数値（変種の重みつき平均）と二値/カテゴリの指標。"""
    m = master.copy()
    m["goal_slope_rise"] = _num(m.get("goal_slope_rise")).where(m.get("goal_slope_exists", False).fillna(False).astype(bool), 0.0).fillna(0.0)
    m["goal_slope_exists_f"] = m.get("goal_slope_exists", False).fillna(False).astype(float)
    m["n_corner_records"] = _num(m.get("n_corner_records")).fillna(_num(m.get("n_corners_geom_A")))
    for c in ("final_straight_m", "lap_length_m", "elevation_range_m", "first_curve_m"):
        m[c] = _num(m[c])
    m["w"] = [VARIANT_WEIGHTS.get((v, s, d), {}).get(var, 1.0) for v, s, d, var in zip(m.venue_code, m.surface, m.distance, m.variant)]
    keys = ["venue_code", "surface", "distance"]
    num = ["final_straight_m", "lap_length_m", "elevation_range_m", "goal_slope_rise", "goal_slope_exists_f", "first_curve_m", "n_corner_records"]
    rows = []
    for k, g in m.groupby(keys):
        w = g["w"].to_numpy(dtype=float)
        r = dict(zip(keys, k))
        for c in num:
            v = g[c].to_numpy(dtype=float)
            ok = ~np.isnan(v)
            r[c] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        r["is_turf"] = float(k[1] == "芝")
        r["is_straight_course"] = float(bool((g.get("straight_course", False) == True).any()) if "straight_course" in g else False)  # noqa: E712
        r["distance_m"] = float(k[2])
        rows.append(r)
    v = pd.DataFrame(rows).set_index(keys)
    # 欠損は同じ面・距離帯の中央値で補う
    v["_band"] = pd.cut(v["distance_m"], [0, 1300, 1700, 2100, 9999]).astype(str)
    for c in num:
        v[c] = v[c].fillna(v.groupby(["is_turf", "_band"])[c].transform("median")).fillna(v[c].median())
    return v.drop(columns="_band")


SHAPE_AXES = ["final_straight_m", "goal_slope_rise", "n_corner_records", "first_curve_m", "elevation_range_m", "lap_length_m", "distance_m"]


def standardize(v: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = v[cols].to_numpy(dtype=float)
    return (x - x.mean(0)) / np.where(x.std(0) < 1e-9, 1.0, x.std(0))


def similarity_kernel(v: pd.DataFrame, cols: list[str] | None = None, weights: dict | None = None, tau: float = 1.5, stats: pd.DataFrame | None = None, stat_weight: float = 0.5) -> np.ndarray:
    """(コース数 × コース数) のカーネル。面（芝/ダート）が違えば 0。距離は標準化した形状の距離（cols に distance_m を含める）。stats を渡すと、統計プロファイルの距離も足す（stat_weight 倍）。"""
    cols = cols or ["final_straight_m", "goal_slope_rise", "n_corner_records", "first_curve_m", "elevation_range_m", "lap_length_m", "distance_m"]
    z = standardize(v, cols)
    w = np.array([(weights or {}).get(c, 1.0) for c in cols])
    d2 = (((z[:, None, :] - z[None, :, :]) ** 2) * w).sum(-1) / w.sum()
    if stats is not None:
        s = standardize(stats.reindex(v.index).fillna(0.0), list(stats.columns))
        d2 = d2 + stat_weight * ((s[:, None, :] - s[None, :, :]) ** 2).mean(-1)
    K = np.exp(-d2 / (tau ** 2))
    turf = v["is_turf"].to_numpy()
    K = K * (turf[:, None] == turf[None, :])
    return K


def horse_similarity_features(df: pd.DataFrame, v: pd.DataFrame, K: np.ndarray, kappa: float = 3.0, ridge: float = 8.0) -> pd.DataFrame:
    """df: race_id・horse_id・date・venue_code・surface・distance・finish_position・field_size（日付順に並べ替え済みである必要はない）。
    戻り値: df と同じ index の sim_aff・sim_n・shape_pred・shape_se。どれも、その行のレースより前の実績だけを使う（同日のレースは使わない）。"""
    idx_map = {k: i for i, k in enumerate(v.index)}
    n_c = len(v)
    ckey = list(zip(df["venue_code"].astype(str), df["surface"].astype(str), df["distance"].astype(int)))
    cid = np.array([idx_map.get(k, -1) for k in ckey])
    res = (0.5 - (df["finish_position"].to_numpy(dtype=float) - 1.0) / np.maximum(df["field_size"].to_numpy(dtype=float) - 1.0, 1.0))
    X = standardize(v, SHAPE_AXES)
    d = X.shape[1] + 1
    order = np.lexsort((df["race_id"].to_numpy(), pd.to_datetime(df["date"]).to_numpy(), df["horse_id"].astype(str).to_numpy()))
    out = np.full((len(df), 4), np.nan)
    hid = df["horse_id"].astype(str).to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy()
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and hid[order[j]] == hid[order[i]]:
            j += 1
        S, N = np.zeros(n_c), np.zeros(n_c)
        A = np.eye(d) * ridge
        b = np.zeros(d)
        tot_s, tot_n = 0.0, 0.0
        pending = []                                   # 同じ日の結果は、その日のレースの特徴量には入れない
        cur_date = None
        for t in range(i, j):
            r = order[t]
            if cur_date is not None and dates[r] != cur_date:
                for (c_, y_) in pending:
                    S[c_] += y_; N[c_] += 1; tot_s += y_; tot_n += 1
                    z = np.concatenate([[1.0], X[c_]]); A += np.outer(z, z); b += z * y_
                pending = []
            cur_date = dates[r]
            c = cid[r]
            if c >= 0:
                parent = tot_s / (tot_n + 1.0)
                w = K[c]
                ws, wn = float(w @ S), float(w @ N)
                out[r, 0] = (ws + kappa * parent) / (wn + kappa)
                out[r, 1] = wn
                z = np.concatenate([[1.0], X[c]])
                beta = np.linalg.solve(A, b)
                out[r, 2] = float(z @ beta)
                out[r, 3] = float(np.sqrt(z @ np.linalg.solve(A, z)))
            if c >= 0 and not np.isnan(res[r]):
                pending.append((c, res[r]))
        i = j
    return pd.DataFrame(out, columns=["sim_aff", "sim_n", "shape_pred", "shape_se"], index=df.index)
