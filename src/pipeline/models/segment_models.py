"""レースのクラス・コース・競馬場ごとの専用モデルが、全体モデルより良いかを測る部品。

- ``grade_bucket`` / ``course_key`` / ``venue_surface``: レース単位の区分（同じレースの馬は必ず同じ区分）。
- ``oof_segmented``: 区分ごとに LightGBM を学習して OOF を作る。
  separate = その区分の学習行だけで学習 / weighted = 全学習行を使い、区分外の行の重みを下げて学習（データを捨てない専用化）。
  学習行が足りない区分は、全体モデルの予測で埋める（fallback）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.models import win_models as wm


def grade_bucket(race_class: pd.Series, surface: pd.Series) -> pd.Series:
    rc = race_class.fillna("").astype(str)
    out = pd.Series("その他", index=rc.index)
    out[rc.str.contains("オープン|ＯＰ|OP|Ｇ|G[123１２３]|重賞|リステッド|L$", regex=True)] = "オープン・重賞"
    out[rc.str.contains("３勝|3勝")] = "3勝"
    out[rc.str.contains("２勝|2勝")] = "2勝"
    out[rc.str.contains("１勝|1勝")] = "1勝"
    out[rc.str.contains("未勝利")] = "未勝利"
    out[rc.str.contains("新馬")] = "新馬"
    out[(rc == "") | surface.astype(str).str.contains("障")] = "障害"
    return out


def course_key(surface: pd.Series, distance: pd.Series) -> pd.Series:
    d = pd.cut(distance.astype(float), [0, 1300, 1700, 2100, 99999], labels=["〜1300", "1400〜1700", "1800〜2100", "2200〜"]).astype(str)
    s = surface.astype(str).map(lambda x: "障害" if "障" in x else ("芝" if "芝" in x else "ダ"))
    return np.where(s == "障害", "障害", s + "/" + d)


def venue_surface(venue_code: pd.Series, surface: pd.Series) -> pd.Series:
    s = surface.astype(str).map(lambda x: "障" if "障" in x else ("芝" if "芝" in x else "ダ"))
    return venue_code.astype(str).str.zfill(2) + "/" + s


def oof_segmented(X: pd.DataFrame, meta: pd.DataFrame, folds, seg: np.ndarray, lgb_params: dict, n_rounds: int, threads: int, mode: str = "separate",
                  min_rows: int = 3000, weight_other: float = 0.25, seed: int = 0, base_oof: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """戻り値: (OOF 予測 race_id・horse_id・year・p_win・p_top2・p_top3, 区分ごとの情報)。
    base_oof（全体モデルの OOF）を渡すと、学習行が足りない区分はその予測で埋める。"""
    race = meta["race_id"].to_numpy()
    fin = meta["finish_position"].to_numpy()
    year = meta["year"].to_numpy()
    seg = np.asarray(seg)
    assert (pd.Series(seg).groupby(race).nunique() == 1).all(), "区分はレース単位でなければなりません"
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int), "finish": fin}
    P = {k: np.full(len(meta), np.nan) for k in ("win", "top2", "top3")}
    info = {}
    for f in folds:
        tr, va = f.train_index, f.valid_index
        for s in np.unique(seg[va]):
            va_s = va[seg[va] == s]
            if mode == "separate":
                tr_s = tr[seg[tr] == s]
                w = None
            else:
                tr_s = tr
                w = np.where(seg[tr] == s, 1.0, weight_other)
            info[(int(year[va[0]]), str(s))] = int((seg[tr] == s).sum())
            if (seg[tr] == s).sum() < min_rows:
                continue
            o = wm.fit_predict("binary", X.iloc[tr_s], {k: v[tr_s] for k, v in Y.items()}, race[tr_s], X.iloc[va_s], race[va_s], threads, seed=seed, params=lgb_params,
                               n_rounds=n_rounds, weight=w)
            for k in P:
                P[k][va_s] = o[k]
    covered = np.zeros(len(meta), bool)
    for f in folds:
        covered[f.valid_index] = True
    out = meta.loc[covered, ["race_id", "horse_id", "year"]].copy()
    for k in P:
        out[f"p_{k}"] = P[k][covered]
    if base_oof is not None:
        b = out[["race_id", "horse_id"]].merge(base_oof[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]], on=["race_id", "horse_id"], how="left")
        for k in P:
            miss = out[f"p_{k}"].isna().to_numpy()
            out.loc[miss, f"p_{k}"] = b.loc[miss, f"p_{k}"].to_numpy()
    out = out.dropna().reset_index(drop=True)
    for k, tot in (("win", 1.0), ("top2", 2.0), ("top3", 3.0)):
        out[f"p_{k}"] = wm.normalize_to_total(out["race_id"].to_numpy(), out[f"p_{k}"].to_numpy(), tot)
    return out, info
