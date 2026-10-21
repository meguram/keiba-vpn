"""tp-56: ユーザーの予想手順（レースタイプ → 馬タイプ → 適合 × 実力優劣 → 印）の特徴量。すべて as-of（その走より前の結果だけ）。

1. レースタイプ: 同じ競馬場・面の直近のレース（同日の先のレースを含む）の結果から『どんな馬が来たか』を指数減衰で数値化
   R_fb（前に行った馬が来た度合い）、R_kb（上がりの速い馬が来た度合い）、R_chalk（堅さ）、R_inside（内枠の有利）、R_attr（消耗戦度）。
2. 馬タイプ: 過去走の指数減衰平均 front_t（初角位置）、kick_t（上がり偏差）、kick_best（直近5走の最速）、stam_t（消耗戦での強さ）。
   6 タイプ（脚質 3 × 上がり 2）+ 不明。
3. 適合度: 直近の同競馬場の結果で、その馬のタイプが馬券内に来た割合（階層縮約）= fit_rate / fit_lift / fit_rank / fit_best。
4. 実力の優劣: 過去の対戦履歴からの条件付き Elo（全体・面×距離帯）。レース内の相対値と平均対戦勝率。
5. 不利の補正: 備考の不利系フラグ（trouble_rem）のある走は、Elo の更新で着順を 3 つ引き上げて扱う（_adj 版）。
先読みの検査は tests/pipeline/test_tp56_yosou.py（日付を切って作り直しても、それ以前の値が完全一致）。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

LAM_RACE = 0.93          # レースタイプの指数減衰（直近 10 レースで半減）
HORSE_ALPHA = 0.30       # 馬タイプの指数平滑（直近 2〜3 走が重い）
PRIOR_K = 12.0           # 適合率の縮約の擬似件数
ELO_BASE = 1500.0
FIT_TYPES = 7            # 0..5 = 脚質(3) × 上がり(2)、6 = 不明
TROUBLE_SHIFT = 3        # 不利走の着順の引き上げ幅（Elo の更新）
RACE_SIG = ["fb", "kb", "chalk", "inside", "attr"]


def dist_band(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, float)
    return np.where(d <= 1400, 0, np.where(d <= 1800, 1, np.where(d <= 2200, 2, 3)))


def _first_pos(s) -> float:
    try:
        return float(str(s).split("-")[0])
    except (TypeError, ValueError):
        return np.nan


def _pace_parts(p) -> tuple[float, float]:
    try:
        d = json.loads(p) if isinstance(p, str) else p
        return float(d.get("first_half_3f")), float(d.get("second_half_3f"))
    except (TypeError, ValueError, AttributeError):
        return np.nan, np.nan


def prepare_runs(df: pd.DataFrame) -> pd.DataFrame:
    """race_result_flat → 1 走 1 行。着順が無い走（取消・除外）は除く。"""
    d = df.copy()
    d["finish_position"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d = d[d["finish_position"].notna()].copy()
    d["field_size"] = pd.to_numeric(d["field_size"], errors="coerce")
    d["last_3f"] = pd.to_numeric(d["last_3f"], errors="coerce")
    d["last_3f"] = d["last_3f"].where((d["last_3f"] >= 30) & (d["last_3f"] <= 50))     # 計測外れ値（0・100 超など）は欠損
    d["popularity"] = pd.to_numeric(d["popularity"], errors="coerce")
    d["bracket_number"] = pd.to_numeric(d["bracket_number"], errors="coerce")
    d["distance"] = pd.to_numeric(d["distance"], errors="coerce")
    d["dnf"] = (d["finish_position"] <= 0).astype(float)          # 中止・失格などは -1。最下位扱い（同順位）にする
    d["finish_position"] = np.where(d["dnf"] > 0, d["field_size"], d["finish_position"])
    d["epn"] = d["passing_order"].map(_first_pos) / d["field_size"]
    parts = d["pace"].map(_pace_parts)
    d["t3f"] = [p[0] for p in parts]
    d["l3f_race"] = [p[1] for p in parts]
    d["band"] = dist_band(d["distance"].to_numpy())
    d = d.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    return d


def _prior_ewm(df: pd.DataFrame, key: str, col: str, alpha: float) -> np.ndarray:
    """馬ごとに『その走より前の走』だけの指数平滑（adjust=True の重み付き平均）。"""
    s = df.groupby(key, sort=False)[col].transform(lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1, ignore_na=True).mean())
    return s.to_numpy(float)


def race_level(runs: pd.DataFrame) -> pd.DataFrame:
    """レースごとの『どんな馬が来たか』の指標（そのレースの結果）。ペース差は同条件の過去分で標準化（as-of）。"""
    d = runs
    g = d.groupby("race_id", sort=False)
    d["l3f_dev"] = g["last_3f"].transform("median") - d["last_3f"]
    top3 = d["finish_position"] <= 3
    d["_epn_top"] = d["epn"].where(top3)
    d["_kb_top"] = d["l3f_dev"].where(top3)
    d["_br"] = (d["bracket_number"] - 1) / 7.0
    d["_br_top"] = d["_br"].where(top3)
    d["_pop_top"] = (1.0 - (d["popularity"] - 1.0) / (d["field_size"] - 1.0).clip(lower=1)).where(top3)
    g = d.groupby("race_id", sort=False)
    r = pd.DataFrame({
        "date": g["date"].first(), "venue": g["venue"].first(), "surface": g["surface"].first(), "band": g["band"].first(),
        "field_size": g["field_size"].first(), "t3f": g["t3f"].first(), "l3f_race": g["l3f_race"].first(),
        "epn_all": g["epn"].mean(), "epn_top": g["_epn_top"].mean(), "kb": g["_kb_top"].mean(),
        "br_all": g["_br"].mean(), "br_top": g["_br_top"].mean(), "chalk": g["_pop_top"].mean(),
    })
    r["fb"] = r["epn_all"] - r["epn_top"]
    r["inside"] = r["br_all"] - r["br_top"]
    r["pace_diff"] = r["l3f_race"] - r["t3f"]
    r = r.sort_values(["date"], kind="stable")
    r = r.reset_index().sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    key = r["venue"].astype(str) + "|" + r["surface"].astype(str) + "|" + r["band"].astype(str)
    pdv = r["pace_diff"].astype(float)
    v0 = pdv.fillna(0.0)
    cnt = pdv.notna().astype(float).groupby(key, sort=False).cumsum().to_numpy() - pdv.notna().astype(float).to_numpy()
    s1 = v0.groupby(key, sort=False).cumsum().to_numpy() - v0.to_numpy()
    s2 = (v0 ** 2).groupby(key, sort=False).cumsum().to_numpy() - (v0 ** 2).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = s2 / cnt - mean ** 2
        z = (pdv.to_numpy() - mean) / np.sqrt(np.clip(var, 1e-6, None))
    z[cnt < 20] = np.nan
    r["attr"] = z
    return r


def race_stream_ema(races: pd.DataFrame, cols: list[str], lam: float = LAM_RACE) -> pd.DataFrame:
    """同じ競馬場・面のレースの流れで、『その前の』レースの指数減衰平均（同日の先のレースを含む）。races は (date, race_id) 昇順。"""
    out = {f"R_{c}": np.full(len(races), np.nan) for c in cols}
    out["R_n"] = np.zeros(len(races))
    key = (races["venue"].astype(str) + "|" + races["surface"].astype(str)).to_numpy()
    idx_by: dict = {}
    for i, k in enumerate(key):
        idx_by.setdefault(k, []).append(i)
    X = races[cols].to_numpy(float)
    for k, idx in idx_by.items():
        S = np.zeros(len(cols))
        W = np.zeros(len(cols))
        n = 0
        for i in idx:
            if n > 0:
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.where(W > 0, S / W, np.nan)
                for j, c in enumerate(cols):
                    out[f"R_{c}"][i] = v[j]
            out["R_n"][i] = n
            x = X[i]
            ok = ~np.isnan(x)
            S = lam * S + np.where(ok, x, 0.0)
            W = lam * W + ok.astype(float)
            n += 1
    return pd.DataFrame(out, index=races.index)


def horse_types(runs: pd.DataFrame) -> pd.DataFrame:
    """馬タイプ（as-of）。"""
    d = runs
    d["perf"] = np.where(d["field_size"] >= 4, 1.0 - (d["finish_position"] - 1.0) / (d["field_size"] - 1.0).clip(lower=1), np.nan)
    d["stam_c"] = (d["perf"] - 0.5) * d["attr_z"]
    out = pd.DataFrame(index=d.index)
    out["front_t"] = _prior_ewm(d, "horse_id", "epn", HORSE_ALPHA)
    out["kick_t"] = _prior_ewm(d, "horse_id", "l3f_dev", HORSE_ALPHA)
    out["stam_t"] = _prior_ewm(d, "horse_id", "stam_c", HORSE_ALPHA)
    out["perf_t"] = _prior_ewm(d, "horse_id", "perf", HORSE_ALPHA)
    out["kick_best"] = d.groupby("horse_id", sort=False)["l3f_dev"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max()).to_numpy(float)
    out["n_runs"] = d.groupby("horse_id", sort=False).cumcount().to_numpy(float)
    ft, kt = out["front_t"].to_numpy(), out["kick_t"].to_numpy()
    style = np.where(ft < 0.33, 0, np.where(ft > 0.62, 2, 1))
    kick = (kt > 0).astype(int)
    tp = style * 2 + kick
    out["tp"] = np.where(np.isnan(ft) | np.isnan(kt), 6, tp).astype(int)
    return out


def type_fit(runs: pd.DataFrame, races: pd.DataFrame, lam: float = LAM_RACE) -> pd.DataFrame:
    """適合度: 直近の同競馬場・面のレースで、各タイプが馬券内に来た割合（縮約）。"""
    rid_to_i = {r: i for i, r in enumerate(races["race_id"])}
    ri = runs["race_id"].map(rid_to_i).to_numpy()
    nrace = len(races)
    start = np.zeros((nrace, FIT_TYPES))
    top3 = np.zeros((nrace, FIT_TYPES))
    tp = runs["tp"].to_numpy()
    np.add.at(start, (ri, tp), 1.0)
    np.add.at(top3, (ri[(runs["finish_position"] <= 3).to_numpy()], tp[(runs["finish_position"] <= 3).to_numpy()]), 1.0)
    key = (races["venue"].astype(str) + "|" + races["surface"].astype(str)).to_numpy()
    idx_by: dict = {}
    for i, k in enumerate(key):
        idx_by.setdefault(k, []).append(i)
    fit_rate = np.full((nrace, FIT_TYPES), np.nan)
    base = np.full(nrace, np.nan)
    for k, idx in idx_by.items():
        S1 = np.zeros(FIT_TYPES)
        S0 = np.zeros(FIT_TYPES)
        n = 0
        for i in idx:
            if n > 0 and S0.sum() > 0:
                r0 = S1.sum() / S0.sum()
                fit_rate[i] = (S1 + PRIOR_K * r0) / (S0 + PRIOR_K)
                base[i] = r0
            S1 = lam * S1 + top3[i]
            S0 = lam * S0 + start[i]
            n += 1
    fr = fit_rate[ri, tp]
    out = pd.DataFrame(index=runs.index)
    out["fit_rate"] = fr
    out["fit_lift"] = fr / base[ri]
    # レース内の適合順位（出走馬のタイプのうち、適合率が高い順。1 = 最適合グループ）
    tmp = pd.DataFrame({"race_id": runs["race_id"].to_numpy(), "tp": tp, "fr": fr})
    ranks = tmp.groupby("race_id", sort=False)["fr"].rank(method="dense", ascending=False)
    out["fit_rank"] = ranks.to_numpy(float)
    out["fit_best"] = (out["fit_rank"] == 1).astype(float).where(~np.isnan(fr))
    return out


def elo_pass(runs: pd.DataFrame, trouble: np.ndarray | None, k_new: float = 32.0, k_old: float = 18.0) -> pd.DataFrame:
    """条件付き Elo（全体と 面×距離帯）。pre-race の値を返す。trouble がある走は着順を TROUBLE_SHIFT 引き上げて更新に使う。"""
    n = len(runs)
    pre_g = np.full(n, ELO_BASE)
    pre_c = np.full(n, ELO_BASE)
    rid = runs["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], n]
    hid = runs["horse_id"].to_numpy()
    fin = runs["finish_position"].to_numpy(float)
    cond = (runs["surface"].astype(str) + runs["band"].astype(str)).to_numpy()
    rg: dict = {}
    rc: dict = {}
    cnt: dict = {}
    for a, b in zip(starts, ends):
        ids = hid[a:b]
        m = b - a
        c = cond[a]
        g0 = np.array([rg.get(h, ELO_BASE) for h in ids])
        c0 = np.array([rc.get((c, h), rg.get(h, ELO_BASE)) for h in ids])
        pre_g[a:b] = g0
        pre_c[a:b] = c0
        if m < 4:
            continue
        rk = fin[a:b].copy()
        if trouble is not None:
            rk = np.maximum(1.0, rk - TROUBLE_SHIFT * trouble[a:b])
        S = (rk[:, None] < rk[None, :]).astype(float) + 0.5 * (rk[:, None] == rk[None, :]).astype(float)
        np.fill_diagonal(S, 0.0)
        kk = np.array([k_new if cnt.get(h, 0) < 5 else k_old for h in ids]) / (m - 1)
        for arr, store, keyf in ((g0, rg, lambda h: h), (c0, rc, lambda h: (c, h))):
            E = 1.0 / (1.0 + 10 ** ((arr[None, :] - arr[:, None]) / 400.0))
            np.fill_diagonal(E, 0.0)
            delta = kk * (S - E).sum(axis=1)
            for h, v, dl in zip(ids, arr, delta):
                store[keyf(h)] = v + dl
        for h in ids:
            cnt[h] = cnt.get(h, 0) + 1
    out = pd.DataFrame({"g": pre_g, "c": pre_c}, index=runs.index)
    return out


def elo_relative(runs: pd.DataFrame, elo: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """レース内の相対値（他馬の平均との差）と、他馬に対する平均の期待勝率（メンバー比較）。"""
    out = pd.DataFrame(index=runs.index)
    rid = runs["race_id"].to_numpy()
    for col in ("g", "c"):
        v = elo[col].to_numpy()
        df = pd.DataFrame({"r": rid, "v": v})
        grp = df.groupby("r", sort=False)["v"]
        s = grp.transform("sum").to_numpy()
        m = grp.transform("size").to_numpy()
        out[f"elo_rel_{col}{suffix}"] = v - (s - v) / np.clip(m - 1, 1, None)
    # 平均対戦勝率（条件付き版）
    v = elo["c"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(rid)]
    pw = np.full(len(rid), np.nan)
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        x = v[a:b]
        E = 1.0 / (1.0 + 10 ** ((x[None, :] - x[:, None]) / 400.0))
        np.fill_diagonal(E, 0.0)
        pw[a:b] = E.sum(axis=1) / (b - a - 1)
    out[f"elo_pw_c{suffix}"] = pw
    return out


def build_features(runs: pd.DataFrame, flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """runs: prepare_runs の出力（全期間）。flags: horse_id, race_id, trouble_rem の表。戻り値は 1 走 1 行の特徴量。"""
    d = runs.copy()
    races = race_level(d)
    d = d.merge(races[["race_id", "attr"]].rename(columns={"attr": "attr_z"}), on="race_id", how="left")
    d = d.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    ema = race_stream_ema(races, RACE_SIG)
    races = pd.concat([races, ema], axis=1)
    ht = horse_types(d)
    d = pd.concat([d, ht], axis=1)
    fit = type_fit(d, races)
    d = pd.concat([d, fit], axis=1)
    d = d.merge(races[["race_id"] + [f"R_{c}" for c in RACE_SIG] + ["R_n"]], on="race_id", how="left")
    d = d.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    # 連続の適合（レースタイプ × 馬タイプ）
    d["m_front"] = d["R_fb"] * (0.5 - d["front_t"])
    d["m_kick"] = d["R_kb"] * d["kick_t"]
    d["m_att"] = d["R_attr"] * d["stam_t"]
    # 不利（備考）
    tr = np.zeros(len(d))
    if flags is not None and len(flags):
        f = flags[["race_id", "horse_id", "trouble_rem", "has_remarks"]].drop_duplicates(["race_id", "horse_id"])
        d = d.merge(f, on=["race_id", "horse_id"], how="left")
        d = d.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
        tr = d["trouble_rem"].fillna(0.0).to_numpy(float)
        d["trouble_run"] = tr
    else:
        d["trouble_run"] = 0.0
    d["trouble_last"] = d.groupby("horse_id", sort=False)["trouble_run"].shift(1).to_numpy(float)
    d["trouble_n3"] = d.groupby("horse_id", sort=False)["trouble_run"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).to_numpy(float)
    e0 = elo_pass(d, None)
    e1 = elo_pass(d, tr)
    rel0 = elo_relative(d, e0, "")
    rel1 = elo_relative(d, e1, "_adj")
    d["elo_g"], d["elo_c"] = e0["g"].to_numpy(), e0["c"].to_numpy()
    d["elo_g_adj"], d["elo_c_adj"] = e1["g"].to_numpy(), e1["c"].to_numpy()
    for c in list(rel0.columns) + list(rel1.columns):
        d[c] = (rel0[c] if c in rel0 else rel1[c]).to_numpy()
    # レース内の相対（馬タイプ）
    for c in ("front_t", "kick_t", "stam_t", "fit_rate"):
        g = d.groupby("race_id", sort=False)[c]
        d[f"{c}_rel"] = (d[c] - g.transform("mean")).to_numpy()
    d["fit_pct"] = d.groupby("race_id", sort=False)["fit_rate"].rank(pct=True).to_numpy()
    keep = ["race_id", "horse_id", "date", "venue", "surface", "distance", "field_size", "finish_position", "popularity", "dnf",
            "R_fb", "R_kb", "R_chalk", "R_inside", "R_attr", "R_n",
            "front_t", "kick_t", "stam_t", "perf_t", "kick_best", "n_runs", "tp",
            "fit_rate", "fit_lift", "fit_rank", "fit_best", "fit_pct", "m_front", "m_kick", "m_att",
            "front_t_rel", "kick_t_rel", "stam_t_rel", "fit_rate_rel",
            "elo_g", "elo_c", "elo_g_adj", "elo_c_adj", "elo_rel_g", "elo_rel_c", "elo_pw_c", "elo_rel_g_adj", "elo_rel_c_adj", "elo_pw_c_adj",
            "trouble_run", "trouble_last", "trouble_n3"]
    return d[keep]


GROUPS = {
    "RT": ["R_fb", "R_kb", "R_chalk", "R_inside", "R_attr"],
    "HT": ["front_t", "kick_t", "stam_t", "kick_best", "front_t_rel", "kick_t_rel", "stam_t_rel"],
    "FIT": ["fit_rate", "fit_lift", "fit_rank", "fit_best", "fit_pct", "fit_rate_rel", "m_front", "m_kick", "m_att"],
    "ELO": ["elo_rel_g", "elo_rel_c", "elo_pw_c"],
    "ELO_ADJ": ["elo_rel_g_adj", "elo_rel_c_adj", "elo_pw_c_adj"],
    "TRB": ["trouble_last", "trouble_n3"],
}
GROUPS["ALL"] = sorted({c for k, v in GROUPS.items() if k != "ELO_ADJ" for c in v})
GROUPS["ALL_ADJ"] = sorted({c for k, v in GROUPS.items() if k != "ELO" for c in v})
