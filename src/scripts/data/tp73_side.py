"""tp-73 母系／父系の高頻度の種牡馬の特性・クラスタ・登場回数の用量反応・血統クラスタ対の同時好走（結果・オッズは分析にだけ使い、特徴量にしない）。

python -m src.scripts.data.tp73_side anc        # 側（母系 mat／父系 pat）ごとの祖先の表（10 世代以内の登場回数・世代別）
python -m src.scripts.data.tp73_side prep       # 分析表（tp-59 master + レース質 + 条件軸 + 残差）
python -m src.scripts.data.tp73_side thin       # 第 1 段 A: 『血が薄くても効く』（深い世代の重み比較）
python -m src.scripts.data.tp73_side profile    # 第 1 段 B: 特性プロファイルとクラスタ（as-of）、タイプ割り当ての検証
python -m src.scripts.data.tp73_side dose       # 第 2 段: 登場回数 n の用量反応
python -m src.scripts.data.tp73_side pair       # 血統クラスタ対の同時好走（lift・市場統制・回収率）

定義は prereg.json（実行前に固定。型の数を含む）。出力 <TP73_OUT>（既定 /home/jovyan/work/tp73_out）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

OUTD = Path(os.environ.get("TP73_OUT") or "/home/jovyan/work/tp73_out")
TP60 = Path(os.environ.get("TP60_OUT") or "/home/jovyan/work/tp60_out")
TP59 = Path(os.environ.get("TP59_OUT") or "/home/jovyan/work/tp59_out")
ROOT = Path(os.environ.get("KEIBA_ROOT") or "/home/jovyan/work/keiba-vpn")
RQ = ROOT / "data/local/modeling/shared_cache/race_quality/race_quality.parquet"
PAYOFF = ROOT / "data/local/modeling/betting/payoff.parquet"
SEED = 20261022
SET = os.environ.get("TP73_SET", "1")                       # 1 = 原定義（2〜80%）、2 = 修正（5〜60%。飽和した創始者を除く）
SFX = "" if SET == "1" else f"_s{SET}"
RANGE = {"1": (0.02, 0.80), "2": (0.05, 0.60)}[SET]
SIDES = ("mat", "pat")

PRE = {
    "freq_share_range": [0.02, 0.80],     # 母集団（35,794 頭）の何割がその祖先を持つか。飽和した創始者（>80%）と稀な祖先（<2%）は対象外
    "top_n_primary": 50, "top_n_sens": [20, 100],
    "count_defs": {"a": "その側の 10 世代以内に祖先として含む馬の頭数（母集団）", "b": "うち 2020〜2025 に出走した馬の頭数", "c": "その側だけで数えた頭数（a と同じ。側ごとに別々）"},
    "sample_floor_entries": 3000,         # その祖先を持つ馬の出走数（別の閾値）
    "bands": {"near": [1, 5], "deep": [6, 10]},
    "weights": {"decay": "Σ 2^-世代", "presence": "あり=1", "weak": "sqrt(Σ 2^-世代)"},
    "cond_families": {"venue": 10, "dist_band": 4, "going": 4, "surface": 2, "pace_cls(oracle)": 3},
    "ridge_c_rel": {"primary": 0.2, "sens": 0.02, "note": "λ = c × 行数（標準化した列に対して）。基底の列（定数・条件・近い世代の血量・もう一方の側）は 1e-3"},
    "responses": {"primary": "R_pro（着順の正規スコアの馬内残差）", "others": ["R_tim（タイム偏差）", "R_meg（めぐ指数 v3a の相対）"], "traditional": ["y_old = top3 - 3/頭数（母集団の単純な成績率）", "y_v4 = top3 - p_top3（V4 の残差、2022 年〜）"], "baseline": "馬ごとの as-of 平均を疑似走数 3 で 0 へ縮約。最低 3 走", "min_prev": 3},
    "y_primary": "top3 - p_top3 (V4 OOF)", "y_secondary": "win - p_win",
    "folds": "LOYO 2022/2023/2024 → 2025 を 1 回 → 2026 は記述のみ",
    "K_clusters": 6, "profile_shrink_k": 200,
    "dose_bins": {"n_near": ["0", "1", "2+"], "n_deep": ["0", "1", "2", "3+"], "above_mean": "n_total >= その祖先を持つ馬の平均 n"},
    "pair_assign": {"A_embK16": "tp-60 構成埋め込みの K=16", "B_embK40": "tp-60 構成埋め込みの K=40", "C_mat": "母系の高頻度の種牡馬のタイプ（K=6 + none）", "D_pat": "父系の同（対照）"},
    "pair_strata": ["all", "turf", "dirt", "sprint", "mile_mid", "long"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ 祖先の表
def ancestor_entries(sire: np.ndarray, dam: np.ndarray, roots: np.ndarray, side: str, max_gen: int = 10):
    """各馬の側（mat: 母から、pat: 父から）の木を世代ごとに展開し、男の祖先（誰かの父の位置）の (馬, 節点, 世代) を返す。
    root が世代 1。pat は root（父）自身が男の祖先（世代 1）。重複（クロス）は別の登場として数える。"""
    h = np.arange(len(roots), dtype=np.int64)
    ok = roots >= 0
    cur_h, cur_n = h[ok], roots[ok]
    H, N, G = [], [], []
    if side == "pat":
        H.append(cur_h); N.append(cur_n); G.append(np.full(len(cur_n), 1, dtype=np.int8))
    for g in range(1, max_gen):
        s, d = sire[cur_n], dam[cur_n]
        ms, md = s >= 0, d >= 0
        H.append(cur_h[ms]); N.append(s[ms]); G.append(np.full(int(ms.sum()), g + 1, dtype=np.int8))
        cur_h = np.concatenate([cur_h[ms], cur_h[md]])
        cur_n = np.concatenate([s[ms], d[md]])
        if len(cur_n) == 0:
            break
    return np.concatenate(H), np.concatenate(N), np.concatenate(G)


def cmd_anc(a) -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    G = pd.read_parquet(TP60 / "embed" / "graph.parquet")
    code = {n: i for i, n in enumerate(G["node_id"])}
    sire = G["sire_id"].map(lambda x: code.get(x, -1) if x else -1).to_numpy(dtype=np.int64)
    dam = G["dam_id"].map(lambda x: code.get(x, -1) if x else -1).to_numpy(dtype=np.int64)
    horses = pd.read_parquet(TP60 / "embed" / "horses.parquet")["horse_id"].astype(str).to_numpy()
    hn = np.array([code.get(h, -1) for h in horses])
    log(f"horses {len(horses):,} / in graph {(hn >= 0).sum():,}")
    sire_h, dam_h = np.where(hn >= 0, sire[np.maximum(hn, 0)], -1), np.where(hn >= 0, dam[np.maximum(hn, 0)], -1)
    names = json.load(open(TP60 / "embed" / "node_names.json", encoding="utf-8"))
    m = pd.read_parquet(TP59 / "master.parquet", columns=["horse_id", "year"])
    raced = set(m["horse_id"].astype(str))
    ran_mask = np.array([h in raced for h in horses])
    birth = pd.read_parquet(TP60 / "ped" / "horse_static.parquet", columns=["horse_id", "birth_year"]).drop_duplicates("horse_id").set_index("horse_id")["birth_year"]
    by = birth.reindex(horses).to_numpy()
    nh = len(horses)
    for side in SIDES:
        roots = sire_h if side == "pat" else dam_h
        H, N, Gn = ancestor_entries(sire, dam, roots, side)
        log(f"{side}: 登場 {len(H):,} 件")
        key = H.astype(np.int64) * len(G) + N
        uk, inv = np.unique(key, return_inverse=True)
        hh, nn = uk // len(G), uk % len(G)
        n_tot = np.bincount(inv)
        near = (Gn <= 5)
        n_near = np.bincount(inv, weights=near.astype(np.float64), minlength=len(uk))
        sh = np.bincount(inv, weights=np.power(2.0, -Gn.astype(np.float64)), minlength=len(uk))
        sh_deep = np.bincount(inv, weights=np.power(2.0, -Gn.astype(np.float64)) * (~near), minlength=len(uk))
        min_g = np.full(len(uk), 99.0); np.minimum.at(min_g, inv, Gn.astype(np.float64))
        T = pd.DataFrame({"h": hh.astype(np.int32), "anc": nn.astype(np.int32), "n_total": n_tot.astype(np.int16), "n_near": n_near.astype(np.int16),
                          "share": sh.astype(np.float32), "share_deep": sh_deep.astype(np.float32), "min_gen": min_g.astype(np.int8)})
        T["n_deep"] = (T["n_total"] - T["n_near"]).astype(np.int16)
        # 残存数（母集団）
        pop = T.groupby("anc").agg(n_pop=("h", "size"), occ=("n_total", "sum"), mean_n=("n_total", "mean"))
        T_ran = T[ran_mask[T["h"].to_numpy()]]
        pop["n_ran"] = T_ran.groupby("anc").size().reindex(pop.index).fillna(0).astype(int)
        # 生年で切った as-of のカウント（その年より前に生まれた馬だけ）
        for yy in (2019, 2021, 2023):
            Ty = T[(by[T["h"].to_numpy()] <= yy)]
            pop[f"n_pop_by{yy}"] = Ty.groupby("anc").size().reindex(pop.index).fillna(0).astype(int)
        pop["share_pop"] = pop["n_pop"] / nh
        pop["name"] = [names.get(G["node_id"].iloc[i], G["node_id"].iloc[i]) for i in pop.index]
        pop["node_id"] = [G["node_id"].iloc[i] for i in pop.index]
        T.to_parquet(OUTD / f"anc_{side}.parquet", index=False)
        pop.reset_index().to_parquet(OUTD / f"pop_{side}.parquet", index=False)
        log(f"{side}: 祖先 {len(pop):,} 頭、母集団の割合 2〜80% は {((pop.share_pop >= .02) & (pop.share_pop <= .8)).sum()} 頭")
    pd.DataFrame({"horse_id": horses, "h": np.arange(nh), "birth_year": by, "raced": ran_mask}).to_parquet(OUTD / "horses_idx.parquet", index=False)
    json.dump(PRE, open(OUTD / "prereg.json", "w"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 分析表
TP72 = Path(os.environ.get("TP72_OUT") or "/home/jovyan/work/tp72_out")
MEGU_ROOT = ROOT / "data/local/modeling/megu_recompute_20261019e"
FAMS = ("venue", "dist_band", "going", "surface", "pace")
LEVELS = {
    "venue": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    "dist_band": ["d_sprint", "d_mile", "d_mid", "d_long"],
    "going": ["良", "稍重", "重", "不良"],
    "surface": ["芝", "ダート"],
    "pace": ["slow", "mid", "high"],
}
K0_HORSE = 3          # 馬の基準の縮約（全体の平均 0 へ、疑似走数）
MIN_PREV = 3          # 最低の過去走数（これ未満の走は相対適性の対象外）
METRICS = ("pro", "tim", "meg")


def cmd_prep(a) -> None:
    from scipy.special import ndtri

    m = pd.read_parquet(TP59 / "master.parquet", columns=["race_id", "date", "venue_code", "surface", "track_condition", "field_size", "finish_position", "horse_id", "year", "dist_band",
                                                          "win", "top3", "popularity", "win_odds", "q_win", "p_win", "p_top2", "p_top3", "horse_number"])
    m = m[m["surface"].isin(["芝", "ダート"])].copy()
    m["horse_id"] = m["horse_id"].astype(str)
    hx = pd.read_parquet(OUTD / "horses_idx.parquet")[["horse_id", "h"]]
    m = m.merge(hx, on="horse_id", how="inner")
    fin = pd.to_numeric(m["finish_position"], errors="coerce")
    n = m["field_size"].astype(float).clip(lower=2)
    ok = fin.notna() & (fin > 0) & (fin <= n)
    m["pro"] = np.where(ok, ndtri(((n - fin + 0.5) / n).clip(1e-3, 1 - 1e-3)), np.nan)
    try:
        mi = pd.read_parquet(TP72 / "mining.parquet", columns=["race_id", "horse_id", "time_sec"])
        mi["horse_id"] = mi["horse_id"].astype(str)
        m = m.merge(mi.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    except Exception as e:  # noqa: BLE001
        log(f"time_sec 読めず: {e}")
        m["time_sec"] = np.nan
    t = m["time_sec"].where(ok & (m["time_sec"] > 0))
    g = t.groupby(m["race_id"])
    z = -(t - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    m["tim"] = z.clip(-3, 3)
    parts = []
    for y in range(2020, 2027):
        f = MEGU_ROOT / str(y) / "megu_index_flat.parquet"
        if not f.exists():
            continue
        cols = list(pd.read_parquet(f).columns)
        col = "megu_index_v3a_feat" if "megu_index_v3a_feat" in cols else "megu_index"
        want = [c for c in {"race_id", "horse_id", col, "megu_v3a_status", "megu_v3a_low_confidence"} if c in cols]
        d = pd.read_parquet(f, columns=want)
        if "megu_v3a_status" in d:
            d = d[d["megu_v3a_status"].notna()]
        if "megu_v3a_low_confidence" in d:
            d = d[~d["megu_v3a_low_confidence"].astype("boolean").fillna(False).astype(bool)]
        d = d[["race_id", "horse_id", col]].rename(columns={col: "meg"})
        parts.append(d)
    meg = pd.concat(parts).drop_duplicates(["race_id", "horse_id"])
    meg["horse_id"] = meg["horse_id"].astype(str)
    m = m.merge(meg, on=["race_id", "horse_id"], how="left")
    q = pd.read_parquet(RQ, columns=["race_id", "pace_cls"]).rename(columns={"pace_cls": "pace"})
    m = m.merge(q, on="race_id", how="left")
    m["going"] = m["track_condition"].astype(str)
    m["venue"] = m["venue_code"].astype(str)
    m = m.sort_values(["h", "date", "race_id"]).reset_index(drop=True)
    gh = m.groupby("h", sort=False)
    for k in METRICS:
        v = m[k]
        valid = v.notna().astype(float)
        cs = v.fillna(0).groupby(m["h"]).cumsum() - v.fillna(0)
        cc = valid.groupby(m["h"]).cumsum() - valid
        mu = float(v[m["year"] <= 2021].mean())
        m[f"b_{k}"] = (cs + K0_HORSE * mu) / (cc + K0_HORSE)
        m[f"nprev_{k}"] = cc
        m[f"R_{k}"] = (v - m[f"b_{k}"]).where(cc >= MIN_PREV)
    m["n_prev"] = m["nprev_pro"]
    m["top3"] = m["top3"].astype(float)
    m["y_old"] = m["top3"] - 3.0 / m["field_size"].astype(float)
    m["y_v4"] = m["top3"] - m["p_top3"]
    m["y_v4w"] = m["win"].astype(float) - m["p_win"]
    for f in FAMS:
        lv = {v: i for i, v in enumerate(LEVELS[f])}
        m[f"L_{f}"] = m[f].astype(str).map(lv).fillna(-1).astype(int)
    keep = ["race_id", "date", "year", "h", "horse_number", "field_size", "finish_position", "win", "top3", "popularity", "win_odds", "q_win", "p_win", "p_top3",
            "pro", "tim", "meg", "R_pro", "R_tim", "R_meg", "n_prev", "y_old", "y_v4", "y_v4w"] + [f"L_{f}" for f in FAMS] + ["surface", "distance"] if "distance" in m else None
    keep = [c for c in keep if c in m.columns] if keep else list(m.columns)
    m[keep].to_parquet(OUTD / "prep.parquet", index=False)
    log(f"prep: {len(m):,} 行、R_pro 有効 {m['R_pro'].notna().sum():,}、R_tim {m['R_tim'].notna().sum():,}、R_meg {m['R_meg'].notna().sum():,}、y_v4 {m['y_v4'].notna().sum():,}")
    for k in ("R_pro", "R_tim", "R_meg"):
        log(f"  {k}: 平均 {m[k].mean():+.4f}、sd {m[k].std():.3f}")


# ------------------------------------------------------------------ 共通: 祖先集合と疎行列
import scipy.sparse as sp  # noqa: E402

NL = sum(len(v) for v in LEVELS.values())
OFF = {}
_o = 0
for _f in FAMS:
    OFF[_f] = _o
    _o += len(LEVELS[_f])
LEVEL_NAMES = [f"{f}:{lv}" for f in FAMS for lv in LEVELS[f]]


def load_side(side: str):
    T = pd.read_parquet(OUTD / f"anc_{side}.parquet")
    pop = pd.read_parquet(OUTD / f"pop_{side}.parquet")
    return T, pop


def pick_set(pop: pd.DataFrame, n_top: int) -> np.ndarray:
    lo, hi = RANGE
    el = pop[(pop["share_pop"] >= lo) & (pop["share_pop"] <= hi)].sort_values("n_pop", ascending=False)
    return el["anc"].to_numpy()[:n_top]


def sparse_from(T: pd.DataFrame, S: np.ndarray, nh: int, vals: np.ndarray | None = None, mask: np.ndarray | None = None, col: str | None = None) -> sp.csr_matrix:
    cmap = pd.Series(np.arange(len(S)), index=S)
    sel = T["anc"].isin(S).to_numpy()
    if mask is not None:
        sel = sel & mask
    r = T["h"].to_numpy()[sel]
    c = cmap.reindex(T["anc"].to_numpy()[sel]).to_numpy()
    v = vals[sel] if vals is not None else T[col].to_numpy(float)[sel]
    return sp.csr_matrix((v, (r, c)), shape=(nh, len(S)))


def cond_onehot(P: pd.DataFrame) -> sp.csr_matrix:
    rows, cols = [], []
    for f in FAMS:
        L = P[f"L_{f}"].to_numpy()
        ok = L >= 0
        rows.append(np.where(ok)[0]); cols.append(OFF[f] + L[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(P), NL))


def deep_cond(P: pd.DataFrame, X: sp.csr_matrix) -> sp.csr_matrix:
    """行ごとの X（馬×祖先）と条件の水準の積: 列 = 祖先 × NL。"""
    Xr = X[P["h"].to_numpy()].tocoo()
    rr, ss, vv = Xr.row, Xr.col, Xr.data
    R, C, V = [], [], []
    for f in FAMS:
        L = P[f"L_{f}"].to_numpy()[rr]
        ok = L >= 0
        R.append(rr[ok]); C.append(ss[ok] * NL + OFF[f] + L[ok]); V.append(vv[ok])
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(len(P), X.shape[1] * NL))


def deep_variants(T: pd.DataFrame, S: np.ndarray, nh: int) -> dict[str, sp.csr_matrix]:
    sd = T["share_deep"].to_numpy(float)
    return {"decay": sparse_from(T, S, nh, vals=sd), "presence": sparse_from(T, S, nh, vals=(T["n_deep"].to_numpy() > 0).astype(float)),
            "weak": sparse_from(T, S, nh, vals=np.sqrt(sd))}


def near_share(T: pd.DataFrame, S: np.ndarray, nh: int) -> sp.csr_matrix:
    return sparse_from(T, S, nh, vals=(T["share"].to_numpy(float) - T["share_deep"].to_numpy(float)))


def gram_by_year(Z: sp.csr_matrix, y: np.ndarray, years: np.ndarray):
    out = {}
    for yr in np.unique(years):
        i = np.where(years == yr)[0]
        Zy = Z[i]
        out[int(yr)] = ((Zy.T @ Zy).toarray(), np.asarray(Zy.T @ y[i]).ravel(), len(i))
    return out


def ridge_fit(grams, train_years, idx, pen_rel, scale):
    G = sum(grams[y][0] for y in train_years)[np.ix_(idx, idx)]
    b = sum(grams[y][1] for y in train_years)[idx]
    N = sum(grams[y][2] for y in train_years)
    sc = scale[idx]
    A = G / np.outer(sc, sc) + np.diag(pen_rel[idx] * N)
    beta = np.linalg.solve(A, b / sc)
    return beta / sc


RESP_ROWS = {"R_pro": "R_pro", "R_tim": "R_tim", "R_meg": "R_meg", "y_old": "y_old", "y_v4": "y_v4"}
DISC_YEARS = {"R_pro": [2020, 2021, 2022, 2023, 2024], "R_tim": [2020, 2021, 2022, 2023, 2024], "R_meg": [2020, 2021, 2022, 2023, 2024],
              "y_old": [2020, 2021, 2022, 2023, 2024], "y_v4": [2022, 2023, 2024]}


def thin_eval(P: pd.DataFrame, T_side: pd.DataFrame, T_other: pd.DataFrame, S: np.ndarray, variant: str, resp: str, c_rel: float, nh: int,
              Xd_all: dict | None = None, Xn: sp.csr_matrix | None = None) -> dict:
    """深い世代（6〜10）の重みの変種 variant を、条件との積（cond）と条件なし（flat）で足したときの、保留した年での二乗誤差の改善（対 base）。"""
    d = P[P[resp].notna() & P["R_pro"].notna()].copy() if resp != "R_pro" else P[P["R_pro"].notna()].copy()
    d = d.reset_index(drop=True)
    y = d[resp].to_numpy(float)
    years = d["year"].to_numpy()
    Xd = (Xd_all or deep_variants(T_side, S, nh))[variant]
    Xn = Xn if Xn is not None else near_share(T_side, S, nh)
    Xo = sparse_from(T_other, S, nh, col="share") if T_other is not None else sp.csr_matrix((nh, len(S)))
    h = d["h"].to_numpy()
    C = cond_onehot(d)
    one = sp.csr_matrix(np.ones((len(d), 1)))
    base = sp.hstack([one, C, Xn[h], Xo[h]]).tocsr()
    B = base.shape[1]
    Zc = sp.hstack([base, deep_cond(d, Xd)]).tocsr()
    Zf = sp.hstack([base, Xd[h]]).tocsr()
    Xsum = sp.csr_matrix(np.asarray(Xd.sum(axis=1)))
    Za = sp.hstack([base, deep_cond(d, Xsum)]).tocsr()                 # 祖先を区別せず、深い世代の合計 × 条件（祖先固有かどうかの切り分け）
    Xthin = sparse_from(T_side, S, nh, vals=np.ones(len(T_side)), mask=((T_side["n_near"].to_numpy() == 0) & (T_side["n_deep"].to_numpy() > 0)))
    thin = np.asarray(Xthin[h].sum(axis=1)).ravel() > 0
    gc, gf, ga = gram_by_year(Zc, y, years), gram_by_year(Zf, y, years), gram_by_year(Za, y, years)
    alla = sum(ga[k][0] for k in ga)
    sc_a = np.sqrt(np.maximum(np.diag(alla) / sum(ga[k][2] for k in ga), 1e-12)); sc_a[sc_a < 1e-6] = 1.0
    pen_a = np.r_[np.full(B, 1e-3), np.full(Za.shape[1] - B, 1e-3)]
    allc = sum(gc[k][0] for k in gc); Ntot = sum(gc[k][2] for k in gc)
    sc_c = np.sqrt(np.maximum(np.diag(allc) / Ntot, 1e-12)); sc_c[sc_c < 1e-6] = 1.0
    allf = sum(gf[k][0] for k in gf)
    sc_f = np.sqrt(np.maximum(np.diag(allf) / Ntot, 1e-12)); sc_f[sc_f < 1e-6] = 1.0
    pen_c = np.r_[np.full(B, 1e-3), np.full(Zc.shape[1] - B, c_rel)]
    pen_f = np.r_[np.full(B, 1e-3), np.full(Zf.shape[1] - B, c_rel)]
    ib = np.arange(B)

    def run(train, test):
        i = np.where(np.isin(years, test))[0]
        b0 = ridge_fit(gc, train, ib, pen_c, sc_c)
        bc = ridge_fit(gc, train, np.arange(Zc.shape[1]), pen_c, sc_c)
        bf = ridge_fit(gf, train, np.arange(Zf.shape[1]), pen_f, sc_f)
        e0 = y[i] - base[i] @ b0
        ec = y[i] - Zc[i] @ bc
        ef = y[i] - Zf[i] @ bf
        ba = ridge_fit(ga, train, np.arange(Za.shape[1]), pen_a, sc_a)
        ea = y[i] - Za[i] @ ba
        return i, e0 ** 2, ec ** 2, ef ** 2, ea ** 2

    out = {"n": int(len(d)), "folds": {}}
    rows = []
    for yr in DISC_YEARS[resp]:
        tr = [k for k in DISC_YEARS[resp] if k != yr]
        i, s0, sc_, sf, sa = run(tr, [yr])
        out["folds"][str(yr)] = {"cond": float(1 - sc_.sum() / s0.sum()), "flat": float(1 - sf.sum() / s0.sum()), "agg": float(1 - sa.sum() / s0.sum()),
                                  "cond_thin": float(1 - sc_[thin[i]].sum() / s0[thin[i]].sum()) if thin[i].sum() > 100 else None, "n": int(len(i)), "n_thin": int(thin[i].sum())}
        rows.append(pd.DataFrame({"race_id": d["race_id"].to_numpy()[i], "dc": s0 - sc_, "df": s0 - sf, "da": s0 - sa, "thin": thin[i]}))
    pooled = pd.concat(rows)
    for k in ("dc", "df", "da"):
        g = pooled.groupby("race_id")[k].sum()
        out[f"t_{k}"] = float(g.mean() / (g.std() / np.sqrt(len(g)))) if len(g) > 2 else None
    for k, test_y in (("test2025", 2025), ("desc2026", 2026)):
        tr = DISC_YEARS[resp] + ([2025] if test_y == 2026 and resp != "y_v4" else []) + ([2025] if test_y == 2026 and resp == "y_v4" else [])
        i, s0, sc_, sf, sa = run(tr, [test_y])
        if len(i) == 0:
            continue
        out[k] = {"cond": float(1 - sc_.sum() / s0.sum()), "flat": float(1 - sf.sum() / s0.sum()), "agg": float(1 - sa.sum() / s0.sum()), "n": int(len(i)),
                  "cond_thin": float(1 - sc_[thin[i]].sum() / s0[thin[i]].sum()) if thin[i].sum() > 100 else None}
    out["thin_share"] = float(thin.mean())
    return out


def cmd_thin_strong(a) -> None:
    """事前に決めた c=0.2/0.02 が過学習（保留した年で base より悪い）だったので、より強い縮約 c=2, 20 を追加で見る（事後の追加。結果を見てから加えたと明記）。"""
    P = pd.read_parquet(OUTD / "prep.parquet")
    nh = int(pd.read_parquet(OUTD / "horses_idx.parquet")["h"].max()) + 1
    res: dict = {}
    for side in SIDES:
        T, pop = load_side(side)
        To, _ = load_side("pat" if side == "mat" else "mat")
        S50 = pick_set(pop, PRE["top_n_primary"])
        Xd50 = deep_variants(T, S50, nh)
        Xn50 = near_share(T, S50, nh)
        for resp in ("R_pro", "y_old", "y_v4"):
            for var in ("decay", "presence", "weak"):
                for c in (2.0, 20.0):
                    key = f"{side}|S50|{var}|{resp}|c{c}"
                    res[key] = thin_eval(P, T, To, S50, var, resp, c, nh, Xd50, Xn50)
                    r = res[key]
                    log(f"{key}: LOYO cond {np.mean([f['cond'] for f in r['folds'].values()]) * 1e4:+.2f}bp flat {np.mean([f['flat'] for f in r['folds'].values()]) * 1e4:+.2f}bp t_c {r['t_dc']:.2f}")
        json.dump(res, open(OUTD / f"thin_strong{SFX}.json", "w"), ensure_ascii=False, indent=1)
    log("thin_strong 完了")


def cmd_thin(a) -> None:
    P = pd.read_parquet(OUTD / "prep.parquet")
    nh = int(pd.read_parquet(OUTD / "horses_idx.parquet")["h"].max()) + 1
    res: dict = {}
    for side in SIDES:
        T, pop = load_side(side)
        To, _ = load_side("pat" if side == "mat" else "mat")
        S50 = pick_set(pop, PRE["top_n_primary"])
        log(f"{side}: S50 = {[pop.set_index('anc').loc[x, 'name'] for x in S50[:12]]} …")
        Xd50 = deep_variants(T, S50, nh)
        Xn50 = near_share(T, S50, nh)
        for resp in ("R_pro", "R_tim", "R_meg", "y_old", "y_v4"):
            for var in ("decay", "presence", "weak"):
                for c in (0.2, 0.02):
                    if c == 0.02 and not (resp == "R_pro"):
                        continue
                    key = f"{side}|S50|{var}|{resp}|c{c}"
                    res[key] = thin_eval(P, T, To, S50, var, resp, c, nh, Xd50, Xn50)
                    r = res[key]
                    log(f"{key}: LOYO cond {np.mean([f['cond'] for f in r['folds'].values()]) * 1e4:+.2f}bp flat {np.mean([f['flat'] for f in r['folds'].values()]) * 1e4:+.2f}bp t_c {r['t_dc']:.2f}")
        for n_top in PRE["top_n_sens"]:
            S = pick_set(pop, n_top)
            Xd = deep_variants(T, S, nh)
            Xn = near_share(T, S, nh)
            for var in ("decay", "presence", "weak"):
                key = f"{side}|S{n_top}|{var}|R_pro|c0.2"
                res[key] = thin_eval(P, T, To, S, var, "R_pro", 0.2, nh, Xd, Xn)
                r = res[key]
                log(f"{key}: LOYO cond {np.mean([f['cond'] for f in r['folds'].values()]) * 1e4:+.2f}bp flat {np.mean([f['flat'] for f in r['folds'].values()]) * 1e4:+.2f}bp t_c {r['t_dc']:.2f}")
        json.dump(res, open(OUTD / f"thin{SFX}.json", "w"), ensure_ascii=False, indent=1)
    log("thin 完了")


# ------------------------------------------------------------------ 第 1 段 B: 特性プロファイルとクラスタ（2020–2021 で固定。as-of）
PROFILE_YEARS = (2020, 2021)
CLUSTER_SET = 100


def clu_ols(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """OLS + レース単位クラスタ頑健共分散。返り値 (beta, se, r2_inc 用の SSE)。"""
    XtX = X.T @ X + 1e-9 * np.eye(X.shape[1])
    inv = np.linalg.inv(XtX)
    beta = inv @ (X.T @ y)
    e = y - X @ beta
    u = pd.DataFrame(X * e[:, None]).groupby(pd.factorize(groups)[0]).sum().to_numpy()
    meat = u.T @ u
    cov = inv @ meat @ inv
    return beta, np.sqrt(np.maximum(np.diag(cov), 0)), float((e ** 2).sum()), cov


def level_profile(P: pd.DataFrame, A: sp.csr_matrix, y_col: str, years=PROFILE_YEARS, k: float = 200.0):
    """祖先ごとの、条件水準ごとの『祖先固有の偏り』d_s[l] = 縮約した水準の平均 − 祖先の全体の平均 − 全馬の水準の偏り。
    全馬に共通の条件の効果（その水準で走る馬の平均成績の偏り）を引くので、祖先を持つ馬だけに出る傾きを測る。
    縮約は『祖先の効果なし』の期待値（祖先の全体の平均 + 全馬の水準の偏り）へ向けて行う。A: 馬×祖先の有無（0/1）。"""
    d = P[P[y_col].notna() & P["year"].isin(list(years))]
    y = d[y_col].to_numpy(float)
    C = cond_onehot(d)
    cnt_g = np.asarray(C.sum(axis=0)).ravel()
    gdev = np.asarray(C.T @ y).ravel() / np.maximum(cnt_g, 1) - y.mean()
    Ar = A[d["h"].to_numpy()]
    cnt = (Ar.T @ C).toarray()
    sm = (Ar.T @ C.multiply(y.reshape(-1, 1)).tocsr()).toarray()
    cnt_all = np.asarray(Ar.sum(axis=0)).ravel()
    sum_all = np.asarray(Ar.T @ y).ravel()
    m_all = np.divide(sum_all, cnt_all, out=np.zeros_like(sum_all), where=cnt_all > 0)
    expect = m_all[:, None] + gdev[None, :]
    ml = (sm + k * expect) / (cnt + k)
    return ml - expect, cnt_all, m_all


def cmd_profile(a) -> None:
    from sklearn.cluster import KMeans

    P = pd.read_parquet(OUTD / "prep.parquet")
    nh = int(pd.read_parquet(OUTD / "horses_idx.parquet")["h"].max()) + 1
    res: dict = {}
    for side in SIDES:
        T, pop = load_side(side)
        To, _ = load_side("pat" if side == "mat" else "mat")
        S = pick_set(pop, CLUSTER_SET)
        names = pop.set_index("anc").loc[S, "name"].tolist()
        A = sparse_from(T, S, nh, vals=np.ones(len(T)))
        share = sparse_from(T, S, nh, col="share")
        out_side: dict = {"S": names}
        for basis in ("R_pro", "y_old"):
            Dm, cnt_all, m_all = level_profile(P, A, basis)
            use = cnt_all >= 500
            Z = Dm[use]
            Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            km = KMeans(n_clusters=PRE["K_clusters"], random_state=0, n_init=10).fit(Zs)
            lab = np.full(len(S), -1); lab[use] = km.labels_
            cent = np.vstack([Dm[lab == k].mean(0) for k in range(PRE["K_clusters"])])
            np.save(OUTD / f"profile{SFX}_{side}_{basis}.npy", Dm)
            np.save(OUTD / f"cluster{SFX}_{side}_{basis}.npy", lab)
            desc = []
            for k in range(PRE["K_clusters"]):
                c = cent[k]
                o = np.argsort(-c)
                desc.append({"k": k, "n_anc": int((lab == k).sum()), "members": [names[i] for i in np.where(lab == k)[0][:8]],
                             "up": [(LEVEL_NAMES[i], round(float(c[i]), 4)) for i in o[:3]], "down": [(LEVEL_NAMES[i], round(float(c[i]), 4)) for i in o[-3:]]})
            out_side[basis] = {"clusters": desc, "n_used": int(use.sum()), "profile_sd": float(Z.std())}
            # 馬のタイプ: クラスタごとの血量の合計の最大（合計が小さい馬は none）
            Hs = share.toarray()
            tot = Hs.sum(1)
            pi = np.zeros((nh, PRE["K_clusters"]))
            for k in range(PRE["K_clusters"]):
                pi[:, k] = Hs[:, lab == k].sum(1)
            typ = np.where(tot >= 0.02, pi.argmax(1), -1)
            pd.DataFrame({"h": np.arange(nh), "type": typ}).to_parquet(OUTD / f"type{SFX}_{side}_{basis}.parquet", index=False)
            # 検証: 予測した偏り d̂ と実績の馬内相対適性 R_pro の傾き（年別）
            pin = pi / np.maximum(pi.sum(1, keepdims=True), 1e-9)
            rows = {}
            for yr in (2022, 2023, 2024, 2025, 2026):
                d = P[(P["year"] == yr) & P["R_pro"].notna()]
                h = d["h"].to_numpy()
                Lall = np.stack([OFF[f] + d[f"L_{f}"].to_numpy() for f in FAMS], 1)
                okL = np.stack([d[f"L_{f}"].to_numpy() >= 0 for f in FAMS], 1)
                dhat_c = np.zeros(len(d))
                dhat_k = np.zeros(len(d))
                w = Hs[h] / np.maximum(Hs[h].sum(1, keepdims=True), 1e-9)
                for j in range(len(FAMS)):
                    lv = np.where(okL[:, j], Lall[:, j], 0)
                    dhat_c += np.where(okL[:, j], (pin[h] * cent[:, lv].T).sum(1), 0)         # クラスタ（ソフト）
                    dhat_k += np.where(okL[:, j], (w * Dm[:, lv].T).sum(1), 0)               # 祖先別（連続）
                has = tot[h] >= 0.02
                y = d["R_pro"].to_numpy(float)
                r = {}
                for nm, x in (("cluster", dhat_c), ("anc", dhat_k)):
                    Cd = cond_onehot(d).toarray()[has][:, 1:]
                    X = np.c_[np.ones(has.sum()), Cd, x[has]]
                    b, se, sse, _ = clu_ols(X, y[has], d["race_id"].to_numpy()[has])
                    r[nm] = {"slope": float(b[-1]), "t": float(b[-1] / se[-1]), "n": int(has.sum()), "sd_x": float(x[has].std())}
                rows[str(yr)] = r
            out_side[basis]["validation"] = rows
            log(f"{side}|{basis}: 検証 " + " ".join(f"{yr}:cl t={v['cluster']['t']:+.1f}/anc t={v['anc']['t']:+.1f}" for yr, v in rows.items()))
        res[side] = out_side
    json.dump(res, open(OUTD / f"profile{SFX}.json", "w"), ensure_ascii=False, indent=1)
    log("profile 完了")


# ------------------------------------------------------------------ 第 2 段: 登場回数 n の用量反応（第 1 段で作ったプロファイルとの一致度）
def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    ranked = p[o] * len(p) / (np.arange(len(p)) + 1)
    q[o] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


def cmd_dose(a) -> None:
    from scipy.stats import norm

    P = pd.read_parquet(OUTD / "prep.parquet")
    nh = int(pd.read_parquet(OUTD / "horses_idx.parquet")["h"].max()) + 1
    P = P[P["R_pro"].notna() & (P["year"] >= 2022)].reset_index(drop=True)
    h = P["h"].to_numpy()
    C = cond_onehot(P).toarray()
    res: dict = {"design": {"families": list(FAMS), "set": "S50", "profile": "2020–2021 固定（R_pro）", "bins": PRE["dose_bins"]}}
    bins = {
        "near1": lambda nn, nd, nt, mn: (nn == 1), "near2p": lambda nn, nd, nt, mn: (nn >= 2),
        "deep1": lambda nn, nd, nt, mn: (nd == 1), "deep2": lambda nn, nd, nt, mn: (nd == 2), "deep3p": lambda nn, nd, nt, mn: (nd >= 3),
    }
    for side in SIDES:
        T, pop = load_side(side)
        To, _ = load_side("pat" if side == "mat" else "mat")
        S100 = pick_set(pop, CLUSTER_SET)
        S = pick_set(pop, PRE["top_n_primary"])
        Dm = np.load(OUTD / f"profile{SFX}_{side}_R_pro.npy")                  # S100 × NL
        pos = pd.Series(np.arange(len(S100)), index=S100).reindex(S).to_numpy()
        Dm = Dm[pos]
        mean_n = pop.set_index("anc").loc[S, "mean_n"].to_numpy(float)
        Nn = sparse_from(T, S, nh, col="n_near").toarray()
        Nd = sparse_from(T, S, nh, col="n_deep").toarray()
        Nt = Nn + Nd
        sh_all = sparse_from(T, S, nh, col="share").toarray().sum(1)
        sh_oth = sparse_from(To, S, nh, col="share").toarray().sum(1)
        Hn, Hd, Ht = Nn[h], Nd[h], Nt[h]
        ctrl = np.c_[np.ones(len(P)), C[:, 1:], sh_all[h], sh_oth[h]]
        yrs = P["year"].to_numpy()
        coef_rows = []
        for fi, f in enumerate(FAMS):
            L = P[f"L_{f}"].to_numpy()
            okf = L >= 0
            Dsel = np.where(okf[:, None], Dm[:, OFF[f] + np.where(okf, L, 0)].T, 0.0)       # 行 × 祖先: その祖先のこの条件への偏り
            feats = {}
            for nm, fn in bins.items():
                feats[nm] = (fn(Hn, Hd, Ht, mean_n) * Dsel).sum(1)
            feats["above"] = (((Ht >= np.maximum(mean_n, 1)) & (Ht >= 1)) * Dsel).sum(1)
            feats["below"] = (((Ht < np.maximum(mean_n, 1)) & (Ht >= 1)) * Dsel).sum(1)
            for resp in ("R_pro", "y_old", "y_v4"):
                y = P[resp].to_numpy(float)
                okr = ~np.isnan(y)
                for period, ysel in (("disc", (yrs >= 2022) & (yrs <= 2024)), ("test2025", yrs == 2025), ("desc2026", yrs == 2026)):
                    m = okr & ysel
                    if m.sum() < 5000:
                        continue
                    for grp, names_ in (("bins", ["near1", "near2p", "deep1", "deep2", "deep3p"]), ("ab", ["above", "below"])):
                        X = np.c_[ctrl[m], np.stack([feats[k][m] for k in names_], 1)]
                        b, se, sse, cov = clu_ols(X, y[m], P["race_id"].to_numpy()[m])
                        k0 = ctrl.shape[1]
                        for j, nm in enumerate(names_):
                            coef_rows.append({"side": side, "fam": f, "resp": resp, "period": period, "grp": grp, "coef": nm, "beta": float(b[k0 + j]), "se": float(se[k0 + j]),
                                              "t": float(b[k0 + j] / se[k0 + j]), "sd_x": float(feats[nm][m].std()), "n": int(m.sum())})
                        if grp == "bins":      # 用量反応: deep3p − deep1
                            d = b[k0 + 4] - b[k0 + 2]
                            v = cov[k0 + 4, k0 + 4] + cov[k0 + 2, k0 + 2] - 2 * cov[k0 + 4, k0 + 2]
                            coef_rows.append({"side": side, "fam": f, "resp": resp, "period": period, "grp": "trend", "coef": "deep3p-deep1", "beta": float(d), "se": float(np.sqrt(max(v, 0))),
                                              "t": float(d / np.sqrt(max(v, 1e-18))), "sd_x": 0.0, "n": int(m.sum())})
        # 祖先ごと: n>=2 と n=1 のプロファイル一致度の差（discovery）
        anc_rows = []
        dm = (yrs >= 2022) & (yrs <= 2024)
        y = P["R_pro"].to_numpy(float)
        for si, s_ in enumerate(S):
            has = (Ht[:, si] >= 1) & dm
            if has.sum() < 3000:
                continue
            for f in FAMS:
                L = P[f"L_{f}"].to_numpy()
                okf = (L >= 0) & has
                if okf.sum() < 3000:
                    continue
                gl = pd.Series(y[dm & (L >= 0)]).groupby(L[dm & (L >= 0)]).mean()          # 全馬のこの水準の平均（共通の条件の効果）を引いてから、祖先固有の傾きを見る
                yadj = y[okf] - gl.reindex(L[okf]).fillna(0).to_numpy()
                mm = Dm[si, OFF[f] + L[okf]]
                n2 = (Ht[okf, si] >= 2).astype(float)
                X = np.c_[np.ones(okf.sum()), mm, mm * n2, n2]
                b, se, _, _ = clu_ols(X, yadj, P["race_id"].to_numpy()[okf])
                j = 1
                anc_rows.append({"side": side, "anc": pop.set_index("anc").loc[s_, "name"], "fam": f, "b_match": float(b[j]), "t_match": float(b[j] / se[j]),
                                 "b_dose": float(b[j + 1]), "t_dose": float(b[j + 1] / se[j + 1]), "n": int(okf.sum())})
        res[f"{side}_anc"] = anc_rows
        res[f"{side}_coef"] = coef_rows
        log(f"{side}: 係数 {len(coef_rows)} 件、祖先別 {len(anc_rows)} 件")
    json.dump(res, open(OUTD / f"dose{SFX}.json", "w"), ensure_ascii=False, indent=1)
    # 要約: disc の R_pro の bins（BH）と確認
    for side in SIDES:
        c = pd.DataFrame(res[f"{side}_coef"])
        d = c[(c.resp == "R_pro") & (c.period == "disc") & (c.grp.isin(["bins", "ab"]))].copy()
        d["p"] = 2 * (1 - norm.cdf(np.abs(d.t)))
        d["q"] = bh(d["p"].to_numpy())
        log(f"{side}: disc R_pro の係数 {len(d)} 本、q<0.1 = {(d.q < 0.1).sum()}、|t|>2 = {(d.t.abs() > 2).sum()}")
        a_ = pd.DataFrame(res[f"{side}_anc"])
        a_["p"] = 2 * (1 - norm.cdf(np.abs(a_.t_dose)))
        a_["q"] = bh(a_["p"].to_numpy())
        log(f"{side}: 祖先別 {len(a_)} 本、用量差 q<0.1 = {(a_.q < 0.1).sum()}、|t|>2 = {(a_.t_dose.abs() > 2).sum()}、一致度 |t|>2 = {(a_.t_match.abs() > 2).sum()}")
    log("dose 完了")


# ------------------------------------------------------------------ 血統クラスタ対の同時好走
def harville_pairs(q: np.ndarray):
    """勝率ベクトル q（和 1）から Harville の『2 頭がともに 3 着以内』P3[i,j] と『1-2 着の 2 頭』P2[i,j]。"""
    n = len(q)
    P3 = np.zeros((n, n))
    idx = np.array(list(combinations(range(n), 3)))
    a, b, c = q[idx[:, 0]], q[idx[:, 1]], q[idx[:, 2]]
    pr = np.zeros(len(idx))
    for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
        pr += x * y / np.maximum(1 - x, 1e-9) * z / np.maximum(1 - x - y, 1e-9)
    for u, v in ((0, 1), (0, 2), (1, 2)):
        np.add.at(P3, (idx[:, u], idx[:, v]), pr)
    P3 = P3 + P3.T
    P2 = q[:, None] * q[None, :] / np.maximum(1 - q[:, None], 1e-9) + q[None, :] * q[:, None] / np.maximum(1 - q[None, :], 1e-9)
    np.fill_diagonal(P3, 0); np.fill_diagonal(P2, 0)
    return P3, P2


def pair_tables(P: pd.DataFrame, assigns: dict[str, np.ndarray]):
    """レースごとに、割り当てごとのクラスタ対の観測数 O と期待数 E（一様 / 市場 / モデル）を集計。"""
    P = P.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)
    info = {k: {"K": int(v.max()) + 1, "tri": None} for k, v in assigns.items()}
    for k, v in info.items():
        K = v["K"]
        iu = np.triu_indices(K)
        pid = np.full((K, K), -1); pid[iu] = np.arange(len(iu[0])); pid = np.maximum(pid, pid.T)
        v["pid"], v["npair"] = pid, len(iu[0])
    races, years, dates = [], [], []
    store = {k: {t: [] for t in ("O3", "O2", "Eu3", "Eu2", "Em3", "Em2", "Ep3", "Ep2")} for k in assigns}
    keep_idx = []
    for rid, g in P.groupby("race_id", sort=False):
        n = len(g)
        fin = g["finish_position"].to_numpy(float)
        if n < 5 or np.isnan(fin).any() or (fin <= 3).sum() != 3 or (fin <= 2).sum() != 2:
            continue
        q = g["q_win"].to_numpy(float)
        if np.isnan(q).any() or q.sum() <= 0:
            continue
        q = q / q.sum()
        pw = g["p_win"].to_numpy(float)
        have_p = not np.isnan(pw).any()
        P3m, P2m = harville_pairs(q)
        if have_p:
            pw = pw / pw.sum()
            P3p, P2p = harville_pairs(pw)
        u3, u2 = 6.0 / (n * (n - 1)), 2.0 / (n * (n - 1))
        ii, jj = np.triu_indices(n, 1)
        t3 = np.where(fin <= 3)[0]
        t2 = np.where(fin <= 2)[0]
        h = g["h"].to_numpy()
        for k, v in info.items():
            c = assigns[k][h]
            ok = (c[ii] >= 0) & (c[jj] >= 0)
            pid = v["pid"][np.maximum(c[ii], 0), np.maximum(c[jj], 0)]
            pid = np.where(ok, pid, -1)
            m = pid >= 0
            npair = v["npair"]
            def bc(w):
                return np.bincount(pid[m], weights=w[m], minlength=npair).astype(np.float32)
            O3 = np.zeros(npair, np.float32); O2 = np.zeros(npair, np.float32)
            for x, y_ in combinations(t3, 2):
                if c[x] >= 0 and c[y_] >= 0:
                    O3[v["pid"][c[x], c[y_]]] += 1
            if c[t2[0]] >= 0 and c[t2[1]] >= 0:
                O2[v["pid"][c[t2[0]], c[t2[1]]]] += 1
            st = store[k]
            st["O3"].append(O3); st["O2"].append(O2)
            st["Eu3"].append(bc(np.full(len(ii), u3))); st["Eu2"].append(bc(np.full(len(ii), u2)))
            st["Em3"].append(bc(P3m[ii, jj])); st["Em2"].append(bc(P2m[ii, jj]))
            if have_p:
                st["Ep3"].append(bc(P3p[ii, jj])); st["Ep2"].append(bc(P2p[ii, jj]))
            else:
                st["Ep3"].append(np.full(npair, np.nan, np.float32)); st["Ep2"].append(np.full(npair, np.nan, np.float32))
        races.append(rid); years.append(int(g["year"].iloc[0])); dates.append(g["date"].iloc[0])
    arr = {k: {t: np.vstack(vv) for t, vv in d.items()} for k, d in store.items()}
    return info, arr, np.array(races), np.array(years)


def lift_stats(O, E, mask, min_e=5.0):
    o, e = O[mask].sum(0), E[mask].sum(0)
    d = O[mask] - E[mask]
    z = d.sum(0) / np.sqrt(np.maximum((d ** 2).sum(0), 1e-12))
    return o, e, z


def roi_for(P: pd.DataFrame, races_sel: set, code: np.ndarray, pairs: set, pay: dict, boot: int = 1000):
    """選んだクラスタ対（(lo,hi) の集合）に該当する馬の組をすべて、馬連・ワイドで 100 円ずつ買ったときの回収（レース単位でブートストラップ）。"""
    rows = []
    for rid, g in P[P["race_id"].isin(races_sel)].groupby("race_id", sort=False):
        c = code[g["h"].to_numpy()]
        hn = g["horse_number"].to_numpy()
        cost = pu = pw = 0
        for i, j in combinations(range(len(g)), 2):
            if c[i] < 0 or c[j] < 0:
                continue
            if (min(c[i], c[j]), max(c[i], c[j])) in pairs:
                cost += 1
                key = (rid, frozenset((int(hn[i]), int(hn[j]))))
                pu += pay.get(("umaren",) + key, 0)
                pw += pay.get(("wide",) + key, 0)
        if cost:
            rows.append((cost, pu, pw))
    if not rows:
        return None
    r = np.array(rows, float)
    rng = np.random.default_rng(SEED)
    out = {"races": int(len(r)), "tickets": int(r[:, 0].sum())}
    for nm, col in (("umaren", 1), ("wide", 2)):
        tot = r[:, col].sum() / (100.0 * r[:, 0].sum()) * 100.0 / 100.0
        bs = []
        for _ in range(boot):
            ix = rng.integers(0, len(r), len(r))
            bs.append(r[ix, col].sum() / (100.0 * r[ix, 0].sum()))
        out[nm] = {"roi": float(r[:, col].sum() / (100.0 * r[:, 0].sum())), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    return out


def cmd_pair(a) -> None:
    from scipy.stats import norm

    P = pd.read_parquet(OUTD / "prep.parquet")
    hx = pd.read_parquet(OUTD / "horses_idx.parquet")
    nh = len(hx)
    assigns: dict[str, np.ndarray] = {}
    for nm, f in (("A_embK16", "cluster_K16.parquet"), ("B_embK40", "cluster_K40.parquet")):
        c = pd.read_parquet(TP60 / "embed" / f)
        assigns[nm] = hx["horse_id"].map(c.set_index("horse_id")["g"]).fillna(-1).astype(int).to_numpy()
    for side, tag in (("mat", "C"), ("pat", "D")):
        for basis, bt in (("R_pro", "rel"), ("y_old", "pop")):
            tp = pd.read_parquet(OUTD / f"type{SFX}_{side}_{basis}.parquet").sort_values("h")
            t = tp["type"].to_numpy().copy()
            t = np.where(t < 0, PRE["K_clusters"], t)                    # none もひとつのタイプとして扱う
            assigns[f"{tag}_{side}_{bt}"] = t
    log("割り当て: " + ", ".join(f"{k}(K={v.max() + 1})" for k, v in assigns.items()))
    info, arr, races, years = pair_tables(P, assigns)
    log(f"レース {len(races):,}")
    pay_df = pd.read_parquet(PAYOFF)
    pay_df = pay_df[pay_df["bet_type"].isin(["umaren", "wide"]) & pay_df["parsed"]]
    pay = {}
    for r in pay_df.itertuples():
        try:
            x, y_ = (int(v) for v in str(r.combo).split("-"))
        except ValueError:
            continue
        pay[(r.bet_type, r.race_id, frozenset((x, y_)))] = pay.get((r.bet_type, r.race_id, frozenset((x, y_))), 0) + int(r.payout)
    disc = (years >= 2020) & (years <= 2024)
    disc_m = (years >= 2022) & (years <= 2024)
    res: dict = {"n_races": int(len(races)), "assignments": {k: int(v.max() + 1) for k, v in assigns.items()}}
    allrows = []
    for k, v in info.items():
        K, pid = v["K"], v["pid"]
        iu = np.triu_indices(K)
        for tgt in ("3", "2"):
            O, Eu, Em, Ep = arr[k][f"O{tgt}"], arr[k][f"Eu{tgt}"], arr[k][f"Em{tgt}"], arr[k][f"Ep{tgt}"]
            o, e, z = lift_stats(O, Em, disc)
            _, eu, _ = lift_stats(O, Eu, disc)
            om, ep, zp = lift_stats(O, np.nan_to_num(Ep), disc_m)
            ypos = np.stack([(O[years == y].sum(0) > Em[years == y].sum(0)) & (Em[years == y].sum(0) > 0.5) for y in range(2020, 2025)], 0).sum(0)
            ok = e >= 5.0
            p = 2 * (1 - norm.cdf(np.abs(z)))
            df = pd.DataFrame({"assign": k, "target": tgt, "lo": iu[0], "hi": iu[1], "O": o, "E_mkt": e, "E_unif": eu, "lift_mkt": o / np.maximum(e, 1e-9), "lift_unif": o / np.maximum(eu, 1e-9),
                               "lift_model": np.where(ep > 5, om / np.maximum(ep, 1e-9), np.nan), "z": z, "p": p, "years_pos": ypos})
            df = df[ok]
            allrows.append(df)
    A = pd.concat(allrows, ignore_index=True)
    A["q_global"] = bh(A["p"].to_numpy())
    A["q_assign"] = A.groupby(["assign", "target"])["p"].transform(lambda s: bh(s.to_numpy()))
    res["n_tests"] = int(len(A))
    res["n_tests_by_assign"] = A.groupby(["assign", "target"]).size().rename(lambda x: str(x)).to_dict() if False else {f"{a_}|{t_}": int(n) for (a_, t_), n in A.groupby(["assign", "target"]).size().items()}
    res["sig_global_q10"] = int((A.q_global < 0.1).sum())
    res["sig_assign_q10"] = {f"{a_}|{t_}": int(((g.q_assign < 0.1)).sum()) for (a_, t_), g in A.groupby(["assign", "target"])}
    res["sig_assign_q10_lift_gt1"] = {f"{a_}|{t_}": int(((g.q_assign < 0.1) & (g.lift_mkt > 1)).sum()) for (a_, t_), g in A.groupby(["assign", "target"])}
    res["z_summary"] = {f"{a_}|{t_}": {"mean_z": float(g.z.mean()), "sd_z": float(g.z.std()), "max_z": float(g.z.max()), "n": int(len(g))} for (a_, t_), g in A.groupby(["assign", "target"])}
    A.to_parquet(OUTD / f"pair_tests{SFX}.parquet", index=False)
    # 2025 / 2026 の確認: discovery の z 上位 5（正）
    conf: dict = {}
    for k, v in info.items():
        for tgt in ("3", "2"):
            g = A[(A["assign"] == k) & (A["target"] == tgt) & (A["lift_mkt"] > 1) & (A["E_mkt"] >= 10)].sort_values("z", ascending=False).head(5)
            O, Em = arr[k][f"O{tgt}"], arr[k][f"Em{tgt}"]
            pids = [v["pid"][int(r.lo), int(r.hi)] for r in g.itertuples()]
            rec: dict = {"pairs": [(int(r.lo), int(r.hi), round(float(r.lift_mkt), 3), round(float(r.z), 2)) for r in g.itertuples()]}
            for nm, msk in (("disc", disc), ("test2025", years == 2025), ("desc2026", years == 2026)):
                if msk.sum() == 0 or not pids:
                    continue
                oo, ee = O[msk][:, pids].sum(), Em[msk][:, pids].sum()
                d_ = (O[msk][:, pids] - Em[msk][:, pids]).sum(1)
                rec[nm] = {"O": float(oo), "E": float(ee), "lift": float(oo / max(ee, 1e-9)), "z": float(d_.sum() / np.sqrt(max((d_ ** 2).sum(), 1e-12)))}
            if pids:
                pairs = {(int(r.lo), int(r.hi)) for r in g.itertuples()}
                code = assigns[k]
                for nm, ysel in (("test2025", 2025), ("desc2026", 2026), ("disc", None)):
                    rs = set(races[(years == ysel)] if ysel else races[disc])
                    rec[f"roi_{nm}"] = roi_for(P, rs, code, pairs, pay, boot=500)
            conf[f"{k}|{tgt}"] = rec
            log(f"{k}|{tgt}: top5 disc lift={rec.get('disc', {}).get('lift', float('nan')):.2f} → 2025 lift={rec.get('test2025', {}).get('lift', float('nan')):.2f} z={rec.get('test2025', {}).get('z', float('nan')):+.2f}")
    res["confirm_top5"] = conf
    # 特徴量化の見込み: 同レースの他馬との（縮約した）対 lift の平均 → 2025 の残差との関連
    feat: dict = {}
    for k, v in info.items():
        K, pid = v["K"], v["pid"]
        O, Em = arr[k]["O3"], arr[k]["Em3"]
        o, e = O[disc].sum(0), Em[disc].sum(0)
        lift_sh = (o + 20.0) / (e + 20.0)
        Lmat = lift_sh[pid]
        code = assigns[k]
        d = P[P["year"].isin([2025])].copy()
        c = code[d["h"].to_numpy()]
        f = np.full(len(d), np.nan)
        cnt = d.groupby("race_id")["h"].transform("size").to_numpy()
        # 馬ごとに、同レースの他馬との lift - 1 の平均
        rid_codes = d.assign(c=c).groupby("race_id")["c"].apply(lambda s: s.to_numpy())
        vals = []
        for rid, g in d.assign(c=c).groupby("race_id", sort=False):
            cc = g["c"].to_numpy()
            M = Lmat[np.maximum(cc, 0)[:, None], np.maximum(cc, 0)[None, :]] - 1.0
            M[(cc < 0)[:, None] | (cc < 0)[None, :]] = 0
            np.fill_diagonal(M, 0)
            vals.append(pd.Series(M.sum(1) / max(len(cc) - 1, 1), index=g.index))
        d["f_comp"] = pd.concat(vals)
        dis = P[P["year"].between(2020, 2024)]
        cd = code[dis["h"].to_numpy()]
        for resp in ("R_pro", "y_v4", "y_old"):
            dd = d[d[resp].notna() & d["f_comp"].notna()].copy()
            if len(dd) < 5000:
                continue
            ok_ = dis[resp].notna() & (cd >= 0)
            gm = float(dis.loc[ok_, resp].mean())
            cs = pd.DataFrame({"c": cd[ok_.to_numpy()], "y": dis.loc[ok_, resp].to_numpy()}).groupby("c")["y"].agg(["sum", "count"])
            cm = ((cs["sum"] + 200.0 * gm) / (cs["count"] + 200.0)).reindex(range(v["K"])).fillna(gm).to_numpy()
            dd["x_own"] = cm[np.maximum(code[dd["h"].to_numpy()], 0)]
            X = np.c_[np.ones(len(dd)), dd["f_comp"].to_numpy()]
            b, se, sse, _ = clu_ols(X, dd[resp].to_numpy(float), dd["race_id"].to_numpy())
            X2 = np.c_[np.ones(len(dd)), dd["x_own"].to_numpy(), dd["f_comp"].to_numpy()]
            b2, se2, _, _ = clu_ols(X2, dd[resp].to_numpy(float), dd["race_id"].to_numpy())
            feat[f"{k}|{resp}"] = {"slope": float(b[1]), "t": float(b[1] / se[1]), "slope_ctrl_own": float(b2[2]), "t_ctrl_own": float(b2[2] / se2[2]), "sd_f": float(dd["f_comp"].std()), "n": int(len(dd))}
    res["feature_prospect_2025"] = feat
    json.dump(res, open(OUTD / f"pair{SFX}.json", "w"), ensure_ascii=False, indent=1, default=float)
    log("pair 完了")


# ------------------------------------------------------------------ 既存 role_lift_profiles との関係
def cmd_rlp(a) -> None:
    rl = pd.read_parquet(ROOT / "data/page_reference/note_aptitude_race.bak/role_lift_profiles.parquet")
    vmap = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
    res: dict = {}
    for side, roles in (("mat", ["maternal_deep", "MF"]), ("pat", ["F", "FF"])):
        T, pop = load_side(side)
        S = pick_set(pop, CLUSTER_SET)
        nid = pop.set_index("anc").loc[S, "node_id"].tolist()
        for basis in ("R_pro", "y_old"):
            Dm = np.load(OUTD / f"profile{SFX}_{side}_{basis}.npy")
            for role in roles:
                r = rl[rl["role"] == role].set_index("stallion_id")
                xs, ys, per = [], [], []
                for si, n_ in enumerate(nid):
                    if n_ not in r.index:
                        continue
                    row = r.loc[n_]
                    lifts = {k: row.get(f"lift_win_v_{k}") for k in vmap}
                    vals = np.array([lifts[k] for k in vmap], float)
                    if np.isfinite(vals).sum() < 5:
                        continue
                    dev = vals - np.nanmean(vals)
                    mine = np.array([Dm[si, OFF["venue"] + LEVELS["venue"].index(vmap[k])] for k in vmap])
                    ok = np.isfinite(dev)
                    xs.extend(mine[ok]); ys.extend(dev[ok])
                    if ok.sum() >= 5:
                        per.append(np.corrcoef(mine[ok], dev[ok])[0, 1])
                if len(xs) > 20:
                    res[f"{side}|{basis}|{role}"] = {"n_anc": len(per), "pooled_r": float(np.corrcoef(xs, ys)[0, 1]), "median_r": float(np.nanmedian(per)), "n_pts": len(xs)}
                    log(f"{side}|{basis}|{role}: 祖先 {len(per)}、競馬場の偏りの相関 pooled r={res[f'{side}|{basis}|{role}']['pooled_r']:+.3f}、中央値 r={res[f'{side}|{basis}|{role}']['median_r']:+.3f}")
    json.dump(res, open(OUTD / f"rlp{SFX}.json", "w"), ensure_ascii=False, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("anc", "prep", "thin", "thin_strong", "profile", "dose", "pair", "rlp"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"anc": cmd_anc, "prep": cmd_prep, "thin": cmd_thin, "profile": cmd_profile, "dose": cmd_dose, "pair": cmd_pair, "thin_strong": cmd_thin_strong, "rlp": cmd_rlp}.get(a.cmd, lambda x: (_ for _ in ()).throw(SystemExit(f"{a.cmd} は未実装")))(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
