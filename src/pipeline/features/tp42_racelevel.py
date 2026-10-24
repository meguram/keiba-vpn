"""tp-42: めぐ指数（v3a の絶対指数）から作る『レースレベル』の特徴量候補。すべて as-of（その走の日付より前に分かる情報だけ）。

入力 d は ``prepare(runs, megu)``（factory_self.prep の列 + 絶対指数 a）で、馬・日付順に並んでいること。
群
  RL_A   過去走が行われたレースの強さ（出走馬の絶対指数の平均・上位3頭の平均。自分を除く）と、その中での自分の指数（相手関係補正）
  RL_A2  過去レースの『後日談』: そのレースの相手が、その後（今回の日付より前）に出した絶対指数 / 相手の事前水準との差
  RL_B   今回のレースの強さ: 出走馬の想定指数（過去5走の指数加重平均）の分布と、その中での自分の位置
  RL_C   自分の想定指数・勢い・面/距離を揃えた想定
先読みを避ける約束
  * 過去走の特徴は馬ごとに 1 走以上ずらす（lags）。自分の今回の結果は入らない
  * 後日談は『次走の日付 < 今回の日付』の相手の次走だけを使う（累積和 + 二分探索）
  * 今回のレースの強さは、出走馬の過去走だけから作る（今回の結果は使わない）
市場オッズ・人気は使わない。キーは race_id / horse_id。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs

KEY = ["race_id", "horse_id"]
K = 5                      # 想定指数に使う過去走数
W = 0.65                   # 1 走古くなるごとの重み
KD = 3                     # 後日談に使う過去レース数
DIST_SCALE = 600.0         # 距離の近さの重み exp(-|差|/600m)
MIN_FIELD = 4              # レースの強さを出す最小の指数あり頭数
BIG = 1_000_000


def prepare(runs: pd.DataFrame, megu: pd.DataFrame) -> pd.DataFrame:
    """走表（fs.load_runs の形）にめぐ指数の絶対指数 a を結合する。megu は tg.load_megu の形（megu_abs 列）。"""
    d = fs.prep(runs)
    d = d.merge(megu[KEY + ["megu_abs"]], on=KEY, how="left")
    d["a"] = pd.to_numeric(d["megu_abs"], errors="coerce").astype("float64")
    d["surf"] = (d["surface"].astype(str).str.startswith("芝")).astype(float)
    d = d.sort_values(["horse_id", "dnum", "race_id"], kind="stable").reset_index(drop=True)
    return d


def _ewm(M: np.ndarray, w: float = W, extra_w: np.ndarray | None = None) -> np.ndarray:
    """M の列 0 が 1 走前。値のある走だけで重み w**k の加重平均（extra_w は走ごとの追加の重み）。"""
    ww = np.broadcast_to(w ** np.arange(M.shape[1]), M.shape).astype(float)
    if extra_w is not None:
        ww = ww * extra_w
    m = np.isfinite(M)
    num = (np.where(m, M, 0.0) * ww * m).sum(1)
    den = (ww * m).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def lags_m(d: pd.DataFrame, col: str, k: int, lag_days: int = 0) -> np.ndarray:
    """fs.lags と同じ (n, k) の過去走の値。今回の日付との差が lag_days 日未満の走は『まだ指数が反映されていない』として欠損にする。
    lag_days は指数の反映遅れ（確定結果が揃うのは金曜の weekly-update 後）を学習側にも再現するための設定。0 なら遅れなし。"""
    M = fs.lags(d, col, k)
    if lag_days > 0:
        gap = d["dnum"].to_numpy(float)[:, None] - fs.lags(d, "dnum", k)
        M = np.where(gap >= lag_days, M, np.nan)
    return M


# ───────────────────────── 過去レースの強さ（事後。自分を除く）
def add_race_level(d: pd.DataFrame) -> pd.DataFrame:
    """各走に、そのレースの『自分を除く出走馬の絶対指数の平均 / 上位3頭の平均』を足す（そのレースの結果を使うので、必ず過去走としてずらして使う）。"""
    d = d.copy()
    a = d["a"]
    known = a.notna().astype(float)
    g = d.groupby("race_id", sort=False)
    s = g["a"].transform("sum")
    n = known.groupby(d["race_id"], sort=False).transform("sum")
    a0 = a.fillna(0.0)
    n_ex = n - known
    with np.errstate(invalid="ignore", divide="ignore"):
        opp_mean = np.where(n_ex >= MIN_FIELD - 1, (s - a0) / n_ex, np.nan)
    d["rl_n"] = n.to_numpy()
    d["opp_mean"] = opp_mean
    # 上位 3 頭の平均（自分が上位 3 に入るときは自分を除いて 4 位まで見る）
    srt = pd.DataFrame({"race_id": d["race_id"].to_numpy(), "a": a.to_numpy(), "_i": np.arange(len(d))})
    srt = srt[srt["a"].notna()].sort_values(["race_id", "a"], ascending=[True, False], kind="stable")
    srt["rk"] = srt.groupby("race_id", sort=False).cumcount()
    top = srt[srt["rk"] < 4]
    piv = top.pivot(index="race_id", columns="rk", values="a").reindex(columns=[0, 1, 2, 3])
    V = piv.reindex(d["race_id"]).to_numpy(float)
    rk_self = np.full(len(d), np.inf)
    rk_self[srt["_i"].to_numpy()] = srt["rk"].to_numpy()
    cnt4 = np.isfinite(V).sum(1)
    sum4 = np.nansum(V, axis=1)
    sum3 = np.nansum(V[:, :3], axis=1)
    in3 = rk_self < 3
    with np.errstate(invalid="ignore", divide="ignore"):
        t3_ex = np.where(in3, np.where(cnt4 >= 4, (sum4 - a0.to_numpy()) / 3.0, np.nan), np.where(cnt4 >= 3, sum3 / 3.0, np.nan))
    d["opp_top3"] = t3_ex
    d["rel_mean"] = a - d["opp_mean"]
    d["rel_top3"] = a - d["opp_top3"]
    return d


def grp_A(d: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    """RL_A: 過去 K 走のレースの強さと、相手関係補正後の自分の指数。"""
    x = add_race_level(d)
    out = pd.DataFrame(index=d.index)
    for name, col in (("oppm", "opp_mean"), ("oppt3", "opp_top3"), ("rel", "rel_mean"), ("rel3", "rel_top3")):
        M = lags_m(x, col, K, lag_days)
        out[f"rlA_{name}_ewm"] = _ewm(M)
        out[f"rlA_{name}_last"] = M[:, 0]
        if name in ("oppm", "oppt3"):
            with np.errstate(invalid="ignore"):
                out[f"rlA_{name}_max"] = np.where(np.isfinite(M).any(1), np.nanmax(np.where(np.isfinite(M), M, -np.inf), axis=1), np.nan)
                out[f"rlA_{name}_min"] = np.where(np.isfinite(M).any(1), np.nanmin(np.where(np.isfinite(M), M, np.inf), axis=1), np.nan)
    out["rlA_oppt3_gap"] = out["rlA_oppt3_ewm"] - out["rlA_oppm_ewm"]
    return out


# ───────────────────────── 後日談（相手のその後。今回の日付より前の次走だけ）
def grp_A2(d: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    """RL_A2: 過去 KD レースの相手が、今回の日付より前に出した次走の絶対指数 / 相手の事前水準との差の指数加重平均。"""
    n = len(d)
    h = d["horse_id"].to_numpy()
    dn = d["dnum"].to_numpy(float)
    a = d["a"].to_numpy(float)
    same_next = np.r_[h[1:] == h[:-1], False]
    next_dn = np.where(same_next, np.r_[dn[1:], np.nan], np.nan)
    next_a = np.where(same_next, np.r_[a[1:], np.nan], np.nan)
    pre = _ewm(lags_m(d, "a", K, lag_days))                                         # その走の前の水準（その走の結果は入らない）
    v1 = next_a
    v2 = next_a - pre
    ok = np.isfinite(next_dn) & np.isfinite(next_a)
    ridx = pd.factorize(d["race_id"])[0].astype(np.int64)
    order = np.lexsort((next_dn[ok], ridx[ok]))
    idx_ok = np.flatnonzero(ok)[order]
    keys = ridx[idx_ok] * BIG + next_dn[idx_ok].astype(np.int64)
    c1 = np.r_[0.0, np.cumsum(v1[idx_ok])]
    v2z = np.where(np.isfinite(v2[idx_ok]), v2[idx_ok], 0.0)
    c2 = np.r_[0.0, np.cumsum(v2z)]
    c2n = np.r_[0.0, np.cumsum(np.isfinite(v2[idx_ok]).astype(float))]
    cnt = np.arange(len(idx_ok) + 1)

    pos_in_group = d.groupby("horse_id", sort=False).cumcount().to_numpy()
    out = pd.DataFrame(index=d.index)
    cols1, cols2, cnts = [], [], []
    for j in range(1, KD + 1):
        src = np.where(pos_in_group >= j, np.arange(n) - j, -1)                       # j 走前の行
        have = src >= 0
        s = np.where(have, src, 0)
        if lag_days > 0:                                                              # j 走前そのものが反映前なら使わない
            have = have & ((dn - dn[s]) >= lag_days)
        q = ridx[s] * BIG
        lo = np.searchsorted(keys, q, side="left")
        cutoff = dn.astype(np.int64) - lag_days + (1 if lag_days > 0 else 0)
        hi = np.searchsorted(keys, q + cutoff, side="left")                          # 次走の日付 < 今回の日付（遅れありなら 日付 <= 今回 - lag_days）
        S1 = c1[hi] - c1[lo]
        S2 = c2[hi] - c2[lo]
        N = (cnt[hi] - cnt[lo]).astype(float)
        N2 = c2n[hi] - c2n[lo]
        if j >= 2:                                                                    # 自分の j 走前→(j-1)走前 の次走は今回より前なので差し引く
            sv1 = np.where(np.isfinite(v1[s]), v1[s], 0.0)
            sv2 = np.where(np.isfinite(v2[s]), v2[s], 0.0)
            own = np.isfinite(v1[s]) & np.isfinite(next_dn[s]) & (next_dn[s] < cutoff)
            S1 = S1 - np.where(own, sv1, 0.0)
            N = N - own
            own2 = own & np.isfinite(v2[s])
            S2 = S2 - np.where(own2, sv2, 0.0)
            N2 = N2 - own2
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = np.where(have & (N >= 2), S1 / N, np.nan)
            m2 = np.where(have & (N2 >= 2), S2 / N2, np.nan)
        cols1.append(m1)
        cols2.append(m2)
        cnts.append(np.where(have, N, np.nan))
    M1, M2 = np.column_stack(cols1), np.column_stack(cols2)
    out["rlA2_next_ewm"] = _ewm(M1)
    out["rlA2_delta_ewm"] = _ewm(M2)
    out["rlA2_next_last"] = M1[:, 0]
    out["rlA2_delta_last"] = M2[:, 0]
    out["rlA2_n_last"] = cnts[0]
    return out


# ───────────────────────── 自分の想定指数
def _self_block(d: pd.DataFrame, lag_days: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    M = lags_m(d, "a", K, lag_days)
    out = pd.DataFrame(index=d.index)
    E = _ewm(M)
    out["rlC_E"] = E
    out["rlC_last"] = M[:, 0]
    fin = np.isfinite(M)
    with np.errstate(invalid="ignore"):
        out["rlC_best"] = np.where(fin.any(1), np.nanmax(np.where(fin, M, -np.inf), axis=1), np.nan)
        out["rlC_mean"] = np.where(fin.any(1), np.nanmean(np.where(fin, M, np.nan), axis=1), np.nan)
        sd = np.nanstd(np.where(fin, M, np.nan), axis=1)
    out["rlC_sd"] = np.where(fin.sum(1) >= 3, sd, np.nan)
    out["rlC_slope"] = fs.slope_last(M)
    out["rlC_n"] = fin.sum(1).astype(float)
    # 今回と同じ面の走を優先 / 距離が近い走を重く
    surf_lag = fs.lags(d, "surf", K)
    today = d["surf"].to_numpy(float)[:, None]
    same = np.where(np.isfinite(surf_lag), (surf_lag == today).astype(float), np.nan)
    Es = _ewm(np.where(same == 1.0, M, np.nan))
    out["rlC_E_samesurf"] = Es
    out["rlC_d_surf"] = Es - E
    dist_lag = fs.lags(d, "dist", K)
    dw = np.exp(-np.abs(dist_lag - d["dist"].to_numpy(float)[:, None]) / DIST_SCALE)
    dw = np.where(np.isfinite(dw), dw, 0.0)
    Ed = _ewm(M, extra_w=dw)
    out["rlC_E_dist"] = Ed
    out["rlC_d_dist"] = Ed - E
    return out, E


def grp_C(d: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    return _self_block(d, lag_days)[0]


# ───────────────────────── 今回のレースの強さ（出走馬の過去走だけ）
def grp_B(d: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    """RL_B: 出走馬の想定指数 E の分布と、その中での自分の位置。E は各馬の過去走だけから作る（今回の結果は使わない）。"""
    E = _self_block(d, lag_days)[1]
    x = pd.DataFrame({"race_id": d["race_id"].to_numpy(), "E": E})
    g = x.groupby("race_id", sort=False)["E"]
    n = g.transform("count").to_numpy(float)
    mean = g.transform("mean").to_numpy(float)
    med = g.transform("median").to_numpy(float)
    sd = g.transform("std").to_numpy(float)
    mx = g.transform("max").to_numpy(float)
    x["rk"] = g.rank(ascending=False, method="min").to_numpy(float)
    srt = x[x["E"].notna()].sort_values(["race_id", "E"], ascending=[True, False], kind="stable")
    srt["r0"] = srt.groupby("race_id", sort=False).cumcount()
    second = srt[srt["r0"] == 1].set_index("race_id")["E"]
    top2 = srt[srt["r0"] < 2].groupby("race_id", sort=False)["E"].mean()
    top3 = srt[srt["r0"] < 3].groupby("race_id", sort=False)["E"].mean()
    sec = d["race_id"].map(second).to_numpy(float)
    t2 = d["race_id"].map(top2).to_numpy(float)
    t3 = d["race_id"].map(top3).to_numpy(float)
    out = pd.DataFrame(index=d.index)
    valid = n >= MIN_FIELD
    out["rlB_n"] = n
    out["rlB_max"] = np.where(valid, mx, np.nan)
    out["rlB_top2"] = np.where(valid, t2, np.nan)
    out["rlB_top3"] = np.where(valid, t3, np.nan)
    out["rlB_mean"] = np.where(valid, mean, np.nan)
    out["rlB_sd"] = np.where(valid, sd, np.nan)
    out["rlB_gap12"] = np.where(valid, mx - sec, np.nan)
    out["rlB_self_minus_max"] = np.where(valid, E - mx, np.nan)
    out["rlB_self_minus_mean"] = np.where(valid, E - mean, np.nan)
    out["rlB_self_minus_med"] = np.where(valid, E - med, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["rlB_self_z"] = np.where(valid & (sd > 0), (E - mean) / sd, np.nan)
    out["rlB_rank"] = np.where(valid & np.isfinite(E), x["rk"].to_numpy(float), np.nan)
    out["rlB_rank_pct"] = np.where(valid & np.isfinite(E), (x["rk"].to_numpy(float) - 1.0) / np.maximum(n - 1.0, 1.0), np.nan)
    out["rlB_gap_sec"] = np.where(valid & np.isfinite(E), 0.1 * (mx - E), np.nan)
    # 自分より強い馬の数 / 自分に近い馬の数（±3 点）
    Ev = np.where(np.isfinite(E), E, np.nan)
    codes = pd.factorize(d["race_id"])[0]
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    ends = np.r_[starts[1:], len(cs)]
    stronger = np.full(len(d), np.nan)
    close = np.full(len(d), np.nan)
    Eo = Ev[order]
    for s0, e0 in zip(starts, ends):
        seg = Eo[s0:e0]
        k = np.isfinite(seg)
        if k.sum() < MIN_FIELD:
            continue
        diff = seg[:, None] - seg[None, :]
        st = np.where(np.isfinite(diff), diff < 0, False).sum(1)             # seg[j] > seg[i] の j の数
        cl = np.where(np.isfinite(diff), np.abs(diff) <= 3.0, False).sum(1) - 1
        stronger[order[s0:e0]] = np.where(k, st, np.nan)
        close[order[s0:e0]] = np.where(k, cl, np.nan)
    out["rlB_n_stronger"] = stronger
    out["rlB_n_close"] = close
    return out


GROUPS = {"RL_A": grp_A, "RL_A2": grp_A2, "RL_B": grp_B, "RL_C": grp_C}


def build_all(d: pd.DataFrame, lag_days: int = 0) -> dict[str, pd.DataFrame]:
    out = {}
    for name, f in GROUPS.items():
        t = f(d, lag_days)
        t.insert(0, "horse_id", d["horse_id"].to_numpy())
        t.insert(0, "race_id", d["race_id"].to_numpy())
        out[name] = t
    return out
