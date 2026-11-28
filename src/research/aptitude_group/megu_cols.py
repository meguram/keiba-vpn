"""血統ベクトルの合い具合の列（めぐ AI に足す。計画: docs/operations/aptitude_group/plan_megu_features.md の第 2 版）。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=2 flock /tmp/claude-1000/heavy.lock nice -n 19 python -m src.research.aptitude_group.megu_cols

- 年 t（2017〜24）の列は、t 年 1 月 1 日より前の走りだけで作った値を、その年の全レースに使う。2016 年の行は空（慣らしの年）。
- 縮め・z の尺度・中核（300 走以上）の判定は、年ごとにその切り口のデータで作り直す（全期間の版は使わない）。
- 馬自身の過去の走りは、父と母父の集計から抜く（その馬の走りだけの集計を引く近似。前の位置・TS は厳密。距離・ギアの傾きと条件の対比は、
  鍵の中の中心化と調和の件数のため近似で、片方の条件だけを走った馬からは対比の分が引かれない・critic 7 回目の 6・7）。
  縮め先（中系統・大系統）と牝系 (B) の祖先の集計の中の本人の分は残る（重みが小さい）。
- 50 走未満の父は、計画の◇（血統構成の予測）ではなく、縮めの上の段の値で埋まる（年ごとの予測の学び直しを省いた。計画からの外れ・log.md）。
  t 年より前に産駒の走りが無い父（初年度の産駒）は、馬の行の中系統 → 大系統の値で埋める。
- 列（9 本）: 父 × 距離・芝ダ・芝の道悪・ダートの道悪（4）・父 × コース 6 本の中心化した和（1）・父の確かさ（1）・
  牝系 (B) × 芝の道悪・ダートの道悪（2）・母父 × 芝ダ（1）。
- 偽の列（null_seed）: 馬の単位で、血統の塊（父・(B)・母父の値と確かさ）を、年ごとに生年の中で入れ替える。
出力: common.OUT / megu_cols.parquet（race_id・horse_id と 9 列 × 本物・偽 3 組）。費用 0 円（手元の表だけ）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.aptitude_group import common
from src.research.aptitude_group.line_vectors import MAT_W, long_frame, pedigree_links, shrink_chain
from src.research.aptitude_group.sire_vectors import COURSE, DIMS, K, MIN_RUNS, flags, runs, stats, stats_base
from src.research.pedigree_lab import data
from src.research.pedigree_lab.aptitude import MAT_POS

COLS = ["apt_f_dist", "apt_f_surf", "apt_f_wetT", "apt_f_wetD", "apt_f_course", "apt_f_rel", "apt_b_wetT", "apt_b_wetD", "apt_m_surf"]
VEC_F = ["dist", "surf", "wet_T", "wet_D"] + COURSE
YEARS = range(2017, 2025)
NULL_SEEDS = (101, 102, 103)


def levels(Fb: pd.DataFrame, k_s: str, k_m: str, k_t: str, base: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """鍵ごとの縮める前の和（d・n）と、縮め先の値（上の段を縮めた値）。中系統・大系統の値（vm・vt）も返す
    （t 年より前に産駒の走りが無い父を、馬の行の中系統・大系統の値で埋めるため・critic 7 回目の 3）。"""
    St, Sm, Ss = stats(Fb, k_t, base), stats(Fb, k_m, base), stats(Fb, k_s, base)
    link = Fb[[k_s, k_m, k_t]].replace("", np.nan).dropna().drop_duplicates(k_s).set_index(k_s)
    mid_top = Fb[[k_m, k_t]].replace("", np.nan).dropna().drop_duplicates(k_m).set_index(k_m)[k_t]
    prior, VM, VT = {}, {}, {}
    for nm in DIMS:
        vt = St[("d", nm)] / (St[("n", nm)] + K)
        vm = (Sm[("d", nm)] + K * Sm.index.to_series().map(mid_top).map(vt).fillna(0.0)) / (Sm[("n", nm)] + K)
        mid = link[k_m].reindex(Ss.index)
        top = link[k_t].reindex(Ss.index)
        prior[nm] = mid.map(vm).fillna(top.map(vt)).fillna(0.0).to_numpy()
        VM[nm], VT[nm] = vm, vt
    return Ss, pd.DataFrame(prior, index=Ss.index), VM, VT


def loo_value(S: pd.DataFrame, prior: pd.DataFrame, keys: np.ndarray, Sh: pd.DataFrame, horses: np.ndarray,
              mids: np.ndarray | None = None, tops: np.ndarray | None = None, VM: dict | None = None, VT: dict | None = None) -> pd.DataFrame:
    """鍵（父・母父）の値から、その馬自身の走りの集計を引いて縮める: (D − d_h + K·先) / (N − n_h + K)。
    鍵が t 年より前の走りに無い（初年度の産駒の父など）ときは、馬の行の中系統 → 大系統の値で埋める（どれも無ければ NaN）。"""
    out = {}
    ok = pd.Index(S.index).get_indexer(keys)
    hk = pd.Index(Sh.index).get_indexer(horses)
    for nm in DIMS:
        D = np.where(ok >= 0, S[("d", nm)].to_numpy()[np.maximum(ok, 0)], np.nan)
        N = np.where(ok >= 0, S[("n", nm)].to_numpy()[np.maximum(ok, 0)], np.nan)
        pr = np.where(ok >= 0, prior[nm].to_numpy()[np.maximum(ok, 0)], np.nan)
        dh = np.where(hk >= 0, Sh[("d", nm)].to_numpy()[np.maximum(hk, 0)], 0.0)
        nh = np.where(hk >= 0, Sh[("n", nm)].to_numpy()[np.maximum(hk, 0)], 0.0)
        v = (D - dh + K * pr) / (np.maximum(N - nh, 0) + K)
        if mids is not None:
            fill = pd.Series(mids).map(VM[nm]).fillna(pd.Series(tops).map(VT[nm])).to_numpy(float)
            v = np.where(ok >= 0, v, fill)
        out[nm] = v
    runs_ = np.where(ok >= 0, S["runs"].to_numpy()[np.maximum(ok, 0)], 0.0) - np.where(hk >= 0, Sh["runs"].to_numpy()[np.maximum(hk, 0)], 0.0)
    out["runs"] = np.maximum(runs_, 0)
    return pd.DataFrame(out)


def zscale(V: pd.DataFrame, runs_col: pd.Series, min_runs: int | None = None) -> tuple[dict, dict]:
    """中核（走数 MIN_RUNS 以上）の、走数の重み付きの平均と sd（その年の切り口で）。中核が無ければ全員で。"""
    core = runs_col >= (MIN_RUNS if min_runs is None else min_runs)
    if not core.any():
        core = runs_col > 0
    w = runs_col[core].to_numpy(float)
    mu = {d: float(np.average(V.loc[core, d], weights=w)) for d in DIMS}
    sd = {d: float(np.sqrt(np.average((V.loc[core, d] - mu[d]) ** 2, weights=w))) or 1.0 for d in DIMS}
    return mu, sd


def year_vectors(F: pd.DataFrame, t: int, P: pd.DataFrame, parent: dict, horses_t: pd.DataFrame) -> pd.DataFrame:
    """年 t の初めに固定した、馬ごとの血統の値（z）。horses_t は年 t に走る馬（列 horse_id・sire・ds）。"""
    Fb = F[F["date"] < pd.Timestamp(f"{t}-01-01")].reset_index(drop=True)
    base = stats_base(Fb)
    Sh = stats(Fb, "horse_id", base)                      # 馬自身の走りの集計（父・母父から引く）
    # 父（縮め先: 父の父 → 4 代上）
    Ss, prs, VMs, VTs = levels(Fb, "sire", "ss", "L4", base)
    Vs_full = pd.DataFrame({nm: (Ss[("d", nm)] + K * prs[nm]) / (Ss[("n", nm)] + K) for nm in DIMS}, index=Ss.index)
    mu_s, sd_s = zscale(Vs_full, Ss["runs"])
    f = loo_value(Ss, prs, horses_t["sire"].to_numpy(), Sh, horses_t["horse_id"].to_numpy(),
                  horses_t["ss"].to_numpy(), horses_t["L4"].to_numpy(), VMs, VTs)
    # 母父（母父としての値。縮め先: 母父の父 → 母父の 3 代上）
    Sb, prb, VMb, VTb = levels(Fb, "ds", "bms_ss", "bms_L4", base)
    Vb_full = pd.DataFrame({nm: (Sb[("d", nm)] + K * prb[nm]) / (Sb[("n", nm)] + K) for nm in DIMS}, index=Sb.index)
    mu_b, sd_b = zscale(Vb_full, Sb["runs"])
    m = loo_value(Sb, prb, horses_t["ds"].to_numpy(), Sh, horses_t["horse_id"].to_numpy(),
                  horses_t["bms_ss"].to_numpy(), horses_t["bms_L4"].to_numpy(), VMb, VTb)
    # 牝系 (B): 母の側 2〜5 代の牡の祖先の値（line_vectors と同じ作り）を、馬の母の側 3〜5 代の祖先について血量の重みで平均
    Lb = long_frame(Fb, P[P["horse_id"].isin(set(Fb["horse_id"]))], MAT_W)
    if len(Lb) and (Lb["w"] > 0).any():
        VB = shrink_chain(stats(Lb, "key", stats_base(Lb)), parent)
        mu_B, sd_B = zscale(VB, VB["runs"]) if (VB["runs"] >= MIN_RUNS).any() else ({d: 0.0 for d in DIMS}, {d: 1.0 for d in DIMS})
        zB = pd.DataFrame({d: (VB[d] - mu_B[d]) / sd_B[d] for d in ("wet_T", "wet_D")}, index=VB.index)
    else:                                                 # 牝系の表が空（試しの小さな表など）
        zB = pd.DataFrame(columns=["wet_T", "wet_D"], dtype=float)
    pos = {(g, p) for g in (3, 4, 5) for p in MAT_POS[g]}
    Pm = P[P["horse_id"].isin(set(horses_t["horse_id"]))]
    Pm = Pm.loc[np.array([(g, p) in pos for g, p in zip(Pm["gen"], Pm["pos"])], dtype=bool)].copy()   # 空の表で列の選びにならないように
    Pm["w"] = 0.5 ** Pm["gen"]
    Pm = Pm[Pm["anc"].isin(zB.index)]
    for d in ("wet_T", "wet_D"):
        Pm[d] = Pm["anc"].map(zB[d]) * Pm["w"]
    agg = Pm.groupby("horse_id")[["wet_T", "wet_D", "w"]].sum()
    out = pd.DataFrame({"horse_id": horses_t["horse_id"].to_numpy()})
    for d in VEC_F:
        out[f"f_{d}"] = (f[d] - mu_s[d]) / sd_s[d]
    out["f_rel"] = f["runs"] / (f["runs"] + K)
    out["m_surf"] = (m["surf"] - mu_b["surf"]) / sd_b["surf"]
    for d in ("wet_T", "wet_D"):
        out[f"b_{d}"] = out["horse_id"].map(agg[d] / agg["w"]).to_numpy()
    out["year"] = t
    return out


def conditions(R: pd.DataFrame, F: pd.DataFrame, t: int) -> pd.DataFrame:
    """年 t のレースの条件（距離の平均と sd・コースの出現率は t 年より前のレースから）。"""
    prev = F[F["date"] < pd.Timestamp(f"{t}-01-01")].drop_duplicates("race_id")
    lx = np.log(prev["distance"])
    mu = lx.groupby(prev["surf"]).mean()
    sd = lx.groupby(prev["surf"]).std()
    fl_prev = flags(prev.reset_index(drop=True))
    rate = {c: pd.Series(fl_prev[c].astype(float)).groupby(prev["surf"].to_numpy()).mean() for c in COURSE}
    fl = flags(R)
    s = R["surf"].to_numpy()
    c = pd.DataFrame(index=R.index)
    c["dist"] = (np.log(R["distance"]) - R["surf"].map(mu)) / R["surf"].map(sd)
    c["surf"] = np.where(s == "D", 1.0, -1.0)
    c["wet_T"] = fl["wet_T"].astype(float)
    c["wet_D"] = fl["wet_D"].astype(float)
    for k in COURSE:
        c[k] = fl[k].astype(float) - R["surf"].map(rate[k]).fillna(0.0).to_numpy()
    return c


def columns(R: pd.DataFrame, c: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    """合い具合の 9 列（R の行の順）。H は馬ごとの血統の値（その年）。"""
    h = R[["horse_id"]].merge(H, on="horse_id", how="left")
    out = pd.DataFrame(index=R.index)
    out["apt_f_dist"] = c["dist"].to_numpy() * h["f_dist"].to_numpy()
    out["apt_f_surf"] = c["surf"].to_numpy() * h["f_surf"].to_numpy()
    out["apt_f_wetT"] = c["wet_T"].to_numpy() * h["f_wet_T"].to_numpy()
    out["apt_f_wetD"] = c["wet_D"].to_numpy() * h["f_wet_D"].to_numpy()
    out["apt_f_course"] = sum(c[k].to_numpy() * h[f"f_{k}"].to_numpy() for k in COURSE)
    out["apt_f_rel"] = h["f_rel"].to_numpy()
    out["apt_b_wetT"] = c["wet_T"].to_numpy() * h["b_wet_T"].to_numpy()
    out["apt_b_wetD"] = c["wet_D"].to_numpy() * h["b_wet_D"].to_numpy()
    out["apt_m_surf"] = c["surf"].to_numpy() * h["m_surf"].to_numpy()
    return out


def permute_block(H: pd.DataFrame, seed: int) -> pd.DataFrame:
    """偽: 馬の単位で、血統の塊（父・(B)・母父の値と確かさ）を生年の中で入れ替える。"""
    rng = np.random.default_rng(seed)
    H = H.reset_index(drop=True)
    by = H["horse_id"].str[:4].to_numpy()
    perm = np.arange(len(H))
    for ii in pd.Series(np.arange(len(H))).groupby(by).indices.values():
        perm[ii] = ii[rng.permutation(len(ii))]
    vals = [c for c in H.columns if c not in ("horse_id", "year")]
    G = H.copy()
    G[vals] = H[vals].to_numpy()[perm]
    return G


def build(F: pd.DataFrame, P: pd.DataFrame, parent: dict, years=YEARS, null_seeds=NULL_SEEDS) -> pd.DataFrame:
    parts = []
    for t in years:
        R = F[F["date"].dt.year == t].reset_index(drop=True)
        if not len(R):
            continue
        hz = R.drop_duplicates("horse_id")[["horse_id", "sire", "ds", "ss", "L4", "bms_ss", "bms_L4"]]
        H = year_vectors(F, t, P, parent, hz)
        c = conditions(R, F, t)
        out = R[["race_id", "horse_id"]].copy()
        cols = columns(R, c, H)
        for k in COLS:
            out[k] = cols[k].to_numpy(np.float32)
        for s in null_seeds:
            nc = columns(R, c, permute_block(H, s * 10000 + t))
            for k in COLS:
                out[f"{k}__null{s}"] = nc[k].to_numpy(np.float32)
        parts.append(out)
        print("year", t, len(out), out[COLS].notna().mean().round(3).to_dict(), flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    common.open_token("select", "megu_cols_v1", "血統ベクトルの合い具合の列（年ごとの切り口・as-of・開発の行 2016〜24。成績は数えない）")
    F = runs()
    F["L4"] = F["L4"].fillna("")
    P, parent = pedigree_links(set(F["horse_id"]))
    X = build(F, P, parent)
    X = X.drop_duplicates(["race_id", "horse_id"])
    dest = common.OUT / "megu_cols.parquet"
    X.to_parquet(dest, index=False)
    print("megu_cols", dest, len(X), flush=True)
    data.finish_all(str(dest))


if __name__ == "__main__":
    main()
