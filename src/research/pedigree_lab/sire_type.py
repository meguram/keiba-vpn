"""データ版の国別の型（種牡馬を産駒の成績の形で群に分ける。log.md「調べ F」。開発の行・時間の順・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.sire_type [--null 1]

- 年 Y ごとに、Y より前の開発の行（split 0・1）で産駒の走りが 100 以上の父について、ダートの走りの割合・3 着内の走りの平均の距離・
  ダート − 芝の 3 着内率・芝の中の道悪（稍重以上）− 良の 3 着内率・2 歳戦 − 3 歳以上の 3 着内率（差は擬似件数 50 走で 0 へ縮める）を作り、標準化して k-means（4 群・種 3）で分ける。
  群の番号は前年の中心へのハンガリー法の対応でそろえる（始めの年はダートの割合の順・n_init 20。2 号）。走りが 100 未満の父は父の父の群を受け継ぐ（無ければ 4 = 不明）。母父も同じ表で付ける。
- 列: ty_s0〜ty_s4（父の群の印）・ty_b0〜ty_b4（母父の群の印）。
- 帰無（--null）: 年ごとに、父 → 群の対応を、群のある父の間で入れ替える（雑音の列の費用の記述だけ。判定は版 − (A)。2 号）。
- 予測の parquet には、父の群を父の父から受け継いだ行の印（_inh）を残す（受け継いだ行の版 − (A) を 2 本目の線として読む）。
- (A) ＋ 10 列を時間の順（学習は 2017 年から・試す年は 2019〜24 年）で当て、行ごとの予測を sire_type_wf{tag}.parquet に残す。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.research.pedigree_lab import ablate, data

NEW = [f"ty_s{i}" for i in range(5)] + [f"ty_b{i}" for i in range(5)]


def profiles(P: pd.DataFrame) -> pd.DataFrame:
    """父ごとの 5 つの値。差の値は件数に応じて 0 へ縮める（擬似件数 50 走。1 号）。"""
    # 3 着内は頭数の差を外した ex_top3（top3 − min(3, n)/n）で比べる（critic）
    P = P.assign(_d=P["surf"].eq("D").astype(float), _t3=P["ex_top3"].astype(float),
                 _dist3=np.where(P["top3"] == 1, pd.to_numeric(P["distance"], errors="coerce"), np.nan),
                 _wet=P["going"].ne("G").astype(float), _y2=P["rage"].eq("A2").astype(float))
    g = P.groupby("sire")
    out = pd.DataFrame({"n": g.size(), "dirt": g["_d"].mean(), "dist3": g["_dist3"].mean()})

    def diff(sub, col):
        t = sub.groupby(["sire", col])["_t3"].agg(["sum", "size"]).unstack()
        r1 = t[("sum", 1.0)] / t[("size", 1.0)]
        r0 = t[("sum", 0.0)] / t[("size", 0.0)]
        n = np.minimum(t[("size", 1.0)], t[("size", 0.0)])
        # 片方の側に走りが無い父は、差を 0 にする（NaN で群から落とさない。critic）
        return ((r1 - r0) * n / (n + 50)).where(n > 0, 0.0).fillna(0.0)
    out["dt"] = diff(P, "_d")                                        # ダート − 芝の 3 着内率
    out["wg"] = diff(P[P["_d"] == 0], "_wet")                        # 芝の中だけの道悪 − 良（ダートの道悪は向きが逆なので混ぜない。1 号）
    out["early"] = diff(P, "_y2")                                    # 2 歳戦 − 3 歳以上の 3 着内率（早熟の度合い。1 号）
    return out


def assign(X: pd.DataFrame, null_seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """年ごとの群。番号は前年の中心へのハンガリー法の対応でそろえる（始めの年はダートの割合の順。2 号）。"""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    yr = X["date"].dt.year.to_numpy()
    ss_of = X.drop_duplicates("sire").set_index("sire")["ss"]
    ss_of = ss_of[ss_of.fillna("").ne("")]
    cs, cb, inh = np.full(len(X), 4), np.full(len(X), 4), np.zeros(len(X), bool)
    rng = np.random.default_rng(null_seed) if null_seed is not None else None
    prev_c, prev_lab = None, None
    cols = ["dirt", "dist3", "dt", "wg", "early"]
    for Y in range(2017, 2025):
        P = X[(yr < Y) & X["split"].isin([0, 1])]
        pr = profiles(P)
        pr = pr[pr["n"] >= 100].dropna()
        if len(pr) < 8:
            continue
        mu, sd = pr[cols].mean(), pr[cols].std()
        Z = ((pr[cols] - mu) / sd).to_numpy()
        km = KMeans(n_clusters=4, n_init=20, random_state=3).fit(Z)
        cen = km.cluster_centers_ * sd.to_numpy() + mu.to_numpy()          # 元の尺度の中心（年をまたいで比べる）
        if prev_c is None:
            perm = np.argsort(np.argsort(cen[:, 0]))                       # 始めの年: ダートの割合の順
        else:
            zc, zp = (cen - mu.to_numpy()) / sd.to_numpy(), (prev_c - mu.to_numpy()) / sd.to_numpy()
            cost = ((zc[:, None, :] - zp[None, :, :]) ** 2).sum(-1)
            r, c = linear_sum_assignment(cost)
            perm = np.empty(4, int); perm[r] = c
        lab = pd.Series(perm[km.labels_], index=pr.index)
        newc = np.zeros_like(cen); newc[perm] = cen
        if prev_lab is not None:
            k = lab.index.intersection(prev_lab.index)
            print(Y, "父", len(pr), "前年と同じ番号に残る割合", round(float((lab[k] == prev_lab[k]).mean()), 3),
                  "ARI", round(float(adjusted_rand_score(prev_lab[k], lab[k])), 3), flush=True)
        else:
            print(Y, "父", len(pr), flush=True)
        prev_c, prev_lab = newc, lab.copy()
        if rng is not None:
            lab = pd.Series(rng.permutation(lab.to_numpy()), index=lab.index)   # 帰無（記述だけ）: 父 → 群の対応を入れ替える
        mp = {k_: v_ for k_, v_ in lab.to_dict().items() if k_ != ""}      # 空の札は不明に寄せる（critic）
        m = yr == Y
        sir = X.loc[m, "sire"]
        own = sir.map(mp)
        anc = sir.map(lambda s_: mp.get(ss_of.get(s_, None), np.nan))
        cs[m] = own.fillna(anc).fillna(4).astype(int).to_numpy()            # 走り 100 未満の父は、父の父の群を受け継ぐ
        inh[m] = (own.isna() & anc.notna()).to_numpy()
        bm = X.loc[m, "ds"]
        cb[m] = bm.map(mp).fillna(bm.map(lambda s_: mp.get(ss_of.get(s_, None), np.nan))).fillna(4).astype(int).to_numpy()
        print("   父の不明", round(float((cs[m] == 4).mean()), 3), "受け継ぎ", round(float(inh[m].mean()), 3), "母父の不明", round(float((cb[m] == 4).mean()), 3), flush=True)
    return cs, cb, inh


def main() -> None:
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "sire_type" + ("_null" if null_seed is not None else ""),
                                   "データ版の国別の型の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    cs, cb, inh = assign(X, null_seed)
    X["_inh"] = inh
    for i in range(5):
        X[f"ty_s{i}"] = (cs == i).astype(float)
        X[f"ty_b{i}"] = (cb == i).astype(float)
    print("父の群の割合（2019 年以降）", pd.Series(cs[X["date"].dt.year >= 2019]).value_counts(normalize=True).round(3).sort_index().to_dict(), flush=True)
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    from src.research.pedigree_lab.walkforward import fit_year
    use = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + NEW
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    o = X.loc[te, ["race_id", "horse_id", "_inh"]].copy()
    o["p_TB"] = p[te]
    path = data.OUT / f"sire_type_wf{'' if null_seed is None else f'_null{null_seed}'}.parquet"
    o.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
