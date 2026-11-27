"""血統の似た馬の近傍（kNN）の as-of の列（log.md「調べ M」。開発の行・時間の順）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.pedigree_knn

- ベクトル: 開発の行に出る馬の 2〜5 代の祖先ごとに 2^−代 を足した値（2 頭以上の馬に出る祖先だけ。父・母は入れない）。近さはコサイン。
  限界: 祖先の語彙と近い馬の候補は 2024 年までの開発の行の全体で決めるので、母集団の先読みがある（結果は上限の値として読む。2 号）。
  近い馬にその日より前の走りが無ければ、その席は 0 件のまま（値は 0 の側へ寄る）。
- 近い馬: 各馬の近い 50 頭（自分と、同じ母の子と、同じ父の子は除く）。近さの表は血統だけで作る（着順を使わない）。
- 列: 近い馬たちの、その日より前の、同じ芝ダ × 距離の帯（knn_sd）・同じ芝ダ（knn_s）の ex_top3 の和を、近さで重み付けた走り数で割り、
  擬似件数 50 走で 0 へ縮めた値。
- (A) ＋ 2 列を時間の順（学習は 2017 年から・試す年は 2019〜24 年）で当て、行ごとの予測を pedigree_knn_wf.parquet に残す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.public_rules import parse

KN, K = 50, 50.0
NEW = ["knn_sd", "knn_s"]


def neighbors(H: pd.Series, A: pd.DataFrame, dam: pd.Series, sire: pd.Series) -> pd.DataFrame:
    A = A[A["horse_id"].isin(set(H)) & (A["gen"] >= 2)]                 # 1 代目（父・母）を除く（1 号・2 号）
    cnt = A.groupby("anc")["horse_id"].nunique()
    A = A[A["anc"].map(cnt) >= 2]
    hi = {h: i for i, h in enumerate(H)}
    ai = {a: j for j, a in enumerate(A["anc"].unique())}
    M = sparse.coo_matrix(((0.5 ** A["gen"]).to_numpy(), (A["horse_id"].map(hi).to_numpy(), A["anc"].map(ai).to_numpy())),
                          shape=(len(H), len(ai))).tocsr()
    nrm = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    M = sparse.diags(1 / np.maximum(nrm, 1e-12)) @ M
    MT = M.T.tocsc()
    damv = dam.reindex(H.to_numpy()).astype(object).fillna("").to_numpy()
    sirev = sire.reindex(H.to_numpy()).astype(object).fillna("").to_numpy()
    sib = pd.Series(np.arange(len(H))).groupby(damv).apply(lambda x: x.to_numpy()).to_dict()
    half = pd.Series(np.arange(len(H))).groupby(sirev).apply(lambda x: x.to_numpy()).to_dict()
    hs, ns, ws = [], [], []
    for s0 in range(0, len(H), 500):                                         # 500 行ずつ（メモリ。critic）
        S = (M[s0:s0 + 500] @ MT).toarray().astype(np.float32)
        for r in range(S.shape[0]):
            i = s0 + r
            S[r, i] = -1
            if damv[i]:
                S[r, sib[damv[i]]] = -1                                    # 同じ母の子を除く
            if sirev[i]:
                S[r, half[sirev[i]]] = -1                                  # 同じ父の子も除く（父の側の 2〜5 代が丸ごと一致して近さを占めるため。critic）
        idx = np.argpartition(-S, KN, axis=1)[:, :KN]
        w = np.take_along_axis(S, idx, axis=1)
        ok = w > 0
        rr = np.repeat(np.arange(s0, s0 + S.shape[0]), KN).reshape(-1, KN)
        hs.append(rr[ok]); ns.append(idx[ok]); ws.append(w[ok])
    Hv = H.to_numpy()
    return pd.DataFrame({"horse_id": Hv[np.concatenate(hs)], "nb": Hv[np.concatenate(ns)], "w": np.concatenate(ws)})


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "pedigree_knn", "血統の似た馬の近傍の列の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    H = pd.Series(X["horse_id"].unique())
    ped = data.load_ped()
    A = parse(ped[ped["horse_id"].isin(set(H))])
    U = X.drop_duplicates("horse_id").set_index("horse_id")
    NB = neighbors(H, A, U["dam"], U["sire"])
    print("近い馬の表", len(NB), "近さの平均", round(float(NB["w"].mean()), 3), flush=True)
    R = X[["horse_id", "date", "surf", "dband", "ex_top3"]].copy()
    R["c_sd"] = R["surf"].astype(str) + "|" + R["dband"].astype(str)
    R["c_s"] = R["surf"].astype(str)
    for nm, ck in (("knn_sd", "c_sd"), ("knn_s", "c_s")):
        # 近い馬ごと × 条件ごとの、日ごとの累計（その日を含む）
        G = R.groupby(["horse_id", ck, "date"])["ex_top3"].agg(["sum", "size"]).groupby(level=[0, 1]).cumsum().reset_index()
        G = G.rename(columns={"horse_id": "nb", ck: "cond"}).sort_values("date")
        Q = X[["horse_id", "date"]].assign(cond=R[ck].to_numpy()).reset_index().rename(columns={"index": "row"})
        vals = np.zeros(len(X)); wn = np.zeros(len(X))
        for yy in range(2016, 2025):
            q = Q[Q["date"].dt.year == yy].merge(NB, on="horse_id").sort_values("date")
            m = pd.merge_asof(q, G, on="date", by=["nb", "cond"], allow_exact_matches=False)   # その日より厳密に前
            m[["sum", "size"]] = m[["sum", "size"]].fillna(0.0)
            m["ws"], m["wn"] = m["w"] * m["sum"], m["w"] * m["size"]
            a = m.groupby("row")[["ws", "wn"]].sum()
            vals[a.index.to_numpy()] = a["ws"].to_numpy(); wn[a.index.to_numpy()] = a["wn"].to_numpy()
            print(nm, yy, flush=True)
        X[nm] = vals / (wn + K)
    print("mean", X[NEW].mean().round(4).to_dict(), "sd", X[NEW].std().round(4).to_dict(), flush=True)
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    from src.research.pedigree_lab.walkforward import fit_year
    use = list(dict.fromkeys([c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + NEW))
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    o = X.loc[te, ["race_id", "horse_id"]].copy()
    o["p_TB"] = p[te]
    path = data.OUT / "pedigree_knn_wf.parquet"
    o.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
