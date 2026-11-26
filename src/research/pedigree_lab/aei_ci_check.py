"""組の残差の持続を、母の質を引いた値でも見る（追加の記述・結果を見た後に足した 1 本。log.md「平均獲得賞金で見た…」）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.aei_ci_check

- 馬ごとの値 = その馬の b の平均 − 同じ母の、ほかの父の子の b の平均（1 頭 1 票・CI 型。ほかの父の子のいる馬だけ）。
- この値で aei_list と同じ組の残差（擬似件数 10 頭・見込みは足し算）を作り、生年の群の 2 期で Spearman と帰無（200 回）を出す。
- 母の質の偏り（生産者の選び方）が組の持続の元なら、この値では持続が消える（1 号の予想: 差は半分以下）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research.pedigree_lab import data
from src.research.pedigree_lab.aei_list import nick_table, perm_within, value


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "aei_ci_check", "組の残差の持続を母の質を引いた値で見る（記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "sire", "ds", "dam"])
    C = pd.read_parquet(data.OUT / "aei_cols.parquet", columns=["race_id", "horse_id", "b"])
    T = X.merge(C, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & (T["date"] >= "2018-01-01") & (T["date"] <= data.DEV_END) & T["b"].notna()]
    T = T[T["sire"].fillna("").ne("") & T["ds"].fillna("").ne("") & T["dam"].fillna("").ne("")]
    H = T.groupby("horse_id").agg(b=("b", "mean"), n=("b", "size"), sire=("sire", "first"), ds=("ds", "first"), dam=("dam", "first")).reset_index()
    H["by"] = H["horse_id"].str[:4]
    H["b"] = H["b"] / H["b"].mean()
    byi0 = H["by"].astype(int)
    H["per"] = np.where(byi0.between(2014, 2016), 1, np.where(byi0.between(2017, 2020), 2, 0))
    H = H[H["per"] > 0].copy()
    # 同じ母の、ほかの父の子の平均は、同じ期（生年の群）の中だけで取る（2 期にまたがる兄弟の共通のぶれを入れない。critic）
    g = H.groupby(["per", "dam"])["b"].agg(["sum", "size"])
    gs = H.groupby(["per", "dam", "sire"])["b"].agg(["sum", "size"])
    k1 = pd.MultiIndex.from_frame(H[["per", "dam"]]); k2 = pd.MultiIndex.from_frame(H[["per", "dam", "sire"]])
    ds_ = k1.map(g["sum"]).to_numpy() - k2.map(gs["sum"]).to_numpy()
    dn_ = k1.map(g["size"]).to_numpy() - k2.map(gs["size"]).to_numpy()
    H = H[dn_ > 0].copy()
    H["b_raw"] = H["b"]
    H["b"] = H["b"] - (ds_[dn_ > 0] / dn_[dn_ > 0]) + 1.0          # 母の質を引いた値（1 が平均になるように 1 を足す）
    byi = H["by"].astype(int)

    def rho(Z):
        out = []
        for P in (Z[byi.loc[Z.index].between(2014, 2016)], Z[byi.loc[Z.index].between(2017, 2020)]):
            P = P.assign(b=P["b"] / P["b"].mean())                    # 期ごとに中心を 1 に合わせる（critic）
            out.append(nick_table(P, value(P, "sire")["v"], value(P, "ds")["v"]))
        k = out[0][out[0]["nh"] >= 5].index.intersection(out[1][out[1]["nh"] >= 5].index)
        return spearmanr(out[0].loc[k, "resid"], out[1].loc[k, "resid"])[0], len(k)
    r_raw, n_raw = rho(H.assign(b=H["b_raw"]))                       # 同じ部分集合で、引く前の値（critic）
    r, n = rho(H)
    rng = np.random.default_rng(3)
    nul, nul_raw = [], []
    for _ in range(200):
        Hn = H.copy()
        Hn["ds"] = perm_within((Hn["by"] + "|" + Hn["sire"]).to_numpy(), Hn["ds"].to_numpy(), rng)
        nul.append(rho(Hn)[0])
        nul_raw.append(rho(Hn.assign(b=Hn["b_raw"]))[0])
    res = {"馬": int(len(H)), "引く前の rho（同じ馬）": round(float(r_raw), 3), "rho": round(float(r), 3), "数": n, "帰無の 95% 点": round(float(np.nanpercentile(nul, 95)), 3),
           "帰無の平均": round(float(np.nanmean(nul)), 3), "帰無が上回る割合": round(float(np.mean(np.array(nul) >= r)), 3),
           "引く前の帰無の 95% 点": round(float(np.nanpercentile(nul_raw, 95)), 3), "引く前の帰無が上回る割合": round(float(np.mean(np.array(nul_raw) >= r_raw)), 3)}
    print(res, flush=True)
    path = data.OUT / "aei_ci_check.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
