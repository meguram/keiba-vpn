"""組の残差の持続を、母の質を引いた値でも見る（追加の記述・結果を見た後に足した 1 本。log.md「平均獲得賞金で見た…」）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.aei_ci_check [--split orig|parity|three]

- 馬ごとの値 = その馬の b の平均 − 同じ母の、ほかの父の子の b の平均（1 頭 1 票・CI 型。ほかの父の子のいる馬だけ）。
- この値で aei_list と同じ組の残差（擬似件数 10 頭・見込みは足し算）を作り、生年の群の 2 期で Spearman と帰無（200 回）を出す。
- 母の質の偏り（生産者の選び方）が組の持続の元なら、この値では持続が消える（1 号の予想: 差は半分以下）。
"""
from __future__ import annotations

import json
import sys

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
    if "--log" in sys.argv:
        H["b"] = np.log1p(H["b"])                                    # 重い裾を抑えた版（1 号）
    H["b"] = H["b"] / H["b"].mean()
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "orig"
    assert split in ("orig", "parity", "three", "dam"), split
    byi0 = H["by"].astype(int)
    # 期の分け方（log.md「延長の続き」の調べ A）: orig = 2014〜16・2017〜20 年生まれ、parity = 偶数年・奇数年（2014〜20）、
    # three = 2014〜15・2016〜17・2018〜20（隣り合う 2 組の ρ の平均）
    if split == "orig":
        per = np.where(byi0.between(2014, 2016), 1, np.where(byi0.between(2017, 2020), 2, 0)); pairs = [(1, 2)]
    elif split == "parity":
        per = np.where(byi0.between(2014, 2020), 1 + (byi0 % 2), 0); pairs = [(1, 2)]
    elif split == "dam":
        # 母の id で分ける（同じ母の子はすべて同じ半分。全兄弟が両方に入らない。1 号の主の案）
        dl = H["dam"].str[-1].map(lambda c: int(c, 16) if c in "0123456789abcdef" else 0)
        per = np.where(byi0.between(2014, 2020), 1 + (dl.to_numpy() % 2), 0); pairs = [(1, 2)]
    else:
        per = np.select([byi0.between(2014, 2015), byi0.between(2016, 2017), byi0.between(2018, 2020)], [1, 2, 3], 0); pairs = [(1, 2), (2, 3)]
    H["per"] = per
    H = H[H["per"] > 0].copy()
    nov = "--nooverlap" in sys.argv
    if nov:
        # 両方の期に子がいる母を落とす（全兄弟が期をまたいで組の残差を共有する分を除く。2 号）
        nper = H.groupby("dam")["per"].nunique()
        H = H[H["dam"].map(nper) == 1].copy()
    # 同じ母の、ほかの父の子の平均は、同じ期（生年の群）の中だけで取る（2 期にまたがる兄弟の共通のぶれを入れない。critic）
    g = H.groupby(["per", "dam"])["b"].agg(["sum", "size"])
    gs = H.groupby(["per", "dam", "sire"])["b"].agg(["sum", "size"])
    k1 = pd.MultiIndex.from_frame(H[["per", "dam"]]); k2 = pd.MultiIndex.from_frame(H[["per", "dam", "sire"]])
    ds_ = k1.map(g["sum"]).to_numpy() - k2.map(gs["sum"]).to_numpy()
    dn_ = k1.map(g["size"]).to_numpy() - k2.map(gs["size"]).to_numpy()
    H = H[dn_ > 0].copy()
    H["b_raw"] = H["b"]
    H["b"] = H["b"] - (ds_[dn_ > 0] / dn_[dn_ > 0]) + 1.0          # 母の質を引いた値（1 が平均になるように 1 を足す）

    def rho(Z):
        tabs = {}
        for k_ in sorted(set(Z["per"])):
            P = Z[Z["per"] == k_]
            P = P.assign(b=P["b"] / P["b"].mean())                    # 期ごとに中心を 1 に合わせる（critic）
            tabs[k_] = nick_table(P, value(P, "sire")["v"], value(P, "ds")["v"])
        rs, ns = [], []
        for x, y in pairs:
            k = tabs[x][tabs[x]["nh"] >= 5].index.intersection(tabs[y][tabs[y]["nh"] >= 5].index)
            rs.append(spearmanr(tabs[x].loc[k, "resid"], tabs[y].loc[k, "resid"])[0] if len(k) >= 3 else np.nan); ns.append(len(k))
        return float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan, int(min(ns))     # 組の数は対ごとの最小（3 期で 2 重に数えない。critic）
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
    lg = "--log" in sys.argv
    res["分け方"] = split + ("・母が重ならない" if nov else "") + ("・log1p" if lg else "")
    path = data.OUT / f"aei_ci_check{'' if split == 'orig' else '_' + split}{'_nov' if nov else ''}{'_log' if lg else ''}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
