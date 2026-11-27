"""同じ母の初仔と 2 番仔の A/E の比（log.md「同じ母の初仔と 2 番仔の比べ」・記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.foal_pair
"""
import pandas as pd, numpy as np
from src.research.pedigree_lab import data
from src.research.pedigree_lab.dam_age import birth_year
data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "foal_pair", "同じ母の初仔と 2 番仔の A/E の比（記述）"))
ped = data.load_ped()[["horse_id", "dam"]]
ped["by"] = birth_year(ped["horse_id"]); ped["dby"] = birth_year(ped["dam"])
ped = ped[ped["dam"].fillna("").ne("") & (ped["dby"] >= 2013) & ped["by"].notna()].sort_values(["dam", "by", "horse_id"])
ped["ord"] = ped.groupby("dam").cumcount() + 1
X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "split", "win", "rage"])
P = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
T = X[X.split.isin([0, 1])].merge(P, on=["race_id", "horse_id"]).merge(ped[["horse_id", "dam", "ord"]], on="horse_id")
T = T[T.ord.isin([1, 2])]
ok = T.groupby("dam")["ord"].nunique(); T = T[T.dam.isin(ok[ok == 2].index)]
def run(T, lab):
    g = T.groupby(["dam", "ord"]).agg(w=("win", "sum"), e=("p_A", "sum")).unstack(fill_value=0)
    W1, E1, W2, E2 = (g[("w", 1)].to_numpy(), g[("e", 1)].to_numpy(), g[("w", 2)].to_numpy(), g[("e", 2)].to_numpy())
    r = (W1.sum() / E1.sum()) / (W2.sum() / E2.sum())
    rng = np.random.default_rng(3); n = len(W1); bs = []
    for _ in range(1000):
        i = rng.integers(0, n, n); bs.append((W1[i].sum() / E1[i].sum()) / (W2[i].sum() / E2[i].sum()))
    print(lab, "母", n, "行", len(T), "初仔 A/E", round(W1.sum() / E1.sum(), 3), "2 番仔 A/E", round(W2.sum() / E2.sum(), 3), "比", round(r, 3), np.percentile(bs, [2.5, 97.5]).round(3))
run(T, "全戦")
T2 = T[T.rage == "A2"]
ok2 = T2.groupby("dam")["ord"].nunique(); T2 = T2[T2.dam.isin(ok2[ok2 == 2].index)]   # 2 歳戦でも両方の仔がいる母だけ（critic）
run(T2, "2 歳戦")
data.finish_all("foal_pair")
