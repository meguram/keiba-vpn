"""追走難度 試行4: 試行3 の plus_lap に、ラップ予想の直接予測（前半/後半 3F・前後半の差）と、それと位置取りの組を足す。"""
import json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
os.environ["TP54_OUT"] = "/home/jovyan/work/retrain_out/tp54_2016"
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.scripts.data import tp54_model as m
from src.pipeline.models import tracking_difficulty_v2 as td2
h = m.load(); X = td2.make_features(h); ok = h["pf_early"].notna()
h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
lap = pd.read_parquet("/home/jovyan/work/retrain_out/tp55_2016/groups/LAP.parquet"); lap["race_id"], lap["horse_id"] = lap["race_id"].astype(str), lap["horse_id"].astype(str)
L = h[["race_id", "horse_id"]].merge(lap.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left").drop(columns=["race_id", "horse_id"])
L.index = h.index
po = pd.read_parquet("/home/jovyan/work/keiba-vpn/data/local/modeling/pos_model/pos_oof.parquet")
po["race_id"], po["horse_id"] = po["race_id"].astype(str), po["horse_id"].astype(str)
g = po.groupby("race_id")["pos_pred"]
po["pp_rank"] = g.rank(pct=True); po["pp_ahead"] = g.rank(method="min") - 1; po["pp_front_n"] = g.transform(lambda x: (x < 0.15).sum())
po["pp_gap_mean"] = po["pos_pred"] - g.transform("mean")
P = h[["race_id", "horse_id"]].merge(po[["race_id", "horse_id", "pp_rank", "pp_ahead", "pp_front_n", "pp_gap_mean"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left").drop(columns=["race_id", "horse_id"])
P.index = h.index
ld = pd.read_parquet("/home/jovyan/work/retrain_out/tp55_2016/groups/LAPD.parquet"); ld["race_id"] = ld["race_id"].astype(str)
D = h[["race_id"]].merge(ld.drop_duplicates("race_id"), on="race_id", how="left").drop(columns=["race_id"]); D.index = h.index
pe = h["pf_early"].astype(float)
D["ld_front_x_f3"] = (1 - pe) * D["ld_f3"]          # 前に行く馬ほど、前半が速いと負荷が大きい
D["ld_rear_x_q"] = pe * D["ld_q"]                   # 後ろの馬ほど、後半が速い（前後半の差が大きい）と不利
B = dict(m.PARAMS, learning_rate=0.015, num_leaves=63, min_child_samples=150)
VAR = {"plus_lap": pd.concat([X, L], axis=1), "plus_lap_direct": pd.concat([X, L, D], axis=1), "plus_lap_direct_pos": pd.concat([X, L, D, P], axis=1)}
res = {k: {} for k in VAR}
for yr in (2023, 2024, 2025):
    coef = m.baseline_fit(h, ok & (h["year"] < yr)); T, y = m.make_target(h, coef)
    tr = ok & y.notna() & (h["year"] < yr); te = ok & y.notna() & (h["year"] == yr)
    for k, XX in VAR.items():
        pr = np.mean([lgb.train(dict(B, seed=s), lgb.Dataset(XX[tr], y[tr].to_numpy()), num_boost_round=1500).predict(XX[te]) for s in (0, 1, 2)], axis=0)
        res[k][yr] = m.corr(pr, y[te])
    print(yr, {k: round(v[yr], 4) for k, v in res.items()}, flush=True)
for k, v in res.items(): print(k, "平均", round(np.mean(list(v.values())), 4), "最小", round(min(v.values()), 4))
json.dump({k: {str(y): c for y, c in v.items()} for k, v in res.items()}, open("/home/jovyan/work/retrain_out/tp54_2016/try4_lapdirect.json", "w"), indent=1)
