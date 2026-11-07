"""ラップ予想 試行3: 直接予測（試行2）に、出走メンバーの追い切りの速さのレース集計を足す。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
from src.scripts.data import pos_model as pm
OUT = "/home/jovyan/work/retrain_out/tp55_2016"
d = pd.read_parquet(f"{OUT}/races_feat.parquet"); d["year"] = d["date"].str[:4].astype(int); d["race_id"] = d["race_id"].astype(str)
o = pm.oikiri_features(); o["year"] = o["race_id"].str[:4]
for c in ("oi_1f", "oi_3f"):
    v = o[c].where(o[c].between(o[c].quantile(0.001), o[c].quantile(0.999)))
    g = o["oi_hill"].astype(str) + "|" + o["year"]
    o[c + "_z"] = (v - v.groupby(g).transform("mean")) / v.groupby(g).transform("std")
agg = o.groupby("race_id").agg(oi1_min=("oi_1f_z", "min"), oi1_mean=("oi_1f_z", "mean"), oi3_min=("oi_3f_z", "min"), oi3_mean=("oi_3f_z", "mean"),
                               oi1_fast=("oi_1f_z", lambda x: (x < -1).sum()), oi_evalA=("oi_eval", lambda x: (x >= 3).sum()), oi_evalB=("oi_eval", lambda x: (x >= 2).sum()),
                               oi_n=("oi_1f_z", "count")).reset_index()
d = d.merge(agg, on="race_id", how="left")
num, cats = lfm.feature_columns(); extra = [c for c in agg.columns if c != "race_id"]
for c in cats: d[c] = d[c].astype("category")
def targets(df, par):
    laps = [json.loads(s) for s in df["laps_json"]]; L = lfm.laps_matrix(laps); P = lfm.par_matrix(df, par); dev = L - P
    n = np.array([len(a) for a in laps])
    return np.array([dev[i, :3].sum() for i in range(len(df))]), np.array([dev[i, n[i]-3:n[i]].sum() for i in range(len(df))])
res = {}
for Y in (2022, 2023, 2024, 2025):
    tr, te = d[d.year < Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    f3tr, l3tr = targets(tr, par); f3te, l3te = targets(te, par)
    out = {}
    for name, ytr, yte in (("f3", f3tr, f3te), ("l3", l3tr, l3te)):
        ok = ~np.isnan(ytr); r = {}
        for tag, cols in (("base", num + cats), ("plus_oikiri", num + cats + extra)):
            ps = [lgb.train(dict(objective="l1", learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                                 lambda_l2=1.0, verbose=-1, num_threads=4, seed=sd), lgb.Dataset(tr.loc[ok, cols], label=ytr[ok]), num_boost_round=1200).predict(te[cols]) for sd in (0, 1, 2)]
            p = np.mean(ps, axis=0); m = ~np.isnan(yte); r[tag] = float(np.mean(np.abs(p[m] - yte[m])))
        out[name] = r
    res[Y] = out; print(Y, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try3_oikiri.json", "w"), indent=1)
