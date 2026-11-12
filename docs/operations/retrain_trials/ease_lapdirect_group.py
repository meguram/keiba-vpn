"""ラップ予想の直接予測（前半/後半 3F・前後半の差。Huber、乱数 3 本）を、LAP 群と同じく年ごとに『その年を除く全年』で学習して作る（2016〜2025）。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
OUT = "/home/jovyan/work/retrain_out/tp55_2016"
d = pd.read_parquet(f"{OUT}/races_feat.parquet"); d["year"] = d["date"].str[:4].astype(int); d = d[d.year <= 2025].reset_index(drop=True)
num, cats = lfm.feature_columns(); cols = num + cats
for c in cats: d[c] = d[c].astype("category")
RP = dict(objective="huber", alpha=0.6, learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
def stats(df, par):
    laps = [json.loads(s) for s in df["laps_json"]]; L = lfm.laps_matrix(laps); P = lfm.par_matrix(df, par); dev = L - P
    n = np.array([len(a) for a in laps]); st = lfm.scenario_stats(dev, n)
    return np.array([dev[i, :3].sum() for i in range(len(df))]), np.array([dev[i, n[i]-3:n[i]].sum() for i in range(len(df))]), st[:, 2] - st[:, 0]
parts = []
for Y in range(2016, 2026):
    tr, te = d[d.year != Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    f3, l3, q = stats(tr, par)
    o = {"race_id": te["race_id"].astype(str).values}
    for name, t in (("ld_f3", f3), ("ld_l3", l3), ("ld_q", q)):
        ok = ~np.isnan(t)
        o[name] = np.mean([lgb.train(dict(RP, seed=s), lgb.Dataset(tr.loc[ok, cols], label=t[ok]), num_boost_round=1200).predict(te[cols]) for s in (0, 1, 2)], axis=0)
    parts.append(pd.DataFrame(o)); print(Y, len(te), flush=True)
pd.concat(parts).to_parquet(f"{OUT}/groups/LAPD.parquet", index=False); print("saved")
