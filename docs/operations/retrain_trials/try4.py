"""ラップ予想 試行4: 直接予測に、位置取り予想（学習モデル）の予測から作る先頭争いのレース集計を足す。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
from src.scripts.data import pos_model as pm
OUT = "/home/jovyan/work/retrain_out/tp55_2016"
d = pd.read_parquet(f"{OUT}/races_feat.parquet"); d["year"] = d["date"].str[:4].astype(int); d["race_id"] = d["race_id"].astype(str)
po = pd.read_parquet(pm.OUT / "pos_oof.parquet")
po["race_id"] = po["race_id"].astype(str)
def _agg(x):
    v = np.sort(x.to_numpy())
    return pd.Series({"pp_min": v[0], "pp_2nd": v[1] if len(v) > 1 else np.nan, "pp_gap12": (v[1] - v[0]) if len(v) > 1 else np.nan,
                      "pp_front": (v < 0.15).sum(), "pp_frontstalk": (v < 0.3).sum(), "pp_mean": v.mean(), "pp_std": v.std(), "pp_top3": v[:3].mean()})
agg = po.groupby("race_id")["pos_pred"].apply(_agg).unstack().reset_index()
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
        for tag, cols in (("base", num + cats), ("plus_pos", num + cats + extra)):
            ps = [lgb.train(dict(objective="l1", learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                                 lambda_l2=1.0, verbose=-1, num_threads=4, seed=sd), lgb.Dataset(tr.loc[ok, cols], label=ytr[ok]), num_boost_round=1200).predict(te[cols]) for sd in (0, 1, 2)]
            p = np.mean(ps, axis=0); m = ~np.isnan(yte); r[tag] = float(np.mean(np.abs(p[m] - yte[m])))
        out[name] = r
    res[Y] = out; print(Y, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try4_pos.json", "w"), indent=1)
