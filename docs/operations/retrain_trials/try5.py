"""ラップ予想 試行5: 前半/後半 3F の直接予測の学習設定（ゆっくり・大きい木・Huber・最近の年を重く）の比較。採否は 2022〜2024 年の平均で決める。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
OUT = "/home/jovyan/work/retrain_out/tp55_2016"
d = pd.read_parquet(f"{OUT}/races_feat.parquet"); d["year"] = d["date"].str[:4].astype(int)
num, cats = lfm.feature_columns(); cols = num + cats
for c in cats: d[c] = d[c].astype("category")
def targets(df, par):
    laps = [json.loads(s) for s in df["laps_json"]]; L = lfm.laps_matrix(laps); P = lfm.par_matrix(df, par); dev = L - P
    n = np.array([len(a) for a in laps])
    return np.array([dev[i, :3].sum() for i in range(len(df))]), np.array([dev[i, n[i]-3:n[i]].sum() for i in range(len(df))])
BASE = dict(objective="l1", learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
CFG = {"base": (BASE, 1200, None), "slow_big": (dict(BASE, learning_rate=0.015, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.7), 3000, None),
       "huber": (dict(BASE, objective="huber", alpha=0.6), 1200, None), "recent_w": (BASE, 1200, 0.85)}
res = {}
for Y in (2022, 2023, 2024, 2025):
    tr, te = d[d.year < Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    f3tr, l3tr = targets(tr, par); f3te, l3te = targets(te, par)
    out = {}
    for name, ytr, yte in (("f3", f3tr, f3te), ("l3", l3tr, l3te)):
        ok = ~np.isnan(ytr); m = ~np.isnan(yte); r = {}
        for tag, (p, rounds, decay) in CFG.items():
            w = None if decay is None else decay ** (Y - 1 - tr.loc[ok, "year"].to_numpy())
            ps = [lgb.train(dict(p, seed=sd), lgb.Dataset(tr.loc[ok, cols], label=ytr[ok], weight=w), num_boost_round=rounds).predict(te[cols]) for sd in (0, 1, 2)]
            r[tag] = float(np.mean(np.abs(np.mean(ps, axis=0)[m] - yte[m])))
        out[name] = r
    res[Y] = out; print(Y, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try5_cfg.json", "w"), indent=1)
for name in ("f3", "l3"):
    avg = {t: np.mean([res[y][name][t] for y in (2022, 2023, 2024)]) for t in CFG}
    best = min(avg, key=avg.get); print(name, "2022〜2024 平均", {k: round(v, 4) for k, v in avg.items()}, "→ 採用", best, "2025", round(res[2025][name][best], 4))
