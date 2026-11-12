"""ラップ予想 試行10: 前半/後半 3F を、区間ごとの直接予測（Huber）の和でも作り、合計の直接予測と混ぜる。採否は 2022〜2024 年の平均。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
OUT = "/home/jovyan/work/retrain_out/tp55_2016"
d = pd.read_parquet(f"{OUT}/races_feat.parquet"); d["year"] = d["date"].str[:4].astype(int)
num, cats = lfm.feature_columns(); cols = num + cats
for c in cats: d[c] = d[c].astype("category")
RP = dict(objective="huber", alpha=0.6, learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
def devm(df, par):
    laps = [json.loads(s) for s in df["laps_json"]]; L = lfm.laps_matrix(laps); P = lfm.par_matrix(df, par); n = np.array([len(a) for a in laps])
    dev = L - P
    first = dev[:, :3]
    last = np.stack([dev[np.arange(len(df)), np.clip(n - 3 + k, 0, None)] for k in range(3)], 1)
    return first, last
def fit_pred(X, y, Xt):
    ok = ~np.isnan(y)
    return np.mean([lgb.train(dict(RP, seed=s), lgb.Dataset(X[ok], label=y[ok]), num_boost_round=1200).predict(Xt) for s in (0, 1, 2)], axis=0)
res = {}
for Y in (2022, 2023, 2024, 2025):
    tr, te = d[d.year < Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    Ftr, Ltr = devm(tr, par); Fte, Lte = devm(te, par)
    out = {}
    for name, A, B in (("f3", Ftr, Fte), ("l3", Ltr, Lte)):
        ysum, tsum = A.sum(1), B.sum(1)
        p_sum = fit_pred(tr[cols], ysum, te[cols])
        p_lap = sum(fit_pred(tr[cols], A[:, k], te[cols]) for k in range(3))
        m = ~np.isnan(tsum); r = {"sum": float(np.mean(np.abs(p_sum[m] - tsum[m]))), "per_lap": float(np.mean(np.abs(p_lap[m] - tsum[m])))}
        for w in (0.3, 0.5, 0.7):
            r[f"mix{w}"] = float(np.mean(np.abs(w * p_lap[m] + (1 - w) * p_sum[m] - tsum[m])))
        out[name] = r
    res[Y] = out; print(Y, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try10_perlap.json", "w"), indent=1)
for m in ("f3", "l3"):
    avg = {t: np.mean([res[y][m][t] for y in (2022, 2023, 2024)]) for t in res[2022][m]}
    best = min(avg, key=avg.get); print(m, "2022〜2024 平均", {k: round(v, 4) for k, v in avg.items()}, "→ 採用", best, "2025", round(res[2025][m][best], 4))
