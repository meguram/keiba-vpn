"""ラップ予想 試行6: Huber の alpha と、今のモデル（展開の混合）との混ぜ合わせ。採否は 2022〜2024 年の平均。"""
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
BASE = dict(objective="huber", learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
res = {}
for Y in (2022, 2023, 2024, 2025):
    tr, te = d[d.year < Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    f3tr, l3tr = targets(tr, par); f3te, l3te = targets(te, par)
    mix = pd.read_parquet(f"{OUT}/main_A_full_{Y}_pred.parquet").set_index("race_id").reindex(te["race_id"])
    out = {}
    for name, ytr, yte, mcol in (("f3", f3tr, f3te, "exp_f3_dev"), ("l3", l3tr, l3te, "exp_l3_dev")):
        ok = ~np.isnan(ytr); mm = mix[mcol].to_numpy(); m = ~np.isnan(yte) & ~np.isnan(mm); r = {}
        for a in (0.3, 0.6, 1.0):
            p = np.mean([lgb.train(dict(BASE, alpha=a, seed=sd), lgb.Dataset(tr.loc[ok, cols], label=ytr[ok]), num_boost_round=1200).predict(te[cols]) for sd in (0, 1, 2)], axis=0)
            r[f"h{a}"] = float(np.mean(np.abs(p[m] - yte[m])))
            for w in (0.6, 0.8):
                r[f"h{a}_mix{w}"] = float(np.mean(np.abs(w * p[m] + (1 - w) * mm[m] - yte[m])))
        out[name] = r
    res[Y] = out; print(Y, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try6_huber_mix.json", "w"), indent=1)
for name in ("f3", "l3"):
    tags = list(res[2022][name]); avg = {t: np.mean([res[y][name][t] for y in (2022, 2023, 2024)]) for t in tags}
    best = min(avg, key=avg.get); print(name, "2022〜2024 平均 最良", best, round(avg[best], 4), "→ 2025", round(res[2025][name][best], 4))
