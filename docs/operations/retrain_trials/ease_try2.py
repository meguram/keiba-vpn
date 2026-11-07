"""追走難度 試行2: 学習の設定（回数・木の大きさ・最小葉・損失・乱数の平均）。指標は v2_full と同じ（予測と Y2 の相関、2023〜2025 年、年ごとに広げる）。"""
import json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
os.environ["TP54_OUT"] = "/home/jovyan/work/retrain_out/tp54_2016"
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.scripts.data import tp54_model as m
from src.pipeline.models import tracking_difficulty_v2 as td2
h = m.load(); X = td2.make_features(h); ok = h["pf_early"].notna()
B = dict(m.PARAMS)
CFG = {"base": (B, 500, (0,)), "seeds3": (B, 500, (0, 1, 2)), "slow_big": (dict(B, learning_rate=0.015, num_leaves=63, min_child_samples=150), 1500, (0, 1, 2)),
       "leaf400": (dict(B, min_child_samples=400), 800, (0, 1, 2)), "huber": (dict(B, objective="huber", alpha=1.0), 800, (0, 1, 2))}
res = {k: {} for k in CFG}
for yr in (2023, 2024, 2025):
    coef = m.baseline_fit(h, ok & (h["year"] < yr)); T, y = m.make_target(h, coef)
    tr = ok & y.notna() & (h["year"] < yr); te = ok & y.notna() & (h["year"] == yr)
    for k, (p, r, seeds) in CFG.items():
        pr = np.mean([lgb.train(dict(p, seed=s), lgb.Dataset(X[tr], y[tr].to_numpy()), num_boost_round=r).predict(X[te]) for s in seeds], axis=0)
        res[k][yr] = m.corr(pr, y[te])
    print(yr, {k: round(v[yr], 4) for k, v in res.items()}, flush=True)
for k, v in res.items():
    print(k, "平均", round(np.mean(list(v.values())), 4), "最小", round(min(v.values()), 4))
json.dump({k: {str(y): c for y, c in v.items()} for k, v in res.items()}, open("/home/jovyan/work/retrain_out/tp54_2016/try2_cfg.json", "w"), indent=1)
