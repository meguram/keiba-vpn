"""ラップ予想 試行7: 展開（スロー/ミドル/ハイ）の分類。今の分類器・大きめの分類器・『前後半の差』の回帰→確率・混ぜ合わせ。採否は 2022〜2024 年の平均。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import norm
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
OUT = "/home/jovyan/work/retrain_out/tp55_2016"
d = pd.read_parquet(f"{OUT}/races_feat.parquet"); d["year"] = d["date"].str[:4].astype(int)
num, cats = lfm.feature_columns(); cols = num + cats
for c in cats: d[c] = d[c].astype("category")
def qstat(df, par):
    laps = [json.loads(s) for s in df["laps_json"]]; L = lfm.laps_matrix(laps); P = lfm.par_matrix(df, par); dev = L - P
    n = np.array([len(a) for a in laps]); st = lfm.scenario_stats(dev, n)
    ok = ~np.isnan(P[np.arange(len(df)), 0]) & ~np.isnan(dev[np.arange(len(df)), 0])
    return st[:, 2] - st[:, 0], ok
def ll(p, y): return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))
def temp_fit(p, y):
    best, bt = 1e9, 1.0
    for t in np.linspace(0.6, 1.8, 25):
        z = np.log(np.clip(p, 1e-9, 1)) / t; z -= z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
        v = ll(q, y); best, bt = (v, t) if v < best else (best, bt)
    return bt
def apply_t(p, t):
    z = np.log(np.clip(p, 1e-9, 1)) / t; z -= z.max(1, keepdims=True); q = np.exp(z); return q / q.sum(1, keepdims=True)
res = {}
for Y in (2022, 2023, 2024, 2025):
    tr, te = d[d.year < Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    qtr, oktr = qstat(tr, par); qte, okte = qstat(te, par)
    tr, qtr = tr[oktr & ~np.isnan(qtr)].reset_index(drop=True), qtr[oktr & ~np.isnan(qtr)]
    te, qte = te[okte & ~np.isnan(qte)].reset_index(drop=True), qte[okte & ~np.isnan(qte)]
    lo, hi = np.nanpercentile(qtr, [25, 75])
    lab = lambda q: np.where(q <= lo, 0, np.where(q >= hi, 2, 1))
    ytr, yte = lab(qtr), lab(qte)
    # 温度合わせ用に学習期間の最後の 1 年を取り分け（その前で学習）
    last = tr["year"] == tr["year"].max()
    out = {}
    def clf(params, rounds):
        b1 = lgb.train(params, lgb.Dataset(tr.loc[~last, cols], label=ytr[~last]), num_boost_round=rounds)
        t = temp_fit(b1.predict(tr.loc[last, cols]), ytr[last])
        b = lgb.train(params, lgb.Dataset(tr[cols], label=ytr), num_boost_round=rounds)
        return apply_t(b.predict(te[cols]), t)
    base = dict(objective="multiclass", num_class=3, learning_rate=0.04, num_leaves=15, min_data_in_leaf=40, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0, verbose=-1, num_threads=4, seed=0)
    p_base = clf(base, 250); out["clf_base"] = ll(p_base, yte)
    p_big = np.mean([clf(dict(base, learning_rate=0.02, num_leaves=31, min_data_in_leaf=60, seed=s), 900) for s in (0, 1, 2)], axis=0); out["clf_big"] = ll(p_big, yte)
    # 回帰 → 確率（残差の標準偏差は取り分けた年で推定）
    rp = dict(objective="huber", alpha=0.8, learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
    b1 = lgb.train(dict(rp, seed=0), lgb.Dataset(tr.loc[~last, cols], label=qtr[~last]), num_boost_round=1200)
    sd = float(np.std(qtr[last] - b1.predict(tr.loc[last, cols])))
    mu = np.mean([lgb.train(dict(rp, seed=s), lgb.Dataset(tr[cols], label=qtr), num_boost_round=1200).predict(te[cols]) for s in (0, 1, 2)], axis=0)
    c_lo, c_hi = norm.cdf((lo - mu) / sd), norm.cdf((hi - mu) / sd)
    p_reg = np.stack([c_lo, c_hi - c_lo, 1 - c_hi], 1); out["reg_prob"] = ll(p_reg, yte)
    for w in (0.3, 0.5, 0.7):
        out[f"blend_big_reg{w}"] = ll(w * p_big + (1 - w) * p_reg, yte)
    res[Y] = out; print(Y, json.dumps({k: round(v, 4) for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try7_classify.json", "w"), indent=1)
avg = {t: np.mean([res[y][t] for y in (2022, 2023, 2024)]) for t in res[2022]}
best = min(avg, key=avg.get); print("2022〜2024 平均 最良", best, round(avg[best], 4), "→ 2025", round(res[2025][best], 4), "（今の分類器の再現 2025:", round(res[2025]["clf_base"], 4), "）")
