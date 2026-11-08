"""ラップ予想 試行9: 芝・ダートで別々に学習したモデルと、まとめたモデルの比較（前半/後半 3F の直接予測 Huber・展開の分類）。採否は 2022〜2024 年の平均。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
OUT = "/home/jovyan/work/retrain_out/tp55_2016"
d = pd.read_parquet(f"{OUT}/races_feat.parquet"); d["year"] = d["date"].str[:4].astype(int)
num, cats = lfm.feature_columns(); cols = num + cats
for c in cats: d[c] = d[c].astype("category")
def stats(df, par):
    laps = [json.loads(s) for s in df["laps_json"]]; L = lfm.laps_matrix(laps); P = lfm.par_matrix(df, par); dev = L - P
    n = np.array([len(a) for a in laps]); st = lfm.scenario_stats(dev, n)
    f3 = np.array([dev[i, :3].sum() for i in range(len(df))]); l3 = np.array([dev[i, n[i]-3:n[i]].sum() for i in range(len(df))])
    ok = ~np.isnan(P[np.arange(len(df)), 0]) & ~np.isnan(dev[np.arange(len(df)), 0])
    return st[:, 2] - st[:, 0], f3, l3, ok
def ll(p, y): return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))
def apply_t(p, t):
    z = np.log(np.clip(p, 1e-9, 1)) / t; z -= z.max(1, keepdims=True); q = np.exp(z); return q / q.sum(1, keepdims=True)
def temp_fit(p, y):
    ts = np.linspace(0.6, 1.8, 25); return float(ts[np.argmin([ll(apply_t(p, t), y) for t in ts])])
RP = dict(objective="huber", alpha=0.6, learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
CP = dict(objective="multiclass", num_class=3, learning_rate=0.04, num_leaves=15, min_data_in_leaf=40, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0, verbose=-1, num_threads=4)
res = {}
for Y in (2022, 2023, 2024, 2025):
    tr, te = d[d.year < Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    qtr, f3tr, l3tr, oktr = stats(tr, par); qte, f3te, l3te, okte = stats(te, par)
    k = oktr & ~np.isnan(qtr); tr, qtr, f3tr, l3tr = tr[k].reset_index(drop=True), qtr[k], f3tr[k], l3tr[k]
    k = okte & ~np.isnan(qte); te, qte, f3te, l3te = te[k].reset_index(drop=True), qte[k], f3te[k], l3te[k]
    lo, hi = np.nanpercentile(qtr, [25, 75]); lab = lambda q: np.where(q <= lo, 0, np.where(q >= hi, 2, 1)); ytr, yte = lab(qtr), lab(qte)
    out = {}
    for tag, parts in (("pooled", [(np.ones(len(tr), bool), np.ones(len(te), bool))]),
                       ("by_surface", [((tr["surface"] == s).to_numpy(), (te["surface"] == s).to_numpy()) for s in tr["surface"].unique()])):
        pf3, pl3 = np.full(len(te), np.nan), np.full(len(te), np.nan); pc = np.full((len(te), 3), np.nan)
        for mtr, mte in parts:
            if not mte.any():
                continue
            for t, store in ((f3tr, pf3), (l3tr, pl3)):
                ok = mtr & ~np.isnan(t)
                store[mte] = np.mean([lgb.train(dict(RP, seed=s), lgb.Dataset(tr.loc[ok, cols], label=t[ok]), num_boost_round=1200).predict(te.loc[mte, cols]) for s in (0, 1, 2)], axis=0)
            last = mtr & (tr["year"] == tr["year"].max()).to_numpy(); early = mtr & ~last
            tt = temp_fit(lgb.train(dict(CP, seed=0), lgb.Dataset(tr.loc[early, cols], label=ytr[early]), num_boost_round=250).predict(tr.loc[last, cols]), ytr[last])
            pc[mte] = np.mean([apply_t(lgb.train(dict(CP, seed=s), lgb.Dataset(tr.loc[mtr, cols], label=ytr[mtr]), num_boost_round=250).predict(te.loc[mte, cols]), tt) for s in (0, 1, 2)], axis=0)
        m3, ml = ~np.isnan(f3te) & ~np.isnan(pf3), ~np.isnan(l3te) & ~np.isnan(pl3)
        out[tag] = {"f3": float(np.mean(np.abs(pf3[m3] - f3te[m3]))), "l3": float(np.mean(np.abs(pl3[ml] - l3te[ml]))), "clf_ll": ll(pc, yte)}
    res[Y] = out; print(Y, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try9_surface.json", "w"), indent=1)
for m in ("f3", "l3", "clf_ll"):
    avg = {t: np.mean([res[y][t][m] for y in (2022, 2023, 2024)]) for t in ("pooled", "by_surface")}
    best = min(avg, key=avg.get); print(m, "2022〜2024 平均", {k: round(v, 4) for k, v in avg.items()}, "→ 採用", best, "2025", round(res[2025][best][m], 4))
