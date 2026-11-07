"""ラップ予想 試行8: 展開の分類に、前後半の差・前半3F・後半3F の直接予測（学習期間はレース単位の 3 分割 OOF）を入力として足す 2 段構え。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
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
RP = dict(objective="huber", alpha=0.8, learning_rate=0.03, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)
CP = dict(objective="multiclass", num_class=3, learning_rate=0.04, num_leaves=15, min_data_in_leaf=40, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, lambda_l2=5.0, verbose=-1, num_threads=4, seed=0)
res = {}
for Y in (2022, 2023, 2024, 2025):
    tr, te = d[d.year < Y].reset_index(drop=True), d[d.year == Y].reset_index(drop=True)
    par = lf.fit_par(pd.DataFrame({"venue": tr["venue"], "surface": tr["surface"], "distance": tr["distance"], "direction": tr["direction"], "laps": [json.loads(x) for x in tr["laps_json"]]}))
    qtr, f3tr, l3tr, oktr = stats(tr, par); qte, _, _, okte = stats(te, par)
    k = oktr & ~np.isnan(qtr); tr, qtr, f3tr, l3tr = tr[k].reset_index(drop=True), qtr[k], f3tr[k], l3tr[k]
    k = okte & ~np.isnan(qte); te, qte = te[k].reset_index(drop=True), qte[k]
    lo, hi = np.nanpercentile(qtr, [25, 75]); lab = lambda q: np.where(q <= lo, 0, np.where(q >= hi, 2, 1)); ytr, yte = lab(qtr), lab(qte)
    S_tr = np.zeros((len(tr), 3)); S_te = np.zeros((len(te), 3))
    for j, t in enumerate((qtr, f3tr, l3tr)):
        ok = ~np.isnan(t)
        for a, b in GroupKFold(3).split(tr, groups=tr["race_id"]):
            a = a[ok[a]]
            S_tr[b, j] = lgb.train(RP, lgb.Dataset(tr.loc[a, cols], label=t[a]), num_boost_round=800).predict(tr.loc[b, cols])
        S_te[:, j] = lgb.train(RP, lgb.Dataset(tr.loc[ok, cols], label=t[ok]), num_boost_round=800).predict(te[cols])
    X_tr = pd.concat([tr[cols], pd.DataFrame(S_tr, columns=["q_hat", "f3_hat", "l3_hat"])], axis=1)
    X_te = pd.concat([te[cols], pd.DataFrame(S_te, columns=["q_hat", "f3_hat", "l3_hat"])], axis=1)
    last = (tr["year"] == tr["year"].max()).to_numpy()
    out = {}
    for tag, Xa, Xb in (("base", tr[cols], te[cols]), ("stack", X_tr, X_te)):
        b1 = lgb.train(CP, lgb.Dataset(Xa[~last], label=ytr[~last]), num_boost_round=250); t = temp_fit(b1.predict(Xa[last]), ytr[last])
        ps = [apply_t(lgb.train(dict(CP, seed=s), lgb.Dataset(Xa, label=ytr), num_boost_round=250).predict(Xb), t) for s in (0, 1, 2)]
        out[tag] = ll(np.mean(ps, axis=0), yte)
    res[Y] = out; print(Y, json.dumps({k: round(v, 4) for k, v in out.items()}), flush=True)
json.dump(res, open(f"{OUT}/try8_stack_clf.json", "w"), indent=1)
avg = {t: np.mean([res[y][t] for y in (2022, 2023, 2024)]) for t in res[2022]}
best = min(avg, key=avg.get); print("2022〜2024 平均", {k: round(v, 4) for k, v in avg.items()}, "→ 採用", best, "2025", round(res[2025][best], 4))
