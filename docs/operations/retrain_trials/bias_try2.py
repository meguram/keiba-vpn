"""馬場の当日予測 試行2: 学習モデル（LightGBM、年ごとに前の年だけで学習）。
入力: コース平均・馬場状態・開催の何日目・レース番号・頭数・距離・場・芝ダ、同じ日の前のレース（半減 2・6 レース）、
同じ開催の前の日すべて（日ごとに半減）、同じ日のもう一方の芝ダの傾き。目的: b − コース平均。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "docs/operations/retrain_trials")
import bias_eval as b

R = b.race_table(); mu = b.course_prior(R)
for t in b.TARGETS:
    R["d_" + t] = R[t] - mu["mu_" + t]
key = R[["day_key", "meet", "day", "rnum", "surface", "going"] + ["d_" + t for t in b.TARGETS]].reset_index().rename(columns={"index": "i"})
F = pd.DataFrame(index=R.index)
F["mu_F"], F["mu_D"] = mu["mu_b_F"], mu["mu_b_D"]
F["going"] = R["going"].map({"良": 0, "稍重": 1, "重": 2, "不良": 3})
F["kai"] = R["race_id"].str[6:8].astype(int); F["day"] = R["day"]; F["rnum"] = R["rnum"]; F["n"] = R["n"]; F["dist"] = R["distance"]
F["turf"] = (R["surface"] == "芝").astype(int); F["venue"] = R["race_id"].str[4:6].astype(int)

def agg(P, name, w):
    for t in b.TARGETS:
        v = P["d_" + t + "_j"].to_numpy(); ok = ~np.isnan(v)
        sw = np.bincount(P["i"].to_numpy()[ok], w[ok], minlength=len(R)); s = np.bincount(P["i"].to_numpy()[ok], (w * v)[ok], minlength=len(R))
        F[f"{name}_{t}"] = np.where(sw > 0, s / np.maximum(sw, 1e-9), np.nan); F[f"{name}_w"] = sw

same = key.merge(key, on=["day_key", "surface"], suffixes=("", "_j")); same = same[same.rnum_j < same.rnum]
for h in (2.0, 6.0):
    agg(same, f"sd{int(h)}", 0.5 ** ((same.rnum - 1 - same.rnum_j).to_numpy() / h))
F["sd_last_F"] = same.sort_values("rnum_j").groupby("i")["d_b_F_j"].last().reindex(R.index)
F["sd_last_D"] = same.sort_values("rnum_j").groupby("i")["d_b_D_j"].last().reindex(R.index)
md = key.merge(key, on=["meet", "surface"], suffixes=("", "_j")); md = md[md.day_j < md.day]
agg(md, "md", 0.5 ** ((md.day - md.day_j - 1).to_numpy() / 1.0) * np.where(md.going != md.going_j, 0.5, 1.0))
F["md_going_chg"] = md.assign(c=(md.going != md.going_j)).groupby("i")["c"].mean().reindex(R.index)
oth = key.merge(key, on=["day_key"], suffixes=("", "_j")); oth = oth[(oth.rnum_j < oth.rnum) & (oth.surface != oth.surface_j)]
agg(oth, "os", np.ones(len(oth)))
for t in b.TARGETS:
    F[f"x_{t}"] = F[f"sd6_{t}"]
pr = pd.DataFrame(index=R.index); P = b.window_pairs(R)
pr["n_prev"] = pd.Series(1.0, index=P.index).groupby(P["i"]).sum().reindex(R.index).fillna(0).to_numpy()
PAR = dict(objective="huber", alpha=0.5, learning_rate=0.02, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4)
for t in b.TARGETS:
    pr["base_" + t] = mu["mu_" + t]; pr["pred_" + t] = np.nan
    for Y in b.EVAL_YEARS:
        tr = (R.year < Y) & (R.year >= 2018) & R["d_" + t].notna(); te = R.year == Y
        ps = [lgb.train(dict(PAR, seed=s), lgb.Dataset(F[tr], R.loc[tr, "d_" + t]), 600).predict(F[te]) for s in range(3)]
        pr.loc[te, "pred_" + t] = mu.loc[te, "mu_" + t] + np.mean(ps, axis=0)
res = b.skill(R, pr, boot=1000)
print(json.dumps(res, ensure_ascii=False))
(b.OUT / "bias_try2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pr.to_parquet(b.OUT / "pred_try2.parquet")
