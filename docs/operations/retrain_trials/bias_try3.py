"""馬場の当日予測 試行3: 静的モデル（当日の結果を使わない LightGBM）を事前値にし、前のレースの「実現 − 静的予測」を
重み付き平均して縮めて足す。重み・縮みは 2019〜2021 年で選び、2022〜2024 年で評価（評価年で選ばない）。
比較は 静的モデル（当日なし）に対する上乗せ。"""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "docs/operations/retrain_trials")
import bias_eval as b

R = b.race_table(); mu = b.course_prior(R)
F = pd.DataFrame(index=R.index)
F["mu_F"], F["mu_D"] = mu["mu_b_F"], mu["mu_b_D"]
F["going"] = R["going"].map({"良": 0, "稍重": 1, "重": 2, "不良": 3})
F["kai"] = R["race_id"].str[6:8].astype(int); F["day"] = R["day"]; F["rnum"] = R["rnum"]; F["n"] = R["n"]; F["dist"] = R["distance"]
F["turf"] = (R["surface"] == "芝").astype(int); F["venue"] = R["race_id"].str[4:6].astype(int)
PAR = dict(objective="huber", alpha=0.5, learning_rate=0.02, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4)
S = pd.DataFrame(index=R.index)
for t in b.TARGETS:
    S[t] = np.nan
    for Y in range(2018, 2026):
        tr = (R.year < Y) & (R.year >= 2017) & R[t].notna() & F["mu_F"].notna(); te = R.year == Y
        S.loc[te, t] = np.mean([lgb.train(dict(PAR, seed=s), lgb.Dataset(F[tr], R.loc[tr, t] - mu.loc[tr, "mu_" + t]), 600).predict(F[te]) for s in range(3)], axis=0) + mu.loc[te, "mu_" + t]
S.to_parquet(b.OUT / "static_pred.parquet")
mu2 = pd.DataFrame({"mu_" + t: S[t] for t in b.TARGETS})
P = b.window_pairs(R)
def run(years, half, prev_w, chg_w, K):
    pr = b.predict(R, P, mu2, half=half, prev_w=prev_w, chg_w=chg_w, K=K)
    old = b.EVAL_YEARS; b.EVAL_YEARS = years
    try:
        return b.skill(R, pr, boot=1000 if years == old else 0), pr
    finally:
        b.EVAL_YEARS = old
best = {t: (None, -9) for t in b.TARGETS}; tune = {}
for half in (2.0, 4.0, 6.0, 12.0, 1e9):
    for prev_w in (0.0, 0.4, 0.7, 1.0):
        for chg_w in (0.0, 0.5, 1.0):
            for K in (3.0, 6.0, 12.0, 24.0, 48.0):
                s, _ = run([2019, 2020, 2021], half, prev_w, chg_w, K)
                tune[f"{half}_{prev_w}_{chg_w}_{K}"] = s["skill"]
                for t in b.TARGETS:
                    if s["skill_" + t] > best[t][1]:
                        best[t] = ((half, prev_w, chg_w, K), s["skill_" + t])
print("選んだ設定（2019-21）", best, flush=True)
pr = pd.DataFrame(index=R.index)
for t in b.TARGETS:
    s, p = run(b.EVAL_YEARS, *best[t][0])
    for c in ("base_", "pred_", "sw_"):
        pr[c + t] = p[c + t]
    pr["n_prev"] = p["n_prev"]
# 静的モデルに対する上乗せ（base_ = 静的予測）と、コース平均に対する技能の両方
vs_static = b.skill(R, pr, boot=1000)
pr2 = pr.copy()
for t in b.TARGETS:
    pr2["base_" + t] = mu["mu_" + t]
vs_course = b.skill(R, pr2, boot=1000)
res = {"chosen": {t: best[t][0] for t in b.TARGETS}, "tune_skill_2019_21": {t: best[t][1] for t in b.TARGETS}, "vs_static": vs_static, "vs_course": vs_course}
print(json.dumps(res, ensure_ascii=False))
(b.OUT / "bias_try3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pr.to_parquet(b.OUT / "pred_try3.parquet")
