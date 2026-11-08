"""追走難度（試行6 の設定）の EASE2 群を作る。tp54_model.run_oof と同じく年ごとに『その年を除く全年』で学習し、
drain（追走の量）と full（全体）に、ラップ予想の見込み（LAP・LAPD 群）・位置取りのメンバー内の関係・馬の過去の終いの落ち（組み合わせの形だけ）を足す。
demand（要求だけ）は従来どおり。出力: retrain_out/tp54_2016/groups/EASE2_t6.parquet"""
import json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
os.environ.setdefault("TP54_OUT", "/home/jovyan/work/retrain_out/tp54_2016")
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.scripts.data import tp54_model as m
from src.pipeline.models import tracking_difficulty_v2 as td2
O = m.OUT
h = m.load(); X = td2.make_features(h); ok = h["pf_early"].notna()
h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
lap = pd.read_parquet("/home/jovyan/work/retrain_out/tp55_2016/groups/LAP.parquet"); lap["race_id"], lap["horse_id"] = lap["race_id"].astype(str), lap["horse_id"].astype(str)
L = h[["race_id", "horse_id"]].merge(lap.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left").drop(columns=["race_id", "horse_id"]); L.index = h.index
ld = pd.read_parquet("/home/jovyan/work/retrain_out/tp55_2016/groups/LAPD.parquet"); ld["race_id"] = ld["race_id"].astype(str)
D = h[["race_id"]].merge(ld.drop_duplicates("race_id"), on="race_id", how="left").drop(columns=["race_id"]); D.index = h.index
pe = h["pf_early"].astype(float); D["ld_front_x_f3"] = (1 - pe) * D["ld_f3"]; D["ld_rear_x_q"] = pe * D["ld_q"]
po = pd.read_parquet("/home/jovyan/work/keiba-vpn/data/local/modeling/pos_model/pos_oof.parquet"); po["race_id"], po["horse_id"] = po["race_id"].astype(str), po["horse_id"].astype(str)
g = po.groupby("race_id")["pos_pred"]
po["pp_rank"], po["pp_ahead"], po["pp_front_n"], po["pp_gap_mean"] = g.rank(pct=True), g.rank(method="min") - 1, g.transform(lambda x: (x < 0.15).sum()), po["pos_pred"] - g.transform("mean")
P = h[["race_id", "horse_id"]].merge(po[["race_id", "horse_id", "pp_rank", "pp_ahead", "pp_front_n", "pp_gap_mean"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left").drop(columns=["race_id", "horse_id"]); P.index = h.index
B = dict(m.PARAMS, learning_rate=0.015, num_leaves=63, min_child_samples=150)
def fitp(XX, tr, te, y, seeds=(0, 1, 2)):
    return np.mean([lgb.train(dict(B, seed=s), lgb.Dataset(XX[tr], y[tr].to_numpy()), num_boost_round=1500).predict(XX[te]) for s in seeds], axis=0)
out = pd.DataFrame(index=h.index, columns=["drain", "full", "demand"], dtype=float)
for yr in sorted(h["year"].unique()):
    coef = m.baseline_fit(h, ok & (h["year"] != yr)); T, y = m.make_target(h, coef)
    gy = y.groupby(h["horse_id"], sort=False); fr = 1 - pe
    hy3 = pd.concat([gy.shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1)
    Hi = pd.DataFrame({"hy3_x_front": hy3 * fr, "hy1_x_front": gy.shift(1) * fr, "hy3_x_press": hy3 * h["pf_pressure"].astype(float), "hy3_x_zhat": hy3 * h["z_hat"].astype(float)}, index=h.index)
    tr = ok & y.notna() & (h["year"] != yr); te = ok & (h["year"] == yr)
    XC = pd.concat([X[m.CORE], L, D, P, Hi], axis=1); XB = pd.concat([X, L, D, P, Hi], axis=1)
    out.loc[te, "drain"] = fitp(XC, tr, te, y); out.loc[te, "full"] = fitp(XB, tr, te, y)
    out.loc[te, "demand"] = m.fit(X.loc[tr, m.DEMAND_ONLY], y[tr].to_numpy()).predict(X.loc[te, m.DEMAND_ONLY])
    print("oof", yr, int(te.sum()), flush=True)
q = np.nanquantile(out["drain"], np.linspace(0, 1, 101)).tolist()
gg = pd.DataFrame({"race_id": h["race_id"], "horse_id": h["horse_id"], "EASE2_drain": out["drain"], "EASE2_demand": out["demand"], "EASE2_full": out["full"],
                   "EASE2_capacity": out["drain"] - out["demand"], "EASE2_pct": td2.to_ease_pct(out["drain"].fillna(np.nan).to_numpy(), q), "EASE2_vs_prev": np.nan})
gg.loc[out["drain"].isna(), "EASE2_pct"] = np.nan
gg.to_parquet(O / "groups" / "EASE2_t6.parquet", index=False); print("wrote", len(gg))
