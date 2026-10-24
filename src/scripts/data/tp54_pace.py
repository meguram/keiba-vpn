"""tp-54 (1/3): ペース想定（先行勢の数・圧力 → 前半 3F の速さ z）の較正し直しと、位置取り側 vs 整合層の食い違いの答え合わせ。

python -m src.scripts.data.tp54_pace            # ホストで実行（data/local/modeling/tp53/{race,horse}.parquet を読む）
出力: data/local/modeling/tp54/pace_eval.json, config/pace_coherence_v2.json の元（係数）

- 実測の z: レースの前半 3F を、同条件（場×面×距離×馬場の良/道悪）の中央値・標準偏差で標準化（小さい値＝速い）
- 現行: pace_coherence.predict_pace_z（線形: 逃げ・先行の数、圧力、面、距離、道悪）
- 新: 逃げ 0 頭の有無、逃げ×先行の交互作用、頭数、既知馬の割合などを入れた線形（JSON 係数で配れる形）と、GBM（上限の目安）
- 評価は年別 walk-forward（学習は評価年より前）。2026 年は使わない
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])
TP53 = ROOT / "data/local/modeling/tp53"
OUT = Path(os.environ.get("TP54_OUT") or ROOT / "data/local/modeling/tp54")   # ホストは repo が root 所有なので TP54_OUT で書ける場所を指定
KEY = ["venue", "surface", "distance", "going_group"]

FEATURES = ["f0", "f1", "f2", "f3p", "stalk", "stalk2", "f0_stalk", "pressure", "known_frac", "field", "turf", "dist", "dist2", "heavy", "turf_dist", "heavy_turf"]


def load_races() -> pd.DataFrame:
    r = pd.read_parquet(TP53 / "race.parquet")
    r["act_t3f"] = pd.to_numeric(r["act_t3f"], errors="coerce")
    r = r[r["act_t3f"].between(28, 45)].copy()
    g = r.groupby(KEY)["act_t3f"]
    r["n_c"], r["med"], r["sd"] = g.transform("size"), g.transform("median"), g.transform("std")
    r = r[(r["n_c"] >= 30) & (r["sd"] > 0)].copy()
    r["z"] = (r["act_t3f"] - r["med"]) / r["sd"]
    return r.reset_index(drop=True)


def design(r: pd.DataFrame) -> pd.DataFrame:
    """出走表の段階で分かる量だけ（先行勢の数・圧力は position_forecast の出力）。"""
    front = r["pf_front"].fillna(0).clip(upper=6)
    stalk = r["pf_stalker"].fillna(0).clip(upper=10)
    field = r["field_size"].fillna(14).astype(float)
    known = r["pf_known"].fillna(0).astype(float)
    turf = (r["surface"] == "芝").astype(float)
    heavy = (r["going_group"] == "道悪").astype(float)
    dist = (r["distance"].astype(float) - 1600.0) / 1000.0
    d = pd.DataFrame({
        "f0": (front == 0).astype(float), "f1": (front == 1).astype(float), "f2": (front == 2).astype(float), "f3p": (front >= 3).astype(float),
        "stalk": stalk / 5.0, "stalk2": (stalk / 5.0) ** 2, "f0_stalk": (front == 0).astype(float) * stalk / 5.0,
        "pressure": r["pf_pressure"].fillna(0).astype(float), "known_frac": (known / field.clip(lower=1)).clip(upper=1.0),
        "field": (field - 14.0) / 4.0, "turf": turf, "dist": dist, "dist2": dist ** 2, "heavy": heavy,
        "turf_dist": turf * dist, "heavy_turf": heavy * turf,
    })
    return d[FEATURES]


def fit_ridge(X: pd.DataFrame, y: np.ndarray, alpha: float = 3.0) -> dict:
    A = X.to_numpy(float)
    mu = A.mean(axis=0)
    Ac = A - mu
    ym = float(np.mean(y))
    coef = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ (y - ym))
    return {"intercept": float(ym - mu @ coef), "coef": [float(c) for c in coef], "features": list(X.columns)}


def predict_ridge(m: dict, X: pd.DataFrame) -> np.ndarray:
    return m["intercept"] + X[m["features"]].to_numpy(float) @ np.asarray(m["coef"])


def classes(z: np.ndarray, fast: float, slow: float) -> np.ndarray:
    return np.where(z <= fast, "high", np.where(z >= slow, "slow", "mid"))


def current_pred(r: pd.DataFrame, cfg: dict) -> np.ndarray:
    from src.pipeline.inference.pace_coherence import predict_pace_z

    return np.array([predict_pace_z(cfg, int(f), int(s), float(p), su, di, go) for f, s, p, su, di, go in
                     zip(r["pf_front"].fillna(0), r["pf_stalker"].fillna(0), r["pf_pressure"].fillna(0), r["surface"], r["distance"], r["track_condition"])])


def metrics(z_true: np.ndarray, z_pred: np.ndarray, th: tuple[float, float]) -> dict:
    ct, cp = classes(z_true, *th), classes(z_pred, *th)
    return {"n": int(len(z_true)), "corr": round(float(np.corrcoef(z_true, z_pred)[0, 1]), 4), "mae": round(float(np.mean(np.abs(z_true - z_pred))), 4),
            "acc3": round(float(np.mean(ct == cp)), 4), "slope": round(float(np.polyfit(z_pred, z_true, 1)[0]), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config/pace_coherence.json").read_text(encoding="utf-8"))
    th = (float(cfg["pace"]["thresholds"]["fast"]), float(cfg["pace"]["thresholds"]["slow"]))
    r = load_races()
    r["cur"] = current_pred(r, cfg)
    X = design(r)
    res: dict = {"thresholds": list(th), "n_races": int(len(r)), "by_year": {}}
    gbm_ok = True
    try:
        import lightgbm as lgb
    except Exception:  # pragma: no cover
        gbm_ok = False
    preds = []
    for y in (2023, 2024, 2025):
        tr, te = r["year"] < y, r["year"] == y
        m = fit_ridge(X[tr], r.loc[tr, "z"].to_numpy())
        p_new = predict_ridge(m, X[te])
        row = {"current": metrics(r.loc[te, "z"].to_numpy(), r.loc[te, "cur"].to_numpy(), th),
               "ridge_v2": metrics(r.loc[te, "z"].to_numpy(), p_new, th)}
        if gbm_ok:
            g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
            g.fit(X[tr], r.loc[tr, "z"])
            p_g = g.predict(X[te])
            row["gbm"] = metrics(r.loc[te, "z"].to_numpy(), p_g, th)
        res["by_year"][str(y)] = row
        d = r.loc[te, ["race_id", "year", "z", "cur", "pf_level", "pf_front", "pf_stalker"]].copy()
        d["new"] = p_new
        preds.append(d)
    pr = pd.concat(preds, ignore_index=True)
    # 食い違いの答え合わせ: 位置取り側ラベル（pf_level）と整合層（現行）の向きが真逆のレース
    cur_cls, new_cls, act_cls = classes(pr["cur"].to_numpy(), *th), classes(pr["new"].to_numpy(), *th), classes(pr["z"].to_numpy(), *th)
    pf_cls = pr["pf_level"].map({"high": "high", "mid": "mid", "slow": "slow"}).to_numpy()
    known = pd.notna(pf_cls)
    opposite = known & (((pf_cls == "slow") & (cur_cls == "high")) | ((pf_cls == "high") & (cur_cls == "slow")))
    dis = known & (pf_cls != cur_cls)
    res["disagreement"] = {
        "n_known": int(known.sum()), "n_disagree_pf_vs_current": int(dis.sum()), "n_opposite_pf_vs_current": int(opposite.sum()),
        "opposite_rate": round(float(opposite.sum() / max(known.sum(), 1)), 4),
        "when_disagree_acc": {"pf_level": round(float(np.mean(pf_cls[dis] == act_cls[dis])), 4), "current": round(float(np.mean(cur_cls[dis] == act_cls[dis])), 4), "ridge_v2": round(float(np.mean(new_cls[dis] == act_cls[dis])), 4)},
        "when_opposite_acc": {"pf_level": round(float(np.mean(pf_cls[opposite] == act_cls[opposite])), 4) if opposite.any() else None,
                              "current": round(float(np.mean(cur_cls[opposite] == act_cls[opposite])), 4) if opposite.any() else None,
                              "ridge_v2": round(float(np.mean(new_cls[opposite] == act_cls[opposite])), 4) if opposite.any() else None},
        "when_opposite_mean_z_by_pf": {k: round(float(pr.loc[opposite & (pf_cls == k), "z"].mean()), 3) for k in ("slow", "high") if (opposite & (pf_cls == k)).any()},
    }
    # 逃げ 0 頭のレースでの実測
    f0 = pr["pf_front"].fillna(0) == 0
    res["front0"] = {"n": int(f0.sum()), "mean_z": round(float(pr.loc[f0, "z"].mean()), 3), "mean_pred_current": round(float(pr.loc[f0, "cur"].mean()), 3), "mean_pred_v2": round(float(pr.loc[f0, "new"].mean()), 3),
                     "share_actual_slow": round(float((act_cls[f0.to_numpy()] == "slow").mean()), 3), "share_current_high": round(float((cur_cls[f0.to_numpy()] == "high").mean()), 3), "share_v2_high": round(float((new_cls[f0.to_numpy()] == "high").mean()), 3)}
    # 全期間で当てはめた最終係数（配布用）
    final = fit_ridge(X, r["z"].to_numpy())
    final["thresholds"] = {"fast": th[0], "slow": th[1]}
    final["trained_years"] = [int(r["year"].min()), int(r["year"].max())]
    final["n_races"] = int(len(r))
    res["final_ridge_v2"] = final
    (OUT / "pace_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pr.to_parquet(OUT / "pace_eval_pred.parquet", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "final_ridge_v2"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
