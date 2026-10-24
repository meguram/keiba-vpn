"""v6 (戦略 4): ペース・ラップの不確実性（シナリオ確率のエントロピー）などのレースレベルの条件を、アンサンブル出力の温度調整に使う。

python -m src.scripts.data.v6_temp     # v5 の LOYO アンサンブル（tp27_v5/ens/v5.parquet、2022〜2024 年）に、年をまたぐ交差検証で温度調整をかけて比べる

各目的（勝率・連対・複勝）の確率 p を、logit(p') = τ·logit(p) + b に変換し、レース内の合計（1/2/3）にそろえ直す。
τ = exp(a0 + a1·エントロピー + a2·頭数(z) + a3·最大勝率(z))、b は定数。係数は評価年以外の 2 年の対数損失を最小化して決め、評価年に適用する
（3 つの評価年を順に回す）。ラップ予想のエントロピーは出走表の段階で分かる値で、結果は使わない。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.models import win_models as wm
from src.scripts.data import tp27_eval as t

KEY = ["race_id", "horse_id"]
V = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline_v4"
OUT = Path(os.environ.get("V6_OUT") or V / "v6")
LAP = Path(os.environ.get("V6_LAP") or "/home/jovyan/work/tp55_out3/groups/LAP.parquet")
TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def transform(p: np.ndarray, race: np.ndarray, tau: np.ndarray, b: float, total: float) -> np.ndarray:
    q = sigmoid(tau * logit(p) + b)
    return wm.normalize_to_total(race, q, total)


def race_features(d: pd.DataFrame, lap: pd.DataFrame) -> pd.DataFrame:
    P = lap[["lp_p_slow", "lp_p_mid", "lp_p_high"]].to_numpy(dtype=float)
    P = np.clip(np.where(np.isfinite(P), P, 1 / 3), 1e-9, 1)
    P = P / P.sum(axis=1, keepdims=True)
    ent = pd.DataFrame({"race_id": lap["race_id"].astype(str).to_numpy(), "ent": -(P * np.log(P)).sum(axis=1)}).drop_duplicates("race_id")
    x = d.merge(ent, on="race_id", how="left")
    x["ent"] = x["ent"].fillna(x["ent"].mean())
    g = x.groupby("race_id")
    x["n"] = g["horse_id"].transform("count")
    x["pmax"] = g["p_win"].transform("max")
    for c in ("ent", "n", "pmax"):
        x[f"z_{c}"] = (x[c] - x[c].mean()) / (x[c].std() + 1e-9)
    return x


def fit_params(x: pd.DataFrame, y: np.ndarray, target: str, feats: list[str]) -> np.ndarray:
    from scipy.optimize import minimize

    race = x["race_id"].to_numpy()
    p = x[f"p_{target}"].to_numpy()
    F = x[feats].to_numpy(dtype=float)

    def loss(th):
        tau = np.exp(th[0] + F @ th[1:1 + len(feats)])
        q = transform(p, race, tau, th[-1], TOT[target])
        return wm.binary_logloss(q, y)

    th0 = np.zeros(1 + len(feats) + 1)
    r = minimize(loss, th0, method="Powell", options={"maxiter": 2000, "xtol": 1e-3, "ftol": 1e-7})
    return r.x


def main() -> None:
    d = pd.read_parquet(V / "tp27_v5/ens/v5.parquet")
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    lap = pd.read_parquet(LAP)
    x = race_features(d, lap)
    fin = fin_map.loc[list(zip(x["race_id"], x["horse_id"]))].to_numpy()
    out_oof = x[KEY + ["year"]].copy()
    rep: dict = {"variants": {}}
    for vname, feats in (("const", []), ("ent", ["z_ent"]), ("ent_n_pmax", ["z_ent", "z_n", "z_pmax"]), ("n_pmax", ["z_n", "z_pmax"])):
        res = {}
        for tg, k in (("win", 1), ("top2", 2), ("top3", 3)):
            yy = (fin <= k).astype(float)
            new = np.zeros(len(x))
            for yr in (2022, 2023, 2024):
                tr, te = x["year"] != yr, x["year"] == yr
                th = fit_params(x[tr], yy[tr.to_numpy()], tg, feats)
                tau = np.exp(th[0] + x.loc[te, feats].to_numpy(dtype=float) @ th[1:1 + len(feats)]) if feats else np.full(int(te.sum()), np.exp(th[0]))
                new[te.to_numpy()] = transform(x.loc[te, f"p_{tg}"].to_numpy(), x.loc[te, "race_id"].to_numpy(), tau, th[-1], TOT[tg])
                res.setdefault(tg, {})[str(yr)] = [float(v) for v in th]
            out_oof[f"p_{tg}"] = new
        base_df = x[KEY + ["p_win", "p_top2", "p_top3"]]
        c = t._cmp(base_df, out_oof[KEY + ["p_win", "p_top2", "p_top3"]], fin_map, f"温度調整 {vname} vs 調整なし")
        rep["variants"][vname] = {"cmp": c, "params": res}
        print(f"{vname}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "temp_results.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
