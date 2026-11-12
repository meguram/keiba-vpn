"""障害の残差モデル（実験）。親 v6e16 の logit を土台（init_score）にして、障害の行だけで小さな木を足し、障害内の勝率 LogLoss が親より良くなるかを年別 LOYO（2022〜2025）で測る。

    python -m src.scripts.data.hurdle_resid --feat /tmp/hurdle/hurdle_features.parquet --out /tmp/hurdle/resid_eval.json

hurdle_model.py の出力（hurdle_preds.parquet は p_win を含む）を入力にする。市場オッズ・人気は特徴量にしない。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.scripts.data.hurdle_model import boot_diff, race_logloss

SEED = 20261111
DROP = {"race_id", "horse_id", "date", "h", "win", "fin", "year", "p_win", "win_odds", "q"}


def run(H: pd.DataFrame, feats: list[str], n_est: int, leaves: int, lr: float) -> dict:
    import lightgbm as lgb

    E = H[H["p_win"].notna()].copy()
    E["off"] = np.log(np.clip(E["p_win"], 1e-9, 1 - 1e-9) / (1 - np.clip(E["p_win"], 1e-9, 1 - 1e-9)))
    E["p_res"] = np.nan
    for y in sorted(E["year"].unique()):
        tr, te = E[E["year"] != y], E[E["year"] == y]
        m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=lr, num_leaves=leaves, min_child_samples=60, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=20.0, random_state=SEED, verbose=-1, n_jobs=2)
        m.fit(tr[feats], tr["win"], init_score=tr["off"])
        raw = m.predict(te[feats], raw_score=True) + te["off"].to_numpy()
        E.loc[te.index, "p_res"] = 1 / (1 + np.exp(-raw))
    base = pd.concat([race_logloss(E[E["year"] == y], E.loc[E["year"] == y, "p_win"].to_numpy()) for y in sorted(E["year"].unique())])
    new = pd.concat([race_logloss(E[E["year"] == y], E.loc[E["year"] == y, "p_res"].to_numpy()) for y in sorted(E["year"].unique())])
    d, lo, hi = boot_diff(new, base)
    by_year = {}
    for y in sorted(E["year"].unique()):
        a = race_logloss(E[E["year"] == y], E.loc[E["year"] == y, "p_res"].to_numpy())
        b = race_logloss(E[E["year"] == y], E.loc[E["year"] == y, "p_win"].to_numpy())
        by_year[int(y)] = float((a - b.reindex(a.index)).mean())
    return {"parent": float(base.mean()), "resid": float(new.mean()), "diff": d, "lo": lo, "hi": hi, "by_year": by_year, "races": int(len(base))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="/tmp/hurdle/hurdle_preds.parquet")
    ap.add_argument("--out", default="/tmp/hurdle/resid_eval.json")
    a = ap.parse_args()
    H = pd.read_parquet(a.feat)
    feats = [c for c in H.columns if c not in DROP and not c.startswith("p_") and c != "lh"]
    res = {"feats": feats}
    for n_est, leaves, lr in ((50, 3, 0.02), (100, 3, 0.02), (200, 3, 0.02), (100, 7, 0.02)):
        k = f"n{n_est}_l{leaves}"
        res[k] = run(H, feats, n_est, leaves, lr)
        print(k, json.dumps(res[k], ensure_ascii=False), flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
