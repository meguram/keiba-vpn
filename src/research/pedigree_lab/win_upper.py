"""R0-PL1 の門 0 の補足: 単勝の上限（モデルの上の値。着順で回収率は出さない）。

E[p_Benter ÷ q] と、モデルの上の期待回収率 Σ p_Benter × 確定オッズ ÷ 点数（控除の後の値）。
p_Benter は場面の上乗せ込みの Benter をレースの単位の 5 分割の交差で当てた勝ちの確率。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.win_upper
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, gates
from src.research.pedigree_lab.hyp_tests import clogit


def benter_cross(X: pd.DataFrame, scene: np.ndarray) -> np.ndarray:
    y = X["win"].to_numpy(float)
    rc = pd.factorize(X["race_id"])[0]
    m = scene.astype(float)
    lpc, lqc = gates.centered(X, X["lp"].to_numpy()), gates.centered(X, X["lq"].to_numpy())
    Z = np.c_[X["lp"], X["lq"], lpc * m, lqc * m, m]
    fo = np.random.default_rng(7).permutation(rc.max() + 1)[rc] % 5
    pb = np.zeros(len(X))
    for k in range(5):
        A = fo != k
        bk = clogit(Z[A], y[A], pd.factorize(X.loc[A, "race_id"])[0])
        eta = Z[~A] @ bk
        rid = X.loc[~A, "race_id"].to_numpy()
        e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
        pb[~A] = e / pd.Series(e).groupby(rid).transform("sum").to_numpy()
    return pb


def main() -> None:
    X = gates.load("dd", reason="R0-PL1 の門 0 の補足（単勝の上限。回収率は見ない）")
    th = gates.thresholds(X)
    R = gates.rules(X, th)
    pb = benter_cross(X, X["FD"].to_numpy())
    q = X["q"].to_numpy()
    odds = X["odds"].to_numpy()
    out = {k: {"E_p_over_q": round(float(np.mean(pb[v] / q[v])), 4), "sum_ratio": round(float(pb[v].sum() / q[v].sum()), 4),
               "mean_odds": round(float(odds[v].mean()), 1), "model_EV_win": round(float(np.mean(pb[v] * odds[v])), 4)}
           for k, v in R.items()}
    print(json.dumps(out, ensure_ascii=False))
    (data.OUT / "gate_win_upper.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    data.finish_all(str(data.OUT / "gate_win_upper.json"))


if __name__ == "__main__":
    main()
