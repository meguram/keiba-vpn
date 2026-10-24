"""特徴量の追加が効くかを、基準モデルとの OOF 比較で判定する（tp-15・tp-16・tp-17 共通）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.models import win_models as wm

KEY = ["race_id", "horse_id"]


def compare_oof(oof_base: pd.DataFrame, oof_new: pd.DataFrame, fin_map: pd.Series, name: str, n_boot: int = 1000, min_effect: float = 2e-4, tol: float = 2e-4) -> dict:
    """採用 = 勝率 LogLoss の差の 95% 信頼区間の上限 < 0、効果 ≥ min_effect、連対・複勝が tol を超えて悪化しない。"""
    m = oof_base.merge(oof_new, on=KEY, suffixes=("_b", "_v"))
    fin = fin_map.loc[list(zip(m["race_id"], m["horse_id"]))].to_numpy()
    out: dict = {"variant": name}
    for t_, k in (("win", 1), ("top2", 2), ("top3", 3)):
        yy = (fin <= k).astype(float)
        rid = m["race_id"].to_numpy()
        bb = wm.paired_bootstrap_diff(wm.per_race_loss(rid, m[f"p_{t_}_b"].to_numpy(), yy), wm.per_race_loss(rid, m[f"p_{t_}_v"].to_numpy(), yy), n_boot=n_boot)
        out[f"{t_}_base"] = wm.binary_logloss(m[f"p_{t_}_b"].to_numpy(), yy)
        out[f"{t_}_new"] = wm.binary_logloss(m[f"p_{t_}_v"].to_numpy(), yy)
        out[f"Δ{t_}"], out[f"{t_}_hi"] = bb["diff"], bb["hi"]
    out["採用"] = bool(out["win_hi"] < 0 and out["Δwin"] <= -min_effect and out["Δtop2"] <= tol and out["Δtop3"] <= tol)
    return out
