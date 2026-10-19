"""tp-38 F2: リッジ + 位置の重みの推定を実行して群を保存する（python -m src.scripts.data.tp38_eval ridge）。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.pipeline.features import tp38_pedigree as P
from src.pipeline.features import tp38_ridge as RG
from src.scripts.data import tp38_eval as E


def windows(start: str = "2021-01-01", end: str = "2026-01-01") -> list[tuple[str, str]]:
    ds = pd.date_range(start, end, freq="6MS")
    return [(str(a.date()), str(b.date())) for a, b in zip(ds[:-1], ds[1:])]


def main() -> None:
    df, base, meta, folds, lp = E.t.setup()
    anc, horse_index, uniq = E._load_static()
    R = P.Runs(df, horse_index)
    t0 = time.time()
    res, phis = RG.ridge_features(R, anc, len(horse_index), len(uniq), windows(), log=E._log)
    E._log(f"リッジ完了 {time.time() - t0:.0f}s")
    meta_out = E._jload(E.OUT / "build_meta.json")
    for k, name in (("fix", "PX_ridge_fix"), ("als", "PX_ridge_als"), ("als2", "PX_ridge_als2")):
        F = E.add_rel(res[k], df["race_id"], ["z", "zc"])
        meta_out[name] = E._save_group(name, F, df)
    E._jsave(E.OUT / "build_meta.json", meta_out)
    E._jsave(E.OUT / "ridge_phi.json", {"phis": phis, "pos_gen": RG.POS_GEN.tolist(), "pos_pos": RG.POS_POS.tolist(), "prior": RG.POS_PRIOR.tolist()})
    phi = np.array([p["phi"] for p in phis])
    for p in (0, 1, 2, 3, 4, 5, 6, 8, 14, 30, 61):
        E._log(f"  φ[{RG.POS_GEN[p]}世代 {P.pos_name(int(RG.POS_GEN[p]), int(RG.POS_POS[p]))}] 平均 {phi[:, p].mean():.3f} (理論 {RG.POS_PRIOR[p]:.3f})")
