"""8 観点の牝系の版の確かめ（1 号・2 号の R2。記述・未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.aptitude_checks <観点> [--boot 200]

- mat_deep: 牝系から母父（2 代）を外し、3〜5 代の母の側の牡の祖先だけで作る（母父との共線の疑い。1 号 1-1・2 号 候補 1）。
- mat_level: 同じ祖先・同じ血量の重みで「全条件の水準 − 全体の水準」（母方の家系の質）を作り、制御に足す（2 号 候補 2）。
- comp: 対になる型（pace_h ↔ pace_s・short ↔ ext）の牝系の列を、この型の行に当てる（適性なら負、質なら正）。
- 母父と牝系の列の相関・別々に当てた係数・2 本の和の係数。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.explore2 import fit_boot
from src.research.pedigree_lab.explore5 import attach, sim_ci

COMP = {"pace_h": "pace_s", "pace_s": "pace_h", "short": "ext", "ext": "short"}


def level_by(D: pd.DataFrame, keys: list[str], base_t: np.ndarray) -> np.ndarray:
    """鍵の全条件の as-of の平均（全体へ擬似件数 K で縮める）− 全体の平均。"""
    g = -D["fpct"].to_numpy()
    s, _ = asof(D.assign(_t=g, _nt=1.0), keys, ["_t", "_nt"])
    mT = (s[:, 0] + ap_.K * base_t) / (s[:, 1] + ap_.K)
    return mT - base_t


def mat_cols(F: pd.DataFrame, A: np.ndarray, keys: list[str], L: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row = L["row"].to_numpy()
    mA0, mT0 = ap_.base_means(F, A, keys)
    d = ap_.apt_by(L, A[row], ["anc"] + keys, base=(mA0[row], mT0[row]))
    lv = level_by(L, ["anc"] + keys, mT0[row])
    wt = L["bw"].to_numpy()
    agg = pd.DataFrame({"row": row, "wd": wt * d, "wl": wt * lv, "w": wt}).groupby("row")[["wd", "wl", "w"]].sum()
    n = np.arange(len(F))
    return ((agg["wd"] / agg["w"]).reindex(n).fillna(0.0).to_numpy(), (agg["wl"] / agg["w"]).reindex(n).fillna(0.0).to_numpy())


def fit(S: pd.DataFrame, cols: list[np.ndarray], names: list[str], market: bool, n_boot: int) -> dict:
    base = [S["lp"].to_numpy(), S["lq"].to_numpy()] if market else [S["lp"].to_numpy()]
    M = np.c_[tuple(base) + tuple(cols)]
    idx = {n: len(base) + i for i, n in enumerate(names)}
    b, bs = fit_boot(S, M, n_boot)
    c, r = sim_ci(b, bs, idx)
    out = {"c": round(c, 3), **{k: {"gamma": v["gamma"], "sim_ci": v["sim_ci"]} for k, v in r.items()}}
    if "bms" in idx and "mat5" in idx:
        sm = bs[:, idx["bms"]] + bs[:, idx["mat5"]]
        out["bms_plus_mat5"] = [round(float(b[idx["bms"]] + b[idx["mat5"]]), 4), np.percentile(sm, [2.5, 97.5]).round(4).tolist()]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what")
    p.add_argument("--boot", type=int, default=200)
    a = p.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"aptitude_checks_{a.what}", f"8 観点の牝系の確かめ: {a.what}（尤度だけ・記述）"))
    F = ap_.base_frame()
    A, keys = ap_.group_a(F, a.what)
    L = ap_.maternal_long(F, drop_common=False)
    Ld = ap_.maternal_long(F, drop_common=False, min_gen=3)
    F["z_sire"] = ap_.apt_by(F, A, ["sire"] + keys)
    F["z_bms"] = ap_.apt_by(F, A, ["ds"] + keys)
    F["z_mat5"], F["z_lv5"] = mat_cols(F, A, keys, L)
    F["z_matd"], F["z_lvd"] = mat_cols(F, A, keys, Ld)
    if a.what in COMP:
        Ac, kc = ap_.group_a(F, COMP[a.what])
        F["z_comp"], _ = mat_cols(F, Ac, kc, L)
    F["_A"] = A
    X = attach(F)
    S = X[X["_A"].astype(bool)].reset_index(drop=True)
    Z = {k: ap_.zA(S, S[f"z_{k}"].to_numpy()) for k in ["sire", "bms", "mat5", "lv5", "matd", "lvd"] + (["comp"] if "z_comp" in S else [])}
    ctrl = ap_.zA(S, S["sire_all_fpct"].fillna(0).to_numpy())
    res = {"what": a.what, "rows": int(len(S)),
           "corr_bms_mat5": round(float(np.corrcoef(Z["bms"], Z["mat5"])[0, 1]), 3),
           "corr_bms_matd": round(float(np.corrcoef(Z["bms"], Z["matd"])[0, 1]), 3),
           "corr_mat5_lv5": round(float(np.corrcoef(Z["mat5"], Z["lv5"])[0, 1]), 3)}
    for market in (False, True):
        tag = "mkt" if market else "nomkt"
        res[f"{tag}_bms_only"] = fit(S, [ctrl, Z["bms"]], ["ctrl", "bms"], market, a.boot)["bms"]
        res[f"{tag}_mat5_only"] = fit(S, [ctrl, Z["mat5"]], ["ctrl", "mat5"], market, a.boot)["mat5"]
        res[f"{tag}_both"] = fit(S, [ctrl, Z["sire"], Z["bms"], Z["mat5"]], ["ctrl", "sire", "bms", "mat5"], market, a.boot)
        res[f"{tag}_deep"] = fit(S, [ctrl, Z["sire"], Z["bms"], Z["matd"]], ["ctrl", "sire", "bms", "matd"], market, a.boot)
        res[f"{tag}_level_ctrl"] = fit(S, [ctrl, Z["lv5"], Z["sire"], Z["bms"], Z["mat5"]], ["ctrl", "lv5", "sire", "bms", "mat5"], market, a.boot)
        if "comp" in Z:
            res[f"{tag}_comp"] = fit(S, [ctrl, Z["lv5"], Z["comp"]], ["ctrl", "lv5", "comp"], market, a.boot)["comp"]
        print(a.what, tag, json.dumps({k: v for k, v in res.items() if k.startswith(tag)}, ensure_ascii=False), flush=True)
    path = data.OUT / f"aptitude_checks_{a.what}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
