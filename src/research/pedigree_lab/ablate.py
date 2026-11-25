"""段 2 の切り分け: 列の束を外す・足すときの対数損失の差（前向きの分け方・勝ち）。試した数として trials.jsonl に残す。

    nice -n 19 python -m src.research.pedigree_lab.ablate --tag v2 --sets age_only,noage,main,sub,main_plus_age
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, model


def col_sets(X: pd.DataFrame) -> dict[str, list[str]]:
    allc = model.feature_cols(X, sub=True)          # 血統の列を先に決める（下の比べの列 st_* が許可の正規表現に当たらないように）
    X["st_age"] = X["age"].astype(float)
    X["st_sex_f"] = (X["sexb"] == "F").astype(float)
    X["st_career"] = X["career"].astype(float)
    X["st_rel_age"] = X["st_age"] - X.groupby("race_id")["st_age"].transform("mean")
    main = [c for c in allc if not pd.Series([c]).str.contains(model.SUB_RE).iloc[0]]
    noage = [c for c in main if "agesurf" not in c]
    return {
        "age_only": ["st_age", "st_sex_f", "st_rel_age"],
        "state_only": ["st_age", "st_sex_f", "st_rel_age", "st_career"],
        "main": main,
        "noage": noage,
        "pure": noage,                                            # 主: 年齢の情報を一切入れない純粋な血統
        "ped_age": noage + ["st_age", "st_sex_f", "st_rel_age"],  # 副: 血統 ＋ 年齢・性（血統ではないと明記して並べる）
        "noage_plus_age": noage + ["st_age", "st_sex_f", "st_rel_age"],
        "main_plus_age": main + ["st_age", "st_sex_f", "st_rel_age"],
        "sub": allc,
        "sub_plus_state": allc + ["st_age", "st_sex_f", "st_rel_age", "st_career"],
        "sire_only": [c for c in noage if c.startswith(("sire_", "rel_sire_", "ss_", "sss_"))],
        "bms_only": [c for c in noage if c.startswith(("bms_", "rel_bms_"))],
        "dam_only": [c for c in noage if c.startswith(("dam_", "dd_", "full_", "rel_dam", "rel_dd", "rel_full"))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--sets", default="age_only,noage,main,main_plus_age,sub")
    ap.add_argument("--split", default="forward")
    ap.add_argument("--target", default="win")
    a = ap.parse_args()
    X = pd.read_parquet(data.OUT / f"feat_{a.tag}.parquet")
    S = col_sets(X)
    for nm in a.sets.split(","):
        t = time.time()
        o = model.fit_eval(X, a.target, a.split, cols=S[nm], verbose=False)
        r = o["res"] | {"tag": a.tag, "set": nm, "sec": round(time.time() - t)}
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                          if k in ("set", "split", "target", "n_feat", "va_gain_vs_uni", "te_gain_vs_uni", "temp")}, ensure_ascii=False), flush=True)
        with open(data.OUT / "trials.jsonl", "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
