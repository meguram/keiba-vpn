"""探す側の記述: 勝率と複勝率で効く束の違い（束を 1 つずつ落とした 2 代母の単位の OOF の損失の増え。未確認・買い方に使わない）。

勝ち: レース内の対数損失の一様からの改善の減り。複勝（3 着内）: 二値の対数損失の改善の減り。束は report_stats.GROUPS。
    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.group_drop
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, v5
from src.research.pedigree_lab.report_stats import GROUPS


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "group_drop", "探す側の記述: 束を落とした損失（勝ちと複勝）"))
    X = v5.base_frame("v4")
    dev = X["split"].isin([0, 1]).to_numpy()
    full = ablate.col_sets(X)["pure"]
    res = {}
    base = v5.oof_for(X, full, "dd", None, ("win", "top3"))
    g0w, g0t = v5.gain(X, base["win"], dev), v5.top3_gain(X, base["top3"], dev)
    res["full"] = {"win": g0w, "top3": g0t, "n": len(full)}
    print("full", res["full"], flush=True)
    for nm, pat in GROUPS:
        cols = [c for c in full if not re.search(pat, c)]
        if len(cols) == len(full):
            continue
        o = v5.oof_for(X, cols, "dd", None, ("win", "top3"))
        gw, gt = v5.gain(X, o["win"], dev), v5.top3_gain(X, o["top3"], dev)
        res[nm] = {"dropped": len(full) - len(cols), "win_loss": round(g0w - gw, 5), "top3_loss": round(g0t - gt, 5),
                   "top3_share_of_win": round((g0t - gt) / g0t / max((g0w - gw) / g0w, 1e-9), 3) if g0w - gw > 0 else None}
        print(nm, res[nm], flush=True)
    path = data.OUT / "group_drop.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
