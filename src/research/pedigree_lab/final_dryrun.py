"""final_eval の予行（2025・26 年に触れない）。開発の行だけで作り直し、2023〜24 年を評価の側に見立てて evaluate を通す。
コードが途中で落ちないことだけを確かめる（数は判定に使わない）。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.final_dryrun
"""
from __future__ import annotations

import json

import pandas as pd

from src.research.pedigree_lab import data, final_eval, gates


def main() -> None:
    X = final_eval.build_all("v4", check_only=True)
    F = gates.scenes(X[gates.FCOLS].copy())
    X["FD"], X["SD"] = F["FD"].to_numpy(), F["SD"].to_numpy()
    p = final_eval.predict(X, "v4")
    old = data.DEV_END
    data.DEV_END = pd.Timestamp("2022-12-31")
    try:
        res = final_eval.evaluate(X, p, 0.08152)
    finally:
        data.DEV_END = old
    print("dryrun ok", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "deciles"})
                                   for k, v in res.items()}, ensure_ascii=False, default=str)[:1500])
    data.finish_all()


if __name__ == "__main__":
    main()
