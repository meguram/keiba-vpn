"""final_eval の予行（2025・26 年に触れない）。開発の行だけで作り直し、2023〜24 年を評価の側に見立てて evaluate を通す。
コードが途中で落ちないことだけを確かめる（数は判定に使わない）。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.final_dryrun
"""
from __future__ import annotations

import pandas as pd

from src.research.pedigree_lab import data, final_eval, gates


def main() -> None:
    X = final_eval.build_all("v4", check_only=True)
    X = X[X["split"] != 2].reset_index(drop=True)          # 保留（split 2）は使わない（critic の指摘 2）
    F = gates.scenes(X[gates.FCOLS].copy())
    X["FD"], X["SD"] = F["FD"].to_numpy(), F["SD"].to_numpy()
    p = final_eval.predict(X, "v4")
    old = data.DEV_END
    data.DEV_END = pd.Timestamp("2022-12-31")
    try:
        res = final_eval.evaluate(X, p, 0.08152)
    finally:
        data.DEV_END = old
    print("dryrun ok（数は出さない）", sorted(res))
    data.finish_all()


if __name__ == "__main__":
    main()
