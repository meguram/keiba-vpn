"""87 列の小さなモデルの時間の順の予測（walkforward.fit_year と同じ作り・勝ちだけ）。出力: OUT/oof_v4_compact.parquet。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.compact_wf
"""
from __future__ import annotations

import numpy as np

from src.research.pedigree_lab import compact_model, data
from src.research.pedigree_lab.walkforward import fit_year


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "compact_wf", "87 列の小さなモデルの時間の順の予測（尤度だけ）"))
    X, sets = compact_model.frame()
    cols = sets["compact"]
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, cols, "win", t, 2000)
        print(t, flush=True)
    out = X.loc[yr >= 2019, ["race_id", "horse_id"]].copy()
    out["p_win"] = p[yr >= 2019]
    path = data.OUT / "oof_v4_compact.parquet"
    out.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
