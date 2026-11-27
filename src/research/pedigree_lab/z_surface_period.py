"""調べ W: 芝ダの z の傾きの差を 2 期（2019〜21 年・2022〜24 年）に分けて、区間つきで見る（記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_surface_period
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.z_bands import fit, load


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_surface_period", "芝ダの差の 2 期（記述）"))
    X = load()
    S = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "surface"])
    n0 = len(X)
    X = X.merge(S, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0 and X["surface"].isin(["芝", "ダート"]).all()
    res = {}
    for nm, (a, b) in {"2019〜21 年": (2019, 2021), "2022〜24 年": (2022, 2024)}.items():
        D = X[X["y"].between(a, b)].reset_index(drop=True)
        res[nm] = fit(D, D["surface"].eq("ダート").to_numpy().astype(int), ["芝", "ダート"], np.random.default_rng(37))
        print(nm, json.dumps(res[nm], ensure_ascii=False), flush=True)
    path = data.OUT / "z_surface_period.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
