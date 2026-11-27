"""調べ U の数を見た後の確かめ: 年ごとの芝ダの z の傾きの差（点の値だけ・記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_surface_years
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_market import irls
from src.research.pedigree_lab.z_bands import load


def slope_diff(D: pd.DataFrame) -> tuple[float, float, float]:
    lq, z = D["lq"].to_numpy(), D["z"].to_numpy()
    ind = D["surface"].eq("ダート").to_numpy(float)
    M = np.column_stack([np.ones(len(D)), lq, lq ** 2, z, ind, z * ind, lq * ind, lq ** 2 * ind])
    c = irls(M, D["win"].to_numpy(float))
    return float(c[3]), float(c[3] + c[5]), float(c[5])


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_surface_years", "調べ U の年ごとの確かめ（記述）"))
    X = load()
    S = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "surface"])
    n0 = len(X)
    X = X.merge(S, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0
    X = X[X["surface"].isin(["芝", "ダート"])]
    res = {}
    for y, D in X.groupby("y"):
        t, d, df = slope_diff(D)
        res[int(y)] = {"芝": round(t, 4), "ダート": round(d, 4), "差": round(df, 4)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "z_surface_years.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
