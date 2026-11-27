"""調べ V: ダートの行の中で、z の傾きが良馬場と道悪で違うか（記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_going

- 行はダートだけ（z_bands.py の load）。帯 = 良・道悪（稍重・重・不良）。式は z_bands.fit（帯ごとの市場の曲線を入れる）。区間は開催日の単位（500 回）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.z_bands import fit, load


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_going", "ダートの z の傾きの馬場の違い（記述）"))
    X = load()
    S = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "surface", "track_condition"])
    n0 = len(X)
    X = X.merge(S, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0
    dirt = X["surface"].eq("ダート")
    assert X.loc[dirt, "track_condition"].isin(["良", "稍重", "重", "不良"]).all()   # 黙って落とさない（critic）
    D = X[dirt].reset_index(drop=True)
    res = {"行": int(len(D)), "良と道悪": fit(D, D["track_condition"].ne("良").to_numpy().astype(int), ["良", "道悪"], np.random.default_rng(31))}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "z_going.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
