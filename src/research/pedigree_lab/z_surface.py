"""z の効きの芝とダートの違い（log.md「調べ U」。記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_surface

- 行と式は z_bands.py と同じ（帯ごとの市場の曲線を入れる）。障害は除く。
- 主の値は z の傾きの差（ダート − 芝）。区間は開催日の単位（500 回）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.z_bands import fit, load


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_surface", "z の効きの芝ダの違い（記述）"))
    X = load()
    S = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "surface"])
    n0 = len(X)
    X = X.merge(S, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0
    X = X[X["surface"].isin(["芝", "ダート"])].reset_index(drop=True)
    rng = np.random.default_rng(17)
    res = {"行": int(len(X)), "芝ダ": fit(X, X["surface"].eq("ダート").to_numpy().astype(int), ["芝", "ダート"], rng)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "z_surface.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
