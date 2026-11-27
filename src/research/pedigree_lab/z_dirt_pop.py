"""調べ X: ダートの z の傾きが人気の帯（1〜3・4〜8・9 番人気以下）で違うか（記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_dirt_pop

- 行はダートだけ（z_bands.py の load）。帯は確定オッズの人気の順。式は z_bands.fit（帯ごとの市場の曲線を入れる）。
- fit は最後の帯 − 最初の帯の差を出す（9 番人気以下 − 1〜3 番人気）。区間は開催日の単位（500 回）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.z_bands import fit, load


def main() -> None:
    # 台帳の範囲は z_bands.py と同じ。使うのは p_A のある 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_dirt_pop", "ダートの z の傾きの人気の帯（記述）"))
    X = load()
    S = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "surface", "popularity"])
    n0 = len(X)
    X = X.merge(S, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    assert len(X) == n0
    D = X[X["surface"].eq("ダート")].reset_index(drop=True)
    pop = pd.to_numeric(D["popularity"], errors="coerce")
    assert pop.notna().all()
    band = np.where(pop <= 3, 0, np.where(pop <= 8, 1, 2))
    res = {"行": int(len(D)), "人気の帯": fit(D, band, ["1〜3 番人気", "4〜8 番人気", "9 番人気以下"], np.random.default_rng(41))}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "z_dirt_pop.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
