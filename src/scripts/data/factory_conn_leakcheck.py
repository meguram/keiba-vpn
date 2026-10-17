"""ファクトリー G2: 実データで『将来の結果を変えても過去行の特徴量が変わらない』ことを確認する。

python -m src.scripts.data.factory_conn_leakcheck [--cut 2022-06-01] [--ents o,to,j]
2022-06-01 以降の着順・上がりをランダムに入れ替えた履歴で特徴量を作り直し、それ以前の行の値が完全に一致するかを列ごとに調べる。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.pipeline.features import factory_conn as fc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2022-06-01")
    ap.add_argument("--ents", default="o,to,j,t")
    a = ap.parse_args()
    ents = a.ents.split(",")
    raw = fc.load_history()
    rng = np.random.default_rng(0)
    raw2 = raw.copy()
    late = pd.to_datetime(raw2["date"]) >= pd.Timestamp(a.cut)
    fin = pd.to_numeric(raw2["finish_position"], errors="coerce")
    sh = fin[late].to_numpy().copy()
    rng.shuffle(sh)
    raw2.loc[late, "finish_position"] = sh
    l3 = pd.to_numeric(raw2["last_3f"], errors="coerce")
    s3 = l3[late].to_numpy().copy()
    rng.shuffle(s3)
    raw2.loc[late, "last_3f"] = s3
    g1 = fc.build_all(fc.prepare(raw), entities=ents, conds=["surf", "lay"])
    g2 = fc.build_all(fc.prepare(raw2), entities=ents, conds=["surf", "lay"])
    early = (pd.to_datetime(raw["date"]) < pd.Timestamp(a.cut)).to_numpy()
    bad, n = [], 0
    for grp, cols in g1.items():
        for c, x in cols.items():
            y = g2[grp][c]
            ok = np.allclose(np.asarray(x)[early], np.asarray(y)[early], rtol=1e-5, atol=1e-6, equal_nan=True)
            n += 1
            if not ok:
                bad.append(f"{grp}.{c}")
    print(f"確認した列 {n} / 先読みの疑い {len(bad)}: {bad[:20]}")


if __name__ == "__main__":
    main()
