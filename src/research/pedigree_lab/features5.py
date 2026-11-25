"""第 5 版の案 E（1 号）: 3・4 代の牡の祖先（12 頭）の子孫の適性の血量。

- 祖先 X の子孫 = 5 代血統に X を持つ馬（2016 年からの出走）。X × 条件（全体・芝ダ・芝ダ × 距離帯）の 3 着内率を as-of
  （その日より前・自分を引く）で集め、擬似件数 200 で条件の全体の率へ縮め、条件の全体の率との差にする。
- 馬ごとに 12 頭の差を 2^−代 で重み付けて平均する（anc_lvl・anc_s・anc_sd）。レースの中の偏差（rel_anc_sd）も付ける。
- 1・2 代（父・母・父父・父母・母父・母母）は、すでに役の列で入っているので入れない。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.features5
出力: OUT/feat_v5e.parquet（第 4 版の列 ＋ 上の列）。費用 0 円。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data
from src.research.pedigree_lab.data import asof

K = 200.0
SLOTS = [(3, 0), (3, 2), (3, 4), (3, 6), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8), (4, 10), (4, 12), (4, 14)]


def anc_table(ped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    want = set(SLOTS)
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        for t in a.split(";"):
            if not t:
                continue
            g, p, i = t.split(":")
            if (int(g), int(p)) in want:
                rows.append((h, int(g), i))
    return pd.DataFrame(rows, columns=["horse_id", "gen", "anc"])


def build(src=None) -> pd.DataFrame:
    X = pd.read_parquet(src or (data.OUT / "feat_v4.parquet"))
    ped = data.load_ped()
    A = anc_table(ped[ped["horse_id"].isin(X["horse_id"].unique())])
    R = X[["date", "horse_id", "race_id", "surf", "dband", "top3"]].reset_index().rename(columns={"index": "row"})
    # 走り × 祖先（約 500 万行）。同じ祖先を 2 つの枠に持つ馬は、近い代の 1 行だけにする
    # （data.asof は「同じ鍵・同じ馬で 1 日 1 行」が前提。重なると今回の着順を自分の分として余分に引く。critic の指摘 1）
    A = A.sort_values("gen").drop_duplicates(["horse_id", "anc"])
    L = R.merge(A, on="horse_id", how="inner")
    L = L.sort_values(["date", "row"], kind="stable").reset_index(drop=True)
    out = {}
    for nm, conds in (("lvl", []), ("s", ["surf"]), ("sd", ["surf", "dband"])):
        s, c = asof(L, ["anc"] + conds, ["top3"])
        # 基準（条件の全体の率）は走りの表（1 走 1 行）で作り、祖先の行へ写す
        bs, bc = asof(R.assign(_k="1"), ["_k"] + conds, ["top3"])
        base_r = np.where(bc > 30, bs[:, 0] / np.maximum(bc, 1), 0.2)
        base = base_r[L["row"].to_numpy()]
        rate = (s[:, 0] + K * base) / (c + K)
        w = 0.5 ** L["gen"].to_numpy()
        d = pd.DataFrame({"row": L["row"], "wd": w * (rate - base), "w": w})
        g = d.groupby("row")[["wd", "w"]].sum()
        out[f"anc_{nm}"] = (g["wd"] / g["w"]).reindex(np.arange(len(X))).to_numpy().astype("float32")
    for k, v in out.items():
        X[k] = v
    X["rel_anc_sd"] = (X["anc_sd"] - X.groupby("race_id")["anc_sd"].transform("mean")).astype("float32")
    return X


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "feat_v5e", "第 5 版の案 E の特徴量（祖先の子孫の適性）"))
    X = build()
    X.to_parquet(data.OUT / "feat_v5e.parquet", index=False)
    print("feat_v5e", X.shape, X[["anc_lvl", "anc_s", "anc_sd"]].describe().round(4).to_string())
    data.finish_all(str(data.OUT / "feat_v5e.parquet"))


if __name__ == "__main__":
    main()
