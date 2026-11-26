"""探す側の記述の 4 本目: 父・母父の埋め込み（産駒の成績の行列の分解。統括役の例 (1)。未確認）。

年 t の行には、t の前の年末までの走りだけで作った行列を使う（as-of は年の単位）。自分の過去走を引くため、行列は
その馬の生まれ年の世代の産駒を丸ごと除いて作る（世代 b × 年 t ごとに 1 回の分解。行列は小さいので軽い）。
- 行列: 役（父・母父）× マス（芝ダ × 距離帯 × レースの年齢条件 = 40）の 3 着内率の、マスの平均からの残差を擬似件数 200 で縮めたもの。
  産駒が 200 走以上の役だけ。
- 特異値分解の上位 R=6 で復元した値のうち、今回のレースのマスの値を特徴量にする（*_lr）。レースの中の偏差も付ける。
    nice -n 19 python -m src.research.pedigree_lab.features6
出力: OUT/feat_v6e.parquet（第 4 版の列 ＋ sire_lr・bms_lr・rel_）。費用 0 円。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data

R = 6
K = 200.0
MIN_RUNS = 200


def lowrank_table(D: pd.DataFrame, role: str) -> pd.DataFrame:
    """D（役・マス・3 着内）から、役 × マスの低い階数の復元の残差。"""
    cell_mean = D.groupby("cell")["top3"].mean()
    g = D.groupby([role, "cell"])["top3"].agg(["sum", "size"])
    n_role = g.groupby(level=0)["size"].sum()
    keep = n_role[n_role >= MIN_RUNS].index
    g = g[g.index.get_level_values(0).isin(keep)]
    if g.empty:
        return pd.DataFrame(columns=[role, "cell", "lr"])
    cm = cell_mean.reindex(g.index.get_level_values(1)).to_numpy()
    resid = (g["sum"].to_numpy() + K * cm) / (g["size"].to_numpy() + K) - cm
    M = pd.Series(resid, index=g.index).unstack(fill_value=0.0)
    U, S, Vt = np.linalg.svd(M.to_numpy(), full_matrices=False)
    r = min(R, len(S))
    L = (U[:, :r] * S[:r]) @ Vt[:r]
    out = pd.DataFrame(L, index=M.index, columns=M.columns).stack().rename("lr").reset_index()
    out.columns = [role, "cell", "lr"]
    return out


def build(src=None) -> pd.DataFrame:
    X = pd.read_parquet(src or (data.OUT / "feat_v4.parquet"))
    X["cell"] = X["surf"] + "_" + X["dband"] + "_" + X["rage"]
    X["year"] = X["date"].dt.year
    X["by"] = X["by"].astype(str)
    for role, col in (("sire", "sire"), ("bms", "ds")):
        X[f"{role}_lr"] = np.nan
        for t in sorted(X["year"].unique()):
            if t <= 2016:
                continue
            past = X.loc[X["year"] < t, [col, "cell", "top3", "by"]]
            rows_t = X["year"] == t
            for b in X.loc[rows_t, "by"].unique():
                D = past[past["by"] != b]                       # その馬の世代を除く（自分を引く）
                T = lowrank_table(D, col)
                m = rows_t & (X["by"] == b)
                sub = X.loc[m, [col, "cell"]].merge(T, on=[col, "cell"], how="left")
                X.loc[m, f"{role}_lr"] = sub["lr"].to_numpy()
        X[f"rel_{role}_lr"] = (X[f"{role}_lr"] - X.groupby("race_id")[f"{role}_lr"].transform("mean")).astype("float32")
    return X.drop(columns=["cell", "year"])


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "feat_v6e", "探す側の記述: 父・母父の埋め込みの列"))
    X = build()
    X.to_parquet(data.OUT / "feat_v6e.parquet", index=False)
    print("feat_v6e", X.shape, X[["sire_lr", "bms_lr"]].describe().round(4).to_string())
    data.finish_all(str(data.OUT / "feat_v6e.parquet"))


if __name__ == "__main__":
    main()
