"""RPCI（レースのペースチェンジ指数）・PCI（馬ごとの道中の脚の使い方）に基づく特徴量 G（接頭辞 pc_）。

- rpci: ラップ（1F ごと）から、50 × （ラスト 3F を除く 1F 平均 ÷ ラスト 3F の 1F 平均）。小さいほど前傾（ハイペース）。
- pci（馬）: 走破タイムと上り 3F から、同じ式を馬ごとに計算。小さいほど道中を速く走った（前に行った）。
- JRA の公式式は未確認のため自前定義。前半 3F と後半 3F の差（pace_bal）と違い、距離で正規化され、中盤も含む。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .common import FIT_END
from .profile import H, ewm


def pace_index(F: pl.DataFrame, P: dict, S: dict | None = None) -> pl.DataFrame:
    frozen = bool(S and S.get("frozen"))
    dm = pl.col("distance").cast(pl.Float64)
    valid = (dm >= 1000) & pl.col("l3f").is_between(30, 50) & (pl.col("time_sec") > 0)
    front = (pl.col("time_sec") - pl.col("l3f")) / ((dm - 600) / 200)
    F = F.with_columns(pl.when(valid).then(50 * front / (pl.col("l3f") / 3)).alias("_pci"))
    F = F.with_columns(pl.when(pl.col("_pci").is_between(30, 80)).then(pl.col("_pci")).alias("_pci"))
    F = F.with_columns((pl.col("_pci") - pl.col("rpci")).alias("_pcrel"))

    # 距離帯ごとの RPCI の分位（前傾／平均／後傾）。FIT_END 以前のレースだけで決める。
    if frozen:
        lo, hi = S["rpci_lo"], S["rpci_hi"]
    else:
        fit = (F.filter((pl.col("date") <= FIT_END) & pl.col("rpci").is_not_null())
               .group_by("race_id").agg(pl.col("rpci").first(), pl.col("dist_band").first()))
        q = fit.group_by("dist_band").agg(pl.col("rpci").quantile(0.33).alias("lo"), pl.col("rpci").quantile(0.67).alias("hi"))
        lo = {int(r["dist_band"]): float(r["lo"]) for r in q.iter_rows(named=True)}
        hi = {int(r["dist_band"]): float(r["hi"]) for r in q.iter_rows(named=True)}
        if S is not None:
            S["rpci_lo"], S["rpci_hi"] = lo, hi
    P["rpci_q"] = {k: (lo[k], hi[k]) for k in lo}
    F = F.with_columns(
        pl.when(pl.col("rpci").is_null()).then(None)
        .when(pl.col("rpci") <= pl.col("dist_band").replace_strict(lo, default=-99.0, return_dtype=pl.Float64)).then(0)
        .when(pl.col("rpci") >= pl.col("dist_band").replace_strict(hi, default=99.0, return_dtype=pl.Float64)).then(2)
        .otherwise(1).alias("_reg_r"))

    F = F.with_columns(
        ewm("_pci", 2).alias("pc_pci_ewm2"), ewm("_pci", 6).alias("pc_pci_ewm6"),
        ewm("_pcrel", 2).alias("pc_pcrel_ewm2"), ewm("_pcrel", 6).alias("pc_pcrel_ewm6"),
        ewm("rpci", 3).alias("pc_rpci_ewm3"),
    ).with_columns((pl.col("pc_pci_ewm2") - pl.col("pc_pci_ewm6")).alias("pc_pci_trend"))

    # 過去走を RPCI の 3 区分に分けた、区分別の相対成績（縮約）。前傾・後傾への相性。
    kap = P["kappa_all"]
    for k in range(3):
        ind = (pl.col("_reg_r") == k).fill_null(False)
        r_k = pl.when(ind).then(pl.col("res")).otherwise(0.0)
        F = F.with_columns((r_k.cum_sum().over(H) - r_k).alias(f"_rs{k}"),
                           (ind.cast(pl.Float64).cum_sum().over(H) - ind.cast(pl.Float64)).alias(f"pc_n_reg{k}"))
        F = F.with_columns((pl.col(f"_rs{k}") / (pl.col(f"pc_n_reg{k}") + kap)).alias(f"pc_a_reg{k}")).drop(f"_rs{k}")
    F = F.with_columns((pl.col("pc_a_reg0") - pl.col("pc_a_reg2")).alias("pc_adv"))
    return F.drop("_pci", "_pcrel", "_reg_r")
