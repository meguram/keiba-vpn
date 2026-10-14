"""標準条件レーティング H（接頭辞 sr_）。

斤量・性別・年齢の違いを取り除いた「標準条件（牡・4 歳以上・56kg 相当）でのパフォーマンス」を過去走ごとに求めて保持し、
今回の条件（斤量・馬体重・性別・年齢）で期待パフォーマンスに戻す。

- 時計は log で扱い、par（芝ダ×距離×クラスの中央値。fit 期間で決定）と当日の馬場差（日×場×芝ダの中央値）を引く。
- 負担の指標は 斤量 × 基準体重(480) ÷ 馬体重。斤量の係数 β（秒/kg、par 時計に対する比で適用）は固定せず、
  候補（BETA_GRID）を fit 期間の「次走の実測パフォーマンスの予測誤差」で距離帯×芝ダ別に選ぶ。
- 性別は芝ダ別の固定効果（レース内比較）、年齢は年齢×月の成長曲線（馬の固定効果）で推定する。いずれも fit 期間のみ。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .common import FIT_END
from .profile import H, ewm

LOAD_REF, BODY_REF = 56.0, 480.0
BETA_GRID = (0.0, 0.05, 0.10, 0.17, 0.25)  # 秒/kg（par 時計の比で適用）
BETA_EST = 0.10  # 性別・年齢の推定時に使う仮の係数
AGE_EDGES = (2.5, 3.0, 3.5, 4.0)  # 年齢 + (月−1)/12 の区切り。4.0 以上が基準


def _age_bin():
    t = pl.col("age").cast(pl.Float64) + (pl.col("date").dt.month().cast(pl.Float64) - 1.0) / 12.0
    return (pl.when(t < AGE_EDGES[0]).then(0).when(t < AGE_EDGES[1]).then(1)
            .when(t < AGE_EDGES[2]).then(2).when(t < AGE_EDGES[3]).then(3).otherwise(4))


def _within(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    return y - (np.bincount(g, y) / np.bincount(g))[g]


def standard_rating(F: pl.DataFrame, P: dict) -> pl.DataFrame:
    F = F.sort([H, "date", "race_id"])
    good = (pl.col("time_sec") > 40) & pl.col("surface").is_in(["芝", "ダート"])
    wt_f = pl.col("wt").forward_fill().over(H).fill_null(BODY_REF).clip(380, 600)
    F = F.with_columns(
        pl.when(good).then(pl.col("time_sec").log()).alias("_ll"),
        (pl.col("jockey_weight") * BODY_REF / wt_f).alias("_load"),
    )
    F = F.with_columns(pl.col("_ll").mean().over("race_id").alias("_rll"),
                       (pl.col("_ll").count().over("race_id") >= 5).alias("_okr"))
    F = F.with_columns(pl.when(pl.col("_okr")).then(pl.col("_rll")).alias("_rll"))

    # par: fit 期間のレースの中央値（細かい条件 → 粗い条件の順にフォールバック）
    fr = (F.filter((pl.col("date") <= FIT_END) & pl.col("_rll").is_not_null())
          .group_by("race_id").agg(pl.col("surface").first(), pl.col("distance").first(), pl.col("class_level").first(),
                                   pl.col("dist_band").first(), pl.col("_rll").first()))
    tabs = [(["surface", "distance", "class_level"], 5), (["surface", "distance"], 5)]
    names = []
    for i, (keys, mn) in enumerate(tabs):
        tb = (fr.group_by(keys).agg(pl.col("_rll").median().alias(f"_par{i}"), pl.len().alias("_n"))
              .filter(pl.col("_n") >= mn).drop("_n"))
        F = F.join(tb, on=keys, how="left")
        names.append(f"_par{i}")
    # 最後のフォールバック: 芝ダ別に log時計 = a + b·log(距離) を fit 期間のレースで回帰（距離帯の平均は 2500〜3600m などで破綻する）
    reg = {}
    for sf in ("芝", "ダート"):
        g = fr.filter(pl.col("surface") == sf)
        if g.height >= 30:
            b, a = np.polyfit(np.log(g["distance"].to_numpy().astype(float)), g["_rll"].to_numpy(), 1)
            reg[sf] = (float(a), float(b))
    P["sr_par_reg"] = reg
    a_s = {k: v[0] for k, v in reg.items()}; b_s = {k: v[1] for k, v in reg.items()}
    F = F.with_columns((pl.col("surface").replace_strict(a_s, default=None, return_dtype=pl.Float64)
                        + pl.col("surface").replace_strict(b_s, default=None, return_dtype=pl.Float64) * pl.col("distance").cast(pl.Float64).log()).alias("_parreg"))
    F = F.sort([H, "date", "race_id"]).with_columns(pl.coalesce(names + ["_parreg"]).alias("_par"))
    F = F.with_columns((pl.col("_rll") - pl.col("_par")).alias("_dev"))
    dv = (F.group_by("race_id").agg(pl.col("date").first(), pl.col("venue_code").first(), pl.col("surface").first(), pl.col("_dev").first())
          .filter(pl.col("_dev").is_not_null()).group_by(["date", "venue_code", "surface"])
          .agg(pl.col("_dev").median().alias("_var")))
    F = F.join(dv, on=["date", "venue_code", "surface"], how="left").sort([H, "date", "race_id"])
    F = F.with_columns((-100.0 * (pl.col("_ll") - pl.col("_par") - pl.col("_var"))).alias("_fig"),
                       pl.col("_par").exp().alias("_ptime"))

    # 性別（芝ダ別・レース内比較）と年齢（年齢×月の成長曲線・馬の固定効果）を fit 期間で推定
    fem = pl.col("sex") == "牝"
    fit = F.filter((pl.col("date") <= FIT_END) & pl.col("_fig").is_not_null() & pl.col("_load").is_not_null()
                   & pl.col("age").is_not_null() & pl.col("sex").is_not_null()).with_columns(
        (pl.col("_fig") + 100.0 * (BETA_EST / pl.col("_ptime")) * (pl.col("_load") - LOAD_REF)).alias("_y0"),
        fem.cast(pl.Float64).alias("_fem"), _age_bin().alias("_ab"))
    sex_b = {}
    for sf in ("芝", "ダート"):
        g = fit.filter(pl.col("surface") == sf)
        if g.height < 1000:
            sex_b[sf] = 0.0
            continue
        rid = np.unique(g["race_id"].to_numpy(), return_inverse=True)[1]
        yd, xd = _within(g["_y0"].to_numpy(), rid), _within(g["_fem"].to_numpy(), rid)
        sex_b[sf] = float((xd @ yd) / max(xd @ xd, 1e-9))
    fit = fit.with_columns(pl.col("surface").replace_strict(sex_b, default=0.0, return_dtype=pl.Float64).alias("_sb"))
    y1 = (fit["_y0"] - fit["_sb"] * fit["_fem"]).to_numpy()
    hid = np.unique(fit["horse_id"].to_numpy(), return_inverse=True)[1]
    ab = fit["_ab"].to_numpy()
    X = np.stack([_within((ab == j).astype(float), hid) for j in range(4)], 1)
    yd = _within(y1, hid)
    g_est = np.linalg.lstsq(X, yd, rcond=1e-6)[0]
    age_g = {j: float(g_est[j]) for j in range(4)} | {4: 0.0}
    P["sr_sex_b"], P["sr_age_g"] = sex_b, {str(k): v for k, v in age_g.items()}
    F = F.with_columns(
        (pl.col("surface").replace_strict(sex_b, default=0.0, return_dtype=pl.Float64) * fem.cast(pl.Float64)).alias("_S"),
        _age_bin().replace_strict(age_g, default=0.0, return_dtype=pl.Float64).alias("_G"),
        (pl.col("surface") + "|" + pl.col("dist_band").cast(pl.Utf8)).alias("_sk"))

    # β を選ぶ: 各 β で標準条件に直した過去走から今回の実測 _fig を予測し、誤差が最小のものを距離帯×芝ダ別に採用
    for j, b in enumerate(BETA_GRID):
        L = -100.0 * (b / pl.col("_ptime")) * (pl.col("_load") - LOAD_REF)
        F = F.with_columns((pl.col("_fig") - L - pl.col("_S") - pl.col("_G")).alias(f"_fs{j}"), L.alias(f"_L{j}"))
        mu = pl.col(f"_fs{j}").shift(1).ewm_mean(half_life=3, adjust=True, ignore_nulls=True).over(H)
        F = F.with_columns(((pl.col("_fig") - (mu + pl.col(f"_L{j}") + pl.col("_S") + pl.col("_G"))) ** 2).alias(f"_e2{j}"))
    nprior = pl.col("_fig").is_not_null().cast(pl.Float64).cum_sum().over(H) - pl.col("_fig").is_not_null().cast(pl.Float64)
    ev = F.filter((pl.col("date") <= FIT_END) & (nprior >= 3) & pl.col("_fig").is_not_null() & pl.col("_load").is_not_null())
    mse = ev.group_by("_sk").agg([pl.col(f"_e2{j}").mean().alias(f"m{j}") for j in range(len(BETA_GRID))] + [pl.len().alias("n")])
    beta_sel, beta_mse = {}, {}
    for r in mse.iter_rows(named=True):
        ms = [r[f"m{j}"] for j in range(len(BETA_GRID))]
        beta_sel[r["_sk"]] = float(BETA_GRID[int(np.nanargmin(ms))]) if r["n"] >= 500 else BETA_EST
        beta_mse[r["_sk"]] = [float(m) for m in ms]
    P["sr_beta"], P["sr_beta_mse"], P["sr_beta_grid"] = beta_sel, beta_mse, list(BETA_GRID)

    # 採用した β で標準条件のパフォーマンスを保持し、今回の条件で期待値に戻す
    F = F.with_columns(pl.col("_sk").replace_strict(beta_sel, default=BETA_EST, return_dtype=pl.Float64).alias("_beta"))
    F = F.with_columns((-100.0 * (pl.col("_beta") / pl.col("_ptime")) * (pl.col("_load") - LOAD_REF)).alias("_Lsel"))
    F = F.with_columns((pl.col("_fig") - pl.col("_Lsel") - pl.col("_S") - pl.col("_G")).alias("_fstd"))
    cnt = pl.col("_fstd").is_not_null().cast(pl.Float64)
    F = F.with_columns(
        ewm("_fstd", 2).alias("sr_perf_ewm2"), ewm("_fstd", 6).alias("sr_perf_ewm6"),
        pl.col("_fstd").shift(1).rolling_max(window_size=5, min_samples=1).over(H).alias("sr_perf_best5"),
        pl.col("_fstd").shift(1).rolling_median(window_size=5, min_samples=1).over(H).alias("sr_perf_med5"),
        pl.col("_fstd").shift(1).rolling_std(window_size=5, min_samples=2).over(H).alias("sr_perf_std5"),
        (cnt.cum_sum().over(H) - cnt).alias("sr_n"),
        pl.col("_load").alias("sr_load_eq"), (pl.col("_load") - LOAD_REF).alias("sr_load_dev"),
        pl.col("_Lsel").alias("sr_load_pts"), (pl.col("_load") - pl.col("_load").shift(1).over(H)).alias("sr_load_chg"),
        pl.col("_S").alias("sr_sex_pts"), pl.col("_G").alias("sr_age_pts"),
    ).with_columns(
        (pl.col("sr_perf_ewm2") + pl.col("_Lsel") + pl.col("_S") + pl.col("_G")).alias("sr_exp"),
        (pl.col("sr_perf_ewm6") + pl.col("_Lsel") + pl.col("_S") + pl.col("_G")).alias("sr_exp6"),
        (pl.col("sr_perf_std5") / pl.col("sr_n").sqrt()).alias("sr_unc"),
    ).with_columns(
        (pl.col("sr_exp") - pl.col("sr_exp").mean().over("race_id")).alias("sr_exp_rel"),
        (pl.col("sr_exp").rank("average").over("race_id") / pl.col("sr_exp").count().over("race_id")).alias("sr_exp_rank"),
    )
    drop = [c for c in F.columns if c.startswith("_")]
    return F.drop(drop).sort([H, "date", "race_id"])
