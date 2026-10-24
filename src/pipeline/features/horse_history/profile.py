"""馬ごとの出走前履歴特徴量 A〜E（厳密 as-of）。全体フィットは FIT_END 以前のデータだけで行う。"""
from __future__ import annotations
import numpy as np
import polars as pl
from .common import FIT_END, CLASS_LEVEL

H = "horse_id"
KEY = ["race_id", "horse_id"]
IR = lambda keys: pl.int_range(pl.len()).over(keys)   # グループ内の通し番号 = 同セルの過去走数


def sh(c):
    return pl.col(c).shift(1)


def ewm(c, hl):
    return sh(c).ewm_mean(half_life=hl, adjust=True, ignore_nulls=True).over(H)


# ---------------------------------------------------------------- 準備（レース内相対化など。履歴の素材）
def prepare(runs_rep: pl.DataFrame, elo: pl.DataFrame) -> pl.DataFrame:
    F = runs_rep.join(elo, on=KEY, how="left").sort([H, "date", "race_id"])
    F = F.with_columns(pl.col("l3f").count().over("race_id").alias("l3f_cnt"))
    F = F.with_columns(
        (pl.col("l3f_cnt") / pl.col("n_fin")).alias("l3f_cov"),
        pl.col("l3f").mean().over("race_id").alias("l3f_mu"),
        pl.col("l3f").std().over("race_id").alias("l3f_sd"),
        pl.col("l3f").min().over("race_id").alias("l3f_min"),
    )
    ok = (pl.col("l3f_cov") >= 0.6) & (pl.col("l3f_sd") > 0.2)
    F = F.with_columns(
        pl.when(ok).then((pl.col("l3f") - pl.col("l3f_mu")) / pl.col("l3f_sd")).alias("l3f_z"),
        pl.when(ok).then(pl.col("l3f") - pl.col("l3f_min")).alias("l3f_gap"),
        pl.when(ok).then(pl.col("l3f").rank("average").over("race_id") / pl.col("l3f_cnt")).alias("l3f_rk"),
    )
    F = F.with_columns(
        pl.when(pl.col("l3f_rk").is_not_null()).then((pl.col("l3f_rk") * pl.col("l3f_cnt") <= 3).cast(pl.Float64)).alias("l3f_top3"),
        pl.when(pl.col("pos_early") > 0.6).then(pl.col("l3f_z")).alias("l3f_z_back"),
        pl.when(pl.col("pos_early") < 0.4).then(pl.col("l3f_z")).alias("l3f_z_front"),
        (pl.col("r_pre") - pl.col("fs_mean")).alias("standing"),
        pl.col("res").clip(-0.35, 0.35).alias("res_w"),
        (pl.col("finish_position") == 1).cast(pl.Float64).alias("is_win"),
        (pl.col("finish_position") <= 3).cast(pl.Float64).alias("is_top3"),
        (pl.col("surface") == "芝").cast(pl.Float64).alias("is_turf"),
        (pl.col("going") >= 2).cast(pl.Float64).alias("is_wet"),
        (pl.col("res") ** 2).alias("res2"),
    )
    return F


# ---------------------------------------------------------------- 全体フィット（FIT_END 以前のみ）
def _eb_kappa(pdf, keys, col="res", lo=2.0, hi=300.0):
    g = pdf.groupby(keys)[col]
    n, v, m = g.transform("size"), g.transform("var"), g.transform("mean")
    sel = n >= 2
    sw = float(v[sel].mean())
    gm = pdf[sel].groupby(keys)[col].agg(["mean", "size"])
    sb = max(float(gm["mean"].var() - (sw / gm["size"]).mean()), 1e-4)
    return float(np.clip(sw / sb, lo, hi))


def fit_basic(F: pl.DataFrame) -> dict:
    fit = F.filter(pl.col("date") <= FIT_END)
    pdf = fit.select(H, "res", "surface", "dist_band", "venue_code", "going", "direction", "distance", "class_level").to_pandas()
    pdf["dev"] = pdf["res"] - pdf.groupby(H)["res"].transform("mean")
    P = {"kappa_all": _eb_kappa(pdf, [H], "res", 2.0, 80.0)}
    P["kappa"] = {k: _eb_kappa(pdf, [H] + ks, "dev") for k, ks in {
        "sd": ["surface", "dist_band"], "venue": ["venue_code", "surface"], "going": ["surface", "going"],
        "dir": ["direction", "surface"], "course": ["venue_code", "surface", "distance"], "class": ["class_level"]}.items()}
    P["res_var"] = float(pdf["res"].var())
    P["class_norm"] = {float(k): float(v) for k, v in fit.group_by("class_level").agg(pl.col("fs_mean").mean()).iter_rows()}
    q = fit.filter(pl.col("pace_bal").is_not_null()).group_by("dist_band").agg(
        pl.col("pace_bal").quantile(0.33).alias("q33"), pl.col("pace_bal").quantile(0.67).alias("q67"))
    P["pace_q"] = {int(r["dist_band"]): (float(r["q33"]), float(r["q67"])) for r in q.iter_rows(named=True)}
    return P


# ---------------------------------------------------------------- 履歴特徴量の本体
def history(F: pl.DataFrame, P: dict) -> pl.DataFrame:
    F = F.sort([H, "date", "race_id"])
    cn = P["class_norm"]
    F = F.with_columns(
        (pl.col("fs_mean") - pl.col("class_level").replace_strict(cn, default=float(np.mean(list(cn.values()))), return_dtype=pl.Float64)).alias("rq"),
        pl.col("pace_bal").alias("_pb"),
    )
    # ---- D: 休養・年齢・体重 / E: 信頼度
    F = F.with_columns(
        (pl.col("date") - pl.col("date").shift(1).over(H)).dt.total_days().cast(pl.Float64).alias("d_rest"),
        (pl.col("date") - pl.col("date").shift(2).over(H)).dt.total_days().cast(pl.Float64).alias("d_rest2"),
        IR(H).cast(pl.Float64).alias("d_nprior"),
        (pl.col("date") - pl.col("date").first().over(H)).dt.total_days().cast(pl.Float64).alias("d_career_days"),
        (pl.col("wt") - sh("wt").rolling_mean(window_size=50, min_samples=1).over(H)).alias("d_wt_dev"),
        (pl.col("wt").shift(1) - pl.col("wt").shift(2)).over(H).alias("d_wt_chg_last"),
        (pl.col("wt") - pl.col("wt").shift(1)).over(H).alias("d_wt_chg_now"),
        pl.col("finish_position").shift(1).over(H).alias("d_prev_fin"),
        pl.col("n_fin").shift(1).over(H).alias("d_prev_field"),
        (pl.col("finish_position").shift(1) / pl.col("n_fin").shift(1)).over(H).alias("d_prev_fin_pct"),
        sh("l3f_z").is_not_null().cast(pl.Float64).cum_sum().over(H).alias("e_n_l3f"),
    )
    # ---- A: 上がりの傾向
    F = F.with_columns(
        ewm("l3f_z", 2).alias("a_l3fz_ewm2"), ewm("l3f_z", 6).alias("a_l3fz_ewm6"),
        sh("l3f_z").rolling_min(window_size=50, min_samples=1).over(H).alias("a_l3fz_best"),
        sh("l3f_z").rolling_median(window_size=5, min_samples=1).over(H).alias("a_l3fz_med5"),
        sh("l3f_z").rolling_std(window_size=5, min_samples=2).over(H).alias("a_l3fz_std5"),
        ewm("l3f_gap", 2).alias("a_gap_ewm2"), ewm("l3f_rk", 2).alias("a_rk_ewm2"), ewm("l3f_top3", 3).alias("a_top3_freq"),
        ewm("l3f_z_back", 3).alias("a_l3fz_back"), ewm("l3f_z_front", 3).alias("a_l3fz_front"),
        ewm("pos_early", 2).alias("a_style"), ewm("pos_late", 2).alias("a_style_late"),
    ).with_columns((pl.col("a_l3fz_ewm2") - pl.col("a_l3fz_ewm6")).alias("a_trend"),
                   (pl.col("a_style_late") - pl.col("a_style")).alias("a_style_move"))
    # ---- C: レースの質・相手関係・ロバスト統計
    F = F.with_columns(
        ewm("res_w", 2).alias("c_res_ewm2"), ewm("res_w", 6).alias("c_res_ewm6"),
        sh("res").rolling_median(window_size=5, min_samples=1).over(H).alias("c_res_med5"),
        sh("res").rolling_std(window_size=5, min_samples=2).over(H).alias("e_res_std5"),
        sh("res").rolling_max(window_size=5, min_samples=1).over(H).alias("c_res_max5"),
        sh("res").rolling_min(window_size=5, min_samples=1).over(H).alias("c_res_min5"),
        sh("res").rolling_sum(window_size=5, min_samples=4).over(H).alias("_s5"),
        sh("res").is_not_null().cast(pl.Float64).rolling_sum(window_size=5, min_samples=1).over(H).alias("_n5"),
        ewm("fs_mean", 2).alias("c_fs_ewm2"), ewm("fs_top3", 2).alias("c_fstop3_ewm2"),
        ewm("standing", 2).alias("c_standing_ewm2"), ewm("rq", 2).alias("c_rq_ewm2"),
        ewm("class_level", 2).alias("c_cls_ewm2"), pl.col("class_level").shift(1).over(H).alias("c_cls_prev"),
        pl.col("res").shift(1).over(H).alias("c_res_prev1"),
    ).with_columns(
        pl.when(pl.col("_n5") >= 4).then((pl.col("_s5") - pl.col("c_res_max5") - pl.col("c_res_min5")) / (pl.col("_n5") - 2)).alias("c_res_trim5"),
        (pl.col("standing") - pl.col("c_standing_ewm2")).alias("c_opp_step"),
        (pl.col("class_level") - pl.col("c_cls_ewm2")).alias("c_cls_step"),
        (pl.col("fs_mean") - pl.col("c_fs_ewm2")).alias("c_fs_step"),
        (pl.col("c_res_prev1") - pl.col("c_res_ewm6")).alias("c_last_dev"),
    ).drop("_s5", "_n5")
    F = F.with_columns(
        (pl.col("res").cum_sum().over(H) - pl.col("res")).alias("_s1"),
        (pl.col("res2").cum_sum().over(H) - pl.col("res2")).alias("_s2"),
    ).with_columns(
        pl.when(pl.col("d_nprior") >= 2).then(((pl.col("_s2") / pl.col("d_nprior") - (pl.col("_s1") / pl.col("d_nprior")) ** 2).clip(lower_bound=0) * pl.col("d_nprior") / (pl.col("d_nprior") - 1)).sqrt()).alias("e_res_std_all"),
        (pl.col("_s1") / (pl.col("d_nprior") + P["kappa_all"])).alias("b_aff_all"),
    ).drop("_s1", "_s2")
    # ---- B: 得意な舞台（セル別の相対成績を階層縮約）
    kp = P["kappa"]
    def cell(name, keys, parent, kappa):
        s = (pl.col("res").cum_sum().over([H] + keys) - pl.col("res")).alias(f"_s_{name}")
        n = IR([H] + keys).cast(pl.Float64).alias(f"b_n_{name}")
        return s, n, (pl.col(f"_s_{name}") + kappa * pl.col(parent)) / (pl.col(f"b_n_{name}") + kappa)
    chain = [("sd", ["surface", "dist_band"], "b_aff_all"), ("venue", ["venue_code", "surface"], "b_aff_sd"),
             ("going", ["surface", "going"], "b_aff_sd"), ("dir", ["direction", "surface"], "b_aff_sd"),
             ("course", ["venue_code", "surface", "distance"], "b_aff_venue"), ("class", ["class_level"], "b_aff_all")]
    for name, keys, parent in chain:
        s, n, a = cell(name, keys, parent, kp[name])
        F = F.with_columns(s, n).with_columns(a.alias(f"b_aff_{name}")).drop(f"_s_{name}")
    # 距離適性曲線（log距離の二次・リッジ。過去走のモーメントの累積から閉形式で解く）
    x, y = pl.col("logd"), pl.col("res")
    mom = {"m1": x, "m2": x ** 2, "m3": x ** 3, "m4": x ** 4, "sy": y, "sxy": x * y, "sx2y": x ** 2 * y}
    F = F.with_columns(*[(e.cum_sum().over(H) - e).alias(f"_{k}") for k, e in mom.items()], IR(H).cast(pl.Float64).alias("_m0"))
    M = F.select("_m0", "_m1", "_m2", "_m3", "_m4", "_sy", "_sxy", "_sx2y", "logd").to_numpy()
    m0, m1, m2, m3, m4, sy, sxy, sx2y, x0 = [M[:, i] for i in range(9)]
    A = np.zeros((len(M), 3, 3)); bvec = np.stack([sy, sxy, sx2y], 1)
    A[:, 0] = np.stack([m0, m1, m2], 1); A[:, 1] = np.stack([m1, m2, m3], 1); A[:, 2] = np.stack([m2, m3, m4], 1)
    lam = np.array([P["kappa_all"] * 0.5 + 1e-3, 8.0, 8.0]) * 1.0
    A = A + np.diag(lam)[None]
    beta = np.linalg.solve(A, bvec[:, :, None])[:, :, 0]
    z = np.stack([np.ones_like(x0), x0, x0 ** 2], 1)
    pred = (beta * z).sum(1)
    se = np.sqrt(P["res_var"] * np.einsum("ni,nij,nj->n", z, np.linalg.inv(A), z))
    slope = beta[:, 1] + 2 * beta[:, 2] * x0
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = np.where(beta[:, 2] < -1e-3, -beta[:, 1] / (2 * beta[:, 2]), np.nan)
    gap = np.clip(x0 - vertex, -1.0, 1.0)
    nprior = m0
    F = F.with_columns(pl.Series("b_dist_pred", np.where(nprior > 0, pred, np.nan)), pl.Series("b_dist_se", se),
                       pl.Series("b_dist_slope", np.where(nprior > 0, slope, np.nan)), pl.Series("b_dist_gap", gap)).drop([c for c in F.columns if c.startswith("_m") or c in ("_sy", "_sxy", "_sx2y")])
    return F
