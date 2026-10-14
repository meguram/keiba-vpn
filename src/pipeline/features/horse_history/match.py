"""出走レースとのマッチ度 F1/F2/F5、レーティング系 F3、特徴量の組み立て。"""
from __future__ import annotations
import numpy as np
import polars as pl
from .common import FIT_END
from .profile import H, KEY, prepare, fit_basic, history, sh, ewm
from .pace import pace_index


def f3(F, P):
    return F.with_columns(
        pl.col("r_pre").alias("f3_r"), pl.col("E").alias("f3_E"), pl.col("fs_mean").alias("f3_fs_mean"),
        pl.col("fs_top3").alias("f3_fs_top3"), pl.col("fs_std").alias("f3_fs_std"),
        (pl.col("r_pre") - pl.col("fs_mean")).alias("f3_r_gap"), (pl.col("r_pre") - pl.col("fs_top3")).alias("f3_r_gap_top3"),
        (pl.col("r_pre").rank("average").over("race_id") / pl.col("n_fin")).alias("f3_r_rank"),
        pl.col("n_pre").alias("f3_n_pre"),
    )


def f2(F, P):
    kn = pl.col("a_style").is_not_null().sum().over("race_id")
    F = F.with_columns(
        kn.alias("f2_nknown"), (pl.col("a_style") < 0.25).sum().over("race_id").alias("f2_nfront"),
        (pl.col("a_style") < 0.40).sum().over("race_id").alias("f2_nfront40"),
        (pl.col("a_style").rank("average").over("race_id") / kn).alias("f2_early_rank"),
    ).with_columns(
        pl.when(pl.col("f2_nknown") >= 0.5 * pl.col("n_fin")).then(pl.col("f2_nfront") / pl.col("f2_nknown")).alias("f2_pressure"))
    fitr = F.filter(pl.col("date") <= FIT_END).group_by("race_id").agg(pl.col("f2_pressure").first(), pl.col("dist_band").first(), pl.col("pace_bal").first())
    pq = fitr.select(pl.col("f2_pressure").quantile(0.33).alias("a"), pl.col("f2_pressure").quantile(0.67).alias("b")).row(0)
    P["press_q"] = pq
    q = P["pace_q"]
    lo = {k: v[0] for k, v in q.items()}; hi = {k: v[1] for k, v in q.items()}
    F = F.with_columns(
        pl.when(pl.col("f2_pressure").is_null()).then(None).when(pl.col("f2_pressure") <= pq[0]).then(0).when(pl.col("f2_pressure") <= pq[1]).then(1).otherwise(2).alias("f2_bin"),
        pl.when(pl.col("pace_bal").is_null()).then(None)
        .when(pl.col("pace_bal") <= pl.col("dist_band").replace_strict(lo, default=-99.0, return_dtype=pl.Float64)).then(0)
        .when(pl.col("pace_bal") >= pl.col("dist_band").replace_strict(hi, default=99.0, return_dtype=pl.Float64)).then(2).otherwise(1).alias("reg_act"))
    fr = F.filter(pl.col("date") <= FIT_END).group_by("race_id").agg(pl.col("f2_bin").first(), pl.col("reg_act").first()).drop_nulls()
    cnt = np.ones((3, 3))
    for b, r in zip(fr["f2_bin"].to_list(), fr["reg_act"].to_list()):
        cnt[int(b), int(r)] += 1
    P["pmap"] = (cnt / cnt.sum(1, keepdims=True)).tolist()
    kap = P["kappa_all"]
    for k in range(3):
        ind = (pl.col("reg_act") == k).fill_null(False)
        r_k = pl.when(ind).then(pl.col("res")).otherwise(0.0)
        F = F.with_columns((r_k.cum_sum().over(H) - r_k).alias(f"_rs{k}"), (ind.cast(pl.Float64).cum_sum().over(H) - ind.cast(pl.Float64)).alias(f"f2_n_reg{k}"))
        F = F.with_columns((pl.col(f"_rs{k}") / (pl.col(f"f2_n_reg{k}") + kap)).alias(f"f2_a_reg{k}")).drop(f"_rs{k}")
    pm = np.array(P["pmap"]); marg = pm.mean(0)
    bins = F["f2_bin"].to_numpy()
    probs = np.array([pm[int(b)] if b == b and b is not None else marg for b in bins])
    a = F.select("f2_a_reg0", "f2_a_reg1", "f2_a_reg2").to_numpy()
    F = F.with_columns(pl.Series("f2_pace_match", (probs * a).sum(1)), (pl.col("f2_a_reg0") - pl.col("f2_a_reg2")).alias("f2_pace_adv"),
                       pl.Series("f2_p_fast", probs[:, 0]))
    # 脚質×想定展開の素直な相互作用
    return F.with_columns((pl.col("a_style") * pl.col("f2_pressure")).alias("f2_style_x_press"),
                          ((pl.col("a_style") < 0.25) & (pl.col("f2_nfront") <= 1)).cast(pl.Float64).alias("f2_lone_front"))


def _comp(df):
    f = pl.Float64
    return df.select(
        (pl.col("logd") - pl.col("logd_h")).abs(), (pl.col("surface") != pl.col("surface_h")).cast(f),
        (pl.col("venue_code") != pl.col("venue_h")).cast(f), (pl.col("going") - pl.col("going_h")).abs().cast(f),
        (pl.col("direction") != pl.col("dir_h")).cast(f), (pl.col("class_level") - pl.col("class_h")).abs(),
        (pl.col("n_fin") - pl.col("nfin_h")).abs().cast(f) / 10.0, (pl.col("date") - pl.col("date_h")).dt.total_days().cast(f) / 365.0,
    ).to_numpy()


def sim(F, P):
    t = F.select(H, "date", "logd", "surface", "venue_code", "going", "direction", "class_level", "n_fin").with_row_index("ti")
    h = F.select(H, pl.col("date").alias("date_h"), pl.col("logd").alias("logd_h"), pl.col("surface").alias("surface_h"),
                 pl.col("venue_code").alias("venue_h"), pl.col("going").alias("going_h"), pl.col("direction").alias("dir_h"),
                 pl.col("class_level").alias("class_h"), pl.col("n_fin").alias("nfin_h"), pl.col("res").alias("res_h"), pl.col("l3f_z").alias("z_h"))
    pr = t.join(h, on=H).filter(pl.col("date_h") < pl.col("date"))
    N = F.height
    res_t = F["res"].to_numpy()
    # θ を FIT 期間の対象だけで探索（ランダム探索 + 座標微調整）
    fp = pr.filter(pl.col("date") <= FIT_END)
    C, ti, rh = _comp(fp), fp["ti"].to_numpy(), fp["res_h"].to_numpy()
    fitmask = (F["date"] <= FIT_END).to_numpy()

    def score(th):
        w = np.exp(-(C @ th)); sw = np.bincount(ti, w, N); s = np.bincount(ti, w * rh, N)
        sim_ = s / (sw + 1.0); m = fitmask & (sw > 0.3)
        return np.corrcoef(sim_[m], res_t[m])[0, 1] if m.sum() > 100 else -1
    rng = np.random.default_rng(0)
    base = score(np.zeros(C.shape[1])); best, bs = np.zeros(C.shape[1]), base
    for _ in range(80):
        th = np.exp(rng.uniform(np.log(0.05), np.log(4.0), C.shape[1]))
        sc = score(th)
        if sc > bs: best, bs = th, sc
    for _ in range(2):
        for j in range(C.shape[1]):
            for mult in (0.5, 2.0):
                th = best.copy(); th[j] = max(th[j], 0.05) * mult; sc = score(th)
                if sc > bs: best, bs = th, sc
    P["sim_theta"] = best.tolist(); P["sim_fit_corr"] = [float(base), float(bs)]
    C2, ti2 = _comp(pr), pr["ti"].to_numpy()
    w = np.exp(-(C2 @ best)); rh2 = pr["res_h"].to_numpy(); zh = pr["z_h"].to_numpy(); zv = ~np.isnan(zh)
    sw = np.bincount(ti2, w, N); s = np.bincount(ti2, w * rh2, N); sw2 = np.bincount(ti2, w * w, N)
    swz = np.bincount(ti2, w * zv, N); sz = np.bincount(ti2, w * np.where(zv, zh, 0.0), N)
    return F.with_columns(pl.Series("f1_sim_res", s / (sw + 1.0)), pl.Series("f1_sw", sw),
                          pl.Series("f1_neff", np.where(sw2 > 0, sw ** 2 / np.where(sw2 > 0, sw2, 1), 0.0)),
                          pl.Series("f1_sim_z", np.where(swz > 0, sz / (swz + 1.0), np.nan)))


def clusters(F, P, k=24):
    from sklearn.cluster import KMeans
    F = F.with_columns(ewm("logd", 3).alias("_p_logd"), ewm("is_turf", 3).alias("_p_turf"), ewm("is_wet", 3).alias("_p_wet"))
    cols = ["a_style", "a_l3fz_ewm2", "_p_logd", "_p_turf", "_p_wet", "c_cls_ewm2"]
    X = F.select(cols).to_numpy().astype(float)
    valid = (F["d_nprior"].to_numpy() >= 3)
    fitm = valid & (F["date"] <= FIT_END).to_numpy()
    mu, sd = np.nanmean(X[fitm], 0), np.nanstd(X[fitm], 0) + 1e-9
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    km = KMeans(k, n_init=3, random_state=0).fit(Z[fitm])
    lab = np.where(valid, km.predict(Z), -1)
    F = F.with_columns(pl.Series("f5_cl", lab).cast(pl.Int32)).drop("_p_logd", "_p_turf", "_p_wet")
    keys = ["f5_cl", "surface", "dist_band"]
    D = (F.filter(pl.col("f5_cl") >= 0).group_by(keys + ["date"]).agg(pl.col("res").sum().alias("s"), pl.col("is_win").sum().alias("w"), pl.len().alias("n"))
         .sort("date").with_columns(pl.col("s").cum_sum().over(keys).alias("cs"), pl.col("w").cum_sum().over(keys).alias("cw"), pl.col("n").cum_sum().over(keys).alias("cn"))
         .select(keys + ["date", "cs", "cw", "cn"]))
    T = F.select(keys + ["date"]).with_row_index("ti").sort("date")
    J = T.join_asof(D, on="date", by=keys, strategy="backward", allow_exact_matches=False).sort("ti")
    cn = J["cn"].fill_null(0).to_numpy().astype(float); cs = J["cs"].fill_null(0).to_numpy().astype(float); cw = J["cw"].fill_null(0).to_numpy().astype(float)
    ok = F["f5_cl"].to_numpy() >= 0
    return F.with_columns(pl.Series("f5_aff", np.where(ok, cs / (cn + 30.0), np.nan)), pl.Series("f5_win", np.where(ok, (cw + 0.07 * 50) / (cn + 50.0), np.nan)))  # 累積件数 cn は時間で単調増加するため出力しない


def build_all(runs_rep, elo):
    F = prepare(runs_rep, elo)
    P = fit_basic(F)
    F = history(F, P)
    F = pace_index(F, P)
    F = f3(F, P)
    F = f2(F, P)
    F = sim(F, P)
    F = clusters(F, P)
    return F, P




def families(cols):
    fam = {"A": [c for c in cols if c.startswith("a_")],
           "B": [c for c in cols if c.startswith(("b_aff_", "b_n_", "b_dist_"))],
           "C": [c for c in cols if c.startswith("c_")],
           "D": [c for c in cols if c.startswith("d_")],
           "E": [c for c in cols if c.startswith("e_")],
           "G": [c for c in cols if c.startswith("pc_")],
           "F1": [c for c in cols if c.startswith("f1_")],
           "F2": [c for c in cols if c.startswith("f2_")],
           "F3": [c for c in cols if c.startswith("f3_")],
           "F5": [c for c in cols if c.startswith("f5_")]}
    return fam
