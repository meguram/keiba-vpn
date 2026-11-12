"""tp-83: 出走馬全員のめぐ指数の履歴から『馬ごとの能力の分布』を作り、レース内で同時にぶつけた確率と差・形の特徴量にする。as-of（今回の日付より前の走だけ）。

馬ごとの能力 = 過去 K 走の絶対指数の加重平均を母集団へ寄せた μ（走数が少ないほど強く。外れた走は重みを下げる）と、予測の不確かさ σ（走数が少ない・ばらつきが大きいほど大）。履歴の無い馬は母集団の事前分布。
  FS_A  自分の分布と、レース内の他馬との比較（同時確率 p_win/p_top3、最強の相手との差 / √(σ²+σ²)、近い相手の頭数、期待して勝てる相手の割合）
  FS_B  レース全体の形（有効頭数・エントロピー・本命の確率・上位2頭の差・μ のばらつき/σ・履歴の無い馬の割合）
  FC_*  上の μ を、今回の面・距離に近い走ほど重くして作った版（条件を揃えた能力）
先読みを避ける約束: 過去走は 1 走以上ずらし（tp42 の lags_m）、今回の結果は使わない。母集団の定数は 2021 年以前の走だけから作る。
市場オッズ・人気は使わない。キーは race_id / horse_id。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln, log_ndtr, ndtr, stdtr

from src.pipeline.features import factory_self as fs
from src.pipeline.features import tp42_racelevel as rl

K = rl.K
W = rl.W
K0 = 3.0                   # ばらつきを母集団へ寄せる強さ（擬似的な走数）
MIN_FIELD = 4
GRID = 300
N_MC = 2000
SEED = 12345
CONST_YEAR = 2021          # 母集団の定数を作る年（この年以前の走だけ）
COND_SURF_W = 0.6          # 今回と違う面の走の重み（条件を揃えた版）

prepare = rl.prepare


def _weights(d: pd.DataFrame, cond: bool) -> np.ndarray:
    ww = np.broadcast_to(W ** np.arange(K), (len(d), K)).astype(float).copy()
    if cond:
        surf_lag = fs.lags(d, "surf", K)
        today = d["surf"].to_numpy(float)[:, None]
        ws = np.where(surf_lag == today, 1.0, COND_SURF_W)
        dist_lag = fs.lags(d, "dist", K)
        wd = np.exp(-np.abs(dist_lag - d["dist"].to_numpy(float)[:, None]) / rl.DIST_SCALE)
        ww = ww * np.where(np.isfinite(ws), ws, 1.0) * np.where(np.isfinite(wd), wd, 1.0)
    return ww


def _stats(M: np.ndarray, ww: np.ndarray) -> dict[str, np.ndarray]:
    """加重平均 E・単走のばらつき var（不偏）・走数 n・有効走数 n_eff。"""
    m = np.isfinite(M)
    w = np.where(m, ww, 0.0)
    sw = w.sum(1)
    sw2 = (w ** 2).sum(1)
    Mz = np.where(m, M, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        E = np.where(sw > 0, (w * Mz).sum(1) / sw, np.nan)
        den = sw - sw2 / sw
        var = np.where((m.sum(1) >= 2) & (den > 1e-9), (w * (Mz - E[:, None]) ** 2).sum(1) / den, np.nan)
        n_eff = np.where(sw2 > 0, sw ** 2 / sw2, np.nan)
    return {"E": E, "var": var, "n": m.sum(1).astype(float), "n_eff": n_eff}


def population_constants(d: pd.DataFrame, lag_days: int = 0) -> dict[str, float]:
    """R2: 単走のばらつきの母集団値（履歴 3 走以上の馬の var の平均）。mu0/sd0: 履歴の無い馬の最初の指数の平均・標準偏差。2021 年以前の走だけで作る。"""
    M = rl.lags_m(d, "a", K, lag_days)
    st = _stats(M, _weights(d, False))
    yr = d["date"].dt.year.to_numpy()
    ok = (yr <= CONST_YEAR) & (st["n"] >= 3) & np.isfinite(st["var"])
    R2 = float(np.mean(st["var"][ok]))
    ev = (yr <= CONST_YEAR) & (st["n"] >= 3) & np.isfinite(st["E"])
    tau2 = max(float(np.var(st["E"][ev]) - R2 * np.mean(1.0 / st["n_eff"][ev])), 1.0)     # 馬の真の能力の分散（観測の分散から測定誤差を引く）
    kappa = float(np.clip(R2 / tau2, 0.5, 20.0))                                           # 母集団へ寄せる強さ（擬似的な走数）
    a_all = d["a"].to_numpy(float)[(yr <= CONST_YEAR) & d["a"].notna().to_numpy()]
    deb = (yr == CONST_YEAR) & (st["n"] == 0) & d["a"].notna().to_numpy()
    a = d["a"].to_numpy(float)[deb]
    return {"R2": R2, "tau2": tau2, "kappa": kappa, "m_all": float(np.mean(a_all)), "mu0": float(np.mean(a)), "sd0": float(np.std(a)), "n_R2": int(ok.sum()), "n_debut": int(deb.sum())}


def ability(d: pd.DataFrame, consts: dict, lag_days: int = 0, cond: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各走の μ・σ・走数。履歴の無い馬は事前分布。
    μ = 母集団の平均を擬似 κ 走ぶんの観測として足した加重平均。huber_k>0 なら、自分の μ から大きく外れた走（良くも悪くも）の重みを k·√R2 を超えた分だけ下げて、μ・σ の両方から外れ値の影響を抑える。"""
    M = rl.lags_m(d, "a", K, lag_days)
    ww = _weights(d, cond)
    R2, kappa, m_all = consts["R2"], consts["kappa"], consts["m_all"]
    hk = float(consts.get("huber_k", 0.0))
    fin = np.isfinite(M)
    Mz = np.where(fin, M, 0.0)
    hw = np.ones_like(Mz)
    for _ in range(7 if hk > 0 else 1):
        w = np.where(fin, ww * hw, 0.0)
        sw, sw2 = w.sum(1), (w ** 2).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            E = np.where(sw > 0, (w * Mz).sum(1) / sw, 0.0)
            ne = np.where(sw2 > 0, sw ** 2 / sw2, 0.0)
        c = (ne * E + kappa * m_all) / (ne + kappa)
        if hk <= 0:
            break
        r = np.abs(Mz - c[:, None]) / np.sqrt(R2)
        hw = np.where(fin, np.minimum(1.0, hk / np.maximum(r, 1e-9)), 1.0)
    st = _stats(M, w)
    n, var = st["n"], st["var"]
    s2 = np.where(n >= 2, (K0 * R2 + (n - 1) * np.nan_to_num(var, nan=R2)) / (K0 + n - 1), R2)
    sd = np.sqrt(s2 * (1.0 + 1.0 / (ne + kappa))) * float(consts.get("sd_mult", 1.0))
    if consts.get("sd_n_mult") is not None:
        sd = sd * np.asarray(consts["sd_n_mult"], float)[np.clip(n, 0, K).astype(int)]
    mu = np.where(ne > 0, c, np.nan)
    none = n == 0
    mu = np.where(none, consts["mu0"], mu)
    sd = np.where(none, consts["sd0"], sd)
    return mu, sd, n


def fit_sd_by_n(d: pd.DataFrame, consts: dict, lag_days: int = 0, cond: bool = False, min_rows: int = 200) -> list[float]:
    """走数 n（0〜K）ごとの σ の倍率 = 実績との差 z=(a-μ)/σ の標準偏差。2021 年（定数を作る年）の走だけで合わせる。走数が少ない馬の σ は大きく見すぎ・多い馬は小さく見すぎ、の偏りを直す。"""
    mu, sd, n = ability(d, {**consts, "sd_n_mult": None}, lag_days, cond)
    a = d["a"].to_numpy(float)
    sel = (d["date"].dt.year.to_numpy() == CONST_YEAR) & np.isfinite(a) & (n >= 1)
    z = (a - mu) / sd
    out = np.ones(K + 1)
    for k in range(1, K + 1):
        m = sel & (n == k)
        if m.sum() >= min_rows:
            out[k] = float(np.std(z[m]))
    return out.tolist()


def _race_block(mu: np.ndarray, sd: np.ndarray, nn: np.ndarray, rng: np.random.Generator, df: float = 0.0) -> dict[str, np.ndarray]:
    """df>2 なら、標準偏差が sd になるように尺度を揃えた t 分布（裾が重い＝大きな番狂わせを許す）。df=0 は正規分布。"""
    N = len(mu)
    # 勝つ確率: ∫ φ_i(x) Π_{j≠i} Φ_j(x) dx（格子で積分）
    xs = np.linspace((mu - 5 * sd).min(), (mu + 5 * sd).max(), GRID)
    if df > 2:
        sc = sd * np.sqrt((df - 2.0) / df)
        U = (xs[None, :] - mu[:, None]) / sc[:, None]
        lphi = np.log(np.clip(stdtr(df, U), 1e-300, 1.0))
        lpdf = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - (df + 1) / 2 * np.log1p(U ** 2 / df)
                - np.log(sc)[:, None])
    else:
        U = (xs[None, :] - mu[:, None]) / sd[:, None]
        lphi = log_ndtr(U)
        lpdf = -0.5 * U ** 2 - np.log(sd)[:, None] - 0.5 * np.log(2 * np.pi)
    tot = lphi.sum(0)
    pw = np.exp(lpdf + tot[None, :] - lphi).sum(1) * (xs[1] - xs[0])
    pw = pw / pw.sum()
    # 3 着以内の確率（乱数。レースごとに固定の乱数）
    if N <= 3:
        p3 = np.ones(N)
    else:
        if df > 2:
            X = mu[None, :] + sd[None, :] * np.sqrt((df - 2.0) / df) * rng.standard_t(df, (N_MC, N))
        else:
            X = mu[None, :] + sd[None, :] * rng.standard_normal((N_MC, N))
        top = np.argpartition(-X, 2, axis=1)[:, :3]
        p3 = np.bincount(top.ravel(), minlength=N) / N_MC
    # 2 頭ずつの比較
    D = mu[:, None] - mu[None, :]
    S = np.sqrt(sd[:, None] ** 2 + sd[None, :] ** 2)
    Z = D / S
    off = ~np.eye(N, dtype=bool)
    beaten = (ndtr(Z) * off).sum(1) / (N - 1)
    close = ((np.abs(Z) <= 1.0) & off).sum(1).astype(float)
    Zm = np.where(off, mu[None, :], -np.inf)
    jb = Zm.argmax(1)                                    # 自分以外で μ が最大の馬
    gap_best = Z[np.arange(N), jb]
    # 勝つ確率が最大の相手との差
    pm = np.where(off, pw[None, :], -np.inf)
    jp = pm.argmax(1)
    gap_threat = Z[np.arange(N), jp]
    order = np.argsort(-mu)
    z12 = Z[order[0], order[1]]
    ps = np.sort(pw)[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(-(pw * np.log(np.where(pw > 0, pw, 1.0))).sum())
    return {
        "p_win": pw, "p_top3": p3, "lp_rel": np.log(np.clip(pw, 1e-6, 1) * N), "p_share_max": pw / ps[0],
        "beaten": beaten, "close": close, "gap_best_z": gap_best, "gap_threat_z": gap_threat,
        "rank_pw": (-pw).argsort().argsort().astype(float) + 1.0,
        "f_neff": np.full(N, 1.0 / float((pw ** 2).sum())), "f_entropy": np.full(N, ent / np.log(N)),
        "f_pmax": np.full(N, ps[0]), "f_ptop3": np.full(N, ps[:3].sum()), "f_z12": np.full(N, z12),
        "f_sep": np.full(N, float(np.std(mu) / np.mean(sd))), "f_sd_mean": np.full(N, float(np.mean(sd))),
        "f_debut": np.full(N, float(np.mean(nn == 0))),
    }


SELF_COLS = ["mu", "sd", "n", "p_win", "p_top3", "lp_rel", "p_share_max", "rank_pw", "beaten", "close", "gap_best_z", "gap_threat_z"]
FIELD_COLS = ["f_neff", "f_entropy", "f_pmax", "f_ptop3", "f_z12", "f_sep", "f_sd_mean", "f_debut"]


def build(d: pd.DataFrame, consts: dict, lag_days: int = 0, cond: bool = False) -> pd.DataFrame:
    """全列（SELF_COLS + FIELD_COLS）。出走頭数が MIN_FIELD 未満のレースは欠損。"""
    mu, sd, nn = ability(d, consts, lag_days, cond)
    codes = pd.factorize(d["race_id"])[0]
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    ends = np.r_[starts[1:], len(cs)]
    out = {c: np.full(len(d), np.nan) for c in SELF_COLS + FIELD_COLS}
    out["mu"], out["sd"], out["n"] = mu.copy(), sd.copy(), nn.copy()
    for s0, e0 in zip(starts, ends):
        if e0 - s0 < MIN_FIELD:
            continue
        idx = order[s0:e0]
        rng = np.random.default_rng(SEED + int(codes[idx[0]]))
        blk = _race_block(mu[idx], sd[idx], nn[idx], rng, float(consts.get("t_df", 0.0)))
        for c, v in blk.items():
            out[c][idx] = v
    return pd.DataFrame(out, index=d.index)


def build_all(d: pd.DataFrame, lag_days: int = 0, opts: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """opts: kappa_mult（母集団へ寄せる強さの倍率）, huber_k（外れ値の重み下げ。0 で無し）, t_df（0 で正規分布）, sd_mult。"""
    consts = population_constants(d, lag_days)
    o = dict(opts or {})
    consts["kappa"] = consts["kappa"] * float(o.pop("kappa_mult", 1.0))
    consts.update({k: float(v) for k, v in o.items()})
    consts["sd_n_mult"] = fit_sd_by_n(d, consts, lag_days, False)
    consts["sd_n_mult_cond"] = fit_sd_by_n(d, consts, lag_days, True)
    groups: dict[str, pd.DataFrame] = {}
    for tag, cond in (("FS", False), ("FC", True)):
        t = build(d, {**consts, "sd_n_mult": consts["sd_n_mult_cond"] if cond else consts["sd_n_mult"]}, lag_days, cond)
        pre = tag.lower() + "_"
        parts = {"A": SELF_COLS, "B": FIELD_COLS, "ALL": SELF_COLS + FIELD_COLS}
        for k, cols in parts.items():
            g = t[cols].rename(columns={c: pre + c for c in cols})
            g.insert(0, "horse_id", d["horse_id"].to_numpy())
            g.insert(0, "race_id", d["race_id"].to_numpy())
            groups[f"{tag}_{k}"] = g
    return groups, consts
