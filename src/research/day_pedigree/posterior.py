"""日の中の更新（段 1-b・1-e）。着順を使う（台帳 select・開発の行 2016〜24 だけ）。

- π̃: η = a + b·log q + c·(log q)² を、探す側（part 0）の 2016 年の 5 着内で当てて固定する。
  レースごとに c_r を解き、Σ expit(η + c_r) = m_r（5 着以内の頭数。同着を含む）にそろえる。
- レース r の U_r = Σ (y − π̃)(x − x̄_w)、I_r = Σ w (x − x̄_w)(x − x̄_w)ᵀ（w = π̃(1 − π̃)・重み付きの中心化）。
- 事前 N(μ, diag τ²): μ = 探す側のその年より前の日の (ΣI)⁻¹ ΣU（静的な効き）。τ² は特徴ごとの DerSimonian–Laird（日の単位）。
- 事後（第 k レースの時点・その日のレースのうち、確定の時刻が締め切り〔発走 − lag 分〕より前のものだけ）: 精度 = φ Σ I_r + T⁻¹、平均 = 精度⁻¹ (φ Σ U_r + T⁻¹ μ)。
  φ は U の分散 ÷ I の比（π̃ から m_r 頭の集合を引いて測る・特徴の平均の 1 つの数）。Var(U) = φ·I なので、
  U も I も 1/φ 倍して使う（擬似尤度。θ̂ = I⁻¹U は変わらず、区間が φ 倍に広がる）。
- 当てはめの点 s_j = (θ̂ − μ)ᵀ (x_j − x̄_w)。日の効きだけを使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

TOP = 5


def fit_eta(lq: np.ndarray, y: np.ndarray) -> np.ndarray:
    """logit P(5 着内) = a + b·lq + c·lq² の最尤（ニュートン法）。"""
    X = np.column_stack([np.ones_like(lq), lq, lq ** 2])
    b = np.zeros(3)
    for _ in range(50):
        p = expit(X @ b)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        st = np.linalg.solve(H, g)
        b += st
        if np.abs(st).max() < 1e-10:
            break
    return b


def calibrate(eta: np.ndarray, rid: np.ndarray, m: np.ndarray) -> np.ndarray:
    """レースごとに c_r を足して Σ π̃ = m_r にする。rid は 0..R−1 の連番、m は行ごとのそのレースの m_r。"""
    R = rid.max() + 1
    c = np.zeros(R)
    mr = np.bincount(rid, weights=m, minlength=R) / np.bincount(rid, minlength=R)
    for _ in range(60):
        p = expit(eta + c[rid])
        f = np.bincount(rid, weights=p, minlength=R) - mr
        d = np.bincount(rid, weights=p * (1 - p), minlength=R)
        st = f / np.maximum(d, 1e-12)
        c -= st
        if np.abs(f).max() < 1e-9:
            break
    return expit(eta + c[rid])


def race_stats(X: np.ndarray, y: np.ndarray, pt: np.ndarray, rid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """レースごとの U [R, F]・I [R, F, F] と、行ごとの中心化した x [n, F]。"""
    R = rid.max() + 1
    w = pt * (1 - pt)
    sw = np.bincount(rid, weights=w, minlength=R)
    xbar = np.stack([np.bincount(rid, weights=w * X[:, f], minlength=R) for f in range(X.shape[1])], axis=1) / np.maximum(sw, 1e-12)[:, None]
    Xc = X - xbar[rid]
    r = y - pt
    U = np.stack([np.bincount(rid, weights=r * Xc[:, f], minlength=R) for f in range(X.shape[1])], axis=1)
    Fn = X.shape[1]
    I = np.zeros((R, Fn, Fn))
    for a in range(Fn):
        for b in range(a, Fn):
            v = np.bincount(rid, weights=w * Xc[:, a] * Xc[:, b], minlength=R)
            I[:, a, b] = v
            I[:, b, a] = v
    return U, I, Xc


def phi_check(X: np.ndarray, pt: np.ndarray, rid: np.ndarray, m: np.ndarray, n_rep: int = 200, seed: int = 11) -> np.ndarray:
    """π̃ を真として「ちょうど m_r 頭が 5 着内」の集合を引き（独立のベルヌーイの棄却）、U の分散 ÷ I の和を特徴ごとに出す。
    使うときは特徴の平均（1 つの数）にする。"""
    rng = np.random.default_rng(seed)
    R = rid.max() + 1
    mr = (np.bincount(rid, weights=m, minlength=R) / np.bincount(rid, minlength=R)).round().astype(int)
    _, I, Xc = race_stats(X, np.zeros(len(pt)), pt, rid)
    Us = []
    for _ in range(n_rep):
        y = np.zeros(len(pt))
        todo = np.arange(R)
        while len(todo):
            sel = np.isin(rid, todo)
            draw = rng.random(sel.sum()) < pt[sel]
            cnt = np.bincount(rid[sel], weights=draw, minlength=R)[todo]
            ok = todo[cnt == mr[todo]]
            okm = np.isin(rid[sel], ok)
            y[np.flatnonzero(sel)[okm]] = draw[okm]
            todo = todo[cnt != mr[todo]]
        r = y - pt
        Us.append(np.stack([np.bincount(rid, weights=r * Xc[:, f], minlength=R) for f in range(X.shape[1])], axis=1).sum(0))
    Us = np.array(Us)
    return Us.var(axis=0) / np.diagonal(I, axis1=1, axis2=2).sum(0)


def phi_pearson(U: np.ndarray, I: np.ndarray) -> np.ndarray:
    """実データの U の散らばり（Pearson の比）: Σ_r U_r² / Σ_r I_r（特徴ごと）。
    レース 1 本の I は小さい（約 0.5）ので、日の効きの分（τ²·I²）はほぼ入らず、モデルのずれの過分散を拾う。"""
    return (U ** 2).sum(0) / np.diagonal(I, axis1=1, axis2=2).sum(0)


def dl_tau2(Ud: np.ndarray, Id: np.ndarray, phi: float) -> np.ndarray:
    """特徴ごとの DerSimonian–Laird（日の単位）。Ud [D, F]・Id [D, F]（対角）。1/φ で直した尺度。
    1 変量の日の推定 u/i と比べる中心も、同じ 1 変量の周辺の μ_f = Σu/Σi にそろえる（自由度 D − 1。critic の 9）。
    多変量の事前の τ² としては近似（交絡した周辺の値で測る）。"""
    U, I = Ud / phi, Id / phi
    ok = I > 1e-9
    tau = np.zeros(U.shape[1])
    for f in range(U.shape[1]):
        u, i = U[ok[:, f], f], I[ok[:, f], f]
        if len(u) < 10:
            continue
        mf = u.sum() / i.sum()
        Q = np.sum((u - mf * i) ** 2 / i)
        den = i.sum() - (i ** 2).sum() / i.sum()
        tau[f] = max(0.0, (Q - (len(u) - 1)) / den)
    return tau


def priors(days: pd.DataFrame, U: np.ndarray, I: np.ndarray, phi: float, years) -> dict:
    """芝ダ × 年ごとの μ（多変量の静的な効き）と τ²。days はレースの表（race 連番の順に day・surf・year・part）。
    年 y の事前は、探す側（part 0）の y より前の年の日だけで作る。"""
    out = {}
    D = days.reset_index(drop=True)
    for s in ("T", "D"):
        for y in years:
            m = ((D["surf"] == s) & (D["year"] < y) & (D["part"] == 0)).to_numpy()
            if m.sum() < 50:
                out[(s, y)] = None
                continue
            mu = np.linalg.solve(I[m].sum(0) + 1e-6 * np.eye(U.shape[1]), U[m].sum(0))     # φ は打ち消し合う
            g = D.loc[m, "day"].to_numpy()
            gi = pd.factorize(g)[0]
            Ud = np.stack([np.bincount(gi, weights=U[m][:, f]) for f in range(U.shape[1])], axis=1)
            Id = np.stack([np.bincount(gi, weights=I[m][:, f, f]) for f in range(U.shape[1])], axis=1)
            out[(s, y)] = {"mu": mu, "tau2": dl_tau2(Ud, Id, phi), "days": int(gi.max() + 1)}
    return out


def day_posterior(days: pd.DataFrame, U: np.ndarray, I: np.ndarray, phi: float, pri: dict, lag_min: float,
                  carry: dict | None = None, w_carry: float = 0.0, keep_path: bool = False):
    """レースごとに、その日・その場・その芝ダの「確定の時刻 < そのレースの発走 − lag_min 分」のレースだけで作る事後。

    days はレースの表（race 連番の順）で、day・surf・year・t_start・t_conf（確定とみなす時刻）を持つ。
    lag_min = 0 は表示用（発走の時点で確定しているレース全部）、60 は検定用（rules-council の T-60）。
    返す: 平均 [R, F]・分散（対角）[R, F]・使ったレース数 [R]。keep_path なら日ごとの使ったレースの id の列も返す。
    """
    R, Fn = U.shape
    mean = np.full((R, Fn), np.nan)
    var = np.full((R, Fn), np.nan)
    nprev = np.zeros(R, dtype=int)
    used: dict[int, list[int]] = {}
    D = days.reset_index(drop=True)
    ts = D["t_start"].to_numpy("datetime64[m]")
    tc = D["t_conf"].to_numpy("datetime64[m]")
    rno = D["rno"].to_numpy() if "rno" in D else None
    conf = np.timedelta64(15, "m")
    lag = np.timedelta64(int(lag_min), "m")
    for (day, s, y), g in D.groupby(["day", "surf", "year"], sort=False):
        p = pri.get((s, y))
        if p is None:
            continue
        Tinv = np.diag(1.0 / np.maximum(p["tau2"], 1e-12))
        mu0 = p["mu"].copy()
        if carry is not None and w_carry and day in carry:
            mu0 = mu0 + w_carry * (carry[day] - p["mu"])
        targets = g.index.to_numpy()[np.argsort(ts[g.index.to_numpy()], kind="stable")]
        srcs = g.index.to_numpy()[np.argsort(tc[g.index.to_numpy()], kind="stable")]
        cu = np.zeros(Fn)
        ci = np.zeros((Fn, Fn))
        n = 0
        for r in targets:
            cut = ts[r] - lag
            while n < len(srcs) and tc[srcs[n]] < cut:
                cu = cu + U[srcs[n]] / phi
                ci = ci + I[srcs[n]] / phi
                n += 1
            # 先読みの検査（rules-council の条件）。足し込みの条件（確定の列）とは別の情報で確かめる（critic の 7）:
            # 入力のレースは、発走 + 15 分 < このレースの発走 − lag、R 番号がこのレースより前、このレース自身を含まない
            if n:
                used_now = srcs[:n]
                assert (ts[used_now] + conf < ts[r] - lag).all(), "日の効きの入力に、締め切りより後に確定したレースが入った"
                assert rno is None or (rno[used_now] < rno[r]).all(), "日の効きの入力に、後の R 番号のレースが入った"
                assert r not in set(used_now.tolist()), "日の効きの入力に、そのレース自身が入った"
            Pinv = np.linalg.inv(ci + Tinv)
            mean[r] = Pinv @ (cu + Tinv @ mu0)
            var[r] = np.diag(Pinv)
            nprev[r] = n
            if keep_path:
                used[int(r)] = srcs[:n].tolist()
    return (mean, var, nprev, used) if keep_path else (mean, var, nprev)


def offset_alpha(y: np.ndarray, pt: np.ndarray, s: np.ndarray, cl: np.ndarray) -> dict:
    """logit P = logit π̃ + α·s の α（1 次元のニュートン）と、開催の束ねの頑健な標準誤差。"""
    off = logit(np.clip(pt, 1e-9, 1 - 1e-9))
    a = 0.0
    for _ in range(50):
        p = expit(off + a * s)
        g = np.sum((y - p) * s)
        h = np.sum(p * (1 - p) * s * s)
        if h <= 0:
            return {"alpha": float("nan"), "se": float("nan"), "info": 0.0}
        st = g / h
        a += st
        if abs(st) < 1e-10:
            break
    p = expit(off + a * s)
    sc = pd.Series((y - p) * s).groupby(cl).sum().to_numpy()
    h = np.sum(p * (1 - p) * s * s)
    return {"alpha": float(a), "se": float(np.sqrt(np.sum(sc ** 2)) / h), "info": float(h)}
