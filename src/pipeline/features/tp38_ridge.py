"""tp-38 F2: 祖先の効果 × 位置の重み（φ）のリッジ回帰（交互最小二乗）。すべて as-of（その作り直し日より前の走だけで学習）。

モデル: 走 i の成績 y_i（レース内の着順スコア）を、血統表の 62 の位置 p にいる祖先 a(p) の効果 δ で説明する。
    y_i ≈ Σ_p φ_p · ( δ_G[a(p)] + δ_C[k(i), a(p)] )
  δ_G … 祖先ごとの全体の効果、δ_C … 条件 k（面×距離帯）ごとの上乗せ（δ_G より強く縮める）、φ_p … 位置 p の重み（データから推定。理論値は 2^-(gen-1)）。
  φ を固定して δ をリッジで解き、δ を固定して φ を 62 次元の回帰で解く、を繰り返す（φ は理論値へ縮約、φ_父 = 1 に正規化して報告）。
作り直しは半年ごと。その馬自身の結果が δ に入らないよう、馬を fold に分け、その fold の馬を除いた走で学習したモデルで予測する（クロスフィット）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from src.pipeline.features import tp38_pedigree as P

N_POS = 62
POS_GEN = np.concatenate([np.full(2 ** g, g) for g in range(1, 6)])
POS_POS = np.concatenate([np.arange(2 ** g) for g in range(1, 6)])
POS_SIDE = POS_POS >> (POS_GEN - 1)
POS_TM = (POS_POS == 0)
POS_PRIOR = 0.5 ** (POS_GEN - 1.0)


def pos_index(gen, pos) -> np.ndarray:
    return ((1 << np.asarray(gen, dtype=np.int64)) - 2 + np.asarray(pos, dtype=np.int64)).astype(np.int32)


class Spec:
    lam_g = 30.0
    lam_c = 100.0
    lam_phi = 0.05          # φ の事前（理論値）への縮約の強さ（CᵀC の平均対角に対する比）
    lam_phi_strong = 1.0    # 強く縮める版（als2）
    n_als = 3
    min_desc_g = 6
    min_desc_c = 40
    n_fold = 3
    iters = 200


def universes(anc: pd.DataFrame, min_g: int, min_c: int):
    """全体の効果・条件別の上乗せを持つ祖先（産駒の馬数が min 以上）の祖先コード → 列番号。"""
    cnt = anc.groupby("a")["h"].nunique()
    ga = np.sort(cnt.index[cnt >= min_g].to_numpy())
    ca = np.sort(cnt.index[cnt >= min_c].to_numpy())
    return ga, ca


def pos_ancestor_matrix(anc: pd.DataFrame, n_horses: int, universe: np.ndarray, n_anc: int) -> np.ndarray:
    """馬 × 62 の位置 → 祖先の列番号（universe に無ければ -1）。"""
    col = np.full(n_anc, -1, dtype=np.int32)
    col[universe] = np.arange(len(universe), dtype=np.int32)
    M = np.full((n_horses, N_POS), -1, dtype=np.int32)
    M[anc["h"].to_numpy(), pos_index(anc["gen"].to_numpy(), anc["pos"].to_numpy())] = col[anc["a"].to_numpy()]
    return M


def _design(hr: np.ndarray, kr: np.ndarray | None, PAg: np.ndarray, PAc: np.ndarray, n_g: int, n_c: int, n_k: int, phi: np.ndarray, scale_c: float) -> sp.csr_matrix:
    n = len(hr)
    rows = np.repeat(np.arange(n, dtype=np.int64), N_POS)
    jg = PAg[hr].ravel()
    m = jg >= 0
    Xg = sp.coo_matrix((np.tile(phi, n)[m], (rows[m], jg[m])), shape=(n, n_g)).tocsr()
    jc = PAc[hr].ravel()
    kk = np.repeat(kr, N_POS)
    m2 = (jc >= 0) & (kk >= 0)
    Xc = sp.coo_matrix((scale_c * np.tile(phi, n)[m2], (rows[m2], kk[m2].astype(np.int64) * n_c + jc[m2])), shape=(n, n_k * n_c)).tocsr()
    return sp.hstack([Xg, Xc], format="csr")


def _contrib(hr: np.ndarray, kr: np.ndarray, PAg, PAc, th_g: np.ndarray, th_c: np.ndarray, n_c: int, scale_c: float) -> np.ndarray:
    """走 × 62 位置の『祖先の効果』（φ を掛ける前）。"""
    jg = PAg[hr]
    cg = np.where(jg >= 0, th_g[np.maximum(jg, 0)], 0.0)
    jc = PAc[hr]
    ok = (jc >= 0) & (kr[:, None] >= 0)
    idx = np.where(ok, kr[:, None].astype(np.int64) * n_c + np.maximum(jc, 0), 0)
    cc = np.where(ok, scale_c * th_c[idx], 0.0)
    return cg + cc, cg, cc


def fit_one(hr, kr, y, PAg, PAc, n_g, n_c, n_k, sp_: type[Spec] = Spec, phi0: np.ndarray | None = None, lam_phi: float | None = None):
    """学習。戻り値: (履歴 [(φ, θ_G, θ_C)]（先頭 = φ を理論値に固定した解、末尾 = ALS 後）, scale_c)。"""
    scale_c = float(np.sqrt(sp_.lam_g / sp_.lam_c))
    lam_phi = sp_.lam_phi if lam_phi is None else lam_phi
    phi = POS_PRIOR.copy() if phi0 is None else phi0.copy()
    hist = []
    th_g = th_c = None
    for it in range(sp_.n_als + 1):
        X = _design(hr, kr, PAg, PAc, n_g, n_c, n_k, phi, scale_c)
        sol = lsqr(X, y, damp=float(np.sqrt(sp_.lam_g)), atol=1e-5, btol=1e-5, iter_lim=sp_.iters)[0]
        th_g, th_c = sol[:n_g], sol[n_g:]
        hist.append((phi.copy(), th_g.copy(), th_c.copy()))
        if it == sp_.n_als:
            break
        C, _, _ = _contrib(hr, kr, PAg, PAc, th_g, th_c, n_c, scale_c)
        CtC = C.T @ C
        lam = lam_phi * float(np.trace(CtC)) / N_POS
        phi = np.linalg.solve(CtC + lam * np.eye(N_POS), C.T @ y + lam * POS_PRIOR)
        phi = np.maximum(phi, 0.0)
    return hist, scale_c


def _features(hr, kr, PAg, PAc, th_g, th_c, n_c, scale_c, phi, y=None):
    """予測の表と、位置ごとの『成績との共分散 mean(y·φ_p·C_p)』『寄与の標準偏差』（y があるとき）。"""
    C, cg, cc = _contrib(hr, kr, PAg, PAc, th_g, th_c, n_c, scale_c)
    W = C * phi[None, :]
    imp = None
    if y is not None:
        ok = ~np.isnan(y)
        imp = {"cov": (W[ok] * y[ok, None]).mean(axis=0).tolist(), "sd": W[ok].std(axis=0).tolist(), "n": int(ok.sum())}
    f = {"z": C @ phi, "zg": cg @ phi, "zc": cc @ phi, "z_s0": C[:, POS_SIDE == 0] @ phi[POS_SIDE == 0], "z_s1": C[:, POS_SIDE == 1] @ phi[POS_SIDE == 1],
         "z_g12": C[:, POS_GEN <= 2] @ phi[POS_GEN <= 2], "z_g3p": C[:, POS_GEN >= 3] @ phi[POS_GEN >= 3], "z_tm": C[:, POS_TM] @ phi[POS_TM]}
    return pd.DataFrame(f), imp


def ridge_features(R: P.Runs, anc: pd.DataFrame, n_horses: int, n_anc: int, windows: list[tuple[str, str]], cond: str = "sd", spec: type[Spec] = Spec, log=None):
    """windows: [(作り直し日 T, 次の作り直し日)]。T より前の走で学習し、[T, 次) の走を予測する。
    戻り値: ({'fix': 表, 'als': 表}, 作り直しごとの φ の記録)。固定 = φ を理論値 2^-(gen-1) に固定、als = φ を推定。"""
    ga, ca = universes(anc, spec.min_desc_g, spec.min_desc_c)
    PAg = pos_ancestor_matrix(anc, n_horses, ga, n_anc)
    PAc = pos_ancestor_matrix(anc, n_horses, ca, n_anc)
    n_g, n_c = len(ga), len(ca)
    kcodes = R.cond_codes[cond]
    n_k = int(kcodes.max()) + 1
    rng = np.random.default_rng(0)
    fold_of_h = rng.integers(0, spec.n_fold, size=n_horses)           # 馬の fold（静的な割り当て。成績は使わない）
    cols = ["z", "zg", "zc", "z_s0", "z_s1", "z_g12", "z_g3p", "z_tm"]
    out = {k: pd.DataFrame(np.nan, index=np.arange(R.n), columns=cols) for k in ("fix", "als", "als2")}
    phis = []
    y_all, h_all, d_all = R.y, R.h, R.date
    for (T0, T1) in windows:
        t0, t1 = np.datetime64(T0, "D"), np.datetime64(T1, "D")
        tr_base = (d_all < t0) & (h_all >= 0) & ~np.isnan(y_all)
        ap = np.flatnonzero((d_all >= t0) & (d_all < t1) & (h_all >= 0))
        if tr_base.sum() < 1000 or len(ap) == 0:
            continue
        for f in range(spec.n_fold):
            tr = np.flatnonzero(tr_base & (fold_of_h[np.maximum(h_all, 0)] != f))
            ap_f = ap[fold_of_h[h_all[ap]] == f]
            if len(ap_f) == 0:
                continue
            hist, scale_c = fit_one(h_all[tr], kcodes[tr], y_all[tr], PAg, PAc, n_g, n_c, n_k, spec)
            hist2, _ = fit_one(h_all[tr], kcodes[tr], y_all[tr], PAg, PAc, n_g, n_c, n_k, spec, lam_phi=spec.lam_phi_strong)
            rec = {"T": T0, "fold": f, "phi": hist[-1][0].tolist(), "phi2": hist2[-1][0].tolist(), "n_train": int(len(tr))}
            for name, h_ in (("fix", hist[0]), ("als", hist[-1]), ("als2", hist2[-1])):
                phi, th_g, th_c = h_
                F, imp = _features(h_all[ap_f], kcodes[ap_f], PAg, PAc, th_g, th_c, n_c, scale_c, phi, y_all[ap_f])
                out[name].loc[ap_f, F.columns] = F.to_numpy()
                rec[f"imp_{name}"] = imp
            phis.append(rec)
            if log:
                log(f"  ridge T={T0} fold={f}: train {len(tr):,} apply {len(ap_f):,} φ(父)={hist[-1][0][0]:.3f} φ(母父)={hist[-1][0][4]:.3f} φ(父父)={hist[-1][0][2]:.3f}")
    return {k: v.astype("float32") for k, v in out.items()}, phis
