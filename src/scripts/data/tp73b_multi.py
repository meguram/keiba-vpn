"""tp-73b 複数の軸が同時に近いペア・組の分析（tp-41 の拡張）。市場オッズ・払戻は分析（回収率）にだけ使い、特徴量にしない。モデルの学習はしない。

python -m src.scripts.data.tp73b_multi pairs      # 2022〜2026 のペア表（tp-41 の軸 + 母系タイプ）と期待（割引 Harville）。V4 の予測は master の p_win/p_top2/p_top3
python -m src.scripts.data.tp73b_multi describe   # 近い軸の数 k = 0/1/2/3+ の 観測/期待、2 軸の全組合せ、相乗か加算か（ロジット・オフセット GLM、レース単位クラスタ頑健）
python -m src.scripts.data.tp73b_multi joint      # 馬連・ワイド・3連複・馬単の対数損失: 補正なし／単一軸／並列条件（±d_str4）。2022-23→2024、2022-24→2025（1 回）、2022-25→2026（記述）
python -m src.scripts.data.tp73b_multi bet        # 補正した確率 × 確定オッズ ≥ θ で選ぶ回収率（補正あり/なし）

事前に固定: 軸 = 力(d_str)・位置取り(d_sty)・馬番(d_gate)・適性(d_apt)・ともに先行(both_front)・母系タイプが同じ(tmat)。『近い』= 連続軸は発見期間（2022–2024）の下位 20%。
季節は同じレースの中では全馬同じなので軸にならない。馬場適性（pref_hard）は被覆が 16〜19% で外した。出力 <TP73B_OUT>（既定 /home/jovyan/work/tp73b_out）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("KEIBA_ROOT") or "/home/jovyan/work/keiba-vpn")
OUT = Path(os.environ.get("TP73B_OUT") or "/home/jovyan/work/tp73b_out")
TP73 = Path(os.environ.get("TP73_OUT") or "/home/jovyan/work/tp73_out")
TP59 = Path(os.environ.get("TP59_OUT") or "/home/jovyan/work/tp59_out")
BET = ROOT / "data/local/modeling/betting"
AX = ["str", "sty", "gate", "apt", "front", "type"]
CONT = {"str": "d_str", "sty": "d_sty", "gate": "d_gate", "apt": "d_apt"}
PAIRS_AX = list(combinations(range(len(AX)), 2))
SEED = 20261022


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ ペア表
def cmd_pairs(a=None) -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from src.pipeline.features import factory_ped as fp
    from src.pipeline.features import factory_self as fs
    from src.pipeline.features import tp41_similar as ts
    from src.pipeline.inference import exotic_probabilities as ex

    OUT.mkdir(parents=True, exist_ok=True)
    tables = ROOT / "data/page_reference/tables"
    years = list(range(2020, 2027))
    runs = fs.load_runs(tables, years)
    megu = ts.load_megu(tables, years)
    T = ts.runner_table(runs, megu)
    A, _ = fp.build_ped_tables(T["horse_id"].unique())
    T = ts.add_static(T, A)
    T = ts.standardize_strength(T)
    T = ts.sort_for_races(T)
    m = pd.read_parquet(TP59 / "master.parquet", columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"])
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    m = m.dropna(subset=["p_win", "p_top2", "p_top3"]).drop_duplicates(["race_id", "horse_id"])
    full = T.groupby("race_id").size()
    T = T.merge(m, on=["race_id", "horse_id"], how="inner")
    cnt = T.groupby("race_id").size()
    keep = cnt.index[(cnt == full.reindex(cnt.index)) & (cnt >= 5)]
    T = ts.sort_for_races(T[T["race_id"].isin(keep)])
    T["year"] = T["date"].dt.year
    hx = pd.read_parquet(TP73 / "horses_idx.parquet")[["horse_id", "h"]]
    ty = pd.read_parquet(TP73 / "type_s2_mat_R_pro.parquet")
    T = T.merge(hx, on="horse_id", how="left")
    T["tmat"] = T["h"].map(ty.set_index("h")["type"]).fillna(-1).astype(int)
    T = T.reset_index(drop=True)
    log(f"レース {T['race_id'].nunique()}（行 {len(T)}）、年 {sorted(T['year'].unique())}")
    Pp = ts.race_pairs(T, None, None)
    i, j = Pp["i"].to_numpy(), Pp["j"].to_numpy()
    fin, nf = T["fin"].to_numpy(float), T["nfield"].to_numpy(float)
    Pp["o3"], Pp["o2"] = ((fin <= 3) & (fin > 0))[i] & ((fin <= 3) & (fin > 0))[j], ((fin <= 2) & (fin > 0))[i] & ((fin <= 2) & (fin > 0))[j]
    Pp["o3"], Pp["o2"] = Pp["o3"].astype(float), Pp["o2"].astype(float)
    p3 = np.clip(T["p_top3"].to_numpy(float), 1e-6, 1 - 1e-6)
    lg = np.log(p3 / (1 - p3))
    Pp["d_str4"] = np.abs(lg[i] - lg[j])
    Pp["year"] = T["year"].to_numpy()[i]
    Pp["nf"] = nf[i]
    t = T["tmat"].to_numpy()
    Pp["same_type"] = ((t[i] == t[j]) & (t[i] >= 0)).astype(float)
    W3, W2 = np.full(len(Pp), np.nan), np.full(len(Pp), np.nan)
    rid = T["race_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, len(rid)]
    pos, t0 = 0, time.time()
    for k, (a_, b_) in enumerate(zip(starts[:-1], starts[1:])):
        mm = b_ - a_
        npair = mm * (mm - 1) // 2
        dist, _ = ex.fit_rank_distribution(T["p_win"].to_numpy(float)[a_:b_], T["p_top2"].to_numpy(float)[a_:b_], T["p_top3"].to_numpy(float)[a_:b_])
        wide, uma = dist.all_wide(), dist.all_umaren()
        ii, jj = np.triu_indices(mm, 1)
        W3[pos:pos + npair] = [wide[(int(x), int(y))] for x, y in zip(ii, jj)]
        W2[pos:pos + npair] = [uma[(int(x), int(y))] for x, y in zip(ii, jj)]
        pos += npair
        if k % 3000 == 0:
            log(f"  期待 {k}/{len(starts) - 1} ({time.time() - t0:.0f}s)")
    Pp["w3"], Pp["w2"] = W3, W2
    Pp.to_parquet(OUT / "pairs.parquet", index=False)
    T.to_parquet(OUT / "runners.parquet", index=False)
    log(f"ペア {len(Pp):,} 件、同時3着以内 観測 {Pp['o3'].mean():.4f} / 期待 {np.nanmean(W3):.4f}")


def add_flags(Pp: pd.DataFrame, cuts: dict | None = None):
    """『近い』の旗。連続軸は発見期間（2022–2024）の下位 20%。"""
    if cuts is None:
        d = Pp[Pp["year"].between(2022, 2024)]
        cuts = {a: float(d[c].quantile(0.2)) for a, c in CONT.items()}
    F = pd.DataFrame(index=Pp.index)
    for a, c in CONT.items():
        F[a] = (Pp[c] <= cuts[a]).astype(float)
    F["front"] = Pp["both_front"].fillna(0.0)
    F["type"] = Pp["same_type"]
    F["k"] = F[AX].sum(axis=1)
    return F, cuts


# ------------------------------------------------------------------ 観測/期待・用量反応・2 軸・相乗/加算
def oe_boot(race, o, e, mask, B: int = 400, seed: int = 0):
    r = pd.factorize(race[mask])[0]
    R = r.max() + 1 if len(r) else 0
    if R == 0:
        return None
    O = np.bincount(r, weights=o[mask], minlength=R)
    E = np.bincount(r, weights=e[mask], minlength=R)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(B)]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = (W @ O) / (W @ E)
    return {"n": int(mask.sum()), "O": float(O.sum()), "E": float(E.sum()), "oe": float(O.sum() / max(E.sum(), 1e-12)), "lo": float(np.nanpercentile(rr, 2.5)), "hi": float(np.nanpercentile(rr, 97.5))}


def glm_fit(y, X, off, groups):
    import statsmodels.api as sm

    r = sm.GLM(y, sm.add_constant(X, has_constant="add"), family=sm.families.Binomial(), offset=off).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return r


def cmd_describe(a=None) -> None:
    from scipy.stats import norm

    Pp = pd.read_parquet(OUT / "pairs.parquet")
    F, cuts = add_flags(Pp)
    race = Pp["race_id"].to_numpy()
    periods = {"disc": Pp["year"].between(2022, 2024).to_numpy(), "test2025": (Pp["year"] == 2025).to_numpy(), "desc2026": (Pp["year"] == 2026).to_numpy()}
    res: dict = {"cuts": cuts, "n_pairs": {k: int(v.sum()) for k, v in periods.items()}, "coverage": {a: float(Pp[c].notna().mean()) for a, c in CONT.items()}}
    d4z = ((Pp["d_str4"] - Pp.loc[periods["disc"], "d_str4"].mean()) / Pp.loc[periods["disc"], "d_str4"].std()).to_numpy()
    for tgt, o, e in (("w3", "o3", "w3"), ("w2", "o2", "w2")):
        ov, ev = Pp[o].to_numpy(float), Pp[e].to_numpy(float)
        R: dict = {}
        for per, pm in periods.items():
            P: dict = {"k": {}, "single": {}, "two": {}}
            for kk, lab in ((0, "0"), (1, "1"), (2, "2"), (3, "3+")):
                mk = pm & ((F["k"] == kk) if kk < 3 else (F["k"] >= 3)).to_numpy()
                P["k"][lab] = oe_boot(race, ov, ev, mk)
            k5 = F[["str", "sty", "gate", "apt", "type"]].sum(axis=1)          # 補足（事後）: 先行の旗を除いた近い軸の数（both_front は tp-41 で逆向きと分かっていた）
            P["k5"] = {}
            for kk, lab in ((0, "0"), (1, "1"), (2, "2"), (3, "3+")):
                mk = pm & ((k5 == kk) if kk < 3 else (k5 >= 3)).to_numpy()
                P["k5"][lab] = oe_boot(race, ov, ev, mk)
            for ai, ax in enumerate(AX):
                P["single"][ax] = oe_boot(race, ov, ev, pm & (F[ax] == 1).to_numpy())
            for (ia, ib) in PAIRS_AX:
                mk = pm & ((F[AX[ia]] == 1) & (F[AX[ib]] == 1)).to_numpy()
                c = oe_boot(race, ov, ev, mk)
                if c:
                    s1, s2 = P["single"][AX[ia]]["oe"], P["single"][AX[ib]]["oe"]
                    c["mult"], c["add"] = s1 * s2, 1 + (s1 - 1) + (s2 - 1)
                P["two"][f"{AX[ia]}×{AX[ib]}"] = c
            R[per] = P
        # GLM（ロジット・オフセット）: 主効果のみ vs 並列条件の項。d_str4 の有無
        gl: dict = {}
        for per, pm in periods.items():
            idx = np.where(pm & np.isfinite(ev) & (ev > 0) & (ev < 1))[0]
            y = ov[idx]
            off = np.log(ev[idx] / (1 - ev[idx]))
            grp = pd.factorize(race[idx])[0]
            base = np.c_[F.loc[Pp.index[idx], AX].to_numpy(), d4z[idx]]
            names = AX + ["d_str4"]
            r0 = glm_fit(y, base, off, grp)
            gl[per] = {"main": {n: [float(r0.params[i + 1]), float(r0.bse[i + 1])] for i, n in enumerate(names)}}
            k_ = F["k"].to_numpy()[idx]
            Xc = np.c_[base, (k_ == 2).astype(float), (k_ >= 3).astype(float)]
            r1 = glm_fit(y, Xc, off, grp)
            gl[per]["count_terms"] = {"k2": [float(r1.params[-2]), float(r1.bse[-2])], "k3p": [float(r1.params[-1]), float(r1.bse[-1])], "llf_main": float(r0.llf), "llf_count": float(r1.llf)}
            inter = {}
            for (ia, ib) in PAIRS_AX:
                x = (F[AX[ia]].to_numpy()[idx] * F[AX[ib]].to_numpy()[idx])
                if x.sum() < 200:
                    continue
                ri = glm_fit(y, np.c_[base, x], off, grp)
                inter[f"{AX[ia]}×{AX[ib]}"] = {"coef": float(ri.params[-1]), "se": float(ri.bse[-1]), "z": float(ri.params[-1] / ri.bse[-1]), "n_cell": int(x.sum())}
            gl[per]["inter"] = inter
        R["glm"] = gl
        res[tgt] = R
        log(f"{tgt}: 完了")
    json.dump(res, open(OUT / "describe.json", "w"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 組合せ券の確率の補正
def feature_sets(Pp: pd.DataFrame, F: pd.DataFrame):
    d = Pp[Pp["year"].between(2022, 2024)]
    z = lambda c: ((Pp[c] - d[c].mean()) / d[c].std()).fillna(0.0)
    d4 = z("d_str4").to_numpy()
    single = np.c_[z("d_str").to_numpy(), z("d_sty").to_numpy(), z("d_gate").to_numpy(), z("d_apt").to_numpy(), Pp["both_front"].fillna(0.0).to_numpy()]
    flags = F[AX].to_numpy()
    inter = np.stack([flags[:, a] * flags[:, b] for a, b in PAIRS_AX], 1)
    k = F["k"].to_numpy()
    par = np.c_[flags, inter, (k == 2).astype(float), (k >= 3).astype(float)]
    return {"base": np.zeros((len(Pp), 0)), "d_str4": d4[:, None], "single": single, "single+d4": np.c_[single, d4],
            "parallel": par, "parallel+d4": np.c_[par, d4], "single+par": np.c_[single, par], "single+par+d4": np.c_[single, par, d4]}


def build_races(T: pd.DataFrame, Pp: pd.DataFrame, FS: dict):
    """レースごとの 馬連/3連複/馬単 の log P とペア特徴、実際の位置。"""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.pipeline.inference import exotic_probabilities as ex

    rid = T["race_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, len(rid)]
    fin = T["fin"].to_numpy(float)
    years = T["year"].to_numpy()
    hn = T["horse_number"].to_numpy().astype(int)
    out, pos = [], 0
    for a_, b_ in zip(starts[:-1], starts[1:]):
        m = b_ - a_
        npair = m * (m - 1) // 2
        ii, jj = np.triu_indices(m, 1)
        pid = np.full((m, m), -1)
        pid[ii, jj] = np.arange(npair)
        pid[jj, ii] = np.arange(npair)
        dist, _ = ex.fit_rank_distribution(T["p_win"].to_numpy(float)[a_:b_], T["p_top2"].to_numpy(float)[a_:b_], T["p_top3"].to_numpy(float)[a_:b_])
        trio = dist._trio_sums()
        keys = np.array(list(trio.keys()))
        lp3 = np.log(np.maximum(np.array([trio[k] for k in trio]), 1e-300))
        um = dist.all_umaren()
        ukeys = np.array(list(um.keys()))
        lp2 = np.log(np.maximum(np.array([um[k] for k in um]), 1e-300))
        okeys = np.array([(x, y) for x in range(m) for y in range(m) if x != y])
        lpo = np.log(np.maximum(np.array([dist.umatan(int(x), int(y)) for x, y in okeys]), 1e-300))
        w = dist.all_wide()
        wk = np.array(list(w.keys()))
        w3 = np.array([w[k] for k in w])
        order = np.argsort(fin[a_:b_])
        top3, top2 = tuple(sorted(order[:3].tolist())), tuple(sorted(order[:2].tolist()))
        key_of = lambda arr: {tuple(r): i for i, r in enumerate(arr.tolist())}
        k3, k2, ko = key_of(keys), key_of(ukeys), key_of(okeys)
        wmap = {tuple(r): i for i, r in enumerate(wk.tolist())}
        top3set = set(order[:3].tolist())
        wy = np.array([1.0 if (int(x) in top3set and int(y) in top3set) else 0.0 for x, y in wk])
        out.append({"year": int(years[a_]), "rid": str(rid[a_]), "hn": hn[a_:b_], "pos": (pos, pos + npair),
                    "tri_p": np.stack([pid[keys[:, 0], keys[:, 1]], pid[keys[:, 0], keys[:, 2]], pid[keys[:, 1], keys[:, 2]]], 1), "tri_lp": lp3, "tri_act": k3.get(top3, -1), "tri_keys": keys,
                    "um_p": pid[ukeys[:, 0], ukeys[:, 1]], "um_lp": lp2, "um_act": k2.get(top2, -1), "um_keys": ukeys,
                    "or_p": pid[okeys[:, 0], okeys[:, 1]], "or_lp": lpo, "or_act": ko.get((int(order[0]), int(order[1])), -1), "or_keys": okeys,
                    "w_p": pid[wk[:, 0], wk[:, 1]], "w_off": np.log(np.clip(w3, 1e-9, 1 - 1e-9) / (1 - np.clip(w3, 1e-9, 1 - 1e-9))), "w_y": wy, "w_keys": wk})
        pos += npair
    return out


def soft_nll(g, races, F, kind):
    """softmax（組合せ全体の中で 1 つが起きる）の負の対数尤度と勾配。kind: tri/um/or"""
    from scipy.special import logsumexp

    tot, n = 0.0, 0
    grad = np.zeros_like(g)
    for r in races:
        act = r[f"{kind}_act"]
        if act < 0:
            continue
        a_, b_ = r["pos"]
        Fr = F[a_:b_]
        ps = Fr @ g if len(g) else np.zeros(b_ - a_)
        p = r[f"{kind}_p"]
        sc = r[f"{kind}_lp"] + (ps[p].sum(1) if p.ndim == 2 else ps[p])
        lse = logsumexp(sc)
        tot -= sc[act] - lse
        if len(g):
            pr = np.exp(sc - lse)
            wpair = np.zeros(b_ - a_)
            if p.ndim == 2:
                for c in range(3):
                    np.add.at(wpair, p[:, c], pr)
                    wpair[p[act, c]] -= 1.0
            else:
                np.add.at(wpair, p, pr)
                wpair[p[act]] -= 1.0
            grad += Fr.T @ wpair
        n += 1
    return tot / n, grad / n


def wide_nll(g, races, F):
    tot, n = 0.0, 0
    grad = np.zeros_like(g)
    for r in races:
        a_, b_ = r["pos"]
        Fr = F[a_:b_]
        ps = Fr @ g if len(g) else np.zeros(b_ - a_)
        z = r["w_off"] + ps[r["w_p"]]
        p = 1 / (1 + np.exp(-z))
        y = r["w_y"]
        tot += float(-(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).sum())
        n += len(y)
        if len(g):
            wp = np.zeros(b_ - a_)
            np.add.at(wp, r["w_p"], p - y)
            grad += Fr.T @ wp
    return tot / n, grad / n


def fit_g(races, F, kind):
    from scipy.optimize import minimize

    k = F.shape[1]
    if k == 0:
        return np.zeros(0)
    fn = (lambda g: wide_nll(g, races, F)) if kind == "w" else (lambda g: soft_nll(g, races, F, kind))

    def f(g):
        v, gr = fn(g)
        return v + 1e-4 * float(g @ g), gr + 2e-4 * g

    return minimize(f, np.zeros(k), jac=True, method="L-BFGS-B", options={"maxiter": 120}).x


def race_losses(g, races, F, kind):
    """レースごとの負の対数尤度（ワイドは 1 ペア平均×ペア数でなくレースの合計）。"""
    from scipy.special import logsumexp

    out = []
    for r in races:
        a_, b_ = r["pos"]
        Fr = F[a_:b_]
        ps = Fr @ g if len(g) else np.zeros(b_ - a_)
        if kind == "w":
            z = r["w_off"] + ps[r["w_p"]]
            p = 1 / (1 + np.exp(-z))
            y = r["w_y"]
            out.append(float(-(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).sum()) / len(y))
            continue
        act = r[f"{kind}_act"]
        if act < 0:
            out.append(np.nan)
            continue
        p = r[f"{kind}_p"]
        sc = r[f"{kind}_lp"] + (ps[p].sum(1) if p.ndim == 2 else ps[p])
        out.append(float(-(sc[act] - logsumexp(sc))))
    return np.array(out)


def cmd_joint(a=None) -> None:
    Pp = pd.read_parquet(OUT / "pairs.parquet")
    T = pd.read_parquet(OUT / "runners.parquet")
    Fl, cuts = add_flags(Pp)
    FS = feature_sets(Pp, Fl)
    races = build_races(T, Pp, FS)
    log(f"レース {len(races)}")
    folds = {"2024": ([2022, 2023], [2024]), "2025": ([2022, 2023, 2024], [2025]), "2026": ([2022, 2023, 2024, 2025], [2026])}
    res: dict = {}
    for kind, name in (("um", "馬連"), ("w", "ワイド"), ("tri", "3連複"), ("or", "馬単")):
        res[name] = {}
        for fold, (tr_y, te_y) in folds.items():
            tr = [r for r in races if r["year"] in tr_y]
            te = [r for r in races if r["year"] in te_y]
            base_loss = None
            for vname, F in FS.items():
                g = fit_g(tr, F, kind)
                loss = race_losses(g, te, F, kind)
                if vname == "base":
                    base_loss = loss
                d = loss - base_loss
                ok = ~np.isnan(d)
                rng = np.random.default_rng(0)
                bs = np.array([d[ok][rng.integers(0, ok.sum(), ok.sum())].mean() for _ in range(300)])
                res[name].setdefault(vname, {})[fold] = {"nll": float(np.nanmean(loss)), "diff": float(np.nanmean(d)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(ok.sum()),
                                                      "g": [float(x) for x in g]}
                log(f"{name} {fold} {vname}: nll {np.nanmean(loss):.5f} 差 {np.nanmean(d):+.5f} [{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}]")
    json.dump({"cuts": cuts, "res": res, "parallel_cols": AX + [f"{AX[a]}×{AX[b]}" for a, b in PAIRS_AX] + ["k2", "k3p"]}, open(OUT / "joint.json", "w"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 買い目（確率 × 確定オッズ ≥ θ）
def cmd_bet(a=None) -> None:
    from scipy.special import logsumexp

    Pp = pd.read_parquet(OUT / "pairs.parquet")
    T = pd.read_parquet(OUT / "runners.parquet")
    Fl, cuts = add_flags(Pp)
    FS = feature_sets(Pp, Fl)
    races = build_races(T, Pp, FS)
    od = pd.read_parquet(BET / "odds_pair.parquet")
    od = od[od["bet_type"].isin(["umaren", "wide", "umatan"])]
    odds = {}
    for r in od.itertuples(index=False):
        v = r.odds if (r.odds == r.odds) else r.odds_min
        if v == v:
            odds[(r.bet_type, r.race_id, r.combo)] = float(v)
    pay_df = pd.read_parquet(BET / "payoff.parquet")
    pay_df = pay_df[pay_df["bet_type"].isin(["umaren", "wide", "umatan"]) & pay_df["parsed"]]
    pay = {}
    for r in pay_df.itertuples(index=False):
        pay[(r.bet_type, r.race_id, r.combo)] = pay.get((r.bet_type, r.race_id, r.combo), 0) + int(r.payout)
    res: dict = {}
    thetas = (1.0, 1.1, 1.2)
    for kind, bt, name in (("um", "umaren", "馬連"), ("w", "wide", "ワイド"), ("or", "umatan", "馬単")):
        res[name] = {}
        for fold, (tr_y, te_y) in (("2024", ([2022, 2023], [2024])), ("2025", ([2022, 2023, 2024], [2025]))):
            tr = [r for r in races if r["year"] in tr_y]
            te = [r for r in races if r["year"] in te_y]
            for vname in ("base", "single+d4", "parallel+d4", "single+par+d4"):
                F = FS[vname]
                g = fit_g(tr, F, kind)
                rows = {th: [] for th in thetas}
                for r in te:
                    a_, b_ = r["pos"]
                    ps = F[a_:b_] @ g if len(g) else np.zeros(b_ - a_)
                    hn = r["hn"]
                    if kind == "w":
                        p = 1 / (1 + np.exp(-(r["w_off"] + ps[r["w_p"]])))
                        keys = r["w_keys"]
                        key_str = [f"{min(hn[x], hn[y])}-{max(hn[x], hn[y])}" for x, y in keys]
                    else:
                        kk = r[f"{kind}_keys"]
                        sc = r[f"{kind}_lp"] + ps[r[f"{kind}_p"]]
                        p = np.exp(sc - logsumexp(sc))
                        key_str = [(f"{min(hn[x], hn[y])}-{max(hn[x], hn[y])}" if kind == "um" else f"{hn[x]}>{hn[y]}") for x, y in kk]
                    ov = np.array([odds.get((bt, r["rid"], k), np.nan) for k in key_str])
                    ev = p * ov
                    for th in thetas:
                        sel = np.where(ev >= th)[0]
                        if len(sel) == 0:
                            rows[th].append((0.0, 0.0))
                            continue
                        ret = sum(pay.get((bt, r["rid"], key_str[s]), 0) for s in sel) / 100.0
                        rows[th].append((float(len(sel)), float(ret)))
                for th in thetas:
                    D = np.array(rows[th])
                    st, rt = D[:, 0].sum(), D[:, 1].sum()
                    rng = np.random.default_rng(0)
                    bs = []
                    for _ in range(300):
                        ix = rng.integers(0, len(D), len(D))
                        s_ = D[ix, 0].sum()
                        bs.append(D[ix, 1].sum() / s_ if s_ > 0 else np.nan)
                    res[name].setdefault(vname, {}).setdefault(fold, {})[str(th)] = {"bets": int(st), "races_bet": int((D[:, 0] > 0).sum()), "roi": float(rt / st) if st > 0 else None,
                                                                                    "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))] if st > 0 else None}
                    x = res[name][vname][fold][str(th)]
                    log(f"{name} {fold} {vname} θ={th}: 買い目 {x['bets']:,} 回収率 {x['roi']}")
    json.dump(res, open(OUT / "bet.json", "w"), ensure_ascii=False, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("pairs", "describe", "joint", "bet"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"pairs": cmd_pairs, "describe": cmd_describe, "joint": cmd_joint, "bet": cmd_bet}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
