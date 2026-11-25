"""穴馬 × コースの条件探しの、合成データの確かめ（1 号の決定 D1 改訂 R3 の段 0・模擬 A・模擬 B。着順は使わない）。

- 勝ち・3 着内は、レースの全頭の確定の q × 帯の比（2016〜21 の穴馬の 勝ち ÷ q、穴馬以外は 1）から Plackett-Luce（Harville）で引き直す。
  効きを入れるときは、穴馬の行の重みに倍率を掛けてからレースの中で比を作る。
- 段 0: 推定 τ²（排他を入れた積率法）が、合成の真の τ²（雑音なしで同じ式に通した値）を取り戻すか。真の値 0・約 0.01・約 0.02 を各 20 回。
- 模擬 A: 単勝 E・複勝 E3 の下限の候補（組で流す）ごとに、条件の数・覆う割合と、各コースに深さ 1 の条件を 1 つ選んで比を 1.3 倍にした合成での超過の勝ち数 S（1 号の決定 R3-2）。
- 段 1 の検出力（power1）: 効きなし 200 回の 97.5% 点を線に、×1.3・×1.15 で線を超える割合（R3-3 の規則に使った）。
- 記録の追加（power2）: 下限を満たす大きな条件に ×1.3・×1.5・×2 を植えた場合（レビューの指摘で追加。判断には使っていない）。
- 模擬 B: 真の回収率 1.15・0.78 で、段 2 の線を通る確率と点数の関係。

使い方（nice -n 19）:
  python -m src.research.course.longshot_synth stage0 --reps 20
  python -m src.research.course.longshot_synth simA --reps 20
  python -m src.research.course.longshot_synth simB --reps 2000
出力: <COURSE_OUT>/longshot_rules/synth/*.json。費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.research.course import longshot_rules as LR

OUTS = LR.OUTW / "synth"
FIT_END = 2021        # 関門と合成は 2016〜21 だけ
SIGMAS = (0.0, 0.18, 0.25, 0.4)   # 真の τ²（単勝の条件）が 0・約 0.01・約 0.02・約 0.034 になる倍率のばらつき


def load():
    d, L = LR.load_pop()
    R = pd.read_parquet(LR.OUTW / "race_all.parquet")
    return d, L, R


def band_ratio(d: pd.DataFrame) -> np.ndarray:
    """2016〜21 の穴馬の、確定オッズの帯ごとの 勝ち ÷ q（帯の比。本命・大穴の偏りの分）。"""
    s = d[pd.to_numeric(d["year"]).to_numpy() <= FIT_END]
    g = s.groupby("obin")
    return (g["win"].sum() / g["q"].sum()).reindex(range(LR.NB)).fillna(1.0).to_numpy()


class Sim:
    """レースの全頭の表 R の上で、穴馬の行（pop_row ≥ 0）の勝ち・3 着内を引き直す。"""

    def __init__(self, d: pd.DataFrame, R: pd.DataFrame, rho: np.ndarray):
        self.R = R
        self.rid = pd.factorize(R["race_id"])[0]
        self.pr = R["pop_row"].to_numpy()
        self.is_ls = self.pr >= 0
        self.base = R["q"].to_numpy(float).copy()
        ob = d["obin"].to_numpy()
        self.base[self.is_ls] = self.base[self.is_ls] * rho[ob[self.pr[self.is_ls]]]
        N = R.groupby(self.rid)["field"].transform("first").fillna(12).to_numpy(float)
        self.T = np.where(N <= 7, 2, 3)
        self.n_pop = len(d)

    def weights(self, mult: np.ndarray | None) -> np.ndarray:
        w = self.base.copy()
        if mult is not None:
            w[self.is_ls] *= mult[self.pr[self.is_ls]]
        return w

    def win_prob(self, mult: np.ndarray | None) -> np.ndarray:
        w = self.weights(mult)
        tot = np.bincount(self.rid, weights=w)
        p = w / tot[self.rid]
        out = np.zeros(self.n_pop)
        out[self.pr[self.is_ls]] = p[self.is_ls]
        return out

    def draw(self, mult: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Plackett-Luce（Gumbel の最大）で順位を引き、穴馬の行の 1 着・払戻の枠の中を返す。"""
        w = self.weights(mult)
        key = np.log(w) + rng.gumbel(size=len(w))
        order = np.lexsort((-key, self.rid))
        rank = np.empty(len(w), int)
        start = np.r_[0, np.flatnonzero(np.diff(self.rid[order])) + 1]
        grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(w)]))
        rank[order] = np.arange(len(w)) - start[grp]
        win, top = np.zeros(self.n_pop), np.zeros(self.n_pop)
        ls = self.is_ls
        win[self.pr[ls]] = (rank[ls] == 0)
        top[self.pr[ls]] = (rank[ls] < self.T[ls])
        return win, top


def inject(d: pd.DataFrame, L: pd.DataFrame, sigma: float, rng: np.random.Generator, n_types: int = 3) -> np.ndarray:
    """コースごとに型を n_types 個えらび、その水準ごとに対数正規の倍率（平均 1）を付け、行の倍率をその積にする。"""
    mult = np.ones(len(d))
    if sigma <= 0:
        return mult
    units = d["unit"].to_numpy()
    cols = list(L.columns)
    for u in np.unique(units):
        r = np.flatnonzero(units == u)
        for t in rng.choice(cols, size=n_types, replace=False):
            lv = L[t].to_numpy()[r].astype(str)
            uniq = np.unique(lv)
            g = dict(zip(uniq, np.exp(rng.normal(-sigma ** 2 / 2, sigma, len(uniq)))))
            mult[r] *= np.array([g[x] for x in lv])
    return mult


def tau_on(d: pd.DataFrame, O: np.ndarray, rows: np.ndarray, win: np.ndarray, top: np.ndarray, emin: tuple[float, float],
           noise: bool = True) -> dict:
    """rows（2016〜21）だけで、与えた勝ち・3 着内（期待値でもよい）の τ² を推す。"""
    dd = d.copy()
    dd["win"], dd["top3"] = win, top
    s = np.zeros(len(d), bool)
    s[rows] = True
    rate, _ = LR.group_rates(dd, s)
    Y = LR.ymat(dd)
    ul = LR.make_ulist(dd, O, emin, rows)
    sh = LR.estimate_shrink(dd, O, Y, ul, {"rate": rate, "s": s}, emin, noise=noise)
    return {k: v for k, v in sh.items() if k.startswith("tau2") or k.startswith("n_")}


def cmd_stage0(reps: int, emin: tuple[float, float]) -> None:
    OUTS.mkdir(parents=True, exist_ok=True)
    d, L, R = load()
    rho = band_ratio(d)
    sim = Sim(d, R, rho)
    O, _, _ = LR.onehot(L)
    rows = np.flatnonzero(pd.to_numeric(d["year"]).to_numpy() <= FIT_END)
    res = {"rho": rho.tolist(), "emin": list(emin), "levels": []}
    for sigma in SIGMAS:
        rng = np.random.default_rng(int(sigma * 1000) + 7)
        mult = inject(d, L, sigma, rng)
        pw = sim.win_prob(mult)
        # 3 着内の期待値は 200 回の引き直しの平均
        acc = np.zeros(len(d))
        for _ in range(200):
            acc += sim.draw(mult, rng)[1]
        p3 = acc / 200
        truth = tau_on(d, O, rows, pw, p3, emin, noise=False)
        ests = []
        t0 = time.time()
        for k in range(reps):
            w, t = sim.draw(mult, np.random.default_rng(10_000 * (k + 1) + int(sigma * 1000)))
            ests.append(tau_on(d, O, rows, w, t, emin))
        E = pd.DataFrame(ests)
        res["levels"].append({"sigma": sigma, "truth": truth, "est_mean": E.mean().to_dict(), "est_sd": E.std().to_dict(),
                              "est_se": (E.std() / np.sqrt(len(E))).to_dict(), "seconds": round(time.time() - t0)})
        print(json.dumps(res["levels"][-1], ensure_ascii=False), flush=True)
    (OUTS / f"stage0_{emin[0]:g}_{emin[1]:g}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


GRID_B = [100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000]


def _lines_b(roi, roi_lo, mr, mr_lo, be, base) -> np.ndarray:
    """段 2 の線のうち、模擬で当てられるもの（帰無の p・種・前向き・§77 は除く）。"""
    return (roi >= 1.0) & (roi_lo > base) & (mr >= be) & (mr_lo > 1.0)


def cmd_simB(reps: int) -> None:
    """真の回収率 1.15・0.78 で、点数ごとに段 2 の線を通る確率。点は実際の穴馬（2016〜24）の q・確定オッズ・開催から引く。
    単勝: 勝つ確率 = 真の回収率 ÷ 確定オッズ。複勝: 3 着内の確率 = 真の回収率 ÷ 帯・頭数の群の複勝の払戻の平均。
    区間は開催で束ねた正規近似（片側 97.5%）。基準線は 0.78 とする（1 号の決定）。"""
    OUTS.mkdir(parents=True, exist_ok=True)
    d, _, _ = load()
    rate, pay = LR.group_rates(d, np.ones(len(d), bool))
    odds, q = d["odds_final"].to_numpy(float), d["q"].to_numpy(float)
    g = d["e3g"].to_numpy()
    e3, F = rate[g], pay[g]
    kai = pd.factorize(d["kaisai"])[0]
    base = 0.78
    out = {"base": base, "grid": GRID_B, "reps": reps, "kinds": {}}
    for kind in ("t", "f"):
        res = {}
        for R in (1.15, 0.78):
            pr = []
            for n in GRID_B:
                rng = np.random.default_rng(n * 7 + int(R * 100) + (0 if kind == "t" else 1))
                ok = 0
                for _ in range(reps):
                    i = rng.integers(len(d), size=n)
                    if kind == "t":
                        p = np.minimum(R / odds[i], 1.0)
                        hit = rng.random(n) < p
                        ret, exp = np.where(hit, odds[i], 0.0), q[i]
                        be = 1.0 / np.mean(q[i] * odds[i])
                    else:
                        p = np.minimum(R / F[i], 1.0)
                        hit = rng.random(n) < p
                        ret, exp = np.where(hit, F[i], 0.0), e3[i]
                        be = 1.0 / np.mean(e3[i] * F[i])
                    c = kai[i]
                    roi = ret.sum() / n
                    mr = hit.sum() / exp.sum()
                    Rc, Nc = np.bincount(c, weights=ret), np.bincount(c).astype(float)
                    Hc, Ec = np.bincount(c, weights=hit.astype(float)), np.bincount(c, weights=exp)
                    se_roi = np.sqrt(np.sum((Rc - roi * Nc) ** 2)) / n
                    se_mr = np.sqrt(np.sum((Hc - mr * Ec) ** 2)) / exp.sum()
                    ok += bool(_lines_b(roi, roi - 1.96 * se_roi, mr, mr - 1.96 * se_mr, be, base))
                pr.append(ok / reps)
            res[str(R)] = pr
        good = [n for n, a, b in zip(GRID_B, res["1.15"], res["0.78"]) if a >= 0.5 and b <= 0.025]
        res["min_points"] = good[0] if good else None
        out["kinds"][kind] = res
        print(kind, json.dumps(res), flush=True)
    (OUTS / "simB.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


CANDS_A = [(8.0, 20.0), (12.0, 30.0), (15.0, 45.0)]   # (単勝 E, 複勝 E3) の候補（券種ごとに独立に判定する）
EFFECT_MULT = 1.3


def subset_2021(d: pd.DataFrame, L: pd.DataFrame, R: pd.DataFrame):
    """2016〜21 の穴馬と、そのレースの全頭の表（pop_row を付け直す）。"""
    keep = pd.to_numeric(d["year"]).to_numpy() <= FIT_END
    newrow = np.full(len(d), -1)
    newrow[keep] = np.arange(keep.sum())
    d6, L6 = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True)
    races = set(d6["race_id"])
    R6 = R[R["race_id"].isin(races)].reset_index(drop=True).copy()
    pr = R6["pop_row"].to_numpy()
    R6["pop_row"] = np.where(pr >= 0, newrow[np.maximum(pr, 0)], -1)
    return d6, L6, R6


def effect_mult(d: pd.DataFrame, O: np.ndarray, rng: np.random.Generator, ratio: float = EFFECT_MULT) -> tuple[np.ndarray, list[dict]]:
    """コースごとに、深さ 1 の条件を 1 つだけ等確率で選び、その行の比を 1.3 倍にする（1 号の決定 D1 改訂 R3-2）。
    選べる条件は、コースの穴馬の 5〜50% を覆い、異なる馬が 30 頭以上のもの。選んだ条件の大きさ（走・E）も返す。"""
    units = d["unit"].to_numpy()
    horse = pd.factorize(d["horse_id"].astype(str))[0]
    q = d["q"].to_numpy(float)
    mult = np.ones(len(d))
    picked = []
    for u in np.unique(units):
        r = np.flatnonzero(units == u)
        o = O[r] > 0
        share = o.mean(0)
        cand = [c for c in np.flatnonzero((share >= 0.05) & (share <= 0.50)) if len(np.unique(horse[r[o[:, c]]])) >= 30]
        if not cand:
            continue
        c = cand[int(rng.integers(len(cand)))]
        rows = r[o[:, c]]
        mult[rows] *= ratio
        picked.append({"unit": u, "rows": int(len(rows)), "E": float(q[rows].sum()), "share": float(share[c])})
    return mult, picked


def _synth_frame(d: pd.DataFrame, win: np.ndarray, top: np.ndarray, F: np.ndarray) -> pd.DataFrame:
    dd = d.copy()
    dd["win"], dd["top3"] = win.astype(float), top.astype(float)
    dd["rt"] = dd["odds_final"].to_numpy(float) * dd["win"].to_numpy()
    dd["rf"] = F * dd["top3"].to_numpy()
    return dd


def _simA_one(args) -> dict:
    ew, ep, reps, n_null = args
    P = LR.prereg()
    d, L, R = load()
    d, L, R = subset_2021(d, L, R)
    rho = band_ratio(d)
    sim = Sim(d, R, rho)
    O, _, _ = LR.onehot(L)
    sp = LR.Split(d["fold"].to_numpy(), list(range(LR.NFOLD)), LR.NFOLD)
    _, pay = LR.group_rates(d, np.ones(len(d), bool))           # 複勝の払戻の尺度（群の平均。合成の払戻に使う）
    F = pay[d["e3g"].to_numpy()]
    emin = (ew, ep)
    # 効きなしの期待値（超過の勝ち数 S の基準）
    p0w = sim.win_prob(None)
    rng0 = np.random.default_rng(99)
    acc = np.zeros(len(d))
    for _ in range(200):
        acc += sim.draw(None, rng0)[1]
    p0t = acc / 200
    # 較正: 効きなしの合成 n_null 回の τ² の平均（fold ごと）と、そのときの買う点数
    cal, null_bets = {k: [] for k in sp.holds}, []
    for i in range(n_null):
        w, t = sim.draw(None, np.random.default_rng(500 + i))
        dd = _synth_frame(d, w, t, F)
        r0 = LR.run_split(dd, L, P, sp, emin=emin)
        for k in sp.holds:
            sh = r0["shrink"][k]
            cal[k].append((sh["tau2_cell_w"], sh["tau2_cell_p"], sh["tau2_course_p"]))
    calib = {k: {"cell_w": float(np.mean([x[0] for x in v])), "cell_p": float(np.mean([x[1] for x in v])),
                 "course_p": float(np.mean([x[2] for x in v]))} for k, v in cal.items()}
    for i in range(n_null):
        w, t = sim.draw(None, np.random.default_rng(700 + i))
        r0 = LR.run_split(_synth_frame(d, w, t, F), L, P, sp, calib=calib, emin=emin)
        null_bets.append((int(r0["bet_t"].sum()), int(r0["bet_f"].sum())))
    # 効きありの合成 reps 回
    out = []
    for i in range(reps):
        rng = np.random.default_rng(1000 + i)
        mult, picked = effect_mult(d, O, rng)
        w, t = sim.draw(mult, rng)
        r1 = LR.run_split(_synth_frame(d, w, t, F), L, P, sp, calib=calib, emin=emin)
        bt, bf = r1["bet_t"], r1["bet_f"]
        sh0 = r1["shrink"][0]
        out.append({"S_w": float((w[bt] - p0w[bt]).sum()), "S_p": float((t[bf] - p0t[bf]).sum()),
                    "bets_t": int(bt.sum()), "bets_f": int(bf.sum()),
                    "purity_t": float((mult[bt] > 1).mean()) if bt.any() else None, "purity_f": float((mult[bf] > 1).mean()) if bf.any() else None,
                    "share_effect_rows": float((mult > 1).mean()), "picked_rows_median": float(np.median([x["rows"] for x in picked])),
                    "picked_E_median": float(np.median([x["E"] for x in picked])), "picked_E_q90": float(np.quantile([x["E"] for x in picked], 0.9)),
                    "picked_courses": len(picked),
                    "n_courses_w": sh0["n_courses_w"], "n_courses_p": sh0["n_courses_p"], "n_cells_w": sh0["n_cells_w"], "n_cells_p": sh0["n_cells_p"]})
    E = pd.DataFrame(out)
    # 覆う割合（着順なし）: 2016〜21 の全行で、下限に届く深さ 1 の条件のどれかに当たる穴馬の割合
    rate0, _ = LR.group_rates(d.assign(top3=p0t), np.ones(len(d), bool))
    e = rate0[d["e3g"].to_numpy()]
    q = d["q"].to_numpy(float)
    units = d["unit"].to_numpy()
    cov_w, cov_p, cour_w, cour_p = np.zeros(len(d), bool), np.zeros(len(d), bool), 0, 0
    for u in np.unique(units):
        r = np.flatnonzero(units == u)
        o = O[r] > 0
        cw = np.flatnonzero(o.T.astype(float) @ q[r] >= ew * 0.8)        # 探す側は全体の 4/5
        cp = np.flatnonzero(o.T.astype(float) @ e[r] >= ep * 0.8)
        cour_w += len(cw) > 0
        cour_p += len(cp) > 0
        if len(cw):
            cov_w[r[o[:, cw].any(1)]] = True
        if len(cp):
            cov_p[r[o[:, cp].any(1)]] = True
    res = {"e_min_win": ew, "e3_min_place": ep, "reps": reps, "calib": calib, "null_bets": null_bets,
           "mean": E.mean(numeric_only=True).to_dict(), "se": (E.std(numeric_only=True) / np.sqrt(len(E))).to_dict(),
           "count_no_outcome": {"courses_w": int(cour_w), "courses_p": int(cour_p), "cover_w": float(cov_w.mean()), "cover_p": float(cov_p.mean())}}
    print(json.dumps({k: v for k, v in res.items() if k != "calib"}, ensure_ascii=False), flush=True)
    return res


def cmd_simA(reps: int, n_null: int, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    OUTS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_simA_one, [(ew, ep, reps, n_null) for ew, ep in CANDS_A]))
    out = {"cands": res, "rule": "券種ごとに、条件が残るコース ≥ 47 の候補のうち S の期待値が最大の下限。差が S の SE 以内なら大きい下限"}
    for kind, key, ck in (("win", "S_w", "n_courses_w"), ("place", "S_p", "n_courses_p")):
        ok = [r for r in res if r["mean"][ck] >= 47]
        if not ok:
            out[kind] = None
            continue
        best = max(ok, key=lambda r: r["mean"][key])
        se = best["se"][key] if best["se"][key] == best["se"][key] else 0.0      # reps=1 のときは SE が NaN
        near = [r for r in ok if best["mean"][key] - r["mean"][key] <= se] or [best]
        pick = max(near, key=lambda r: r["e_min_win" if kind == "win" else "e3_min_place"])
        out[kind] = pick["e_min_win" if kind == "win" else "e3_min_place"]
    (OUTS / "simA.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("pick", out["win"], out["place"], flush=True)


def _power1_one(args) -> list:
    ratio, seeds, emin = args
    d, L, R = load()
    d, L, R = subset_2021(d, L, R)
    sim = Sim(d, R, band_ratio(d))
    O, _, _ = LR.onehot(L)
    rows = np.arange(len(d))
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        mult = effect_mult(d, O, rng, ratio)[0] if ratio > 1 else None
        w, t = sim.draw(mult, rng)
        tt = tau_on(d, O, rows, w, t, emin)
        out.append({"ratio": ratio, "seed": sd, "tau2_cell_w": tt["tau2_cell_w"], "tau2_cell_p": tt["tau2_cell_p"]})
    return out


def cmd_power1(n_null: int, n_eff: int, workers: int) -> None:
    """段 1 の検出力（1 号の決定 D1 改訂 R3-3）。2016〜21・下限は prereg の値。効きなしの合成 n_null 回の τ² の 97.5% 点を線に、
    効き ×1.3・×1.15（コースごとに 1 条件、5〜50% を覆う）を n_eff 回ずつ流し、線を超える確率を券種ごとに出す。"""
    from concurrent.futures import ProcessPoolExecutor
    P = LR.prereg()
    emin = (P["search"]["e_min_win"], P["search"]["e3_min_place"])
    jobs = []
    for ratio, n, base in ((1.0, n_null, 30_000), (1.3, n_eff, 40_000), (1.15, n_eff, 50_000)):
        seeds = list(range(base, base + n))
        for i in range(workers):
            if seeds[i::workers]:
                jobs.append((ratio, seeds[i::workers], emin))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_power1_one, jobs) for r in part]
    T = pd.DataFrame(rows)
    out = {"emin": list(emin), "n_null": n_null, "n_eff": n_eff}
    for kind, col in (("win", "tau2_cell_w"), ("place", "tau2_cell_p")):
        null = T.loc[T.ratio == 1.0, col].to_numpy()
        line = float(np.quantile(null, 0.975))
        res = {"line_q975": line, "null_mean": float(null.mean()), "null_sd": float(null.std())}
        for r in (1.3, 1.15):
            v = T.loc[T.ratio == r, col].to_numpy()
            res[f"power_{r}"] = float((v > line).mean())
            res[f"mean_{r}"] = float(v.mean())
        res["run_stage1"] = res["power_1.3"] >= 0.5
        out[kind] = res
    (OUTS / "power1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    T.to_parquet(OUTS / "power1_rows.parquet", index=False)
    print(json.dumps(out, ensure_ascii=False), flush=True)


def effect_mult_big(d: pd.DataFrame, O: np.ndarray, rng: np.random.Generator, ratio: float, e_row: np.ndarray, emin: float) -> np.ndarray:
    """記録用の追加の模擬（レビューの指摘）: コースごとに、下限を満たす大きさ（2016〜21 の全体で見込みの数 ≥ emin、コースの 50% 以下）の
    深さ 1 の条件を 1 つ選び、その行の比を ratio 倍にする。e_row は行の見込み（単勝は q、複勝は効きなしの 3 着内の確率）。"""
    units = d["unit"].to_numpy()
    mult = np.ones(len(d))
    for u in np.unique(units):
        r = np.flatnonzero(units == u)
        o = O[r] > 0
        E = o.T.astype(float) @ e_row[r]
        cand = np.flatnonzero((E >= emin) & (o.mean(0) <= 0.50))
        if not len(cand):
            continue
        c = cand[int(rng.integers(len(cand)))]
        mult[r[o[:, c]]] *= ratio
    return mult


def _power2_one(args) -> list:
    kind, ratio, seeds, emin = args
    d, L, R = load()
    d, L, R = subset_2021(d, L, R)
    sim = Sim(d, R, band_ratio(d))
    O, _, _ = LR.onehot(L)
    rows = np.arange(len(d))
    p0w = sim.win_prob(None)
    rng0 = np.random.default_rng(99)
    acc = np.zeros(len(d))
    for _ in range(200):
        acc += sim.draw(None, rng0)[1]
    p0t = acc / 200
    e_row, em = (d["q"].to_numpy(float), emin[0]) if kind == "win" else (p0t, emin[1])
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        mult = effect_mult_big(d, O, rng, ratio, e_row, em)
        w, t = sim.draw(mult, rng)
        tt = tau_on(d, O, rows, w, t, emin)
        eff = mult > 1
        # 植えた行の実際の比（単: 勝つ確率の比、複: 3 着内の確率の比。複は 100 回の引き直しの平均）
        if kind == "win":
            real = float(sim.win_prob(mult)[eff].sum() / p0w[eff].sum()) if eff.any() else None
        else:
            a3 = np.zeros(len(d))
            for _ in range(100):
                a3 += sim.draw(mult, rng)[1]
            real = float((a3 / 100)[eff].sum() / p0t[eff].sum()) if eff.any() else None
        out.append({"kind": kind, "ratio": ratio, "seed": sd, "tau2": tt["tau2_cell_w" if kind == "win" else "tau2_cell_p"],
                    "share_rows": float(eff.mean()), "real_ratio": real})
    return out


def cmd_power2(n_eff: int, workers: int) -> None:
    """記録用の追加の模擬。下限を満たす大きさの条件に ×1.3・×1.5・×2 を植え、段 1 の線（power1 の効きなし 200 回の 97.5% 点）を超える割合。"""
    from concurrent.futures import ProcessPoolExecutor
    P = LR.prereg()
    emin = (P["search"]["e_min_win"], P["search"]["e3_min_place"])
    p1 = json.loads((OUTS / "power1.json").read_text(encoding="utf-8"))
    jobs = []
    for kind in ("win", "place"):
        for k, ratio in enumerate((1.3, 1.5, 2.0)):
            seeds = list(range(60_000 + 1000 * k + (0 if kind == "win" else 500), 60_000 + 1000 * k + (0 if kind == "win" else 500) + n_eff))
            for i in range(2):
                jobs.append((kind, ratio, seeds[i::2], emin))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_power2_one, jobs) for r in part]
    T = pd.DataFrame(rows)
    out = {"emin": list(emin), "n_eff": n_eff, "note": "記録用（閉じた判断は変えない）。線は power1 の効きなし 200 回の 97.5% 点"}
    for kind in ("win", "place"):
        line = p1[kind]["line_q975"]
        res = {"line_q975": line}
        for ratio in (1.3, 1.5, 2.0):
            g = T[(T.kind == kind) & (T.ratio == ratio)]
            res[str(ratio)] = {"power": float((g["tau2"] > line).mean()), "tau2_mean": float(g["tau2"].mean()),
                               "share_rows": float(g["share_rows"].mean()), "real_ratio": float(g["real_ratio"].dropna().mean())}
        out[kind] = res
    (OUTS / "power2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage0", "simA", "simB", "power1", "power2"])
    ap.add_argument("--neff", type=int, default=40)
    ap.add_argument("--ew", type=float, default=8.0)
    ap.add_argument("--ep", type=float, default=20.0)
    ap.add_argument("--null", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    if LR._stop_window():
        raise SystemExit("開催日の 07:30〜17:00 は重い計算を止める")
    if a.cmd == "stage0":
        cmd_stage0(a.reps, (a.ew, a.ep))
    elif a.cmd == "power2":
        cmd_power2(a.neff, a.workers)
    elif a.cmd == "power1":
        cmd_power1(a.null, a.neff, a.workers)
    elif a.cmd == "simA":
        cmd_simA(a.reps, a.null, a.workers)
    elif a.cmd == "simB":
        cmd_simB(a.reps)


if __name__ == "__main__":
    main()
