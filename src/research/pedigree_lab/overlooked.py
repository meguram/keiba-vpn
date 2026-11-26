"""注目されていない馬を血統で見抜けるか・血統 1 番手を人気に関わらず買う（log.md「注目されていない馬を血統で見抜けるか」。開発の行・時間の順・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.overlooked

- 予測: oof_v4_pure（第 4 版）と oof_v4_compact（87 列）の時間の順の p_win。行は 2020〜24 年の開発の行で、全頭に両方の予測があるレース。
- 人気は確定と T-60 の再現（pre_odds.simulate・5 回それぞれで順位と q を作り、物差しを回ごとに出して平均）。払戻は payoff の確定の値。
- 区間は開催で束ねたブートストラップ 2,000 回。T-60 の版は 5 回の値の平均を点とし、回ごとのブートストラップの平均で区間を作る。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck
from src.research.pedigree_lab import data, pre_odds
from src.research.pedigree_lab.pop_compare import payouts

N_TESTS = 40          # 人気の側 3 × 2 つの q ＋ 情報の側 6 × 2 つの q を 2 つのモデルで 36、ダートの牝系の 4（critic）
YEARS_FULL = 5 * 0.85
SKIPPED: list[int] = []
B_ROI: list = []


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "overlooked",
                                   "注目されていない馬と血統 1 番手の成績（的中・A/E・回収率の記述・合否にしない）"))
    cols = ["race_id", "horse_id", "date", "split", "win", "top3", "finish_position", "odds", "horse_number", "n_run", "surf",
            "debut", "career", "first_surf", "layoff", "dchg", "sire_all_n"]
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=cols)
    # ダートの牝系 5 代（凍結の作り）
    F = ap_.base_frame()
    A, keys = ap_.group_a(F, "dirt")
    L = ap_.maternal_long(F, drop_common=False)
    F["mat5"], _ = ck.mat_cols(F, A, keys, L)
    X = X.merge(F[["race_id", "horse_id", "mat5"]], on=["race_id", "horse_id"], how="left")
    P4 = pd.read_parquet(data.OUT / "oof_v4_pure.parquet")[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_v4"})
    PC = pd.read_parquet(data.OUT / "oof_v4_compact.parquet")[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_c"})
    S = X[X["split"].isin([0, 1]) & (X["odds"] > 0)].merge(P4, on=["race_id", "horse_id"]).merge(PC, on=["race_id", "horse_id"])
    S = S[S.groupby("race_id")["horse_id"].transform("size") == S["n_run"]].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    S = S[S["date"].dt.year >= 2019].reset_index(drop=True)
    rid = S["race_id"].to_numpy()
    yr = S["date"].dt.year.to_numpy()
    inv = 1 / S["odds"].to_numpy(float)
    q = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    rk_f = S.groupby("race_id")["odds"].rank(method="first").to_numpy()
    pool = pre_odds.load_drift("t60")
    sims = pre_odds.simulate(rid, S["odds"].to_numpy(float), rk_f, pool, n=5)
    views = [("final", rk_f, q, S["odds"].to_numpy(float))]
    for o in sims:
        qq = (1 / o) / pd.Series(1 / o).groupby(rid).transform("sum").to_numpy()
        views.append(("t60", pd.Series(o).groupby(rid).rank(method="first").to_numpy(), qq, o))
    wp, pp = payouts()
    S["win_pay"] = [wp.get(r, {}).get(int(h), 0.0) for r, h in zip(S["race_id"], S["horse_number"])]
    S["place_pay"] = [pp.get(r, {}).get(int(h), 0.0) for r, h in zip(S["race_id"], S["horse_number"])]
    w, t3 = S["win"].to_numpy(float), S["top3"].to_numpy(float)
    t2 = (S["finish_position"] <= 2).to_numpy(float)
    winp, plp = S["win_pay"].to_numpy(), S["place_pay"].to_numpy()
    mt = S["race_id"].str[:8].to_numpy()
    mu, minv = np.unique(mt, return_inverse=True)
    rng = np.random.default_rng(21)
    Wb = np.stack([np.bincount(rng.integers(0, len(mu), len(mu)), minlength=len(mu)) for _ in range(2000)]).astype(float)
    zb = norm.ppf(1 - 0.025 / N_TESTS)
    late = yr >= 2020

    def msum(v, mk):
        return np.bincount(minv[mk], weights=v[mk], minlength=len(mu))

    def mean_ci(v, mk):
        n_, s_ = msum(np.ones(len(v)), mk), msum(v, mk)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (Wb @ s_) / (Wb @ n_)                                  # 空の回は NaN（0 にしない。critic）
        return (float(s_.sum() / n_.sum()) if n_.sum() else np.nan), b

    def wmatch(v, band, pick, rk):
        """帯の馬を全部買う回収率を、1 番手の人気の順の件数で重み付けた値（点と回ごと）。"""
        rr = np.clip(rk.astype(int), 1, 18)
        nb = 19
        sm = np.zeros((len(mu), nb)); cn = np.zeros((len(mu), nb)); pk = np.zeros((len(mu), nb))
        np.add.at(sm, (minv[band], rr[band]), v[band]); np.add.at(cn, (minv[band], rr[band]), 1)
        np.add.at(pk, (minv[pick], rr[pick]), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m0 = sm.sum(0) / cn.sum(0)
            pt = np.nansum(pk.sum(0) * m0) / pk.sum() if pk.sum() else np.nan
            mb = (Wb @ sm) / (Wb @ cn)
            bb = np.nansum((Wb @ pk) * mb, axis=1) / (Wb @ pk).sum(1)
        return float(pt), bb

    res = {"rows": int(late.sum()), "races": int(pd.Series(rid[late]).nunique()), "n_tests": N_TESTS, "z_bonf": round(float(zb), 3)}
    # ---- 調べ 1: 血統 1 番手を買う ----
    part1 = {}
    for mname in ("p_v4", "p_c"):
        top1 = (pd.Series(S[mname].to_numpy()).groupby(rid).rank(ascending=False, method="first").to_numpy() == 1) & late
        for vname in ("final", "t60"):
            vs = [v for v in views if v[0] == vname]
            for bnm, lo, hi in (("全部", 1, 99), ("1〜3 番人気", 1, 3), ("4 番人気以降", 4, 99), ("4〜6 番人気", 4, 6), ("7〜9 番人気", 7, 9), ("10 番人気以下", 10, 99)):
                for snm, smk in (("全体", np.ones(len(S), bool)), ("芝", (S["surf"] == "T").to_numpy()), ("ダート", (S["surf"] == "D").to_numpy())):
                    acc = {k: [] for k in ("n", "win", "ren", "top3", "wroi", "proi", "wbase", "pbase")}
                    bw_, bp_ = [], []
                    for _, rk, qv, _o in vs:
                        mk = top1 & (rk >= lo) & (rk <= hi) & smk
                        band = late & (rk >= lo) & (rk <= hi) & smk          # 同じ人気の帯の馬を全部買う（下で 1 番手の人気の順の重みに直す）
                        acc["n"].append(mk.sum())
                        for k, v in (("win", w), ("ren", t2), ("top3", t3)):
                            acc[k].append(v[mk].mean() if mk.sum() else np.nan)
                        r1, b1 = mean_ci(winp, mk); r2, b2 = mean_ci(plp, mk)
                        # 基準線: 帯の各人気の順の回収率を、1 番手の人気の順ごとの件数で重み付ける（帯の中の人気の寄りを消す。critic）
                        r3, b3 = wmatch(winp, band, mk, rk); r4, b4 = wmatch(plp, band, mk, rk)
                        acc["wroi"].append(r1); acc["proi"].append(r2); acc["wbase"].append(r3); acc["pbase"].append(r4)
                        bw_.append(b1 - b3); bp_.append(b2 - b4); acc.setdefault("bw1", []).append(b1); acc.setdefault("bp1", []).append(b2)
                    bw_, bp_ = np.mean(bw_, 0), np.mean(bp_, 0)
                    dw = float(np.mean(acc["wroi"]) - np.mean(acc["wbase"]))
                    dp = float(np.mean(acc["proi"]) - np.mean(acc["pbase"]))
                    key = f"{mname}|{vname}|{bnm}|{snm}"
                    part1[key] = {"per_year": round(float(np.mean(acc["n"])) / YEARS_FULL, 1), "win": round(float(np.nanmean(acc["win"])), 4),
                                  "ren": round(float(np.nanmean(acc["ren"])), 4), "top3": round(float(np.nanmean(acc["top3"])), 4),
                                  "win_roi": round(float(np.mean(acc["wroi"])), 1), "place_roi": round(float(np.mean(acc["proi"])), 1),
                                  "band_win_roi": round(float(np.mean(acc["wbase"])), 1), "band_place_roi": round(float(np.mean(acc["pbase"])), 1),
                                  "win_roi_ci": np.nanpercentile(np.mean(acc["bw1"], 0), [2.5, 97.5]).round(1).tolist(),
                                  "place_roi_ci": np.nanpercentile(np.mean(acc["bp1"], 0), [2.5, 97.5]).round(1).tolist(),
                                  "win_minus_band": [round(dw, 1), np.nanpercentile(bw_, [2.5, 97.5]).round(1).tolist()],
                                  "place_minus_band": [round(dp, 1), np.nanpercentile(bp_, [2.5, 97.5]).round(1).tolist()]}
                    print("p1", key, part1[key], flush=True)
    res["part1"] = part1
    # ---- 調べ 2: 注目されていない馬 ----
    info = {"新馬": (S["debut"] == "Y").to_numpy(), "出走 1〜2 走": S["career"].isin([1, 2]).to_numpy(),
            "初ダート・初芝": (S["first_surf"] == "Y").to_numpy(), "距離の大きな変化": S["dchg"].isin(["U", "D"]).to_numpy(),
            "休み明け": (S["layoff"] == "Y").to_numpy(), "新種牡馬の産駒": (np.expm1(S["sire_all_n"].to_numpy()) < 100)}
    part2 = {}

    def deciles(score, pop):
        """その年より前の母集団の行の分位で 10 等分（2020 年は 2019 年の行）。上 10% と下 10% の印。"""
        top, bot = np.zeros(len(S), bool), np.zeros(len(S), bool)
        for y in range(2020, 2025):
            prior = pop & (yr < y) & (yr >= 2019)
            if prior.sum() < 200:
                SKIPPED.append(int(y))                                    # 飛ばした年を残す（critic）
                continue
            hi, lo = np.quantile(score[prior], [0.9, 0.1])
            cur = pop & (yr == y)
            top |= cur & (score >= hi)
            bot |= cur & (score <= lo)
        return top, bot

    def ae_block(pop_mk_list, score):
        """pop_mk_list: [(view, 母集団の印)]。上位・下位の A/E・比・回収率。"""
        out = {k: [] for k in ("ae_top", "ae_bot", "ae_all", "ratio", "win_top", "top3_top", "wroi_top", "proi_top", "wroi_all", "n_top", "n_all")}
        b_ratio, b_aetop = [], []
        for (vname, rk, qv, _o), pop in pop_mk_list:
            top, bot = deciles(score, pop)
            allm = pop & late
            ae = {}
            for nm, mk in (("top", top), ("bot", bot), ("all", allm)):
                sw, sq = msum(w, mk), msum(qv, mk)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ae[nm] = (sw.sum() / sq.sum() if sq.sum() > 0 else np.nan, (Wb @ sw) / (Wb @ sq))
            out["ae_top"].append(ae["top"][0]); out["ae_bot"].append(ae["bot"][0]); out["ae_all"].append(ae["all"][0])
            with np.errstate(invalid="ignore", divide="ignore"):
                out["ratio"].append(ae["top"][0] / ae["bot"][0] if ae["bot"][0] and ae["bot"][0] > 0 else np.nan)
                b_ratio.append(np.where(ae["bot"][1] > 0, ae["top"][1] / ae["bot"][1], np.nan)); b_aetop.append(ae["top"][1])
            out["win_top"].append(w[top].mean() if top.any() else np.nan); out["top3_top"].append(t3[top].mean() if top.any() else np.nan)
            r_, bwr = mean_ci(winp, top); r2_, bpr = mean_ci(plp, top)
            out["wroi_top"].append(r_); out["proi_top"].append(r2_); out["wroi_all"].append(winp[allm].mean() if allm.any() else np.nan)
            B_ROI.append((bwr, bpr))
            out["n_top"].append(top.sum()); out["n_all"].append(allm.sum())
        b_ratio, b_aetop = np.nanmean(b_ratio, 0), np.nanmean(b_aetop, 0)
        r = {k: (round(float(np.nanmean(v)), 4) if np.isfinite(np.nanmean(v)) else None) for k, v in out.items()}
        r["wroi_top_ci"] = np.nanpercentile(np.nanmean([b[0] for b in B_ROI[-len(pop_mk_list):]], 0), [2.5, 97.5]).round(1).tolist()
        r["proi_top_ci"] = np.nanpercentile(np.nanmean([b[1] for b in B_ROI[-len(pop_mk_list):]], 0), [2.5, 97.5]).round(1).tolist()
        r["n_top_per_year"] = round(r.pop("n_top") / YEARS_FULL, 1)
        r["n_all_per_year"] = round(r.pop("n_all") / YEARS_FULL, 1)
        for nm, b, pt in (("ratio", b_ratio, r["ratio"]), ("ae_top", b_aetop, r["ae_top"])):
            if pt is None or not np.isfinite(np.nanmean(b)):
                r[f"{nm}_ci95"], r[f"{nm}_ci_bonf"] = None, None
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                se = float(np.nanstd(np.log(b[b > 0])))
            r[f"{nm}_ci95"] = np.nanpercentile(b, [2.5, 97.5]).round(3).tolist()
            r[f"{nm}_ci_bonf"] = [round(float(pt * np.exp(-zb * se)), 3), round(float(pt * np.exp(zb * se)), 3)]
        return r
    for mname in ("p_v4", "p_c"):
        score = S[mname].to_numpy()
        for vname in ("final", "t60"):
            vs = [v for v in views if v[0] == vname]
            line = 1.27 if vname == "final" else 1.27 / 0.86
            for pnm, f in (("6 番人気以下", lambda v: v[1] >= 6), ("単勝 10〜50 倍", lambda v: (v[3] >= 10) & (v[3] <= 50)), ("単勝 50 倍超", lambda v: v[3] > 50)):
                key = f"{mname}|{vname}|人気|{pnm}"
                part2[key] = ae_block([(v, f(v)) for v in vs], score) | {"line": round(line, 3)}
                print("p2", key, part2[key], flush=True)
        for inm, mk in info.items():
            for vname in ("final", "t60"):
                vs = [v for v in views if v[0] == vname]
                key = f"{mname}|{vname}|情報|{inm}"
                part2[key] = ae_block([(v, mk) for v in vs], score) | {"line": round(1.27 if vname == "final" else 1.27 / 0.86, 3)}
                print("p2", key, part2[key], flush=True)
    # ダートの牝系 5 代の点（ダートの行・記述）
    dirt = (S["surf"] == "D").to_numpy()
    zc = S["mat5"].fillna(0).to_numpy() - pd.Series(S["mat5"].fillna(0).to_numpy()).groupby(rid).transform("mean").to_numpy()
    for vname in ("final", "t60"):
        vs = [v for v in views if v[0] == vname]
        for pnm, f in (("6 番人気以下", lambda v: v[1] >= 6), ("単勝 10〜50 倍", lambda v: (v[3] >= 10) & (v[3] <= 50)), ("単勝 50 倍超", lambda v: v[3] > 50)):
            key = f"mat5|{vname}|人気|{pnm}|ダート"
            part2[key] = ae_block([(v, f(v) & dirt) for v in vs], zc) | {"line": round(1.27 if vname == "final" else 1.27 / 0.86, 3)}
            print("p2", key, part2[key], flush=True)
        for inm, mk in info.items():
            key = f"mat5|{vname}|情報|{inm}|ダート"
            part2[key] = ae_block([(v, mk & dirt) for v in vs], zc) | {"line": round(1.27 if vname == "final" else 1.27 / 0.86, 3)}
            print("p2", key, part2[key], flush=True)
    res["part2"] = part2
    res["skipped_years"] = sorted(set(SKIPPED))
    path = data.OUT / "overlooked.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
