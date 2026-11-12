"""最終層 F: 合成。A〜E の方策を、会場・斉齢限定/その他・レース水準の舞台ごとに、全体の重みへ縮約して選び直す。

各方策は『1 レース 1 単位の予算』に正規化した回収倍率 m と、同じ予算で最も人気の券を買う対照 mc を持つ。
舞台ごとの選択 = 学習年での (Σ 差 + n0·全体の差) / (Σ 買った数 + n0) が最大の方策（全て負なら見送り）。
2 年で選び、残りの 1 年で評価する（入れ子）。
python -m src.scripts.data.fl_f
"""

from __future__ import annotations

import json

import numpy as np

from src.scripts.data import fl_c
from src.scripts.data import fl_core as C
from src.scripts.data.fl_b import soft
from src.scripts.data.fl_d import basket_sanren, gen_alloc

N0 = 300.0
POLICIES = ["A_nagashi5", "B0_long1", "B3_ev1.0", "C_axisB0_nag5", "D1_alloc", "D2_basket", "E1_ev1.0", "E1_top30"]


def make_policies(D, S, ridx, thr_top30=None):
    """S: A0, A1, B0, B3, D1, E1 のスコア行列 [R,N]。戻り値 {名前: (tk, ctrl)}, top30 のしきい値。"""
    pop = np.where(D.mask, -D.pop, -np.inf)
    cand = D.mask & (D.win_odds >= C.LONG_ODDS)
    q_alloc = np.where(D.mask, D.q, 0.0)
    out = {}
    out["A_nagashi5"] = (C.gen_bundle(D, S["A0"], "sanrenpuku", 5, "nagashi", ridx=ridx), C.gen_bundle(D, pop, "sanrenpuku", 5, "nagashi", ridx=ridx))
    out["B0_long1"] = (C.gen_single(D, S["B0"], "tansho", 1, cand=cand, ridx=ridx), C.gen_single(D, pop, "tansho", 1, ridx=ridx))
    for nm, key in (("B3_ev1.0", "B3"), ("E1_ev1.0", "E1")):
        p = soft(S[key])
        sel_c = cand & (p * D.win_odds >= 1.0)
        sc = np.where(sel_c, p * D.win_odds, -np.inf)
        out[nm] = (C.gen_single(D, sc, "tansho", 1, cand=sel_c, ridx=ridx), C.gen_single(D, pop, "tansho", 1, sel=sel_c.any(1), ridx=ridx))
    s_ax = fl_c.with_axis(S["A1"], S["B0"], cand)
    out["C_axisB0_nag5"] = (C.gen_bundle(D, s_ax, "sanrenpuku", 5, "nagashi", ridx=ridx), C.gen_bundle(D, pop, "sanrenpuku", 5, "nagashi", ridx=ridx))
    out["D1_alloc"] = (gen_alloc(D, soft(S["D1"]), "tansho", ridx), gen_alloc(D, q_alloc, "tansho", ridx))
    tk, used = basket_sanren(D, soft(S["A1"]), ridx, "edge", 0.0)
    ct, _ = basket_sanren(D, soft(S["A1"]), ridx, "pop", races=used)
    out["D2_basket"] = (tk, ct)
    top_sc = np.where(cand, S["E1"], -np.inf).max(1)
    if thr_top30 is None:
        v = top_sc[ridx]
        thr_top30 = float(np.quantile(v[np.isfinite(v)], 0.70))
    sel = top_sc >= thr_top30
    out["E1_top30"] = (C.gen_single(D, S["E1"], "tansho", 1, sel=sel, cand=cand, ridx=ridx), C.gen_single(D, pop, "tansho", 1, sel=sel, ridx=ridx))
    return out, thr_top30


def multiples(D, pol):
    """各方策の (買ったか b, 倍率 m, 対照倍率 mc) を [R] で返す。"""
    b, m, mc = {}, {}, {}
    for nm, (tk, ct) in pol.items():
        s, r = C._per_race(D, tk)
        cs, cr = C._per_race(D, ct)
        ok = (s > 0) & (cs > 0)
        b[nm] = ok
        m[nm] = np.where(ok, r / np.where(s > 0, s, 1), 0.0)
        mc[nm] = np.where(ok, cr / np.where(cs > 0, cs, 1), 0.0)
    return b, m, mc


def segments(D):
    rc = np.array([str(x) for x in D.race_class])
    gr = np.array([str(x) for x in D.grade])
    age = np.where(((np.char.find(rc, "２歳") >= 0) | (np.char.find(rc, "３歳") >= 0)) & (np.char.find(rc, "以上") < 0), "限定", "その他")
    low = (np.char.find(gr, "新馬") >= 0) | (np.char.find(gr, "未勝利") >= 0)
    mid = (np.char.find(gr, "1勝") >= 0) | (np.char.find(gr, "2勝") >= 0)
    level = np.where(low, "低", np.where(mid, "中", "高"))
    venue, surf = np.array([str(x) for x in D.venue]), np.array([str(x) for x in D.surface])
    return {"全体": np.array(["all"] * D.R), "会場": venue, "斉齢": age, "水準": level, "芝ダ": surf,
            "会場×芝ダ": np.char.add(np.char.add(venue, "|"), surf), "斉齢×水準": np.char.add(np.char.add(age, "|"), level)}


def choose(seg, tr, B, M, MC, n0=N0):
    """舞台ごとの方策（なければ None = 見送り）を学習レース tr から決める。"""
    g = {}
    for p in POLICIES:
        k = B[p][tr]
        g[p] = float((M[p][tr] - MC[p][tr])[k].mean()) if k.any() else 0.0
    best_global = max(POLICIES, key=lambda p: g[p])
    choice = {}
    for c in np.unique(seg[tr]):
        inc = tr[seg[tr] == c]
        sc = {}
        for p in POLICIES:
            k = B[p][inc]
            d = float((M[p][inc] - MC[p][inc])[k].sum())
            sc[p] = (d + n0 * g[p]) / (k.sum() + n0)
        pbest = max(sc, key=lambda p: sc[p])
        choice[c] = pbest if sc[pbest] > 0 else None
    return choice, best_global, g


def apply(seg, rows, choice, B, M, MC, default=None):
    """rows のレースに選んだ方策を当てはめ、(買った行, 倍率, 対照倍率) を返す。"""
    use = np.zeros(len(rows), bool)
    m = np.zeros(len(rows))
    mc = np.zeros(len(rows))
    for j, r in enumerate(rows):
        p = choice.get(seg[r], default)
        if p is not None and B[p][r]:
            use[j], m[j], mc[j] = True, M[p][r], MC[p][r]
    return use, m, mc


def stats(use, m, mc, B=1000):
    n = len(use)
    if use.sum() == 0:
        return {"empty": True}
    s, r, cr = use.astype(float), m * use, mc * use
    idx = C._boot_idx(n, B)
    S, R_, CR = s[idx].sum(1), r[idx].sum(1), cr[idx].sum(1)
    ok = S > 0
    d = R_[ok] / S[ok] - CR[ok] / S[ok]
    roi = r.sum() / s.sum()
    rb = np.sort(r[use])[::-1]
    return {"n_bought": int(use.sum()), "roi": float(roi), "roi_ci": [float(np.quantile(R_[ok] / S[ok], 0.025)), float(np.quantile(R_[ok] / S[ok], 0.975))],
            "ctrl_roi": float(cr.sum() / s.sum()), "diff": float(roi - cr.sum() / s.sum()), "diff_ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "roi_ex_top1": float((r.sum() - rb[:1].sum()) / s.sum()), "roi_ex_top3": float((r.sum() - rb[:3].sum()) / s.sum()),
            "race_hit_rate": float((r[use] > 0).mean())}


def main() -> None:
    D = C.load()
    ev = C.races_of(D, C.OOF_YEARS)
    sA0, sA1 = (np.nan_to_num(np.load(C.OUT / f"score_{k}.npy"), nan=-np.inf) for k in ("A0", "A1"))
    sB = np.load(C.OUT / "scores_B.npz")
    sD = np.load(C.OUT / "scores_D.npz")
    sE = np.load(C.OUT / "scores_E.npz")
    nn = lambda a: np.where(np.isnan(a), -np.inf, a)
    S = {"A0": sA0, "A1": sA1, "B0": nn(sB["B0_resid"]), "B3": nn(sB["B3_offset"]), "D1": nn(sD["D1w_f0.001(RL期待値)"]), "E1": nn(sE["E1"])}
    pol, thr = make_policies(D, S, ev)
    B, M, MC = multiples(D, pol)
    segs = segments(D)
    res: dict = {"thr_top30": thr, "single": {}, "compose": {}, "choices": {}}
    yr = D.year
    print("単独方策（各レース 1 単位に正規化、買ったレース上の平均）", flush=True)
    for p in POLICIES:
        k = B[p] & np.isin(yr, C.OOF_YEARS)
        u = k[ev]
        st = stats(u, M[p][ev], MC[p][ev])
        st["by_year"] = {int(y): float((M[p][k & (yr == y)] - MC[p][k & (yr == y)]).mean()) for y in C.OOF_YEARS}
        res["single"][p] = st
        print(f"  {p:16s} 買い {st['n_bought']:5d} 回収 {st['roi']:.3f} 対照 {st['ctrl_roi']:.3f} 差 {st['diff']:+.3f}[{st['diff_ci'][0]:+.3f},{st['diff_ci'][1]:+.3f}] 年別 " + " ".join(f"{y}:{v:+.3f}" for y, v in st["by_year"].items()), flush=True)

    print("合成（2 年で選び 1 年で評価、3 年連結）。『最良の単独』は同じく学習年で選んだ単独方策", flush=True)
    for sn, seg in segs.items():
        U, Mm, MCm, GU, GM, GMC, byy, byy_g, ch = [], [], [], [], [], [], {}, {}, {}
        for yv in C.OOF_YEARS:
            tr = C.races_of(D, [y for y in C.OOF_YEARS if y != yv])
            te = C.races_of(D, [yv])
            choice, bg, g = choose(seg, tr, B, M, MC)
            u, m, mc = apply(seg, te, choice, B, M, MC)
            U.append(u); Mm.append(m); MCm.append(mc)
            ug, mg, mcg = apply(seg, te, {}, B, M, MC, default=bg)
            GU.append(ug); GM.append(mg); GMC.append(mcg)
            byy[int(yv)] = float(((m - mc)[u]).mean()) if u.any() else float("nan")
            byy_g[int(yv)] = float(((mg - mcg)[ug]).mean()) if ug.any() else float("nan")
            ch[int(yv)] = {str(k): v for k, v in choice.items()}
        U, Mm, MCm = np.concatenate(U), np.concatenate(Mm), np.concatenate(MCm)
        GU, GM, GMC = np.concatenate(GU), np.concatenate(GM), np.concatenate(GMC)
        st, sg = stats(U, Mm, MCm), stats(GU, GM, GMC)
        st["by_year"], sg["by_year"] = byy, byy_g
        res["compose"][sn] = {"composed": st, "best_single": sg}
        res["choices"][sn] = ch
        name = f"F/{sn}"
        prof_c = np.concatenate([(Mm - 1.0) * U])
        C.TRIALS.append({"name": name, "group": "F", **st})
        print(f"  {sn:10s} 合成 買い {st['n_bought']:5d} 回収 {st['roi']:.3f}[{st['roi_ci'][0]:.3f},{st['roi_ci'][1]:.3f}] 対照 {st['ctrl_roi']:.3f} 差 {st['diff']:+.3f}[{st['diff_ci'][0]:+.3f},{st['diff_ci'][1]:+.3f}] 年別 "
              + " ".join(f"{y}:{v:+.3f}" for y, v in byy.items()) + f" 最大除き {st['roi_ex_top1']:.3f}/{st['roi_ex_top3']:.3f}"
              + f" || 最良単独 {sg['roi']:.3f} 差 {sg['diff']:+.3f}[{sg['diff_ci'][0]:+.3f},{sg['diff_ci'][1]:+.3f}] 年別 " + " ".join(f"{y}:{v:+.3f}" for y, v in byy_g.items()), flush=True)

    # 全年で選んだ最終の舞台別選択（テスト 2025 に使う）
    final = {}
    for sn, seg in segs.items():
        choice, bg, g = choose(seg, ev, B, M, MC)
        final[sn] = {"choice": {str(k): v for k, v in choice.items()}, "best_global": bg, "g": g}
    res["final"] = final
    print("全体の最良（全 3 年）:", {sn: final[sn]["best_global"] for sn in ("全体",)}, "g=", {k: round(v, 3) for k, v in final["全体"]["g"].items()}, flush=True)
    for sn in ("会場", "斉齢", "水準"):
        print(f"  最終選択 {sn}:", {k: v for k, v in final[sn]["choice"].items()}, flush=True)

    (C.OUT / "stage_F.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    C.save_trials("F")
    np.savez_compressed(C.OUT / "mult_F.npz", **{f"b_{p}": B[p] for p in POLICIES}, **{f"m_{p}": M[p] for p in POLICIES}, **{f"mc_{p}": MC[p] for p in POLICIES})


if __name__ == "__main__":
    main()
