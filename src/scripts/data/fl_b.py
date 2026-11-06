"""最終層 B: 人気薄発掘派。人気に引きずられない確率を作り、人気薄（単勝 7 倍以上）の中から選ぶ。

手段（全部試して LOYO で比べる）:
 B0 残差: log p_model − log p_market を人気（対数オッズ・人気順位）で直交化した残差
 B1 逆確率の重み: 勝ち馬の市場確率の逆数 ^γ（γ=0, 0.5, 1）で学習（人気薄が勝ったレースを重くする）
 B2 人気との相関の罰則 λ: 得点と対数市場確率の相関² に λ を掛けて引く
 B3 Benter 型（市場の対数確率を offset、人気薄側だけに乗せる）
 B4 学習を 7 倍以上の馬だけに制限
python -m src.scripts.data.fl_b
"""

from __future__ import annotations

import json

import numpy as np

from src.scripts.data import fl_core as C

# 人気薄側だけに効くモデルの差（B3）
C.FEATS["gapL"] = lambda D: np.where(D.win_odds >= C.LONG_ODDS, D.lp - D.lq, 0.0)


class Resid:
    """gap = lp − lq を [1, lo, lo², prk, fav] に回帰し、その残差を得点にする（人気と無相関な上乗せ）。"""

    def fit(self, D, idx):
        m = D.mask[idx]
        X = np.stack([np.ones(m.sum()), C.FEATS["lo"](D)[idx][m], C.FEATS["lo2"](D)[idx][m], C.FEATS["prk"](D)[idx][m], C.FEATS["fav"](D)[idx][m]], 1)
        y = (D.lp - D.lq)[idx][m]
        self.theta = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def score(self, D, idx):
        X = np.stack([np.ones(D.lp[idx].shape), C.FEATS["lo"](D)[idx], C.FEATS["lo2"](D)[idx], C.FEATS["prk"](D)[idx], C.FEATS["fav"](D)[idx]], -1)
        r = (D.lp - D.lq)[idx] - X @ self.theta
        return np.where(D.mask[idx], r, -np.inf)


def soft(s):
    s = np.where(np.isnan(s), -np.inf, s)
    mx = np.max(np.where(np.isfinite(s), s, -1e18), 1, keepdims=True)
    e = np.exp(np.where(np.isfinite(s), s - mx, -np.inf))
    z = e.sum(1, keepdims=True)
    return np.where(z > 0, e / np.where(z > 0, z, 1), 0.0)


def pick_stats(D, score, cand, ridx, sel=None):
    """k=1 で選んだ人気薄の、的中数 と 市場が見込む的中数（Σ市場確率）。比 > 1 なら市場より当たっている。"""
    w = sq = n = 0
    w10 = sq10 = n10 = 0
    for r in ridx:
        if sel is not None and not sel[r]:
            continue
        sc = np.where(cand[r], score[r], -np.inf)[: D.n[r]]
        if not np.isfinite(sc).any():
            continue
        i = int(np.argmax(sc))
        n += 1; w += int(D.win[r, i]); sq += D.q[r, i]
        if D.win_odds[r, i] >= 10:
            n10 += 1; w10 += int(D.win[r, i]); sq10 += D.q[r, i]
    return {"picks": n, "wins": w, "market_expected": float(sq), "ratio": float(w / sq) if sq else None,
            "picks_ge10": n10, "wins_ge10": w10, "market_expected_ge10": float(sq10), "ratio_ge10": float(w10 / sq10) if sq10 else None}


def main() -> None:
    D = C.load()
    ev = C.races_of(D, C.OOF_YEARS)
    pop_score = np.where(D.mask, -D.pop, -np.inf)
    cand = D.mask & (D.win_odds >= C.LONG_ODDS)
    res: dict = {"prob": {}, "theta": {}, "pick": {}}
    models = {
        "B0_resid": lambda: Resid(),
        "B1_g0": lambda: C.CLogit(["lp", "lq"], gamma=0.0),
        "B1_g0.5": lambda: C.CLogit(["lp", "lq"], gamma=0.5),
        "B1_g1": lambda: C.CLogit(["lp", "lq"], gamma=1.0),
        "B2_lam1": lambda: C.CLogit(["lp", "l2", "l3", "lq"], lam=1.0),
        "B2_lam10": lambda: C.CLogit(["lp", "l2", "l3", "lq"], lam=10.0),
        "B2_lam10_g0.5": lambda: C.CLogit(["lp", "l2", "l3", "lq"], lam=10.0, gamma=0.5),
        "B3_offset": lambda: C.CLogit(["gapL"], offset=lambda D, idx: D.lq[idx]),
        "B4_longonly": lambda: C.CLogit(["lp", "l2", "l3", "lq", "lo"], only_long=True),
    }
    scores = {}
    for nm, mk in models.items():
        s, th = C.loyo_scores(D, mk)
        s = np.where(np.isnan(s), -np.inf, s)
        scores[nm] = s
        res["theta"][nm] = th
        if nm == "B4_longonly":
            ql = np.where(cand, D.q, 0.0).sum(1, keepdims=True)
            sl = np.where(cand, s, -np.inf)
            p = soft(sl) * ql
        elif nm == "B0_resid":
            p = None
        else:
            p = soft(s)
        if p is not None:
            res["prob"][nm] = C.prob_metrics(D, p)
            v = res["prob"][nm]
            print(f"{nm:16s} win_ll {v['win_ll']:.4f} | 10倍以上 ll {v['long10_ll']:.4f} Σp {v['long10_sum_p']:.1f}/実 {v['long10_wins']:.0f} θ {th}", flush=True)
            scores[nm + "_p"] = p
        else:
            print(f"{nm:16s} (確率なし) θ {th}", flush=True)

    out = []
    uni = {bt: C.gen_uniform(D, bt, cand, 1.0, ridx=ev) for bt in ("tansho", "fukusho")}
    for nm in models:
        sc = scores[nm]
        # 人気薄の中から最上位 1 頭（毎レース）と、得点が上位 30% のレースだけ
        top_sc = np.where(cand, sc, -np.inf).max(1)
        thr = np.quantile(top_sc[ev][np.isfinite(top_sc[ev])], 0.70)
        for tag, sel in (("all", None), ("top30", top_sc >= thr)):
            res["pick"][f"{nm}/{tag}"] = pick_stats(D, sc, cand, ev, sel)
            for bt in ("tansho", "fukusho"):
                tk = C.gen_single(D, sc, bt, 1, sel=sel, cand=cand, ridx=ev)
                ct = C.gen_single(D, pop_score, bt, 1, sel=sel, ridx=ev)
                o = C.summarize(D, f"B/{nm}/{tag}/{bt}_long1", tk, ct, group="B", params={"model": nm, "sel": tag, "bt": bt})
                if tag == "all":
                    o2 = C.summarize(D, "tmp", tk, uni[bt], group="B", log=False, detail=False)
                    o["vs_uniform_long"] = {"ctrl_roi": o2.get("ctrl_roi"), "diff": o2.get("diff"), "diff_ci": o2.get("diff_ci")}
                out.append(o)
                ps = res["pick"][f"{nm}/{tag}"]
                print(C.fmt(o) + f" | 人気薄の的中/市場期待 {ps['wins']}/{ps['market_expected']:.1f} (≥10倍 {ps['wins_ge10']}/{ps['market_expected_ge10']:.1f})"
                      + (f" 対 一様買い差 {o['vs_uniform_long']['diff']:+.3f}" if "vs_uniform_long" in o else ""), flush=True)
    # B3: 較正後の確率 × 確定オッズ ≥ しきい値（人気薄のみ）
    p3 = scores["B3_offset_p"]
    for th in (1.0, 1.2):
        sel_c = cand & (p3 * D.win_odds >= th)
        sc = np.where(sel_c, p3 * D.win_odds, -np.inf)
        for bt in ("tansho",):
            tk = C.gen_single(D, sc, bt, 1, cand=sel_c, ridx=ev)
            ct = C.gen_single(D, pop_score, bt, 1, sel=sel_c.any(1), ridx=ev)
            o = C.summarize(D, f"B/B3_offset/ev{th}/{bt}_long1", tk, ct, group="B", params={"model": "B3_offset", "ev": th, "bt": bt})
            out.append(o); print(C.fmt(o), flush=True)
    res["policies"] = out
    (C.OUT / "stage_B.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(C.OUT / "scores_B.npz", **{k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in scores.items() if not k.endswith("_p")})
    C.save_trials("B")


if __name__ == "__main__":
    main()
