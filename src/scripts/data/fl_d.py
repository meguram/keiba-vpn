"""最終層 D: 期待値の直接最適化派。

D1 微分可能な賭け金配分: レース内 softmax の配分 π を、実現した払戻での期待対数成長
    −mean log(1−f+f·Σπ_i·ret_i) + 人気馬への配分罰則 で学習（f=賭け比率。f→0 は『期待回収を直接最大化する方策』= 1 段の方策勾配の厳密期待値）。
D2 レース内『どれが当たっても予算の 5 倍』の制約: Σ1/odds_i ≤ 0.2 の範囲で、モデル確率×オッズが高い順に詰める（単勝は確定オッズ、3 連複は想定オッズ 0.75/市場ハーヴィル確率）。
python -m src.scripts.data.fl_d
"""

from __future__ import annotations

import json
from itertools import combinations, permutations

import numpy as np

from src.scripts.data import fl_core as C
from src.scripts.data.fl_b import soft

NAMES = ["lp", "lq", "lo", "lo2", "prk", "fav", "l3"]
NAMES_MKT = ["lq", "lo", "lo2", "prk", "fav"]


class Alloc:
    def __init__(self, names, f, lam=0.0, kind="win", l2=1e-3):
        self.names, self.f, self.lam, self.kind, self.l2 = names, f, lam, kind, l2

    def fit(self, D, idx):
        import torch

        X = C.feat_tensor(D, self.names, idx)
        m = D.mask[idx]
        self.mu, self.sd = X[m].mean(0), X[m].std(0) + 1e-9
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        mt = torch.tensor(m)
        ret = torch.tensor(np.where(m, (D.ret_win if self.kind == "win" else D.ret_place)[idx] / 100.0, 0.0))
        popm = torch.tensor((D.pop[idx] <= 3) & m, dtype=torch.float64)
        th = torch.zeros(X.shape[-1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)
        f, lam = self.f, self.lam

        def closure():
            opt.zero_grad()
            pi = torch.softmax((Xt @ th).masked_fill(~mt, -1e9), 1)
            R = (pi * ret).sum(1)
            loss = -torch.log(1 - f + f * R).mean() + lam * f * (pi * popm).sum(1).mean() + self.l2 * f * (th ** 2).sum()
            (loss / f).backward()
            return loss / f

        opt.step(closure)
        self.theta = th.detach().numpy().copy()
        return self

    def score(self, D, idx):
        s = ((C.feat_tensor(D, self.names, idx) - self.mu) / self.sd) @ self.theta
        return np.where(D.mask[idx], s, -np.inf)


def gen_alloc(D, pi, bt, ridx, thr=1e-3):
    tk = C.Tk()
    ret = D.ret_win if bt == "tansho" else D.ret_place
    for r in ridx:
        n = D.n[r]
        ids = np.where(pi[r, :n] > thr)[0]
        if len(ids) == 0:
            continue
        w = pi[r, ids] / pi[r, ids].sum()
        for i, st in zip(ids, w):
            tk.add(r, st, st * ret[r, i] / 100.0, D.win_odds[r, i])
    return tk


def extra(D, tk, ev):
    """配分の人気薄寄り度: 賭け金の中で単勝 10 倍以上に置いた割合。"""
    ri, st, rt, mo = tk.arr()
    return float(st[mo >= 10].sum() / st.sum()) if len(st) else float("nan")


# ── D2: 5 倍制約のバスケット ──
def basket_tansho(D, p, ridx, order, thr_roi=0.0, races=None, cap=0.2):
    tk = C.Tk()
    used = []
    for r in (ridx if races is None else races):
        n = D.n[r]
        o, pp = np.clip(D.win_odds[r, :n], 1.0, None), p[r, :n]
        if order == "edge":
            key = np.argsort(-(pp * o))
        elif order == "pop":
            key = np.argsort(o)
        elif order == "long":
            key = np.argsort(-o)
        else:
            key = np.random.default_rng(int(r)).permutation(n)
        S, cum = [], 0.0
        for i in key:
            if cum + 1.0 / o[i] <= cap:
                S.append(i); cum += 1.0 / o[i]
        if not S:
            continue
        S = np.array(S)
        if order == "edge" and thr_roi and pp[S].sum() / (1.0 / o[S]).sum() < thr_roi:
            continue
        w = 1.0 / o[S]
        s = w / w.sum()
        for i, st in zip(S, s):
            tk.add(r, st, st * D.ret_win[r, i] / 100.0, o[i])
        used.append(r)
    return tk, np.array(used, int)


_CI: dict = {}


def tri_idx(n):
    if n not in _CI:
        _CI[n] = np.array(list(combinations(range(n), 3)), int)
    return _CI[n]


def harville3(p, idx):
    a, b, c = p[idx[:, 0]], p[idx[:, 1]], p[idx[:, 2]]
    tot = 0.0
    for x, y, z in permutations((a, b, c)):
        tot = tot + x * y / np.clip(1 - x, 1e-9, None) * z / np.clip(1 - x - y, 1e-9, None)
    return tot


def basket_sanren(D, p, ridx, order, thr_roi=0.0, races=None, cap_q=0.15):
    """想定オッズ = 0.75/市場ハーヴィル確率。Σ(1/想定オッズ) ≤ 0.2 ⇔ Σ市場確率 ≤ 0.15。賭け金は想定オッズの逆数に比例（当たれば想定上どれも同額）。"""
    tk = C.Tk()
    used = []
    for r in (ridx if races is None else races):
        n = D.n[r]
        idx = tri_idx(n)
        qc = harville3(D.q[r, :n], idx)
        if order == "edge":
            pc = harville3(p[r, :n], idx)
            key = np.argsort(-(pc / np.clip(qc, 1e-9, None)))
        elif order == "pop":
            key = np.argsort(-qc)
        elif order == "long":
            key = np.argsort(qc)
        else:
            key = np.random.default_rng(int(r)).permutation(len(qc))
        cs = np.cumsum(qc[key])
        S = key[cs <= cap_q]
        if len(S) == 0:
            continue
        if order == "edge" and thr_roi and 0.75 * pc[S].sum() / qc[S].sum() < thr_roi:
            continue
        s = qc[S] / qc[S].sum()
        hn, hd = D.hn[r], D.hits[r].get("sanrenpuku", {})
        for k, st in zip(S, s):
            kk = tuple(sorted(int(hn[j]) for j in idx[k]))
            tk.add(r, st, st * hd.get(kk, 0.0) / 100.0, 1.0 / (qc[k] / 0.75))
        used.append(r)
    return tk, np.array(used, int)


def hit_multiples(D, tk):
    ri, st, rt, mo = tk.arr()
    s, r = np.bincount(ri, st, D.R), np.bincount(ri, rt, D.R)
    h = (r > 0)
    mult = r[h] / s[h]
    return {"hit_races": int(h.sum()), "mean_multiple_on_hit": float(mult.mean()) if h.any() else 0.0, "share_hit_ge5": float((mult >= 5).mean()) if h.any() else 0.0}


def main() -> None:
    D = C.load()
    ev = C.races_of(D, C.OOF_YEARS)
    res: dict = {"theta": {}, "policies": []}
    pop = np.where(D.mask, -D.pop, -np.inf)
    sA1 = np.nan_to_num(np.load(C.OUT / "score_A1.npy"), nan=-np.inf)
    sB = np.load(C.OUT / "scores_B.npz")
    q_alloc = np.where(D.mask, D.q, 0.0)
    scores = {}

    # ── D1 ──
    specs = {
        "D1w_f0.001(RL期待値)": (0.001, 0.0, "win"), "D1w_f0.05": (0.05, 0.0, "win"), "D1w_f0.3": (0.3, 0.0, "win"),
        "D1w_f0.3_pen1": (0.3, 1.0, "win"), "D1w_f0.3_pen3": (0.3, 3.0, "win"),
        "D1p_f0.3": (0.3, 0.0, "place"), "D1p_f0.3_pen1": (0.3, 1.0, "place"),
    }
    for nm, (f, lam, kind) in specs.items():
        s, th = C.loyo_scores(D, lambda f=f, lam=lam, kind=kind: Alloc(NAMES, f, lam, kind))
        s = np.where(np.isnan(s), -np.inf, s)
        scores[nm] = s
        res["theta"][nm] = th
        pi = soft(s)
        bt = "tansho" if kind == "win" else "fukusho"
        tk = gen_alloc(D, pi, bt, ev)
        ct = gen_alloc(D, q_alloc, bt, ev)
        o = C.summarize(D, f"D1/{nm}/{bt}_alloc", tk, ct, group="D", params={"f": f, "lam": lam, "kind": kind, "mode": "alloc"})
        o["stake_share_ge10"] = extra(D, tk, ev)
        res["policies"].append(o)
        print(C.fmt(o) + f" | 10倍以上への賭け金割合 {o['stake_share_ge10']:.3f}（q比例 {extra(D, ct, ev):.3f}）θ={th[2024]}", flush=True)
        tk1 = C.gen_single(D, s, bt, 1, ridx=ev)
        ct1 = C.gen_single(D, pop, bt, 1, ridx=ev)
        o1 = C.summarize(D, f"D1/{nm}/{bt}_top1", tk1, ct1, group="D", params={"f": f, "lam": lam, "kind": kind, "mode": "top1"})
        res["policies"].append(o1)
        print(C.fmt(o1), flush=True)

    # 市場情報だけの同型モデル（人気薄寄りの偏りだけで稼げる分）
    for f in (0.001, 0.05):
        s, th = C.loyo_scores(D, lambda f=f: Alloc(NAMES_MKT, f, 0.0, "win"))
        s = np.where(np.isnan(s), -np.inf, s)
        scores[f"D1w_f{f}_mktonly"] = s
        res["theta"][f"D1w_f{f}_mktonly"] = th
        tkm = gen_alloc(D, soft(s), "tansho", ev)
        o = C.summarize(D, f"D1/D1w_f{f}_mktonly/tansho_alloc", tkm, gen_alloc(D, q_alloc, "tansho", ev), group="D", params={"f": f, "mode": "alloc_mktonly"})
        o["stake_share_ge10"] = extra(D, tkm, ev)
        res["policies"].append(o)
        print(C.fmt(o) + f" | 10倍以上への賭け金割合 {o['stake_share_ge10']:.3f}", flush=True)
        key = [k for k in scores if k.startswith("D1w_f") and "_" not in k[5:].replace("(", "_").split("_")[0] and k.startswith(f"D1w_f{f}(") ]
        for k in key:
            tkp = gen_alloc(D, soft(scores[k]), "tansho", ev)
            o2 = C.summarize(D, "tmp", tkp, tkm, group="D", log=False, detail=False)
            res["policies"].append({"name": f"D1/{k}/対 市場のみ同型", "diff": o2["diff"], "diff_ci": o2["diff_ci"], "by_year": o2["by_year"]})
            print(f"      {k} − 市場のみ同型 = {o2['diff']:+.3f}[{o2['diff_ci'][0]:+.3f},{o2['diff_ci'][1]:+.3f}] 年別 " + " ".join(f"{y}:{v['diff']:+.3f}" for y, v in o2["by_year"].items()), flush=True)
    for k in ("D1w_f0.05",):
        tkp = gen_alloc(D, soft(scores[k]), "tansho", ev)
        tkm = gen_alloc(D, soft(scores["D1w_f0.05_mktonly"]), "tansho", ev)
        o2 = C.summarize(D, "tmp", tkp, tkm, group="D", log=False, detail=False)
        res["policies"].append({"name": f"D1/{k}/対 市場のみ同型", "diff": o2["diff"], "diff_ci": o2["diff_ci"], "by_year": o2["by_year"]})
        print(f"      {k} − 市場のみ同型 = {o2['diff']:+.3f}[{o2['diff_ci'][0]:+.3f},{o2['diff_ci'][1]:+.3f}] 年別 " + " ".join(f"{y}:{v['diff']:+.3f}" for y, v in o2["by_year"].items()), flush=True)

    # ── D2: 5 倍制約 ──
    pA1 = soft(sA1)
    pB3 = soft(np.where(np.isnan(sB["B3_offset"]), -np.inf, sB["B3_offset"]))
    for pn, p in (("A1", pA1), ("B3", pB3)):
        for th in (0.0, 1.0, 1.1):
            tk, used = basket_tansho(D, p, ev, "edge", th)
            ct, _ = basket_tansho(D, p, ev, "pop", races=used)
            o = C.summarize(D, f"D2/tansho_{pn}_roi{th}", tk, ct, group="D", params={"p": pn, "thr": th, "bt": "tansho"})
            if o.get("empty"):
                print(f"D2/tansho_{pn}_roi{th}: 買いなし"); continue
            o["hit_multiples"] = hit_multiples(D, tk)
            res["policies"].append(o)
            print(C.fmt(o) + f" | レース {len(used)} 的中レース率 {o['race_hit_rate']:.3f} 当たり時の倍率 平均 {o['hit_multiples']['mean_multiple_on_hit']:.2f} ≥5倍の割合 {o['hit_multiples']['share_hit_ge5']:.2f}", flush=True)
            for od in ("long", "rand"):
                c2, _ = basket_tansho(D, p, ev, od, races=used)
                o2 = C.summarize(D, "tmp", tk, c2, group="D", log=False, detail=False)
                o[f"vs_{od}"] = {"ctrl_roi": o2["ctrl_roi"], "diff": o2["diff"], "diff_ci": o2["diff_ci"], "by_year": o2["by_year"]}
                print(f"      対 {od}順 {o2['ctrl_roi']:.3f} 差 {o2['diff']:+.3f}[{o2['diff_ci'][0]:+.3f},{o2['diff_ci'][1]:+.3f}] 年別 " + " ".join(f"{y}:{v['diff']:+.3f}" for y, v in o2["by_year"].items()), flush=True)
    for pn, p in (("A1", pA1),):
        for th in (0.0, 0.9, 1.0):
            tk, used = basket_sanren(D, p, ev, "edge", th)
            ct, _ = basket_sanren(D, p, ev, "pop", races=used)
            o = C.summarize(D, f"D2/sanrenpuku_{pn}_roi{th}", tk, ct, group="D", params={"p": pn, "thr": th, "bt": "sanrenpuku"})
            if o.get("empty"):
                print(f"D2/sanrenpuku_{pn}_roi{th}: 買いなし"); continue
            o["hit_multiples"] = hit_multiples(D, tk)
            res["policies"].append(o)
            print(C.fmt(o) + f" | レース {len(used)} 的中レース率 {o['race_hit_rate']:.3f} 当たり時の倍率 平均 {o['hit_multiples']['mean_multiple_on_hit']:.2f} ≥5倍の割合 {o['hit_multiples']['share_hit_ge5']:.2f}", flush=True)
            for od in ("long", "rand"):
                c2, _ = basket_sanren(D, p, ev, od, races=used)
                o2 = C.summarize(D, "tmp", tk, c2, group="D", log=False, detail=False)
                o[f"vs_{od}"] = {"ctrl_roi": o2["ctrl_roi"], "diff": o2["diff"], "diff_ci": o2["diff_ci"], "by_year": o2["by_year"]}
                print(f"      対 {od}順 {o2['ctrl_roi']:.3f} 差 {o2['diff']:+.3f}[{o2['diff_ci'][0]:+.3f},{o2['diff_ci'][1]:+.3f}] 年別 " + " ".join(f"{y}:{v['diff']:+.3f}" for y, v in o2["by_year"].items()), flush=True)

    (C.OUT / "stage_D.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(C.OUT / "scores_D.npz", **{k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in scores.items()})
    C.save_trials("D")


if __name__ == "__main__":
    main()
