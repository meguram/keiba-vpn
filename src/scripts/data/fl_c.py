"""最終層 C: レース単位の一致派。選んだ頭数の束に 3 着以内の 3 頭が全部入る確率を直接上げる。

C1 学習: PL3（1〜3 着の順序つき尤度）／BCE3（3 着内かどうかの 2 値）／MARG（3 着内の最下位と 3 着外の最上位の差を広げる）
   × 市場あり・なし × 重み γ（3 頭の想定 3 連複オッズ ∝ 1/(q1 q2 q3) の γ 乗でレースを重くする）
C2 荒れるレースの分類（3 着内に単勝 10 倍以上が入るか）→ 上位 30% / 50% のレースだけ束を買う
C3 束の買い方: 3 連複 box / 軸流し、ワイド box / 流し。軸を B の人気薄発掘で決める併用
python -m src.scripts.data.fl_c
"""

from __future__ import annotations

import json

import numpy as np

from src.scripts.data import fl_core as C

F_MKT = ["lp", "l2", "l3", "lq", "lo", "lo2", "prk", "fav", "lpz"]
F_NOM = ["lp", "l2", "l3", "lpz"]


class Rank3:
    def __init__(self, names, kind="pl3", gamma=0.0, l2=1e-3):
        self.names, self.kind, self.gamma, self.l2 = names, kind, gamma, l2

    def fit(self, D, idx):
        import torch
        import torch.nn.functional as Fn

        X = C.feat_tensor(D, self.names, idx)
        m = D.mask[idx]
        fp = D.fp[idx]
        self.mu, self.sd = X[m].mean(0), X[m].std(0) + 1e-9
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        mt = torch.tensor(m)
        t3n = (fp <= 3) & m
        prod = np.where(t3n, D.q[idx], 1.0).prod(1)
        w = (1.0 / np.clip(prod, 1e-8, None)) ** self.gamma if self.gamma else np.ones(len(idx))
        w = torch.tensor(w / w.mean(), dtype=torch.float64)
        t3 = torch.tensor(t3n)
        o = [torch.tensor(np.argmax(fp == k, 1)) for k in (1, 2, 3)]
        ar = torch.arange(len(idx))
        th = torch.zeros(X.shape[-1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
        kind = self.kind

        def closure():
            opt.zero_grad()
            s = (Xt @ th).masked_fill(~mt, -1e9)
            if kind == "pl3":
                s2 = s.scatter(1, o[0][:, None], -1e9)
                s3 = s2.scatter(1, o[1][:, None], -1e9)
                ll = (s[ar, o[0]] - torch.logsumexp(s, 1)) + (s2[ar, o[1]] - torch.logsumexp(s2, 1)) + (s3[ar, o[2]] - torch.logsumexp(s3, 1))
                loss = -(w * ll).sum() / w.sum()
            elif kind == "bce3":
                e = Fn.binary_cross_entropy_with_logits(s + b, t3.double(), reduction="none").masked_fill(~mt, 0.0).sum(1)
                loss = (w * e).sum() / w.sum() / 3
            else:
                a = -torch.logsumexp((-s).masked_fill(~t3, -1e9), 1)
                c = torch.logsumexp(s.masked_fill(t3 | ~mt, -1e9), 1)
                loss = (w * Fn.softplus(c - a)).sum() / w.sum()
            loss = loss + self.l2 * (th ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.theta = th.detach().numpy().copy()
        return self

    def score(self, D, idx):
        s = ((C.feat_tensor(D, self.names, idx) - self.mu) / self.sd) @ self.theta
        return np.where(D.mask[idx], s, -np.inf)


def cov_vec(D, score, K, ev):
    out = np.zeros((len(ev), 2))
    for j, r in enumerate(ev):
        c = int(D.top3[r, C._order(score[r], D.n[r])[:K]].sum())
        out[j] = (c == 3, c >= 2)
    return out


def boot_diff(a, b, B=1000):
    d = a - b
    idx = C._boot_idx(len(d), B)
    bs = d[idx].mean(1)
    return float(d.mean()), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def auc(score, y):
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def race_features(D):
    q = np.where(D.mask, D.q, 0.0)
    qs = -np.sort(-q, 1)
    ln = np.log(D.n)
    p = np.where(D.mask, D.p_win, 0.0)
    ent_q = -(q * np.log(np.clip(q, 1e-12, None))).sum(1) / ln
    ent_p = -(p * np.log(np.clip(p, 1e-12, None))).sum(1) / ln
    kl = (p * np.where(D.mask, D.lp - D.lq, 0.0)).sum(1)
    long_ = D.mask & (D.win_odds >= C.LONG_ODDS)
    gap = np.where(long_, D.lp - D.lq, -5.0).max(1)
    dirt = np.array([1.0 if "ダ" in str(s) or "dirt" in str(s).lower() else 0.0 for s in D.surface])
    return np.stack([ln, qs[:, 0], qs[:, :3].sum(1), ent_q, ent_p, kl, long_.sum(1) / D.n, gap, D.distance / 1000.0, dirt], 1)


def rough_loyo(D, y, years=C.OOF_YEARS, l2=1e-2, Xr=None):
    import torch

    Xr = race_features(D) if Xr is None else Xr
    pred, thr = np.full(D.R, np.nan), {}
    for yv in years:
        tr, te = C.races_of(D, [x for x in years if x != yv]), C.races_of(D, [yv])
        mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9
        Xt = torch.tensor((Xr - mu) / sd, dtype=torch.float64)
        yt = torch.tensor(y.astype(float))
        th = torch.zeros(Xr.shape[1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
        trt = torch.tensor(tr)

        def closure():
            opt.zero_grad()
            z = Xt[trt] @ th[1:] + th[0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt[trt]) + l2 * (th[1:] ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        z = (Xt @ th[1:] + th[0]).detach().numpy()
        pred[te] = z[te]
        thr[yv] = {q: float(np.quantile(z[tr], q)) for q in (0.5, 0.7)}
    return pred, thr


def with_axis(base, axis_score, cand):
    out = np.where(np.isfinite(base), base, -np.inf).copy()
    sc = np.where(cand, axis_score, -np.inf)
    has = np.isfinite(sc).any(1)
    ai = np.argmax(sc, 1)
    out[np.where(has)[0], ai[has]] = 1e9
    return out


def main() -> None:
    D = C.load()
    ev = C.races_of(D, C.OOF_YEARS)
    print("surface 値:", sorted(set(map(str, D.surface)))[:6], flush=True)
    pop = np.where(D.mask, -D.pop, -np.inf)
    sA1 = np.nan_to_num(np.load(C.OUT / "score_A1.npy"), nan=-np.inf)
    sB = np.load(C.OUT / "scores_B.npz")
    cand = D.mask & (D.win_odds >= C.LONG_ODDS)
    res: dict = {"cov": {}, "theta": {}, "rough": {}, "policies": []}

    # ── C1: 学習 ──
    specs = {
        "PL3_mkt": (F_MKT, "pl3", 0.0), "PL3_nomkt": (F_NOM, "pl3", 0.0), "PL3_mkt_g0.5": (F_MKT, "pl3", 0.5), "PL3_mkt_g1": (F_MKT, "pl3", 1.0),
        "BCE3_mkt": (F_MKT, "bce3", 0.0), "BCE3_mkt_g1": (F_MKT, "bce3", 1.0),
        "MARG_mkt": (F_MKT, "marg", 0.0), "MARG_nomkt": (F_NOM, "marg", 0.0), "MARG_mkt_g0.5": (F_MKT, "marg", 0.5), "MARG_mkt_g1": (F_MKT, "marg", 1.0),
    }
    scores = {"A1": sA1}
    for nm, (fe, kd, g) in specs.items():
        s, th = C.loyo_scores(D, lambda fe=fe, kd=kd, g=g: Rank3(fe, kd, g))
        scores[nm] = np.where(np.isnan(s), -np.inf, s)
        res["theta"][nm] = th
    cv_pop = {K: cov_vec(D, pop, K, ev) for K in (3, 4, 5, 6)}
    yr = D.year[ev]
    print("上位 K 頭に 3 着内の 3 頭が全部入る割合（差 = モデル − 人気順、95%区間）", flush=True)
    for nm, sc in scores.items():
        row = []
        for K in (3, 4, 5, 6):
            cv = cov_vec(D, sc, K, ev)
            d, ci = boot_diff(cv[:, 0], cv_pop[K][:, 0])
            by = [float((cv[yr == y, 0] - cv_pop[K][yr == y, 0]).mean()) for y in C.OOF_YEARS]
            res["cov"][f"{nm}_K{K}"] = {"all3": float(cv[:, 0].mean()), "pop_all3": float(cv_pop[K][:, 0].mean()), "diff": d, "ci": ci, "by_year": by, "ge2": float(cv[:, 1].mean()), "pop_ge2": float(cv_pop[K][:, 1].mean())}
            row.append(f"K{K} {cv[:, 0].mean():.3f}(人気 {cv_pop[K][:, 0].mean():.3f}) 差{d:+.3f}[{ci[0]:+.3f},{ci[1]:+.3f}] 年{'/'.join(f'{b:+.3f}' for b in by)}")
        print(f"{nm:14s} " + " | ".join(row), flush=True)

    # ── C2: 荒れるレースの分類 ──
    nlong3 = (D.top3 & D.mask & (D.win_odds >= 10)).sum(1)
    y_rough = nlong3 >= 1
    z, thr = rough_loyo(D, y_rough)
    res["rough"] = {"base_rate": float(y_rough[ev].mean()), "auc": {int(yv): auc(z[C.races_of(D, [yv])], y_rough[C.races_of(D, [yv])]) for yv in C.OOF_YEARS}, "auc_all": auc(z[ev], y_rough[ev]),
                    "frac_ge2": float((nlong3[ev] >= 2).mean())}
    print(f"荒れるレース(3着内に10倍以上が1頭以上) 基準率 {res['rough']['base_rate']:.3f} AUC {res['rough']['auc_all']:.3f} 年別 {res['rough']['auc']}", flush=True)
    sels = {}
    for qn in (0.5, 0.7):
        sel = np.zeros(D.R, bool)
        for yv in C.OOF_YEARS:
            te = C.races_of(D, [yv])
            sel[te] = z[te] >= thr[yv][qn]
        sels[qn] = sel
        print(f"  選んだレース上位{int((1-qn)*100)}%: 荒れる率 {y_rough[sel & np.isin(D.year, C.OOF_YEARS)].mean():.3f}", flush=True)
    res["rough"]["thr"] = thr

    # ── C3: 束の買い方 ──
    def run(name, sc, bt, K, kind, sel=None, ctrl_sc=None, extra=None):
        tk = C.gen_bundle(D, sc, bt, K, kind, sel=sel, ridx=ev)
        ct = C.gen_bundle(D, pop if ctrl_sc is None else ctrl_sc, bt, K, kind, sel=sel, ridx=ev)
        o = C.summarize(D, name, tk, ct, group="C", params={"bt": bt, "K": K, "kind": kind, "sel": bool(sel is not None)})
        res["policies"].append(o)
        print(C.fmt(o), flush=True)
        return o, tk

    for nm in ("A1", "PL3_mkt", "PL3_mkt_g1", "BCE3_mkt", "MARG_mkt", "MARG_mkt_g1", "MARG_nomkt", "PL3_nomkt"):
        sc = scores[nm]
        for bt, K, kind in (("sanrenpuku", 5, "box"), ("sanrenpuku", 6, "box"), ("sanrenpuku", 5, "nagashi"), ("wide", 4, "box")):
            run(f"C/{nm}/{bt}_{kind}{K}", sc, bt, K, kind)
    for nm in ("A1", "MARG_mkt", "PL3_mkt_g1"):
        for qn, sel in sels.items():
            for bt, K, kind in (("sanrenpuku", 5, "box"), ("sanrenpuku", 6, "box"), ("sanrenpuku", 5, "nagashi")):
                run(f"C/{nm}/rough{int((1-qn)*100)}/{bt}_{kind}{K}", scores[nm], bt, K, kind, sel=sel)

    # 軸を B の人気薄発掘で決め、相手は A1 / 人気順
    for bn in ("B0_resid", "B1_g1"):
        axis = np.where(np.isnan(sB[bn]), -np.inf, sB[bn])
        for pn, psc in (("A1", sA1), ("MARG_mkt", scores["MARG_mkt"])):
            s_ax = with_axis(psc, axis, cand)
            s_ax_pop = with_axis(pop, axis, cand)
            for bt, K, kind in (("sanrenpuku", 5, "nagashi"), ("sanrenpuku", 6, "nagashi"), ("wide", 5, "nagashi"), ("umaren", 5, "nagashi")):
                o, _ = run(f"C/axis_{bn}/partners_{pn}/{bt}_{kind}{K}", s_ax, bt, K, kind)
                tk = C.gen_bundle(D, s_ax, bt, K, kind, ridx=ev)
                ct2 = C.gen_bundle(D, s_ax_pop, bt, K, kind, ridx=ev)
                o2 = C.summarize(D, "tmp", tk, ct2, group="C", log=False, detail=False)
                print(f"      同じ軸で相手だけ人気順との差 {o2['diff']:+.3f}[{o2['diff_ci'][0]:+.3f},{o2['diff_ci'][1]:+.3f}] 年別 " + " ".join(f"{y}:{v['diff']:+.3f}" for y, v in o2["by_year"].items()), flush=True)
                o["vs_same_axis_pop_partners"] = {"diff": o2["diff"], "ci": o2["diff_ci"], "by_year": o2["by_year"]}

    (C.OUT / "stage_C.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(C.OUT / "scores_C.npz", **{k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in scores.items()}, rough_z=z)
    C.save_trials("C")


if __name__ == "__main__":
    main()
