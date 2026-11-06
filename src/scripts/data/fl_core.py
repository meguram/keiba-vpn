"""最終層の共通部品: 入力表の読み込み・条件付きロジット・券の生成・評価（対照との差の区間など）。

評価の固定ルール（実験前に決めた）:
- 主指標 = 同じ点数・同じレースで『最も人気の券』を買う対照との回収率の差（レース単位の対ブートストラップ、95%）。
- 併記 = 回収率の絶対値、最大の的中 1 本／上位 3 本を除いた回収率、年別の差、的中率、平均払戻、オッズ帯別、最長連敗、破産確率。
- 選択は 2022〜2024 年の年抜き（2 年で学習し 1 年で評価）の結果だけ。3 年とも差の符号が同じものだけ候補にする。
- 試した設定はすべて TRIALS に記録し、最後に Reality Check をかける。
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(os.environ.get("FL_OUT", "/home/jovyan/work/fl_out"))
OOF_YEARS = (2022, 2023, 2024)
TEST_YEAR = 2025
LONG_ODDS = 7.0
BANDS = [(0, 3, "<3"), (3, 7, "3-7"), (7, 15, "7-15"), (15, 30, "15-30"), (30, 1e9, "30+")]
UNORDERED = ("umaren", "wide", "sanrenpuku")


class Data:
    pass


def load() -> Data:
    H = pd.read_parquet(OUT / "horses.parquet")
    hits_raw = pickle.loads((OUT / "hits.pkl").read_bytes())
    H = H.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    rids = H["race_id"].unique()
    codes = pd.Series(np.arange(len(rids)), index=rids)
    H["ri"] = codes.loc[H["race_id"]].to_numpy()
    n = np.bincount(H["ri"].to_numpy())
    R, N = len(rids), int(n.max())
    pos = H.groupby("ri").cumcount().to_numpy()
    D = Data()
    D.R, D.N, D.n, D.rids, D.H = R, N, n, rids, H
    D.mask = np.zeros((R, N), bool)
    D.mask[H["ri"], pos] = True
    D.pos = pos

    def pad(col, fill=np.nan, dtype=float):
        a = np.full((R, N), fill, dtype=dtype)
        a[H["ri"], pos] = H[col].to_numpy()
        return a

    D.pad = pad
    D.hn = pad("horse_number", 0, int)
    D.fp = pad("fp", 99, int)
    D.win_odds = pad("win_odds", 9999.0)
    D.place_mid = pad("place_mid", np.nan)
    D.p_win, D.p_top2, D.p_top3 = pad("p_win", 1e-9), pad("p_top2", 1e-9), pad("p_top3", 1e-9)
    D.q = pad("q_mkt", 1e-9)
    D.pop = pad("pop_rank", 99.0)
    D.ret_win, D.ret_place = pad("ret_win", 0.0), pad("ret_place", 0.0)
    r1 = H.groupby("ri").first()
    D.year = r1["year"].to_numpy()
    D.month = r1["month"].to_numpy()
    D.date = r1["date"].to_numpy()
    D.venue = r1["venue"].to_numpy()
    D.surface = r1["surface"].to_numpy()
    D.distance = r1["distance"].to_numpy()
    D.grade = r1["grade"].to_numpy()
    D.race_class = r1["race_class"].to_numpy()
    D.hits = []
    for rid in rids:
        h = {}
        for bt, dd in hits_raw[rid].items():
            h[bt] = {(tuple(sorted(k)) if bt in UNORDERED else k): v for k, v in dd.items()}
        D.hits.append(h)
    D.lp, D.lq = np.log(np.clip(D.p_win, 1e-9, None)), np.log(np.clip(D.q, 1e-9, None))
    D.win = (D.fp == 1)
    D.top3 = (D.fp <= 3)
    D.winner_idx = np.argmax(D.win, axis=1)
    return D


def races_of(D: Data, years) -> np.ndarray:
    return np.where(np.isin(D.year, list(years)))[0]


# ───────────────────────── 条件付きロジット ─────────────────────────
FEATS = {
    "lp": lambda D: D.lp,
    "lq": lambda D: D.lq,
    "gap": lambda D: D.lp - D.lq,
    "l2": lambda D: np.log(np.clip(D.p_top2, 1e-9, None)),
    "l3": lambda D: np.log(np.clip(D.p_top3, 1e-9, None)),
    "lo": lambda D: np.log(np.clip(D.win_odds, 1.0, None)),
    "lo2": lambda D: np.log(np.clip(D.win_odds, 1.0, None)) ** 2,
    "prk": lambda D: np.where(D.mask, D.pop / D.n[:, None], 0.0),
    "fav": lambda D: (D.pop == 1).astype(float),
    "lpz": lambda D: _race_z(D, D.lp),
    "lnN": lambda D: np.broadcast_to(np.log(D.n)[:, None], D.lp.shape),
}


def _race_z(D: Data, a: np.ndarray) -> np.ndarray:
    m = np.where(D.mask, a, np.nan)
    mu, sd = np.nanmean(m, 1, keepdims=True), np.nanstd(m, 1, keepdims=True) + 1e-9
    return (a - mu) / sd


def feat_tensor(D: Data, names, idx: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(FEATS[k](D), float)[idx] for k in names], -1)


class CLogit:
    """レース内 softmax の条件付きロジット。score = X·θ + offset。"""

    def __init__(self, names, l2=1e-3, gamma=0.0, lam=0.0, offset=None, only_long=False, lq_name="lq"):
        self.names, self.l2, self.gamma, self.lam, self.offset, self.only_long, self.lq_name = names, l2, gamma, lam, offset, only_long, lq_name

    def _prep(self, D, idx):
        X = feat_tensor(D, self.names, idx)
        m = D.mask[idx].copy()
        if self.only_long:
            m &= (D.win_odds[idx] >= LONG_ODDS)
        return X, m

    def fit(self, D: Data, idx: np.ndarray):
        import torch

        X, m = self._prep(D, idx)
        win = D.winner_idx[idx]
        keep = m[np.arange(len(idx)), win]          # 勝ち馬が対象外のレースは学習に使わない（only_long のとき）
        X, m, win, idx2 = X[keep], m[keep], win[keep], idx[keep]
        flat = X[m]
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-9
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        mt = torch.tensor(m)
        wt = torch.tensor(win)
        w = np.ones(len(idx2))
        if self.gamma:
            w = (1.0 / np.clip(D.q[idx2, win], 1e-4, None)) ** self.gamma
            w = w / w.mean()
        w = torch.tensor(w, dtype=torch.float64)
        off = torch.zeros_like(mt, dtype=torch.float64) if self.offset is None else torch.tensor(self.offset(D, idx2), dtype=torch.float64)
        lqi = self.names.index(self.lq_name) if self.lam and self.lq_name in self.names else None
        lqt = torch.tensor(D.lq[idx2], dtype=torch.float64)
        th = torch.zeros(X.shape[-1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
        ar = torch.arange(len(idx2))
        pen = [False]

        def closure():
            opt.zero_grad()
            s = (Xt @ th + off).masked_fill(~mt, -1e9)
            ll = s[ar, wt] - torch.logsumexp(s, 1)
            loss = -(w * ll).sum() / w.sum() + self.l2 * (th ** 2).sum()
            if self.lam and pen[0]:
                mm = mt.double()
                sc = (s.masked_fill(~mt, 0.0) - (s.masked_fill(~mt, 0.0) * mm).sum(1, keepdim=True) / mm.sum(1, keepdim=True)) * mm
                qc = (lqt - (lqt * mm).sum(1, keepdim=True) / mm.sum(1, keepdim=True)) * mm
                corr = (sc * qc).sum() / torch.sqrt((sc ** 2).sum() * (qc ** 2).sum() + 1e-12)
                loss = loss + self.lam * corr ** 2
            loss.backward()
            return loss

        opt.step(closure)
        if self.lam:                                  # 罰則なしの解から始める（θ=0 では相関の勾配が発散する）
            pen[0] = True
            opt = torch.optim.LBFGS([th], lr=1.0, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
            opt.step(closure)
        self.theta = th.detach().numpy().copy()
        return self

    def score(self, D: Data, idx: np.ndarray) -> np.ndarray:
        X = feat_tensor(D, self.names, idx)
        s = ((X - self.mu) / self.sd) @ self.theta
        if self.offset is not None:
            s = s + self.offset(D, idx)
        return np.where(D.mask[idx], s, -np.inf)

    def prob(self, D: Data, idx: np.ndarray) -> np.ndarray:
        s = self.score(D, idx)
        if self.only_long:
            s = np.where(D.win_odds[idx] >= LONG_ODDS, s, -np.inf)
        e = np.exp(s - np.max(s, 1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def loyo_scores(D: Data, make, years=OOF_YEARS, kind="score"):
    """年抜き: 他の年で学習して 1 年を予測。戻り値 [R, N]（評価年以外は NaN）。"""
    out = np.full((D.R, D.N), np.nan)
    thetas = {}
    for y in years:
        tr, te = races_of(D, [x for x in years if x != y]), races_of(D, [y])
        mdl = make().fit(D, tr)
        out[te] = mdl.score(D, te) if kind == "score" else mdl.prob(D, te)
        thetas[y] = mdl.theta.round(3).tolist()
    return out, thetas


# ───────────────────────── 券の生成 ─────────────────────────
class Tk:
    def __init__(self):
        self.ri, self.stake, self.ret, self.mo, self.hit = [], [], [], [], []

    def add(self, ri, stake, ret, mo):
        self.ri.append(ri); self.stake.append(stake); self.ret.append(ret); self.mo.append(mo)

    def arr(self):
        return (np.array(self.ri, int), np.array(self.stake, float), np.array(self.ret, float), np.array(self.mo, float))


def _order(score_row, n):
    s = np.where(np.isfinite(score_row[:n]), score_row[:n], -1e18)
    return np.argsort(-s, kind="stable")


def gen_single(D: Data, score, bt: str, k: int, sel=None, cand=None, ridx=None) -> Tk:
    """各レースで score 上位 k 頭の単勝／複勝を 1 単位ずつ買う。cand は候補馬の真偽 [R,N]、sel は買うレースの真偽 [R]。"""
    tk = Tk()
    ridx = range(D.R) if ridx is None else ridx
    for r in ridx:
        if sel is not None and not sel[r]:
            continue
        n = D.n[r]
        sc = score[r].copy()
        if cand is not None:
            sc = np.where(cand[r], sc, -np.inf)
        if not np.isfinite(sc[:n]).any():
            continue
        o = _order(sc, n)[:k]
        o = [i for i in o if np.isfinite(sc[i])]
        for i in o:
            rt = D.ret_win[r, i] if bt == "tansho" else D.ret_place[r, i]
            tk.add(r, 1.0, rt / 100.0, D.win_odds[r, i])
    return tk


def gen_uniform(D: Data, bt: str, cand, total=1.0, sel=None, ridx=None) -> Tk:
    """候補馬すべてに総額 total を等分（市場の中立な対照）。"""
    tk = Tk()
    ridx = range(D.R) if ridx is None else ridx
    for r in ridx:
        if sel is not None and not sel[r]:
            continue
        ids = np.where(cand[r][: D.n[r]])[0]
        if len(ids) == 0:
            continue
        for i in ids:
            rt = D.ret_win[r, i] if bt == "tansho" else D.ret_place[r, i]
            tk.add(r, total / len(ids), rt / 100.0 * total / len(ids), D.win_odds[r, i])
    return tk


def gen_bundle(D: Data, score, bt: str, K: int, kind="box", sel=None, ridx=None) -> Tk:
    """score 上位 K 頭の束。box = 全組合せ、nagashi = 1 位を軸に残りを相手（馬連・ワイドは軸−相手、3連複は軸+相手2頭）。"""
    tk = Tk()
    ridx = range(D.R) if ridx is None else ridx
    for r in ridx:
        if sel is not None and not sel[r]:
            continue
        n = D.n[r]
        if n < max(K, 4):
            continue
        o = _order(score[r], n)[:K]
        hn = D.hn[r]
        hd = D.hits[r].get(bt, {})
        if kind == "box":
            sets = combinations(o, 3 if bt == "sanrenpuku" else 2)
        else:
            ax, pa = o[0], o[1:]
            sets = ([(ax, x) for x in pa] if bt != "sanrenpuku" else [(ax, x, y) for x, y in combinations(pa, 2)])
        for s in sets:
            key = tuple(sorted(int(hn[i]) for i in s))
            tk.add(r, 1.0, hd.get(key, 0.0) / 100.0, float(max(D.win_odds[r, i] for i in s)))
    return tk


# ───────────────────────── 評価 ─────────────────────────
TRIALS: list[dict] = []
SERIES: dict[str, tuple] = {}
_BOOT: dict = {}


def _boot_idx(nr: int, B: int, seed: int = 0) -> np.ndarray:
    key = (nr, B, seed)
    if key not in _BOOT:
        _BOOT[key] = np.random.default_rng(seed).integers(0, nr, size=(B, nr), dtype=np.int32)
    return _BOOT[key]


def _per_race(D, tk: Tk):
    ri, st, rt, mo = tk.arr()
    return np.bincount(ri, st, D.R) if len(ri) else np.zeros(D.R), np.bincount(ri, rt, D.R) if len(ri) else np.zeros(D.R)


def _ex_top(rt: np.ndarray, st: float, k: int) -> float:
    if len(rt) == 0 or st <= 0:
        return float("nan")
    return float((rt.sum() - np.sort(rt)[::-1][:k].sum()) / st)


def summarize(D: Data, name: str, tk: Tk, ctrl: Tk, years=OOF_YEARS, B=1000, group="", params=None, detail=True, log=True) -> dict:
    ev = races_of(D, years)
    s, r = _per_race(D, tk)
    cs, cr = _per_race(D, ctrl)
    s, r, cs, cr = s[ev], r[ev], cs[ev], cr[ev]
    yr = D.year[ev]
    out: dict = {"name": name, "group": group, "params": params or {}, "years": list(years)}
    if s.sum() <= 0:
        out["empty"] = True
        return out
    roi, croi = r.sum() / s.sum(), (cr.sum() / cs.sum() if cs.sum() > 0 else float("nan"))
    idx = _boot_idx(len(ev), B)
    S, Rr, CS, CR = s[idx].sum(1), r[idx].sum(1), cs[idx].sum(1), cr[idx].sum(1)
    ok = (S > 0) & (CS > 0)
    b_roi, b_d = Rr[ok] / S[ok], (Rr[ok] / S[ok] - CR[ok] / CS[ok])
    out.update({"n_races_bought": int((s > 0).sum()), "n_tickets": int(len(tk.ri)), "stake": float(s.sum()), "roi": float(roi),
                "roi_ci": [float(np.quantile(b_roi, 0.025)), float(np.quantile(b_roi, 0.975))], "ctrl_roi": float(croi),
                "diff": float(roi - croi), "diff_ci": [float(np.quantile(b_d, 0.025)), float(np.quantile(b_d, 0.975))],
                "diff_p_le0": float((b_d <= 0).mean())})
    by_year = {}
    for y in years:
        m = yr == y
        if s[m].sum() > 0 and cs[m].sum() > 0:
            by_year[int(y)] = {"roi": float(r[m].sum() / s[m].sum()), "ctrl": float(cr[m].sum() / cs[m].sum()), "diff": float(r[m].sum() / s[m].sum() - cr[m].sum() / cs[m].sum())}
    out["by_year"] = by_year
    out["same_sign_all_years"] = bool(len(by_year) == len(years) and (all(v["diff"] > 0 for v in by_year.values())))
    ri, st, rt, mo = tk.arr()
    keep = np.isin(ri, ev)
    ri, st, rt, mo = ri[keep], st[keep], rt[keep], mo[keep]
    out["roi_ex_top1"], out["roi_ex_top3"] = _ex_top(rt, st.sum(), 1), _ex_top(rt, st.sum(), 3)
    hit = rt > 0
    out["ticket_hit_rate"] = float(np.average(hit, weights=st))
    out["mean_payout_per_hit"] = float(rt[hit].sum() / st[hit].sum()) if hit.any() else 0.0
    prof = r - s
    rb = (s > 0)
    out["race_hit_rate"] = float((r[rb] > 0).mean())
    if detail:
        bd = {}
        for lo, hi, lab in BANDS:
            m = (mo >= lo) & (mo < hi)
            if m.any() and st[m].sum() > 0:
                bd[lab] = {"tickets": int(m.sum()), "roi": float(rt[m].sum() / st[m].sum()), "hit": float(np.average(hit[m], weights=st[m]))}
        out["band"] = bd
        pb = prof[rb]
        streak = best = 0
        for p in pb:
            streak = streak + 1 if p < 0 else 0
            best = max(best, streak)
        out["longest_losing_streak_races"] = int(best)
        u = float(s[rb].mean())
        rng = np.random.default_rng(1)
        Nn = max(1, len(pb) // len(years))
        sims = (pb / u)[rng.integers(0, len(pb), size=(2000, Nn))].cumsum(1).min(1)
        out["ruin_30u"], out["ruin_100u"] = float((sims <= -30).mean()), float((sims <= -100).mean())
    if log:
        TRIALS.append(out)
        SERIES[name] = (prof.astype(np.float32), (cr - cs).astype(np.float32))
    return out


def fmt(o: dict) -> str:
    if o.get("empty"):
        return f"{o['name']}: 買いなし"
    yy = " ".join(f"{y}:{v['diff']:+.3f}" for y, v in o["by_year"].items())
    return (f"{o['name']:46s} 回収率 {o['roi']:.3f}[{o['roi_ci'][0]:.3f},{o['roi_ci'][1]:.3f}] 対照 {o['ctrl_roi']:.3f} 差 {o['diff']:+.3f}"
            f"[{o['diff_ci'][0]:+.3f},{o['diff_ci'][1]:+.3f}] 年別差 {yy} 最大除き {o['roi_ex_top1']:.3f}/{o['roi_ex_top3']:.3f} 的中 {o['ticket_hit_rate']:.3f} 点 {o['n_tickets']}")


def coverage(D: Data, score, K: int, years=OOF_YEARS, sel=None) -> dict:
    """上位 K 頭に 3 着以内の馬が何頭入ったか（レース単位）。"""
    c = []
    for r in races_of(D, years):
        if sel is not None and not sel[r]:
            continue
        n = D.n[r]
        o = _order(score[r], n)[:K]
        c.append(int(D.top3[r, o].sum()))
    c = np.array(c)
    return {"K": K, "races": int(len(c)), "all3": float((c == 3).mean()), "ge2": float((c >= 2).mean()), "ge1": float((c >= 1).mean())}


def prob_metrics(D: Data, p, years=OOF_YEARS) -> dict:
    """勝率の対数損失・Brier・（単勝 10 倍以上の）人気薄での対数損失と較正。p は [R,N]（レース内で合計 1）。"""
    ev = races_of(D, years)
    pp, w, m, od = p[ev], D.win[ev], D.mask[ev], D.win_odds[ev]
    pw = np.clip(np.where(w, pp, np.nan), 1e-9, None)
    ll = float(-np.nanmean(np.log(pw[np.isfinite(pw)])))
    br = float(np.mean(np.sum(np.where(m, (pp - w) ** 2, 0.0), 1)))
    L = m & (od >= 10)
    pl, yl = np.clip(pp[L], 1e-9, 1 - 1e-9), w[L].astype(float)
    return {"win_ll": ll, "brier": br, "long10_n": int(L.sum()), "long10_ll": float(-np.mean(yl * np.log(pl) + (1 - yl) * np.log(1 - pl))),
            "long10_sum_p": float(pl.sum()), "long10_wins": float(yl.sum())}


def save_trials(stage: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"trials_{stage}.json").write_text(json.dumps(TRIALS, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(OUT / f"series_{stage}.npz", **{k.replace("/", "_"): np.stack(v) for k, v in SERIES.items()})
