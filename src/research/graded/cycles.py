"""重賞のルールベース予想家: ゴール（◎ の勝率 30% 以上・単勝の回収率 120% 以上）とサブゴール（消した馬の 95% 以上が 4 着以下、
1 レース 3 頭以上を消す）に向けた分析サイクル。

- 表: 2018〜2026 年の重賞の全出走（着順の無い馬は除く）。列は as-of の型ごとの点 x（前日版の点と同じ tally を縮めたもの）、
  前日版・当日版の点、前日版の順位の印（上位 3・5）など。オッズ・人気は使わない。
- ◎ の決め方 = 列の重み付きの和（重みの 1 列が 1 つの決め方）。同点は前日版の点 → 馬番。
- 消しの決め方 = ある点の下位 k 頭（k = 3〜6）。
- A（当てはめ版）: 2025・2026 年で一番良い決め方を選ぶ（重賞ごと、または全体で 1 つ）。評価ではなく当てはめ。
- B（正直な版）: 2018〜2024 年で一番良い決め方を選び、2025・2026 年に当てはめる。
計算は手元だけ（費用 0 円）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WIN_GOAL, ROI_GOAL = 0.30, 1.20
KILL_GOAL, KILL_MIN = 0.95, 3
KS = (3, 4, 5, 6)
DEV, EVAL = (2018, 2024), (2025, 2026)


class Table:
    """行 = 馬（レースの中は 前日版の点の降順 → 馬番 で並べる）。F は特徴の行列、names は列名。"""

    def __init__(self, rows: pd.DataFrame, F: np.ndarray, names: list[str]):
        o = np.lexsort((rows["hn"].to_numpy(), -rows["score_pre"].to_numpy(), rows["race_id"].to_numpy()))
        self.r = rows.iloc[o].reset_index(drop=True)
        self.F = F[o].astype(np.float64)   # 点は倍精度で計算する（まとめた行列の積と 1 列の積で、同点の扱いが変わらないように）
        self.names = names
        rid = self.r["race_id"].to_numpy()
        self.starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
        self.race_of = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, len(rid)]))
        R = self.r.iloc[self.starts]
        self.race = pd.DataFrame({"race_id": R["race_id"].to_numpy(), "rkey": R["rkey"].to_numpy(), "grade": R["grade"].to_numpy(),
                                  "year": R["year"].to_numpy(), "pay_ok": self.r.groupby(self.race_of)["pay_ok"].max().to_numpy().astype(bool),
                                  "field": np.diff(np.r_[self.starts, len(rid)])})
        self.y1 = self.r["y1"].to_numpy(np.float32)
        self.y3 = self.r["y3"].to_numpy(np.float32)
        self.rt = np.where(self.r["pay_ok"].to_numpy(bool), self.r["ret_tan"].fillna(0).to_numpy(), 0).astype(np.float32)
        self.rf = np.where(self.r["pay_ok"].to_numpy(bool), self.r["ret_fuku"].fillna(0).to_numpy(), 0).astype(np.float32)
        self.bad = (self.r["fin"].to_numpy() >= 4).astype(np.float32)   # 4 着以下（消しの的中）

    def col(self, name: str) -> int:
        return self.names.index(name)


def honmei(t: Table, W: np.ndarray) -> np.ndarray:
    """決め方（W: 特徴 × 決め方）ごとの、レースごとの ◎ の行番号（レース × 決め方）。同点はレースの中の並び（前日版の点 → 馬番）の先。"""
    Sc = t.F @ W
    mx = np.maximum.reduceat(Sc, t.starts, axis=0)
    hit = Sc >= mx[t.race_of] - 1e-9 * np.maximum(1.0, np.abs(mx[t.race_of]))
    idx = np.where(hit, np.arange(len(Sc))[:, None], np.iinfo(np.int64).max)
    return np.minimum.reduceat(idx, t.starts, axis=0)


def killed(t: Table, K: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """消しの決め方（K: 特徴 × 決め方、点の低い順に消す）ごとの、レースごとの (消した頭数, 消した馬のうち 4 着以下の数)。
    k 頭を消す。頭数が k + 3 未満のレースは k を頭数 − 3 に減らす（3 頭は残す）。同点はレースの中の並びの後ろ（前日版の点が低い・馬番が大きい）から消す。"""
    Sc = t.F @ K
    nr, m = len(t.starts), K.shape[1]
    n_k = np.zeros((nr, m), np.float32)
    n_b = np.zeros((nr, m), np.float32)
    ends = np.r_[t.starts[1:], len(Sc)]
    for i, (a, b) in enumerate(zip(t.starts, ends)):
        kk = max(0, min(k, (b - a) - 3))
        if kk == 0:
            continue
        o = (b - a - 1) - np.argsort(Sc[a:b][::-1], axis=0, kind="stable")[:kk]
        n_k[i] = kk
        n_b[i] = t.bad[a:b][o].sum(0)
    return n_k, n_b


def objective(win: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """ゴールまでの近さ（1 以上で達成）: min(勝率 / 0.30, 単回収 / 1.20)。"""
    return np.minimum(win / WIN_GOAL, roi / ROI_GOAL)


def agg_honmei(t: Table, H: np.ndarray, sel_races: np.ndarray) -> dict[str, np.ndarray]:
    """レースの集合（真偽）での決め方ごとの ◎ の勝率・単回収・複勝率・複回収と、払戻の解けたレースの数。"""
    rs = np.flatnonzero(sel_races)
    h = H[rs]
    pay = t.race["pay_ok"].to_numpy()[rs][:, None]
    n = np.maximum(pay.sum(), 1)
    return {"win": t.y1[h].mean(0), "place": t.y3[h].mean(0), "roi_t": (t.rt[h] * pay).sum(0) / n, "roi_f": (t.rf[h] * pay).sum(0) / n,
            "races": np.full(h.shape[1], len(rs))}


# ------------------------------------------------------------------ 選び方の状態（サイクルをまたいで、一番良い決め方を持ち続ける）
class Best:
    """A・B × 重賞ごと・全体で 1 つの、◎ と消しの一番良い決め方と、その 2025・2026 年の値。"""

    def __init__(self, t: Table, dv: np.ndarray | None = None, ev: np.ndarray | None = None, rkeys: list[str] | None = None):
        """dv・ev = 開発（B を選ぶ）・評価（値を数える）のレースの真偽。既定は 2018〜2024 年・2025〜2026 年。
        重賞ごとの窓（条件の更新のバッチ、versions.py）では、重賞ごとの発見の回を dv、検証の回を ev にして渡す。"""
        self.t = t
        rk = t.race["rkey"].to_numpy()
        yr = t.race["year"].to_numpy()
        self.ev = ev.astype(bool) if ev is not None else (yr >= EVAL[0]) & (yr <= EVAL[1])
        self.dv = dv.astype(bool) if dv is not None else (yr >= DEV[0]) & (yr <= DEV[1])
        self.rkeys = sorted(rkeys) if rkeys is not None else sorted(set(rk[(t.race["year"] >= EVAL[0]).to_numpy()]))
        ri = {k: i for i, k in enumerate(self.rkeys)}
        self.rk_idx = np.array([ri.get(k, -1) for k in rk])
        nk = len(self.rkeys)
        self.G = np.zeros((nk, len(rk)), np.float32)
        ok = self.rk_idx >= 0
        self.G[self.rk_idx[ok], np.flatnonzero(ok)] = 1
        self.pay = t.race["pay_ok"].to_numpy().astype(np.float32)
        self.tried = 0
        neg = -np.inf
        # 重賞ごと: A は 評価の目的 → 開発の目的、B は 開発の目的 → 評価の目的 で比べる
        self.h = {m: {"key": np.full((nk, 2), neg), "val": np.zeros((nk, 5)), "sel": [None] * nk} for m in ("A", "B")}
        self.hg = {m: {"key": (neg, neg), "val": None, "sel": None} for m in ("A", "B")}
        self.k = {m: {"key": np.full((nk, 3), neg), "val": np.zeros((nk, 3)), "sel": [None] * nk} for m in ("A", "B")}
        self.kg = {m: {"key": (neg, neg, neg), "val": None, "sel": None} for m in ("A", "B")}

    def _sums(self, X: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """レース × 決め方の値を、重賞ごとに足す（sel のレースだけ）。"""
        return self.G @ (X * sel[:, None])

    def update(self, W: np.ndarray, labels: list[str], chunk: int = 1500) -> None:
        t = self.t
        for a in range(0, W.shape[1], chunk):
            Wc, lab = W[:, a:a + chunk], labels[a:a + chunk]
            H = honmei(t, Wc)
            win, plc = t.y1[H], t.y3[H]
            rt, rf = t.rt[H] * self.pay[:, None], t.rf[H] * self.pay[:, None]
            self._upd_honmei(win, plc, rt, rf, lab)
            for k in KS:
                nk, nb = killed(t, Wc, k)
                self._upd_kill(nk, nb, lab, k)
            self.tried += Wc.shape[1]

    def _upd_honmei(self, win, plc, rt, rf, lab):
        ev, dv = self.ev.astype(np.float32), self.dv.astype(np.float32)
        S = {}
        for nm, per in (("ev", ev), ("dv", dv)):
            S[nm] = {"win": self._sums(win, per), "plc": self._sums(plc, per), "rt": self._sums(rt, per), "rf": self._sums(rf, per),
                     "n": self.G @ per, "np": self.G @ (per * self.pay)}
        def obj(s):
            n, npay = s["n"][:, None], s["np"][:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                o = objective(s["win"] / n, s["rt"] / np.maximum(npay, 1))
            return np.where(n > 0, o, -np.inf)
        oe, od = obj(S["ev"]), obj(S["dv"])
        # B の同点は評価の値で決めない（評価の漏れになる）。先に見つけた決め方を残す
        z = np.zeros_like(od)
        for m, (k1, k2) in (("A", (oe, od)), ("B", (od, z))):
            cur = self.h[m]
            best_j = np.argmax(k1 + 1e-6 * np.nan_to_num(k2, neginf=-1e3), axis=1)
            for i in range(len(self.rkeys)):
                jj = best_j[i]
                key = (k1[i, jj], k2[i, jj])
                if key > tuple(cur["key"][i]):
                    cur["key"][i] = key
                    e = S["ev"]
                    cur["val"][i] = [e["win"][i, jj], e["plc"][i, jj], e["rt"][i, jj], e["rf"][i, jj], 0]
                    cur["sel"][i] = lab[jj]
        # 全体で 1 つ
        tot = {nm: {k: S[nm][k].sum(0) for k in ("win", "plc", "rt", "rf")} | {"n": S[nm]["n"].sum(), "np": S[nm]["np"].sum()} for nm in S}
        ge = objective(tot["ev"]["win"] / tot["ev"]["n"], tot["ev"]["rt"] / tot["ev"]["np"])
        gd = objective(tot["dv"]["win"] / tot["dv"]["n"], tot["dv"]["rt"] / tot["dv"]["np"])
        for m, (k1, k2) in (("A", (ge, gd)), ("B", (gd, np.zeros_like(gd)))):
            jj = int(np.argmax(k1 + 1e-6 * k2))
            if (k1[jj], k2[jj]) > self.hg[m]["key"]:
                self.hg[m] = {"key": (float(k1[jj]), float(k2[jj])), "sel": lab[jj],
                              "val": {"win": float(tot["ev"]["win"][jj] / tot["ev"]["n"]), "place": float(tot["ev"]["plc"][jj] / tot["ev"]["n"]),
                                      "roi_t": float(tot["ev"]["rt"][jj] / tot["ev"]["np"]), "roi_f": float(tot["ev"]["rf"][jj] / tot["ev"]["np"]),
                                      "dev_win": float(tot["dv"]["win"][jj] / tot["dv"]["n"]), "dev_roi_t": float(tot["dv"]["rt"][jj] / tot["dv"]["np"])}}

    def _upd_kill(self, nk, nb, lab, k):
        ev, dv = self.ev.astype(np.float32), self.dv.astype(np.float32)
        Ke, Be = self._sums(nk, ev), self._sums(nb, ev)
        Kd, Bd = self._sums(nk, dv), self._sums(nb, dv)
        with np.errstate(divide="ignore", invalid="ignore"):
            pe, pd_ = np.where(Ke > 0, Be / Ke, -np.inf), np.where(Kd > 0, Bd / Kd, -np.inf)
        for m, (k1, k2) in (("A", (pe, pd_)), ("B", (pd_, np.zeros_like(pd_)))):
            cur = self.k[m]
            best_j = np.argmax(k1 + 1e-6 * np.nan_to_num(k2, neginf=-1e3), axis=1)
            for i in range(len(self.rkeys)):
                jj = best_j[i]
                key = (k1[i, jj], float(k), k2[i, jj])
                if key > tuple(cur["key"][i]):
                    cur["key"][i] = key
                    cur["val"][i] = [Ke[i, jj], Be[i, jj], k]
                    cur["sel"][i] = f"{lab[jj]}・下位 {k} 頭"
        te, tb, tdk, tdb = Ke.sum(0), Be.sum(0), Kd.sum(0), Bd.sum(0)
        ge, gd = tb / np.maximum(te, 1), tdb / np.maximum(tdk, 1)
        for m, (k1, k2, KK, BB) in (("A", (ge, gd, te, tb)), ("B", (gd, np.zeros_like(gd), te, tb))):
            jj = int(np.argmax(k1 + 1e-6 * k2))
            if (float(k1[jj]), float(k), float(k2[jj])) > self.kg[m]["key"]:
                self.kg[m] = {"key": (float(k1[jj]), float(k), float(k2[jj])), "sel": f"{lab[jj]}・下位 {k} 頭",
                              "val": {"precision": float(BB[jj] / max(KK[jj], 1)), "killed": float(KK[jj]),
                                      "killed_per_race": float(KK[jj] / max(self.ev.sum(), 1))}}

    def report(self) -> dict:
        """今の一番良い決め方の、2025・2026 年の値（A・B × 重賞ごと・全体で 1 つ、◎ と消し、全体と G1〜G3 別）。"""
        t = self.t
        gr = {}
        for k in self.rkeys:
            g = t.race.loc[t.race["rkey"] == k, "grade"]
            gr[k] = g.iloc[-1]
        ne = self.G @ self.ev.astype(np.float32)
        npe = self.G @ (self.ev * self.pay).astype(np.float32)
        out = {"tried": int(self.tried), "n_rkeys": len(self.rkeys), "eval_races": int(self.ev.sum())}
        for m in ("A", "B"):
            v, sel = self.h[m]["val"], self.h[m]["sel"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ach = (v[:, 0] / ne >= WIN_GOAL) & (v[:, 2] / np.maximum(npe, 1) >= ROI_GOAL)
            res = {"achieved": int(ach.sum())}
            for gname in ("all", "G1", "G2", "G3"):
                mk = np.array([gname == "all" or gr[k] == gname for k in self.rkeys])
                n, npay = ne[mk].sum(), npe[mk].sum()
                res[gname] = {"races": int(n), "win": float(v[mk, 0].sum() / n) if n else None, "place": float(v[mk, 1].sum() / n) if n else None,
                              "roi_t": float(v[mk, 2].sum() / max(npay, 1)) if n else None, "roi_f": float(v[mk, 3].sum() / max(npay, 1)) if n else None,
                              "achieved": int(ach[mk].sum()), "rkeys": int(mk.sum())}
            nodev = ~np.isfinite(self.h[m]["key"][:, 0])
            res["per_rkey"] = [{"rkey": k, "grade": gr[k], "races": int(ne[i]), "win": float(v[i, 0] / ne[i]) if ne[i] else None,
                                "roi_t": float(v[i, 2] / npe[i]) if npe[i] else None, "achieved": bool(ach[i]), "rule": sel[i], "default": bool(nodev[i])}
                               for i, k in enumerate(self.rkeys)]
            kv, ks = self.k[m]["val"], self.k[m]["sel"]
            kres = {}
            for gname in ("all", "G1", "G2", "G3"):
                mk = np.array([gname == "all" or gr[k] == gname for k in self.rkeys])
                kres[gname] = {"precision": float(kv[mk, 1].sum() / max(kv[mk, 0].sum(), 1)), "killed_per_race": float(kv[mk, 0].sum() / max(ne[mk].sum(), 1))}
            with np.errstate(divide="ignore", invalid="ignore"):
                kach = kv[:, 1] / np.maximum(kv[:, 0], 1) >= KILL_GOAL
            kres["achieved"] = int(kach.sum())
            knodev = ~np.isfinite(self.k[m]["key"][:, 0])
            kres["per_rkey"] = [{"rkey": k, "precision": float(kv[i, 1] / kv[i, 0]) if kv[i, 0] else None, "k": int(kv[i, 2]), "rule": ks[i],
                                 "default": bool(knodev[i])}
                                for i, k in enumerate(self.rkeys)]
            out[m] = {"honmei_by_rkey": res, "honmei_global": self.hg[m], "kill_by_rkey": kres, "kill_global": self.kg[m]}
        return out


# ------------------------------------------------------------------ サイクルの仮説（決め方の候補）
BASE = ("score_pre", "score_day", "top3_pre", "top5_pre")


def _vec(t: Table, terms: dict[str, float], tie: float = 1e-3) -> np.ndarray:
    w = np.zeros(len(t.names), np.float32)
    for k, v in terms.items():
        w[t.col(k)] += v
    if "score_pre" not in terms:
        w[t.col("score_pre")] += tie
    return w


def _types(t: Table) -> list[str]:
    return [n for n in t.names if n.startswith("x_")]


def space(c: int, t: Table, rng: np.random.Generator) -> tuple[str, list[tuple[dict, str]]]:
    """サイクル c の仮説と、決め方の候補 [(重み, 名前)]。"""
    X = _types(t)
    nm = lambda d: " ＋ ".join(f"{v:g}×{k.removeprefix('x_')}" for k, v in d.items())   # noqa: E731
    if c == 1:
        return "基準: 前日版・当日版の点（凍結の規則のまま）", [({"score_pre": 1}, "前日版の点"), ({"score_day": 1}, "当日版の点")]
    if c == 2:
        return "型 1 つの点だけで並べる（118 型、同点は前日版の点）", [({x: 1}, nm({x: 1})) for x in X]
    if c == 3:
        return "前日版の点 ＋ α × 型（α = ±0.5・±1・±2・±4）", [({"score_pre": 1, x: a}, nm({"score_pre": 1, x: a})) for x in X for a in (-4, -2, -1, -0.5, 0.5, 1, 2, 4)]
    if c == 4:
        return "型の点を逆向きに使う（点の低い馬を ◎）: −型、−前日版の点", [({x: -1}, nm({x: -1})) for x in X] + [({"score_pre": -1}, "−前日版の点"), ({"score_day": -1}, "−当日版の点")]
    if c == 5:
        return "前日版の上位 3・5 頭に絞ってから、型 1 つで選ぶ（±）", [({b: 100, x: s}, f"{b}の中で {nm({x: s})}") for b in ("top3_pre", "top5_pre") for x in X for s in (1, -1)]
    if c == 6:
        return "型の対の和（2 つの AND の重み）", [({a: 1, b: 1}, nm({a: 1, b: 1})) for i, a in enumerate(X) for b in X[i + 1:]]
    if c == 7:
        return "型の対（片方を逆向き）", [({a: 1, b: -1}, nm({a: 1, b: -1})) for a in X for b in X if a != b]
    if c in (8, 9, 10):
        n = {8: 3, 9: 4, 10: 5}[c]
        out = []
        for _ in range(20000):
            ks = rng.choice(X, n, replace=False)
            d = {k: float(rng.choice([-1, 1])) for k in ks}
            out.append((d, nm(d)))
        return f"型 {n} つのランダムな組（向きもランダム、20,000 通り）", out
    if c in (11, 12):
        out = []
        for _ in range(20000):
            ks = rng.choice(X, 3 if c == 11 else 2, replace=False)
            d = {"score_pre": 1.0} | {k: float(rng.normal(0, 2)) for k in ks}
            d = {k: round(v, 2) for k, v in d.items()}
            out.append((d, nm(d)))
        return f"前日版の点 ＋ 型 {3 if c == 11 else 2} つの連続な重み（20,000 通り）", out
    if c in (13, 14):
        out = []
        for _ in range(20000):
            ks = rng.choice(X, 3, replace=False)
            b = "top5_pre" if c == 13 else "top3_pre"
            d = {b: 100.0} | {k: float(rng.choice([-1, 1])) for k in ks}
            out.append((d, f"{b}の中で " + nm({k: v for k, v in d.items() if k != b})))
        return f"前日版の上位 {5 if c == 13 else 3} 頭に絞ってから、型 3 つのランダムな組で選ぶ（20,000 通り）", out
    if c in (15, 16):
        KH = [n for n in t.names if n.startswith("khit_")]
        out = []
        if c == 15:
            for kh in KH:
                for a in (0.5, 1, 2, 5, 10):
                    d = {"score_pre": 1.0, kh: -float(a)}
                    out.append((d, nm(d)))
            return "消しの版（基準を通った 8 版）の当たりの数を、前日版の点から引く（消しの決め方）", out
        for _ in range(20000):
            ks = rng.choice(X, 2, replace=False)
            d = {"score_pre": 1.0, str(rng.choice(KH)): -float(rng.choice([1, 2, 5]))} | {k: float(rng.normal(0, 1.5)) for k in ks}
            d = {k: round(v, 2) for k, v in d.items()}
            out.append((d, nm(d)))
        return "前日版の点 − 消しの当たりの数 ＋ 型 2 つの連続な重み（20,000 通り）", out
    if c <= 20:
        out = []
        n = int(rng.integers(6, 12))
        for _ in range(20000):
            ks = rng.choice(X, n, replace=False)
            d = {k: round(float(rng.normal(0, 1)), 2) for k in ks}
            out.append((d, nm(d)))
        return f"型 {n} つの連続でランダムな重み（20,000 通り）", out
    raise ValueError(c)


def matrix(t: Table, cand: list[tuple[dict, str]]) -> tuple[np.ndarray, list[str]]:
    return np.stack([_vec(t, d) for d, _ in cand], axis=1), [n for _, n in cand]


def goal_met(rep: dict) -> dict:
    """ゴール・サブゴールを満たしたか。重賞ごと（A）: 全重賞で ◎ が 30%・120% を達成し、全体の値も 30%・120% 以上。
    全体で 1 つ（A）: 全重賞に同じ決め方で 30%・120% 以上。サブゴール: 消した馬の 95% 以上が 4 着以下で、1 レース 3 頭以上を消す。
    止めるのは、重賞ごと・全体で 1 つの両方で、ゴールとサブゴールを満たしたとき（「全重賞で」の 2 つの読み方の両方）。"""
    a = rep["A"]
    h, k = a["honmei_by_rkey"], a["kill_by_rkey"]
    g, kg = a["honmei_global"]["val"], a["kill_global"]["val"]
    out = {"goal_rkey": h["achieved"] == rep["n_rkeys"] and h["all"]["win"] >= WIN_GOAL and h["all"]["roi_t"] >= ROI_GOAL,
           "sub_rkey": k["achieved"] == rep["n_rkeys"] and k["all"]["precision"] >= KILL_GOAL and k["all"]["killed_per_race"] >= KILL_MIN,
           "goal_global": g["win"] >= WIN_GOAL and g["roi_t"] >= ROI_GOAL,
           "sub_global": kg["precision"] >= KILL_GOAL and kg["killed_per_race"] >= KILL_MIN}
    out = {k: bool(v) for k, v in out.items()}
    out["all"] = all(out.values())
    return out


# ------------------------------------------------------------------ 前向きの記録（C）
def rule_of(label: str, terms: dict[str, dict]) -> dict:
    """決め方の名前（消しは「…・下位 k 頭」）から、重みと消す頭数を返す。"""
    k = None
    if label.endswith(" 頭") and "・下位 " in label:
        label, ks = label.rsplit("・下位 ", 1)
        k = int(ks.removesuffix(" 頭"))
    return {"label": label, "terms": terms[label], "k": k}


def apply_rule(t: Table, rule: dict, races: np.ndarray) -> dict:
    """1 つの規則をレース（真偽）に当てはめる。◎ の規則は ◎ の馬番・着順・単勝、消しの規則は消した馬番・4 着以下の数。"""
    w = _vec(t, rule["terms"])[:, None]
    out = []
    if rule["k"] is None:
        H = honmei(t, w)[:, 0]
        for i in np.flatnonzero(races):
            h = H[i]
            out.append({"race_id": t.race["race_id"].iloc[i], "hn": int(t.r["hn"].iloc[h]), "fin": float(t.r["fin"].iloc[h]),
                        "win": float(t.y1[h]), "ret_tan": float(t.rt[h]), "pay_ok": bool(t.race["pay_ok"].iloc[i])})
    else:
        Sc = (t.F @ w)[:, 0]
        ends = np.r_[t.starts[1:], len(Sc)]
        for i in np.flatnonzero(races):
            a, b = t.starts[i], ends[i]
            kk = max(0, min(rule["k"], (b - a) - 3))
            o = (b - a - 1) - np.argsort(Sc[a:b][::-1], kind="stable")[:kk] + a
            out.append({"race_id": t.race["race_id"].iloc[i], "killed": [int(x) for x in t.r["hn"].to_numpy()[o]], "bad": int(t.bad[o].sum()), "n": int(kk)})
    return {"races": out}
