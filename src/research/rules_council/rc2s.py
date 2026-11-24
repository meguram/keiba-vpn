"""RC-2 の版 2（1 号の決定「RC-2 の段 B の判断」）: 段 S' の 2 段の門と、その検出力の模擬（合成だけ・着順なし）。

段 S'（2016〜21 の全体で 1 回）:
- 1 段目: 水準ごとの帯の比に対する z（z_band）が、帰無の max-T の 95% 点以上（増やす向き）／ −z_band が同じ点以上（消し、記録だけ）。
- 2 段目: 1 段目を通った k 個だけで、損益分岐の z（z_be）が Holm の線（片側 α = 0.05）を越える。
- 候補: 2 段目を通り、前向きの量（2024 年の走の数、着順なし）≥ 900。再現 5 本・5 本すべて内の線は本物の段 S' で当てる（模擬は種 1000 の 1 本）。
段 F（模擬では 2022〜24 にあたる合成の 3 年）: 候補（z_be の上位 3 つまで）で買い、回収率 ≥ 1.00・帯の比（2016〜21 で出す）に対する市場比の片側 p ≤ 0.01（開催で束ねたブートストラップ 2,000 回）・点数 ≥ 300。
模擬の腕: 下限 E ≥ 200 と E ≥ 1,000。植える: 回収率 1.00・1.05・1.10・1.15。各 40 回、効きなし 200 回。
関門（腕ごと）: ① 回収率 1.10 で植えた水準が候補 ≥ 50% ② 回収率 1.00 で植えた水準が候補 ≤ 5% ③ 効きなしで 1 つでも候補 ≤ 5% ④ 回収率 1.10 で段 S' と段 F を両方 ≥ 30%。

2022〜24 の本物の着順は、この道具では読み込んだ直後に消す（合成の勝ちだけを使う）。合成の元の帯の比は 2016〜21 の本物の 勝ち ÷ q（帯 × クラス群）。
使い方: python -m src.research.rules_council.rc2s power --reps 40 --null 200（nice -n 19）
出力: <COURSE_OUT>/rules_council/rc2s_power.json。費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import norm

from src.research.rules_council import engine as E
from src.research.rules_council import rc_frame as F

PRE_HI = 10.0
SEARCH_END = 2021
FWD_YEAR, FWD_MIN = 2024, 900
ALPHA2 = 0.05
F_ALPHA, F_N_MIN, F_MAX_CAND = 0.01, 300, 3
EXCLUDE_T60 = ("G06", "G07", "H07")   # 発走前（T-60）に分からないことがある型。検定に入れない（prereg）


def holm_pass(z: np.ndarray, alpha: float) -> np.ndarray:
    """片側の Holm（step-down）。通った位置の真偽。"""
    k = len(z)
    if k == 0:
        return np.zeros(0, bool)
    p = norm.sf(z)
    order = np.argsort(p)
    ok = np.zeros(k, bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (k - i):
            ok[j] = True
        else:
            break
    return ok


class World:
    def __init__(self, emins=(200.0, 1000.0), mode: str = "k70"):
        """mode = k70（版 3: 2016〜24 の開催の 70% で探し、30% で確かめる。統括役の指示）か year21（版 2: 2016〜21 → 2022〜24）。
        確かめの側の本物の着順・払戻は、読み込んだ直後に消す（合成の勝ちだけを使う）。"""
        P = E.load_pop(0.0, PRE_HI, pre_seed=0, outcome_until=SEARCH_END if mode == "year21" else None, exclude_types=EXCLUDE_T60)
        d = P.d
        self.year = d["year"].astype(int).to_numpy()
        fwd = (self.year > SEARCH_END) if mode == "year21" else ~E.search70(d["kaisai"].to_numpy())
        for c in E.OUTCOME_COLS:
            if c in d:
                d[c] = d[c].astype(float)
                d.loc[fwd, c] = np.nan
        self.P = P
        self.flat = P.flat()
        self.q = d["q"].to_numpy()
        self.odds = d["odds_final"].to_numpy(float)
        self.kidx = d["kidx"].to_numpy()
        self.hidx = d["hidx"].to_numpy()
        self.robust = mode == "k70"
        self.S = np.flatnonzero(~fwd)
        self.Fr = np.flatnonzero(fwd)
        self.s_mask = ~fwd
        (self.Eq,) = E.level_sums(self.flat, self.S, [self.q], P.n_levels)
        (self.fwd_n,) = E.level_sums(self.flat, np.flatnonzero(self.year == FWD_YEAR), [np.ones(len(d))], P.n_levels)
        self.elig = {e: np.flatnonzero(self.Eq >= e) for e in emins}
        real_win = d["win"].to_numpy(float).copy()
        real_win[fwd] = 0.0
        self.rho0 = E.band_ratio(d, self.s_mask, real_win)    # 探す側の本物の帯の比（合成の元）
        self.sim = E.Sim(P, self.rho0)

    def zs(self, w: np.ndarray, rt: np.ndarray) -> tuple[dict, np.ndarray]:
        rho = E.band_ratio(self.P.d, self.s_mask, np.nan_to_num(w))
        e = rho * self.q
        W, Eb, V, N, RT, O1 = E.level_sums(self.flat, self.S, [w, e, e * (1 - e), np.ones(len(w)), rt, self.odds - 1.0], self.P.n_levels)
        if self.robust:     # 版 3: 馬で束ねた分散との大きい方（本物の段 S' と同じ作り）
            Cb, Cw = E.level_cluster_ss(self.flat, self.S, [np.nan_to_num(rt) - 1.0, np.nan_to_num(w) - e], self.hidx, self.P.n_levels)
            O1, V = np.maximum(O1, Cb), np.maximum(V, Cw)
        return {"z_band": (W - Eb) / np.sqrt(np.maximum(V, 1e-9)), "z_be": (RT - N) / np.sqrt(np.maximum(O1, 1e-9))}, rho

    def stage_s(self, z: dict, emin: float, t_up: float, t_dn: float) -> dict:
        el = self.elig[emin]
        up1 = el[z["z_band"][el] >= t_up]
        dn1 = el[-z["z_band"][el] >= t_dn]
        ok2 = holm_pass(z["z_be"][up1], ALPHA2)
        cand = up1[ok2]
        cand = cand[self.fwd_n[cand] >= FWD_MIN]
        cand = cand[np.argsort(-z["z_be"][cand])]
        return {"up1": up1, "down1": dn1, "cand": cand}

    def stage_f(self, cand: np.ndarray, w: np.ndarray, rt: np.ndarray, rho: np.ndarray, rng: np.random.Generator) -> dict:
        frozen = cand[:F_MAX_CAND]
        if not len(frozen):
            return {"pass": False, "n": 0}
        sel = np.zeros(self.P.n_levels, bool)
        sel[frozen] = True
        f = self.flat[self.Fr]
        hit = np.where(f >= 0, sel[np.maximum(f, 0)], False).any(1)
        b = self.Fr[hit]
        n = len(b)
        if n == 0:
            return {"pass": False, "n": 0}
        roi = float(rt[b].sum() / n)
        e = rho[b] * self.q[b]
        k = np.unique(self.kidx[b], return_inverse=True)[1]
        nk = int(k.max()) + 1
        A = np.bincount(k, weights=w[b], minlength=nk)
        B = np.bincount(k, weights=e, minlength=nk)
        idx = rng.integers(0, nk, size=(2000, nk))
        r = A[idx].sum(1) / B[idx].sum(1)
        p = float((np.sum(r <= 1.0) + 1) / (len(r) + 1))
        return {"pass": bool(roi >= 1.0 and p <= F_ALPHA and n >= F_N_MIN), "n": n, "roi": roi, "p": p, "frozen": [int(x) for x in frozen]}


def cmd_power(reps: int, n_null: int, seed: int, mode: str = "k70") -> None:
    t0 = time.time()
    Wd = World(mode=mode)
    rng = np.random.default_rng(seed)
    out = {"mode": mode, "n_search": int(len(Wd.S)), "n_forward": int(len(Wd.Fr)), "reps": reps, "n_null": n_null,
           "n_elig": {str(int(e)): int(len(v)) for e, v in Wd.elig.items()}, "arms": {}}

    def draw(mult):
        w, _ = Wd.sim.draw(mult, rng)
        return w, w * Wd.odds

    nulls = []
    for i in range(n_null):
        E.check_stop()
        w, rt = draw(None)
        z, rho = Wd.zs(w, rt)
        nulls.append((z, rho, w, rt))
    thr = {}
    for e, el in Wd.elig.items():
        mu = np.array([np.max(z["z_band"][el]) for z, *_ in nulls])
        md = np.array([np.max(-z["z_band"][el]) for z, *_ in nulls])
        thr[e] = (mu, md)
        fw = {"cand": 0, "down1": 0, "up1": 0}
        for i, (z, rho, w, rt) in enumerate(nulls):           # 1 つ抜きの帰無の線で当てる
            r = Wd.stage_s(z, e, np.quantile(np.delete(mu, i), 0.95), np.quantile(np.delete(md, i), 0.95))
            fw["cand"] += len(r["cand"]) > 0; fw["down1"] += len(r["down1"]) > 0; fw["up1"] += len(r["up1"]) > 0
        out["arms"][str(int(e))] = {"t_up": float(np.quantile(mu, 0.95)), "t_down": float(np.quantile(md, 0.95)),
                                    "fwer": {k: v / n_null for k, v in fw.items()}, "plant": {}}
    print("null done", {k: (v["t_up"], v["fwer"]) for k, v in out["arms"].items()}, round(time.time() - t0), flush=True)
    d = Wd.P.d
    for roi_t in (1.00, 1.05, 1.10, 1.15):
        for e, el in Wd.elig.items():
            A = out["arms"][str(int(e))]
            hits = {"cand": 0, "up1": 0, "cand_and_F": 0, "any_cand": 0}
            ms, es = [], []
            miss = 0
            for _ in range(reps):
                E.check_stop()
                j = int(rng.choice(el))
                rows = np.flatnonzero((Wd.flat == j).any(1))
                rows_s = rows[Wd.year[rows] <= SEARCH_END]
                m, got = Wd.sim.calibrate(rows_s, Wd.odds, roi_t)     # 2016〜21 の期待の回収率で合わせ、同じ倍率を 2022〜24 にも掛ける
                if abs(got - roi_t) > 0.005:
                    miss += 1
                mult = np.ones(len(d))
                mult[rows] = m
                w, rt = draw(mult)
                z, rho = Wd.zs(w, rt)
                r = Wd.stage_s(z, e, A["t_up"], A["t_down"])
                c = j in r["cand"]
                hits["cand"] += c; hits["up1"] += j in r["up1"]; hits["any_cand"] += len(r["cand"]) > 0
                if c:
                    fr = Wd.stage_f(r["cand"], w, rt, rho, rng)
                    hits["cand_and_F"] += fr["pass"] and j in fr["frozen"]
                ms.append(m); es.append(float(Wd.Eq[j]))
            A["plant"][f"{roi_t:.2f}"] = {k: v / reps for k, v in hits.items()} | {"mult_median": float(np.median(ms)), "E_median": float(np.median(es)), "calib_miss": miss}
            print(e, roi_t, A["plant"][f"{roi_t:.2f}"], round(time.time() - t0), flush=True)
    for e, A in out["arms"].items():
        P_ = A["plant"]
        A["gate"] = {"1_roi110_cand_ge_0.5": P_["1.10"]["cand"] >= 0.5, "2_roi100_cand_le_0.05": P_["1.00"]["cand"] <= 0.05,
                     "3_fwer_le_0.05": A["fwer"]["cand"] <= 0.05, "4_roi110_S_and_F_ge_0.3": P_["1.10"]["cand_and_F"] >= 0.3}
        A["gate"]["pass"] = all(A["gate"].values())
        A["se_gate1"] = float(np.sqrt(P_["1.10"]["cand"] * (1 - P_["1.10"]["cand"]) / reps))
    passed = [e for e, A in out["arms"].items() if A["gate"]["pass"]]
    if len(passed) == 2:
        a, b = out["arms"]["200"]["plant"]["1.10"]["cand"], out["arms"]["1000"]["plant"]["1.10"]["cand"]
        se = max(out["arms"]["200"]["se_gate1"], out["arms"]["1000"]["se_gate1"])
        out["chosen_arm"] = "1000" if (b >= a or abs(a - b) <= se) else "200"
    else:
        out["chosen_arm"] = passed[0] if passed else None
    out["seconds"] = round(time.time() - t0)
    (F.OUTW / ("rc2s_power.json" if mode == "year21" else "rc2s_power_k70.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("power")
    s.add_argument("--reps", type=int, default=40)
    s.add_argument("--null", type=int, default=200)
    s.add_argument("--seed", type=int, default=20261126)
    s.add_argument("--mode", choices=["k70", "year21"], default="k70")
    a = ap.parse_args()
    cmd_power(a.reps, a.null, a.seed, a.mode)


if __name__ == "__main__":
    main()
