"""RC-2（議題 1）: 一桁オッズ（T-60 の再現 < 10 倍）・全クラス・全体の単位・深さ 1・単勝・2 方向（1 号の決定 RC-0）。

段 S（2016〜21 の 1 回だけ）の作り:
- 検定する水準: 2016〜21 の母集団で E（確定の q の和）≥ 200 の深さ 1 の水準（着順を使わずに決まる）。
- 増やす向き: 損益分岐の z = Σ(払戻 − 1) ÷ √Σ(確定オッズ − 1) ≥ Z（Bonferroni、検定の数 = 水準 × 2 方向）。
- 帯の比の帰無の max-T（行を丸ごと入れ替え）の p ≤ 0.05。
- 5 分割の外側（2016〜21 の中で 4 つで選び残り 1 つに当てる）: 回収率 ≥ 1.00、市場比 ÷ 帯の比 > 1、点数の合計 ≥ 1,000。
- 前向きの量: 2024 年の走の数（着順なし）≥ 900。
- 消しの向き: 帯の比の z ≤ −Z と max-T の p ≤ 0.05 を記録だけ。
段 B（検出力の模擬）: Plackett-Luce の合成で、E ≥ 200 の水準に回収率 1.00・1.05・1.10（と帯の比 ×1.10・1.15・1.25）を植えて、上の線を全部越える確率と、
効きなしで 1 つでも越える確率（FWER）を出す。年で分けた版（2016〜18 で選び 2019〜21 に当てる）も出す。

使い方（nice -n 19。段 B は着順を使わない。帯の比の合成の元には 2016〜21 の帯ごとの 勝ち ÷ q を使う）:
  python -m src.research.rules_council.rc2 power --reps 40 --null 200
出力: <COURSE_OUT>/rules_council/rc2_*.json。費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.rules_council import engine as E
from src.research.rules_council import rc_frame as F

PRE_HI = 10.0
YEAR_END = 2021
EMIN = 200.0
FWD_YEAR, FWD_MIN = 2024, 900
OUTER_N_MIN = 1000
EXCLUDE_T60 = ("G06", "G07", "H07")   # 馬体重・増減（発表は発走の約 70 分前）・馬場状態（T-60 の後に変わりうる）。検定に入れない（prereg）
NULL_EDGES = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0])   # 帰無の層の確定オッズの 9 帯（＋10 倍以上）


def z_line(n_tests: int, alpha: float = 0.05) -> float:
    return float(norm.isf(alpha / n_tests))


class RC2:
    """1 つの再現の種での母集団（2016〜21 と、前向きの量を数える 2024 年の符号）。"""

    def __init__(self, pre_seed: int = 0, mode: str = "year21"):
        """mode = year21（2016〜21 を探す。版 1・版 2 の記録）か k70（2016〜24 の開催の 70% を探す。版 3）。
        探す側の外の行の着順・払戻は読み込んだ直後に消す。"""
        self.mode = mode
        P = E.load_pop(0.0, PRE_HI, pre_seed=pre_seed, outcome_until=YEAR_END if mode == "year21" else None, exclude_types=EXCLUDE_T60)
        yr = P.d["year"].astype(int).to_numpy()
        if mode == "k70":
            srch = E.search70(P.d["kaisai"].to_numpy())
            for c in E.OUTCOME_COLS:
                if c in P.d:
                    P.d[c] = P.d[c].astype(float)
                    P.d.loc[~srch, c] = np.nan
        else:
            srch = yr <= YEAR_END
        self.flat_all = P.flat()
        self.fwd_rows = np.flatnonzero(yr == FWD_YEAR)          # 2024 年は符号だけ使う（着順を読まない）
        keep = np.flatnonzero(srch)
        self.P = E.Pop(P.d.iloc[keep].reset_index(drop=True), P.codes[keep], P.types, P.levels, P.full_rows[keep])
        self.flat = self.P.flat()
        d = self.P.d
        self.q = d["q"].to_numpy()
        self.odds = d["odds_final"].to_numpy(float)
        self.fold = d["fold"].to_numpy()
        self.year = d["year"].astype(int).to_numpy()
        self.kidx = d["kidx"].to_numpy()
        self.hidx = d["hidx"].to_numpy()
        allr = np.arange(len(d))
        (self.Eq,) = E.level_sums(self.flat, allr, [self.q], self.P.n_levels)
        self.elig = np.flatnonzero(self.Eq >= EMIN)
        self.n_tests = 2 * len(self.elig)
        self.Z = z_line(self.n_tests)
        (fwd_n,) = E.level_sums(self.flat_all, self.fwd_rows, [np.ones(len(P.d))], P.n_levels)
        self.fwd_n = fwd_n
        self.null_key = self._null_key()

    # ------------------------------------------------------------ 帰無の層
    def _era(self) -> np.ndarray:
        """時代: year21 は 2016〜18・19〜21 の 2 つ、k70 は 2016〜18・19〜21・22〜24 の 3 つ。"""
        if self.mode == "k70":
            return np.select([self.year <= 2018, self.year <= 2021], [0, 1], 2)
        return (self.year >= 2019).astype(int)

    def _null_key(self) -> np.ndarray:
        d = self.P.d
        ob = np.searchsorted(NULL_EDGES, self.odds, side="right")
        surf = (d["surf"].astype(str) == "芝").astype(int).to_numpy()
        era = self._era()
        cg = d["cgrp"].to_numpy()
        fg = d["fgrp"].to_numpy()
        full = ((ob * 5 + cg) * 3 + era) * 2 * 3 + surf * 3 + fg
        k1 = ((ob * 5 + cg) * 3 + era) * 2 + surf + 10_000_000          # 頭数をまとめる
        k2 = (ob * 5 + cg) * 3 + era + 20_000_000                        # さらに芝ダをまとめる
        key = full.copy()
        n = pd.Series(full).map(pd.Series(full).value_counts()).to_numpy()
        key[n < 30] = k1[n < 30]
        n1 = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
        key[n1 < 30] = k2[n1 < 30]
        return key

    def permute(self, seed: int) -> np.ndarray:
        """水準の行を丸ごと層の中で入れ替える。レースの型は クラス群 × 時代 × 芝ダ の中でレースの束ごと。通し番号の flat を返す。"""
        rng = np.random.default_rng(seed)
        P = self.P
        d = P.d
        perm = np.arange(len(d))
        for idx in pd.Series(np.arange(len(d))).groupby(self.null_key).indices.values():
            if len(idx) > 1:
                perm[idx] = idx[rng.permutation(len(idx))]
        rt = [j for j, t in enumerate(P.types) if t in E.RACE_LEVEL_TYPES]
        ht = [j for j in range(len(P.types)) if j not in rt]
        out = self.flat.copy()
        out[:, ht] = self.flat[perm][:, ht]
        if rt:
            ridx = d["ridx"].to_numpy()
            first = np.r_[0, np.flatnonzero(np.diff(ridx)) + 1]
            rk = ((d["cgrp"].to_numpy() * 3 + self._era()) * 2 + (d["surf"].astype(str) == "芝").to_numpy())[first]
            src = first.copy()
            for idx in pd.Series(np.arange(len(first))).groupby(rk).indices.values():
                if len(idx) > 1:
                    src[idx] = first[idx[rng.permutation(len(idx))]]
            assert np.array_equal(ridx[first], np.arange(len(first))), "行がレースの順に並んでいない"
            rmap = np.empty(int(ridx.max()) + 1, int)
            rmap[np.arange(len(first))] = src
            out[:, rt] = self.flat[rmap[ridx]][:, rt]
        return out

    # ------------------------------------------------------------ 数え上げ
    def stats(self, flat: np.ndarray, rows: np.ndarray, win: np.ndarray, rt: np.ndarray, rho: np.ndarray, robust: bool = False) -> dict:
        """robust = True なら、分散を「モデルの分散」と「馬で束ねた分散 Σ_馬 (Σ x)²」の大きい方にする（版 3。同じ馬の繰り返しを入れる）。"""
        e = rho * self.q
        W, Eb, V, N, RT, Ob1 = E.level_sums(flat, rows, [win, e, e * (1 - e), np.ones(len(win)), rt, self.odds - 1.0], self.P.n_levels)
        if robust:
            Cb, Cw = E.level_cluster_ss(flat, rows, [rt - 1.0, win - e], self.hidx, self.P.n_levels)
            Ob1, V = np.maximum(Ob1, Cb), np.maximum(V, Cw)
        z_be = (RT - N) / np.sqrt(np.maximum(Ob1, 1e-9))
        z_band = (W - Eb) / np.sqrt(np.maximum(V, 1e-9))
        return {"W": W, "Eb": Eb, "N": N, "RT": RT, "z_be": z_be, "z_band": z_band}

    def rho(self, rows: np.ndarray, win: np.ndarray) -> np.ndarray:
        s = np.zeros(len(win), bool)
        s[rows] = True
        return E.band_ratio(self.P.d, s, win)

    def select(self, st: dict) -> tuple[np.ndarray, np.ndarray]:
        el = self.elig
        up = el[st["z_be"][el] >= self.Z]
        down = el[st["z_band"][el] <= -self.Z]
        return up, down

    def max_t(self, st: dict) -> float:
        return float(np.max(st["z_band"][self.elig]))

    def outer(self, flat: np.ndarray, win: np.ndarray, rt: np.ndarray, split: str = "cv") -> dict:
        """外側の成績。split = cv（2016〜21 の中の開催 5 分割）か year（2016〜18 で選び 2019〜21 に当てる）。
        水準ごとの外側の値（選ばれた分割だけ足す）と、選んだ水準の和集合で買った値。"""
        parts = [(self.fold != k, self.fold == k) for k in range(5)] if split == "cv" else [(self.year <= 2018, self.year >= 2019)]
        nl = self.P.n_levels
        per = {k: np.zeros(nl) for k in ("N", "RT", "W", "Eb")}
        U = {"N": 0.0, "RT": 0.0, "W": 0.0, "Eb": 0.0}
        bet = np.zeros(len(win), bool)
        for tr, ho in parts:
            trr, hor = np.flatnonzero(tr), np.flatnonzero(ho)
            rho = self.rho(trr, win)
            up, _ = self.select(self.stats(flat, trr, win, rt, rho))
            if not len(up):
                continue
            e = rho * self.q
            sh = self.stats(flat, hor, win, rt, rho)
            for k in per:
                per[k][up] += sh[k][up]
            sel = np.zeros(nl, bool)
            sel[up] = True
            f = flat[hor]
            hit = (np.where(f >= 0, sel[np.maximum(f, 0)], False)).any(1)
            b = hor[hit]
            bet[b] = True
            U["N"] += len(b); U["RT"] += rt[b].sum(); U["W"] += win[b].sum(); U["Eb"] += e[b].sum()
        return {"per": per, "union": U, "bet": bet}

    def evaluate(self, win: np.ndarray, rt: np.ndarray, maxt_null: np.ndarray, flat: np.ndarray | None = None,
                 splits=("cv", "year")) -> dict:
        """段 S の線を全部当てる。maxt_null は帯の比の帰無の max-T の分布（並べ替えか合成で作った値）。"""
        flat = self.flat if flat is None else flat
        allr = np.arange(len(win))
        rho = self.rho(allr, win)
        st = self.stats(flat, allr, win, rt, rho)
        up, down = self.select(st)
        res = {"n_up": int(len(up)), "n_down": int(len(down)), "max_t": self.max_t(st)}
        p_up = np.array([(1 + np.sum(maxt_null >= st["z_band"][j])) / (1 + len(maxt_null)) for j in up])
        p_dn = np.array([(1 + np.sum(maxt_null >= -st["z_band"][j])) / (1 + len(maxt_null)) for j in down])  # 対称とみる
        res["down_pass"] = [int(j) for j, p in zip(down, p_dn) if p <= 0.05]
        for sp in splits:
            o = self.outer(flat, win, rt, sp)
            U = o["union"]
            res[f"union_{sp}"] = {"N": U["N"], "roi": U["RT"] / U["N"] if U["N"] else None,
                                  "mr_band": U["W"] / U["Eb"] if U["Eb"] else None}
            passed = []
            for j, p in zip(up, p_up):
                n = o["per"]["N"][j]
                ok = (p <= 0.05 and n >= OUTER_N_MIN and o["per"]["RT"][j] / max(n, 1) >= 1.0
                      and o["per"]["W"][j] / max(o["per"]["Eb"][j], 1e-9) > 1.0 and self.fwd_n[j] >= FWD_MIN)
                if ok:
                    passed.append(int(j))
            res[f"pass_{sp}"] = passed
        res["up"] = [int(j) for j in up]
        res["p_up"] = [float(x) for x in p_up]
        return res


# ---------------------------------------------------------------- 段 B 検出力の模擬
def synth_base_rho(R: RC2) -> np.ndarray:
    """合成の元の帯の比（2016〜21 の 帯 × クラス群 の 勝ち ÷ q。母集団の記述の値で、条件の選び方には使わない）。"""
    win = R.P.d["win"].to_numpy(float)
    return R.rho(np.arange(len(win)), win)


def cmd_power(reps: int, n_null: int, seed: int) -> None:
    t0 = time.time()
    R = RC2(0)
    rho0 = synth_base_rho(R)
    sim = E.Sim(R.P, rho0)
    rng = np.random.default_rng(seed)
    out = {"n_rows": int(len(R.q)), "n_elig": int(len(R.elig)), "n_tests": R.n_tests, "Z": R.Z, "reps": reps, "n_null": n_null}
    # 効きなし: max-T の分布と FWER
    nulls = []
    null_runs = []
    for i in range(n_null):
        E.check_stop()
        w, _ = sim.draw(None, rng)
        rt = w * R.odds
        st = R.stats(R.flat, np.arange(len(w)), w, rt, R.rho(np.arange(len(w)), w))
        nulls.append(R.max_t(st))
        null_runs.append((w, rt))
    nulls = np.array(nulls)
    out["null_maxt_q95"] = float(np.quantile(nulls, 0.95))
    fw = {"cv": 0, "year": 0, "down": 0, "up_selected": 0}
    # FWER は、帰無の分布を自分以外の帰無で作って当てる（自分を含めると甘くなるので 1 つ抜き）
    for i, (w, rt) in enumerate(null_runs[: min(n_null, 200)]):
        r = R.evaluate(w, rt, np.delete(nulls, i))
        fw["cv"] += bool(r["pass_cv"]); fw["year"] += bool(r["pass_year"]); fw["down"] += bool(r["down_pass"]); fw["up_selected"] += r["n_up"] > 0
    nfw = min(n_null, 200)
    out["fwer"] = {k: v / nfw for k, v in fw.items()}
    print("null done", out["null_maxt_q95"], out["fwer"], round(time.time() - t0), flush=True)
    # 植える: 回収率の目標（1.00・1.05・1.10）と、帯の比に対する倍率（1.10・1.15・1.25）
    d = R.P.d
    allr = np.arange(len(d))
    e0 = rho0 * R.q
    _, Eb0, N0, BE0 = E.level_sums(R.flat, allr, [np.zeros(len(d)), e0, np.ones(len(d)), e0 * R.odds], R.P.n_levels)
    plans = [("roi", 1.00), ("roi", 1.05), ("roi", 1.10), ("mult", 1.10), ("mult", 1.15), ("mult", 1.25)]
    out["plant"] = {}
    for kind, val in plans:
        hits = {"cv": 0, "year": 0, "selected": 0}
        mults, sizes = [], []
        for _ in range(reps):
            E.check_stop()
            j = int(rng.choice(R.elig))
            rows = np.flatnonzero((R.flat == j).any(1))
            target = val if kind == "roi" else val * BE0[j] / N0[j]     # 帯の比 ×val は、回収率 val × 帯の比のときの回収率
            m, _ = sim.calibrate(rows, R.odds, target)                 # レースの中で直した後に目標になる倍率
            mult = np.ones(len(d))
            mult[rows] = m
            w, _ = sim.draw(mult, rng)
            rt = w * R.odds
            r = R.evaluate(w, rt, nulls)
            hits["cv"] += j in r["pass_cv"]; hits["year"] += j in r["pass_year"]; hits["selected"] += j in r["up"]
            mults.append(m); sizes.append(float(R.Eq[j]))
        out["plant"][f"{kind}_{val:.2f}"] = {k: v / reps for k, v in hits.items()} | {
            "mult_median": float(np.median(mults)), "E_median": float(np.median(sizes))}
        print(kind, val, out["plant"][f"{kind}_{val:.2f}"], round(time.time() - t0), flush=True)
    out["gate"] = {"roi_1.10_cv_ge_0.5": out["plant"]["roi_1.10"]["cv"] >= 0.5, "roi_1.10_year_ge_0.5": out["plant"]["roi_1.10"]["year"] >= 0.5,
                   "fwer_cv_le_0.05": out["fwer"]["cv"] <= 0.05, "fwer_year_le_0.05": out["fwer"]["year"] <= 0.05}
    out["gate"]["pass"] = all(out["gate"].values())
    out["seconds"] = round(time.time() - t0)
    F.OUTW.mkdir(parents=True, exist_ok=True)
    (F.OUTW / "rc2_power.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("power")
    s.add_argument("--reps", type=int, default=40)
    s.add_argument("--null", type=int, default=200)
    s.add_argument("--seed", type=int, default=20261125)
    a = ap.parse_args()
    if a.cmd == "power":
        cmd_power(a.reps, a.null, a.seed)


if __name__ == "__main__":
    main()
