"""tp-62 共通部品: 月次過半・年間 100% 超の馬券戦略のウォークフォワード探索。

入力は tp-50 のチケット表（1 行 = レース × 買い目。ret は 100 円あたりの払戻倍率、odds は確定オッズ）。
戦略 = 買い目ルール（券種 × スコア × 閾値 × オッズ帯 × レース内上位 K 点）× 賭け方（定額 / レース均等）× レース絞り込み。
評価は『その年より前の年だけで選び、その年に適用』（ウォークフォワード）。帰無（市場が正しい）の下で同じ選び方をした結果を、
市場の確率から着順を乱数で作って再現する（simulate_ret）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TP50 = Path(os.environ.get("TP50_OUT", str(ROOT / "data/local/modeling/betting/tp50")))
OUT = Path(os.environ.get("TP62_OUT", "/home/jovyan/work/tp62_out"))
BT_NAMES = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrentan"]
RATE = np.array([0.8, 0.8, 0.775, 0.775, 0.75, 0.725])
THETAS = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)
KS = (1, 2, 3, 5)
SCORES = ("ev_mix", "ev_model")
BAND_EDGES = {
    0: (3.0, 10.0, 30.0),
    1: (2.0, 4.0, 8.0),
    2: (30.0, 100.0, 300.0),
    3: (10.0, 30.0, 100.0),
    4: (100.0, 300.0, 1000.0),
    5: (300.0, 1000.0, 5000.0),
}
MIN_RACES_PER_YEAR = 40
Z90 = 1.2816
COLS = ["race_id", "date", "year", "n", "r_p1", "r_kl", "r_ent", "r_q1", "r_dom", "r_even", "r_upset", "bt", "odds", "p_model", "p_mix", "q", "ret", "h1", "h2", "h3"]


def band_bounds(bt: int, band: int) -> tuple[float, float]:
    e = BAND_EDGES[bt]
    lo = (0.0, 0.0, e[0], e[1], e[2])[band]
    hi = (np.inf, e[0], e[1], e[2], np.inf)[band]
    return lo, hi


def attach_overlay(d: pd.DataFrame) -> pd.DataFrame:
    """tp62_overlay の旗（res2/ex2）を買い目の馬ごとに引き、h1 だけ（anchor）/ 全馬（all）の 4 列を付ける。"""
    o = pd.read_parquet(OUT / "overlay.parquet")
    key = pd.MultiIndex.from_arrays([o["race_id"].astype(str), o["horse_number"].astype(int)])
    for col in ("res2", "ex2"):
        vals = o[col].to_numpy()
        flags = []
        for h in ("h1", "h2", "h3"):
            hv = d[h].to_numpy()
            ix = key.get_indexer(pd.MultiIndex.from_arrays([d["race_id"].astype(str), hv]))
            f = np.where(ix >= 0, vals[np.maximum(ix, 0)], 0).astype(bool)
            flags.append(np.where(hv > 0, f, True))
        d[f"{col}_h1"] = flags[0]
        d[f"{col}_all"] = flags[0] & flags[1] & flags[2]
    return d


def load_tickets(years=(2022, 2023, 2024, 2025, 2026), max_races: int = 0, overlay: bool = False) -> pd.DataFrame:
    parts = []
    for y in years:
        f = TP50 / f"tickets_{y}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=COLS))
    return prepare(pd.concat(parts, ignore_index=True), max_races, overlay)


def prepare(d: pd.DataFrame, max_races: int = 0, overlay: bool = False) -> pd.DataFrame:
    d["ev_mix"] = (d["p_mix"] * d["odds"]).astype(np.float32)
    d["ev_model"] = (d["p_model"] * d["odds"]).astype(np.float32)
    d = d.sort_values(["date", "race_id", "bt"], kind="stable").reset_index(drop=True)
    d["ridx"] = pd.factorize(d["race_id"])[0].astype(np.int32)
    if max_races:
        d = d[d["ridx"] % max_races == 0].copy()          # 動作確認用: max_races 件おきに間引く
        d["ridx"] = pd.factorize(d["race_id"])[0].astype(np.int32)
    if overlay:
        d = attach_overlay(d)
    return d


@dataclass
class Races:
    race_id: np.ndarray
    date: np.ndarray
    year: np.ndarray
    month: np.ndarray        # 全期間通しの月番号 0..M-1
    month_label: list
    month_year: np.ndarray   # 月番号 → 年
    feats: pd.DataFrame
    masks: np.ndarray        # [F, Rn] bool
    mask_names: list


def build_races(d: pd.DataFrame) -> Races:
    first = d.groupby("ridx", sort=True).head(1)
    f = first[["race_id", "date", "year", "n", "r_p1", "r_kl", "r_ent", "r_q1", "r_dom", "r_even", "r_upset"]].reset_index(drop=True)
    ym = f["date"].astype(str).str.slice(0, 7)
    labels = sorted(ym.unique())
    month = ym.map({l: i for i, l in enumerate(labels)}).to_numpy()
    my = np.array([int(l[:4]) for l in labels])
    ref = f[f["year"] <= 2025]
    q = {c: np.quantile(ref[c], [1 / 3, 2 / 3]) for c in ("r_p1", "r_kl", "r_ent", "r_q1")}
    names, masks = [], []

    def add(nm, m):
        names.append(nm)
        masks.append(np.asarray(m, bool))

    n = f["n"].to_numpy()
    add("all", np.ones(len(f), bool))
    add("n<=9", n <= 9)
    add("n10-13", (n >= 10) & (n <= 13))
    add("n>=14", n >= 14)
    for c, lab in (("r_p1", "モデル本命の強さ"), ("r_kl", "モデルと市場の乖離"), ("r_ent", "混戦度(エントロピー)"), ("r_q1", "市場1番人気の強さ")):
        v = f[c].to_numpy()
        add(f"{lab}:低", v <= q[c][0])
        add(f"{lab}:高", v >= q[c][1])
    add("荒れ型(upset)", f["r_upset"].to_numpy() == 1)
    add("非荒れ型", f["r_upset"].to_numpy() == 0)
    add("突出型(dom)", f["r_dom"].to_numpy() == 1)
    add("拮抗型(even)", f["r_even"].to_numpy() == 1)
    return Races(f["race_id"].to_numpy(), f["date"].astype(str).to_numpy(), f["year"].to_numpy(), month, labels, my, f, np.stack(masks), names)


OV_KINDS = ("none", "res2_h1", "res2_all", "ex2_h1", "ex2_all")
OV_THETAS = (1.0, 1.2, 1.5)


def build_rules(d: pd.DataFrame, overlay: bool = False):
    """ルールごとの選択行（ペア配列）。戻り値: meta(DataFrame), pair_rule(int32), pair_row(int32)。
    overlay=True のときは、騎手・調教師の条件（OV_KINDS）を買い目に重ねたルール（スコアは ev_mix、閾値は OV_THETAS）を加える。"""
    ridx = d["ridx"].to_numpy()
    odds = d["odds"].to_numpy(np.float64)
    meta, pr, pw = [], [], []
    plan = [(s_, "none", THETAS) for s_ in SCORES]
    if overlay:
        plan += [("ev_mix", ov, OV_THETAS) for ov in OV_KINDS[1:]]
    rid = 0
    for bt in sorted(d["bt"].unique()):
        idx = np.flatnonzero(d["bt"].to_numpy() == bt)
        for sname, ov, thetas in plan:
            sc = d[sname].to_numpy()[idx]
            order = np.lexsort((-sc, ridx[idx]))
            rows = idx[order]
            sc_s = sc[order]
            od_s = odds[rows]
            r_s = ridx[rows]
            okov = np.ones(len(rows), bool) if ov == "none" else d[ov].to_numpy()[rows]
            chg = np.flatnonzero(np.diff(r_s, prepend=r_s[0] - 1))
            cnts = np.diff(np.append(chg, len(r_s)))
            seg_start = np.repeat(chg, cnts)
            for th in thetas:
                for band in range(5):
                    lo, hi = band_bounds(int(bt), band)
                    ps = (sc_s >= th) & (od_s >= lo) & (od_s < hi) & okov
                    cs = np.cumsum(ps)
                    base = np.where(seg_start > 0, cs[np.maximum(seg_start - 1, 0)], 0)
                    rank = cs - base
                    for k in KS:
                        sel = rows[ps & (rank <= k)]
                        meta.append({"rule": rid, "bt": int(bt), "score": sname, "ov": ov, "theta": th, "band": band, "k": k})
                        pr.append(np.full(len(sel), rid, np.int32))
                        pw.append(sel.astype(np.int32))
                        rid += 1
    return pd.DataFrame(meta), np.concatenate(pr), np.concatenate(pw)


class Grid:
    """ルール × 賭け方 × レース絞り込みの全戦略に対し、月・年の賭け金と払戻を持つ。"""

    def __init__(self, d: pd.DataFrame, races: Races, meta: pd.DataFrame, pair_rule: np.ndarray, pair_row: np.ndarray):
        self.d, self.races, self.meta = d, races, meta
        self.U = len(meta)
        self.Rn = len(races.race_id)
        self.pair_row = pair_row
        self.key = (pair_rule.astype(np.int64) * self.Rn + d["ridx"].to_numpy()[pair_row]).astype(np.int64)
        cnt = np.bincount(self.key, minlength=self.U * self.Rn).reshape(self.U, self.Rn).astype(np.float32)
        self.cnt = cnt
        self.S = np.concatenate([cnt, (cnt > 0).astype(np.float32)], 0)          # [2U, Rn]  先頭 U が定額、後半 U がレース均等
        self.F = races.masks.shape[0]
        self.M = len(races.month_label)
        self.years = sorted(set(races.year.tolist()))
        self.Y = len(self.years)
        yidx = {y: i for i, y in enumerate(self.years)}
        self.race_year_i = np.array([yidx[y] for y in races.year])
        # 月 × マスク の指示行列
        G = np.zeros((self.Rn, self.M * self.F), np.float32)
        Gy = np.zeros((self.Rn, self.Y * self.F), np.float32)
        for f in range(self.F):
            m = races.masks[f]
            rr = np.flatnonzero(m)
            G[rr, races.month[rr] * self.F + f] = 1.0
            Gy[rr, self.race_year_i[rr] * self.F + f] = 1.0
        self.G, self.Gy = G, Gy
        self.Smon = (self.S @ G).reshape(2 * self.U, self.M, self.F)
        self.Bmon = ((self.S > 0).astype(np.float32) @ G).reshape(2 * self.U, self.M, self.F)
        self.SS = ((self.S * self.S) @ Gy).reshape(2 * self.U, self.Y, self.F)
        bt_ = meta["bt"].to_numpy()
        self.avail = np.zeros((6, self.Y), bool)
        for b in range(6):
            for y in self.years:
                self.avail[b, yidx[y]] = bool(((d["bt"] == b) & (d["year"] == y)).any())
        self.rule_bt = np.concatenate([bt_, bt_])
        self._ys = None
        self.set_returns(d["ret"].to_numpy(np.float64))

    def set_returns(self, ret: np.ndarray) -> None:
        self._ys = None
        rs = np.bincount(self.key, weights=ret[self.pair_row], minlength=self.U * self.Rn).reshape(self.U, self.Rn)
        cnt = np.maximum(self.cnt, 1.0)
        R = np.concatenate([rs, rs / cnt], 0).astype(np.float32)               # 定額: 払戻合計 / 均等: 1 点あたり平均
        self.Rmon = (R @ self.G).reshape(2 * self.U, self.M, self.F)
        self.RR = ((R * R) @ self.Gy).reshape(2 * self.U, self.Y, self.F)
        self.RS = ((R * self.S) @ self.Gy).reshape(2 * self.U, self.Y, self.F)

    # ---- 集計 ----
    def year_sums(self):
        if self._ys is not None:
            return self._ys
        my = self.races.month_year
        yi = np.array([self.years.index(y) for y in my])
        S = np.zeros((2 * self.U, self.Y, self.F))
        R = np.zeros_like(S)
        B = np.zeros_like(S)
        for m in range(self.M):
            S[:, yi[m]] += self.Smon[:, m]
            R[:, yi[m]] += self.Rmon[:, m]
            B[:, yi[m]] += self.Bmon[:, m]
        self._ys = (S, R, B)
        return self._ys

    def describe(self, u: int, f: int) -> str:
        i = u % self.U
        m = self.meta.iloc[i]
        mode = "定額" if u < self.U else "レース均等"
        lo, hi = band_bounds(int(m["bt"]), int(m["band"]))
        hi_s = "∞" if np.isinf(hi) else f"{hi:g}"
        ov = "" if m["ov"] == "none" else f"|騎手調教師:{m['ov']}"
        return f"{BT_NAMES[int(m['bt'])]}|{m['score']}>={m['theta']:g}{ov}|odds[{lo:g},{hi_s})|上位{int(m['k'])}点|{mode}|{self.races.mask_names[f]}"


def select(grid: Grid, train_years: list[int], mode: str, nmin: int = MIN_RACES_PER_YEAR):
    """train_years の情報だけで戦略を 1 つ選ぶ。戻り値 (u, f, info) または None。
    mode: 'A' 下側信頼限界の最大 / 'B' 全訓練年で 100% 超かつ月の勝ち越しを条件に下限最大 / 'C' 訓練回収率の最大（過学習の見本）"""
    S, R, B = grid.year_sums()
    yi = [grid.years.index(y) for y in train_years]
    St, Rt, Bt = S[:, yi].sum(1), R[:, yi].sum(1), B[:, yi].sum(1)
    RRt, RSt, SSt = grid.RR[:, yi].sum(1), grid.RS[:, yi].sum(1), grid.SS[:, yi].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        roi = np.where(St > 0, Rt / St, 0.0)
        var = np.maximum(RRt - 2 * roi * RSt + roi * roi * SSt, 0.0)
        se = np.where(St > 0, np.sqrt(var) / St, np.inf)
    zlow = roi - Z90 * se
    # 各訓練年（その券種が存在する年）で最低件数
    elig = np.ones(roi.shape, bool)
    any_year = np.zeros(roi.shape, bool)
    for y in yi:
        av = grid.avail[grid.rule_bt, y]
        elig &= np.where(av[:, None], B[:, y] >= nmin, True)
        any_year |= av[:, None]
    elig &= any_year
    ok = elig.copy()
    if mode == "B":
        for y in yi:
            av = grid.avail[grid.rule_bt, y]
            with np.errstate(divide="ignore", invalid="ignore"):
                ok &= np.where(av[:, None], np.where(S[:, y] > 0, R[:, y] / S[:, y], 0) > 1.0, True)
        my = grid.races.month_year
        mi = np.flatnonzero(np.isin(my, train_years))
        with np.errstate(divide="ignore", invalid="ignore"):
            mroi = np.where(grid.Smon[:, mi] > 0, grid.Rmon[:, mi] / np.where(grid.Smon[:, mi] > 0, grid.Smon[:, mi], 1), 0)
        has = grid.Smon[:, mi] > 0
        win = ((mroi > 1.0) & has).sum(1)
        share = win / np.maximum(has.sum(1), 1)
        ok &= share >= 0.5
    score = (roi if mode == "C" else zlow)
    score = np.where(ok, score, -np.inf)
    j = int(np.argmax(score))
    if not np.isfinite(score.flat[j]):
        return None
    u, f = np.unravel_index(j, roi.shape)
    return int(u), int(f), {"train_roi": float(roi[u, f]), "train_zlow": float(zlow[u, f]), "train_bets_races": float(Bt[u, f]), "train_stake": float(St[u, f]), "n_eligible": int(ok.sum())}


def test_result(grid: Grid, u: int, f: int, year: int) -> dict:
    my = grid.races.month_year
    mi = np.flatnonzero(my == year)
    Sm, Rm = grid.Smon[u, mi, f].astype(float), grid.Rmon[u, mi, f].astype(float)
    S, R = Sm.sum(), Rm.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        mroi = np.where(Sm > 0, Rm / np.where(Sm > 0, Sm, 1), np.nan)
    n_months = len(mi)
    win_m = int(np.nansum(mroi > 1.0))
    bought = int((Sm > 0).sum())
    half = n_months / 2.0
    yi = grid.years.index(year)
    return {
        "year": year, "stake": float(S), "ret": float(R), "roi": float(R / S) if S > 0 else float("nan"),
        "bet_races": float(grid.Bmon[u, mi, f].sum()), "months": n_months, "bought_months": bought, "win_months": win_m,
        "A": bool(S > 0 and R / S > 1.0), "B": bool(win_m > half), "B_bought": bool(bought > 0 and win_m > bought / 2.0),
        "AB": bool(S > 0 and R / S > 1.0 and win_m > half),
        "monthly_roi": [None if np.isnan(x) else round(float(x), 4) for x in mroi],
        "labels": [grid.races.month_label[m] for m in mi],
    }


def insample_pass_count(grid: Grid, year: int, nmin: int = MIN_RACES_PER_YEAR) -> dict:
    """その年の中で（事後に）基準 A・B を満たした戦略の数（最小購入レース数を満たすもの）。帰無との比較用。"""
    S, R, B = grid.year_sums()
    yi = grid.years.index(year)
    mi = np.flatnonzero(grid.races.month_year == year)
    Sm, Rm = grid.Smon[:, mi], grid.Rmon[:, mi]
    with np.errstate(divide="ignore", invalid="ignore"):
        mroi = np.where(Sm > 0, Rm / np.where(Sm > 0, Sm, 1), 0)
    win = ((mroi > 1.0) & (Sm > 0)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        roi = np.where(S[:, yi] > 0, R[:, yi] / S[:, yi], 0)
    el = B[:, yi] >= nmin
    a = el & (roi > 1.0)
    ab = a & (win > len(mi) / 2.0)
    return {"eligible": int(el.sum()), "A": int(a.sum()), "AB": int(ab.sum())}


# ---------------- 帰無（市場が正しい）の再現 ----------------
class NullSim:
    """帰無: すべての買い目の期待回収率が控除率相当（RATE）。買い目ごとに『的中確率 = RATE / オッズ』で的中を決める。
    同時に 2 つは当たらない券種（単勝・馬連・馬単・3連単）は、レース × 券種の中で 1 つの乱数を区間に分けて 1 点だけ的中させる（排他）。
    複勝・ワイドは独立。券種間の相関は入れない（排他を入れないぶん、分散は実際よりやや大きめ = 運で通る確率を多めに見積もる向き）。
    （初版は、着順を単勝の市場確率から乱数で作って各買い目の的中を決めたが、組合せ券の確定オッズと整合せず、選んだ戦略の検証 ROI の中央値が
    150% を超える『帰無でない帰無』になったため、この形に作り直した。）"""

    EXCL = (0, 2, 4, 5)

    def __init__(self, d: pd.DataFrame, lam: float = 1.0):
        self.d = d
        self.bt = d["bt"].to_numpy()
        self.odds = d["odds"].to_numpy(np.float64)
        self.pi = np.minimum(1.0, RATE[self.bt] / np.maximum(self.odds, 1e-9))
        ridx = d["ridx"].to_numpy().astype(np.int64)
        self.excl = np.isin(self.bt, self.EXCL)
        idx = np.flatnonzero(self.excl)
        key = ridx[idx] * 6 + self.bt[idx]
        o = np.argsort(key, kind="stable")
        self.ex_order = idx[o]
        ko = key[o]
        new = np.r_[True, ko[1:] != ko[:-1]]
        self.ex_gid = np.cumsum(new) - 1
        p = self.pi[self.ex_order]
        cs = np.cumsum(p)
        start = (cs - p)[new]
        self.ex_lo = cs - p - start[self.ex_gid]
        self.ex_hi = cs - start[self.ex_gid]
        self.n_groups = int(self.ex_gid[-1]) + 1 if len(self.ex_gid) else 0
        self.nonex = np.flatnonzero(~self.excl)
        self.c = np.ones(6)
        self.raw_mean = RATE.copy()

    def draw_hits(self, rng: np.random.Generator) -> np.ndarray:
        hit = np.zeros(len(self.bt), bool)
        hit[self.nonex] = rng.random(len(self.nonex)) < self.pi[self.nonex]
        u = rng.random(self.n_groups)[self.ex_gid]
        hit[self.ex_order] = (u >= self.ex_lo) & (u < self.ex_hi)
        return hit

    def calibrate(self, n: int = 0, seed: int = 0) -> np.ndarray:
        return self.c

    def draw_ret(self, rng: np.random.Generator) -> np.ndarray:
        return self.draw_hits(rng) * self.odds


def power_curve(grid: Grid, sim: NullSim, u: int, f: int, year: int, targets=(0.9, 1.0, 1.05, 1.1, 1.2, 1.3), reps: int = 200, seed: int = 0) -> dict:
    """固定の戦略 (u, f) について、真の期待回収率が target のとき、その年に基準 A・B を満たす確率。
    市場が正しい帰無の払戻（simulate）を、券種ごとの控除率で割った倍率（target / RATE）だけ一様に増減して作る。"""
    ur = u % grid.U
    sel = np.flatnonzero(grid.key // grid.Rn == ur)
    rows = grid.pair_row[sel]
    race = (grid.key[sel] % grid.Rn).astype(np.int64)
    mask = grid.races.masks[f][race]
    rows, race = rows[mask], race[mask]
    ym = grid.races.month[race]
    yr = grid.races.year[race]
    inyear = yr == year
    rows, race, ym = rows[inyear], race[inyear], ym[inyear]
    months = np.flatnonzero(grid.races.month_year == year)
    mpos = {m: i for i, m in enumerate(months)}
    mi = np.array([mpos[m] for m in ym])
    flat = u < grid.U
    cnt = np.bincount(race, minlength=grid.Rn).astype(float)
    bt = sim.bt[rows]
    out = {t: {"A": 0, "B": 0, "AB": 0} for t in targets}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        hit = sim.draw_hits(rng)[rows] * sim.odds[rows]                     # 帰無: 期待値は RATE
        for t in targets:
            base = hit * sim.c[bt]                                         # 控除率 RATE の期待に較正済み
            ret = base * (t / RATE[bt])
            if flat:
                S_m = np.bincount(mi, minlength=len(months)).astype(float)
                R_m = np.bincount(mi, weights=ret, minlength=len(months))
            else:
                per_race = np.bincount(race, weights=ret, minlength=grid.Rn) / np.maximum(cnt, 1.0)
                rr = np.unique(race)
                rmi = grid.races.month[rr]
                mj = np.array([mpos[m] for m in rmi])
                S_m = np.bincount(mj, minlength=len(months)).astype(float)
                R_m = np.bincount(mj, weights=per_race[rr], minlength=len(months))
            S, R = S_m.sum(), R_m.sum()
            win = int(((S_m > 0) & (R_m > S_m)).sum())
            a = bool(S > 0 and R > S)
            b = win > len(months) / 2.0
            out[t]["A"] += a
            out[t]["B"] += b
            out[t]["AB"] += a and b
    return {str(t): {k: v / reps for k, v in o.items()} for t, o in out.items()}
