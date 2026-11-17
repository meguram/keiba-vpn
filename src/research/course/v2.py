"""全レース版の第 2 版（時間で分けて採る版。計画書 docs/ideas/course_class_handicapper.md §8）。

- 発見 2016〜2021 年で条件を探し（第 1 版と同じ find_cells。下限は期間の長さに合わせて縮める）、確認 2022〜2024 年でも
  買い（複勝率 70% 以上・単か複の回収率 100% 超）と消し（4 着以下の割合）を保った条件だけを採る。
- 20 サイクルの採否は確認の期間の ◎ の値（本番の再現）で行う。発見の期間の値は並べるだけ。
  注意: 確認の期間で条件を選ぶので、確認の値もいくらか高めに出る。偏りのない値は、凍結の後の 2025・2026 年の評価だけ。
- オッズは条件の切り口に使わない。買う／買わないの最後の絞り込み（◎ のオッズ帯・人気帯・期待値）にだけ使う。
  本番の再現: 判定は発走前（T-60）のオッズ、払戻は確定。過去の年の発走前のオッズは、2026 年の T-60 と確定の組から作った
  「T-60 → 確定」のずれの分布（確定の人気帯ごと）を、確定オッズに逆に当てて 20 通り作る（shift_model。着順・払戻は見ない）。
- 2025・2026 年は、凍結・commit の後に 1 回だけ（第 1 版の評価で一度見ているので、2 回目）。

使い方（COURSE_OUT は第 1 版と同じ。第 2 版の出力は <COURSE_OUT>/v2）:
  python -m src.research.course.v2 shift      # ずれの分布（2026 年のオッズの組だけ）
  python -m src.research.course.v2 cycles     # 20 サイクル（2016〜2024 年だけ）
  python -m src.research.course.v2 freeze     # 凍結（prereg_course_v2.json を commit する）
  python -m src.research.course.v2 evaluate   # 2025・2026 年で 1 回だけ（2 回目）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import search as SE

OUT2 = CR.OUT / "v2"
PREREG2 = Path(__file__).with_name("prereg_course_v2.json")
EVAL_MARK2 = Path(__file__).with_name("EVALUATED_COURSE_V2.json")
DISC_END, CONF = 2021, (2022, 2024)
PRE_ODDS = CR.REPO / "data/local/research/persona1/pre_odds.parquet"
POP_BANDS = [(1, 1), (2, 3), (4, 6), (7, 99)]
ODDS_EDGES = [0.0, 3.0, 10.0, 30.0, np.inf]     # 確定オッズの帯（人気帯と組み合わせる。組の数が少ない帯は人気帯だけの分布で代える）
MIN_PAIRS = 30
N_DRAWS = 20
MIN_BET_CONF = 500          # 採否に要る、確認の期間で ◎ を買ったレースの数（小さい集まりの偶然で採らない）
SEED = 20261117
NOTE_CONF = "確認の期間（2022〜2024）で条件を選ぶので、確認の値もいくらか高めに出る。偏りのない値は、凍結の後の 2025・2026 年の評価だけ"


@dataclass(frozen=True)
class P2:
    # 発見（第 1 版の Params と同じ意味。値は全期間の物差しで書き、発見の期間では縮めて使う）
    depth: int = 2
    n_min: int = 2
    kill_n: int = 20
    kill_depth: int = 1
    min_races: int = 10
    weight: str = "rate"
    groups: int = 4
    # 確認で採る
    c_min: int = 5            # 買い: 確認の走数の下限
    shrink: float = 0.0       # 買い: 確認の複勝率を、単位の確認の複勝率へ寄せる強さ（擬似の走数）
    roi_mode: str = "either"  # 買い: either = 単か複の回収率 100% 超、tan = 単の回収率 100% 超
    k_min: int = 10           # 消し: 確認の走数の下限
    kill_rule: str = "point"  # 消し: point = 4 着以下の割合 ≥ kill_prec、lb = その信頼区間の下限 ≥ kill_prec
    kill_prec: float = 0.95
    # 買う／買わないの最後の絞り込み（発走前のオッズで判定）
    odds_min: float = 0.0     # ◎ の単勝オッズの下限（0 = 絞らない）
    pop_min: int = 1          # ◎ の人気の下限（2 = 1 番人気は買わない）
    ev_min: float = 0.0       # 期待値（◎ の確認の勝率 × オッズ）の下限（0 = 絞らない）

    def disc_params(self, frac: float) -> SE.Params:
        sc = lambda v: max(1, round(v * frac))   # noqa: E731
        return SE.Params(depth=self.depth, n_min=self.n_min if self.n_min <= 2 else sc(self.n_min), kill_n=sc(self.kill_n),
                         kill_depth=self.kill_depth, min_races=sc(self.min_races), weight=self.weight, groups=self.groups)

    def disc_key(self) -> tuple:
        return (self.depth, self.n_min, self.kill_n, self.kill_depth, self.min_races, self.groups)

    def name(self) -> str:
        f = []
        if self.odds_min:
            f.append(f"オッズ{self.odds_min}倍以上")
        if self.pop_min > 1:
            f.append(f"{self.pop_min}番人気以下")
        if self.ev_min:
            f.append(f"期待値{self.ev_min}以上")
        return (f"群{self.groups}・深さ{self.depth}・走数{self.n_min}・主{self.min_races}回・{self.weight}・確認{self.c_min}走"
                f"{'・縮め' + str(self.shrink) if self.shrink else ''}・{self.roi_mode}・消し{self.kill_n}走{self.kill_depth}段・確認{self.k_min}走"
                f"・{self.kill_rule}{self.kill_prec}" + ("・" + "・".join(f) if f else ""))


def wilson_lb(k: float, n: float, z: float = 1.96) -> float:
    if not n:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / d


# ---------------------------------------------------------------- オッズ（確定）と、発走前のオッズの再現
def attach_final_odds(d: pd.DataFrame, years) -> pd.DataFrame:
    """d に確定の単勝オッズ・人気（race_result_flat の odds・popularity）を付ける（買う／買わないの判定と、ずれの当てはめだけに使う）。"""
    parts = []
    for y in years:
        f = CR.REPO / "data/page_reference/tables" / str(y) / "race_result_flat.parquet"
        x = pd.read_parquet(f, columns=["race_id", "horse_number", "odds", "popularity"])
        parts.append(x)
    o = pd.concat(parts, ignore_index=True)
    o["race_id"] = o["race_id"].astype(str)
    o["hn"] = pd.to_numeric(o["horse_number"], errors="coerce")
    o = o.dropna(subset=["hn"]).drop_duplicates(["race_id", "hn"])
    k = pd.DataFrame({"race_id": d["race_id"].astype(str).to_numpy(), "hn": pd.to_numeric(d["hn"], errors="coerce").to_numpy()})
    m = k.merge(o[["race_id", "hn", "odds", "popularity"]], on=["race_id", "hn"], how="left")
    d = d.copy()
    d["odds_final"] = pd.to_numeric(m["odds"], errors="coerce").to_numpy()
    d["pop_final"] = pd.to_numeric(m["popularity"], errors="coerce").to_numpy()
    return d


def band_of(pop: np.ndarray) -> np.ndarray:
    b = np.full(len(pop), len(POP_BANDS) - 1)
    for i, (lo, hi) in enumerate(POP_BANDS):
        b[(pop >= lo) & (pop <= hi)] = i
    return b


def oband_of(odds: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(ODDS_EDGES, np.nan_to_num(odds, nan=0.0), side="right") - 1, 0, len(ODDS_EDGES) - 2)


def cell_of(pop: np.ndarray, odds: np.ndarray) -> np.ndarray:
    return band_of(pop) * (len(ODDS_EDGES) - 1) + oband_of(odds)


def build_shift() -> dict:
    """2026 年の発走前（T-60）と確定のオッズの組から、log(確定 / T-60) の分布を確定の人気帯ごとに作る（着順・払戻は見ない）。"""
    p = pd.read_parquet(PRE_ODDS)
    p = p[p["phase"] == "t60"][["race_id", "horse_number", "win_odds", "popularity"]]
    p["race_id"] = p["race_id"].astype(str)
    years = sorted({int(r[:4]) for r in p["race_id"]})
    f = pd.concat([pd.read_parquet(CR.REPO / "data/page_reference/tables" / str(y) / "race_result_flat.parquet",
                                   columns=["race_id", "horse_number", "odds", "popularity"]) for y in years])
    f["race_id"] = f["race_id"].astype(str)
    m = p.merge(f, on=["race_id", "horse_number"], suffixes=("_t60", "_final"))
    m = m[(m["win_odds"] > 0) & (m["odds"] > 0)]
    lr = np.log(m["odds"].to_numpy() / m["win_odds"].to_numpy())
    pop = pd.to_numeric(m["popularity_final"]).to_numpy()
    b = band_of(pop)
    c = cell_of(pop, m["odds"].to_numpy(float))
    nc = len(POP_BANDS) * (len(ODDS_EDGES) - 1)
    return {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "source": str(PRE_ODDS.relative_to(CR.REPO)), "phase": "t60",
            "races": int(m["race_id"].nunique()), "pairs": int(len(m)), "bands": [list(x) for x in POP_BANDS],
            "odds_edges": [float(x) for x in ODDS_EDGES[:-1]], "min_pairs": MIN_PAIRS,
            "lr": [sorted(float(v) for v in lr[b == i]) for i in range(len(POP_BANDS))],
            "lr_cell": [sorted(float(v) for v in lr[c == i]) for i in range(nc)],
            "all": sorted(float(v) for v in lr),
            "note": "2026 年（評価の期間）の発走前と確定のオッズの組だけを使った。着順・払戻は見ていない（統括役の承認、2026-11-17）"}


def simulate_pre(d: pd.DataFrame, shift: dict, draw: int) -> tuple[np.ndarray, np.ndarray]:
    """確定オッズにずれを逆に当てた発走前のオッズ（draw 通り目）と、レースの中で作り直した人気。オッズの無い馬は NaN。"""
    rng = np.random.default_rng(SEED + draw)
    of = d["odds_final"].to_numpy(float)
    pf = np.nan_to_num(d["pop_final"].to_numpy(float), nan=99)
    b = band_of(pf)
    lr = np.zeros(len(d))
    # 人気帯 × 確定オッズ帯の分布（組が MIN_PAIRS 以上）→ 人気帯だけの分布 → 全体の分布の順に使う（空の帯でずれ 0 にしない）
    cells = shift.get("lr_cell")
    if cells:
        c = cell_of(pf, of)
        for i in range(len(cells)):
            idx = np.flatnonzero(c == i)
            arr = cells[i] if len(cells[i]) >= shift.get("min_pairs", MIN_PAIRS) else (shift["lr"][i // (len(ODDS_EDGES) - 1)] or shift.get("all"))
            if len(idx) and arr:
                lr[idx] = rng.choice(np.asarray(arr), size=len(idx), replace=True)
    else:
        for i, arr in enumerate(shift["lr"]):
            idx = np.flatnonzero(b == i)
            arr = arr or shift.get("all")
            if len(idx) and arr:
                lr[idx] = rng.choice(np.asarray(arr), size=len(idx), replace=True)
    pre = np.round(of / np.exp(lr), 1)
    pre = np.where(of > 0, np.maximum(pre, 1.0), np.nan)
    pop = pd.Series(pre).groupby(d["race_id"].astype(str).to_numpy()).rank(method="min").to_numpy()
    return pre, pop


# ---------------------------------------------------------------- 条件の採り方
def _codes(L: pd.DataFrame, cells_all: dict) -> tuple[np.ndarray, list[str], dict]:
    from src.research.graded import buy as BY
    types = sorted({t for cu in cells_all.values() for kind in ("buy", "kill") for c in cu[kind] for t in c["types"]})
    miss = [t for t in types if t not in L.columns]
    if miss:
        raise ValueError(f"表に型の列が無い: {miss}")
    X, names = BY.encode(L, types) if types else (np.zeros((len(L), 0), np.int32), {})
    return X, types, {t: {v: i for i, v in enumerate(names[t])} for t in types}


def _mask_names(X: np.ndarray, types: list[str], code: dict, c: dict) -> np.ndarray:
    m = np.ones(len(X), bool)
    for t, lv in zip(c["types"], c["levels"]):
        m &= X[:, types.index(t)] == code[t].get(lv, -2)
    return m


def confirm_cells(cells_all: dict, d: pd.DataFrame, L: pd.DataFrame, unit: np.ndarray, p: P2) -> tuple[dict, dict]:
    """発見の条件（名前）を、確認の期間の同じ単位の走に当てて数え、基準を保つ条件だけを残す。
    返り値: 残した条件（単位 → buy・kill。buy には確認の値 n_c・place_c・win_c・roi_t_c・roi_f_c、点 rate は確認の複勝率（縮め後））と、数の記録。"""
    X, types, code = _codes(L, cells_all)
    Y = SE.y_matrix(d)
    course = d["course"].astype(str).to_numpy()
    kept, log = {}, {"buy_found": 0, "buy_small": 0, "buy_kept": 0, "buy_unmatched": 0, "kill_found": 0, "kill_small": 0, "kill_kept": 0,
                     "kill_unmatched": 0, "sub_units": 0}
    for u, cu in cells_all.items():
        # 副の単位（発見の期間で回数が少なく、同じコースの全群で探した単位）は、確認でも同じコースの全群の走で数える
        sub = cu.get("layer") == "副"
        idx = np.flatnonzero(course == u.split("|", 1)[0]) if sub else np.flatnonzero(unit == u)
        log["sub_units"] += int(sub)
        if len(idx) == 0:
            kept[u] = {"layer": cu.get("layer"), "buy": [], "kill": []}
            continue
        Xu, Yu = X[idx], Y[idx]
        p0 = Yu[:, 1].mean() if len(Yu) else 0.0
        kb, kk = [], []
        for c in cu["buy"]:
            log["buy_found"] += 1
            if any(lv not in code[t] for t, lv in zip(c["types"], c["levels"])):
                log["buy_unmatched"] += 1     # 確認の期間に、その水準の名前が一度も出ない（引退した騎手など）
                continue
            m = _mask_names(Xu, types, code, c)
            s = Yu[m].sum(0)
            n, k, w, npay = s[0], s[1], s[2], s[5]
            if n < p.c_min:
                log["buy_small"] += 1
                continue
            place = (k + p.shrink * p0) / (n + p.shrink)
            rt, rf = (s[6] / npay if npay else 0.0), (s[7] / npay if npay else 0.0)
            ok_roi = rt > 1.0 if p.roi_mode == "tan" else (rt > 1.0 or rf > 1.0)
            if place >= 0.70 and ok_roi:
                kb.append(c | {"n_c": int(n), "k_c": int(k), "place_c": float(place), "win_c": float(w / n), "roi_t_c": float(rt), "roi_f_c": float(rf),
                               "rate": float(place)})
        for c in cu["kill"]:
            log["kill_found"] += 1
            if any(lv not in code[t] for t, lv in zip(c["types"], c["levels"])):
                log["kill_unmatched"] += 1
                continue
            m = _mask_names(Xu, types, code, c)
            n, k = int(m.sum()), int(Yu[m, 1].sum())
            if n < p.k_min:
                log["kill_small"] += 1
                continue
            prec = (n - k) / n
            val = wilson_lb(n - k, n) if p.kill_rule == "lb" else prec
            if val >= p.kill_prec:
                kk.append(c | {"n_c": n, "prec_c": float(prec)})
        kept[u] = {"layer": cu.get("layer"), "buy": kb, "kill": kk}
        log["buy_kept"] += len(kb)
        log["kill_kept"] += len(kk)
    return kept, log


def apply_named(d: pd.DataFrame, L: pd.DataFrame, unit: np.ndarray, kept: dict, weight: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """残した条件（名前）を当てて、◎ の行・◎ の確認の勝率（期待値の判定用）・消す行を返す（第 1 版と同じ選び方 = search.apply_cells）。"""
    X, types, code = _codes(L, kept)
    race = d["race_id"].astype(str).to_numpy()
    hn = pd.to_numeric(d["hn"], errors="coerce").fillna(99).to_numpy()
    H, Pw, K = [], [], np.zeros(len(d), bool)
    for u in sorted(set(unit)):
        cu = kept.get(u)
        if not cu or not cu["buy"] and not cu["kill"]:
            continue
        idx = np.flatnonzero(unit == u)

        def conv(c):
            return {"ts": tuple(types.index(t) for t in c["types"]), "cs": tuple(code[t].get(lv, -2) for t, lv in zip(c["types"], c["levels"])),
                    "rate": c.get("rate", 1.0)}
        cc = {"buy": [conv(c) for c in cu["buy"]], "kill": [conv(c) for c in cu["kill"]]}
        hm, kl = SE.apply_cells(X[idx], race[idx], hn[idx], cc, weight)
        if hm:
            rows = idx[list(hm.values())]
            # ◎ の勝率の見込み: ◎ に当たった買いの条件のうち、確認の勝率が最も高いもの（期待値の判定だけに使う）
            Xi = X[rows]
            pw = np.zeros(len(rows))
            for c, c2 in zip(cu["buy"], cc["buy"]):
                mm = SE._mask(Xi, c2["ts"], c2["cs"])
                pw = np.where(mm, np.maximum(pw, c["win_c"]), pw)
            H.append(rows)
            Pw.append(pw)
        K[idx[kl]] = True
    H = np.concatenate(H) if H else np.array([], int)
    Pw = np.concatenate(Pw) if Pw else np.array([])
    return H, Pw, K


def odds_filter(p: P2, odds: np.ndarray, pop: np.ndarray, pw: np.ndarray) -> np.ndarray:
    """買う／買わないの最後の絞り込み（◎ の行ごと）。オッズが無い ◎ は、絞り込みのある版では買わない。"""
    keep = np.ones(len(odds), bool)
    if p.odds_min:
        keep &= np.nan_to_num(odds, nan=0) >= p.odds_min
    if p.pop_min > 1:
        keep &= np.nan_to_num(pop, nan=0) >= p.pop_min
    if p.ev_min:
        keep &= pw * np.nan_to_num(odds, nan=0) >= p.ev_min
    return keep


def has_filter(p: P2) -> bool:
    return bool(p.odds_min or p.pop_min > 1 or p.ev_min)


def _summ(draws: list[dict], base: dict) -> dict:
    rep = {}
    for key in ("bet_races", "coverage", "win", "place", "roi_t", "roi_f", "roi_t_capped"):
        v = np.array([x[key] for x in draws if x[key] is not None], float)
        rep[key] = float(v.mean()) if len(v) else None
        rep[key + "_p05"] = float(np.percentile(v, 5)) if len(v) else None
        rep[key + "_p95"] = float(np.percentile(v, 95)) if len(v) else None
    return rep | {k: base[k] for k in ("races", "killed", "kill_races", "kill_coverage", "kill_per_race", "kill_precision")}


def score(d: pd.DataFrame, L: pd.DataFrame, unit: np.ndarray, kept: dict, p: P2, shift: dict | None, real_pre: dict | None = None,
          variants: list[P2] | None = None) -> dict:
    """ある期間の値: 楽観（確定オッズで判定）と本番の再現（発走前のオッズを 20 通り作って判定。平均・5〜95%）。消しの的中。
    real_pre = {(race_id, 馬番): (T-60 のオッズ, 人気)} があれば、T-60 のあるレースだけで、本物の T-60 で判定した値も出す。
    variants（しきい値の隣の版）があれば、同じ ◎ で絞り込みだけ変えた本番の再現の単回収も並べる。"""
    Y = SE.y_matrix(d)
    race = d["race_id"].astype(str).to_numpy()
    nr = len(set(race))
    hn_all = pd.to_numeric(d["hn"], errors="coerce").fillna(-1).astype(int).to_numpy()
    H, Pw, K = apply_named(d, L, unit, kept, p.weight)
    kr = np.flatnonzero(K)
    base = SE.metrics(Y, race, H, kr, nr)
    out = {"no_filter": base}
    of, pf = d["odds_final"].to_numpy(float), d["pop_final"].to_numpy(float)
    filt = has_filter(p)
    pres = [simulate_pre(d, shift, k) for k in range(N_DRAWS)] if (filt or variants) and shift else []
    if filt:
        out["optimistic"] = SE.metrics(Y, race, H[odds_filter(p, of[H], pf[H], Pw)], kr, nr)
        out["reproduction"] = _summ([SE.metrics(Y, race, H[odds_filter(p, pre[H], pop[H], Pw)], kr, nr) for pre, pop in pres], base)
    else:
        out["optimistic"] = out["reproduction"] = base
    if variants:
        out["neighbors"] = [{"name": v.name(), "odds_min": v.odds_min, "pop_min": v.pop_min, "ev_min": v.ev_min,
                             "optimistic_roi_t": SE.metrics(Y, race, H[odds_filter(v, of[H], pf[H], Pw)], kr, nr)["roi_t"],
                             "reproduction": {k: x for k, x in _summ([SE.metrics(Y, race, H[odds_filter(v, pre[H], pop[H], Pw)], kr, nr)
                                                                      for pre, pop in pres], base).items() if k.startswith(("roi_t", "bet_races"))}}
                            for v in variants]
    if real_pre:
        rr = {k[0] for k in real_pre} & set(race)          # T-60 があり、この表（良・単位の中）にもあるレースだけ
        if rr:
            m = np.isin(race, list(rr))
            Hs = H[m[H]]
            ro = np.array([real_pre.get((race[h], hn_all[h]), (np.nan, np.nan))[0] for h in Hs])
            rp = np.array([real_pre.get((race[h], hn_all[h]), (np.nan, np.nan))[1] for h in Hs])
            pw = Pw[m[H]]
            ks = kr[m[kr]]
            out["real_t60"] = SE.metrics(Y, race, Hs[odds_filter(p, ro, rp, pw)] if filt else Hs, ks, len(rr))
            out["real_t60_optimistic"] = SE.metrics(Y, race, Hs[odds_filter(p, of[Hs], pf[Hs], pw)] if filt else Hs, ks, len(rr))
            out["real_t60_missing_odds"] = int(np.isnan(ro).sum())
    return out


def neighbors(p: P2) -> list[P2]:
    """しきい値のすぐ隣の版（判定がしきい値の近くに左右されていないかを見る）。"""
    out = []
    if p.odds_min:
        out += [replace(p, odds_min=round(p.odds_min * f, 1)) for f in (0.8, 1.2)]
    if p.ev_min:
        out += [replace(p, ev_min=round(p.ev_min + dv, 2)) for dv in (-0.1, 0.1)]
    if p.pop_min > 1:
        out += [replace(p, pop_min=p.pop_min + 1)]
    return out


# ---------------------------------------------------------------- 20 サイクル
CYCLES = [
    {"hyp": "基準: 第 1 版の最良の発見（群 4・深さ 2・主 10 回・rate）、確認で買い 5 走・消し 10 走以上・点 95%", "delta": {}},
    {"hyp": "買いの確認の下限 3 走", "delta": {"c_min": 3}},
    {"hyp": "買いの確認の下限 10 走", "delta": {"c_min": 10}},
    {"hyp": "確認の複勝率を単位の平均へ寄せる（擬似 10 走）", "delta": {"shrink": 10.0}},
    {"hyp": "買いは単の回収率 100% 超に限る", "delta": {"roi_mode": "tan"}},
    {"hyp": "◎ の点を条件の数にする", "delta": {"weight": "count"}},
    {"hyp": "発見を深さ 1 にする", "delta": {"depth": 1}},
    {"hyp": "発見の買いの走数の下限 5", "delta": {"n_min": 5}},
    {"hyp": "主の回数の下限 20", "delta": {"min_races": 20}},
    {"hyp": "クラスの群を 3 つ", "delta": {"groups": 3}},
    {"hyp": "消しの確認の下限 20 走", "delta": {"k_min": 20}},
    {"hyp": "消しの基準を信頼区間の下限 95% にする", "delta": {"kill_rule": "lb"}},
    {"hyp": "消しの基準を 90% にする（どこまで届くかを見る）", "delta": {"kill_prec": 0.90}},
    {"hyp": "発見の消しの走数の下限 10", "delta": {"kill_n": 10}},
    {"hyp": "発見の消しを 2 つの AND まで", "delta": {"kill_depth": 2}},
    {"hyp": "絞り込み: ◎ の発走前のオッズ 2 倍以上", "delta": {"odds_min": 2.0}},
    {"hyp": "絞り込み: ◎ の発走前のオッズ 3 倍以上", "delta": {"odds_min": 3.0}},
    {"hyp": "絞り込み: 発走前の 1 番人気の ◎ は買わない", "delta": {"pop_min": 2}},
    {"hyp": "絞り込み: 期待値（確認の勝率 × 発走前のオッズ）1.0 以上", "delta": {"ev_min": 1.0}},
    {"hyp": "絞り込み: 期待値 1.2 以上", "delta": {"ev_min": 1.2}},
]


def load_split():
    """2016〜2024 年の単位の表（第 1 版の frame と水準）に確定オッズを付け、発見・確認に分ける。2025 年以降は読まない。"""
    d, L = CR.load_units()
    yr = pd.to_numeric(d["year"]).to_numpy()
    if (yr > CR.YEAR_END).any():
        raise ValueError("2025 年以降の行が入っている")
    d = attach_final_odds(d, range(2016, CR.YEAR_END + 1))
    dis, conf = yr <= DISC_END, (yr >= CONF[0]) & (yr <= CONF[1])
    frac = d.loc[dis, "race_id"].nunique() / d["race_id"].nunique()
    return d, L, dis, conf, frac


def run_one(p: P2, d, L, dis, conf, frac, shift, disc_cache: dict) -> dict:
    t0 = time.time()
    key = p.disc_key()
    if key not in disc_cache:
        fit = CR.run_config(d[dis].reset_index(drop=True), L[dis].reset_index(drop=True), p.disc_params(frac), keep_cells=True)
        disc_cache[key] = fit["cells"]
    cells = disc_cache[key]
    dc, Lc = d[conf].reset_index(drop=True), L[conf].reset_index(drop=True)
    dd, Ld = d[dis].reset_index(drop=True), L[dis].reset_index(drop=True)
    uc, ud = CR.regroup(dc, p.groups).to_numpy(), CR.regroup(dd, p.groups).to_numpy()
    kept, clog = confirm_cells(cells, dc, Lc, uc, p)
    sc = score(dc, Lc, uc, kept, p, shift, variants=neighbors(p) or None)
    sd = score(dd, Ld, ud, kept, p, shift)
    return {"params": asdict(p) | {"name": p.name()}, "conf": sc, "disc": sd, "confirm_log": clog, "kept": kept,
            "seconds": round(time.time() - t0, 1), "mem": CR.mem_now()}


def qualified(r: dict) -> bool:
    rep = r["conf"]["reproduction"]
    return (rep.get("bet_races") or 0) >= MIN_BET_CONF and rep.get("roi_t") is not None


def adopt_key(r: dict) -> tuple:
    """採否: 確認の期間の本番の再現で、◎ を買ったレースが MIN_BET_CONF 以上の版のうち、単回収の高い方（下限に届かない版は、届く版より下）。"""
    return (qualified(r), r["conf"]["reproduction"].get("roi_t") or 0.0)


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "kept"}


def cmd_shift() -> None:
    if PREREG2.exists() or EVAL_MARK2.exists() or (OUT2 / "course_v2_cycles.json").exists():
        sys.exit("サイクル・凍結・評価の後に、ずれの分布を作り直さない")
    OUT2.mkdir(parents=True, exist_ok=True)
    s = build_shift()
    (OUT2 / "shift.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    print("shift", s["races"], "R", s["pairs"], "組", [len(x) for x in s["lr"]], [round(float(np.median(x)), 3) for x in s["lr"]])


def cmd_cycles() -> None:
    if PREREG2.exists() or EVAL_MARK2.exists():
        sys.exit("凍結・評価の後にサイクルを流し直さない")
    shift = json.loads((OUT2 / "shift.json").read_text(encoding="utf-8"))
    d, L, dis, conf, frac = load_split()
    log = {"note": "全レース版の第 2 版の 20 サイクル（発見 2016〜2021・確認 2022〜2024。2025・2026 年は見ていない）", "note_conf": NOTE_CONF,
           "min_bet_conf": MIN_BET_CONF, "frac": frac, "shift_sha": hashlib.sha256((OUT2 / "shift.json").read_bytes()).hexdigest(),
           "started_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cycles": []}
    cur, best, cache = P2(), None, {}
    path = OUT2 / "course_v2_cycles.json"
    for c, cy in enumerate(CYCLES, 1):
        CR.wait_resources()
        cand = replace(cur, **cy["delta"])
        r = run_one(cand, d, L, dis, conf, frac, shift, cache)
        adopt = best is None or adopt_key(r) > adopt_key(best)
        if adopt:
            best, cur = r | {"cycle": c}, cand
        log["cycles"].append({"cycle": c, "hyp": cy["hyp"], "adopted": adopt, "qualified": qualified(r)} | _slim(r))
        log["best_cycle"] = best["cycle"]
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        rc, rd = r["conf"]["reproduction"], r["disc"]["reproduction"]
        print(f"cycle {c} {cy['hyp']} 確認(再現) 買う {rc.get('bet_races')} 勝率 {rc.get('win')} 単回収 {rc.get('roi_t')}"
              f"（5〜95%: {rc.get('roi_t_p05')}〜{rc.get('roi_t_p95')}） 楽観 {r['conf']['optimistic'].get('roi_t')}"
              f" 消しの的中 {rc.get('kill_precision')} 消しのレース {rc.get('kill_coverage')} ｜ 発見 単回収 {rd.get('roi_t')}"
              f" 採用 {adopt} {r['seconds']}s メモリ {r['mem']}", flush=True)
    log["finished_at"] = datetime.now(CR.JST).isoformat(timespec="seconds")
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT2 / "course_v2_best.json").write_text(json.dumps(best, ensure_ascii=False, default=float), encoding="utf-8")


def cmd_freeze() -> None:
    if EVAL_MARK2.exists():
        sys.exit("評価済み。凍結し直さない")
    if PREREG2.exists():
        sys.exit(f"{PREREG2} は凍結済み")
    L = json.loads((OUT2 / "course_v2_cycles.json").read_text(encoding="utf-8"))
    if not L.get("finished_at"):
        sys.exit("サイクルが最後まで回っていない")
    B = json.loads((OUT2 / "course_v2_best.json").read_text(encoding="utf-8"))
    bc = next(c for c in L["cycles"] if c["cycle"] == L["best_cycle"])
    if bc["params"] != B["params"]:
        sys.exit("course_v2_best.json の版が記録の最良と違う")
    if not bc.get("qualified"):
        sys.exit(f"最良の版が、確認の期間で ◎ を買ったレース {MIN_BET_CONF} 以上に届かない（凍結しない。記録を見て作法を見直す）")
    if CR._sha_file(OUT2 / "shift.json") != L.get("shift_sha"):
        sys.exit("ずれの分布が、サイクルを回したときと違う")
    cells = OUT2 / "course_v2_cells.json"
    cells.write_text(json.dumps(B["kept"], ensure_ascii=False, sort_keys=True), encoding="utf-8")
    body = {"frozen_at": datetime.now(CR.JST).isoformat(timespec="seconds"),
            "note": "全レース版の第 2 版。発見 2016〜2021・確認 2022〜2024 で選んだ。2025・2026 年はこの版では見ていない（第 1 版の評価で一度見た。この版の評価は 2 回目）",
            "note_conf": NOTE_CONF, "params": B["params"], "best_cycle": L["best_cycle"], "cycles_finished_at": L["finished_at"],
            "conf_2022_2024": B["conf"], "disc_2016_2021": B["disc"], "confirm_log": B["confirm_log"],
            "cells_file": "course_v2_cells.json", "sha256_cells": CR._sha_file(cells),
            "shift_file": "shift.json", "sha256_shift": CR._sha_file(OUT2 / "shift.json"), "n_draws": N_DRAWS, "seed": SEED}
    PREREG2.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("frozen", PREREG2)


def cmd_evaluate() -> None:
    """凍結の後に 1 回だけ（2025・2026 年を見るのは 2 回目）。凍結が commit されていて、条件・ずれの sha が一致するときだけ。
    評価済みの印は計算の前に立てる。2025・2026 年の表は、第 1 版の評価で作った master（<COURSE_OUT>/eval/master）から作り直す。"""
    import secrets
    if EVAL_MARK2.exists():
        sys.exit("評価は済んでいる（1 回だけ）")
    P = json.loads(PREREG2.read_text(encoding="utf-8"))
    if CR._sha_file(OUT2 / P["cells_file"]) != P["sha256_cells"] or CR._sha_file(OUT2 / P["shift_file"]) != P["sha256_shift"]:
        sys.exit("条件の表・ずれの分布が凍結の sha と違う")
    r = subprocess.run(["git", "-C", str(CR.REPO), "ls-files", "--error-unmatch", str(PREREG2)], capture_output=True)
    if r.returncode != 0 or subprocess.run(["git", "-C", str(CR.REPO), "diff", "--quiet", "HEAD", "--", str(PREREG2)]).returncode != 0:
        sys.exit("prereg_course_v2.json が commit されていない")
    master = CR.OUT / "eval" / "master" / "master.parquet"
    if not master.exists():
        sys.exit(f"2026 年までの master が無い: {master}")
    tok = secrets.token_hex(16)
    EVAL_MARK2.write_text(json.dumps({"started_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "prereg_sha_cells": P["sha256_cells"],
                                      "token_sha": hashlib.sha256(tok.encode()).hexdigest()}), encoding="utf-8")
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval"), "COURSE_EVAL_TOKEN": tok}
    subprocess.run([sys.executable, "-m", "src.research.course.v2", "_eval_inner"], env=e, cwd=CR.REPO, check=True)


def _eval_inner() -> None:
    from src.research.graded import features as F
    from src.research.graded.run import with_returns
    tok = os.environ.get("COURSE_EVAL_TOKEN", "")
    mark = json.loads(EVAL_MARK2.read_text(encoding="utf-8")) if EVAL_MARK2.exists() else {}
    if not tok or mark.get("token_sha") != hashlib.sha256(tok.encode()).hexdigest() or "evaluated_at" in mark:
        sys.exit("評価は cmd_evaluate から 1 回だけ流す")
    P = json.loads(PREREG2.read_text(encoding="utf-8"))
    if (CR._sha_file(OUT2 / P["cells_file"]) != P["sha256_cells"] or mark.get("prereg_sha_cells") != P["sha256_cells"]
            or CR._sha_file(OUT2 / P["shift_file"]) != P["sha256_shift"]):
        sys.exit("条件の表・ずれの分布が凍結の sha と違う")
    d, L = CR.frame_and_levels(with_returns(F.build(out=None)))
    keep = ((pd.to_numeric(d["year"]) >= 2025) & (d["track"].astype(str) == "良") & d["cls"].map(CR.group_of).notna()).to_numpy()
    d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True).drop(columns=["__race_id", "__horse_id"])
    d["group"] = d["cls"].map(CR.group_of)
    d = attach_final_odds(d, (2025, 2026))
    p = P2(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT2 / P["cells_file"]).read_text(encoding="utf-8"))
    shift = json.loads((OUT2 / P["shift_file"]).read_text(encoding="utf-8"))
    pre = pd.read_parquet(PRE_ODDS)
    pre = pre[pre["phase"] == "t60"]
    real = {(str(r), int(h)): (float(o), float(pp)) for r, h, o, pp in pre[["race_id", "horse_number", "win_odds", "popularity"]].itertuples(index=False)}
    unit = CR.regroup(d, p.groups).to_numpy()
    res = score(d, L, unit, kept, p, shift, real_pre=real, variants=neighbors(p) or None)
    _, _, code = _codes(L, kept)
    res["unmatched_cells"] = sum(any(lv not in code[t] for t, lv in zip(c["types"], c["levels"])) for cu in kept.values()
                                 for kind in ("buy", "kill") for c in cu[kind])
    per = []
    for u in sorted(set(unit)):
        m = unit == u
        su = score(d[m].reset_index(drop=True), L[m].reset_index(drop=True), unit[m], kept, p, shift)
        per.append({"unit": u, "optimistic": su["optimistic"], "reproduction": su["reproduction"]})
    out = {"evaluated_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "round": 2,
           "note": "2025・2026 年を見るのは 2 回目（第 1 版の評価で一度見た）。この版の条件は 2016〜2024 年だけで決め、凍結してから流した",
           "all": res, "per_unit": per, "params": P["params"], "conf_2022_2024": P["conf_2022_2024"], "races": int(d["race_id"].nunique())}
    (OUT2 / "eval_course_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    EVAL_MARK2.write_text(json.dumps(mark | {"evaluated_at": out["evaluated_at"]}), encoding="utf-8")
    a = res["reproduction"]
    print(f"評価 2025・2026（2 回目）: 再現 単回収 {a.get('roi_t')}（{a.get('roi_t_p05')}〜{a.get('roi_t_p95')}） 楽観 {res['optimistic'].get('roi_t')}"
          f" 本物の T-60 {(res.get('real_t60') or {}).get('roi_t')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shift", "cycles", "freeze", "evaluate", "_eval_inner"])
    a = ap.parse_args()
    {"shift": cmd_shift, "cycles": cmd_cycles, "freeze": cmd_freeze, "evaluate": cmd_evaluate, "_eval_inner": _eval_inner}[a.cmd]()


if __name__ == "__main__":
    main()
