"""めぐサイクルの評価器: 買い方の成績（回収率・件数・市場比・同じ人気の基準線・開催日で束ねた区間・並べ替えの帰無）と、2022〜24 の確かめの台帳。

買い方は「表 d と、判定のオッズの版 k を受け取り、1 走ごとの賭け金（0 なら買わない）を返す関数」。券種は単勝（ret_tan）・複勝（ret_fuku）。
判定のオッズは T-60 の再現の 5 通り（odds_pre0〜4）。成績は 5 通りの平均（回収率は 5 通りの払戻の和 ÷ 賭け金の和）。
市場比 = 勝ち数（複勝は 3 着内の数）÷ 市場の見込みの和。見込みは確定の単勝オッズの 1/オッズ をレースで和 1 に直した値（複勝は Harville の 3 着内）。
確定の市場は物差しにだけ使い、判定には使わない。

費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.research.megu_cycles import mc_data as D

GOAL = json.loads(D.GOAL.read_text(encoding="utf-8"))
POP_CAP = 13
ODDS_BANDS = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e9])
Strategy = Callable[[pd.DataFrame, int], np.ndarray]


def period_mask(d: pd.DataFrame, name: str) -> np.ndarray:
    a, b = GOAL["periods"][name]
    s = d["date"].astype(str).str[:10]
    return ((s >= a) & (s <= b)).to_numpy()


def seg_mask(d: pd.DataFrame, seg: str) -> np.ndarray:
    g = d["graded"].to_numpy(bool)
    return g if seg == "graded" else ~g


def race_norm(d: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    s = pd.Series(x).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    return x / np.where(s > 0, s, np.nan)


def q_pre(d: pd.DataFrame, k: int) -> np.ndarray:
    return race_norm(d, 1.0 / d[f"odds_pre{k}"].to_numpy(float))


def harville_top3(d: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """勝率 p（レースで和 1）から、Harville の 3 着内の確率。"""
    out = np.zeros(len(d))
    rid = d["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(d)]
    for s, e in zip(starts, ends):
        q = p[s:e]
        n = len(q)
        if n <= 2:
            out[s:e] = 1.0
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            m = q[:, None] * q[None, :] / (1 - q[:, None])            # m[i, j] = P(1 着 i, 2 着 j)
            np.fill_diagonal(m, 0)
            w = m / (1 - q[:, None] - q[None, :])
            np.fill_diagonal(w, 0)
        m, w = np.nan_to_num(m, posinf=0), np.nan_to_num(w, posinf=0)
        p2 = m.sum(0)
        p3 = q * (w.sum() - w.sum(1) - w.sum(0))                     # 3 着 k = Σ_{i,j≠k} m[i,j]·q_k/(1−q_i−q_j)
        out[s:e] = np.clip(q + p2 + (p3 if n >= 8 else 0.0), 0, 1)       # 7 頭以下の複勝は 2 着まで
    return out


_H3: dict = {}


def harville_top3_final(d: pd.DataFrame) -> np.ndarray:
    key = (hashlib.md5(d["q_final"].to_numpy(float).tobytes() + d["race_id"].astype(str).str.cat().encode()).hexdigest(), "final")
    if key not in _H3:
        _H3[key] = harville_top3(d, d["q_final"].to_numpy(float))
    return _H3[key]


def harville_top3_cached(d: pd.DataFrame, k: int) -> np.ndarray:
    key = (hashlib.md5(d[f"odds_pre{k}"].to_numpy(float).tobytes() + d["race_id"].astype(str).str.cat().encode()).hexdigest(), k)
    if key not in _H3:
        _H3[key] = harville_top3(d, q_pre(d, k))
    return _H3[key]


def _ret(d: pd.DataFrame, kind: str) -> np.ndarray:
    """1 単位の賭け金あたりの払戻。winplace は単勝と複勝に半分ずつ。"""
    if kind == "winplace":
        return (_ret(d, "win") + _ret(d, "place")) / 2
    return np.nan_to_num(d["ret_tan" if kind == "win" else "ret_fuku"].to_numpy(float))


def place_slots(d: pd.DataFrame) -> np.ndarray:
    """複勝の払われる着順の数（JRA: 7 頭以下は 2 着まで、8 頭以上は 3 着まで）。"""
    f = pd.to_numeric(d["field"], errors="coerce").to_numpy() if "field" in d else np.full(len(d), 8)
    return np.where(f <= 7, 2, 3)


def _hit(d: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "winplace":
        return (_hit(d, "win") + _hit(d, "place")) / 2
    if kind == "win":
        return d["y1"].fillna(0).to_numpy(float)
    fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy() if "fin" in d else np.where(d["y3"].fillna(0) > 0, 1, 99)
    return (fin <= place_slots(d)).astype(float)


def baseline_same_pop(d: pd.DataFrame, stake: np.ndarray, kind: str) -> float | None:
    """同じ人気の基準線: 同じ表（期間・区分で絞った d）の確定の人気ごとの全馬の回収率を、買った馬の人気の分布（賭け金）で重みづけ。"""
    pop = np.clip(np.nan_to_num(d["pop_final"].to_numpy(float), nan=POP_CAP), 1, POP_CAP).astype(int)
    r = _ret(d, kind)
    roi_pop = {p: r[pop == p].mean() for p in np.unique(pop)}
    w = pd.Series(stake).groupby(pop).sum()
    if w.sum() <= 0:
        return None
    return float(sum(roi_pop[p] * w[p] for p in w.index) / w.sum())


def evaluate(d: pd.DataFrame, strat: Strategy, kind: str = "win", n_boot: int = 2000, seed: int = 0, ks=range(D.N_PRE)) -> dict:
    """d（期間・区分で絞った表）での成績。5 通りの判定のオッズの平均。"""
    ks = list(ks)
    S = np.vstack([np.asarray(strat(d, k), float) for k in ks])          # (版, 行)
    S = np.where(np.isfinite(S) & (S > 0), S, 0.0)
    ret, hit = _ret(d, kind), _hit(d, kind)
    stake = S.sum(0) / len(ks)
    pay = stake * ret
    # 市場比の物差しは確定の市場（払戻を決める市場）。判定には使わない。T-60 の再現の q は人気馬を低く見るので、物差しにすると市場比が水増しになる（サイクル 1 で分かった）
    Q = (d["q_final"].to_numpy(float) if kind == "win" else harville_top3_final(d) if kind == "place"
         else (d["q_final"].to_numpy(float) + harville_top3_final(d)) / 2)
    exp_hit = (S * Q[None, :]).sum(0) / len(ks)
    hits = stake * hit
    n_bets = (S > 0).sum(1).mean()
    years = d["year"].nunique()
    tot_s, tot_p = stake.sum(), pay.sum()
    out = {"kind": kind, "bets": float(n_bets), "bets_per_year": float(n_bets / max(years, 1)), "races": int(d.loc[stake > 0, "race_id"].nunique()),
           "stake": float(tot_s), "roi": float(tot_p / tot_s) if tot_s > 0 else None,
           "hit_rate": float(hits.sum() / tot_s) if tot_s > 0 else None,
           "market_ratio": float(hits.sum() / exp_hit.sum()) if exp_hit.sum() > 0 else None,
           "base_same_pop": baseline_same_pop(d, stake, kind), "years": int(years)}
    if out["roi"] is not None and out["base_same_pop"] is not None:
        out["roi_minus_base"] = out["roi"] - out["base_same_pop"]
    if tot_s > 0 and n_boot:
        dates = d["date"].astype(str).str[:10].to_numpy()
        g = pd.DataFrame({"dt": dates, "s": stake, "p": pay, "h": hits, "e": exp_hit}).groupby("dt").sum()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        A = g.to_numpy()
        B = A[idx].sum(1)
        roi_b = B[:, 1] / np.where(B[:, 0] > 0, B[:, 0], np.nan)
        mr_b = B[:, 2] / np.where(B[:, 3] > 0, B[:, 3], np.nan)
        out["roi_ci"] = [float(np.nanpercentile(roi_b, 2.5)), float(np.nanpercentile(roi_b, 97.5))]
        out["mr_ci"] = [float(np.nanpercentile(mr_b, 2.5)), float(np.nanpercentile(mr_b, 97.5))]
        out["p_roi_le_1"] = float(np.mean(roi_b <= 1.0))
    return out


def null_perm(d: pd.DataFrame, strat: Strategy, cols: list[str], kind: str = "win", n: int = 50, seed: int = 0) -> dict:
    """並べ替えの帰無: 区分 × 確定のオッズ帯の層の中で、判定に使う列（cols）を入れ替えて、同じ買い方の回収率の分布を作る。"""
    band = np.searchsorted(ODDS_BANDS[1:-1], d["odds_final"].to_numpy(float), side="right")
    key = pd.Series(d["graded"].astype(int).to_numpy() * 100 + band)
    groups = [v for v in key.groupby(key).indices.values() if len(v) > 1]
    real = evaluate(d, strat, kind, n_boot=0)["roi"]
    rng = np.random.default_rng(20261123 + seed)
    vals = []
    for _ in range(n):
        e = d.copy()
        perm = np.arange(len(d))
        for idx in groups:
            perm[idx] = idx[rng.permutation(len(idx))]
        for c in cols:
            e[c] = d[c].to_numpy()[perm]
        r = evaluate(e, strat, kind, n_boot=0)["roi"]
        vals.append(np.nan if r is None else r)
    vals = np.array(vals, float)
    return {"real": real, "null_mean": float(np.nanmean(vals)), "null_p95": float(np.nanpercentile(vals, 95)),
            "p": float((np.sum(vals >= (real or 0)) + 1) / (np.sum(np.isfinite(vals)) + 1))}


def gate(a: dict, b: dict, null: dict | None) -> tuple[bool, list[str]]:
    """2022〜24 に当てる前の関門（goal.json の gate_to_confirm）。"""
    G = GOAL["gate_to_confirm"]
    why = []
    if (a.get("roi") or 0) < G["roi_select_a_min"]:
        why.append(f"2017〜19 の回収率 {a.get('roi')} < 1")
    if (b.get("roi") or 0) < G["roi_select_b_min"]:
        why.append(f"2020〜21 の回収率 {b.get('roi')} < 1")
    if a.get("roi") and b.get("roi") and b["roi"] < G["select_b_over_a_min"] * a["roi"]:
        why.append("2020〜21 が 2017〜19 の 0.9 倍未満")
    if (b.get("mr_ci") or [0])[0] <= G["market_ratio_select_b_ci_low_min"]:
        why.append(f"2020〜21 の市場比の区間の下限 {b.get('mr_ci')} ≤ 1")
    if b.get("bets_per_year", 0) < G["select_b_bets_per_year_min"]:
        why.append(f"2020〜21 の件数 {b.get('bets_per_year')}/年 < 50")
    if null is None:
        why.append("並べ替えの帰無を流していない")
    elif null["p"] > 0.05:
        why.append(f"並べ替えの帰無 p={null['p']:.3f} > 0.05")
    return (not why), why


def goal_met(s: dict, c: dict) -> tuple[bool, list[str]]:
    G = GOAL["goal"]
    why = []
    if (s.get("roi") or 0) < G["roi_select_min"]:
        why.append("選ぶ期間 < 100%")
    if (c.get("roi") or 0) < G["roi_confirm_min"]:
        why.append("確かめ < 100%")
    if s.get("roi") and c.get("roi") and c["roi"] < G["confirm_over_select_min"] * s["roi"]:
        why.append("確かめ < 選ぶ期間 × 0.9")
    if c.get("bets_per_year", 0) < G["confirm_bets_per_year_min"]:
        why.append("確かめの件数 < 50/年")
    return (not why), why


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def confirm(d_all: pd.DataFrame, strat: Strategy, spec: dict, kind: str, seg: str, reason: str) -> dict:
    """2022〜24 に当てる（関門を通った候補だけ）。台帳に confirm を申告する。通算 max_candidates まで。"""
    from src.ops import data_ledger as DL
    log = confirm_log()
    n_led = ledger_confirms()
    if max(len(log), n_led) >= GOAL["confirm_budget"]["max_candidates"]:
        raise SystemExit("2022〜24 に当てる候補の上限に達した")
    v = spec_hash(spec)
    if any(r.get("version") == v for r in log):
        raise SystemExit(f"同じ設定（{v}）はもう 2022〜24 に当てた")
    frozen = [D.GOAL, D.ROOT / "src/research/megu_cycles/mc_eval.py", D.ROOT / "src/research/megu_cycles/cycles.py", D.ROOT / "src/research/megu_cycles/mc_data.py"]
    with DL.guard(tool="src/research/megu_cycles/mc_eval.py:confirm", family="megu_cycles", version=v, period=("2022-01-01", "2024-12-31"),
                  purpose="confirm", reason=reason, frozen_paths=frozen, meta={"spec": spec, "n_confirm_before": max(len(log), n_led)}) as tok:
        m = period_mask(d_all, "confirm") & seg_mask(d_all, seg)
        e = d_all[m].reset_index(drop=True)
        tok.check_dates(e["date"].astype(str))
        r = evaluate(e, strat, kind)
        tok.finish()
    s_m = period_mask(d_all, "select") & seg_mask(d_all, seg)
    s = evaluate(d_all[s_m].reset_index(drop=True), strat, kind)
    ok, why = goal_met(s, r)
    row = {"version": v, "spec": spec, "kind": kind, "seg": seg, "select": s, "confirm": r, "goal_met": ok, "why": why}
    log.append(row)
    CONFIRM_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return row


CONFIRM_LOG = D.ROOT / "docs/operations/megu_cycles/confirm_log.json"


def ledger_confirms() -> int:
    """台帳の megu_cycles の confirm の数（2022〜24）。手元の記録と食い違っても、多い方で上限を数える。"""
    from src.ops import data_ledger as DL
    rows = DL.load_rows()
    return sum(1 for r in rows if (r.get("target") or {}).get("family") == "megu_cycles" and r.get("requested_purpose", r.get("purpose")) == "confirm"
               and r.get("kind") in ("start", "record"))


def confirm_log() -> list:
    return json.loads(CONFIRM_LOG.read_text(encoding="utf-8")) if CONFIRM_LOG.exists() else []
