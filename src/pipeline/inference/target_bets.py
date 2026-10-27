"""推奨馬券の既定方策 `target`（目標払戻つきの 3連複）。

どれか 1 つの買い目が的中すれば、レース全体の払戻が予算の M 倍（既定 5 倍）に届くように、買い目ごとの購入額を決める。
検証（決定ログ §78）で採用した設定は「M=5・3連複のみ・最大 3 点・k=2.0・τ=1.0」。

- 的中確率: 自前の勝率・連対率・複勝率 → MB60（exotic_probabilities.fit_blend_distribution）で 3連複の全組合せ。
- 想定オッズ: 想定単勝・想定複勝（final_odds の予測）→ MB60 の確率 P_hat → exp(poly2(log(1/P_hat)))。実オッズは使わない・出さない。
- 配分: portfolio_target.select_portfolio（購入額 = ceil(M·B·k / 想定オッズ)、100 円単位、合計 ≤ B、的中確率÷購入額の大きい順）。
当たったときの払戻水準を固定する方式で、期待値・回収率 100% 超を保証しない（2025 年の検証区間は 0.99〜1.46）。
旧方策へは環境変数 KEIBA_BET_STRATEGY=legacy で戻せる。
"""

from __future__ import annotations

import json
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from functools import lru_cache
from itertools import permutations

from src.pipeline.inference import portfolio_pattern as PP
from src.pipeline.inference import portfolio_target as PT
from src.pipeline.inference.exotic_probabilities import fit_blend_distribution, normalize_marginal

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(os.environ.get("KEIBA_TARGET_BETS_CONFIG") or ROOT / "config" / "target_bets.json")
ODDS_PRED_PATH = Path(os.environ.get("KEIBA_ODDS_PRED_CONFIG") or ROOT / "config" / "odds_pred" / "odds_pred_config.json")

MARK_SYMBOLS = {"honmei": "◎", "pair": "○", "show_val": "▲", "star": "★", "anchor": "△", "kesi": "消"}
N_KEEP_EV, N_KEEP_P = 40, 25   # 検証（portfolio_target_eval）と同じ候補の刈り込み: 期待値上位 or 的中確率上位
MIN_FIELD = 5                  # 検証と同じ。5 頭未満は見送り
_cache: dict[str, Any] = {}


def strategy() -> str:
    """推奨の方策。target（既定）/ legacy。"""
    v = (os.environ.get("KEIBA_BET_STRATEGY") or "").strip().lower()
    return "legacy" if v == "legacy" else "target"


def _load(path: Path) -> dict[str, Any]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    ent = _cache.get(str(path))
    if ent and ent[0] == mtime:
        return ent[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    _cache[str(path)] = (mtime, data)
    return data


def config() -> dict[str, Any]:
    return _load(CONFIG_PATH)


def odds_pred_config() -> dict[str, Any]:
    return _load(ODDS_PRED_PATH)


def reload() -> None:
    _cache.clear()


def available() -> bool:
    """方策 target を使えるか（設定と学習済みの係数がそろっている）。"""
    return strategy() == "target" and bool(config()) and bool((odds_pred_config().get("poly2") or {}).get("sanrenpuku"))


def require_sec77() -> bool:
    """推奨馬券を、§77 のモデル（sec77）の想定オッズがあるレースにだけ出すか。検証・集計と同じ基準に固定するための安全弁。
    stg / prod（KEIBA_ENV が明示）では既定で有効。KEIBA_TARGET_REQUIRE_SEC77=0/1 で上書き。"""
    v = (os.environ.get("KEIBA_TARGET_REQUIRE_SEC77") or "").strip().lower()
    if v:
        return v not in ("0", "false", "no", "off")
    return (os.environ.get("KEIBA_ENV") or "").strip().lower() in ("stg", "prod")


def pending_odds_plan(race_id: str, cap: int | None) -> dict[str, Any]:
    """sec77 の想定オッズが未準備のレース。従来モデルの想定オッズでは推奨を作らず、理由を付けて見送りの形で返す（odds_pending=True）。"""
    budget = int(cap) if cap else default_budget()
    prm = params(budget)
    return {"strategy": "target", "race_id": race_id, "budget_yen": budget, "target_multiple": prm.mult, "safety_k": prm.k, "max_bets": prm.max_bets,
            "unit_yen": PT.UNIT, "status": "skip", "level": 0, "bets": [], "total_yen": 0, "odds_basis": "pending", "odds_pending": True,
            "skip_reason": "想定オッズ（検証と同じモデル）の準備中のため、推奨馬券はまだ出せません"}


def default_budget() -> int:
    return int(config().get("default_budget_unlimited", PT.DEFAULT_BUDGET))


def params(budget: int) -> PT.TargetParams:
    c = config().get("params") or {}
    return PT.TargetParams(mult=float(c.get("mult", 5.0)), k=float(c.get("k", 2.0)), tau=float(c.get("tau", 1.0)),
                           max_bets=int(c.get("max_bets", 3)), budget=int(budget))


def marginals_from_single(win_odds: Sequence[float], place_mid_odds: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """想定単勝・想定複勝（中央）オッズ → 勝率・連対率・複勝率（検証の odds_pred_eval._marginals と同じ）。"""
    w = 1.0 / np.clip(np.asarray(win_odds, dtype=float), 1.0, None)
    w = w / w.sum()
    p3 = normalize_marginal(1.0 / np.clip(np.asarray(place_mid_odds, dtype=float), 1.0, None), 3.0)
    p3 = np.maximum(p3, w)
    p2 = normalize_marginal(np.maximum((w + p3) / 2, w), 2.0)
    return w, p2, np.maximum(p3, p2)


def _trio_probs(dist, n: int) -> tuple[np.ndarray, np.ndarray]:
    """全 3 頭組合せの index と 3連複の確率。"""
    tri = np.array(list(combinations(range(n), 3)))
    P3 = dist._p3
    ti, tj, tk = tri[:, 0], tri[:, 1], tri[:, 2]
    pt = P3[ti, tj, tk] + P3[ti, tk, tj] + P3[tj, ti, tk] + P3[tj, tk, ti] + P3[tk, ti, tj] + P3[tk, tj, ti]
    return tri, pt


def _hat_marginals(entries: list[dict], single: dict | None, own: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """想定オッズ側の周辺確率。想定単勝・複勝があればそれ、無ければ自前確率から機械的に作る（basis で区別）。"""
    if single:
        by = {int(e["horse_number"]): e for e in (single.get("entries") or []) if e.get("horse_number") and e.get("win_odds")}
        if all(int(e["horse_number"]) in by for e in entries):
            win = [float(by[int(e["horse_number"])]["win_odds"]) for e in entries]
            mid = []
            for e in entries:
                s = by[int(e["horse_number"])]
                lo, hi = s.get("place_odds_min"), s.get("place_odds_max")
                mid.append((float(lo) + float(hi)) / 2 if lo and hi else (float(lo) if lo else max(float(s["win_odds"]) * 0.35, 1.1)))
            return marginals_from_single(win, mid), ("sec77" if single.get("odds_model") == "sec77" else "predicted")
    w, _p2, p3 = own
    return marginals_from_single(0.8 / np.clip(w, 1e-4, None), 0.8 / np.clip(p3, 0.02, None)), "own"


def sanrenpuku_candidates(entries: list[dict], single: dict | None) -> dict[str, Any] | None:
    """3連複の全組合せの的中確率 p と想定オッズ odds。entries は馬番つきの予測（win_prob / top2_prob / top3_prob）。"""
    ents = sorted((e for e in entries if int(e.get("horse_number") or 0) > 0), key=lambda e: int(e["horse_number"]))
    n = len(ents)
    if n < MIN_FIELD:
        return None
    hn = np.array([int(e["horse_number"]) for e in ents])
    w = np.array([float(e.get("win_prob") or 0.0) for e in ents])
    t2 = np.array([float(e.get("top2_prob") or e.get("place_prob") or 0.0) for e in ents])
    t3 = np.array([float(e.get("top3_prob") or e.get("show_prob") or 0.0) for e in ents])
    if w.sum() <= 0 or t3.sum() <= 0:
        return None
    w = w / w.sum()
    own_dist, _ = fit_blend_distribution(w, t2, t3)
    own = (own_dist.win, own_dist.top2(), own_dist.top3() if n >= 8 else own_dist.top2())
    (wh, p2h, p3h), basis = _hat_marginals(ents, single, own)
    hat_dist, _ = fit_blend_distribution(wh, p2h, p3h)
    tri, p_own = _trio_probs(own_dist, n)
    _, p_hat = _trio_probs(hat_dist, n)
    co = (odds_pred_config().get("poly2") or {}).get("sanrenpuku")
    if not co:
        return None
    odds = np.clip(np.exp(np.polyval(co, np.log(1.0 / np.clip(p_hat, 1e-9, None)))), 1.0, None)
    ev = p_own * odds
    # 刈り込み（検証と同じ）: 期待値上位 N_KEEP_EV 件 または 的中確率上位 N_KEEP_P 件
    r_ev = (-ev).argsort(kind="stable").argsort(kind="stable") + 1
    r_p = (-p_own).argsort(kind="stable").argsort(kind="stable") + 1
    keep = (r_ev <= N_KEEP_EV) | (r_p <= N_KEEP_P)
    return {"hn": hn, "tri": tri[keep], "p": p_own[keep], "odds": odds[keep], "basis": basis, "entries": ents, "n": n}


def prioritize(bets: list[dict], evs: list[float]) -> list[dict]:
    """買い目を推奨度の高い順（想定オッズで見た期待値＝的中確率×想定倍率の大きい順。同値は的中率の高い順）に並べ、priority（1 が最優先）を付ける。
    期待値そのものは応答に出さない。利用者は上から、直前の実オッズが期待値ライン以上のものを選ぶ。"""
    order = sorted(range(len(bets)), key=lambda i: (-evs[i], -bets[i]["hit_rate_pct"]))
    out = []
    for rank, i in enumerate(order, 1):
        out.append({**bets[i], "priority": rank})
    return out


def _ev_line(p: float) -> float | None:
    """期待値ライン: 実際のオッズ（倍）がこの値以上なら、自前の的中確率から見て期待値が 1 以上（損益分岐＝1÷的中確率）。"""
    p = float(p)
    return round(1.0 / p, 1) if p > 0 else None


def tier_level(total_hit_prob: float | None) -> int:
    """推奨度（1〜4）: 選んだ買い目のどれかが当たる確率の大きさ。しきい値は config/target_bets.json（検証期間の四分位）。"""
    if total_hit_prob is None:
        return 0
    th = [float(x) for x in (config().get("tier_thresholds") or [0.05, 0.075, 0.11])]
    return 1 + sum(total_hit_prob >= t for t in th)


def build_target_portfolio(entries: list[dict], single: dict | None, max_total_bet: int | None, *, race_id: str = "") -> dict[str, Any]:
    """推奨ポートフォリオ。戻り値は API 応答の本体（view に依らない共通部分）。"""
    try:
        pl = build_plans(entries, single, max_total_bet, race_id=race_id)
    except Exception:  # noqa: BLE001
        logger.exception("プラン方式の計算に失敗したため 3連複のみの方策にフォールバック: %s", race_id)
        pl = None
    if pl is not None:
        return pl
    budget = int(max_total_bet) if max_total_bet else default_budget()
    prm = params(budget)
    base = {"strategy": "target", "race_id": race_id, "budget_yen": budget, "target_multiple": prm.mult, "safety_k": prm.k,
            "max_bets": prm.max_bets, "unit_yen": PT.UNIT}
    cand = sanrenpuku_candidates(entries, single)
    if cand is None:
        return {**base, "status": "skip", "skip_reason": "出走頭数が少ない、または確率が得られないため計画できません", "bets": [], "total_yen": 0,
                "level": 0}
    idx, stakes = PT.select_portfolio(cand["p"], cand["odds"], prm)
    if len(idx) == 0:
        return {**base, "status": "skip", "skip_reason": "払戻が購入額の約%g倍に届く買い目の組み合わせがありません（条件を満たす買い目なし）" % prm.mult,
                "bets": [], "total_yen": 0, "level": 0, "odds_basis": cand["basis"]}
    by_hn = {int(e["horse_number"]): e for e in cand["entries"]}
    bets = []
    for j, st in zip(idx, stakes):
        combo = [int(cand["hn"][i]) for i in cand["tri"][j]]
        legs = [{"horse_number": h, "horse_name": by_hn[h].get("horse_name") or "", "mark": MARK_SYMBOLS.get(str(by_hn[h].get("mark_type") or ""), "")} for h in combo]
        mult = float(cand["odds"][j])
        bets.append({"bet_type": "sanrenpuku", "bet_label": "3連複", "combo": combo, "combo_label": "-".join(str(h) for h in combo), "legs": legs,
                     "yen": int(st), "est_multiple": round(mult, 1), "est_payout_yen": int(round(st / PT.UNIT * mult * PT.UNIT, -1)),
                     "hit_rate_pct": round(float(cand["p"][j]) * 100, 2), "ev_line_multiple": _ev_line(cand["p"][j])})
    bets = prioritize(bets, [float(cand["p"][j]) * float(cand["odds"][j]) for j in idx])
    total = int(sum(b["yen"] for b in bets))
    any_hit = float(min(sum(cand["p"][j] for j in idx), 1.0))
    return {**base, "status": "bet", "skip_reason": None, "bets": bets, "total_yen": total, "any_hit_rate_pct": round(any_hit * 100, 1),
            "target_payout_yen": int(prm.mult * budget), "level": tier_level(any_hit), "odds_basis": cand["basis"],
            **_portfolio_ev(cand, np.asarray(idx), np.asarray(stakes))}


# ───────────────────────── 全券種・プラン方式 ─────────────────────────
BT_KEY = {0: "tansho", 1: "fukusho", 2: "umaren", 3: "wide", 4: "umatan", 5: "sanrenpuku", 6: "sanrentan"}
BT_JA = {0: "単勝", 1: "複勝", 2: "馬連", 3: "ワイド", 4: "馬単", 5: "3連複", 6: "3連単"}
BT_SEP = {0: "", 1: "", 2: "-", 3: "-", 4: ">", 5: "-", 6: ">"}


@lru_cache(maxsize=24)
def _perms(n: int) -> PP.Perms:
    return PP.Perms(n)


def _type_arrays(dist, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """券種 0〜6 ごとの (組の馬 index 配列[m,k], 確率[m])。検証（odds_pred_eval）と同じ定義。"""
    P2, P3 = dist._p2, dist._p3
    iu, ju = np.triu_indices(n, 1)
    T = P3 + P3.transpose(0, 2, 1) + P3.transpose(1, 0, 2) + P3.transpose(1, 2, 0) + P3.transpose(2, 0, 1) + P3.transpose(2, 1, 0)
    ii, jj = np.where(~np.eye(n, dtype=bool))
    tri = np.array(list(combinations(range(n), 3)))
    perm = _perms(n)
    one = np.arange(n)[:, None]
    return [(one, dist.win), (one, dist.top3() if n >= 8 else dist.top2()),
            (np.stack([iu, ju], 1), (P2 + P2.T)[iu, ju]), (np.stack([iu, ju], 1), T.sum(axis=2)[iu, ju]),
            (np.stack([ii, jj], 1), P2[ii, jj]), (tri, T[tri[:, 0], tri[:, 1], tri[:, 2]]),
            (np.stack([perm.a, perm.b, perm.c], 1), P3[perm.a, perm.b, perm.c])]


def _single_odds(ents: list[dict], single: dict | None, own_w: np.ndarray, own_p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_lo = float(config().get("r_lo", 0.7705))
    if single:
        by = {int(e["horse_number"]): e for e in (single.get("entries") or []) if e.get("horse_number") and e.get("win_odds")}
        if all(int(e["horse_number"]) in by for e in ents):
            win, lo = [], []
            for e in ents:
                s_ = by[int(e["horse_number"])]
                a, b = s_.get("place_odds_min"), s_.get("place_odds_max")
                win.append(float(s_["win_odds"]))
                lo.append(float(a) if a else (float(b) * r_lo if b else max(float(s_["win_odds"]) * 0.3, 1.1)))
            return np.array(win), np.array(lo)
    return 0.8 / np.clip(own_w, 1e-4, None), r_lo * 0.8 / np.clip(own_p3, 0.02, None)


def all_candidates(entries: list[dict], single: dict | None) -> dict[str, Any] | None:
    """単勝〜3連単の全候補（的中確率は自前 MB60、想定オッズは odds_pred の構成）。複数頭券種は期待値上位40＋確率上位25に刈り込み（検証と同じ）。"""
    ents = sorted((e for e in entries if int(e.get("horse_number") or 0) > 0), key=lambda e: int(e["horse_number"]))
    n = len(ents)
    if n < MIN_FIELD:
        return None
    hn = np.array([int(e["horse_number"]) for e in ents])
    w = np.array([float(e.get("win_prob") or 0.0) for e in ents])
    t2 = np.array([float(e.get("top2_prob") or e.get("place_prob") or 0.0) for e in ents])
    t3 = np.array([float(e.get("top3_prob") or e.get("show_prob") or 0.0) for e in ents])
    if w.sum() <= 0 or t3.sum() <= 0:
        return None
    w = w / w.sum()
    own_dist, _ = fit_blend_distribution(w, t2, t3)
    own = (own_dist.win, own_dist.top2(), own_dist.top3() if n >= 8 else own_dist.top2())
    (wh, p2h, p3h), basis = _hat_marginals(ents, single, own)
    hat_dist, _ = fit_blend_distribution(wh, p2h, p3h)
    co = odds_pred_config().get("poly2") or {}
    if not all(BT_KEY[t] in co for t in range(2, 7)):
        return None
    win_odds, place_lo = _single_odds(ents, single, own[0], own[2])
    A, H = _type_arrays(own_dist, n), _type_arrays(hat_dist, n)
    bts, combos, ps, os_ = [], [], [], []
    for t in range(7):
        idx, p_own = A[t]
        if t == 0:
            odds = win_odds
        elif t == 1:
            odds = place_lo
        else:
            odds = np.clip(np.exp(np.polyval(co[BT_KEY[t]], np.log(1.0 / np.clip(H[t][1], 1e-12, None)))), 1.0, None)
        if t >= 2:
            ev = p_own * odds
            r_ev = (-ev).argsort(kind="stable").argsort(kind="stable") + 1
            r_p = (-p_own).argsort(kind="stable").argsort(kind="stable") + 1
            keep = (r_ev <= N_KEEP_EV) | (r_p <= N_KEEP_P)
            idx, p_own, odds = idx[keep], p_own[keep], odds[keep]
        bts.append(np.full(len(p_own), t)); combos += [tuple(int(x) for x in r) for r in idx]; ps.append(p_own); os_.append(odds)
    perm = _perms(n)
    pp = own_dist._p3[perm.a, perm.b, perm.c]
    return {"bt": np.concatenate(bts), "combo": combos, "p": np.concatenate(ps), "odds": np.concatenate(os_), "perm_p": pp / pp.sum(),
            "perms": perm, "hn": hn, "entries": ents, "basis": basis, "n": n}


def _portfolio_ev(cand: dict, idx: np.ndarray, stakes: np.ndarray) -> dict[str, Any]:
    """買い目全体を購入したときの期待払戻（円）と期待回収率（倍）。『的中確率 × 購入額 × 想定倍率』の和（同時に当たる組合せがあっても期待値は足し算でよい）。"""
    st = np.asarray(stakes, dtype=float)
    ret = float(np.sum(np.asarray(cand["p"])[idx] * st * np.asarray(cand["odds"])[idx]))
    total = float(np.sum(st))
    return {"expected_payout_yen": int(round(ret, -1)), "portfolio_ev": round(ret / total, 2) if total > 0 else None}


def _fmt_bets(cand: dict, idx: np.ndarray, stakes: np.ndarray) -> list[dict]:
    by_hn = {int(e["horse_number"]): e for e in cand["entries"]}
    bets = []
    for j, st in zip(idx, stakes):
        t = int(cand["bt"][j])
        nums = [int(cand["hn"][i]) for i in cand["combo"][j]]
        if t in (2, 3, 5):
            nums = sorted(nums)
        legs = [{"horse_number": h, "horse_name": by_hn[h].get("horse_name") or "", "mark": MARK_SYMBOLS.get(str(by_hn[h].get("mark_type") or ""), "")} for h in nums]
        mult = float(cand["odds"][j])
        bets.append({"bet_type": BT_KEY[t], "bet_label": BT_JA[t], "combo": nums, "combo_label": BT_SEP[t].join(str(h) for h in nums) if t not in (0, 1) else str(nums[0]),
                     "legs": legs, "yen": int(st), "est_multiple": round(mult, 1), "est_payout_yen": int(round(st / PT.UNIT * mult * PT.UNIT, -1)),
                     "hit_rate_pct": round(float(cand["p"][j]) * 100, 2), "ev_line_multiple": _ev_line(cand["p"][j])})
    return prioritize(bets, [float(cand["p"][j]) * float(cand["odds"][j]) for j in idx])


def race_ev_level(ev: float | None) -> int:
    """レース期待値（パターンの期待回収率）に基づく推奨度 0〜4。しきい値は config の race_ev_tiers（先頭が推奨するレースの下限）。"""
    if ev is None:
        return 0
    th = [float(x) for x in (config().get("race_ev_tiers") or [2.0, 2.15, 2.41, 2.66])]
    return sum(ev >= t for t in th)


def build_plans(entries: list[dict], single: dict | None, max_total_bet: int | None, *, race_id: str = "") -> dict[str, Any] | None:
    cfg = config()
    plans_cfg = cfg.get("plans") or []
    if not plans_cfg:
        return None
    cand = all_candidates(entries, single)
    if cand is None:
        return None
    budget = int(max_total_bet) if max_total_bet else default_budget()
    out = []
    for pc in plans_cfg:
        item = {"id": pc["id"], "label": pc["label"], "note": pc.get("note", ""), "evidence": pc.get("evidence", ""), "kind": pc["kind"]}
        if pc["kind"] == "pattern":
            pat = cfg.get("pattern") or {}
            prm = PP.PatternParams(mult=float(pat.get("mult", 5.0)), k=float(pat.get("k", 2.0)), tau=float(pat.get("tau", 1.0)), bet_ev_min=float(pat.get("bet_ev_min", 0.9)),
                                   top_n=int(pat.get("top_n", 50)), max_k=int(pat.get("max_k", 5)), beam=int(pat.get("beam", 12)), budget=budget)
            r = PP.select_pattern(cand["bt"], cand["combo"], cand["p"], cand["odds"], cand["perm_p"], cand["perms"], prm)
            idx, st, ev, hit = r["idx"], r["stakes"], r["ev"], r["hit_prob"]
            thr = float(cfg.get("race_ev_threshold", 2.0))
            if len(idx) == 0:
                item.update(status="skip", skip_reason="払戻が購入額の約%g倍に届く買い目の組み合わせがありません" % prm.mult, bets=[], total_yen=0, level=0, race_ev=None)
            elif ev is None or ev < thr:
                item.update(status="skip", skip_reason="レースの期待値が水準（%.2f）に届きません（このレース %.2f）" % (thr, ev or 0.0), bets=[], total_yen=0, level=0, race_ev=round(ev, 2) if ev else None)
            else:
                item.update(status="bet", skip_reason=None, bets=_fmt_bets(cand, idx, st), total_yen=int(np.sum(st)), any_hit_rate_pct=round(hit * 100, 1),
                            level=race_ev_level(ev), race_ev=round(ev, 2), target_payout_yen=int(prm.mult * budget), **_portfolio_ev(cand, idx, st))
        else:
            prm = PT.TargetParams(mult=float((cfg.get("pattern") or {}).get("mult", 5.0)), k=float(pc.get("k", 2.0)), tau=float(pc.get("tau", 1.0)),
                                  max_bets=pc.get("max_bets"), budget=budget)
            idx, st = PT.select_portfolio(cand["p"], cand["odds"], prm, np.isin(cand["bt"], pc.get("bt") or []))
            if len(idx) == 0:
                item.update(status="skip", skip_reason="このプランの条件を満たす買い目がありません", bets=[], total_yen=0, level=0, race_ev=None)
            else:
                any_hit = float(min(np.sum(cand["p"][idx]), 1.0))
                item.update(status="bet", skip_reason=None, bets=_fmt_bets(cand, idx, st), total_yen=int(np.sum(st)), any_hit_rate_pct=round(any_hit * 100, 1),
                            level=tier_level(any_hit), race_ev=None, target_payout_yen=int(prm.mult * budget), **_portfolio_ev(cand, idx, st))
        out.append(item)
    default_id = cfg.get("default_plan") or out[0]["id"]
    d = next((x for x in out if x["id"] == default_id), out[0])
    base = {"strategy": "target", "race_id": race_id, "budget_yen": budget, "target_multiple": float((cfg.get("pattern") or {}).get("mult", 5.0)),
            "safety_k": float((cfg.get("pattern") or {}).get("k", 2.0)), "unit_yen": PT.UNIT, "odds_basis": cand["basis"], "default_plan": d["id"], "plans": out}
    keys = ("status", "skip_reason", "bets", "total_yen", "level", "race_ev", "any_hit_rate_pct", "target_payout_yen", "expected_payout_yen", "portfolio_ev")
    return {**base, **{k: d[k] for k in keys if k in d}, "max_bets": len(d.get("bets") or [])}
