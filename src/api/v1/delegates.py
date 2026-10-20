"""Flask /api/v1 向けビジネスロジック（FastAPI 非依存）。"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

# 1 レースの購入合計の既定上限（円）。UI の既定値と揃える（frontend/components/BetRecommend.tsx）。
DEFAULT_MAX_TOTAL_BET = 2000


def _predicted_odds_for(c, predicted: dict | None) -> float | None:
    """単勝・複勝候補の予測オッズ（複勝は下限、既存の fukusho_use_min_odds 方針）。"""
    if not predicted or c.bet_type not in ("tansho", "fukusho"):
        return None
    for e in predicted.get("entries") or []:
        if e.get("horse_number") == c.pair[0]:
            v = e.get("win_odds") if c.bet_type == "tansho" else e.get("place_odds_min")
            return float(v) if v else None
    return None


def optimize_betting(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer
    from src.pipeline.inference.odds_source import predicted_single_odds, resolve_odds
    from src.pipeline.inference.race_prediction_service import load_cached
    from src.scraper.storage import HybridStorage

    race_id = body.get("race_id", "")
    if not race_id:
        return {"error": "race_id is required"}, 400

    storage = HybridStorage()
    cached = load_cached(storage, race_id)
    predictions = (cached or {}).get("predictions") or []
    if not predictions:
        return {"error": "予測結果がありません。先に推論を実行してください"}, 404

    odds_mode = str(body.get("odds_mode") or "auto").lower()
    odds_phase = str(body.get("odds_phase") or "latest").lower()
    odds = resolve_odds(storage, race_id, mode=odds_mode, phase=odds_phase)
    if odds["source"] is None and odds_mode in ("actual", "predicted"):
        what = "実オッズ" if odds_mode == "actual" else "予測オッズ"
        return {
            "error": f"{what}がありません（odds_mode={odds_mode}）。{odds['note']}",
            "available_odds": odds["available"],
        }, 404

    bet_types = body.get("bet_types", ["tansho", "fukusho", "umaren", "wide"])
    allowed = odds["allowed_bet_types"]
    if allowed is not None:
        bet_types = [b for b in bet_types if b in allowed]
        if not bet_types:
            return {
                "error": "予測オッズでは単勝・複勝のみ評価できます。bet_types に tansho / fukusho を含めてください",
                "available_odds": odds["available"],
            }, 400

    pair_odds = odds["pair_odds"]
    single_odds = odds["single_odds"]
    if odds["source"] is None:  # auto でオッズ無し: 従来どおり predictions 内のオッズ欄にフォールバック
        pair_odds = storage.load("race_pair_odds", race_id) or {}
        single_odds = storage.load("race_odds", race_id)

    # 1 レースの購入合計の上限（円）。未指定は 2000 円。null / 0 は上限なし。
    raw_cap = body.get("max_total_bet", DEFAULT_MAX_TOTAL_BET)
    max_total_bet: int | None = None
    if raw_cap not in (None, 0):
        try:
            max_total_bet = int(raw_cap)
        except (TypeError, ValueError):
            max_total_bet = -1
        if max_total_bet < 100:
            return {"error": "max_total_bet は 100 円以上の整数で指定してください（上限なしは null か 0）"}, 400

    config = BettingConfig(
        max_total_bet=max_total_bet,
        bet_types=bet_types,
        kelly_fraction=body.get("kelly_fraction", 0.25),
        min_ev=body.get("min_ev", 1.05),
        min_prob=body.get("min_prob", 0.02),
        max_candidates=body.get("max_candidates", 15),
        top_n_for_pairs=body.get("top_n_for_pairs", 6),
    )
    optimizer = BettingOptimizer(config)
    portfolio = optimizer.optimize(
        predictions,
        pair_odds,
        body.get("bankroll", 100000),
        single_odds=single_odds,
    )

    predicted = None
    if odds["source"] == "actual":
        predicted = predicted_single_odds(storage, race_id)

    candidates_out = []
    for c in portfolio["candidates"]:
        item = {
            "bet_type": c.bet_type,
            "pair": list(c.pair),
            "pair_label": c.pair_label,
            "horse_names": list(c.horse_names),
            "odds": c.odds,
            "prob": c.prob,
            "ev": c.ev,
            "kelly_fraction": c.kelly_fraction,
            "bet_amount": c.bet_amount,
            "expected_return": c.expected_return,
        }
        pred_odds = _predicted_odds_for(c, predicted)
        if pred_odds:
            item["predicted_odds"] = pred_odds
            item["odds_gap"] = round(c.odds / pred_odds - 1, 3)
        candidates_out.append(item)

    calibrated = _calibrate_candidates(candidates_out)
    return {
        "race_id": race_id,
        "bankroll": body.get("bankroll", 100000),
        "total_bet": portfolio["total_bet"],
        "expected_return": portfolio["expected_return"],
        "roi_pct": round(portfolio["expected_roi"] * 100, 1),
        **calibrated,
        "max_total_bet": max_total_bet,
        "probability_model": portfolio.get("probability_model"),
        "candidates": candidates_out,
        "odds_source": odds["source"],
        "odds_phase": odds["phase"],
        "odds_as_of": odds["as_of"],
        "available_odds": odds["available"],
        "odds_note": odds["note"],
    }, 200


def _levels_now():
    from src.utils.ev_calibration import value_levels

    return value_levels()


def _calibrate_candidates(candidates_out: list[dict]) -> dict:
    """推奨買い目に、較正後の EV（ev_calibrated）と購入額加重の期待回収率を付ける。EV の較正が無効のときは何も付けない。"""
    from src.utils.ev_calibration import LABEL, calibrate_portfolio, load_config

    cal = calibrate_portfolio(candidates_out)
    if cal is None:
        return {}
    kinds = load_config().get("kinds") or {}
    for item, e in zip(candidates_out, cal["evs"]):
        item["ev_calibrated"] = round(e, 3)
        spec = kinds.get(f"cand_{item['bet_type']}")
        item["ev_calibration_validated"] = bool(spec and spec.get("validated"))
    rng = load_config().get("range") or {}
    return {"expected_roi_calibrated": cal["expected_roi"], "expected_return_calibrated": cal["expected_return"],
            "roi_pct_calibrated": None if cal["expected_roi"] is None else round(cal["expected_roi"] * 100, 1),
            "roi_calibrated_capped": bool(cal.get("capped")), "ev_label": LABEL,
            "roi_range_calibrated": [rng["lo"], rng["hi"]] if "lo" in rng and "hi" in rng else None}


# ── レースの「価値」一覧（レース選択の見た目用） ──

# 推奨ポートフォリオの期待回収率（= Σ金額×期待値 / 購入合計）でレースを 4 段階にする。
# hot: 特に高い / good: 高い / ok: プラス期待 / pass: 見送り水準（買い目なし・期待回収率が基準未満）
VALUE_LEVELS: tuple[tuple[str, float], ...] = (("hot", 1.5), ("good", 1.25), ("ok", 1.05))
_VALUE_CACHE: dict[tuple, tuple[float, dict]] = {}
_VALUE_TTL_SEC = 20.0


def value_level(expected_roi: float | None, n_bets: int) -> str:
    """expected_roi は呼び出し側で較正した値を渡す（EV の較正が有効なら較正後の期待回収率、無効なら従来の値）。閾値も較正の設定に従う。"""
    from src.utils.ev_calibration import value_levels

    if not n_bets or expected_roi is None:
        return "pass"
    for name, floor in value_levels():
        if expected_roi >= floor:
            return name
    return "pass"


def _value_row(race: dict[str, Any], payload: dict[str, Any], code: int) -> dict[str, Any]:
    rid = race["race_id"]
    base = {"race_id": rid, "round": race.get("round"), "venue": race.get("venue"), "race_name": race.get("race_name")}
    if code != 200:
        return {**base, "level": "none", "reason": payload.get("error", "計算できません"), "expected_roi": None, "n_bets": 0}
    cands = payload.get("candidates") or []
    total = payload.get("total_bet") or 0
    roi_raw = (payload.get("expected_return") or 0) / total if total else None
    roi_cal = payload.get("expected_roi_calibrated")             # EV の較正が有効なときだけ入る
    roi = roi_cal if roi_cal is not None else roi_raw
    ev_key = "ev_calibrated" if roi_cal is not None else "ev"
    top = max(cands, key=lambda c: c.get(ev_key) or 0) if cands else None
    return {
        **base,
        "level": value_level(roi, len(cands)),
        "expected_roi": None if roi is None else round(roi, 3),
        "expected_roi_raw": None if roi_raw is None else round(roi_raw, 3),
        "calibrated": roi_cal is not None,
        "max_ev": None if top is None else round(top.get(ev_key) or 0, 3),
        "n_bets": len(cands),
        "total_bet": total,
        "top_bet": None if top is None else {"bet_type": top["bet_type"], "label": top["pair_label"], "ev": round(top.get(ev_key) or 0, 3)},
        "odds_source": payload.get("odds_source"),
        "odds_phase": payload.get("odds_phase"),
    }


HOT_TOP_FRACTION = 0.2  # hot にできるのは、その日の期待回収率が上位 20% のレースだけ


def _apply_day_ranking(rows: list[dict[str, Any]]) -> None:
    """その日の期待回収率で順位を付け、hot は「絶対水準を超え、かつ上位 20%」に絞る（全レースが hot になって区別できなくなるのを防ぐ）。"""
    ranked = sorted((r for r in rows if r.get("expected_roi") is not None), key=lambda r: -r["expected_roi"])
    cutoff = max(1, math.ceil(len(ranked) * HOT_TOP_FRACTION))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        if r["level"] == "hot" and i > cutoff:
            r["level"] = "good"
    for r in rows:
        r.setdefault("rank", None)


def race_value_summary(
    date: str,
    *,
    odds_mode: str = "auto",
    odds_phase: str = "latest",
    max_total_bet: int | None = DEFAULT_MAX_TOTAL_BET,
) -> dict[str, Any]:
    """開催日の全レースについて、推奨買い目の期待回収率とレベルを返す（レース選択の強調表示に使う）。"""
    import time

    from src.scraper.storage import HybridStorage

    key = (date, odds_mode, odds_phase, max_total_bet)
    hit = _VALUE_CACHE.get(key)
    if hit and time.time() - hit[0] < _VALUE_TTL_SEC:
        return hit[1]

    storage = HybridStorage()
    races = (storage.load("race_lists", date) or {}).get("races") or []
    rows = []
    for r in races:
        if not r.get("race_id"):
            continue
        body: dict[str, Any] = {
            "race_id": r["race_id"], "bankroll": 100000, "odds_mode": odds_mode, "odds_phase": odds_phase,
            "max_total_bet": max_total_bet, "bet_types": ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"],
        }
        try:
            payload, code = optimize_betting(body)
        except Exception as exc:  # 1 レースの失敗で一覧全体を落とさない
            logger.warning("race value 計算失敗 %s: %s", r["race_id"], exc)
            payload, code = {"error": str(exc)}, 500
        rows.append(_value_row(r, payload, code))
    _apply_day_ranking(rows)
    out = {
        "date": date,
        "levels": [{"level": n, "min_roi": f} for n, f in _levels_now()],
        "max_total_bet": max_total_bet,
        "races": rows,
    }
    _VALUE_CACHE[key] = (time.time(), out)
    return out


def get_final_odds(race_id: str, refresh: bool = False) -> dict[str, Any]:
    from src.pipeline.inference.final_odds_service import get_or_compute
    from src.scraper.storage import HybridStorage

    return get_or_compute(
        HybridStorage(),
        race_id,
        force_refresh=refresh,
        allow_scrape=refresh,
    )


def get_horse_aptitude(horse_id: str | None, horse_name: str | None) -> tuple[dict, int]:
    if not horse_id and not horse_name:
        return {"error": "horse_id or horse_name required"}, 400
    try:
        from src.research.pedigree.horse_aptitude_profile import HorseAptitudeProfileCalc

        calc = HorseAptitudeProfileCalc()
        result = calc.compute(horse_id=horse_id, horse_name=horse_name)
    except FileNotFoundError as exc:
        return {"error": f"artifacts not found: {exc}"}, 500
    if "error" in result:
        return result, 404
    return result, 200


def get_race_note_3d_v2(race_id: str) -> tuple[dict, int]:
    from src.research.pedigree.race_note_3d_v2 import build_race_note_v2
    from src.scraper.storage import HybridStorage

    try:
        payload = build_race_note_v2(HybridStorage(), race_id)
    except Exception as exc:
        logger.exception("race-note-3d-v2 failed")
        return {"error": str(exc)}, 500
    if payload.get("error"):
        return payload, 404
    return payload, 200


def query_pedigree_race_stats(params: dict[str, Any]) -> tuple[dict, int]:
    """血統×レース条件クエリ — 内部 FastAPI ブリッジまたは直接 SQLite。"""
    bridge = os.environ.get("KEIBA_LEGACY_API", "").strip()
    if bridge:
        import httpx

        r = httpx.get(
            f"{bridge.rstrip('/')}/api/pedigree-race-stats/query",
            params=params,
            timeout=60.0,
        )
        return r.json(), r.status_code

    try:
        from src.pipeline.data.bloodline_sqlite import get_connection, load_race_result_slim
        import pandas as pd

        slim = load_race_result_slim(get_connection())
        if slim is None or slim.empty:
            return {
                "error": "インデックス未生成。build_pedigree_race_index + migrate_bloodline_to_sqlite を実行してください。"
            }, 503

        df = slim.copy()
        if params.get("date_from"):
            df = df[df["date"] >= params["date_from"]]
        if params.get("date_to"):
            df = df[df["date"] <= params["date_to"]]
        if params.get("venues"):
            vlist = [v.strip() for v in str(params["venues"]).split(",") if v.strip()]
            df = df[df["venue"].isin(vlist)]

        top_n = int(params.get("top_n", 50))
        return {
            "total_entries": int(len(df)),
            "unique_horses": int(df["horse_id"].nunique()) if "horse_id" in df.columns else 0,
            "cat1": [],
            "cat2": [],
            "cat3": [],
            "note": "簡易集計。フル血統クロス集計は KEIBA_LEGACY_API で FastAPI ブリッジを使用",
        }, 200
    except Exception as exc:
        return {"error": str(exc)}, 500
