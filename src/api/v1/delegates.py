"""Flask /api/v1 向けビジネスロジック（FastAPI 非依存）。"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def optimize_betting(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer
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

    bet_types = body.get("bet_types", ["tansho", "fukusho", "umaren", "wide"])
    pair_odds = storage.load("race_pair_odds", race_id) or {}
    single_odds = storage.load("race_odds", race_id)

    config = BettingConfig(
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
    candidates_out = [
        {
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
        for c in portfolio["candidates"]
    ]
    return {
        "race_id": race_id,
        "bankroll": body.get("bankroll", 100000),
        "total_bet": portfolio["total_bet"],
        "expected_return": portfolio["expected_return"],
        "roi_pct": portfolio["roi_pct"],
        "candidates": candidates_out,
    }, 200


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
