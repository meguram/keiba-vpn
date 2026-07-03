"""最終オッズ予測 inference サービス。

キャッシュフロー: get_or_compute() が read-through の入口。
  1. load_cached_response() で HybridStorage (LRU → disk L2 → GCS) を確認
  2. キャッシュミス or force_refresh なら build_final_odds_response() で推論
  3. 結果を save_cached_response() で GCS + LRU に書き込み
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.pipeline.mlflow.inference_cache import InferenceCacheMixin

logger = logging.getLogger(__name__)


class FinalOddsCache(InferenceCacheMixin):
    model_key = "final_odds"
    cache_version = 1


def cache_enabled() -> bool:
    return FinalOddsCache.cache_enabled()


def load_cached_response(storage, race_id: str) -> dict | None:
    return FinalOddsCache.load_cached(storage, race_id)


def save_cached_response(
    storage, race_id: str, payload: dict, *, source: str = "api"
) -> None:
    FinalOddsCache.save_cached(storage, race_id, payload, source=source)


def _build_race_data_from_storage(race_id: str, storage) -> dict:
    """storage から build_race_features() 用 race_data を組み立てる。"""
    race_data: dict = {"race_id": race_id}

    shutuba = storage.load("race_shutuba", race_id)
    if shutuba:
        race_data["race_card"] = shutuba

    past = storage.load("race_shutuba_past", race_id)
    if past:
        race_data["shutuba_past"] = past

    oikiri = storage.load("race_oikiri", race_id)
    if oikiri:
        race_data["oikiri"] = oikiri

    race_index = storage.load("race_index", race_id)
    if race_index:
        race_data["speed_index"] = race_index

    # horse_result を並行ロード
    card = shutuba or {}
    entries = card.get("entries") or []
    horse_ids = [e.get("horse_id", "") for e in entries if e.get("horse_id")]
    if horse_ids:
        horses: dict[str, Any] = {}
        workers = min(len(horse_ids), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                hid: pool.submit(storage.load, "horse_result", hid)
                for hid in horse_ids
            }
            for hid, fut in futures.items():
                try:
                    data = fut.result()
                    if data:
                        horses[hid] = data
                except Exception:
                    pass
        race_data["horses"] = horses

    return race_data


def build_final_odds_response(
    race_id: str,
    storage,
    *,
    allow_scrape: bool = False,
) -> dict[str, Any]:
    """想定オッズ推論を実行して JSON 形式で返す。キャッシュ書き込みは呼び出し側が行う。"""
    from src.pipeline.features.feature_builder import build_race_features
    from src.pipeline.models.final_odds_predictor import predict_final_odds

    t0 = time.perf_counter()

    race_data = _build_race_data_from_storage(race_id, storage)
    shutuba = race_data.get("race_card") or {}

    if not shutuba and allow_scrape:
        from src.scraper.run import ScraperRunner

        runner = ScraperRunner(interval=1.0, cache=True, auto_login=True)
        card = runner.scrape_race_card(race_id, skip_existing=False)
        if card:
            race_data["race_card"] = card
            shutuba = card

    if not shutuba:
        return {
            "race_id": race_id,
            "error": "race_shutuba が storage にありません",
            "entries": [],
        }

    features_df = build_race_features(race_data)
    if features_df.empty:
        return {
            "race_id": race_id,
            "error": "特徴量テーブルが空です (出馬表データ不足)",
            "entries": [],
        }

    try:
        predictions = predict_final_odds(features_df)
    except Exception as exc:
        logger.warning("final_odds 推論失敗 %s: %s", race_id, exc)
        return {"race_id": race_id, "error": str(exc), "entries": []}

    entries = [
        {
            "horse_number": pred.horse_number,
            "predicted_win_odds": pred.predicted_win_odds,
            "predicted_place_odds_min": pred.predicted_place_odds_min,
            "predicted_place_odds_max": pred.predicted_place_odds_max,
            "source": getattr(pred, "source", "final_odds_ml"),
        }
        for pred in predictions
    ]

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "race_id": race_id,
        "race_name": shutuba.get("race_name", ""),
        "entries": entries,
        "elapsed_sec": elapsed,
    }


def get_or_compute(
    storage,
    race_id: str,
    *,
    force_refresh: bool = False,
    allow_scrape: bool = False,
) -> dict[str, Any]:
    """キャッシュ優先でオッズ予測を返す。キャッシュミス時は推論 → 保存。"""
    if not force_refresh:
        cached = load_cached_response(storage, race_id)
        if cached is not None:
            return cached

    payload = build_final_odds_response(race_id, storage, allow_scrape=allow_scrape)
    if not payload.get("error") and payload.get("entries"):
        save_cached_response(
            storage,
            race_id,
            payload,
            source="api" if not force_refresh else "refresh",
        )
    return payload


def precompute_final_odds_for_race_ids(
    race_ids: list[str],
    storage=None,
) -> dict[str, int]:
    """前日18時トリガー等: race_ids の想定オッズをバッチ precompute して storage に保存。"""
    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()

    ok = skip = fail = 0
    for rid in race_ids:
        try:
            payload = build_final_odds_response(rid, storage, allow_scrape=False)
            if payload.get("entries"):
                save_cached_response(
                    storage, rid, payload, source="raceday_eve_precompute"
                )
                ok += 1
            else:
                skip += 1
        except Exception as exc:
            logger.warning("final_odds precompute 失敗 %s: %s", rid, exc)
            fail += 1
    return {"ok": ok, "skip": skip, "fail": fail, "total": len(race_ids)}


def predict_odds_for_features(features_df):
    """後方互換: 特徴量 DataFrame から想定オッズを返す（キャッシュなし）。"""
    from src.pipeline.models.final_odds_predictor import predict_final_odds

    return predict_final_odds(features_df)
