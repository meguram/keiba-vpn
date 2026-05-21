"""
追走難度 API レスポンスの組み立て・storage キャッシュ。

バッチ事前計算・薄い GET API の共通ロジック。
キャッシュは ``src.pipeline.mlflow.inference_cache`` + カタログ ``tracking_difficulty``。
"""

from __future__ import annotations

import time
from typing import Any

from src.pipeline.mlflow.catalog import get_model_spec
from src.pipeline.mlflow.inference_cache import InferenceCacheMixin
from src.pipeline.mlflow.runtime import infer_backend_label
from src.utils.logger import get_logger
from src.utils.race_entries import normalize_race_entries

logger = get_logger("TrackingDifficultyService")

_SPEC = get_model_spec("tracking_difficulty")
CACHE_CATEGORY = _SPEC.cache_category or "tracking_difficulty"


class _TrackingDifficultyCache(InferenceCacheMixin):
    model_key = "tracking_difficulty"
    cache_version = 6


def cache_enabled() -> bool:
    return _TrackingDifficultyCache.cache_enabled()


def load_cached_response(storage, race_id: str) -> dict | None:
    return _TrackingDifficultyCache.load_cached(storage, race_id)


def save_cached_response(
    storage,
    race_id: str,
    payload: dict,
    *,
    source: str = "api",
) -> None:
    _TrackingDifficultyCache.save_cached(storage, race_id, payload, source=source)


def load_race_bundle(
    storage,
    race_id: str,
    *,
    allow_scrape: bool = False,
    pre_race_only: bool = True,
) -> tuple[dict, dict[str, list[dict]]]:
    """
    race_data と horse_histories を読み込む。

    pre_race_only=True（既定）: 前日18時・事前計算向け。
      race_shutuba のみ使用し、当該レースの race_result は読まない（事後リーク防止）。
    pre_race_only=False: UI の refresh 等。result があればフォールバック可。
    """
    race_data: dict = {}
    shutuba = storage.load("race_shutuba", race_id)
    if shutuba:
        shutuba = dict(shutuba)
        raw_entries = shutuba.get("entries") or []
        if raw_entries:
            shutuba["entries"] = normalize_race_entries(raw_entries)
        race_data["race_shutuba"] = shutuba
    elif not pre_race_only:
        card = storage.load("race_card", race_id)
        if card:
            race_data["race_card"] = card

    if not pre_race_only:
        result = storage.load("race_result", race_id)
        if result:
            race_data["race_result"] = result

    shutuba_data = race_data.get("race_shutuba") or race_data.get("race_card") or {}
    result_data = race_data.get("race_result") or {} if not pre_race_only else {}
    entries = shutuba_data.get("entries") or result_data.get("entries") or []
    horses_data: dict[str, list[dict]] = {}
    hids = [e.get("horse_id", "") for e in entries if e.get("horse_id")]

    if hids:
        from concurrent.futures import ThreadPoolExecutor

        def _load_hr(hid: str):
            hr = storage.load("horse_result", hid)
            if hr and "race_history" in hr:
                return hid, hr["race_history"]
            return hid, None

        workers = min(len(hids), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for hid, hist in pool.map(_load_hr, hids):
                if hist is not None:
                    horses_data[hid] = hist

    if not race_data.get("race_shutuba") and not race_data.get("race_card") and allow_scrape:
        from src.scraper.run import ScraperRunner

        runner = ScraperRunner(interval=1.0, cache=True, auto_login=True)
        if pre_race_only:
            card = runner.scrape_race_card(race_id, skip_existing=False)
            if card:
                race_data["race_shutuba"] = card
        else:
            race_data = runner.scrape_race_all(race_id, smart_skip=True)
        horses_data = {}

    if pre_race_only:
        race_data["_pre_race_only"] = True

    return race_data, horses_data


def build_tracking_difficulty_response(
    race_id: str,
    storage,
    *,
    allow_scrape: bool = False,
    pre_race_only: bool = True,
    lgbm_backend: str | None = None,
) -> dict[str, Any]:
    """
    追走難度ページ用 JSON を組み立てる（キャッシュ・MLflow serve 対応）。
    """
    from src.pipeline.models.tracking_difficulty import (
        _date_to_yyyymmdd,
        _filter_history_before_race,
        build_horse_profile,
        predict_position_flow,
        predict_race_pace,
        predict_tracking_difficulty,
    )

    t0 = time.perf_counter()
    race_data, horses_data = load_race_bundle(
        storage,
        race_id,
        allow_scrape=allow_scrape,
        pre_race_only=pre_race_only,
    )

    shutuba = race_data.get("race_shutuba") or race_data.get("race_card") or {}
    result_data = {} if pre_race_only else (race_data.get("race_result") or {})
    entries = normalize_race_entries(
        shutuba.get("entries") or result_data.get("entries") or []
    )
    if shutuba.get("entries"):
        shutuba = {**shutuba, "entries": entries}
    if not entries:
        return {
            "race_id": race_id,
            "error": "出馬表・結果データがありません",
            "entries": [],
        }

    tracking_results, field_prev_stats = predict_tracking_difficulty(
        race_data,
        horse_histories=horses_data if horses_data else None,
        storage=storage,
    )

    date_raw = str(
        shutuba.get("date", "") or result_data.get("date", "") or ""
    )
    ctx_before = _date_to_yyyymmdd(date_raw)
    ctx_rid = str(race_id)
    cur_surface = str(
        shutuba.get("surface", "") or result_data.get("surface", "") or "芝"
    )
    cur_distance = int(
        shutuba.get("distance", 0) or result_data.get("distance", 0) or 1600
    )

    horse_profiles: dict[int, dict] = {}
    for e in entries:
        hid = e.get("horse_id", "")
        hn = e.get("horse_number", 0)
        history = horses_data.get(hid, []) if horses_data else []
        hist_before = _filter_history_before_race(
            history, before_date=ctx_before, exclude_race_id=ctx_rid
        )
        horse_profiles[hn] = build_horse_profile(
            hist_before,
            surface=cur_surface,
            distance=cur_distance,
            current_jockey_id=str(e.get("jockey_id") or ""),
        )

    race_info = {
        "race_id": race_id,
        "distance": shutuba.get("distance", 0) or result_data.get("distance", 0),
        "surface": shutuba.get("surface", "") or result_data.get("surface", ""),
        "track_condition": (
            shutuba.get("track_condition", "")
            or ("" if pre_race_only else result_data.get("track_condition", ""))
            or "良"
        ),
        "venue": shutuba.get("venue", "") or result_data.get("venue", ""),
        "race_name": shutuba.get("race_name", "") or result_data.get("race_name", ""),
        "grade": shutuba.get("grade", "") or result_data.get("grade", ""),
    }

    pace_prediction = predict_race_pace(
        entries, horse_profiles, race_info, field_prev_stats=field_prev_stats,
    )
    position_flow = predict_position_flow(
        entries,
        horse_profiles,
        tracking_results,
        pace_prediction,
        field_prev_stats=field_prev_stats,
        race_info=race_info,
        horse_histories=horses_data,
    )
    if position_flow:
        bl = (position_flow[0].get("expected_last_3f") or {}).get("baseline_sec")
        if bl:
            pace_prediction = {
                **pace_prediction,
                "last_3f_baseline_sec": bl,
            }

    race_date = ctx_before

    from src.utils.race_result_availability import race_result_status

    result_status = race_result_status(storage, race_id)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    backend = lgbm_backend or infer_backend_label("tracking_difficulty")

    return {
        "race_id": race_id,
        "race_date": race_date,
        "result_viewable": result_status.get("viewable", False),
        "result_kind": result_status.get("kind"),
        "result_view_url": result_status.get("result_view_url"),
        "race_name": race_info["race_name"],
        "venue": race_info["venue"],
        "surface": race_info["surface"],
        "distance": race_info["distance"],
        "track_condition": race_info["track_condition"],
        "field_size": len(tracking_results),
        "pace_prediction": pace_prediction,
        "position_flow": position_flow,
        "entries": tracking_results,
        "field_prev_stats": field_prev_stats,
        "_compute_meta": {
            "elapsed_ms": round(elapsed_ms, 1),
            "lgbm_backend": backend,
            "model_key": "tracking_difficulty",
            "n_horses": len(tracking_results),
            "pre_race_only": pre_race_only,
            "data_source": "race_shutuba" if shutuba else "none",
        },
    }


def precompute_tracking_for_race_ids(
    race_ids: list[str],
    storage=None,
    *,
    pre_race_only: bool = True,
) -> dict[str, Any]:
    """前日 eve 等: 出馬表保存後に追走難度キャッシュを一括生成。"""
    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()

    ok = 0
    skip = 0
    fail = 0
    for rid in race_ids:
        try:
            payload = build_tracking_difficulty_response(
                rid,
                storage,
                allow_scrape=False,
                pre_race_only=pre_race_only,
            )
            n = len(payload.get("entries") or [])
            if n:
                save_cached_response(
                    storage, rid, payload, source="raceday_eve_precompute"
                )
                ok += 1
            else:
                skip += 1
        except Exception as exc:
            logger.warning("追走難度 precompute 失敗 %s: %s", rid, exc)
            fail += 1
    return {"ok": ok, "skip": skip, "fail": fail, "total": len(race_ids)}


def get_or_compute(
    storage,
    race_id: str,
    *,
    force_refresh: bool = False,
    allow_scrape: bool = False,
    pre_race_only: bool | None = None,
) -> dict[str, Any]:
    if pre_race_only is None:
        # refresh=true は直前再取得（result フォールバック可）。通常閲覧は前日想定。
        pre_race_only = not (force_refresh and allow_scrape)

    if not force_refresh:
        cached = load_cached_response(storage, race_id)
        if cached is not None:
            return cached
    payload = build_tracking_difficulty_response(
        race_id,
        storage,
        allow_scrape=allow_scrape,
        pre_race_only=pre_race_only,
    )
    if "error" not in payload or payload.get("entries"):
        save_cached_response(
            storage,
            race_id,
            payload,
            source="api" if not force_refresh else "refresh",
        )
    return payload
