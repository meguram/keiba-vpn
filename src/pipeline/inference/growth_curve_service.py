"""
成長曲線 API レスポンスの組み立て・ローカル蓄積。

- 正本: ``data/calculated_data/growth_curve/{horse_id}.json``
- GET はローカル優先。計算成功時は常にローカルへ保存（随時蓄積）
- 金曜 18:00: horse_result が存在する馬のみ、鮮度 7 日以内はスキップ
- API 既定は中央競馬（JRA）のみ表示。全会場は ``jra_only=false``
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.pipeline.inference.growth_curve_store import (
    apply_limit,
    is_local_fresh,
    load_local,
    save_local,
    update_index_meta,
)
from src.scraper.export_tables import is_jra_race
from src.utils.logger import get_logger

logger = get_logger("GrowthCurveService")

# race_id 欠損キャッシュ向け: netkeiba 会場表記に中央 10 競馬場が含まれるか
_JRA_VENUE_MARKERS = (
    "札幌",
    "函館",
    "福島",
    "新潟",
    "東京",
    "中山",
    "中京",
    "京都",
    "阪神",
    "小倉",
)


def is_jra_growth_race_row(race: dict) -> bool:
    """成長曲線 1 行が中央競馬か（race_id 優先、無ければ会場表記）。"""
    rid = str(race.get("race_id") or "").strip()
    if rid and len(rid) >= 6:
        return is_jra_race(rid)
    venue = str(race.get("venue") or "")
    return any(m in venue for m in _JRA_VENUE_MARKERS)


def _recompute_race_aggregates(payload: dict) -> None:
    """payload['races'] から total_races / 平均体重 / 着順系を再計算。"""
    races: list[dict] = list(payload.get("races") or [])
    weights: list[float] = []
    for r in races:
        w = r.get("weight")
        if w is None:
            continue
        try:
            weights.append(float(w))
        except (TypeError, ValueError):
            continue
    ranks: list[int] = []
    for r in races:
        rk = r.get("rank")
        if rk is None:
            continue
        try:
            ranks.append(int(rk))
        except (TypeError, ValueError):
            continue
    payload["total_races"] = len(races)
    payload["avg_weight"] = sum(weights) / len(weights) if weights else 0.0
    payload["weight_range"] = [min(weights), max(weights)] if weights else [0, 0]
    payload["best_rank"] = min(ranks) if ranks else None
    payload["avg_rank"] = sum(ranks) / len(ranks) if ranks else None


def filter_growth_curve_for_jra(payload: dict, *, jra_only: bool) -> dict:
    """レスポンス用に中央のみ／全会場へ切り替え（正本 races は書き換えないようコピー）。"""
    out = dict(payload)
    races_all: list[dict] = list(payload.get("races") or [])
    if not jra_only:
        out["jra_filter_active"] = False
        out["excluded_non_jra_count"] = 0
        return out
    jra_races = [dict(r) for r in races_all if is_jra_growth_race_row(r)]
    out["races"] = jra_races
    out["jra_filter_active"] = True
    out["excluded_non_jra_count"] = max(0, len(races_all) - len(jra_races))
    _recompute_race_aggregates(out)
    return out

_HORSE_RESULT_CHUNK = 2000


def _try_load_race_index(
    storage,
    race_id: str,
    *,
    allow_gcs: bool,
) -> dict | None:
    """race_index をローカル（L2 / mirror）優先で読む。allow_gcs=False なら GCS に行かない。"""
    if storage._is_locally_cached("race_index"):
        data = storage._read_local_cache("race_index", race_id)
        if data:
            return data
    if storage.local_mirror_exists("race_index", race_id):
        try:
            return json.loads(
                storage._local_mirror_path("race_index", race_id).read_text(encoding="utf-8")
            )
        except Exception:
            pass
    if allow_gcs:
        return storage.load("race_index", race_id)
    return None


def _enrich_races_with_speed_index(
    results: list[dict],
    storage,
    *,
    force_refresh: bool = False,
    race_index_gcs: bool = True,
    enqueue_missing: bool = True,
) -> dict[str, int]:
    """2024-01-01 以降のレースでタイム指数を race_index から補完（ローカル優先）。"""
    scraping_status = {
        "required_races": 0,
        "completed_races": 0,
        "pending_races": 0,
    }

    for race in results:
        if race.get("time_index") and race.get("time_index") > 0:
            scraping_status["completed_races"] += 1
        elif race.get("date", "").replace("/", "-") >= "2024-01-01":
            scraping_status["required_races"] += 1

    if force_refresh or scraping_status["required_races"] <= 0:
        return scraping_status

    from concurrent.futures import ThreadPoolExecutor

    need_idx = [
        (i, race)
        for i, race in enumerate(results)
        if (not race.get("time_index") or race.get("time_index") == 0)
        and race.get("horse_number")
        and race.get("race_id")
        and race.get("date", "").replace("/", "-") >= "2024-01-01"
    ]
    if not need_idx:
        return scraping_status

    rids = list({r.get("race_id") for _, r in need_idx if r.get("race_id")})
    idx_map: dict[str, dict] = {}

    def _load_idx(rid: str):
        return rid, _try_load_race_index(storage, rid, allow_gcs=race_index_gcs)

    with ThreadPoolExecutor(max_workers=min(len(rids), 20)) as pool:
        for rid, data in pool.map(_load_idx, rids):
            if data:
                idx_map[rid] = data

    jobs: list[dict] = []
    queue = None
    if enqueue_missing:
        from src.scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()

    for i, race in need_idx:
        race_id = race.get("race_id")
        horse_number = race.get("horse_number")
        speed_data = idx_map.get(race_id or "")
        if speed_data and "entries" in speed_data:
            for entry in speed_data["entries"]:
                if entry.get("horse_number") == horse_number:
                    ti = (
                        entry.get("time_index_m")
                        or entry.get("speed_max")
                        or entry.get("speed_avg")
                    )
                    if ti and ti > 0:
                        race["time_index"] = ti
                        results[i] = race
                        scraping_status["completed_races"] += 1
                        scraping_status["required_races"] -= 1
                    break
        elif not speed_data and queue and enqueue_missing:
            jobs.append({
                "job_kind": "race",
                "target_id": race_id,
                "tasks": ["race_index"],
                "priority": 1,
                "job_label": f"タイム指数: {race_id}",
            })

    if jobs and queue:
        queue.bulk_add_jobs(jobs)
        scraping_status["pending_races"] += len(jobs)

    return scraping_status


def _load_horse_data(
    storage,
    horse_id: str,
    *,
    force_refresh: bool = False,
) -> dict | None:
    if force_refresh:
        year = horse_id[:4]
        local_path = os.path.join(
            storage._local_dir, "horse_result", year, f"{horse_id}.json"
        )
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info("古いhorse_resultを削除: %s", horse_id)

        from src.scraper.run import ScraperRunner

        runner = ScraperRunner(interval=1.0, auto_login=True)
        runner.storage = storage
        return runner.scrape_horse(horse_id, skip_existing=False, with_history=True)

    if storage.local_mirror_exists("horse_result", horse_id):
        try:
            return json.loads(
                storage._local_mirror_path("horse_result", horse_id).read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            pass

    year = horse_id[:4]
    legacy = Path(storage._local_dir) / "horse_result" / year / f"{horse_id}.json"
    if legacy.is_file():
        try:
            return json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:
            pass

    return storage.load("horse_result", horse_id)


def filter_horse_ids_with_horse_result(storage, horse_ids: list[str]) -> list[str]:
    """horse_result がローカルまたは GCS に存在する馬 ID のみ残す（順序維持）。"""
    if not horse_ids:
        return []

    found: set[str] = set()
    for hid in horse_ids:
        if storage.local_mirror_exists("horse_result", hid):
            found.add(hid)

    remaining = [h for h in horse_ids if h not in found]
    for hid in remaining:
        year = hid[:4]
        legacy = Path(storage._local_dir) / "horse_result" / year / f"{hid}.json"
        if legacy.is_file():
            found.add(hid)

    remaining = [h for h in horse_ids if h not in found]
    if remaining and storage.gcs_enabled:
        for i in range(0, len(remaining), _HORSE_RESULT_CHUNK):
            chunk = remaining[i : i + _HORSE_RESULT_CHUNK]
            exists = storage.batch_check_keys("horse_result", chunk)
            found.update(exists.keys())

    return [h for h in horse_ids if h in found]


def build_growth_curve_response(
    horse_id: str,
    storage,
    *,
    fetch_speed_index: bool = True,
    force_refresh: bool = False,
    race_index_gcs: bool = True,
    enqueue_missing: bool = True,
) -> dict[str, Any]:
    """成長曲線 JSON を組み立てる（全出走・limit なし）。"""
    horse_data = _load_horse_data(storage, horse_id, force_refresh=force_refresh)
    if not horse_data:
        return {
            "horse_id": horse_id,
            "error": f"馬ID {horse_id} のデータが見つかりません",
            "status": "no_horse_result",
            "races": [],
        }

    horse_name = horse_data.get("horse_name", "不明")
    results_all = horse_data.get("race_history", horse_data.get("results", []))
    total_race_count = len(results_all)
    results = sorted(results_all, key=lambda r: r.get("date", ""), reverse=True)

    scraping_status: dict[str, int] = {}
    if fetch_speed_index:
        scraping_status = _enrich_races_with_speed_index(
            results,
            storage,
            force_refresh=force_refresh,
            race_index_gcs=race_index_gcs,
            enqueue_missing=enqueue_missing,
        )

    if not results:
        return {
            "horse_id": horse_id,
            "error": "出走履歴が見つかりません",
            "status": "no_races",
            "races": [],
        }

    results_sorted = sorted(results, key=lambda r: r.get("date", ""), reverse=False)
    races: list[dict] = []
    prev_date = None
    weights: list[float] = []
    ranks: list[int] = []

    for race in results_sorted:
        date_str = race.get("date", "")
        weight = race.get("weight")
        weight_diff = race.get("weight_change", race.get("weight_diff"))

        rank = race.get("finish_position", race.get("rank"))
        if rank and isinstance(rank, str):
            try:
                rank = int(rank) if rank.isdigit() else None
            except ValueError:
                rank = None
        elif rank == 0 or rank == -1:
            rank = None

        interval_days = None
        if prev_date and date_str:
            try:
                curr_date = datetime.strptime(date_str.replace("/", "-"), "%Y-%m-%d")
                prev_date_obj = datetime.strptime(prev_date.replace("/", "-"), "%Y-%m-%d")
                interval_days = (curr_date - prev_date_obj).days
            except ValueError:
                pass

        time_index = race.get("time_index") or race.get("speed_index")
        rid = str(race.get("race_id") or "").strip()
        races.append({
            "race_id": rid,
            "date": date_str,
            "venue": race.get("venue", ""),
            "race_name": race.get("race_name", ""),
            "surface": race.get("surface", ""),
            "distance": race.get("distance"),
            "track_condition": race.get("track_condition", ""),
            "rank": rank,
            "field_size": race.get("field_size"),
            "weight": weight,
            "weight_diff": weight_diff,
            "weight_change": weight_diff,
            "interval_days": interval_days,
            "time": race.get("finish_time", race.get("time", "")),
            "time_index": time_index,
        })

        if weight:
            weights.append(weight)
        if rank:
            ranks.append(rank)
        prev_date = date_str

    debut_weight = None
    debut_date = None
    for r in sorted(results_all, key=lambda x: x.get("date", "")):
        w = r.get("weight")
        if w:
            debut_weight = w
            debut_date = r.get("date")
            break

    races.reverse()

    response: dict[str, Any] = {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "total_races": len(races),
        "avg_weight": sum(weights) / len(weights) if weights else 0,
        "weight_range": [min(weights), max(weights)] if weights else [0, 0],
        "best_rank": min(ranks) if ranks else None,
        "avg_rank": sum(ranks) / len(ranks) if ranks else None,
        "total_all_races": total_race_count,
        "debut_weight": debut_weight,
        "debut_date": debut_date,
        "races": races,
    }

    if scraping_status.get("required_races", 0) > 0 or scraping_status.get(
        "pending_races", 0
    ) > 0:
        response["scraping_status"] = scraping_status

    return response


def get_growth_curve(
    storage,
    horse_id: str,
    *,
    fetch_speed_index: bool = False,
    force_refresh: bool = False,
    limit: int | None = None,
    jra_only: bool = True,
    allow_compute_on_miss: bool | None = None,
    race_index_gcs: bool | None = None,
    enqueue_missing: bool | None = None,
) -> dict[str, Any]:
    """
    成長曲線を返す。ローカルにあればそれを使用し、計算時は随時ローカルへ保存する。
    """
    if allow_compute_on_miss is None:
        allow_compute_on_miss = (
            os.environ.get("KEIBA_GROWTH_CURVE_ALLOW_COMPUTE_ON_MISS", "1").strip().lower()
            in ("1", "true", "yes")
        ) or force_refresh

    if race_index_gcs is None:
        race_index_gcs = fetch_speed_index and (
            os.environ.get("KEIBA_GROWTH_CURVE_RACE_INDEX_GCS", "0").strip().lower()
            in ("1", "true", "yes")
        )

    if enqueue_missing is None:
        enqueue_missing = race_index_gcs

    if not force_refresh:
        cached = load_local(horse_id)
        if cached is not None:
            scoped = filter_growth_curve_for_jra(cached, jra_only=jra_only)
            return apply_limit(scoped, limit)

    if not allow_compute_on_miss:
        from src.config.data_paths import GROWTH_CURVE_DIR

        return {
            "horse_id": horse_id,
            "error": (
                "成長曲線の事前計算データがありません。"
                f" 金曜バッチまたは refresh で {GROWTH_CURVE_DIR} に生成してください。"
            ),
            "status": "not_precomputed",
            "races": [],
        }

    payload = build_growth_curve_response(
        horse_id,
        storage,
        fetch_speed_index=fetch_speed_index,
        force_refresh=force_refresh,
        race_index_gcs=race_index_gcs,
        enqueue_missing=enqueue_missing,
    )

    if payload.get("races"):
        save_local(
            horse_id,
            payload,
            source="refresh" if force_refresh else "api",
        )

    scoped = filter_growth_curve_for_jra(payload, jra_only=jra_only)
    return apply_limit(scoped, limit)


def iter_index_horse_ids(base_dir: str | os.PathLike) -> list[str]:
    from src.utils.horse_name_index import load_horse_name_index

    data = load_horse_name_index(base_dir)
    horses = data.get("horses") or []
    if isinstance(horses, dict):
        return sorted(horses.keys())
    return sorted(
        str(h.get("horse_id", "")).strip()
        for h in horses
        if isinstance(h, dict) and h.get("horse_id")
    )


def precompute_growth_curves(
    storage,
    horse_ids: list[str],
    *,
    skip_existing: bool = False,
    force_refresh: bool = False,
    max_age_days: float = 7.0,
    fetch_speed_index: bool = True,
    race_index_gcs: bool = False,
    enqueue_missing: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    """対象馬の成長曲線を一括計算してローカルに保存。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not horse_ids:
        return {"status": "skipped", "reason": "no_horse_ids", "ok": 0, "skip": 0, "fail": 0}

    workers = min(8, max(1, workers))
    ok = skip = fail = 0
    t0 = time.perf_counter()
    gcs_before = getattr(storage, "_gcs_call_count", 0)

    def _one(hid: str) -> str:
        if skip_existing and not force_refresh and is_local_fresh(
            hid, max_age_days=max_age_days
        ):
            return "skip"
        try:
            payload = build_growth_curve_response(
                hid,
                storage,
                fetch_speed_index=fetch_speed_index,
                force_refresh=force_refresh,
                race_index_gcs=race_index_gcs,
                enqueue_missing=enqueue_missing,
            )
            if payload.get("races"):
                save_local(hid, payload, source="batch_weekly")
                return "ok"
            return "fail"
        except Exception as exc:
            logger.warning("成長曲線バッチ失敗 %s: %s", hid, exc)
            return "fail"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, hid): hid for hid in horse_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            st = fut.result()
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
            if i % 500 == 0 or i == len(horse_ids):
                logger.info(
                    "成長曲線バッチ [%d/%d] ok=%d skip=%d fail=%d",
                    i,
                    len(horse_ids),
                    ok,
                    skip,
                    fail,
                )

    gcs_reads = getattr(storage, "_gcs_call_count", 0) - gcs_before
    meta = update_index_meta(
        batch_source="precompute_growth_curve",
        total_targets=len(horse_ids),
        ok=ok,
        skip=skip,
        fail=fail,
        elapsed_sec=round(time.perf_counter() - t0, 1),
        gcs_reads_estimate=gcs_reads,
        skip_existing=skip_existing,
        max_age_days=max_age_days,
        race_index_gcs=race_index_gcs,
    )
    return {"status": "ok", **meta}


def run_weekly_growth_curve_update(base_dir: str | os.PathLike) -> dict[str, Any]:
    """金曜 18:00 用: horse_result がある馬のみ、7 日以内のキャッシュはスキップ。"""
    from src.scraper.storage import HybridStorage

    storage = HybridStorage(str(base_dir))
    index_ids = iter_index_horse_ids(base_dir)
    horse_ids = filter_horse_ids_with_horse_result(storage, index_ids)
    logger.info(
        "成長曲線週次更新: インデックス %d 頭 → horse_result あり %d 頭",
        len(index_ids),
        len(horse_ids),
    )
    return precompute_growth_curves(
        storage,
        horse_ids,
        skip_existing=True,
        force_refresh=False,
        max_age_days=7.0,
        fetch_speed_index=True,
        race_index_gcs=False,
        enqueue_missing=False,
        workers=4,
    )
