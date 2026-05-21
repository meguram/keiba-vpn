"""
追走難度のレイテンシ・通信ベンチ（キャッシュ / フル計算 / MLflow serve）。

例:
  python -m src.scripts.maintenance.bench_tracking_inference 202603010301
  python -m src.scripts.maintenance.bench_tracking_inference 202603010301 --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

from src.pipeline.inference.mlflow_serve_client import get_tracking_serve_client
from src.pipeline.inference.tracking_difficulty_service import (
    build_tracking_difficulty_response,
    get_or_compute,
    load_cached_response,
)
from src.scraper.storage import HybridStorage
from src.utils.mlflow_client import init_mlflow


def _bench_local(storage, race_id: str) -> dict:
    rows: dict[str, float | str | bool | None] = {}

    t0 = time.perf_counter()
    cached = load_cached_response(storage, race_id)
    rows["cache_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    rows["cache_hit"] = cached is not None

    t1 = time.perf_counter()
    full = build_tracking_difficulty_response(race_id, storage, allow_scrape=False)
    rows["full_compute_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    meta = full.get("_compute_meta") or {}
    rows["lgbm_backend"] = meta.get("lgbm_backend")
    rows["n_horses"] = len(full.get("entries") or [])

    t2 = time.perf_counter()
    via_get = get_or_compute(storage, race_id, force_refresh=False, allow_scrape=False)
    rows["get_or_compute_ms"] = round((time.perf_counter() - t2) * 1000, 1)
    rows["get_or_compute_from_cache"] = bool(via_get.get("_from_cache"))

    return rows


def _bench_mlflow() -> dict:
    out: dict = {}
    t0 = time.perf_counter()
    uri = init_mlflow()
    out["tracking_uri"] = uri
    if str(uri).startswith("http"):
        try:
            r = requests.get(f"{uri.rstrip('/')}/health", timeout=5)
            out["tracking_health_ok"] = r.status_code == 200
        except Exception as e:
            out["tracking_health_ok"] = False
            out["tracking_error"] = str(e)
    else:
        out["tracking_health_ok"] = True
        out["tracking_error"] = "file_store"
    out["tracking_health_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    client = get_tracking_serve_client()
    if not client:
        out["serve_configured"] = False
        return out

    out["serve_configured"] = True
    out["serve_uri"] = client.base_uri
    t1 = time.perf_counter()
    out["serve_health_ok"] = client.is_available(force_check=True)
    out["serve_health_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    if out["serve_health_ok"]:
        from src.pipeline.models.tracking_difficulty import FEATURE_COLUMNS

        row = {c: 0.0 for c in FEATURE_COLUMNS}
        matrix = [[row.get(c, 0) for c in FEATURE_COLUMNS]]
        t2 = time.perf_counter()
        try:
            preds = client.predict_dataframe(FEATURE_COLUMNS, matrix)
            out["serve_invocation_ok"] = len(preds) == 1
            out["serve_invocation_ms"] = round((time.perf_counter() - t2) * 1000, 1)
        except Exception as e:
            out["serve_invocation_ok"] = False
            out["serve_invocation_error"] = str(e)
            out["serve_invocation_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    return out


def _bench_api(base: str, race_id: str) -> dict:
    out: dict = {}
    base = base.rstrip("/")

    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base}/api/inference/health", timeout=15)
        out["inference_health_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["inference_health_status"] = r.status_code
        if r.ok:
            out["inference_health"] = r.json()
    except Exception as e:
        out["inference_health_error"] = str(e)
        out["inference_health_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    try:
        r = requests.get(
            f"{base}/api/race/{race_id}/tracking-difficulty",
            timeout=120,
        )
        out["api_tracking_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        out["api_tracking_status"] = r.status_code
        if r.ok:
            body = r.json()
            out["api_field_size"] = body.get("field_size")
    except Exception as e:
        out["api_tracking_error"] = str(e)
        out["api_tracking_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="追走難度レイテンシベンチ")
    parser.add_argument("race_id", help="ベンチ対象レース ID")
    parser.add_argument(
        "--api",
        default=os.environ.get("KEIBA_API_BASE", ""),
        help="FastAPI ベース URL（例 http://127.0.0.1:8000）",
    )
    args = parser.parse_args(argv)

    storage = HybridStorage()
    report = {
        "race_id": args.race_id,
        "cache_env": os.environ.get("KEIBA_TRACKING_DIFFICULTY_CACHE", "1"),
        "serve_env": os.environ.get("KEIBA_MLFLOW_SERVE_TRACKING_URI", ""),
        "local": _bench_local(storage, args.race_id),
        "mlflow": _bench_mlflow(),
    }
    if args.api:
        report["api"] = _bench_api(args.api, args.race_id)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
