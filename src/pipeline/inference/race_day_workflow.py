"""開催日の予測ワークフロー（各レースの発走45分前に予測を起動する）。

流れ::

    発走時刻表 → (各レースの 発走-45分) → 特徴量を作る → アンサンブルで予測 → 結果を保存
                                                                         ↓
                                     VPS の /api/race/{id}/predictions が保存済みの結果を返す

起動方法は2通り（同じ ``predict_race`` を呼ぶ）:
  - ``enqueue_race_day_tasks``: 開催日の朝に全レースを Cloud Tasks の**予約配信**として一括登録する。
    時刻になると Cloud Tasks が ``POST /api/internal/cloud-tasks/process-job`` を呼ぶ（GCP側ワーカー）。
  - ``run_race_day_local``: 1つのプロセスが発走-45分まで待って順に実行する（開発・VPS実行用）。

特徴量ビルダーは ``KEIBA_FEATURE_BUILDER``（現状は疑似ビルダーのみ）、モデルは
``KEIBA_MODEL_STORE``（``gs://bucket/prefix`` またはパス）の現行版を使う。
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np

from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

logger = logging.getLogger("pipeline.inference.race_day_workflow")

_JST = ZoneInfo("Asia/Tokyo")
PREDICT_LEAD_MINUTES = 45
JOB_KIND = "predict_race"
DEFAULT_MODEL_CACHE_DIR = "models/ensemble_cache"

_PREDICTOR_CACHE: dict[str, Any] = {}


# ── モデルの取得 ───────────────────────────────────────


def load_predictor(*, force: bool = False):
    """``KEIBA_MODEL_STORE`` の現行版（``latest.json``）を取得して ``EnsemblePredictor`` を返す。

    同じ版は再ダウンロード・再読み込みしない（プロセス内キャッシュ）。``KEIBA_MODEL_STORE`` が
    未設定なら、学習結果をそのまま置いた ``models/ensemble`` を読む。
    """
    from src.pipeline.models.ensemble_predictor import EnsemblePredictor
    from src.pipeline.models.model_registry import fetch_latest, open_store

    url = (os.environ.get("KEIBA_MODEL_STORE") or "").strip()
    if not url:
        return EnsemblePredictor.load("models/ensemble")

    cache_dir = os.environ.get("KEIBA_MODEL_CACHE_DIR") or DEFAULT_MODEL_CACHE_DIR
    strict = (os.environ.get("KEIBA_MODEL_STRICT_VERSIONS", "1").strip().lower() not in ("0", "false", "no"))
    model_dir = fetch_latest(open_store(url), cache_dir, strict_versions=strict)
    if model_dir is None:
        raise FileNotFoundError(f"モデルストア {url!r} に公開済みの版（latest.json）がありません")

    key = str(model_dir)
    if force or key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE.clear()  # 旧版のメモリを手放す
        _PREDICTOR_CACHE[key] = EnsemblePredictor.load(model_dir)
    return _PREDICTOR_CACHE[key]


# ── 1レースの予測 ──────────────────────────────────────


def _format_predictions(meta_df, scores: np.ndarray) -> list[dict]:
    meta = meta_df[["horse_number", "horse_name", "horse_id"]].copy()
    meta["pred_score"] = scores
    meta = meta.sort_values("pred_score", ascending=False).reset_index(drop=True)
    meta["pred_rank"] = range(1, len(meta) + 1)

    raw = meta["pred_score"].values.astype(float)
    centered = raw - raw.mean()
    exp_scores = np.exp(centered / max(centered.std(), 1e-6))
    meta["softmax_prob"] = exp_scores / exp_scores.sum()

    # 画面と API が読む確率。保存前に 1 度だけ計算する（読み出しのたびに再計算しない）
    win = meta["softmax_prob"].values.astype(float)
    place = harville_top2_prob(win)
    show = harville_top3_prob(win)

    return [
        {
            "pred_rank": int(r["pred_rank"]),
            "horse_number": int(r["horse_number"]),
            "horse_name": r["horse_name"],
            "horse_id": r["horse_id"],
            "pred_score": round(float(r["pred_score"]), 4),
            "normalized_score": round(float(r["softmax_prob"]), 4),
            "win_prob": round(float(win[i]), 4),
            "place_prob": round(float(place[i]), 4),
            "show_prob": round(float(show[i]), 4),
        }
        for i, (_, r) in enumerate(meta.iterrows())
    ]


def predict_race(race_id: str, storage, *, builder=None, predictor=None, source: str = "t45", persist: bool = True) -> dict:
    """1レースの予測を作って保存する。出馬表が無い等は ``status="error"`` で返す（保存しない）。

    ``persist=False`` なら保存せずに結果だけ返す（メモリ・時間の計測用）。
    """
    from src.pipeline.features.pseudo_builder import get_feature_builder
    from src.pipeline.inference.race_prediction_service import (
        build_race_data_from_storage,
        save_cached,
    )

    t0 = time.perf_counter()
    race_data = build_race_data_from_storage(race_id, storage)
    card = race_data.get("race_card") or {}
    if not card.get("entries"):
        return {"race_id": race_id, "status": "error", "error": "出馬表データがありません", "predictions": []}

    builder = builder or get_feature_builder()
    predictor = predictor or load_predictor()

    features = builder.build(race_data)
    if features.empty:
        return {"race_id": race_id, "status": "error", "error": "特徴量テーブルが空", "predictions": []}
    scores = predictor.predict(features)

    elapsed = round(time.perf_counter() - t0, 2)
    payload = {
        "race_id": race_id,
        "race_name": card.get("race_name", ""),
        "venue": card.get("venue", ""),
        "round": card.get("round", 0),
        "surface": card.get("surface", ""),
        "distance": card.get("distance", 0),
        "track_condition": card.get("track_condition", ""),
        "status": "success",
        "model_type": "ensemble_stacking",
        "model_version": predictor.version,
        "feature_builder": getattr(builder, "name", type(builder).__name__),
        "feature_count": len(predictor.feature_names),
        "total_horses": len(features),
        "elapsed_sec": elapsed,
        "predictions": _format_predictions(features, scores),
        "_compute_meta": {"source": source, "elapsed_sec": elapsed},
    }
    if not persist:
        payload["persisted"] = False
        return payload
    payload["persisted"] = bool(save_cached(storage, race_id, payload, source=source))
    if not payload["persisted"]:
        logger.error("予測は計算できたが保存できなかった race_id=%s（GCS未接続/バックオフ中の可能性）", race_id)
    return payload


# ── 発走時刻表 ─────────────────────────────────────────


def load_day_schedule(storage, date_fmt: str) -> list[dict]:
    """``race_day_schedule`` を読む。無ければ ``race_lists`` と ``race_shutuba`` から合成する。"""
    from src.scraper.race_day_schedule import (
        schedule_payload_to_runtime_list,
        synthesize_race_day_schedule_payload,
    )

    payload = storage.load("race_day_schedule", date_fmt)
    if not payload or not payload.get("slots"):
        payload = synthesize_race_day_schedule_payload(storage, date_fmt)
    return schedule_payload_to_runtime_list(payload)


def plan_race_day(storage, date_fmt: str, *, lead_minutes: int = PREDICT_LEAD_MINUTES) -> list[dict]:
    """各レースの起動時刻（発走の ``lead_minutes`` 分前）を計算して返す。"""
    plan = []
    for row in load_day_schedule(storage, date_fmt):
        post = row["post_time"]
        plan.append({**row, "run_at": post - timedelta(minutes=lead_minutes)})
    plan.sort(key=lambda r: r["run_at"])
    return plan


# ── 起動方法1: Cloud Tasks に予約配信として一括登録 ───────────


def enqueue_race_day_tasks(
    storage,
    date_fmt: str,
    *,
    lead_minutes: int = PREDICT_LEAD_MINUTES,
    enqueue_fn: Callable[..., str] | None = None,
    now_fn: Callable[[], datetime] | None = None,
    catch_up_delay_sec: int = 10,
) -> list[dict]:
    """全レースの予測を Cloud Tasks に予約配信として登録する（開催日の朝に1回実行）。

    - 配信時刻は 発走-``lead_minutes`` 分。タスク名を ``predict-<race_id>`` に固定し、
      二重登録しても重複実行されない（``AlreadyExists`` は ``status="exists"`` として扱う）。
    - 登録時点で既に起動時刻を過ぎているが発走前のレースは、``catch_up_delay_sec`` 秒後に実行する。
    - 発走済みのレースは登録しない（``status="skipped_past"``）。
    """
    if enqueue_fn is None:
        from src.scraper.cloud_tasks_queue import enqueue_via_cloud_tasks

        enqueue_fn = enqueue_via_cloud_tasks
    now = (now_fn or (lambda: datetime.now(_JST)))()

    results: list[dict] = []
    for row in plan_race_day(storage, date_fmt, lead_minutes=lead_minutes):
        rid = row["race_id"]
        if row["post_time"] <= now:
            results.append({"race_id": rid, "status": "skipped_past"})
            continue
        run_at = row["run_at"] if row["run_at"] > now else now + timedelta(seconds=catch_up_delay_sec)
        payload = {"job_kind": JOB_KIND, "race_id": rid, "date_fmt": date_fmt}
        try:
            name = enqueue_fn(payload, schedule_time=run_at, task_id=f"predict-{rid}")
            results.append({"race_id": rid, "status": "enqueued", "run_at": run_at.isoformat(), "task": name})
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "AlreadyExists":
                results.append({"race_id": rid, "status": "exists", "run_at": run_at.isoformat()})
            else:
                logger.error("予測タスクの登録に失敗 race_id=%s: %s", rid, e)
                results.append({"race_id": rid, "status": "error", "error": str(e)})
    return results


def handle_predict_job(payload: dict, storage=None) -> dict:
    """Cloud Tasks のワーカー（``/api/internal/cloud-tasks/process-job``）から呼ばれる。"""
    rid = str(payload.get("race_id") or "").strip()
    if not rid:
        return {"status": "error", "error": "race_id が必要です"}
    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage(".")
    res = predict_race(rid, storage, source="t45_cloud_tasks")
    if res.get("status") == "success" and not res.get("persisted"):
        # 保存できていないと画面に出ないため失敗として返し、Cloud Tasks に再試行させる
        return {**res, "status": "error", "error": "予測結果を保存できませんでした（GCS未接続等）"}
    return res


# ── 起動方法2: 1プロセスが発走-45分まで待って順に実行 ─────────


def run_race_day_local(
    storage,
    date_fmt: str,
    *,
    lead_minutes: int = PREDICT_LEAD_MINUTES,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    builder=None,
    predictor=None,
    skip_past: bool = True,
) -> list[dict]:
    """各レースの起動時刻まで待って ``predict_race`` を実行する。発走済みのレースは飛ばす。"""
    now_fn = now_fn or (lambda: datetime.now(_JST))
    plan = plan_race_day(storage, date_fmt, lead_minutes=lead_minutes)
    if not plan:  # 非開催日: モデルを読み込まずに終了する（cron を毎日回しても無害）
        logger.info("%s は開催レースなし。何もせず終了", date_fmt)
        return []
    predictor = predictor or load_predictor()

    results: list[dict] = []
    for row in plan:
        rid = row["race_id"]
        now = now_fn()
        if skip_past and row["post_time"] <= now:
            results.append({"race_id": rid, "status": "skipped_past"})
            continue
        wait = (row["run_at"] - now).total_seconds()
        if wait > 0:
            logger.info("%s まで待機 %.0f 秒 (race_id=%s)", row["run_at"].strftime("%H:%M"), wait, rid)
            sleep_fn(wait)
        try:
            res = predict_race(rid, storage, builder=builder, predictor=predictor, source="t45_local")
        except Exception as e:  # noqa: BLE001 - 1レースの失敗で後続を止めない
            logger.error("予測失敗 race_id=%s: %s", rid, e, exc_info=True)
            res = {"race_id": rid, "status": "error", "error": str(e)}
        status = res.get("status")
        if status == "success" and not res.get("persisted"):
            status = "not_persisted"
        results.append({"race_id": rid, "status": status, "error": res.get("error"), "elapsed_sec": res.get("elapsed_sec")})
    return results
