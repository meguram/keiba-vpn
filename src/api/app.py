"""
FastAPI アプリケーション

予測結果の表示・エージェントステータス・追加要件API・MLflow モデル予測APIを提供する。
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import sys
import math
import threading
import unicodedata
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _scrub_nan(obj: Any) -> Any:
    """float NaN / Inf を None に変換する（JSON シリアライズ前の正規化用）。"""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _scrub_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_nan(v) for v in obj]
    return obj

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from contextlib import asynccontextmanager
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.api.auth import (
    is_developer,
    requires_auth,
    is_public_path,
    verify_password,
    create_session_response,
    clear_session_response,
    COOKIE_NAME,
)
from src.utils.keiba_logging import standard_log_formatter


# ── 毎朝の構造チェックスケジューラ ──────────────────────
_scheduler_thread: threading.Thread | None = None
_scheduler_stop = threading.Event()
_scheduler_state: dict[str, Any] = {
    "running": False,
    "last_run": None,
    "last_result": None,
    "next_run": None,
    "run_count": 0,
}
_scheduler_lock = threading.Lock()

# ── スクレイピングキューワーカー ──────────────────────
# 各スロットのスレッドを保持（インデックス = slot_id）
_queue_worker_threads: list[threading.Thread] = []
_queue_worker_stop = threading.Event()
# 新規ジョブ追加時に set() → 待機中スロットを即起床させる
_queue_new_job_event = threading.Event()
# マルチワーカー時は flock で1プロセスだけがキューループを回す（JSON 競合防止）
_queue_runner_leader_fh = None
_queue_is_leader = False

# ワーカー健康状態（ファイルベースで全プロセスから参照可能）
_WORKER_HEALTH_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "queue", ".worker_health.json",
)
_CRON_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "queue", ".cron_state.json",
)

# ── data/cache 定期クリーンアップ（HybridStorage.cleanup_disk_cache） ──
_disk_cache_cleanup_thread: threading.Thread | None = None
_disk_cache_cleanup_stop = threading.Event()
_disk_cache_cleanup_state: dict[str, Any] = {
    "running": False, "last_run": None, "last_result": None,
    "next_run": None, "run_count": 0,
}
_disk_cache_cleanup_lock = threading.Lock()

# ── キュー定期メンテ（1時間ごと: 失敗→待機、完了レコード削除）リーダーのみ ──
_queue_maintain_thread: threading.Thread | None = None
_queue_maintain_stop = threading.Event()
_queue_maintain_state: dict[str, Any] = {
    "running": False, "last_run": None, "last_result": None,
    "next_run": None, "run_count": 0,
}
_queue_maintain_lock = threading.Lock()

# ── logs/ 保持: 直近1週間超の *.log を毎日 12:00 JST に削除（リーダー1プロセス・fcntl） ──
_logs_retention_thread: threading.Thread | None = None
_logs_retention_stop = threading.Event()
_log_retention_leader_fh = None
_logs_retention_state: dict[str, Any] = {
    "running": False, "last_run": None, "last_result": None,
    "next_run": None, "run_count": 0,
}
_logs_retention_lock = threading.Lock()

# ── 未来レース出馬表 毎日自動取得（race_lists → race_shutuba、リーダーのみ） ──
_daily_shutuba_thread: threading.Thread | None = None
_daily_shutuba_stop = threading.Event()
_daily_shutuba_state: dict[str, Any] = {
    "running": False, "last_run": None, "last_result": None,
    "next_run": None, "run_count": 0,
}
_daily_shutuba_lock = threading.Lock()


_AUTO_RESUME_DELAY = 120  # アクセス一時停止後の自動復帰待機（秒）


def _write_worker_health(*, heartbeat: float = 0, error: str = "", error_ts: float = 0, crash_count: int = 0):
    """ワーカースレッドの状態をファイルに書き込む（リーダーのみ呼ぶ）。"""
    try:
        data = {}
        try:
            with open(_WORKER_HEALTH_FILE, "r") as f:
                data = json.loads(f.read())
        except (OSError, json.JSONDecodeError):
            pass
        if heartbeat:
            data["heartbeat"] = heartbeat
            data["pid"] = os.getpid()
            # スロット稼働状況を記録
            live = sum(1 for t in _queue_worker_threads if t is not None and t.is_alive())
            data["slot_count"] = len(_queue_worker_threads)
            data["live_slots"] = live
        if error:
            data["last_error"] = error
            data["last_error_ts"] = error_ts
        if crash_count:
            data["crash_count"] = crash_count
        with open(_WORKER_HEALTH_FILE, "w") as f:
            f.write(json.dumps(data))
    except OSError:
        pass


def get_worker_health() -> dict:
    """ワーカースレッドの健康状態を返す（全プロセスから呼べる）。"""
    from src.scraper.job_queue import LOCK_FILE
    try:
        with open(_WORKER_HEALTH_FILE, "r") as f:
            data = json.loads(f.read())
    except (OSError, json.JSONDecodeError):
        data = {}
    hb = float(data.get("heartbeat", 0) or 0)
    pid = int(data.get("pid", 0) or 0)
    age = _time.time() - hb if hb > 0 else -1

    alive = False
    if not data:
        # 未記録は「停止」にしない（リーダーが先に heartbeat を書き込む）
        alive = True
    else:
        pid_alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                pid_alive = True
            except (ProcessLookupError, PermissionError, OSError):
                pass

        if pid_alive:
            if age >= 0 and age < 120:
                alive = True
            elif LOCK_FILE.exists():
                alive = True
        elif pid <= 0 and hb <= 0 and not (data.get("last_error") or data.get("crash_count")):
            alive = True

    processing = LOCK_FILE.exists() and alive

    return {
        "alive": alive,
        "processing": processing,
        "heartbeat": hb,
        "heartbeat_age_sec": round(age, 1) if age >= 0 else None,
        "last_error": data.get("last_error"),
        "last_error_ts": data.get("last_error_ts"),
        "crash_count": data.get("crash_count", 0),
        "slot_count": data.get("slot_count", 0),
        "live_slots": data.get("live_slots", 0),
    }


def _write_cron_state(job_id: str, partial: dict) -> None:
    """クロンジョブの状態をファイルに書き込む（マルチワーカー対応）。"""
    try:
        try:
            with open(_CRON_STATE_FILE) as f:
                data = json.loads(f.read())
        except (OSError, json.JSONDecodeError):
            data = {}
        if job_id not in data:
            data[job_id] = {}
        data[job_id].update({k: v for k, v in partial.items() if v is not None or k in ("last_run", "last_result")})
        with open(_CRON_STATE_FILE, "w") as f:
            f.write(json.dumps(data))
    except OSError:
        pass


def _read_cron_state(job_id: str) -> dict:
    """ファイルからクロンジョブの状態を読む（マルチワーカー対応）。"""
    try:
        with open(_CRON_STATE_FILE) as f:
            data = json.loads(f.read())
        return data.get(job_id, {})
    except (OSError, json.JSONDecodeError):
        return {}


def _ensure_worker_slots() -> None:
    """停止しているスロットを検出して再起動する。リーダーでなければリーダー権を奪取してから実行。"""
    global _queue_is_leader, _queue_worker_threads
    if _queue_worker_stop.is_set():
        return
    if not _queue_is_leader:
        health = get_worker_health()
        if health.get("alive"):
            return
        if not _try_acquire_queue_runner_leader():
            return
        _queue_is_leader = True
        logger.warning("前リーダーが停止したため、このプロセスが新リーダーになります")

    from src.scraper.job_queue import _queue_parallel_workers
    n_target = _queue_parallel_workers()

    # スロットリストが足りなければ拡張
    while len(_queue_worker_threads) < n_target:
        _queue_worker_threads.append(None)  # type: ignore[arg-type]

    for slot_id in range(n_target):
        t = _queue_worker_threads[slot_id]
        if t is not None and t.is_alive():
            continue
        health = get_worker_health()
        cc = health.get("crash_count", 0) + 1
        logger.warning("スロット[%d] 停止 — 再起動 (クラッシュ累計: %d)", slot_id, cc)
        _write_worker_health(crash_count=cc)
        new_t = threading.Thread(
            target=_queue_slot_worker, args=(slot_id,),
            daemon=True, name=f"queue-worker-{slot_id}",
        )
        new_t.start()
        _queue_worker_threads[slot_id] = new_t


def _queue_slot_worker(slot_id: int) -> None:
    """
    キュースロットワーカー。SCRAPE_QUEUE_PARALLEL 本並列起動する。
    各スロットが独立して claim_next_pending_job → _process_claimed_job を繰り返す。
    スロット 0 がストール回復・ストレージプレチェックを担当する。
    """
    from src.scraper.job_queue import ScrapeJobQueue
    from src.scraper.scrape_access_pause import read_access_pause, clear_access_pause
    from src.scraper.queue_worker_log import ensure_queue_worker_log_handler, mark_queue_worker_active

    log = logging.getLogger("queue.worker")
    log.setLevel(logging.INFO)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(standard_log_formatter())
        log.addHandler(_h)
    ensure_queue_worker_log_handler()

    queue = ScrapeJobQueue()

    if slot_id == 0:
        log.info("キュースロット[0] 起動 — ストール回復・プレチェック実行")
        queue.requeue_stale_running_jobs(assume_lock_holder=False)
        queue._run_storage_precheck()
    else:
        # スロット間の起動タイミングをずらして同時アクセスを回避
        _queue_worker_stop.wait(timeout=slot_id * 0.3)

    mark_queue_worker_active(True)
    _write_worker_health(heartbeat=_time.time())
    log.info("キュースロット[%d] 待機開始", slot_id)

    # アクセスエラー時にこのスロットの現在ジョブを中断するためのイベント
    slot_stop = threading.Event()

    try:
        while not _queue_worker_stop.is_set():
            _write_worker_health(heartbeat=_time.time())
            try:
                # ── アクセス一時停止チェック（ユーザーが「再開」を押すまで待機）──
                pause = read_access_pause()
                if pause.get("active"):
                    # 自動復帰しない。UI の「再開」ボタン押下まで 30 秒間隔でポーリング。
                    if _queue_worker_stop.wait(timeout=30):
                        break
                    continue

                # スロット 0: precheck ジョブを pending/completed へ移行
                if slot_id == 0:
                    _pc = queue.count_precheck_jobs()
                    if _pc > 0:
                        log.info("プレチェック実行: %d 件のストレージ確認中…", _pc)
                    _pr = queue._run_storage_precheck()
                    if _pc > 0 and isinstance(_pr, dict):
                        _skip = _pr.get("to_completed", 0)
                        _enq  = _pr.get("to_pending", 0)
                        log.info(
                            "プレチェック完了: スキップ %d 件 / キュー投入 %d 件 (残 precheck: %d 件)",
                            _skip, _enq, queue.count_precheck_jobs(),
                        )

                # ── ジョブ取得・実行 ──
                slot_stop.clear()
                job = queue.claim_next_pending_job()
                if job:
                    log.info(
                        "[slot-%d] ジョブ開始: %s",
                        slot_id,
                        job.get("job_label") or job.get("target_id") or "?",
                    )
                    queue._process_claimed_job(job, slot_stop, start_delay_sec=0)
                    invalidate_status_cache()
                else:
                    # キューが空 — 新ジョブ通知か最大5秒後に再試行
                    _queue_new_job_event.wait(timeout=5)
                    _queue_new_job_event.clear()

            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                _write_worker_health(error=msg, error_ts=_time.time())
                log.error("[slot-%d] エラー: %s", slot_id, e, exc_info=True)
                if _queue_worker_stop.wait(timeout=10):
                    break
    finally:
        mark_queue_worker_active(False)
        log.info("[slot-%d] 終了", slot_id)


STRUCTURE_CHECK_HOUR_JST = 6
STRUCTURE_CHECK_MINUTE_JST = 0


def _scheduler_loop():
    """毎朝 JST 指定時刻に構造チェックを実行するデーモンスレッド。"""
    from datetime import timezone as _tz, timedelta as _td, datetime as _dt
    _JST = _tz(_td(hours=9))

    _sched_log = logging.getLogger("scheduler.structure")
    _sched_log.setLevel(logging.INFO)
    if not _sched_log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(standard_log_formatter())
        _sched_log.addHandler(_h)
    _sched_log.info("構造チェックスケジューラ起動 (毎朝 %02d:%02d JST)",
                    STRUCTURE_CHECK_HOUR_JST, STRUCTURE_CHECK_MINUTE_JST)

    while not _scheduler_stop.is_set():
        now = _dt.now(_JST)
        target_today = now.replace(
            hour=STRUCTURE_CHECK_HOUR_JST,
            minute=STRUCTURE_CHECK_MINUTE_JST,
            second=0, microsecond=0)

        if now >= target_today:
            target = target_today + _td(days=1)
        else:
            target = target_today

        wait_seconds = (target - now).total_seconds()
        with _scheduler_lock:
            _scheduler_state["next_run"] = target.isoformat()

        _sched_log.info("次回構造チェック: %s (%.0f秒後)", target.strftime("%Y-%m-%d %H:%M JST"), wait_seconds)

        if _scheduler_stop.wait(timeout=wait_seconds):
            break

        _sched_log.info("=== 定期構造チェック開始 ===")
        with _scheduler_lock:
            _scheduler_state["running"] = True

        try:
            from src.scraper.structure_monitor import run_daily_check
            result = run_daily_check(auto_reparse=True, notify=True)
            with _scheduler_lock:
                _scheduler_state["last_run"] = _dt.now(_JST).isoformat()
                _scheduler_state["last_result"] = result
                _scheduler_state["run_count"] += 1
            severity = result.get("severity", "OK")
            critical = result.get("critical", 0)
            _sched_log.info("定期構造チェック完了: severity=%s, critical=%d", severity, critical)
        except Exception as e:
            _sched_log.error("定期構造チェック失敗: %s", e)
            with _scheduler_lock:
                _scheduler_state["last_run"] = _dt.now(_JST).isoformat()
                _scheduler_state["last_result"] = {"status": "error", "error": str(e)}
        finally:
            with _scheduler_lock:
                _scheduler_state["running"] = False


def _try_acquire_queue_runner_leader() -> bool:
    """
    Uvicorn 複数ワーカーそれぞれがキューループを回すと同一 JSON を壊すため、
    非ブロッキング flock で先着1プロセスだけが True を返す。
    """
    global _queue_runner_leader_fh
    try:
        import fcntl
    except ImportError:
        return True
    from src.scraper.job_queue import QUEUE_DIR

    path = QUEUE_DIR / ".queue_runner_leader"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _queue_runner_leader_fh = open(path, "a+")
        fcntl.flock(_queue_runner_leader_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        try:
            if _queue_runner_leader_fh:
                _queue_runner_leader_fh.close()
        except OSError:
            pass
        _queue_runner_leader_fh = None
        return False


@asynccontextmanager
async def lifespan(app):
    global _scheduler_thread, _queue_worker_threads, _queue_runner_leader_fh, _disk_cache_cleanup_thread, _queue_maintain_thread, _logs_retention_thread, _log_retention_leader_fh, _daily_shutuba_thread

    # 構造チェックスケジューラ起動
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="structure-scheduler")
    _scheduler_thread.start()

    # data/cache 定期クリーンアップ（週次アクセス L2 と整合）
    _disk_cache_cleanup_stop.clear()
    _disk_cache_cleanup_thread = threading.Thread(
        target=_disk_cache_cleanup_loop, daemon=True, name="disk-cache-cleanup"
    )
    _disk_cache_cleanup_thread.start()
    logger.info("data/cache クリーンアップスレッド起動")

    # ワーカーログリングハンドラを起動時に登録（process_queue 前でも有効にするため）
    from src.scraper.queue_worker_log import ensure_queue_worker_log_handler
    ensure_queue_worker_log_handler()

    # デプロイずれ検知: ワーカーが読み込んだ result_table セレクタ数（起動時1回）
    try:
        import logging as _logging
        from src.scraper.parsers import RaceResultParser

        _rt = RaceResultParser._RESULT_TABLE.selectors
        _logging.getLogger("uvicorn.error").info(
            "RaceResultParser._RESULT_TABLE: n=%d head=%s",
            len(_rt),
            _rt[:3],
        )
    except Exception as _e:
        import logging as _logging

        _logging.getLogger("uvicorn.error").warning(
            "RaceResultParser セルフチェック失敗: %s", _e
        )

    # スクレイピングキューワーカー（マルチワーカー時はリーダー1プロセスのみ）
    global _queue_is_leader
    if _try_acquire_queue_runner_leader():
        _queue_is_leader = True
        _queue_worker_stop.clear()
        _queue_new_job_event.clear()
        from src.scraper.job_queue import _queue_parallel_workers
        n_slots = _queue_parallel_workers()
        for _slot_id in range(n_slots):
            _t = threading.Thread(
                target=_queue_slot_worker, args=(_slot_id,),
                daemon=True, name=f"queue-worker-{_slot_id}",
            )
            _t.start()
            _queue_worker_threads.append(_t)
        logger.info(
            "スクレイピングキュースロットワーカー %d 本起動（当プロセスがリーダー）",
            n_slots,
        )
    else:
        _queue_is_leader = False
        logger.info("スクレイピングキューワーカーは別プロセスが担当（スキップ）")

    if _queue_is_leader:
        _queue_maintain_stop.clear()
        _queue_maintain_thread = threading.Thread(
            target=_queue_hourly_maintain_loop,
            daemon=True,
            name="queue-hourly-maintain",
        )
        _queue_maintain_thread.start()
        logger.info("キュー定期メンテスレッド起動（当プロセスがリーダー）")
    else:
        _queue_maintain_thread = None

    _logs_retention_stop.clear()
    _ld = (os.environ.get("LOGS_RETENTION_DISABLED") or "").strip().lower()
    if _ld in ("1", "true", "yes", "on"):
        _logs_retention_thread = None
        logger.info("logs/ 保持クリーンアップ: 無効 (LOGS_RETENTION_DISABLED)")
    elif _try_acquire_logs_retention_leader():
        _logs_retention_thread = threading.Thread(
            target=_logs_retention_loop,
            daemon=True,
            name="logs-retention",
        )
        _logs_retention_thread.start()
        logger.info("logs/ 保持クリーンアップ起動（毎日 12:00 JST ・直近7日超の *.log 削除、当プロセス）")
    else:
        _logs_retention_thread = None
        logger.info("logs/ 保持クリーンアップは別ワーカーが担当")

    # 未来レース 出馬表 毎日自動取得（リーダーのみ）
    if _queue_is_leader:
        _daily_shutuba_stop.clear()
        _daily_shutuba_thread = threading.Thread(
            target=_daily_shutuba_enqueue_loop,
            daemon=True,
            name="daily-shutuba-enqueue",
        )
        _daily_shutuba_thread.start()
        logger.info("出馬表 毎日自動取得スレッド起動（当プロセスがリーダー）")
    else:
        _daily_shutuba_thread = None
        logger.info("出馬表 毎日自動取得は別プロセスが担当")

    try:
        from src.scraper.queue_worker_log import ensure_queue_worker_log_handler

        ensure_queue_worker_log_handler()
    except Exception as _e:
        logger.warning("キューワーカー用ログバッファの初期化をスキップ: %s", _e)

    # キュー関連ルートの存在確認（古いプロセスが動いていると UI のタスク一覧が 404 になる）
    _queue_api_required = (
        "/api/scrape-queue/tasks",
        "/api/scrape-queue/add-job",
        "/api/scrape-queue/enqueue-period-horses",
        "/api/scrape-queue/enqueue-period-races",
        "/api/scrape-queue/enqueue-scrape-period",
        "/api/scrape-queue/worker-logs",
        "/api/scrape-queue/progress",
        "/api/scrape-queue/resume",
        "/api/scrape-queue/verify-horse-coverage",
    )
    _paths = {getattr(r, "path", "") for r in app.routes if getattr(r, "path", None)}
    _missing_q = [p for p in _queue_api_required if p not in _paths]
    if _missing_q:
        logger.error(
            "scrape-queue の API ルートが未登録です（コードとプロセスが不一致の可能性）: %s — サーバを再起動してください",
            _missing_q,
        )
    else:
        logger.info(
            "scrape-queue API（tasks / add-job / enqueue-period-horses / worker-logs / resume）登録済み"
        )

    # bloodline-cluster の重いデータをバックグラウンドで先読みしておく (初回リクエスト高速化)
    def _preload_bloodline_cluster():
        try:
            from src.api.bloodline_meta_cluster import (
                _load_stats_assets,
                _load_pedigree_10gen,
                _load_horse_prize_map,
            )
            _load_stats_assets()
            _load_pedigree_10gen()
            _load_horse_prize_map()
            logger.info("bloodline-cluster 先読み完了")
        except Exception as _e:
            logger.warning("bloodline-cluster 先読み失敗: %s", _e)

    threading.Thread(target=_preload_bloodline_cluster, daemon=True, name="bloodline-preload").start()

    def _horse_name_index_bootstrap():
        if (os.environ.get("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            logger.info("馬名インデックス起動時生成: 無効 (HORSE_NAME_INDEX_DISABLE_BOOTSTRAP)")
            return
        try:
            from src.utils import horse_name_index as hni

            r = hni.ensure_horse_name_index(BASE_DIR)
            logger.info(
                "馬名インデックス起動時 ensure: status=%s total=%s",
                r.get("status"),
                r.get("total_horses"),
            )
        except Exception as _e:
            logger.warning("馬名インデックス起動時 ensure 失敗: %s", _e)

    threading.Thread(
        target=_horse_name_index_bootstrap, daemon=True, name="horse-name-index-ensure"
    ).start()

    yield

    # 終了処理
    _scheduler_stop.set()
    _queue_worker_stop.set()
    _disk_cache_cleanup_stop.set()
    _queue_maintain_stop.set()
    _logs_retention_stop.set()
    _daily_shutuba_stop.set()

    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)
    for _t in _queue_worker_threads:
        if _t is not None:
            _t.join(timeout=5)
    if _disk_cache_cleanup_thread:
        _disk_cache_cleanup_thread.join(timeout=5)
    if _queue_maintain_thread:
        _queue_maintain_thread.join(timeout=5)
    if _logs_retention_thread:
        _logs_retention_thread.join(timeout=5)
    if _daily_shutuba_thread:
        _daily_shutuba_thread.join(timeout=5)
    if _log_retention_leader_fh:
        try:
            _log_retention_leader_fh.close()
        except OSError:
            pass
        _log_retention_leader_fh = None
    if _queue_runner_leader_fh:
        try:
            _queue_runner_leader_fh.close()
        except OSError:
            pass
        _queue_runner_leader_fh = None

    try:
        _get_storage().flush_weekly_access()
    except Exception as _e:
        logger.warning("weekly access flush 失敗: %s", _e)


app = FastAPI(title="ML-AutoPilot Keiba", version="3.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if requires_auth(path) and not is_developer(request):
            if path.startswith("/api/"):
                return JSONResponse(
                    {"error": "認証が必要です", "login_url": "/login"},
                    status_code=401,
                )
            from fastapi.responses import RedirectResponse
            return RedirectResponse(
                url=f"/login?next={path}", status_code=302
            )
        return await call_next(request)


app.add_middleware(AuthMiddleware)

JRA_PLACE_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}


def _is_jra_race(race_id: str) -> bool:
    """race_id が中央競馬 (JRA 10会場) のレースかどうか判定する。"""
    return len(race_id) >= 6 and race_id[4:6] in JRA_PLACE_CODES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/data/image", StaticFiles(directory=os.path.join(BASE_DIR, "data", "local", "image")), name="data_image")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
templates.env.globals["is_dev_check"] = is_developer

PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "predictions.json")


def _load_predictions() -> dict | None:
    if os.path.exists(PREDICTIONS_PATH):
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════
# ヘルスチェック
# ═══════════════════════════════════════════════════════

_SERVER_START_TIME = _time.time()


@app.get("/api/health", response_class=JSONResponse)
async def health_check():
    """サーバーの稼働状態を返す。監視cron用。"""
    uptime = _time.time() - _SERVER_START_TIME
    return JSONResponse({
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "timestamp": datetime.now().isoformat(),
    })


@app.post("/api/html-archive/cleanup", response_class=JSONResponse)
async def html_archive_cleanup(dry_run: bool = False, keep: int = 10):
    """HTML アーカイブをクリーンアップし、カテゴリごとに keep 件のみ残す。"""
    archive = _get_html_archive()
    if not archive.gcs_enabled:
        return JSONResponse({"error": "GCS 未接続"}, status_code=503)

    def _run():
        return archive.cleanup(keep_per_category=keep, dry_run=dry_run)

    result = await asyncio.to_thread(_run)
    return JSONResponse(result)


# ═══════════════════════════════════════════════════════
# 既存エンドポイント
# ═══════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/"):
    if is_developer(request):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=next, status_code=302)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "current_page": "login",
        "breadcrumbs": [],
        "next_url": next,
        "error": "",
    })


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    password = form.get("password", "")
    next_url = form.get("next", "/")

    if verify_password(password):
        return create_session_response(redirect_to=next_url, request=request)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "current_page": "login",
        "breadcrumbs": [],
        "next_url": next_url,
        "error": "パスワードが正しくありません",
    })


@app.get("/logout")
async def logout():
    return clear_session_response(redirect_to="/", request=request)


@app.get("/api/auth/status", response_class=JSONResponse)
async def auth_status(request: Request):
    dev = is_developer(request)
    return JSONResponse({"is_developer": dev})


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_page": "home",
        "is_dev": is_developer(request),
        "breadcrumbs": [],
    })


@app.get("/api/predictions", response_class=JSONResponse)
async def get_predictions():
    data = _load_predictions()
    if data is None:
        return JSONResponse({"error": "予測データが見つかりません"}, status_code=404)
    return JSONResponse(data)


@app.get("/api/gcs-stats", response_class=JSONResponse)
async def get_gcs_stats():
    """GCS API コール数とキャッシュ統計を返す (コスト監視用)。"""
    storage = _get_storage()
    with _race_detail_cache_lock:
        race_detail_cache_size = len(_race_detail_cache)
    return JSONResponse({
        "gcs_api_calls": storage._gcs_call_count,
        "gcs_healthy": getattr(storage, "_gcs_healthy", None),
        "gcs_backoff_remaining_s": max(0, round(
            storage._gcs_backoff - (_time.time() - storage._gcs_last_failure), 1
        )) if storage._gcs_last_failure > 0 else 0,
        "load_cache_size": len(storage._load_cache),
        "load_cache_max": getattr(storage, "_mem_cache_max", None),
        "load_cache_ttl_s": getattr(storage, "_mem_cache_ttl", None),
        "blob_list_cache_size": len(getattr(storage, "_blob_list_cache", {}) or {}),
        "blob_list_cache_ttl_current_s": getattr(storage, "_BLOB_LIST_TTL_CURRENT", None),
        "blob_list_cache_ttl_past_s": getattr(storage, "_BLOB_LIST_TTL_PAST", None),
        "status_cache_entries": len(_status_cache),
        "status_cache_ttl_s": _STATUS_CACHE_TTL,
        "horse_ids_cache_size": len(_horse_ids_cache),
        "race_detail_cache_size": race_detail_cache_size,
    })



# ═══════════════════════════════════════════════════════
# スクレイピング モニタリングボード (GCS プライマリ)
# ═══════════════════════════════════════════════════════


_storage_lock = threading.Lock()


def _get_storage():
    """HybridStorage のシングルトンインスタンスを返す。"""
    from src.scraper.storage import HybridStorage
    if not hasattr(_get_storage, "_inst"):
        with _storage_lock:
            if not hasattr(_get_storage, "_inst"):
                _get_storage._inst = HybridStorage(BASE_DIR)
    return _get_storage._inst


def _disk_cache_cleanup_loop():
    """
    起動直後に 1 回、その後 DISK_CACHE_CLEANUP_INTERVAL_SEC ごとに
    data/cache の古い *.json を削除する。
    """
    import os as _os

    _log = logging.getLogger("scheduler.disk_cache")
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(
            standard_log_formatter()
        )
        _log.addHandler(_h)
    from datetime import datetime as _dt2, timezone as _tz2, timedelta as _td2
    _JST2 = _tz2(_td2(hours=9))
    interval = float(_os.environ.get("DISK_CACHE_CLEANUP_INTERVAL_SEC", "86400"))
    _log.info(
        "data/cache クリーンアップ: 初回即時、その後 %.0fs ごと (max_age=%s)",
        interval,
        _os.environ.get("DISK_CACHE_CLEANUP_MAX_AGE_SEC", "604800"),
    )
    while not _disk_cache_cleanup_stop.is_set():
        with _disk_cache_cleanup_lock:
            _disk_cache_cleanup_state["running"] = True
            _disk_cache_cleanup_state["next_run"] = None
        try:
            st = _get_storage()
            r = st.cleanup_disk_cache()
            _log.info("disk cache cleanup: %s", r)
            sn = st.cleanup_snapshot_files(max_age_seconds=86400)
            if sn:
                _log.info("snapshot cleanup: removed %d files", sn)
            with _disk_cache_cleanup_lock:
                _disk_cache_cleanup_state["last_run"] = _dt2.now(_JST2).isoformat()
                _disk_cache_cleanup_state["last_result"] = {**r, "snapshot_removed": sn}
                _disk_cache_cleanup_state["run_count"] += 1
        except Exception as e:
            _log.warning("disk cache cleanup 失敗: %s", e)
            with _disk_cache_cleanup_lock:
                _disk_cache_cleanup_state["last_run"] = _dt2.now(_JST2).isoformat()
                _disk_cache_cleanup_state["last_result"] = {"error": str(e)}
        finally:
            with _disk_cache_cleanup_lock:
                _disk_cache_cleanup_state["running"] = False
                next_ts = (_dt2.now(_JST2).timestamp() + interval)
                _disk_cache_cleanup_state["next_run"] = _dt2.fromtimestamp(next_ts, tz=_JST2).isoformat()
        if _disk_cache_cleanup_stop.wait(timeout=interval):
            break
    _log.info("data/cache クリーンアップスレッド終了")


def _queue_hourly_maintain_loop():
    """
    一定間隔で失敗ジョブを待機に戻し、完了レコードをキューから取り除く。
    終了目安は /api/scrape-queue/status の queue_eta が都度再計算される。
    """
    import os as _os

    _m_log = logging.getLogger("scheduler.queue_maintain")
    _m_log.setLevel(logging.INFO)
    if not _m_log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(
            standard_log_formatter()
        )
        _m_log.addHandler(_h)

    try:
        interval = float(_os.environ.get("SCRAPE_QUEUE_HOURLY_MAINTENANCE_SEC", "3600"))
    except (TypeError, ValueError):
        interval = 3600.0
    if interval <= 0:
        _m_log.info("キュー定期メンテ: 無効 (SCRAPE_QUEUE_HOURLY_MAINTENANCE_SEC<=0)")
        return

    _m_log.info("キュー定期メンテ: 初回以降 %.0f 秒ごと (失敗→待機 + 完了レコード削除)", interval)

    from src.scraper.job_queue import run_hourly_queue_maintenance
    from datetime import datetime as _dt3, timezone as _tz3, timedelta as _td3
    _JST3 = _tz3(_td3(hours=9))

    while not _queue_maintain_stop.is_set():
        next_ts = _dt3.now(_JST3).timestamp() + interval
        with _queue_maintain_lock:
            _queue_maintain_state["next_run"] = _dt3.fromtimestamp(next_ts, tz=_JST3).isoformat()
        if _queue_maintain_stop.wait(timeout=interval):
            break
        if _queue_maintain_stop.is_set():
            break
        with _queue_maintain_lock:
            _queue_maintain_state["running"] = True
        try:
            r = run_hourly_queue_maintenance()
            if r.get("ok"):
                _m_log.info(
                    "キュー定期メンテ完了: 失敗→待機 %d 件, 完了レコード削除 %d 件",
                    int(r.get("requeued") or 0),
                    int(r.get("completed_removed") or 0),
                )
            else:
                _m_log.warning("キュー定期メンテ: %s", r.get("error") or r)
            with _queue_maintain_lock:
                _queue_maintain_state["last_run"] = _dt3.now(_JST3).isoformat()
                _queue_maintain_state["last_result"] = r
                _queue_maintain_state["run_count"] += 1
        except Exception as e:
            _m_log.error("キュー定期メンテ失敗: %s", e, exc_info=True)
            with _queue_maintain_lock:
                _queue_maintain_state["last_run"] = _dt3.now(_JST3).isoformat()
                _queue_maintain_state["last_result"] = {"error": str(e)}
        finally:
            with _queue_maintain_lock:
                _queue_maintain_state["running"] = False

    _m_log.info("キュー定期メンテスレッド終了")


def _try_acquire_logs_retention_leader() -> bool:
    """``logs/`` 保持期限切れ削除を1プロセスだけが実行（uvicorn マルチワーカー対策）。"""
    global _log_retention_leader_fh
    try:
        import fcntl
    except ImportError:
        return True
    from pathlib import Path

    p = Path(BASE_DIR) / "logs" / ".log_retention_leader"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        _log_retention_leader_fh = open(p, "a+", encoding="utf-8", errors="replace")
        fcntl.flock(_log_retention_leader_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        try:
            if _log_retention_leader_fh:
                _log_retention_leader_fh.close()
        except OSError:
            pass
        _log_retention_leader_fh = None
        return False


def _logs_retention_loop():
    """
    毎日 JST 12:00 に ``logs/*.log`` のうち最終更新が保持日数（既定7日）を超えたものを削除。
    環境変数: LOGS_RETENTION_DAYS, LOGS_RETENTION_HOUR_JST, LOGS_RETENTION_MINUTE_JST, LOGS_RETENTION_DISABLED
    """
    import os as _os
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td

    _JST = _tz(_td(hours=9))
    _log = logging.getLogger("scheduler.logs_retention")
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(standard_log_formatter())
        _log.addHandler(_h)
    try:
        h = int(_os.environ.get("LOGS_RETENTION_HOUR_JST", "12"))
        m = int(_os.environ.get("LOGS_RETENTION_MINUTE_JST", "0"))
    except ValueError:
        h, m = 12, 0
    h = max(0, min(23, h))
    m = max(0, min(59, m))
    _log.info("logs/ 保持クリーンアップ起動 (毎日 %02d:%02d JST, LOGS_RETENTION_DAYS 超を削除)", h, m)

    while not _logs_retention_stop.is_set():
        now = _dt.now(_JST)
        target_today = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now >= target_today:
            target = target_today + _td(days=1)
        else:
            target = target_today
        wait_seconds = (target - now).total_seconds()
        with _logs_retention_lock:
            _logs_retention_state["next_run"] = target.isoformat()
        _log.info(
            "次回 logs 保持期限切れ削除: %s (%.0f秒後)",
            target.strftime("%Y-%m-%d %H:%M JST"),
            wait_seconds,
        )
        if _logs_retention_stop.wait(timeout=wait_seconds):
            break
        if _logs_retention_stop.is_set():
            break
        with _logs_retention_lock:
            _logs_retention_state["running"] = True
        try:
            from src.utils.log_retention import run_log_retention_once

            r = run_log_retention_once(BASE_DIR)
            _log.info(
                "logs 保持クリーンアップ: removed=%s bytes_freed=%s max_age_days=%s",
                r.get("removed"),
                r.get("bytes_freed"),
                r.get("max_age_days"),
            )
            if r.get("errors"):
                _log.warning("logs 保持クリーンアップ: 一部失敗 %s", r.get("errors"))
            with _logs_retention_lock:
                _logs_retention_state["last_run"] = _dt.now(_JST).isoformat()
                _logs_retention_state["last_result"] = r
                _logs_retention_state["run_count"] += 1
        except Exception as e:
            _log.warning("logs 保持クリーンアップ失敗: %s", e, exc_info=True)
            with _logs_retention_lock:
                _logs_retention_state["last_run"] = _dt.now(_JST).isoformat()
                _logs_retention_state["last_result"] = {"error": str(e)}
        finally:
            with _logs_retention_lock:
                _logs_retention_state["running"] = False
    _log.info("logs/ 保持クリーンアップスレッド終了")


def _daily_shutuba_enqueue_loop() -> None:
    """
    毎日指定時刻 (JST) に、今日から DAYS_AHEAD 日先までの race_lists を走査し
    未取得の race_shutuba をキューへ投入する。smart_skip=True なので取得済みはスキップ。

    環境変数:
        DAILY_SHUTUBA_HOUR_JST   (デフォルト 7)
        DAILY_SHUTUBA_MINUTE_JST (デフォルト 0)
        DAILY_SHUTUBA_DAYS_AHEAD (デフォルト 14)
    """
    import os as _os
    from datetime import date as _date, datetime as _dt, timedelta as _td, timezone as _tz

    _JST = _tz(_td(hours=9))
    _log = logging.getLogger("scheduler.daily_shutuba")
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(standard_log_formatter())
        _log.addHandler(_h)

    try:
        h = int(_os.environ.get("DAILY_SHUTUBA_HOUR_JST", "7"))
        m = int(_os.environ.get("DAILY_SHUTUBA_MINUTE_JST", "0"))
        days_ahead = int(_os.environ.get("DAILY_SHUTUBA_DAYS_AHEAD", "14"))
    except ValueError:
        h, m, days_ahead = 7, 0, 14
    h = max(0, min(23, h))
    m = max(0, min(59, m))
    days_ahead = max(1, min(60, days_ahead))

    _log.info("出馬表 毎日自動取得スレッド起動 (%02d:%02d JST, 今日〜+%d日)", h, m, days_ahead)

    while not _daily_shutuba_stop.is_set():
        now = _dt.now(_JST)
        target_today = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now >= target_today:
            target = target_today + _td(days=1)
        else:
            target = target_today
        wait_seconds = (target - now).total_seconds()
        with _daily_shutuba_lock:
            _daily_shutuba_state["next_run"] = target.isoformat()
        _log.info(
            "次回 出馬表 自動投入: %s (%.0f秒後)",
            target.strftime("%Y-%m-%d %H:%M JST"),
            wait_seconds,
        )
        if _daily_shutuba_stop.wait(timeout=wait_seconds):
            break
        if _daily_shutuba_stop.is_set():
            break
        with _daily_shutuba_lock:
            _daily_shutuba_state["running"] = True
        try:
            today = _date.today()
            end = today + _td(days=days_ahead)
            body = {
                "start_date": today.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "tasks": ["race_shutuba"],
                "smart_skip": True,
                "dry_run": False,
                "limit": 500,
                "jra_only": True,
                "priority": 10,
            }
            result = _sync_enqueue_period_race_tasks(body)
            added = result.get("created", 0)
            skipped = result.get("skipped", 0)
            total = result.get("candidate_races", 0)
            _log.info(
                "出馬表 自動投入完了: added=%d skipped=%d total=%d (%s〜%s)",
                added, skipped, total,
                body["start_date"], body["end_date"],
            )
            if added:
                _kick_scrape_queue_worker()
            with _daily_shutuba_lock:
                _daily_shutuba_state["last_run"] = _dt.now(_JST).isoformat()
                _daily_shutuba_state["last_result"] = result
                _daily_shutuba_state["run_count"] += 1
                _state_snap = dict(_daily_shutuba_state)
            _write_cron_state("daily_shutuba", {
                "last_run": _state_snap["last_run"],
                "last_result": _state_snap["last_result"],
                "run_count": _state_snap["run_count"],
            })
        except Exception as e:
            _log.warning("出馬表 自動投入失敗: %s", e, exc_info=True)
            with _daily_shutuba_lock:
                _daily_shutuba_state["last_run"] = _dt.now(_JST).isoformat()
                _daily_shutuba_state["last_result"] = {"error": str(e)}
            _write_cron_state("daily_shutuba", {
                "last_run": _daily_shutuba_state["last_run"],
                "last_result": _daily_shutuba_state["last_result"],
            })
        finally:
            with _daily_shutuba_lock:
                _daily_shutuba_state["running"] = False
    _log.info("出馬表 毎日自動取得スレッド終了")


def _kick_scrape_queue_worker() -> None:
    """新規ジョブが追加されたことをスロットワーカーに通知して即起床させる。"""
    _queue_new_job_event.set()


def _kick_urgent_worker() -> None:
    """最優先ジョブ専用のファストレーンワーカーを起動。通常ワーカーと並走可能。"""
    try:
        from src.scraper.job_queue import kick_urgent_process_queue_background

        kick_urgent_process_queue_background()
    except Exception as e:
        logger.warning("urgent ワーカー起動スキップ: %s", e)


_html_archive_lock = threading.Lock()


def _get_html_archive():
    """HtmlArchive のシングルトンインスタンスを返す。"""
    from src.scraper.html_archive import HtmlArchive
    if not hasattr(_get_html_archive, "_inst"):
        with _html_archive_lock:
            if not hasattr(_get_html_archive, "_inst"):
                _get_html_archive._inst = HtmlArchive()
    return _get_html_archive._inst


@app.get("/monitor", response_class=HTMLResponse)
async def scrape_monitor_page(request: Request):
    """スクレイピング状態のリアルタイムモニタリングボード"""
    return templates.TemplateResponse("admin/monitor.html", {
        "request": request,
        "current_page": "monitor",
        "breadcrumbs": [],
    })


MONITOR_SOURCES = [
    "race_shutuba",
    "race_result_on_time",
    "race_result",
    "race_index",
    "race_odds",
    "race_pair_odds",
    "race_paddock",
    "race_barometer",
    "race_trainer_comment",
    "smartrc_race",
]

_STEEPLECHASE_NA_CATS = frozenset({
    "race_index", "race_barometer", "race_pair_odds", "smartrc_race",
})

# 障害レースかどうかを race_name から判定
import re as _re
_STEEPLECHASE_PATTERN = _re.compile(r"障害|ジャンプ|ハードル|スティープル")


# ── レスポンスキャッシュ (stale-while-revalidate) ──────────────────

_status_cache: dict[str, tuple[float, dict]] = {}
_status_cache_lock = threading.Lock()
_STATUS_CACHE_TTL = 30        # fresh: 30 seconds (active scraping でもリアルタイム反映)
_STATUS_CACHE_STALE = 300     # stale-while-revalidate: 5 minutes
_status_bg_refreshing: set[str] = set()
_status_bg_refreshing_lock = threading.Lock()

_horse_ids_cache: dict[str, list[str]] = {}
_horse_ids_cache_lock = threading.Lock()


def invalidate_status_cache(date: str = ""):
    """指定日のステータスキャッシュを無効化。空文字なら全日付。"""
    with _status_cache_lock:
        if date:
            _status_cache.pop(date, None)
        else:
            _status_cache.clear()


def _get_cached_status(date: str) -> tuple[dict | None, bool]:
    """Return (cached_data, needs_refresh). Stale data is returned immediately."""
    with _status_cache_lock:
        entry = _status_cache.get(date)
        if not entry:
            return None, True
        age = _time.time() - entry[0]
        if age < _STATUS_CACHE_TTL:
            return entry[1], False
        if age < _STATUS_CACHE_STALE:
            return entry[1], True
        return None, True


def _set_cached_status(date: str, data: dict):
    with _status_cache_lock:
        _status_cache[date] = (_time.time(), data)


def _bg_refresh_status(date: str):
    """Background thread to refresh stale cache."""
    with _status_bg_refreshing_lock:
        if date in _status_bg_refreshing:
            return
        _status_bg_refreshing.add(date)
    try:
        result = _build_scrape_status(date)
        _set_cached_status(date, result)
    except Exception:
        pass
    finally:
        with _status_bg_refreshing_lock:
            _status_bg_refreshing.discard(date)


def _build_scrape_status(date: str) -> dict:
    """
    バッチ GCS リスト方式でスクレイピング進捗を構築する。

    旧方式: 36レース × 10ソース × 3-5 GCS call = 1,600+ 回
    新方式: カテゴリごとに list_blobs 1回 = ~15 回
    """
    from datetime import timezone as _tz, timedelta as _td
    _JST = _tz(_td(hours=9))

    storage = _get_storage()

    race_list_data = storage.load("race_lists", date)
    if not race_list_data:
        return {
            "date": date,
            "total_races": 0,
            "race_list_exists": False,
            "gcs_enabled": storage.gcs_enabled,
            "progress": {"completed": 0, "total": 0, "percentage": 0,
                         "fresh": 0, "fresh_percentage": 0},
            "sources": MONITOR_SOURCES + ["horse_result", "horse_pedigree_5gen"],
            "races": [],
        }

    raw_races = race_list_data.get("races", [])
    race_ids = [r["race_id"] for r in raw_races
                if r.get("race_id") and _is_jra_race(r["race_id"])]
    race_id_set = set(race_ids)

    if not race_ids:
        # 一覧ファイルはあるが JRA レースが 0 件（開催なし・未取得の空一覧など）
        struct_versions = storage._load_structure_versions()
        return {
            "date": date,
            "total_races": 0,
            "race_list_exists": True,
            "gcs_enabled": storage.gcs_enabled,
            "all_fresh": False,
            "structure_versions": struct_versions,
            "progress": {
                "completed": 0,
                "total": 0,
                "percentage": 0,
                "fresh": 0,
                "fresh_percentage": 0,
            },
            "sources": MONITOR_SOURCES + ["horse_result", "horse_pedigree_5gen"],
            "races": [],
        }

    year = date[:4]
    struct_versions = storage._load_structure_versions()

    # GCS 疎通確認: バックオフ中でなければ素早くチェック
    if storage.gcs_enabled and storage._gcs_healthy:
        storage.check_gcs_connectivity(quick=True)

    # カテゴリごとに list_blobs 1回 → {key: updated_ts} を取得
    # 並列実行で更に高速化
    cat_blobs: dict[str, dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=len(MONITOR_SOURCES) + 1) as pool:
        futures = {
            cat: pool.submit(storage.batch_list_blobs, cat, year)
            for cat in MONITOR_SOURCES
        }
        for cat, f in futures.items():
            cat_blobs[cat] = f.result()

    # race_result からグレード情報を取得
    result_blobs = cat_blobs.get("race_result", {})
    race_grades: dict[str, str] = {}
    rids_need_grade = [rid for rid in race_ids if rid in result_blobs]
    if rids_need_grade:
        with ThreadPoolExecutor(max_workers=16) as pool:
            grade_futures = {
                rid: pool.submit(storage.load, "race_result", rid)
                for rid in rids_need_grade
            }
            for rid, f in grade_futures.items():
                res = f.result()
                if res:
                    race_grades[rid] = res.get("grade", "")

    # race_shutuba から horse_ids を取得 (長期キャッシュ付き)
    shutuba_blobs = cat_blobs.get("race_shutuba", {})
    horse_ids_per_race: dict[str, list[str]] = {}
    card_entry_counts: dict[str, int] = {}
    all_horse_ids: set[str] = set()

    uncached_rids = []
    with _horse_ids_cache_lock:
        for rid in race_ids:
            if rid in _horse_ids_cache:
                hids = _horse_ids_cache[rid]
                horse_ids_per_race[rid] = hids
                card_entry_counts[rid] = len(hids)
                all_horse_ids.update(hids)
            elif rid in shutuba_blobs:
                uncached_rids.append(rid)

    if uncached_rids:
        with ThreadPoolExecutor(max_workers=16) as pool:
            shutuba_futures = {
                rid: pool.submit(storage.load, "race_shutuba", rid)
                for rid in uncached_rids
            }
            with _horse_ids_cache_lock:
                for rid, f in shutuba_futures.items():
                    card = f.result()
                    if card:
                        entries = card.get("entries", [])
                        hids = [e["horse_id"] for e in entries
                                if e.get("horse_id")]
                        _horse_ids_cache[rid] = hids
                        horse_ids_per_race[rid] = hids
                        card_entry_counts[rid] = len(entries)
                        all_horse_ids.update(hids)

    # horse_result / horse_pedigree_5gen のバッチ存在チェック
    horse_blobs: dict[str, float] = {}
    ped_blobs: dict[str, float] = {}
    if all_horse_ids:
        horse_blobs = storage.batch_check_keys(
            "horse_result", list(all_horse_ids))
        ped_blobs = storage.batch_check_keys(
            "horse_pedigree_5gen", list(all_horse_ids))

    # 構造鮮度判定のための changed_at_unix を事前計算
    # is_fresh() と同一ロジック: changed_at_unix を直接使用
    # version==1 (構造変更なし) のカテゴリは常に fresh
    cat_changed_at: dict[str, float] = {}
    _version1_cats: set[str] = set()
    for cat in MONITOR_SOURCES + ["horse_result", "horse_pedigree_5gen"]:
        struct_cat = storage.DATA_TO_STRUCTURE_MAP.get(cat)
        if struct_cat:
            sv = struct_versions.get(struct_cat, {})
            if sv.get("version", 1) <= 1:
                _version1_cats.add(cat)
            else:
                changed_unix = sv.get("changed_at_unix", 0)
                if changed_unix:
                    cat_changed_at[cat] = float(changed_unix)

    horse_struct_cat = storage.DATA_TO_STRUCTURE_MAP.get("horse_result")
    horse_ver = None
    horse_ver_changed = ""
    if horse_struct_cat:
        sv = struct_versions.get(horse_struct_cat, {})
        horse_ver = sv.get("version")
        horse_ver_changed = sv.get("changed_at", "")

    now = _time.time()
    races_info: list[dict] = []
    for race in raw_races:
        rid = race.get("race_id", "")
        if rid not in race_id_set:
            continue

        race_name = race.get("race_name", "")
        is_steeplechase = bool(_STEEPLECHASE_PATTERN.search(race_name))

        sources: dict[str, Any] = {}
        for cat in MONITOR_SOURCES:
            na = is_steeplechase and cat in _STEEPLECHASE_NA_CATS

            blob_ts = cat_blobs.get(cat, {}).get(rid, 0.0)
            exists = blob_ts > 0
            fresh = False
            scraped_jst = ""
            age_hours = -1.0

            if na and not exists:
                sv = struct_versions.get(
                    storage.DATA_TO_STRUCTURE_MAP.get(cat, ""), {})
                sources[cat] = {
                    "gcs": False, "html": False, "ok": True,
                    "fresh": True, "na": True,
                    "scraped_at_jst": "", "age_hours": -1,
                    "structure_version": sv.get("version") if sv else None,
                    "structure_changed_at": sv.get("changed_at", "") if sv else "",
                }
                continue

            if exists:
                age_hours = round((now - blob_ts) / 3600, 1)
                scraped_jst = datetime.fromtimestamp(blob_ts, tz=_JST).strftime("%Y-%m-%d %H:%M")

                struct_cat = storage.DATA_TO_STRUCTURE_MAP.get(cat)
                if struct_cat is None:
                    fresh = True
                elif cat in _version1_cats:
                    fresh = True
                elif cat not in cat_changed_at:
                    fresh = True
                else:
                    fresh = blob_ts >= cat_changed_at[cat]

            sv = struct_versions.get(
                storage.DATA_TO_STRUCTURE_MAP.get(cat, ""), {})

            sources[cat] = {
                "gcs": exists,
                "html": False,
                "ok": exists,
                "fresh": fresh,
                "na": na,
                "scraped_at_jst": scraped_jst,
                "age_hours": age_hours,
                "structure_version": sv.get("version") if sv else None,
                "structure_changed_at": sv.get("changed_at", "") if sv else "",
            }

        hids = horse_ids_per_race.get(rid, [])
        expected = card_entry_counts.get(rid, len(hids))
        horses_on_gcs = sum(1 for h in hids if h in horse_blobs)
        horse_v1 = "horse_result" in _version1_cats
        horse_changed = cat_changed_at.get("horse_result", 0)
        horses_fresh = sum(
            1 for h in hids
            if h in horse_blobs and (
                horse_v1 or horse_changed == 0 or horse_blobs[h] >= horse_changed
            )
        )
        sources["horse_result"] = {
            "total": len(hids),
            "scraped": horses_on_gcs,
            "ok": len(hids) > 0 and horses_on_gcs == len(hids),
            "expected": expected,
            "fresh": len(hids) > 0 and horses_fresh == len(hids),
            "fresh_count": horses_fresh,
            "na": False,
            "structure_version": horse_ver,
            "structure_changed_at": horse_ver_changed,
        }

        ped_on_gcs = sum(1 for h in hids if h in ped_blobs)
        ped_v1 = "horse_pedigree_5gen" in _version1_cats
        ped_changed = cat_changed_at.get("horse_pedigree_5gen", 0)
        ped_fresh_count = sum(
            1 for h in hids
            if h in ped_blobs and (
                ped_v1 or ped_changed == 0 or ped_blobs[h] >= ped_changed
            )
        )
        sources["horse_pedigree_5gen"] = {
            "total": len(hids),
            "scraped": ped_on_gcs,
            "ok": len(hids) > 0 and ped_on_gcs == len(hids),
            "expected": expected,
            "fresh": len(hids) > 0 and ped_fresh_count == len(hids),
            "fresh_count": ped_fresh_count,
            "na": False,
        }

        all_fresh = all(
            s.get("fresh", False) or s.get("na", False)
            for c in MONITOR_SOURCES
            if (s := sources.get(c))
        ) and sources["horse_result"].get("fresh", False)

        grade = race_grades.get(rid, "")
        if not grade:
            shutuba_data = storage.load("race_shutuba", rid) if rid in shutuba_blobs else None
            if shutuba_data:
                grade = shutuba_data.get("grade", "")

        races_info.append({
            "race_id": rid,
            "round": race.get("round", 0),
            "venue": race.get("venue", ""),
            "race_name": race_name,
            "grade": grade,
            "is_steeplechase": is_steeplechase,
            "sources": sources,
            "all_fresh": all_fresh,
        })

    total_items = 0
    completed_items = 0
    fresh_items = 0
    for r in races_info:
        s = r["sources"]
        for cat in MONITOR_SOURCES:
            info = s.get(cat)
            if not isinstance(info, dict):
                total_items += 1
                continue
            if info.get("na"):
                continue
            total_items += 1
            if info.get("ok"):
                completed_items += 1
            if info.get("fresh"):
                fresh_items += 1
        for hcat in ("horse_result", "horse_pedigree_5gen"):
            hr = s.get(hcat, {})
            total_items += 1
            if isinstance(hr, dict) and hr.get("ok"):
                completed_items += 1
            if isinstance(hr, dict) and hr.get("fresh"):
                fresh_items += 1

    all_dates_fresh = all(
        r.get("all_fresh", False) for r in races_info
    ) if races_info else False

    return {
        "date": date,
        "total_races": len(races_info),
        "race_list_exists": True,
        "gcs_enabled": storage.gcs_enabled,
        "all_fresh": all_dates_fresh,
        "structure_versions": struct_versions,
        "progress": {
            "completed": completed_items,
            "total": total_items,
            "percentage": round(completed_items / total_items * 100, 1)
            if total_items > 0 else 0,
            "fresh": fresh_items,
            "fresh_percentage": round(fresh_items / total_items * 100, 1)
            if total_items > 0 else 0,
        },
        "sources": MONITOR_SOURCES + ["horse_result", "horse_pedigree_5gen"],
        "races": races_info,
    }


@app.get("/api/scrape-status", response_class=JSONResponse)
async def get_scrape_status(date: str = "", force: str = ""):
    """指定日付のスクレイピング進捗を返す (stale-while-revalidate)。"""
    if not date or not (date.isdigit() and len(date) == 8):
        return JSONResponse(
            {"error": "日付(YYYYMMDD)を指定してください"}, status_code=400)

    if force == "1":
        result = await asyncio.to_thread(_build_scrape_status, date)
        _set_cached_status(date, result)
        return JSONResponse(result)

    cached, needs_refresh = _get_cached_status(date)
    if cached:
        if needs_refresh:
            threading.Thread(
                target=_bg_refresh_status, args=(date,), daemon=True
            ).start()
        return JSONResponse(cached)

    result = await asyncio.to_thread(_build_scrape_status, date)
    _set_cached_status(date, result)
    return JSONResponse(result)


@app.get("/api/monitor/missing-dates-summary", response_class=JSONResponse)
async def api_monitor_missing_dates_summary(
    max_dates: int = 200,
    year: int | None = None,
):
    """
    モニター用: 各開催日の JRA レースについて、scrape_status_detail 基準の未取得件数を集計。
    キュー投入の優先度付け・カレンダー可視化に使う。
    """
    try:
        from src.scraper.monitor_backlog import summarize_missing_dates

        cap = max(1, min(int(max_dates), 500))

        def _run():
            return summarize_missing_dates(
                _get_storage(),
                max_dates=cap,
                year=year,
            )

        return JSONResponse(await asyncio.to_thread(_run))
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


_scrape_dates_cache: dict[str, Any] = {"data": None, "ts": 0.0}
_SCRAPE_DATES_RAW_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}

# 開催日セレクタ用: list_keys の結果をプロセス内にキャッシュ（GCS 全件列挙のコストを下げる）
_RACE_LIST_STEMS_CACHE: dict[str, Any] = {"stems": None, "ts": 0.0}
_RACE_LIST_STEMS_TTL = 120.0  # 2分 (daily-race-lists バッチ後の反映を早める)

# picker_past_days 応答の短時間キャッシュ
_PICKER_SCRAPE_DATES_CACHE: dict[str, Any] = {"key": None, "data": None, "ts": 0.0}
_PICKER_SCRAPE_DATES_TTL = 60.0

_NO_STORE_HEADERS = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}


def _invalidate_race_list_caches() -> None:
    """race_lists 関連の全インメモリキャッシュを即時無効化する。"""
    _RACE_LIST_STEMS_CACHE["stems"] = None
    _RACE_LIST_STEMS_CACHE["ts"] = 0.0
    _scrape_dates_cache["data"] = None
    _scrape_dates_cache["ts"] = 0.0
    _SCRAPE_DATES_RAW_CACHE["data"] = None
    _SCRAPE_DATES_RAW_CACHE["ts"] = 0.0
    _PICKER_SCRAPE_DATES_CACHE["data"] = None
    _PICKER_SCRAPE_DATES_CACHE["ts"] = 0.0
    try:
        from src.scraper.monitor_backlog import invalidate_cache as _mb_invalidate
        _mb_invalidate()
    except Exception:
        pass


def _cached_race_list_stems(storage) -> list[str]:
    """race_lists のキーを stem（通常 YYYYMMDD）の昇順で返す。2分キャッシュ。"""
    now = _time.time()
    c = _RACE_LIST_STEMS_CACHE
    if c["stems"] is not None and (now - c["ts"]) < _RACE_LIST_STEMS_TTL:
        return c["stems"]
    stems = storage.list_keys("race_lists")
    c["stems"] = stems
    c["ts"] = now
    return stems


@app.get("/api/scrape-dates", response_class=JSONResponse)
async def get_scraped_dates(
    meeting_only: str | None = Query(None),
    raw_keys: str | None = Query(None),
    picker_past_days: int | None = Query(
        None,
        ge=1,
        le=150,
        description="UI 用: 直近 N 日は data/page_reference/race_lists のファイル存在だけ確認（全件 glob しない）",
    ),
):
    """
    スクレイピング済みの日付一覧。

    既定: race_lists を読み、開催なし・JRA0・未来の平日（重賞なし）等を除いた「開催日相当」のみ。
    raw_keys=1 … ディレクトリのキー一覧のみ（高速・フィルタなし。バッチ用）。
    meeting_only=1 … 既定と同じ（互換用）。
    picker_past_days=N … 日付セレクタ向け。直近 N 日ぶんローカル race_lists の有無だけ見る（JSON 非読込・全件列挙なし）。

    フィルタ適用時は no-store。raw_keys のみ 60 秒キャッシュ。
    """
    now = _time.time()

    if picker_past_days is not None:
        ck = str(int(picker_past_days))
        pd = _PICKER_SCRAPE_DATES_CACHE
        if (
            pd.get("key") == ck
            and pd.get("data") is not None
            and (now - float(pd.get("ts") or 0)) < _PICKER_SCRAPE_DATES_TTL
        ):
            return JSONResponse(pd["data"], headers=_NO_STORE_HEADERS)

        def _picker_dates_only():
            """直近 N 日を 1 日ずつ path.exists() のみ（race_lists 全件 glob / list_keys を避ける）。"""
            from datetime import datetime, timedelta
            from pathlib import Path

            from src.config.data_paths import RACE_LISTS_DIR

            storage = _get_storage()
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            race_dir = RACE_LISTS_DIR
            n = int(picker_past_days)
            out: list[str] = []
            for i in range(1, n + 1):
                d = today - timedelta(days=i)
                stem = d.strftime("%Y%m%d")
                if (race_dir / f"{stem}.json").is_file():
                    out.append(stem)
            return {
                "dates": out,
                "gcs_enabled": storage.gcs_enabled,
                "picker_past_days": n,
                "picker_fast": True,
                "picker_source": "local_exists",
            }

        result = await asyncio.to_thread(_picker_dates_only)
        pd["key"] = ck
        pd["data"] = result
        pd["ts"] = now
        return JSONResponse(result, headers=_NO_STORE_HEADERS)

    use_raw = str(raw_keys or "").strip().lower() in ("1", "true", "yes", "on")
    disable_filter = str(meeting_only or "").strip().lower() in ("0", "false", "no", "off")
    use_fast_keys = use_raw or disable_filter

    if use_fast_keys:
        if _SCRAPE_DATES_RAW_CACHE["data"] and (now - _SCRAPE_DATES_RAW_CACHE["ts"]) < 60:
            return JSONResponse(_SCRAPE_DATES_RAW_CACHE["data"])
    else:
        if _scrape_dates_cache["data"] and (now - _scrape_dates_cache["ts"]) < 60:
            return JSONResponse(_scrape_dates_cache["data"], headers=_NO_STORE_HEADERS)

    def _load():
        storage = _get_storage()
        keys = _cached_race_list_stems(storage)
        if use_fast_keys:
            return {
                "dates": sorted(keys, reverse=True),
                "gcs_enabled": storage.gcs_enabled,
                "raw_keys": True,
            }

        from concurrent.futures import ThreadPoolExecutor

        from src.scraper.monitor_future_eligible import include_date_in_data_viewer_race_list

        def _check_one(d: str) -> tuple[str, bool]:
            rl = storage.load("race_lists", d)
            raw = (rl.get("races") or []) if rl else []
            meta = rl.get("_meta") if isinstance(rl, dict) else None
            return (d, include_date_in_data_viewer_race_list(d, raw, meta))

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(_check_one, keys))
        dates = [d for d, ok in results if ok]
        dates.sort(reverse=True)
        return {
            "dates": dates,
            "gcs_enabled": storage.gcs_enabled,
            "raceday_filtered": True,
            "filter": "data_viewer_raceday",
        }

    result = await asyncio.to_thread(_load)
    if use_fast_keys:
        _SCRAPE_DATES_RAW_CACHE["data"] = result
        _SCRAPE_DATES_RAW_CACHE["ts"] = now
        return JSONResponse(result)
    _scrape_dates_cache["data"] = result
    _scrape_dates_cache["ts"] = now
    return JSONResponse(result, headers=_NO_STORE_HEADERS)


@app.get("/api/race-list/{date}", response_class=JSONResponse)
async def get_race_list_for_date(date: str, with_result_status: bool = True):
    """指定日付のレース一覧を会場別に返す（結果データ有無オプション付き）。"""
    def _load():
        from src.utils.race_result_availability import batch_race_result_status

        storage = _get_storage()
        data = storage.load("race_lists", date)
        if not data:
            return {"date": date, "venues": [], "races": []}
        raw = data.get("races", [])
        jra = [r for r in raw if r.get("race_id") and _is_jra_race(r["race_id"])]
        venues = sorted(set(r.get("venue", "") for r in jra if r.get("venue")))
        race_ids = [r["race_id"] for r in jra]
        status_map = (
            batch_race_result_status(storage, race_ids, date=date)
            if with_result_status and race_ids
            else {}
        )
        races = []
        for r in sorted(jra, key=lambda x: (x.get("venue", ""), x.get("round", 0))):
            rid = r["race_id"]
            st = status_map.get(rid, {})
            row = {
                "race_id": rid,
                "round": r.get("round", 0),
                "venue": r.get("venue", ""),
                "race_name": r.get("race_name", ""),
            }
            if with_result_status:
                row["has_race_result"] = bool(st.get("has_confirmed"))
                row["has_result_flash"] = bool(st.get("has_flash"))
                row["result_viewable"] = bool(st.get("viewable"))
                row["result_kind"] = st.get("kind")
                row["result_view_url"] = st.get("result_view_url")
            races.append(row)
        return {"date": date, "venues": venues, "races": races}
    result = await asyncio.to_thread(_load)
    return JSONResponse(result)


_scrape_summary_cache: dict = {"data": None, "ts": 0.0, "gen": 0}
_SCRAPE_SUMMARY_FILTER_GEN = 4  # フィルタ条件変更時にインクリメントしてキャッシュ無効化


_SUMMARY_SOURCES = [
    "race_shutuba", "race_result", "race_index",
    "race_odds", "race_pair_odds", "race_paddock", "race_barometer",
    "race_trainer_comment",
]


@app.get("/api/scrape-summary-all", response_class=JSONResponse)
async def get_scrape_summary_all(force: str = ""):
    """
    全日付のスクレイピング進捗サマリーを一括返す。
    年ごとにカテゴリ blob 一覧を取得し、日付ごとの完了率を算出。
    force=1 でサーバ側キャッシュを破棄して再計算。
    """
    if force == "1":
        _scrape_summary_cache["data"] = None
        _scrape_summary_cache["ts"] = 0.0
        _scrape_summary_cache["gen"] = 0
    result = await asyncio.to_thread(_build_scrape_summary_all)
    return JSONResponse(result)


def _build_scrape_summary_all():
    cache_ttl = 300  # 5min
    now = _time.time()
    if (
        _scrape_summary_cache["data"]
        and (now - _scrape_summary_cache["ts"]) < cache_ttl
        and _scrape_summary_cache.get("gen") == _SCRAPE_SUMMARY_FILTER_GEN
    ):
        return _scrape_summary_cache["data"]

    storage = _get_storage()
    dates = sorted(storage.list_keys("race_lists"), reverse=True)

    # 1) 全日付の race_list を並列読み込み（JRA ゼロ・空ファイルも行として残す）
    def _load_one(date: str):
        race_list_data = storage.load("race_lists", date)
        if not race_list_data:
            return {
                "date": date,
                "race_list_exists": False,
                "race_list_row_count": 0,
                "total_races": 0,
                "venues": [],
                "race_ids": [],
                "_raw_races": [],
                "_meta": None,
            }
        raw_races = race_list_data.get("races") or []
        jra_races = [r for r in raw_races
                     if r.get("race_id") and _is_jra_race(r["race_id"])]
        venue_names = set()
        for r in jra_races:
            name = r.get("venue", "") or r.get("place_name", "")
            if name:
                venue_names.add(name)
        return {
            "date": date,
            "race_list_exists": True,
            "race_list_row_count": len(raw_races),
            "total_races": len(jra_races),
            "venues": list(venue_names),
            "race_ids": [r["race_id"] for r in jra_races],
            "_raw_races": raw_races,
            "_meta": race_list_data.get("_meta") if isinstance(race_list_data, dict) else None,
        }

    with ThreadPoolExecutor(max_workers=16) as pool:
        summaries_raw = list(pool.map(_load_one, dates))

    from src.scraper.monitor_future_eligible import include_date_in_monitor_summary

    summaries_raw = [
        s for s in summaries_raw
        if include_date_in_monitor_summary(
            s["date"],
            s.get("_raw_races") or [],
            s.get("_meta"),
        )
    ]
    for s in summaries_raw:
        s.pop("_raw_races", None)
        s.pop("_meta", None)

    # 2) 年ごとにカテゴリ blob 一覧を取得 (年×カテゴリ ≒ 数十回のGCSコール)
    #    事前にGCS疎通チェックを行い、失敗時はローカルキャッシュフォールバック
    storage.check_gcs_connectivity(quick=True)

    years = set()
    for s in summaries_raw:
        years.add(s["date"][:4])

    year_cat_blobs: dict[str, dict[str, set[str]]] = {}
    fetch_tasks = []
    for year in years:
        year_cat_blobs[year] = {}
        for cat in _SUMMARY_SOURCES:
            fetch_tasks.append((year, cat))

    def _fetch_blob_keys(year_cat):
        y, c = year_cat
        blob_dict = storage.batch_list_blobs(c, y)
        return (y, c, set(blob_dict.keys()))

    with ThreadPoolExecutor(max_workers=20) as pool:
        blob_results = list(pool.map(_fetch_blob_keys, fetch_tasks))

    for y, c, keys in blob_results:
        year_cat_blobs[y][c] = keys

    # 3) 日付ごとに完了率を計算
    n_cats = len(_SUMMARY_SOURCES)
    summaries = []
    for s in summaries_raw:
        rids = s["race_ids"]
        year = s["date"][:4]
        blobs_for_year = year_cat_blobs.get(year, {})

        if not rids:
            per_cat_empty = {cat: 0 for cat in _SUMMARY_SOURCES}
            summaries.append({
                "date": s["date"],
                "total_races": s["total_races"],
                "venues": s["venues"],
                "race_list_exists": s["race_list_exists"],
                "race_list_row_count": s.get("race_list_row_count", 0),
                "progress": {
                    "filled": 0,
                    "total": 0,
                    "pct": 0,
                    "per_cat": per_cat_empty,
                },
            })
            continue

        filled = 0
        total = len(rids) * n_cats
        per_cat: dict[str, int] = {}
        for cat in _SUMMARY_SOURCES:
            cat_keys = blobs_for_year.get(cat, set())
            count = sum(1 for rid in rids if rid in cat_keys)
            per_cat[cat] = count
            filled += count

        pct = round(filled / total * 100, 1) if total > 0 else 0

        summaries.append({
            "date": s["date"],
            "total_races": s["total_races"],
            "venues": s["venues"],
            "race_list_exists": s["race_list_exists"],
            "race_list_row_count": s.get("race_list_row_count", 0),
            "progress": {
                "filled": filled,
                "total": total,
                "pct": pct,
                "per_cat": per_cat,
            },
        })

    payload = {
        "dates": summaries,
        "gcs_enabled": storage.gcs_enabled,
    }
    _scrape_summary_cache["data"] = payload
    _scrape_summary_cache["ts"] = now
    _scrape_summary_cache["gen"] = _SCRAPE_SUMMARY_FILTER_GEN
    return payload


def _sync_enqueue_incomplete_summary_dates(body: dict | None) -> dict[str, Any]:
    """
    モニター「データ保有率」と同じ scrape-summary-all の基準で、
    progress.pct < 100 の開催日を新しい日付から順に date_all（キュー）へ投入する。
    """
    body = body or {}
    force = bool(body.get("force_summary")) or str(body.get("force") or "") == "1"
    if force:
        _scrape_summary_cache["data"] = None
        _scrape_summary_cache["ts"] = 0.0
        _scrape_summary_cache["gen"] = 0

    payload = _build_scrape_summary_all()
    dates = payload.get("dates") or []

    all_incomplete: list[str] = []
    for d in dates:
        prog = d.get("progress") or {}
        pct = prog.get("pct")
        try:
            pct_f = float(pct) if pct is not None else 0.0
        except (TypeError, ValueError):
            pct_f = 0.0
        if pct_f < 100.0 - 1e-6:
            ds = d.get("date")
            if ds:
                all_incomplete.append(str(ds))

    max_dates = int(body.get("max_dates") or 500)
    max_dates = max(1, min(max_dates, 2000))
    to_enqueue = all_incomplete[:max_dates]

    from src.scraper.scrape_policy import coerce_bool

    smart_skip = body.get("smart_skip", True)
    if isinstance(smart_skip, str):
        smart_skip = smart_skip.strip().lower() not in ("0", "false", "no")
    else:
        smart_skip = bool(smart_skip)

    from src.scraper.job_queue import ScrapeJobQueue

    queue = ScrapeJobQueue()
    created = requeued = duplicate = 0
    details: list[dict[str, Any]] = []
    for date in to_enqueue:
        job_spec: dict[str, Any] = {
            "job_kind": "date",
            "target_id": date,
            "tasks": ["date_all"],
            "smart_skip": smart_skip,
        }
        if "overwrite" in body:
            job_spec["overwrite"] = coerce_bool(body.get("overwrite"), default=False)
        result = queue.add_job(job_spec)
        act = result.get("action", "created")
        if act == "created":
            created += 1
        elif act == "requeued":
            requeued += 1
        else:
            duplicate += 1
        details.append({"date": date, "action": act, "job_id": result.get("job_id")})

    return {
        "status": "ok",
        "gcs_enabled": payload.get("gcs_enabled"),
        "total_incomplete": len(all_incomplete),
        "enqueued_dates": len(to_enqueue),
        "max_dates_cap": max_dates,
        "created": created,
        "requeued": requeued,
        "duplicate": duplicate,
        "order": "newest_first",
        "details": details,
    }


@app.get("/api/data/{category}/{key}", response_class=JSONResponse)
async def get_raw_data(category: str, key: str):
    """GCS上の生JSONデータを返す。"""
    def _load():
        if category == "race_performance":
            year = (key or "")[:4]
            if not year.isdigit() or len(key) < 10:
                return None, "race_performance"
            # GCS → L2 ディスクキャッシュ経由
            data = _get_storage().load("race_performance", key)
            if data is not None:
                return data, None
            # フォールバック: page_reference ローカルファイル
            from src.config.data_paths import RACE_PERFORMANCE_DIR
            path = RACE_PERFORMANCE_DIR / "races" / year / f"{key}.json"
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as f:
                        return json.load(f), None
                except Exception:
                    pass
            return None, "race_performance"
        storage = _get_storage()
        if category in ("smartrc_runners", "smartrc_horses", "smartrc_fullresults"):
            data = storage.load("smartrc_race", key)
            if data is None:
                return None, category
            if category == "smartrc_runners":
                return {
                    "race_id": data.get("race_id", key),
                    "rcode": data.get("rcode", ""),
                    "source": "smartrc",
                    "runners": data.get("runners", []),
                    "_meta": data.get("_meta", {}),
                }, None
            if category == "smartrc_horses":
                return {
                    "race_id": data.get("race_id", key),
                    "source": "smartrc",
                    "horses": data.get("horses", {}),
                    "_meta": data.get("_meta", {}),
                }, None
            return {
                "race_id": data.get("race_id", key),
                "source": "smartrc",
                "fullresults": data.get("fullresults", {}),
                "_meta": data.get("_meta", {}),
            }, None
        data = storage.load(category, key)
        if data is None:
            return None, category
        return data, None

    result, err_cat = await asyncio.to_thread(_load)
    if result is None:
        return JSONResponse(
            {"error": f"{err_cat}/{key} not found"}, status_code=404)
    return JSONResponse(result)


@app.get("/api/horse/{horse_id}/detail", response_class=JSONResponse)
async def api_horse_detail(horse_id: str, race_id: str = ""):
    """馬詳細情報 (netkeiba + SmartRC + 統計) を集約して返す。"""
    def _load():
        storage = _get_storage()
        result: dict[str, Any] = {"horse_id": horse_id}
        hr = storage.load("horse_result", horse_id)
        if hr:
            meta = hr.pop("_meta", {})
            result["info"] = {
                k: v for k, v in hr.items() if k != "race_history"
            }
            result["race_history"] = hr.get("race_history", [])
            result["_meta_netkeiba"] = meta
        else:
            result["info"] = {}
            result["race_history"] = []
        smartrc_horse = None
        smartrc_fullresults: list[dict] = []
        if race_id:
            smartrc = storage.load("smartrc_race", race_id)
            if smartrc:
                horses_dict = smartrc.get("horses", {})
                smartrc_horse = horses_dict.get(horse_id)
                fr_dict = smartrc.get("fullresults", {})
                smartrc_fullresults = fr_dict.get(horse_id, [])
        result["smartrc_horse"] = smartrc_horse
        result["smartrc_fullresults"] = smartrc_fullresults
        ped5 = None
        try:
            ped5 = storage.load("horse_pedigree_5gen", horse_id)
        except Exception:
            ped5 = None
        pedigree = _build_pedigree(hr, smartrc_horse, ped5)
        result["pedigree"] = pedigree
        stats = _calc_horse_stats(result["race_history"], smartrc_fullresults)
        result["stats"] = stats
        return result

    result = await asyncio.to_thread(_load)
    return JSONResponse(result)


def _yyyymmdd_from_date_str(raw: str) -> str:
    """YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD → YYYYMMDD。"""
    s = (raw or "").strip()
    if not s:
        return ""
    if len(s) >= 8 and s[:8].isdigit():
        return s[:8]
    if len(s) >= 10 and s[4] in "-/":
        y, m, d = s[:4], s[5:7], s[8:10]
        if y.isdigit() and m.isdigit() and d.isdigit():
            return f"{y}{m}{d}"
    return ""


def _filter_horse_history_before(
    history: list[dict],
    *,
    limit: int,
    before_date: str = "",
    exclude_race_id: str = "",
) -> list[dict]:
    """race_history（新しい順想定）から、指定日より前・除外 race_id 以外を最大 limit 件。"""
    cutoff = _yyyymmdd_from_date_str(before_date)
    ex = (exclude_race_id or "").strip()
    out: list[dict] = []
    for rec in history:
        if ex and str(rec.get("race_id") or "").strip() == ex:
            continue
        rdate = _yyyymmdd_from_date_str(str(rec.get("date") or rec.get("race_date") or ""))
        if cutoff and rdate and rdate >= cutoff:
            continue
        out.append(rec)
        if len(out) >= limit:
            break
    return out


@app.get("/api/horse/{horse_id}/recent_races", response_class=JSONResponse)
async def api_horse_recent_races(
    horse_id: str,
    limit: int = Query(10, ge=1, le=30),
    before_date: str = Query("", description="YYYYMMDD — この開催日より前のみ（当日含まず）"),
    exclude_race_id: str = Query("", description="除外する race_id（分析対象レース）"),
):
    """horse_result の race_history から直近 N 走（新しい順）を返す。"""
    def _load():
        storage = _get_storage()
        hr = storage.load("horse_result", horse_id)
        if not hr:
            return {"horse_id": horse_id, "horse_name": "", "races": [], "count": 0}
        history = list(hr.get("race_history") or [])
        if before_date or exclude_race_id:
            races = _filter_horse_history_before(
                history,
                limit=limit,
                before_date=before_date,
                exclude_race_id=exclude_race_id,
            )
        else:
            races = history[:limit]
        return {
            "horse_id": horse_id,
            "horse_name": hr.get("horse_name") or "",
            "races": races,
            "count": len(races),
        }

    return JSONResponse(await asyncio.to_thread(_load))


def _horse_race_performance_history_rows(horse_id: str, limit: int) -> list[dict[str, Any]]:
    """馬の戦績に紐づくレースについて、race_performance の per-race JSON から当該馬の行を収集する。"""
    from src.config.data_paths import RACE_PERFORMANCE_DIR

    storage = _get_storage()
    hr = storage.load("horse_result", horse_id)
    if not hr:
        return []
    history = hr.get("race_history") or []
    rows: list[dict[str, Any]] = []
    # 戦績行数が多い馬でもディスク読み取りが暴れないよう上限
    max_reads = min(len(history), 400)
    for rec in history[:max_reads]:
        rid = (rec.get("race_id") or "").strip()
        if len(rid) != 12 or not rid.isdigit():
            continue
        year = rid[:4]
        # GCS → L2 ディスクキャッシュ経由で取得
        data = storage.load("race_performance", rid)
        if data is None:
            # フォールバック: page_reference ローカルファイル
            path = RACE_PERFORMANCE_DIR / "races" / year / f"{rid}.json"
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
        if data is None:
            continue
        match: dict[str, Any] | None = None
        for ent in data.get("entries") or []:
            if str(ent.get("horse_id")) == str(horse_id):
                match = ent
                break
        if not match:
            continue
        rows.append(
            {
                "race_id": rid,
                "race_date": match.get("race_date") or rec.get("date"),
                "venue": match.get("venue") or rec.get("venue"),
                "race_name": rec.get("race_name"),
                "surface": match.get("surface") or rec.get("surface"),
                "distance": match.get("distance") or rec.get("distance"),
                "finish_position": match.get("finish_position"),
                "run_performance_raw": match.get("run_performance_raw"),
                "run_performance_final": match.get("run_performance_final"),
                "run_performance_final_std": match.get("run_performance_final_std"),
                "run_performance_final_pct": match.get("run_performance_final_pct"),
                "time_figure": match.get("time_figure"),
            }
        )

    def _sort_key(x: dict[str, Any]) -> str:
        return str(x.get("race_date") or "")[:10]

    rows.sort(key=_sort_key, reverse=True)
    return rows[:limit]


@app.get("/api/horse/{horse_id}/race_performance_history", response_class=JSONResponse)
async def api_horse_race_performance_history(
    horse_id: str,
    limit: int = Query(80, ge=1, le=200),
):
    """戦績に現れるレースのうち、race_performance が生成済みのものだけを返す。"""
    rows = await asyncio.to_thread(_horse_race_performance_history_rows, horse_id, limit)
    return JSONResponse(
        {
            "horse_id": horse_id,
            "rows": rows,
            "count": len(rows),
        }
    )


def _tail_sire_line(
    ped5: dict | None,
    smartrc_horse: dict | None,
    hr: dict | None,
) -> list[dict[str, str]]:
    """
    父 → 母父 → 母母父 → 母母母父 → 母母母母父（各段の「父」）。
    horse_pedigree_5gen の ancestors は generation 1..5, position 0 始まり。
    """
    slots: list[tuple[str, tuple[int, int]]] = [
        ("父", (1, 0)),
        ("母父", (2, 2)),
        ("母母父", (3, 6)),
        ("母母母父", (4, 14)),
        ("母母母母父", (5, 30)),
    ]
    by_pos: dict[tuple[int, int], str] = {}
    if ped5 and isinstance(ped5.get("ancestors"), list):
        for a in ped5["ancestors"]:
            if not isinstance(a, dict):
                continue
            try:
                g = int(a["generation"])
                p = int(a["position"])
            except (KeyError, TypeError, ValueError):
                continue
            nm = (a.get("name") or "").strip()
            if nm:
                by_pos[(g, p)] = nm
    out: list[dict[str, str]] = []
    for label, (g, p) in slots:
        name = by_pos.get((g, p), "")
        out.append({"label": label, "name": name})
    if smartrc_horse:
        if not out[0]["name"]:
            out[0]["name"] = (smartrc_horse.get("f_name") or "").strip()
        if not out[1]["name"]:
            out[1]["name"] = (smartrc_horse.get("mf_name") or "").strip()
        if not out[2]["name"]:
            out[2]["name"] = (smartrc_horse.get("mmf_name") or "").strip()
    elif hr:
        if not out[0]["name"]:
            out[0]["name"] = (hr.get("sire") or "").strip()
        if not out[1]["name"]:
            out[1]["name"] = (hr.get("dam_sire") or "").strip()
    return out


def _build_pedigree(
    hr: dict | None,
    smartrc_horse: dict | None,
    ped5: dict | None = None,
) -> dict:
    """血統ツリーを構築。SmartRC のデータを優先し netkeiba で補完。"""
    ped: dict[str, Any] = {"sire": {}, "dam": {}}
    ped["tail_sire_line"] = _tail_sire_line(ped5, smartrc_horse, hr)

    if smartrc_horse:
        ped["sire"] = {
            "name": smartrc_horse.get("f_name", ""),
            "code": smartrc_horse.get("f_code", ""),
            "ll": smartrc_horse.get("f_llcode", ""),
            "sl": smartrc_horse.get("f_slcode", ""),
            "cl": smartrc_horse.get("f_clcode", ""),
            "country": smartrc_horse.get("f_country", ""),
        }
        ped["sire"]["sire"] = {
            "name": smartrc_horse.get("ff_name", ""),
            "ll": smartrc_horse.get("ff_llcode", ""),
            "sl": smartrc_horse.get("ff_slcode", ""),
            "cl": smartrc_horse.get("ff_clcode", ""),
        }
        ped["sire"]["dam"] = {
            "name": smartrc_horse.get("fm_name", ""),
            "ll": smartrc_horse.get("fm_llcode", ""),
        }
        ped["sire"]["sire"]["sire"] = {"name": smartrc_horse.get("fff_name", "")}
        ped["sire"]["sire"]["dam"] = {"name": smartrc_horse.get("ffm_name", "")}
        ped["sire"]["dam"]["sire"] = {"name": smartrc_horse.get("fmf_name", "")}
        ped["sire"]["dam"]["dam"] = {"name": smartrc_horse.get("fmm_name", "")}

        ped["dam"] = {
            "name": smartrc_horse.get("m_name", ""),
            "code": smartrc_horse.get("m_code", ""),
        }
        ped["dam"]["sire"] = {
            "name": smartrc_horse.get("mf_name", ""),
            "ll": smartrc_horse.get("mf_llcode", ""),
            "sl": smartrc_horse.get("mf_slcode", ""),
            "cl": smartrc_horse.get("mf_clcode", ""),
            "country": smartrc_horse.get("mf_country", ""),
        }
        ped["dam"]["dam"] = {
            "name": smartrc_horse.get("mm_name", ""),
        }
        ped["dam"]["sire"]["sire"] = {"name": smartrc_horse.get("mff_name", "")}
        ped["dam"]["sire"]["dam"] = {"name": smartrc_horse.get("mfm_name", "")}
        ped["dam"]["dam"]["sire"] = {"name": smartrc_horse.get("mmf_name", "")}
        ped["dam"]["dam"]["dam"] = {"name": smartrc_horse.get("mmm_name", "")}

    elif hr:
        ped["sire"]["name"] = hr.get("sire", "")
        ped["dam"]["name"] = hr.get("dam", "")
        ped["dam"]["sire"] = {"name": hr.get("dam_sire", "")}

    return ped


def _calc_horse_stats(
    race_history: list[dict],
    smartrc_results: list[dict],
) -> dict:
    """成績統計を計算。"""
    stats: dict[str, Any] = {}
    records = race_history or []
    if not records:
        return stats

    total = len(records)
    wins = sum(1 for r in records if r.get("finish_position") == 1)
    place2 = sum(1 for r in records if r.get("finish_position") in (1, 2))
    place3 = sum(1 for r in records if r.get("finish_position") in (1, 2, 3))
    stats["overall"] = {
        "runs": total, "wins": wins, "top2": place2, "top3": place3,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "top2_rate": round(place2 / total * 100, 1) if total else 0,
        "top3_rate": round(place3 / total * 100, 1) if total else 0,
    }

    by_surface: dict[str, list] = {}
    by_distance: dict[str, list] = {}
    by_venue: dict[str, list] = {}
    by_condition: dict[str, list] = {}

    for r in records:
        pos = r.get("finish_position")
        if not isinstance(pos, (int, float)):
            continue

        surface = r.get("surface", "")
        if surface:
            by_surface.setdefault(surface, []).append(pos)

        dist = r.get("distance", 0)
        if dist:
            bucket = f"{(dist // 200) * 200}-{(dist // 200) * 200 + 199}m"
            by_distance.setdefault(bucket, []).append(pos)

        venue = r.get("venue", "")
        if venue:
            import re as _re
            m = _re.search(r"(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)", venue)
            if m:
                by_venue.setdefault(m.group(1), []).append(pos)

        cond = r.get("track_condition", "")
        if cond:
            by_condition.setdefault(cond, []).append(pos)

    def _summarize(groups: dict[str, list]) -> dict:
        out = {}
        for key, positions in sorted(groups.items()):
            n = len(positions)
            w = sum(1 for p in positions if p == 1)
            t3 = sum(1 for p in positions if p <= 3)
            avg = round(sum(positions) / n, 1)
            out[key] = {
                "runs": n, "wins": w, "top3": t3,
                "win_rate": round(w / n * 100, 1),
                "top3_rate": round(t3 / n * 100, 1),
                "avg_pos": avg,
            }
        return out

    stats["by_surface"] = _summarize(by_surface)
    stats["by_distance"] = _summarize(by_distance)
    stats["by_venue"] = _summarize(by_venue)
    stats["by_condition"] = _summarize(by_condition)

    times = [r.get("last_3f") for r in records if r.get("last_3f") and r["last_3f"] > 0]
    if times:
        stats["last_3f"] = {
            "best": min(times),
            "avg": round(sum(times) / len(times), 1),
            "recent3": times[:3],
        }

    return stats


_BUNDLE_CATEGORIES = [
    "race_shutuba", "race_result", "race_result_on_time", "race_index",
    "race_odds", "race_pair_odds", "race_paddock", "race_barometer",
    "race_trainer_comment",
]

_bundle_cache: dict[str, tuple[float, dict]] = {}
_bundle_cache_lock = threading.Lock()
_BUNDLE_CACHE_TTL = 300


@app.get("/api/race/{race_id}/bundle", response_class=JSONResponse)
async def api_race_bundle(race_id: str):
    """
    全カテゴリ + SmartRC + 全馬horse_resultを一括で返す。
    並列GCSリードで最小レイテンシを実現。
    """
    now = _time.time()
    with _bundle_cache_lock:
        cached = _bundle_cache.get(race_id)
    if cached and (now - cached[0]) < _BUNDLE_CACHE_TTL:
        return JSONResponse(cached[1])

    def _load():
        storage = _get_storage()
        pool = ThreadPoolExecutor(max_workers=12)
        futures = {}
        for cat in _BUNDLE_CATEGORIES:
            futures[cat] = pool.submit(storage.load, cat, race_id)
        futures["smartrc_race"] = pool.submit(storage.load, "smartrc_race", race_id)
        bundle: dict[str, Any] = {"race_id": race_id}
        for cat in _BUNDLE_CATEGORIES:
            bundle[cat] = futures[cat].result()
        smartrc = futures["smartrc_race"].result()
        if smartrc:
            bundle["smartrc_runners"] = smartrc.get("runners", [])
            bundle["smartrc_horses"] = smartrc.get("horses", {})
            bundle["smartrc_fullresults"] = smartrc.get("fullresults", {})
            bundle["smartrc_meta"] = smartrc.get("_meta", {})
        else:
            bundle["smartrc_runners"] = []
            bundle["smartrc_horses"] = {}
            bundle["smartrc_fullresults"] = {}
        horse_ids: list[str] = []
        shutuba = bundle.get("race_shutuba")
        if shutuba and isinstance(shutuba, dict):
            for e in shutuba.get("entries", []):
                hid = e.get("horse_id", "")
                if hid:
                    horse_ids.append(hid)
        horse_futures = {
            hid: pool.submit(storage.load, "horse_result", hid) for hid in horse_ids
        }
        ped_futures = {
            hid: pool.submit(storage.load, "horse_pedigree_5gen", hid)
            for hid in horse_ids
        }
        horses_data: dict[str, Any] = {}
        for hid, fut in horse_futures.items():
            hr = fut.result()
            if hr:
                hr.pop("_meta", None)
                horses_data[hid] = hr
        bundle["horse_results"] = horses_data
        sc_horses = bundle.get("smartrc_horses") or {}
        tail_map: dict[str, list[dict[str, str]]] = {}
        for hid in horse_ids:
            ped5 = None
            try:
                ped5 = ped_futures[hid].result()
            except Exception:
                ped5 = None
            sc = sc_horses.get(hid) if isinstance(sc_horses, dict) else None
            tail_map[hid] = _tail_sire_line(ped5, sc, horses_data.get(hid))
        bundle["horse_pedigree_tail"] = tail_map
        pool.shutdown(wait=False)
        return bundle

    bundle = await asyncio.to_thread(_load)
    with _bundle_cache_lock:
        _bundle_cache[race_id] = (now, bundle)
    return JSONResponse(bundle)


_person_api_cache: dict[str, tuple[float, dict]] = {}
_person_api_cache_lock = threading.Lock()
_PERSON_API_CACHE_TTL = 300


@app.get("/api/person/{ptype}/{person_id}/stats", response_class=JSONResponse)
async def api_person_stats(ptype: str, person_id: str, race_id: str = ""):
    """騎手・調教師の成績情報を返す。"""
    if ptype not in ("jockey", "trainer"):
        return JSONResponse({"error": "type must be jockey or trainer"}, 400)

    now = _time.time()
    api_key = f"{ptype}/{person_id}/{race_id}"
    with _person_api_cache_lock:
        cached = _person_api_cache.get(api_key)
    if cached and (now - cached[0]) < _PERSON_API_CACHE_TTL:
        return JSONResponse(cached[1])

    def _load():
        storage = _get_storage()
        profile = _fetch_person_profile(ptype, person_id)
        race_records = _collect_person_records(
            storage, ptype, person_id, profile.get("name", ""), race_id
        )
        return {
            "person_id": person_id,
            "type": ptype,
            **profile,
            "race_records": race_records,
        }

    result = await asyncio.to_thread(_load)
    with _person_api_cache_lock:
        _person_api_cache[api_key] = (now, result)
    return JSONResponse(result)


_person_cache: dict[str, tuple[float, dict]] = {}
_person_cache_lock = threading.Lock()
_PERSON_CACHE_TTL = 7 * 86400


def _fetch_person_profile(ptype: str, person_id: str) -> dict:
    """netkeiba プロフィールページから基本情報 + 年度別成績を取得。"""
    import time as _tm
    cache_key = f"{ptype}/{person_id}"
    with _person_cache_lock:
        cached = _person_cache.get(cache_key)
    if cached and (_tm.time() - cached[0]) < _PERSON_CACHE_TTL:
        return cached[1]

    from pathlib import Path
    if len(person_id) > 200:
        return {}
    from src.config.data_paths import person_profile_path

    local_path = person_profile_path(ptype, person_id)
    if local_path.exists():
        try:
            data = json.loads(local_path.read_text(encoding="utf-8"))
            fetched = data.get("_fetched_at", 0)
            if _tm.time() - fetched < _PERSON_CACHE_TTL:
                with _person_cache_lock:
                    _person_cache[cache_key] = (_tm.time(), data)
                return data
        except Exception:
            pass

    try:
        result = _scrape_person_profile(ptype, person_id)
        result["_fetched_at"] = _tm.time()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with _person_cache_lock:
            _person_cache[cache_key] = (_tm.time(), result)
        return result
    except Exception as e:
        logger.warning("Person profile scrape failed [%s/%s]: %s", ptype, person_id, e)
        return {"name": "", "yearly_stats": []}


def _scrape_person_profile(ptype: str, person_id: str) -> dict:
    """netkeiba 騎手/調教師プロフィール + 年度別成績をパース。"""
    import requests as _req
    from bs4 import BeautifulSoup as _BS

    _hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
    }

    result: dict[str, Any] = {"person_id": person_id, "type": ptype}

    resp = _req.get(
        f"https://db.netkeiba.com/{ptype}/{person_id}/",
        headers=_hdrs, timeout=15,
    )
    resp.encoding = "euc-jp"
    soup = _BS(resp.text, "html.parser")

    name_div = soup.find("div", class_="db_head_name")
    if name_div:
        h1 = name_div.find("h1")
        if h1:
            raw = h1.get_text(strip=True)
            result["name"] = raw.split("（")[0].split("\n")[0].strip()
            if "（" in raw and "）" in raw:
                result["name_kana"] = raw.split("（")[1].split("）")[0]

    txt_el = soup.find("p", class_="txt_01")
    if txt_el:
        txt = txt_el.get_text("\n", strip=True)
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        if lines:
            result["birthdate"] = lines[0]
        for line in lines:
            if "[" in line and "]" in line:
                result["affiliation"] = line

    tables = soup.find_all("table", class_="nk_tb_common")
    if tables:
        for row in tables[0].find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td:
                key = th.get_text(strip=True)
                val = td.get_text(strip=True)
                if "デビュー" in key:
                    result["debut_year"] = val
                elif "通算勝利" in key:
                    result["career_wins_text"] = val
                elif "本年勝利" in key:
                    result["current_year_wins_text"] = val
                elif "通算獲得" in key:
                    result["career_earnings_text"] = val
                elif "GI" in key or "G1" in key:
                    result["g1_wins_text"] = val
                elif "重賞" in key and "勝利" in key:
                    result["graded_wins_text"] = val
                elif "身長" in key:
                    result["physical"] = val
                elif "出身" in key:
                    result["birthplace"] = val

    def _si(s):
        return int(s.replace(",", "")) if s.replace(",", "").isdigit() else 0

    def _sf(s):
        s = s.replace("％", "").replace("%", "").strip()
        if s.startswith("."):
            s = "0" + s
        try:
            v = float(s)
            return round(v * 100, 1) if v < 1 else round(v, 1)
        except ValueError:
            return 0.0

    resp2 = _req.get(
        f"https://db.netkeiba.com/{ptype}/result/{person_id}/",
        headers=_hdrs, timeout=15,
    )
    resp2.encoding = "euc-jp"
    soup2 = _BS(resp2.text, "html.parser")

    yearly: list[dict] = []
    result_table = soup2.find("table", class_="nk_tb_common")
    if result_table:
        ths = [th.get_text(strip=True) for th in result_table.find_all("th")]
        for row in result_table.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 6:
                continue
            vals = [td.get_text(strip=True) for td in tds]
            year_str = vals[0].strip()

            if year_str == "累計":
                result["career_total"] = {
                    "wins": _si(vals[2]),
                    "seconds": _si(vals[3]),
                    "thirds": _si(vals[4]),
                    "others": _si(vals[5]),
                    "rides": _si(vals[2]) + _si(vals[3]) + _si(vals[4]) + _si(vals[5]),
                }
                if len(vals) > 16:
                    result["career_total"]["win_rate"] = _sf(vals[16])
                if len(vals) > 17:
                    result["career_total"]["top2_rate"] = _sf(vals[17])
                if len(vals) > 18:
                    result["career_total"]["top3_rate"] = _sf(vals[18])
                if len(vals) > 19:
                    result["career_total"]["earnings"] = vals[19]
                continue

            if not year_str or not year_str[0].isdigit():
                continue

            try:
                entry: dict[str, Any] = {
                    "year": year_str,
                    "rank": _si(vals[1]),
                    "wins": _si(vals[2]),
                    "seconds": _si(vals[3]),
                    "thirds": _si(vals[4]),
                    "others": _si(vals[5]),
                }
                entry["rides"] = entry["wins"] + entry["seconds"] + entry["thirds"] + entry["others"]

                if len(vals) > 16:
                    entry["win_rate"] = _sf(vals[16])
                if len(vals) > 17:
                    entry["top2_rate"] = _sf(vals[17])
                if len(vals) > 18:
                    entry["top3_rate"] = _sf(vals[18])
                if len(vals) > 19:
                    entry["earnings"] = vals[19]

                yearly.append(entry)
            except (ValueError, IndexError):
                continue

    result["yearly_stats"] = yearly
    return result


def _collect_person_records(
    storage, ptype: str, person_id: str, person_name: str, race_id: str,
) -> list[dict]:
    """
    race_id のレースに出走する全馬の horse_result.race_history から
    該当騎手/調教師のレコードを集約する。

    騎手: race_history の jockey_name でマッチ
    調教師: horse_result.trainer でマッチ → その馬の全戦績を収集
    """
    import re as _re

    records: list[dict] = []
    seen: set[str] = set()

    card = storage.load("race_shutuba", race_id) if race_id else None
    entries = card.get("entries", []) if card else []

    id_key = "jockey_id" if ptype == "jockey" else "trainer_id"
    name_key = "jockey_name" if ptype == "jockey" else "trainer_name"

    name_variants: set[str] = set()
    if person_name:
        clean = _re.sub(r"[▲△☆★◎○]", "", person_name).strip()
        name_variants.add(person_name)
        if clean:
            name_variants.add(clean)

    for e in entries:
        if e.get(id_key, "") == person_id:
            raw = e.get(name_key, "")
            if raw:
                clean = _re.sub(r"[▲△☆★◎○]", "", raw).strip()
                name_variants.add(raw)
                if clean:
                    name_variants.add(clean)

    smartrc = storage.load("smartrc_race", race_id) if race_id else None
    if smartrc:
        sc_id_key = "jcode" if ptype == "jockey" else "tcode"
        sc_name_key = "jname8" if ptype == "jockey" else "tname8"
        for r in smartrc.get("runners", []):
            if r.get(sc_id_key, "") == person_id:
                full = r.get(sc_name_key, "")
                if full:
                    name_variants.add(full)
                    name_variants.add(full.strip())

    if not name_variants:
        return records

    def _name_match(candidate: str) -> bool:
        if not candidate:
            return False
        cand_clean = _re.sub(r"[▲△☆★◎○]", "", candidate).strip()
        for v in name_variants:
            v_clean = _re.sub(r"[▲△☆★◎○]", "", v).strip()
            if v_clean and cand_clean:
                if v_clean in cand_clean or cand_clean in v_clean:
                    return True
        return False

    horse_ids: set[str] = set()
    for e in entries:
        hid = e.get("horse_id", "")
        if hid:
            horse_ids.add(hid)

    pool = ThreadPoolExecutor(max_workers=min(len(horse_ids), 12))
    hr_futures = {hid: pool.submit(storage.load, "horse_result", hid) for hid in horse_ids}
    pool.shutdown(wait=True)

    for hid in horse_ids:
        hr = hr_futures[hid].result()
        if not hr:
            continue
        horse_name = hr.get("horse_name", "")

        if ptype == "trainer":
            trainer_of_horse = hr.get("trainer", "")
            if not _name_match(trainer_of_horse):
                continue
            for rec in hr.get("race_history", []):
                rec_key = f"{rec.get('race_id', '')}/{hid}"
                if rec_key in seen:
                    continue
                seen.add(rec_key)
                records.append({**rec, "horse_name": horse_name, "horse_id": hid})
        else:
            for rec in hr.get("race_history", []):
                if not _name_match(rec.get("jockey_name", "")):
                    continue
                rec_key = f"{rec.get('race_id', '')}/{hid}"
                if rec_key in seen:
                    continue
                seen.add(rec_key)
                records.append({**rec, "horse_name": horse_name, "horse_id": hid})

    records.sort(key=lambda r: r.get("date", ""), reverse=True)
    return records


@app.get("/data-viewer", response_class=HTMLResponse)
async def data_viewer(request: Request, race_id: str = "", category: str = ""):
    """生JSONデータビューア。"""
    return templates.TemplateResponse("admin/data_viewer.html", {
        "request": request,
        "race_id": race_id,
        "category": category,
        "current_page": "data_viewer",
        "breadcrumbs": [],
    })


# ═══════════════════════════════════════════════════════
# スクレイピング トリガー API
# ═══════════════════════════════════════════════════════

_scrape_key_locks: dict[str, threading.Lock] = {}
_scrape_key_locks_lock = threading.Lock()
_scrape_jobs: dict[str, dict] = {}
_scrape_jobs_lock = threading.Lock()
_runner_lock = threading.Lock()

# グローバル並行ジョブ制限: 同時に実行できるスクレイピングジョブ数
_MAX_CONCURRENT_SCRAPE = 3
_scrape_semaphore = threading.Semaphore(_MAX_CONCURRENT_SCRAPE)
_scrape_queue_order: list[str] = []  # キュー待ちジョブIDの順序
_scrape_queue_lock = threading.Lock()


def _get_scrape_lock(key: str) -> threading.Lock:
    """キー別のスクレイピングロックを取得する。異なるレース/日付を同時実行可能にする。"""
    with _scrape_key_locks_lock:
        if key not in _scrape_key_locks:
            _scrape_key_locks[key] = threading.Lock()
        return _scrape_key_locks[key]


def _get_runner():
    """ScraperRunner のシングルトン (重複インスタンス防止)。"""
    if not hasattr(_get_runner, "_inst"):
        with _runner_lock:
            if not hasattr(_get_runner, "_inst"):
                from src.scraper.run import ScraperRunner
                _get_runner._inst = ScraperRunner(
                    interval=1.0, cache=True, auto_login=True,
                )
    return _get_runner._inst


CATEGORY_TO_METHOD = {
    "race_shutuba":         "scrape_race_card",
    "race_result":          "scrape_race_result",
    "race_result_on_time":  "scrape_race_result_on_time",
    "race_index":           "scrape_speed_index",
    "race_odds":            "scrape_odds",
    "race_pair_odds":       "scrape_pair_odds",
    "race_paddock":         "scrape_paddock",
    "race_barometer":       "scrape_barometer",
    "race_trainer_comment": "scrape_trainer_comment",
}


_race_date_index: dict[str, str] = {}
_race_date_index_built = False
_race_date_index_lock = threading.Lock()


def _ensure_race_date_index(storage):
    """race_id → date の逆引きインデックスをローカル race_lists から構築する。"""
    global _race_date_index_built
    if _race_date_index_built:
        return
    with _race_date_index_lock:
        if _race_date_index_built:
            return
    # I/O outside lock to avoid holding lock during I/O
    new_index: dict[str, str] = {}
    dates = storage.list_keys("race_lists")
    for date in dates:
        rl = storage.load("race_lists", date)
        if not rl:
            continue
        for r in rl.get("races", []):
            rid = r.get("race_id", "")
            if rid:
                new_index[rid] = date
    with _race_date_index_lock:
        if _race_date_index_built:
            return
        _race_date_index.update(new_index)
        _race_date_index_built = True


def _resolve_date_for_race(storage, race_id: str) -> str:
    """race_id が属する開催日付 (YYYYMMDD) を逆引きする (キャッシュ付き)。"""
    with _race_date_index_lock:
        if race_id in _race_date_index:
            return _race_date_index[race_id]
    _ensure_race_date_index(storage)
    with _race_date_index_lock:
        return _race_date_index.get(race_id, "")


def _run_scrape_job(job_id: str, race_id: str, category: str,
                    force: bool = False):
    """バックグラウンドスレッドでスクレイピングを実行する。

    グローバルセマフォで同時実行数を制限し、超過分はキュー待ちする。
    force=True の場合、鮮度チェックをスキップして強制再取得する。
    """
    with _scrape_queue_lock:
        _scrape_queue_order.append(job_id)
    with _scrape_jobs_lock:
        job = _scrape_jobs[job_id]
        job["status"] = "queued"
        job["queue_position"] = _get_queue_position(job_id)

    logger.info("ジョブ %s キュー登録 (待ち: %d)", job_id, job["queue_position"])

    _scrape_semaphore.acquire()
    try:
        with _scrape_queue_lock:
            if job_id in _scrape_queue_order:
                _scrape_queue_order.remove(job_id)
        with _scrape_jobs_lock:
            job["status"] = "running"
            job["started_running_at"] = _time.time()
            job["queue_position"] = 0
            job["progress"] = {"current": 0, "total": 0, "current_label": ""}
        _update_queue_positions()

        logger.info("ジョブ %s 実行開始 (%s/%s)", job_id, race_id, category)

        runner = _get_runner()
        lock_key = f"{race_id}:{category}"
        with _get_scrape_lock(lock_key):
            if category == "date_all":
                race_list_data = runner.storage.load("race_lists", race_id)
                if race_list_data:
                    raw_races = race_list_data.get("races", [])
                    jra_races = [r for r in raw_races
                                 if r.get("race_id") and _is_jra_race(r["race_id"])]
                    job["progress"]["total"] = len(jra_races)
                    for i, race in enumerate(jra_races):
                        rid = race["race_id"]
                        label = f"{race.get('venue', '')} {race.get('race_number', i+1)}R"
                        job["progress"]["current"] = i
                        job["progress"]["current_label"] = label
                        runner.scrape_race_all(rid, smart_skip=not force)
                    job["progress"]["current"] = len(jra_races)
                    job["progress"]["current_label"] = "完了"
                else:
                    runner.scrape_date_all(race_id, smart_skip=not force)
            elif category == "all":
                runner.scrape_race_all(race_id, smart_skip=not force)
            elif category == "horse_result":
                card = runner.storage.load("race_shutuba", race_id)
                if card:
                    entries = card.get("entries", [])
                    job["progress"]["total"] = len(entries)
                    for i, entry in enumerate(entries):
                        hid = entry.get("horse_id", "")
                        if hid:
                            job["progress"]["current"] = i
                            job["progress"]["current_label"] = entry.get("horse_name", hid)
                            runner.scrape_horse(hid, skip_existing=not force)
                    job["progress"]["current"] = len(entries)
            elif category in ("horse_pedigree", "horse_pedigree_5gen"):
                card = runner.storage.load("race_shutuba", race_id)
                if card:
                    entries = card.get("entries", [])
                    sire_ids: set[str] = set()
                    horse_id_set: set[str] = set()
                    job["progress"]["total"] = len(entries)
                    for i, entry in enumerate(entries):
                        hid = entry.get("horse_id", "")
                        if hid:
                            horse_id_set.add(hid)
                            job["progress"]["current"] = i
                            job["progress"]["current_label"] = entry.get("horse_name", hid)
                            try:
                                rec = runner.scrape_horse_pedigree_5gen(
                                    hid, skip_existing=True,
                                )
                                if rec:
                                    for a in rec.get("ancestors", []):
                                        if a.get("generation") == 1 and a.get("position") == 0:
                                            sid = a.get("horse_id", "")
                                            if sid:
                                                sire_ids.add(sid)
                                            break
                            except Exception as e:
                                logger.debug("5gen取得失敗 [%s]: %s", hid, e)
                    sire_ids -= horse_id_set
                    sire_list = sorted(sire_ids)
                    job["progress"]["total"] = len(entries) + len(sire_list)
                    for j, sid in enumerate(sire_list):
                        job["progress"]["current"] = len(entries) + j
                        job["progress"]["current_label"] = f"種牡馬: {sid}"
                        try:
                            runner.scrape_horse_pedigree_5gen(
                                sid, skip_existing=True,
                            )
                        except Exception as e:
                            logger.debug("5gen取得失敗(種牡馬) [%s]: %s", sid, e)
                    job["progress"]["current"] = len(entries) + len(sire_list)
            elif category == "race_list":
                runner.scrape_race_list(race_id)
            elif category == "smartrc_race":
                runner.scrape_smartrc(race_id)
            else:
                method_name = CATEGORY_TO_METHOD.get(category)
                if method_name:
                    getattr(runner, method_name)(race_id, skip_existing=not force)
        with _scrape_jobs_lock:
            job["status"] = "done"
    except Exception as e:
        with _scrape_jobs_lock:
            job["status"] = "error"
            job["error"] = str(e)
        logger.error("ジョブ %s エラー: %s", job_id, e)
    finally:
        _scrape_semaphore.release()
        with _scrape_queue_lock:
            if job_id in _scrape_queue_order:
                _scrape_queue_order.remove(job_id)
        _update_queue_positions()
        with _scrape_jobs_lock:
            job["finished_at"] = _time.time()
    _invalidate_status_cache(race_id)


def _get_queue_position(job_id: str) -> int:
    """キュー内での順位を返す (1-indexed, 0=実行中)。"""
    with _scrape_queue_lock:
        if job_id in _scrape_queue_order:
            return _scrape_queue_order.index(job_id) + 1
    return 0


def _update_queue_positions():
    """キュー待ちジョブの順位を更新する。"""
    with _scrape_queue_lock:
        waiting = list(_scrape_queue_order)
    with _scrape_jobs_lock:
        for i, jid in enumerate(waiting):
            if jid in _scrape_jobs and _scrape_jobs[jid]["status"] == "queued":
                _scrape_jobs[jid]["queue_position"] = i + 1


def _invalidate_status_cache(race_id: str):
    """スクレイピング完了後にキャッシュを無効化する。"""
    with _status_cache_lock:
        date_key = _resolve_date_for_race(_get_storage(), race_id)
        if date_key and date_key in _status_cache:
            del _status_cache[date_key]
        if race_id in _status_cache:
            del _status_cache[race_id]
    with _horse_ids_cache_lock:
        _horse_ids_cache.pop(race_id, None)


class ScrapeRequest(BaseModel):
    race_id: str
    category: str = "all"
    force: bool = False


# Race-entity tasks recognized by ScrapeJobQueue
_QUEUE_RACE_TASKS = {
    "race_all", "race_result", "race_result_on_time", "race_result_lap",
    "race_shutuba", "race_odds", "race_pair_odds", "race_index",
    "race_paddock", "race_barometer",
    "race_trainer_comment", "smartrc",
    # race_shutuba 経由で horse_id を解決して処理する horse-from-race タスク
    "horse_result", "horse_pedigree_5gen", "horse_training",
}
# Date-entity tasks recognized by ScrapeJobQueue
_QUEUE_DATE_TASKS = {"race_list", "date_results", "date_cards", "date_all"}


def _trigger_to_queue_spec(race_id: str, category: str, force: bool) -> dict | None:
    """Convert legacy scrape-trigger params to a ScrapeJobQueue job spec.
    Returns None for categories not supported by the queue (e.g. horse_result).
    """
    overwrite = force
    smart_skip = not force
    if category in _QUEUE_DATE_TASKS:
        return {"job_kind": "date", "target_id": race_id, "tasks": [category],
                "overwrite": overwrite, "smart_skip": smart_skip}
    # "all" is the legacy alias for "race_all"
    task = "race_all" if category == "all" else category
    if task in _QUEUE_RACE_TASKS:
        return {"job_kind": "race", "target_id": race_id, "tasks": [task],
                "overwrite": overwrite, "smart_skip": smart_skip}
    return None  # horse_result, horse_pedigree, etc. → legacy path


def _iso_to_unix(s: str | None) -> float | None:
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _queue_status_to_legacy(status: str) -> str:
    if status in ("pending", "precheck"):
        return "queued"
    if status == "running":
        return "running"
    if status == "completed":
        return "done"
    if status == "failed":
        return "error"
    return status


def _category_from_queue_tasks(tasks: list) -> str:
    if not tasks:
        return "unknown"
    t = tasks[0]
    if t == "race_all":
        return "all"
    if t == "smartrc":
        return "smartrc_race"
    return t


@app.post("/api/scrape-trigger", response_class=JSONResponse)
async def trigger_scrape(body: ScrapeRequest):
    """
    指定レース x カテゴリのスクレイピングをバックグラウンドで実行する。

    category:
      - "all": 全データソース一括 (1レース)
      - "date_all": 日付の全レース全データ一括 (race_id に YYYYMMDD を指定)
      - "race_shutuba", "race_result", ... : 個別ソース
      - "horse_result": 当該レースの全馬情報
      - "race_list": レース一覧 (race_id には日付 YYYYMMDD を指定)

    force: true にすると鮮度チェックをスキップして強制再取得する。
    """
    queue_spec = _trigger_to_queue_spec(body.race_id, body.category, body.force)

    if queue_spec is not None:
        # Route through file-based ScrapeJobQueue
        from src.scraper.job_queue import ScrapeJobQueue
        queue = ScrapeJobQueue()
        result = queue.add_job(queue_spec)
        _kick_scrape_queue_worker()
        return JSONResponse({
            "status": result.get("action", result.get("status", "queued")),
            "job_id": result.get("job_id", ""),
            "race_id": body.race_id,
            "category": body.category,
            "force": body.force,
            "queue_action": result.get("action"),
        })

    # Legacy in-memory path for categories not supported by the queue
    # (e.g. horse_result, horse_pedigree, horse_pedigree_5gen)
    job_id = f"{body.race_id}:{body.category}:{int(_time.time())}"

    with _scrape_jobs_lock:
        active = [j for j in _scrape_jobs.values()
                  if j["status"] in ("running", "queued")
                  and j["race_id"] == body.race_id
                  and j["category"] == body.category]
    if active:
        return JSONResponse({
            "status": "already_running",
            "job_id": active[0]["job_id"],
            "message": f"{body.race_id}/{body.category} は既に実行中またはキュー待ちです",
        })

    with _scrape_jobs_lock:
        running_count = sum(1 for j in _scrape_jobs.values()
                           if j["status"] == "running")
        queued_count = sum(1 for j in _scrape_jobs.values()
                          if j["status"] == "queued")

        _scrape_jobs[job_id] = {
            "job_id": job_id,
            "race_id": body.race_id,
            "category": body.category,
            "status": "queued",
            "force": body.force,
            "created_at": _time.time(),
            "started_at": _time.time(),
            "started_running_at": None,
            "finished_at": None,
            "error": None,
            "queue_position": queued_count + 1,
        }

    thread = threading.Thread(
        target=_run_scrape_job,
        args=(job_id, body.race_id, body.category, body.force),
        daemon=True,
    )
    thread.start()

    will_queue = running_count >= _MAX_CONCURRENT_SCRAPE
    return JSONResponse({
        "status": "queued" if will_queue else "started",
        "job_id": job_id,
        "race_id": body.race_id,
        "category": body.category,
        "force": body.force,
        "running_jobs": running_count,
        "queued_jobs": queued_count + (1 if will_queue else 0),
        "max_concurrent": _MAX_CONCURRENT_SCRAPE,
    })


@app.get("/api/scrape-jobs", response_class=JSONResponse)
async def get_scrape_jobs():
    """実行中・キュー待ち・完了済みのスクレイピングジョブ一覧と統計を返す。

    ScrapeJobQueue（ファイルベース）のジョブを正とし、queue_job_progress で進捗を補完。
    キューに乗らないカテゴリ（horse_result 等）はレガシー _scrape_jobs からも補完する。
    """
    now = _time.time()
    cutoff = now - 120  # show completed jobs up to 2 minutes after finish

    from src.scraper.job_queue import ScrapeJobQueue, _queue_parallel_workers
    from src.scraper.queue_job_progress import get_progress_snapshot_for_api

    queue = ScrapeJobQueue()
    q_jobs = queue.load_queue()
    progress_data = get_progress_snapshot_for_api()
    progress_by_job = {str(p["job_id"]): p for p in progress_data.get("jobs", [])}

    result_jobs: list[dict] = []
    for j in q_jobs:
        status = _queue_status_to_legacy(str(j.get("status") or ""))
        completed_at = _iso_to_unix(j.get("completed_at"))
        started_at = _iso_to_unix(j.get("started_at"))
        queued_at = _iso_to_unix(j.get("queued_at"))

        if status in ("done", "error") and completed_at and completed_at < cutoff:
            continue

        tasks = j.get("tasks") or j.get("types") or []
        prog = progress_by_job.get(str(j.get("job_id", "")), {})

        result_jobs.append({
            "job_id": j["job_id"],
            "race_id": j.get("target_id") or j.get("race_id", ""),
            "category": _category_from_queue_tasks(tasks),
            "status": status,
            "force": not bool(j.get("smart_skip", True)),
            "created_at": queued_at,
            "started_at": started_at or queued_at,
            "started_running_at": started_at,
            "finished_at": completed_at,
            "error": j.get("error"),
            "progress": {
                "current": prog.get("done", 0),
                "total": prog.get("total", 0),
                "current_label": prog.get("current_label") or prog.get("label", ""),
            } if prog else None,
            "_source": "queue",
        })

    # Append legacy in-memory jobs (only for categories not routed to queue)
    with _scrape_jobs_lock:
        for j in _scrape_jobs.values():
            if j["status"] in ("running", "queued") or (
                j.get("finished_at") and now - j["finished_at"] < 120
            ):
                result_jobs.append(dict(j, _source="legacy"))

    running_count = sum(1 for j in result_jobs if j["status"] == "running")
    queued_count = sum(1 for j in result_jobs if j["status"] == "queued")
    n_workers = _queue_parallel_workers()

    return JSONResponse({
        "jobs": result_jobs,
        "stats": {
            "running": running_count,
            "queued": queued_count,
            "max_concurrent": n_workers,
            "slots_available": max(0, n_workers - running_count),
            "total_requests": 0,
        },
    })


# ═══════════════════════════════════════════════════════
# 未来のレース & スクレイピングキュー管理
# ═══════════════════════════════════════════════════════

# 既定ウィンドウ（start/end 未指定）の upcoming 応答を短時間キャッシュ（netkeiba 負荷・体感速度）
_UPCOMING_RACES_DEFAULT_CACHE: dict[str, Any] = {"payload": None, "ts": 0.0}
_UPCOMING_RACES_DEFAULT_TTL = 300.0


@app.get("/api/upcoming-races", response_class=JSONResponse)
async def get_upcoming_races(
    start_date: str = None,
    end_date: str = None,
    local_only: str | None = Query(
        None,
        description="1 のとき race_lists ローカルファイルのみ（netkeiba 未取得・UI 初回表示用）",
    ),
):
    """
    未来のレース一覧を取得（JRA は使わない）。

    各日について data/page_reference/race_lists/{YYYYMMDD}.json が存在し、
    件数が is_plausible_race_day_races（中止で 12 の倍数でない日も妥当）ならそれを優先
    （カレンダー取得との整合・ポーリング負荷軽減）。
    極端に少ない件数などのファイルは無視し netkeiba から再取得する。

    上記が無い場合は netkeiba トップと同じ経路（race_list_get_date_list + race_list_sub）で取得。

    Args:
        start_date: 開始日 (YYYY-MM-DD形式、未指定時は明日)
        end_date: 終了日 (YYYY-MM-DD形式、未指定時は7日後)
    """
    now = _time.time()
    skip_remote = str(local_only or "").strip().lower() in ("1", "true", "yes", "on")
    default_window = start_date is None and end_date is None
    if default_window and not skip_remote:
        uc = _UPCOMING_RACES_DEFAULT_CACHE
        if uc["payload"] is not None and (now - float(uc["ts"])) < _UPCOMING_RACES_DEFAULT_TTL:
            return JSONResponse(uc["payload"])

    def _fetch():
        from datetime import datetime, timedelta
        import json
        from pathlib import Path

        from src.scraper.client import NetkeibaClient
        from src.scraper.netkeiba_top_race_list import (
            fetch_races_for_kaisai_date,
            is_plausible_race_day_races,
        )
        from src.scraper.monitor_future_eligible import include_date_in_data_viewer_race_list

        from src.config.data_paths import RACE_LISTS_DIR

        RACE_LIST_DIR = RACE_LISTS_DIR

        today = datetime.now()
        today_date = today.replace(hour=0, minute=0, second=0, microsecond=0)

        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = today + timedelta(days=1)

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = today + timedelta(days=7)

        # ローカル race_lists だけで埋まる日が多いときは NetkeibaClient（ログイン含む）を作らない
        client = None
        all_races = []
        current = start

        while current <= end:
            date_compact = current.strftime("%Y%m%d")
            day_offset = (current - today_date).days

            races: list = []
            blob_meta = None
            list_path = RACE_LIST_DIR / f"{date_compact}.json"
            decided_from_file = False

            if list_path.exists():
                try:
                    with open(list_path, "r", encoding="utf-8") as rf:
                        blob = json.load(rf)
                    blob_meta = blob.get("_meta") if isinstance(blob, dict) else None
                    file_races = blob.get("races") or []
                    if len(file_races) == 0:
                        races = []
                        decided_from_file = True
                    elif is_plausible_race_day_races(file_races):
                        races = file_races
                        decided_from_file = True
                    else:
                        logger.warning(
                            "race_lists/%s.json は件数が不自然 (%d 件) のため無視し netkeiba から取得します",
                            date_compact,
                            len(file_races),
                        )
                except Exception as e:
                    logger.warning("race_lists 読込失敗 %s: %s", list_path, e)

            if not decided_from_file:
                if skip_remote:
                    races = []
                else:
                    if client is None:
                        client = NetkeibaClient(auto_login=True)
                    races = fetch_races_for_kaisai_date(client, date_compact, use_cache=True)

            if races:
                if not include_date_in_data_viewer_race_list(
                    date_compact, races, blob_meta,
                ):
                    races = []

            if races:
                for r0 in races:
                    r = dict(r0)
                    r["day_offset"] = day_offset
                    all_races.append(r)

            current += timedelta(days=1)

        return all_races

    try:
        races = await asyncio.to_thread(_fetch)
        payload = {
            "races": races,
            "start_date": start_date,
            "end_date": end_date,
            "local_only": skip_remote,
        }
        if default_window and not skip_remote:
            _UPCOMING_RACES_DEFAULT_CACHE["payload"] = payload
            _UPCOMING_RACES_DEFAULT_CACHE["ts"] = _time.time()
        return JSONResponse(payload)
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)




@app.post("/api/scrape-queue/add", response_class=JSONResponse)
async def add_to_scrape_queue(request: Request):
    """
    スクレイピングジョブをキューに追加

    Body: {
        "race_id": str,
        "date": str (optional),
        "venue": str (optional),
        "round": int (optional),
        "race_name": str (optional)
    }
    実行内容はレース一式（race_all）相当。細かいタスクは POST /api/scrape-queue/add-job を使用。
    """
    try:
        body = await request.json()
        race_id = body.get("race_id")

        if not race_id:
            return JSONResponse({"error": "race_id required"}, status_code=400)

        from src.scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        result = queue.add_job({
            "race_id": race_id,
            "date": body.get("date", ""),
            "venue": body.get("venue", ""),
            "round": body.get("round", 0),
            "race_name": body.get("race_name", ""),
        })

        _kick_scrape_queue_worker()

        return JSONResponse(result)

    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.get("/api/scrape-queue/status", response_class=JSONResponse)
async def get_scrape_queue_status():
    """スクレイピングキューの状態を取得"""
    try:
        from src.scraper.job_queue import ScrapeJobQueue

        _ensure_worker_slots()

        queue = ScrapeJobQueue()
        if not queue.is_locked():
            queue.requeue_stale_running_jobs(assume_lock_holder=False)

        status = queue.get_status()
        status["worker_health"] = get_worker_health()

        # 未リロードの古い job_queue がメモリに残っていると pending_queue / active_jobs が欠落する。
        # フロントが空表示になるのを防ぐため、欠けていれば生キューから必ず付与する。
        if "pending_queue" not in status or "active_jobs" not in status:
            jobs = queue.load_queue()
            if "pending_queue" not in status:
                runn = [j for j in jobs if j.get("status") == "running"]
                pend = [j for j in jobs if j.get("status") == "pending"]
                status["pending_queue"] = (runn + pend)[:200]
            if "active_jobs" not in status:
                status["active_jobs"] = [j for j in jobs if j.get("status") in ("pending", "running")]
        if "failed_jobs" not in status:
            jobs = queue.load_queue()
            failed = [j for j in jobs if j.get("status") == "failed"]
            failed.sort(
                key=lambda j: (j.get("completed_at") or "", j.get("job_id") or ""),
                reverse=True,
            )
            status["failed_jobs"] = failed[:200]

        from src.scraper.queue_load_settings import get_status_summary

        status["scrape_load_settings"] = get_status_summary()

        return JSONResponse(status)

    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.get("/api/scrape-queue/progress", response_class=JSONResponse)
async def api_scrape_queue_progress():
    """
    キュー実行中ジョブの細かい進捗（ワーカーが `queue_job_progress` に書いた内容）。
    別プロセスのワーカーでは `data/meta/_queue_job_progress.json` 経由。
    """
    try:
        from src.scraper.queue_job_progress import get_progress_snapshot_for_api

        return JSONResponse(get_progress_snapshot_for_api())
    except Exception as e:
        import traceback

        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/scrape-queue/worker-logs", response_class=JSONResponse)
async def api_scrape_queue_worker_logs(
    after: int = Query(-1, description="増分: 前回の max_id。初回は -1"),
    limit: int = Query(300, ge=1, le=800),
):
    """
    キュー実行スレッド中の scraper.* / queue.* / urllib3 等のログ行（メモリリング）。
    """
    try:
        from src.scraper.queue_worker_log import ensure_queue_worker_log_handler, get_worker_logs

        ensure_queue_worker_log_handler()
        return JSONResponse(get_worker_logs(after=after, limit=limit))
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/worker-logs/clear", response_class=JSONResponse)
async def api_scrape_queue_worker_logs_clear():
    try:
        from src.scraper.queue_worker_log import clear_worker_logs

        clear_worker_logs()
        return JSONResponse({"status": "ok"})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/resume", response_class=JSONResponse)
async def api_scrape_queue_resume():
    """
    アクセス系エラーで停止したキューを再開する。
    - 一時停止フラグを解除
    - failed 状態のジョブを全件 pending に戻す（400 ブロックで失敗扱いになったジョブを復元）
    - キューワーカーをキック
    """
    try:
        from src.scraper.scrape_access_pause import (
            clear_access_pause,
            read_access_pause,
            reset_block_400_consecutive,
        )
        from src.scraper.job_queue import ScrapeJobQueue, kick_process_queue_background

        clear_access_pause()
        reset_block_400_consecutive()

        queue = ScrapeJobQueue()
        requeued, err = queue.requeue_failed_jobs(all_failed=True)

        kick_process_queue_background()

        return JSONResponse({
            "status": "ok",
            "requeued_jobs": requeued,
            "requeue_error": err,
            "transport_pause": read_access_pause(),
        })
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/dismiss-auto-cleared-notice", response_class=JSONResponse)
async def api_scrape_queue_dismiss_auto_cleared_notice():
    """HTTP 400 連続による自動全消去の告知バナーのみ閉じる（一時停止は維持）。"""
    try:
        from src.scraper.scrape_access_pause import (
            dismiss_queue_auto_cleared_notice,
            read_access_pause,
        )

        dismiss_queue_auto_cleared_notice()
        return JSONResponse({"status": "ok", "transport_pause": read_access_pause()})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/recover", response_class=JSONResponse)
async def api_scrape_queue_recover():
    """キューが止まったときの緊急対応: running状態の孤児ジョブをpendingに戻し、ロックファイルを削除"""
    try:
        from src.scraper.job_queue import ScrapeJobQueue
        queue = ScrapeJobQueue()
        requeued = queue.requeue_stale_running_jobs(assume_lock_holder=False)
        lock_existed = queue.lock_file.exists()
        lock_removed = False
        if lock_existed and queue._lock_json_pid_dead_or_invalid():
            try:
                queue.lock_file.unlink()
                lock_removed = True
            except OSError:
                pass
        return JSONResponse({
            "status": "ok",
            "requeued_jobs": requeued,
            "lock_existed": lock_existed,
            "lock_removed": lock_removed,
            "message": f"{requeued}件のジョブを待機に戻し、キューを復旧しました"
        })
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.post("/api/scrape-queue/kick", response_class=JSONResponse)
async def api_scrape_queue_kick():
    """待機中のワーカーを即座に起動してキュー処理を開始する"""
    try:
        from src.scraper.job_queue import ScrapeJobQueue

        q = ScrapeJobQueue()
        status = q.get_status()
        pending = status["queue"]["pending"]
        if pending == 0:
            return JSONResponse({"status": "no_jobs", "message": "処理待ちのジョブがありません"})

        # スロットワーカーを起床させる
        _ensure_worker_slots()
        _kick_scrape_queue_worker()

        kicked_urgent = False
        if q.has_urgent_pending() and not q.is_locked_urgent():
            from src.scraper.job_queue import kick_urgent_process_queue_background
            kick_urgent_process_queue_background()
            kicked_urgent = True

        parts = ["スロットワーカー"]
        if kicked_urgent:
            parts.append("ファストレーン（最優先）")
        return JSONResponse({
            "status": "kicked",
            "pending_jobs": pending,
            "kicked_urgent": kicked_urgent,
            "message": f"{' + '.join(parts)} を起動しました（待機中ジョブ: {pending}件）",
        })
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


def _sync_verify_horse_coverage(body: dict) -> dict[str, Any]:
    """
    キュー待機（任意）のあと、指定期間・年代について馬・レース・開催日の
    キュー対象タスクに相当するストレージ有無を検査し、必要なら不足分をジョブ投入する。
    """
    from src.scraper.job_queue import ScrapeJobQueue
    from src.scraper.scrape_policy import coerce_bool
    from src.scraper.verify_horse_scrape_completeness import wait_until_queue_idle
    from src.scraper.verify_scrape_completeness import (
        enqueue_combined_missing as _enqueue_combined_missing,
        public_combined_payload,
        verify_combined_for_period,
        year_to_date_range,
    )

    out: dict[str, Any] = {}
    if bool(body.get("wait_for_queue_idle")):
        try:
            timeout = float(body.get("idle_timeout_sec") or 7200)
        except (TypeError, ValueError):
            timeout = 7200.0
        try:
            poll = float(body.get("idle_poll_sec") or 3)
        except (TypeError, ValueError):
            poll = 3.0
        q = ScrapeJobQueue()
        ok, msg, last = wait_until_queue_idle(
            q, timeout_sec=max(30.0, timeout), poll_sec=poll,
        )
        out["wait_result"] = msg
        out["last_queue_before_verify"] = last
        if not ok:
            out["ok"] = False
            return out

    start_date = body.get("start_date")
    end_date = body.get("end_date")
    y_val = body.get("year")
    year: int | None = None
    if y_val is not None and str(y_val).strip() != "":
        try:
            year = int(y_val)
        except (TypeError, ValueError):
            return {"ok": False, "error": "year must be an integer (e.g. 2024)"}
        try:
            start_date, end_date = year_to_date_range(year)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    if start_date and (len(str(start_date)) != 8 or not str(start_date).isdigit()):
        return {"ok": False, "error": "start_date must be YYYYMMDD"}
    if end_date and (len(str(end_date)) != 8 or not str(end_date).isdigit()):
        return {"ok": False, "error": "end_date must be YYYYMMDD"}
    if start_date and end_date and str(start_date) > str(end_date):
        return {"ok": False, "error": "start_date must be <= end_date"}

    jra_only = body.get("jra_only", True)
    if isinstance(jra_only, str):
        jra_only = jra_only.strip().lower() not in ("0", "false", "no", "")

    def _parse_task_list(k: str) -> list[str] | None:
        raw = body.get(k)
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(t).strip() for t in raw if str(t).strip()]

    include_horse = body.get("include_horse", True)
    if isinstance(include_horse, str):
        include_horse = include_horse.strip().lower() not in ("0", "false", "no", "")
    include_race = body.get("include_race", True)
    if isinstance(include_race, str):
        include_race = include_race.strip().lower() not in ("0", "false", "no", "")
    include_date = body.get("include_date", True)
    if isinstance(include_date, str):
        include_date = include_date.strip().lower() not in ("0", "false", "no", "")
    if not (bool(include_horse) or bool(include_race) or bool(include_date)):
        return {"ok": False, "error": "include_horse / include_race / include_date のいずれかをオンにしてください"}

    tasks_horse = _parse_task_list("tasks_horse")
    tasks_race = _parse_task_list("tasks_race")
    tasks_date = _parse_task_list("tasks_date")
    if tasks_horse is None and body.get("tasks") is not None:
        raw = body.get("tasks")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, (list, tuple)):
            tasks_horse = [str(t).strip() for t in raw if str(t).strip()]

    smode = str(body.get("satisfaction_mode") or "load_default").strip()
    if smode not in ("load_default", "gcs_strict", "mirror_for_selected"):
        return {
            "ok": False,
            "error": "satisfaction_mode must be one of: load_default, gcs_strict, mirror_for_selected",
        }
    raw_mir = body.get("local_mirror_categories")
    if raw_mir is None:
        lm_list: list[str] = []
    elif isinstance(raw_mir, str):
        lm_list = [x.strip() for x in raw_mir.split(",") if str(x).strip()]
    elif isinstance(raw_mir, (list, tuple)):
        lm_list = [str(x).strip() for x in raw_mir if str(x).strip()]
    else:
        lm_list = []

    storage = _get_storage()
    combined = verify_combined_for_period(
        storage,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
        jra_only=bool(jra_only),
        include_horse=bool(include_horse),
        include_race=bool(include_race),
        include_date=bool(include_date),
        tasks_horse=tasks_horse,
        user_race_tasks=tasks_race,
        user_date_tasks=tasks_date,
        year=year,
        satisfaction_mode=smode,
        local_mirror_categories=lm_list,
    )
    pub = public_combined_payload(combined)
    out.update(pub)
    out["ok"] = bool(combined.get("ok"))
    for k, sub in (("verify_horse", combined.get("verify_horse")),
                   ("verify_race", combined.get("verify_race")),
                   ("verify_date", combined.get("verify_date"))):
        if isinstance(sub, dict) and not sub.get("ok"):
            out["ok"] = False

    if not combined.get("ok"):
        for k in ("verify_horse", "verify_race", "verify_date"):
            sub = combined.get(k)
            if isinstance(sub, dict) and sub.get("error"):
                out["error"] = sub.get("error")
                break

    if not body.get("enqueue_missing"):
        if year is not None and "year" not in out:
            out["year"] = year
        return out

    try:
        plim = int(body.get("per_task_limit") or 50_000)
    except (TypeError, ValueError):
        plim = 50_000
    plim = max(1, min(100_000, plim))
    if "smart_skip" in body:
        ss = body.get("smart_skip")
        if isinstance(ss, str):
            smart_skip = ss.strip().lower() not in ("0", "false", "no")
        else:
            smart_skip = bool(ss)
    else:
        smart_skip = True
    overwrite = coerce_bool(body.get("overwrite"), default=False)

    queue = ScrapeJobQueue()
    enr = _enqueue_combined_missing(
        queue, storage, combined,
        per_task_limit=plim,
        smart_skip=smart_skip,
        overwrite=overwrite,
    )
    out["enqueue"] = enr
    if not enr.get("ok", True):
        out["ok"] = False
    if year is not None and "year" not in out:
        out["year"] = year
    return out


@app.post("/api/scrape-queue/verify-horse-coverage", response_class=JSONResponse)
async def api_scrape_queue_verify_horse_coverage(request: Request):
    """
    開発者向け: キューが空になるまで待ったうえで（任意）、
    指定期間または year（西暦）について、馬・レース・開催日のキュー対象タスク相当のストレージ有無を検査。
    不足分は `enqueue_missing: true` で各エンティティ向けにキュー投入（`overwrite` は既定 false）。

    Body: {
      "wait_for_queue_idle"?: true,
      "idle_timeout_sec"?: 7200,
      "year"?: 2024,  // 指定時は 1/1–12/31。start_date/end_date より優先
      "start_date"?, "end_date"?: YYYYMMDD
      "jra_only"?: true,
      "include_horse"?: true, "include_race"?: true, "include_date"?: true,
      "tasks_horse"?: [], "tasks_race"?: [], "tasks_date"?: [],
      "tasks"?: []  // 下位互換: 指定時は tasks_horse 相当
      "enqueue_missing"?: false, "per_task_limit"?: 50000,
      "smart_skip"?: true, "overwrite"?: false,
      "satisfaction_mode"?: "load_default" | "gcs_strict" | "mirror_for_selected",
      "local_mirror_categories"?: ["race_oikiri", ...]  // mirror_for_selected 時
    }
    """
    if not is_developer(request):
        return JSONResponse({"error": "開発者セッションが必要です"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        out = await asyncio.to_thread(_sync_verify_horse_coverage, body)
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )
    enq = (out or {}).get("enqueue") or {}
    tot = int(enq.get("total_enqueued_ops") or 0)
    if bool(body.get("enqueue_missing")) and tot > 0:
        _kick_scrape_queue_worker()
        _kick_urgent_worker()
    return JSONResponse(out or {})


@app.post("/api/scrape-queue/migrate-precheck", response_class=JSONResponse)
async def api_scrape_queue_migrate_precheck(request: Request):
    """
    開発者向け: 既存 ``pending`` のうち上書きなし・スキップあり（smart_skip）の行を
    統一 precheck 列へ戻す（馬・レ・開催日すべて）。
    Body: { \"dry_run\"?: true }（dry_run 時は件数のみ、JSON は更新しない）
    """
    if not is_developer(request):
        return JSONResponse({"error": "開発者セッションが必要です"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    dry = bool(body.get("dry_run"))
    try:
        from src.scraper.job_queue import ScrapeJobQueue

        out = ScrapeJobQueue().migrate_pending_to_storage_precheck(dry_run=dry)
        return JSONResponse(out)
    except Exception as e:
        import traceback
        return JSONResponse(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/scrape-queue/local-mirror-config", response_class=JSONResponse)
async def api_scrape_queue_local_mirror_config_get(request: Request):
    """
    開発者向け: GCS 保存成功時に `data/local/mirror/` へコピーするカテゴリ（追い切り等）の設定を取得。
    併せて、ミラー候補のストレージカテゴリ一覧を返す。
    """
    if not is_developer(request):
        return JSONResponse({"error": "開発者セッションが必要です"}, status_code=403)
    try:
        from src.scraper import local_mirror_config

        st = _get_storage()
        base = st._base_dir
        return JSONResponse(
            {
                "ok": True,
                "config": local_mirror_config.get_local_mirror_config(base),
                "mirrorable_categories": local_mirror_config.mirrorable_storage_categories(),
            }
        )
    except Exception as e:
        import traceback
        return JSONResponse(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/local-mirror-config", response_class=JSONResponse)
async def api_scrape_queue_local_mirror_config_post(request: Request):
    """開発者向け: 上記ミラー設定の保存。Body: { \"enabled\": bool, \"categories\": [\"race_oikiri\", ...] }"""
    if not is_developer(request):
        return JSONResponse({"error": "開発者セッションが必要です"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        from src.scraper import local_mirror_config

        st = _get_storage()
        base = st._base_dir
        en = bool(body.get("enabled"))
        raw = body.get("categories")
        if raw is None:
            cats: list[str] = []
        elif isinstance(raw, str):
            cats = [x.strip() for x in raw.split(",") if str(x).strip()]
        else:
            cats = [str(x).strip() for x in (raw or []) if str(x).strip()]
        saved = local_mirror_config.save_local_mirror_config(
            base, enabled=en, categories=cats
        )
        return JSONResponse({"ok": True, "config": saved})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/scrape-queue/load-settings", response_class=JSONResponse)
async def api_scrape_queue_load_settings_get():
    """ランタイム負荷（並列度・in-flight 等）の上書き設定を取得。"""
    try:
        from src.scraper.queue_load_settings import get_status_summary

        return JSONResponse(get_status_summary())
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, status_code=500,
        )


@app.post("/api/scrape-queue/load-settings", response_class=JSONResponse)
async def api_scrape_queue_load_settings_post(request: Request):
    """
    data/queue/queue_load_settings.json を更新。開発者セッション必須。
    Body: { "parallel"?: 1-32, "stagger_sec"?: number, "eta_sec_per_job"?: number,
            "netkeiba_max_inflight"?: 0-64(0=無制限, null/空=env),
            "note"?: str, "reset_all"?: true }
    """
    if not is_developer(request):
        return JSONResponse({"error": "開発者セッションが必要です"}, status_code=403)
    try:
        from src.scraper.client import reset_netkeiba_inflight_semaphore_cache
        from src.scraper.queue_load_settings import clear_overrides_file, validate_and_save
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON オブジェクトを送ってください"}, status=400)
    if body.get("reset_all"):
        out: dict = clear_overrides_file()  # type: ignore[assignment]
        if out.get("error"):
            return JSONResponse(out, status=400)
        try:
            reset_netkeiba_inflight_semaphore_cache()
        except Exception as _e:
            out["inflight_cache_warning"] = str(_e)
        return JSONResponse(out)
    r = validate_and_save(body)
    if r.get("error"):
        return JSONResponse(r, status=400)
    try:
        reset_netkeiba_inflight_semaphore_cache()
    except Exception as e:
        r = dict(r)
        r["inflight_cache_warning"] = str(e)
    return JSONResponse(r)


@app.get("/api/scrape-queue/tasks", response_class=JSONResponse)
async def scrape_queue_task_catalog():
    """キューに載せられるタスク一覧（UI・API用）。"""
    try:
        from src.scraper.queue_tasks import catalog_for_api

        return JSONResponse({"tasks": catalog_for_api()})
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.post("/api/scrape-queue/add-job", response_class=JSONResponse)
async def scrape_queue_add_generic_job(request: Request):
    """
    job_kind（race|horse|date）+ target_id + tasks[] で任意スクレイプをキュー投入。
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        from src.scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        try:
            result = queue.add_job(body)
        except ValueError as ve:
            return JSONResponse({"error": str(ve)}, status_code=400)

        _ensure_worker_slots()
        _kick_scrape_queue_worker()

        return JSONResponse({"status": "success", **result})
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.post("/api/scrape-queue/enqueue-incomplete-dates", response_class=JSONResponse)
async def scrape_queue_enqueue_incomplete_dates(request: Request):
    """
    モニター全日付一覧の「データ保有率」が 100% 未満の開催日を、新しい日付から順に
    job_kind=date / tasks=date_all でキューへ投入する。

    Body (任意): force_summary (bool|1), max_dates (default 500), smart_skip (default true), overwrite
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        result = await asyncio.to_thread(_sync_enqueue_incomplete_summary_dates, body)
        _kick_scrape_queue_worker()
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/clear", response_class=JSONResponse)
async def clear_scrape_queue():
    """完了・失敗したジョブをクリア"""
    try:
        from src.scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        queue.clear_completed()

        return JSONResponse({"status": "cleared"})

    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.post("/api/scrape-queue/hourly-maintenance-run", response_class=JSONResponse)
async def api_scrape_queue_hourly_maintenance_run():
    """
    手動で定期メンテと同じ処理（失敗→全件待機、完了レコード削除）を実行。
    バックグラウンドの定期実行と同じ `run_hourly_queue_maintenance`。
    """
    try:
        from src.scraper.job_queue import run_hourly_queue_maintenance

        out = await asyncio.to_thread(run_hourly_queue_maintenance)
        return JSONResponse(out)
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


def _sync_scrape_queue_stop_and_clear() -> dict[str, Any]:
    """
    実行中ワーカーが次ループでジョブを取らないよう一瞬一時停止を掛け、
    全ジョブ削除後に必ず一時停止を解除する（残すと get_next_job が常に None で固まる）。
    """
    from src.scraper.job_queue import ScrapeJobQueue
    from src.scraper.scrape_access_pause import read_access_pause

    queue = ScrapeJobQueue()
    removed = queue.clear_all_jobs_with_transport_pause(
        reason="ユーザー操作: キュー全消去処理中（完了後に自動解除されます）"
    )
    return {
        "status": "ok",
        "removed_jobs": removed,
        "transport_pause": read_access_pause(),
    }


@app.post("/api/scrape-queue/stop-and-clear", response_class=JSONResponse)
async def api_scrape_queue_stop_and_clear():
    """
    キューを止めてから空にする:
    - 短く一時停止を掛けたうえで全ジョブ削除し、最後に一時停止を解除する
    - 待機・実行中・完了・失敗のジョブをすべて削除
    実行中のジョブは 1 件分が完了するまで進む場合があります。
    """
    try:
        out = await asyncio.to_thread(_sync_scrape_queue_stop_and_clear)
        _kick_scrape_queue_worker()
        return JSONResponse(out)
    except Exception as e:
        import traceback

        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/failed/requeue", response_class=JSONResponse)
async def scrape_queue_failed_requeue(request: Request):
    """
    失敗ジョブを待機中に戻す。
    Body: { "job_id": "..." } | { "job_ids": ["..."] } | { "all": true }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    jid = str(body.get("job_id") or "").strip()
    jids = body.get("job_ids")
    if isinstance(jids, str):
        jids = [jids]
    job_ids = [str(x).strip() for x in (jids or []) if str(x).strip()]
    if jid:
        job_ids.append(jid)
    all_failed = bool(body.get("all"))
    try:
        from src.scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        n, err = queue.requeue_failed_jobs(job_ids=job_ids or None, all_failed=all_failed)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        _kick_scrape_queue_worker()
        return JSONResponse({"status": "ok", "requeued": n})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/scrape-queue/failed/remove", response_class=JSONResponse)
async def scrape_queue_failed_remove(request: Request):
    """
    失敗ジョブをキューから削除（再実行しない）。
    Body: { "job_id": "..." } | { "job_ids": ["..."] } | { "all": true }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    jid = str(body.get("job_id") or "").strip()
    jids = body.get("job_ids")
    if isinstance(jids, str):
        jids = [jids]
    job_ids = [str(x).strip() for x in (jids or []) if str(x).strip()]
    if jid:
        job_ids.append(jid)
    all_failed = bool(body.get("all"))
    try:
        from src.scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        n, err = queue.remove_failed_jobs(job_ids=job_ids or None, all_failed=all_failed)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        return JSONResponse({"status": "ok", "removed": n})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )




@app.post("/api/scrape-queue/add-batch", response_class=JSONResponse)
async def add_batch_to_scrape_queue(request: Request):
    """複数レースを一括でスクレイピングキューに追加"""
    try:
        body = await request.json()
        races = body.get("races", [])
        
        if not races:
            return JSONResponse({"error": "races required"}, status_code=400)
        
        from src.scraper.job_queue import ScrapeJobQueue
        
        queue = ScrapeJobQueue()
        created = 0
        requeued = 0
        duplicate = 0
        processed = 0

        from src.scraper.scrape_policy import coerce_bool

        batch_extras: dict[str, Any] = {}
        if "smart_skip" in body:
            ss = body.get("smart_skip")
            if isinstance(ss, str):
                batch_extras["smart_skip"] = ss.strip().lower() not in ("0", "false", "no")
            else:
                batch_extras["smart_skip"] = bool(ss)
        if "overwrite" in body:
            batch_extras["overwrite"] = coerce_bool(body.get("overwrite"), default=False)

        for race in races:
            race_id = race.get("race_id")
            if not race_id:
                continue
            processed += 1

            job_data = {
                "race_id": race_id,
                "date": race.get("date", ""),
                "venue": race.get("venue", ""),
                "round": race.get("round", 0),
                "race_name": race.get("race_name", ""),
                **batch_extras,
            }

            result = queue.add_job(job_data)
            act = result.get("action", "created")
            if act == "created":
                created += 1
            elif act == "requeued":
                requeued += 1
            else:
                duplicate += 1

        _kick_scrape_queue_worker()

        scheduled = created + requeued
        return JSONResponse({
            "status": "success",
            "added": scheduled,
            "new_jobs": created,
            "requeued": requeued,
            "already_in_queue": duplicate,
            "processed": processed,
            "total": len(races),
        })
    
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


def _sync_enqueue_missing_races(body: dict) -> dict:
    """
    race_lists 上の JRA レースで未取得のものをキューへ（未来レースに限らない）。
    body: start_date?, end_date? (YYYYMMDD), limit (default 200), dry_run (bool),
          smart_skip?, overwrite?
    """
    from src.scraper.job_queue import ScrapeJobQueue
    from src.scraper.missing_races import find_missing_jra_races_for_queue
    from src.scraper.scrape_policy import coerce_bool

    start_date = body.get("start_date") or None
    end_date = body.get("end_date") or None
    limit = int(body.get("limit") or 200)
    limit = max(1, min(limit, 2000))
    dry_run = bool(body.get("dry_run"))

    job_extras: dict[str, Any] = {}
    if "smart_skip" in body:
        ss = body.get("smart_skip")
        if isinstance(ss, str):
            job_extras["smart_skip"] = ss.strip().lower() not in ("0", "false", "no")
        else:
            job_extras["smart_skip"] = bool(ss)
    if "overwrite" in body:
        job_extras["overwrite"] = coerce_bool(body.get("overwrite"), default=False)

    if start_date and (len(str(start_date)) != 8 or not str(start_date).isdigit()):
        return {"error": "start_date must be YYYYMMDD"}
    if end_date and (len(str(end_date)) != 8 or not str(end_date).isdigit()):
        return {"error": "end_date must be YYYYMMDD"}
    if start_date and end_date and str(start_date) > str(end_date):
        return {"error": "start_date must be <= end_date"}

    storage = _get_storage()
    queue = ScrapeJobQueue()
    skip_ids: set[str] = set()
    for j in queue.load_queue():
        if j.get("status") in ("pending", "running"):
            rid = j.get("race_id")
            if rid:
                skip_ids.add(rid)

    races, meta = find_missing_jra_races_for_queue(
        storage,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
        limit=limit,
        skip_race_ids=skip_ids,
    )

    sample = races[:25]
    if dry_run:
        return {
            "status": "success",
            "dry_run": True,
            "candidate_count": len(races),
            "sample_races": sample,
            "meta": meta,
        }

    created = requeued = duplicate = 0
    for race in races:
        job_data = {
            "race_id": race["race_id"],
            "date": race.get("date", ""),
            "venue": race.get("venue", ""),
            "round": race.get("round", 0),
            "race_name": race.get("race_name", ""),
            **job_extras,
        }
        result = queue.add_job(job_data)
        act = result.get("action", "created")
        if act == "created":
            created += 1
        elif act == "requeued":
            requeued += 1
        else:
            duplicate += 1

    scheduled = created + requeued
    return {
        "status": "success",
        "dry_run": False,
        "candidate_count": len(races),
        "new_jobs": created,
        "requeued": requeued,
        "already_in_queue": duplicate,
        "added": scheduled,
        "sample_races": sample,
        "meta": meta,
    }


def _sync_enqueue_period_horse_tasks(body: dict) -> dict:
    """
    race_lists 上の期間内レースから出走馬を解決し、馬タスクのジョブをキューへ。
    body: start_date?, end_date? (YYYYMMDD), tasks (list[str]), limit, dry_run, jra_only?,
          smart_skip?, overwrite?（上書き再取得。未指定時は SCRAPE_DEFAULT_OVERWRITE）
    """
    from src.scraper.job_queue import ScrapeJobQueue
    from src.scraper.period_runners import enqueue_horse_tasks_for_race_period
    from src.scraper.scrape_policy import coerce_bool

    start_date = body.get("start_date") or None
    end_date = body.get("end_date") or None
    raw_tasks = body.get("tasks") or []
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]
    tasks = [str(t).strip() for t in raw_tasks if str(t).strip()]
    dry_run = bool(body.get("dry_run"))
    try:
        limit = int(body.get("limit") if body.get("limit") is not None else 500)
    except (TypeError, ValueError):
        limit = 500
    # limit=0: 期間内の全出走馬。正の値は 1〜50000（大量時は CLI 推奨）。
    if limit == 0:
        limit = 0
    else:
        limit = max(1, min(limit, 50000))

    jra_only = body.get("jra_only", True)
    if isinstance(jra_only, str):
        jra_only = jra_only.strip().lower() not in ("0", "false", "no", "")

    if start_date and (len(str(start_date)) != 8 or not str(start_date).isdigit()):
        return {"error": "start_date must be YYYYMMDD"}
    if end_date and (len(str(end_date)) != 8 or not str(end_date).isdigit()):
        return {"error": "end_date must be YYYYMMDD"}
    if start_date and end_date and str(start_date) > str(end_date):
        return {"error": "start_date must be <= end_date"}
    if not tasks:
        return {"error": "tasks に馬エンティティのタスクIDを1つ以上指定してください（例: horse_profile）"}

    horse_kw: dict[str, Any] = {}
    if "smart_skip" in body:
        ss = body.get("smart_skip")
        if isinstance(ss, str):
            horse_kw["smart_skip"] = ss.strip().lower() not in ("0", "false", "no")
        else:
            horse_kw["smart_skip"] = bool(ss)
    if "overwrite" in body:
        horse_kw["overwrite"] = coerce_bool(body.get("overwrite"), default=False)
    if "skip_local_mirror" in body:
        horse_kw["skip_local_mirror"] = coerce_bool(
            body.get("skip_local_mirror"), default=False
        )

    storage = _get_storage()
    queue = ScrapeJobQueue()
    try:
        result = enqueue_horse_tasks_for_race_period(
            storage,
            queue,
            start_date=str(start_date) if start_date else None,
            end_date=str(end_date) if end_date else None,
            tasks=tasks,
            limit=limit,
            dry_run=dry_run,
            jra_only=bool(jra_only),
            **horse_kw,
        )
    except ValueError as ve:
        return {"error": str(ve)}

    result["status"] = "success"
    return result


def _sync_enqueue_period_race_tasks(body: dict) -> dict:
    """
    race_lists 上の期間内 JRA レースに対し、レース単位タスクをキューへ。
    body: start_date?, end_date? (YYYYMMDD), tasks (list[str]), limit, dry_run, jra_only?,
          smart_skip?, overwrite?
    """
    from src.scraper.job_queue import ScrapeJobQueue
    from src.scraper.period_runners import enqueue_race_tasks_for_race_period
    from src.scraper.scrape_policy import coerce_bool

    start_date = body.get("start_date") or None
    end_date = body.get("end_date") or None
    raw_tasks = body.get("tasks") or []
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]
    tasks = [str(t).strip() for t in raw_tasks if str(t).strip()]
    dry_run = bool(body.get("dry_run"))
    try:
        limit = int(body.get("limit") or 500)
    except (TypeError, ValueError):
        limit = 500
    limit = max(1, min(limit, 10000))

    jra_only = body.get("jra_only", True)
    if isinstance(jra_only, str):
        jra_only = jra_only.strip().lower() not in ("0", "false", "no", "")

    if start_date and (len(str(start_date)) != 8 or not str(start_date).isdigit()):
        return {"error": "start_date must be YYYYMMDD"}
    if end_date and (len(str(end_date)) != 8 or not str(end_date).isdigit()):
        return {"error": "end_date must be YYYYMMDD"}
    if start_date and end_date and str(start_date) > str(end_date):
        return {"error": "start_date must be <= end_date"}
    if not tasks:
        return {"error": "tasks にレースエンティティのタスクIDを1つ以上指定してください（例: race_shutuba）"}

    race_kw: dict[str, Any] = {}
    if "smart_skip" in body:
        ss = body.get("smart_skip")
        if isinstance(ss, str):
            race_kw["smart_skip"] = ss.strip().lower() not in ("0", "false", "no")
        else:
            race_kw["smart_skip"] = bool(ss)
    if "overwrite" in body:
        race_kw["overwrite"] = coerce_bool(body.get("overwrite"), default=False)
    if "skip_local_mirror" in body:
        race_kw["skip_local_mirror"] = coerce_bool(
            body.get("skip_local_mirror"), default=False
        )
    if "priority" in body:
        try:
            race_kw["priority"] = int(body["priority"])
        except (TypeError, ValueError):
            pass

    storage = _get_storage()
    queue = ScrapeJobQueue()
    try:
        result = enqueue_race_tasks_for_race_period(
            storage,
            queue,
            start_date=str(start_date) if start_date else None,
            end_date=str(end_date) if end_date else None,
            tasks=tasks,
            limit=limit,
            dry_run=dry_run,
            jra_only=bool(jra_only),
            **race_kw,
        )
    except ValueError as ve:
        return {"error": str(ve)}

    result["status"] = "success"
    return result


def _partition_scrape_tasks(
    task_ids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    from src.scraper.queue_tasks import TASK_CATALOG

    m = {t["id"]: t["entity"] for t in TASK_CATALOG}
    h, r, d = [], [], []
    seen_h: set[str] = set()
    seen_r: set[str] = set()
    seen_dt: set[str] = set()
    for x in task_ids:
        s = str(x).strip()
        if not s:
            continue
        e = m.get(s)
        if e == "horse" and s not in seen_h:
            seen_h.add(s)
            h.append(s)
        elif e == "race" and s not in seen_r:
            seen_r.add(s)
            r.append(s)
        elif e == "date" and s not in seen_dt:
            seen_dt.add(s)
            d.append(s)
    return h, r, d


def _sync_enqueue_scrape_period(body: dict) -> dict[str, Any]:
    """
    期間 + タスク（馬 / レース / 開催日の混在可）を一括キュー投入。

    - 馬: 当該期間に出馬履歴のある馬（race_lists → shutuba/result から ID 解決）
    - レース: 当該期間の JRA 各レース（tasks に ``race_result`` / ``race_result_lap`` 等を指定可。``race_result_lap`` は結果ページのラップ系）
    - 開催日: 当該期間の race_lists キー

    Body:
      start_date?, end_date? (YYYYMMDD)
      tasks: list[str]  (TASK_CATALOG id)
      save_mode: "local_gcs" | "gcs_only"  — gcs_only 時 GCS 保存は行うが常設ローカルミラーは付けない
      skip_mode: "use_skip" | "overwrite"  — 上書き時は smart_skip 相当なし
      use_skip かつ少なくとも1件投入した場合、直後に統一 storage precheck（全タスク種。調教は一括キー照会併用）
      limit?, dry_run?, jra_only? (馬・レースのみ)
    """
    from src.scraper.job_queue import ScrapeJobQueue
    from src.scraper.period_runners import (
        enqueue_date_tasks_for_race_period,
        enqueue_horse_tasks_for_race_period,
        enqueue_race_tasks_for_race_period,
    )
    from src.scraper.queue_tasks import TASK_CATALOG

    raw = body.get("tasks") or []
    if isinstance(raw, str):
        raw = [raw]
    task_ids = [str(t).strip() for t in raw if str(t).strip()]
    if not task_ids:
        return {"ok": False, "error": "tasks を1つ以上指定してください"}
    valid = {t["id"] for t in TASK_CATALOG}
    bad = [x for x in task_ids if x not in valid]
    if bad:
        return {"ok": False, "error": f"未登録のタスクID: {bad}"}

    h_tasks, r_tasks, d_tasks = _partition_scrape_tasks(task_ids)
    if not (h_tasks or r_tasks or d_tasks):
        return {
            "ok": False,
            "error": "有効なタスクがありません（race_all・馬・開催日のいずれか）",
        }

    start_date = body.get("start_date") or None
    end_date = body.get("end_date") or None
    if start_date and (len(str(start_date)) != 8 or not str(start_date).isdigit()):
        return {"ok": False, "error": "start_date must be YYYYMMDD"}
    if end_date and (len(str(end_date)) != 8 or not str(end_date).isdigit()):
        return {"ok": False, "error": "end_date must be YYYYMMDD"}
    if start_date and end_date and str(start_date) > str(end_date):
        return {"ok": False, "error": "start_date must be <= end_date"}

    try:
        limit = int(body.get("limit") or 500_000)
    except (TypeError, ValueError):
        limit = 500_000
    limit = max(1, min(limit, 1_000_000))

    dry_run = bool(body.get("dry_run"))
    jra_only = body.get("jra_only", True)
    if isinstance(jra_only, str):
        jra_only = jra_only.strip().lower() not in ("0", "false", "no", "")

    sm = str(body.get("save_mode") or "local_gcs").strip().lower()
    skip_local_mirror = sm in ("gcs_only", "gcs", "gcsonly", "gcs only")

    sk = str(body.get("skip_mode") or "use_skip").strip().lower()
    use_overwrite = sk in (
        "overwrite",
        "no_skip",
        "false",
        "上書き",
    )
    if use_overwrite:
        job_skip = False
        job_ow = True
    else:
        job_skip = True
        job_ow = False

    run_kw: dict[str, Any] = {
        "smart_skip": job_skip,
        "overwrite": job_ow,
        "skip_local_mirror": skip_local_mirror,
    }

    storage = _get_storage()
    queue = ScrapeJobQueue()
    out: dict[str, Any] = {
        "ok": True,
        "partition": {
            "horse_tasks": h_tasks,
            "race_tasks": r_tasks,
            "date_tasks": d_tasks,
        },
        "save_mode": "gcs_only" if skip_local_mirror else "local_gcs",
        "skip_mode": "overwrite" if use_overwrite else "use_skip",
        "limit": limit,
    }
    tot_ops = 0
    if h_tasks:
        try:
            out["horse"] = enqueue_horse_tasks_for_race_period(
                storage,
                queue,
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None,
                tasks=h_tasks,
                limit=limit,
                dry_run=dry_run,
                jra_only=bool(jra_only),
                **run_kw,
            )
        except ValueError as ve:
            return {"ok": False, "error": str(ve)}
        hst = out["horse"]
        if not dry_run and hst:
            tot_ops += int(
                hst.get("created", 0) or 0
            ) + int(hst.get("requeued", 0) or 0)
    if r_tasks:
        try:
            out["race"] = enqueue_race_tasks_for_race_period(
                storage,
                queue,
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None,
                tasks=r_tasks,
                limit=limit,
                dry_run=dry_run,
                jra_only=bool(jra_only),
                **run_kw,
            )
        except ValueError as ve:
            return {"ok": False, "error": str(ve)}
        rst = out["race"]
        if not dry_run and rst:
            tot_ops += int(
                rst.get("created", 0) or 0
            ) + int(rst.get("requeued", 0) or 0)
    if d_tasks:
        try:
            out["date"] = enqueue_date_tasks_for_race_period(
                storage,
                queue,
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None,
                tasks=d_tasks,
                limit=limit,
                dry_run=dry_run,
                **run_kw,
            )
        except ValueError as ve:
            return {"ok": False, "error": str(ve)}
        dst = out["date"]
        if not dry_run and dst:
            tot_ops += int(
                dst.get("created", 0) or 0
            ) + int(dst.get("requeued", 0) or 0)

    if not dry_run and not use_overwrite and (h_tasks or r_tasks or d_tasks):
        out["storage_precheck"] = queue.run_storage_precheck_horse_now()
    if not dry_run and tot_ops > 0:
        _kick_scrape_queue_worker()
    return out


@app.post("/api/scrape-queue/enqueue-scrape-period", response_class=JSONResponse)
async def api_scrape_queue_enqueue_scrape_period(request: Request):
    """
    期間・保存先・スキップ方針をまとめて指定し、馬/レース/開催日タスクを一括投入する。
    詳細は ``_sync_enqueue_scrape_period`` 参照。
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        out = await asyncio.to_thread(_sync_enqueue_scrape_period, body)
    except Exception as e:
        import traceback

        return JSONResponse(
            {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )
    if not out.get("ok", True):
        return JSONResponse(out, status_code=400)
    return JSONResponse(out)


@app.api_route("/api/scrape-queue/enqueue-period-horses", methods=["GET", "POST"])
async def enqueue_period_horse_tasks(request: Request):
    """
    期間内に開催されたレース（race_lists）の出走馬に対し、馬単位タスクをキュー投入。
    出馬表または結果 JSON から horse_id を解決する（レースベース）。

    GET: dry_run のみ。tasks はカンマ区切り（例 ?tasks=horse_pedigree&start_date=20240101）
    POST: JSON { start_date?, end_date?, tasks: [], limit?, dry_run?, jra_only? }
    """
    if request.method == "GET":
        q = request.query_params
        dry_q = (q.get("dry_run") or "1").lower()
        if dry_q in ("0", "false", "no"):
            return JSONResponse(
                {
                    "error": "GET はプレビュー（dry_run）のみです。本投入は POST /api/scrape-queue/enqueue-period-horses に JSON で送ってください。",
                },
                status_code=405,
            )
        try:
            limit_q = int(q.get("limit") or 500)
        except ValueError:
            limit_q = 500
        tasks_raw = q.get("tasks") or "horse_pedigree"
        task_list = [t.strip() for t in tasks_raw.split(",") if t.strip()]
        body = {
            "dry_run": True,
            "limit": limit_q,
            "start_date": q.get("start_date") or None,
            "end_date": q.get("end_date") or None,
            "tasks": task_list,
            "jra_only": q.get("jra_only", "1"),
        }
        if str(q.get("overwrite") or "").strip().lower() in ("1", "true", "yes"):
            body["overwrite"] = True
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}

    try:
        result = await asyncio.to_thread(_sync_enqueue_period_horse_tasks, body)
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        if not result.get("dry_run"):
            _kick_scrape_queue_worker()
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


@app.api_route("/api/scrape-queue/enqueue-period-races", methods=["GET", "POST"])
async def enqueue_period_race_tasks(request: Request):
    """
    期間内に開催された JRA レース（race_lists）に対し、レース単位タスクをキュー投入。

    GET: dry_run のみ。tasks はカンマ区切り（例 ?tasks=race_shutuba&start_date=20240101）
    POST: JSON { start_date?, end_date?, tasks: [], limit?, dry_run?, jra_only? }
    """
    if request.method == "GET":
        q = request.query_params
        dry_q = (q.get("dry_run") or "1").lower()
        if dry_q in ("0", "false", "no"):
            return JSONResponse(
                {
                    "error": "GET はプレビュー（dry_run）のみです。本投入は POST /api/scrape-queue/enqueue-period-races に JSON で送ってください。",
                },
                status_code=405,
            )
        try:
            limit_q = int(q.get("limit") or 500)
        except ValueError:
            limit_q = 500
        tasks_raw = q.get("tasks") or "race_shutuba"
        task_list = [t.strip() for t in tasks_raw.split(",") if t.strip()]
        body = {
            "dry_run": True,
            "limit": limit_q,
            "start_date": q.get("start_date") or None,
            "end_date": q.get("end_date") or None,
            "tasks": task_list,
            "jra_only": q.get("jra_only", "1"),
        }
        if str(q.get("overwrite") or "").strip().lower() in ("1", "true", "yes"):
            body["overwrite"] = True
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}

    try:
        result = await asyncio.to_thread(_sync_enqueue_period_race_tasks, body)
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        if not result.get("dry_run"):
            _kick_scrape_queue_worker()
        return JSONResponse(result)
    except Exception as e:
        import traceback

        return JSONResponse(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


@app.api_route("/api/scrape-queue/enqueue-missing", methods=["GET", "POST"])
async def enqueue_missing_races(request: Request):
    """
    保存済み race_lists に載る JRA レースのうち、未取得（check-scraped と同基準）をキューに追加。
    GET は dry_run のみ（ブラウザ・リバプロ検証用）。本投入は POST。
    """
    if request.method == "GET":
        q = request.query_params
        dry_q = (q.get("dry_run") or "1").lower()
        if dry_q in ("0", "false", "no"):
            return JSONResponse(
                {"error": "GET はプレビュー（dry_run）のみです。本投入は POST /api/scrape-queue/enqueue-missing に JSON で送ってください。"},
                status_code=405,
            )
        try:
            limit_q = int(q.get("limit") or 200)
        except ValueError:
            limit_q = 200
        body = {
            "dry_run": True,
            "limit": limit_q,
            "start_date": q.get("start_date") or None,
            "end_date": q.get("end_date") or None,
        }
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}

    try:
        result = await asyncio.to_thread(_sync_enqueue_missing_races, body)
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        if not result.get("dry_run"):
            _kick_scrape_queue_worker()
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.post("/api/check-scraped-status", response_class=JSONResponse)
async def check_scraped_status(request: Request):
    """複数レースのスクレイピング済み状態を一括確認"""
    try:
        body = await request.json()
        race_ids = body.get("race_ids", [])
        
        if not race_ids:
            return JSONResponse({"error": "race_ids required"}, status_code=400)
        
        from src.scraper.missing_races import scrape_status_detail

        storage = _get_storage()
        result = {rid: scrape_status_detail(storage, rid) for rid in race_ids}
        
        return JSONResponse(result)
    
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


# ── 未来レース情報収集 ──

_calendar_fetch_job = {
    "running": False,
    "status": "idle",
    "progress": {},
    "result": None,
    "error": None,
}

@app.post("/api/fetch-future-calendar", response_class=JSONResponse)
async def fetch_future_calendar(background_tasks: BackgroundTasks, start_date: str = None, end_date: str = None):
    """未来レースのカレンダー情報を収集（バックグラウンド実行・進捗は /status で取得）

    Args:
        start_date: 開始日 (YYYY-MM-DD形式)
        end_date: 終了日 (YYYY-MM-DD形式)
    """
    global _calendar_fetch_job

    if _calendar_fetch_job.get("running"):
        return JSONResponse({
            "status": "already_running",
            "message": "既にカレンダー情報収集が実行中です。画面の進捗表示をご確認ください。",
        })

    _calendar_fetch_job = {
        "running": True,
        "status": "starting",
        "progress": {
            "total_days": 0,
            "completed_days": 0,
            "current_date": None,
            "message": "バックグラウンドでジョブを開始しています…",
        },
        "result": None,
        "error": None,
        "traceback": None,
        "started_at_jst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "range_label": f"{start_date or '（既定）'} 〜 {end_date or '（既定）'}",
    }

    async def _calendar_fetch_task():
        await asyncio.to_thread(_fetch_calendar_background, start_date, end_date)

    background_tasks.add_task(_calendar_fetch_task)

    return JSONResponse({
        "status": "started",
        "message": "カレンダー情報収集を開始しました。進捗は画面に表示されます。",
    })


@app.get("/api/fetch-future-calendar/status", response_class=JSONResponse)
async def get_calendar_fetch_status():
    """カレンダー情報収集の進捗状況を返す"""
    return JSONResponse(_calendar_fetch_job)


def _fetch_calendar_background(start_date: str = None, end_date: str = None):
    """バックグラウンドでカレンダー情報を収集"""
    global _calendar_fetch_job
    
    logger.info(f"=== _fetch_calendar_background開始: {start_date} ~ {end_date} ===")
    
    try:
        from datetime import datetime, timedelta
        from pathlib import Path
        import json
        import time
        from src.scraper.client import NetkeibaClient
        from src.scraper.netkeiba_top_race_list import (
            fetch_races_for_kaisai_date,
            invalidate_race_list_cache,
        )

        from src.config.data_paths import RACE_LISTS_DIR

        RACE_LIST_DIR = RACE_LISTS_DIR
        RACE_LIST_DIR.mkdir(parents=True, exist_ok=True)

        _calendar_fetch_job["status"] = "scraping_races"
        _calendar_fetch_job["progress"]["message"] = "取得期間を計算しています…"

        today = datetime.now()
        today_date = today.replace(hour=0, minute=0, second=0, microsecond=0)

        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = today + timedelta(days=1)

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = today + timedelta(days=7)

        day_list = []
        cur = start
        while cur <= end:
            day_list.append(cur)
            cur += timedelta(days=1)

        n_days = len(day_list)
        _calendar_fetch_job["progress"]["total_days"] = n_days
        _calendar_fetch_job["progress"]["completed_days"] = 0
        _calendar_fetch_job["progress"]["message"] = f"全 {n_days} 日分を netkeiba から取得します"

        client = NetkeibaClient(auto_login=True)
        results = []

        for idx, day in enumerate(day_list, start=1):
            date_display = day.strftime("%Y-%m-%d")
            date_str = day.strftime("%Y%m%d")
            day_offset = (day - today_date).days

            _calendar_fetch_job["progress"]["current_date"] = date_display
            _calendar_fetch_job["progress"]["message"] = (
                f"netkeiba 取得中: {date_display}（{idx}/{n_days} 日目）"
            )

            logger.info(
                "カレンダー収集: %s（netkeiba /top の race_list_get_date_list + race_list_sub）",
                date_display,
            )

            races = fetch_races_for_kaisai_date(client, date_str, use_cache=False)
            venues = sorted({r["venue"] for r in races})

            output_path = None
            if races:
                data = {
                    "date": date_str,
                    "races": races,
                    "_meta": {
                        "scraped_at": time.time(),
                        "scraped_at_jst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "netkeiba_top_race_list",
                        "kaisai_date": date_str,
                        "venues": venues,
                        "day_offset": day_offset,
                    },
                }
                output_path = RACE_LIST_DIR / f"{date_str}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            results.append({
                "date": date_display,
                "date_compact": date_str,
                "venues": venues,
                "race_count": len(races),
                "file": str(output_path) if output_path else None,
            })

            invalidate_race_list_cache(date_str)

            _calendar_fetch_job["progress"]["completed_days"] += 1
            time.sleep(0.5)

        _calendar_fetch_job["running"] = False
        _calendar_fetch_job["status"] = "completed"
        _calendar_fetch_job["progress"]["message"] = (
            f"完了: {sum(r['race_count'] for r in results)} レース（開催日のみ保存）"
        )
        _calendar_fetch_job["result"] = {
            "total_days": len(results),
            "total_races": sum(r["race_count"] for r in results),
            "details": results,
        }

        logger.info(
            "カレンダー収集完了: %d 日処理, %d レース（開催日のみファイル保存）",
            len(results),
            sum(r["race_count"] for r in results),
        )
    
    except Exception as e:
        import traceback
        _calendar_fetch_job["running"] = False
        _calendar_fetch_job["status"] = "error"
        _calendar_fetch_job["error"] = str(e)
        _calendar_fetch_job["traceback"] = traceback.format_exc()
        _calendar_fetch_job.setdefault("progress", {})["message"] = f"エラー: {e}"
        logger.error(f"カレンダー収集エラー: {e}", exc_info=True)

# ═══════════════════════════════════════════════════════
# レース詳細ページ
# ═══════════════════════════════════════════════════════

@app.get("/race/{race_id}", response_class=HTMLResponse)
async def race_detail_page(request: Request, race_id: str):
    """レース詳細ページを表示する。"""
    return templates.TemplateResponse("race/race_detail.html", {
        "request": request,
        "race_id": race_id,
        "current_page": "race_detail",
        "breadcrumbs": [
            {"url": "/", "label": "Home"},
            {"url": "/monitor", "label": "モニター"},
            {"url": "", "label": f"レース {race_id}"},
        ],
    })


_race_detail_cache: dict[str, tuple[float, dict]] = {}
_race_detail_cache_lock = threading.Lock()
_RACE_DETAIL_TTL = 60  # seconds


@app.get("/api/race/{race_id}", response_class=JSONResponse)
async def get_race_detail(race_id: str):
    """レースの全データをまとめて返す (60s キャッシュ)。"""
    with _race_detail_cache_lock:
        cached = _race_detail_cache.get(race_id)
    if cached and (_time.time() - cached[0]) < _RACE_DETAIL_TTL:
        return JSONResponse(cached[1])

    def _load():
        storage = _get_storage()
        result: dict = {"race_id": race_id}
        with ThreadPoolExecutor(max_workers=len(MONITOR_SOURCES)) as pool:
            futures = {cat: pool.submit(storage.load, cat, race_id)
                       for cat in MONITOR_SOURCES}
            for cat, f in futures.items():
                try:
                    result[cat] = f.result()
                except Exception:
                    result[cat] = None
        card = result.get("race_shutuba")
        if card and isinstance(card, dict):
            result["race_name"] = card.get("race_name", "")
            result["venue"] = card.get("venue", "")
            result["round"] = card.get("round", 0)
            result["date"] = card.get("date", "")
            result["surface"] = card.get("surface", "")
            result["distance"] = card.get("distance", 0)
            result["direction"] = card.get("direction", "")
            result["weather"] = card.get("weather", "")
            result["track_condition"] = card.get("track_condition", "")
            result["grade"] = card.get("grade", "")
            result["start_time"] = card.get("start_time", "")
            horse_ids = [e.get("horse_id", "")
                         for e in card.get("entries", []) if e.get("horse_id")]
            with ThreadPoolExecutor(max_workers=16) as pool:
                hf = {hid: pool.submit(storage.load, "horse_result", hid)
                      for hid in horse_ids}
                horses = {}
                for hid, f in hf.items():
                    try:
                        hdata = f.result()
                        if hdata:
                            horses[hid] = hdata
                    except Exception:
                        pass
            result["horses"] = horses
        else:
            race_result_data = result.get("race_result")
            if race_result_data and isinstance(race_result_data, dict):
                result["race_name"] = race_result_data.get("race_name", "")
                result["venue"] = race_result_data.get("venue", "")
                result["round"] = race_result_data.get("round", 0)
                result["date"] = race_result_data.get("date", "")
                result["surface"] = race_result_data.get("surface", "")
                result["distance"] = race_result_data.get("distance", 0)
            result["horses"] = {}
        try:
            result["race_result_lap"] = storage.load("race_result_lap", race_id)
        except Exception:
            result["race_result_lap"] = None

        from src.utils.race_result_display import prepare_race_result_display

        prepared = prepare_race_result_display(
            result.get("race_result"),
            result.get("race_result_on_time"),
            result.get("race_result_lap"),
        )
        if prepared:
            result["race_result"] = prepared

        data_availability = {}
        for cat in MONITOR_SOURCES:
            data_availability[cat] = {
                "json": result.get(cat) is not None,
                "html": False,
            }
        data_availability["race_result_lap"] = {
            "json": result.get("race_result_lap") is not None,
            "html": False,
        }
        result["data_availability"] = data_availability
        return result

    result = await asyncio.to_thread(_load)
    with _race_detail_cache_lock:
        _race_detail_cache[race_id] = (_time.time(), result)
    return JSONResponse(result)


# ═══════════════════════════════════════════════════════
# AI 予測 API
# ═══════════════════════════════════════════════════════

_predict_lock = threading.Lock()


@app.get("/api/race/{race_id}/predictions", response_class=JSONResponse)
async def get_race_predictions(race_id: str):
    """キャッシュ済みの予測結果を返す。なければ 404。"""
    data = await asyncio.to_thread(
        lambda: _get_storage().load("race_predictions", race_id))
    if data:
        return JSONResponse(data)
    return JSONResponse({"status": "not_found"}, status_code=404)


@app.post("/api/race/{race_id}/predict", response_class=JSONResponse)
async def run_race_prediction(request: Request, race_id: str):
    """
    指定レースの AI 予測を実行し、結果を GCS に保存して返す。

    フロー:
      1. scrape_race_all で全データ取得 (smart_skip=True)
      2. feature_builder で特徴量テーブル構築
      3. モデル予測 (MLflow) or フォールバック (ヒューリスティック)
      4. 推奨印 (◎○▲△☆) 付与
      5. GCS に race_predictions/{race_id}.json として保存
    """
    if not is_developer(request):
        return JSONResponse({"error": "管理者権限が必要です"}, status_code=403)
    def _run():
        import time as _t
        import traceback
        from src.pipeline.features.feature_builder import build_race_features

        storage = _get_storage()

        t_start = _t.time()

        runner = _get_runner()
        race_data = runner.scrape_race_all(race_id, smart_skip=True)

        card = race_data.get("race_card") or {}
        result_data = race_data.get("race_result") or {}

        race_info = {
            "race_id": race_id,
            "race_name": card.get("race_name", "") or result_data.get("race_name", ""),
            "venue": card.get("venue", "") or result_data.get("venue", ""),
            "round": card.get("round", 0),
            "surface": card.get("surface", ""),
            "distance": card.get("distance", 0),
            "track_condition": card.get("track_condition", ""),
        }

        # 特徴量構築
        features_df = build_race_features(race_data)
        if features_df.empty:
            return {
                **race_info,
                "status": "error",
                "error": "特徴量テーブルが空 — 出馬表データが不足しています",
                "predictions": [],
            }

        # モデル予測
        model_type = "fallback_heuristic"
        try:
            from src.pipeline.inference.race_day import RaceDayPipeline, PipelineConfig
            pipe = RaceDayPipeline(PipelineConfig(
                model_name="keiba-lgbm-nopi",
                model_stage="latest",
                mlflow_tracking_uri="http://localhost:5000",
            ))
            model = pipe._load_model()
            if model is not None:
                try:
                    X = pipe._prepare_model_input(features_df)
                    scores = model.predict(X)
                    model_type = "mlflow_model"
                except Exception as pred_err:
                    logger.warning("MLflow モデル予測失敗: %s", pred_err)
                    scores = pipe._fallback_score(features_df)
            else:
                scores = pipe._fallback_score(features_df)
        except Exception:
            from src.pipeline.inference.race_day import RaceDayPipeline
            pipe = RaceDayPipeline()
            scores = pipe._fallback_score(features_df)

        meta_cols = ["race_id", "horse_number", "horse_name", "horse_id"]
        meta = features_df[[c for c in meta_cols if c in features_df.columns]].copy()
        meta["pred_score"] = scores
        meta = meta.sort_values("pred_score", ascending=False).reset_index(drop=True)
        meta["pred_rank"] = range(1, len(meta) + 1)

        max_score = meta["pred_score"].max()
        min_score = meta["pred_score"].min()
        score_range = max_score - min_score if max_score != min_score else 1

        # softmax 正規化で確率的解釈を可能にする
        import numpy as np
        raw = meta["pred_score"].values.astype(float)
        centered = raw - raw.mean()
        exp_scores = np.exp(centered / max(centered.std(), 1e-6))
        softmax_probs = exp_scores / exp_scores.sum()
        meta["softmax_prob"] = softmax_probs

        # ── オッズ予測 ──
        # 現在のライブオッズ（あれば推移トラッカーに記録）
        live_odds: dict[int, dict] = {}
        try:
            odds_data = race_data.get("race_odds") or {}
            for e_odds in odds_data.get("entries", []):
                hn = e_odds.get("horse_number", 0)
                if hn:
                    live_odds[hn] = e_odds
        except Exception:
            pass
        if not live_odds:
            try:
                odds_raw = storage.load("race_odds", race_id)
                if odds_raw:
                    for e_odds in odds_raw.get("entries", []):
                        hn = e_odds.get("horse_number", 0)
                        if hn:
                            live_odds[hn] = e_odds
            except Exception:
                pass

        # 予測オッズを取得（推移 > モデル > ヒューリスティック の優先順）
        from src.pipeline.models.odds_predictor import get_predicted_odds
        odds_map = get_predicted_odds(
            features_df, race_id,
            live_odds=live_odds if live_odds else None,
        )

        base_entries = []
        for _, row in meta.iterrows():
            base_entries.append({
                "pred_rank": int(row["pred_rank"]),
                "horse_number": int(row.get("horse_number", 0)),
                "horse_name": row.get("horse_name", ""),
                "horse_id": row.get("horse_id", ""),
                "pred_score": round(float(row["pred_score"]), 2),
                "normalized_score": round(float(row["softmax_prob"]), 4),
            })

        entries, bet_suggestion = _compute_composite_scores(base_entries, odds_map)
        # ソースを判定
        sources = set(v.get("source", "") for v in odds_map.values())
        odds_source = "/".join(sorted(sources)) if sources else "none"
        mark_method = f"composite ({odds_source})"

        feature_highlights = _extract_feature_highlights(features_df, entries)

        elapsed = round(_t.time() - t_start, 2)

        comp_params = _load_composite_params()
        from src.pipeline.inference.composite_optimizer import OPTIM_RESULT_PATH
        is_optimized = OPTIM_RESULT_PATH.exists()

        prediction_result = {
            **race_info,
            "status": "success",
            "model_type": model_type,
            "model_description": "LightGBM (大衆指標排除)" if model_type == "mlflow_model"
                                 else "ヒューリスティック (タイム指数35%+直近成績25%+上がり3F20%+戦績10%+偏差値10%)",
            "mark_method": mark_method,
            "composite_params": {
                "prob_weight": comp_params.get("prob_weight"),
                "source": "simulation" if is_optimized else "default",
            },
            "total_horses": len(entries),
            "elapsed_sec": elapsed,
            "predictions": entries,
            "bet_suggestion": bet_suggestion,
            "feature_highlights": feature_highlights,
        }

        storage.save("race_predictions", race_id, prediction_result)
        return prediction_result

    try:
        with _predict_lock:
            result = _run()
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


def _tracking_difficulty_public_payload(data: dict) -> dict:
    """クライアント向けに内部メタキーを除去。"""
    return {k: v for k, v in data.items() if not k.startswith("_")}


@app.get("/api/race/{race_id}/result-status", response_class=JSONResponse)
async def api_race_result_status(race_id: str):
    """レース結果（確定・速報）の有無と結果ページ URL。"""
    def _run():
        from src.utils.race_result_availability import race_result_status

        return race_result_status(_get_storage(), race_id)

    return JSONResponse(await asyncio.to_thread(_run))


@app.get("/api/race/{race_id}/tracking-difficulty", response_class=JSONResponse)
async def api_tracking_difficulty(race_id: str, refresh: bool = False):
    """追走難度・ペース・位置取り（storage キャッシュ優先、refresh=true で再計算）。"""
    def _run():
        from src.pipeline.inference.tracking_difficulty_service import get_or_compute

        return get_or_compute(
            _get_storage(),
            race_id,
            force_refresh=refresh,
            allow_scrape=refresh,
        )

    try:
        result = await asyncio.to_thread(_run)
        return JSONResponse(_tracking_difficulty_public_payload(result))
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.post("/api/race/{race_id}/tracking-difficulty/precompute", response_class=JSONResponse)
async def api_precompute_tracking_difficulty(race_id: str):
    """追走難度をバッチ計算して storage に保存（推論ワーカー相当）。"""
    def _run():
        from src.pipeline.inference.tracking_difficulty_service import (
            build_tracking_difficulty_response,
            save_cached_response,
        )

        storage = _get_storage()
        payload = build_tracking_difficulty_response(race_id, storage, allow_scrape=False)
        if payload.get("entries"):
            save_cached_response(storage, race_id, payload, source="precompute_api")
        return payload

    try:
        result = await asyncio.to_thread(_run)
        out = _tracking_difficulty_public_payload(result)
        meta = result.get("_compute_meta") or {}
        out["precompute"] = True
        out["compute_meta"] = meta
        return JSONResponse(out)
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.get("/api/inference/health", response_class=JSONResponse)
async def api_inference_health():
    """MLflow Tracking / 全モデル Serving・ローカル Booster・キャッシュの疎通確認。"""
    from src.pipeline.mlflow.runtime import platform_health
    from src.pipeline.inference.tracking_difficulty_service import cache_enabled

    def _run():
        report = platform_health(force_serve_check=True)
        # 後方互換フィールド（追走難度）
        td = next(
            (m for m in report["models"] if m["key"] == "tracking_difficulty"),
            {},
        )
        tr = report["mlflow_tracking"]
        report["mlflow_tracking_uri"] = tr.get("uri")
        report["mlflow_tracking_ok"] = tr.get("ok")
        report["mlflow_tracking_ms"] = tr.get("ms")
        report["mlflow_tracking_error"] = tr.get("error")
        report["mlflow_serve_uri"] = td.get("serve_uri")
        report["mlflow_serve_ok"] = td.get("serve_ok")
        report["mlflow_serve_ms"] = td.get("serve_ms")
        report["tracking_difficulty_cache"] = cache_enabled()
        return report

    return JSONResponse(await asyncio.to_thread(_run))


@app.post("/api/tracking-difficulty/train", response_class=JSONResponse)
async def api_train_tracking_difficulty(request: Request):
    """追走難度モデルの学習を実行する。"""
    def _train():
        from src.pipeline.models.tracking_difficulty import TrackingDifficultyTrainer
        trainer = TrackingDifficultyTrainer(mlflow_tracking_uri="mlflow/runs")
        return trainer.train(storage=_get_storage())

    try:
        result = await asyncio.to_thread(_train)
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


def _rank_recommendation(rank: int, total: int) -> str:
    """ランク順に推奨印を割り当てる（旧ロジック、フォールバック用）。"""
    if rank == 1:
        return "◎ 本命"
    if rank == 2:
        return "○ 対抗"
    if rank == 3:
        return "▲ 単穴"
    if rank <= max(5, total // 3):
        return "△ 連下"
    return "☆ 穴馬"


# ─── 期待値×確率トレードオフ印付けロジック ─────────────────

import math as _math

MARKS = ["◎ 本命", "○ 対抗", "▲ 単穴", "△ 連下", "☆ 穴馬"]

# デフォルトパラメータ（シミュレーション結果があればそちらを優先）
_DEFAULT_PROB_WEIGHT = 0.55
_DEFAULT_MIN_PROB_HONMEI_RATIO = 0.50
_DEFAULT_MIN_PROB_TAIKOU_RATIO = 0.30


def _load_composite_params() -> dict:
    """シミュレーション最適化結果があればロード、なければデフォルト。"""
    try:
        from src.pipeline.inference.composite_optimizer import get_composite_params
        return get_composite_params()
    except Exception:
        return {
            "prob_weight": _DEFAULT_PROB_WEIGHT,
            "min_prob_honmei_ratio": _DEFAULT_MIN_PROB_HONMEI_RATIO,
            "min_prob_taikou_ratio": _DEFAULT_MIN_PROB_TAIKOU_RATIO,
        }


def _compute_composite_scores(
    entries: list[dict],
    odds_map: dict[int, dict],
) -> list[dict]:
    """
    勝率・連対率・複勝率（Harville）と各期待値、MECE 印を付与する。

    composite = top3_prob^α × EV_place^(1-α)（買い目ソート用）
    """
    from src.utils.race_probabilities import (
        assign_mece_marks,
        buy_recommendation_tier,
        derive_race_probabilities,
        estimate_top2_payout_odds,
    )

    params = _load_composite_params()
    alpha = params.get("prob_weight", _DEFAULT_PROB_WEIGHT)

    scores = [float(e.get("pred_score") or 0) for e in entries]
    prob_list = derive_race_probabilities(scores)

    scored: list[dict] = []

    for e, probs in zip(entries, prob_list):
        hn = e["horse_number"]
        win_prob = probs["win_prob"]
        top2_prob = probs["top2_prob"]
        top3_prob = probs["top3_prob"]
        odds_info = odds_map.get(hn, {})

        place_min = odds_info.get("predicted_place_odds_min") or odds_info.get("place_odds_min", 0) or 0
        place_max = odds_info.get("predicted_place_odds_max") or odds_info.get("place_odds_max", 0) or 0
        win_odds = odds_info.get("predicted_win_odds") or odds_info.get("win_odds", 0) or 0
        odds_confidence = odds_info.get("confidence", 1.0)
        odds_source = odds_info.get("source", "live")
        place_avg = (place_min + place_max) / 2 if (place_min and place_max) else 0
        top2_odds = estimate_top2_payout_odds(win_odds) if win_odds else None

        ev_win = round(win_prob * win_odds, 3) if win_odds and win_prob > 0 else None
        ev_top2 = (
            round(top2_prob * top2_odds, 3) if top2_odds and top2_prob > 0 else None
        )
        ev_place = round(top3_prob * place_avg, 3) if place_avg and top3_prob > 0 else None

        if top3_prob > 0 and ev_place:
            composite = (top3_prob ** alpha) * (ev_place ** (1 - alpha))
        elif top3_prob > 0:
            composite = top3_prob
        else:
            composite = 0

        scored.append({
            **e,
            "win_prob": win_prob,
            "top2_prob": top2_prob,
            "top3_prob": top3_prob,
            "place_odds_min": round(place_min, 1) if place_min else None,
            "place_odds_max": round(place_max, 1) if place_max else None,
            "place_odds_avg": round(place_avg, 1) if place_avg else None,
            "win_odds": round(win_odds, 1) if win_odds else None,
            "top2_odds_est": top2_odds,
            "odds_confidence": round(odds_confidence, 2),
            "odds_source": odds_source,
            "estimated_prob": top3_prob,
            "ev_win": ev_win,
            "ev_top2": ev_top2,
            "ev_place": ev_place,
            "expected_value": ev_place,
            "composite_score": round(composite, 4),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, s in enumerate(scored):
        s["composite_rank"] = i + 1

    assign_mece_marks(scored)
    from src.utils.race_bet_suggestion import suggest_race_bets

    bet_suggestion = suggest_race_bets(scored)
    _MARK_LABELS = {
        "honmei": "◎ 1着優位",
        "pair": "○ 2連相手",
        "anchor": "✓ 3列紐",
        "show_val": "▲ 複勝妙味",
        "star": "★ 中穴妙味",
        "none": "",
    }
    for s in scored:
        mt = s.get("mark_type") or "none"
        s["recommendation"] = _MARK_LABELS.get(mt, "")
        s["buy_tier"] = buy_recommendation_tier(s)

    return scored, bet_suggestion


def _extract_feature_highlights(features_df, entries: list[dict]) -> list[dict]:
    """各馬の予測に寄与した主要特徴量を抽出する。"""
    highlights = []
    key_features = [
        ("speed_max", "最高タイム指数", True),
        ("speed_avg", "平均タイム指数", True),
        ("avg_finish_5", "直近5走平均着順", False),
        ("min_finish_5", "直近5走最高着順", False),
        ("top3_count_5", "直近5走3着内回数", True),
        ("avg_last_3f_5", "直近5走平均上がり3F", False),
        ("career_win_rate", "通算勝率", True),
        ("career_top3_rate", "通算複勝率", True),
        ("same_surface_win_rate", "同馬場勝率", True),
        ("same_dist_win_rate", "同距離勝率", True),
        ("training_impression_score", "調教評価", True),
        ("days_since_last", "前走からの日数", False),
    ]

    for entry in entries:
        hn = entry.get("horse_number", 0)
        row = features_df[features_df["horse_number"] == hn]
        if row.empty:
            highlights.append({"horse_number": hn, "factors": []})
            continue

        row = row.iloc[0]
        factors = []
        for col, label, higher_is_better in key_features:
            if col in features_df.columns:
                val = row[col]
                if val and val != 0:
                    factors.append({
                        "name": label,
                        "value": round(float(val), 2) if isinstance(val, (int, float)) else str(val),
                        "higher_is_better": higher_is_better,
                    })
        highlights.append({"horse_number": hn, "factors": factors})

    return highlights


# ═══════════════════════════════════════════════════════
# 構造チェック API
# ═══════════════════════════════════════════════════════

@app.get("/api/structure-status", response_class=JSONResponse)
async def get_structure_status():
    """最新の構造チェック結果を返す。"""
    from pathlib import Path
    check_path = Path(BASE_DIR) / "data" / "local" / "meta" / "structure" / "last_check.json"
    if not check_path.exists():
        return JSONResponse({
            "status": "no_data",
            "message": "構造チェックがまだ実行されていません",
        })
    try:
        data = json.loads(check_path.read_text(encoding="utf-8"))
        return JSONResponse(data)
    except Exception:
        return JSONResponse({"status": "error", "error": traceback.format_exc()}, status_code=500)


@app.get("/api/structure-report", response_class=JSONResponse)
async def get_structure_report():
    """構造チェックレポート (Markdown) を返す。"""
    from pathlib import Path
    report_path = Path(BASE_DIR) / "data" / "local" / "meta" / "structure" / "report.md"
    if not report_path.exists():
        return JSONResponse({"status": "no_data", "report": ""})
    return JSONResponse({
        "status": "ok",
        "report": report_path.read_text(encoding="utf-8"),
    })


@app.post("/api/structure-check", response_class=JSONResponse)
async def trigger_structure_check():
    """構造チェックを手動でトリガーする (バックグラウンド実行)。"""
    job_id = f"structure_check:{int(_time.time())}"

    with _scrape_jobs_lock:
        running = [j for j in _scrape_jobs.values()
                   if j["status"] == "running" and j["category"] == "structure_check"]
    if running:
        return JSONResponse({
            "status": "already_running",
            "job_id": running[0]["job_id"],
        })

    with _scrape_jobs_lock:
        _scrape_jobs[job_id] = {
            "job_id": job_id,
            "race_id": "",
            "category": "structure_check",
            "status": "queued",
            "started_at": _time.time(),
            "finished_at": None,
            "error": None,
        }

    def _run():
        with _scrape_jobs_lock:
            _scrape_jobs[job_id]["status"] = "running"
        try:
            from src.scraper.structure_monitor import run_daily_check
            result = run_daily_check(auto_reparse=True, notify=True)
            with _scrape_jobs_lock:
                _scrape_jobs[job_id]["result"] = result
                _scrape_jobs[job_id]["status"] = "done"
        except Exception as e:
            with _scrape_jobs_lock:
                _scrape_jobs[job_id]["status"] = "error"
                _scrape_jobs[job_id]["error"] = str(e)
        with _scrape_jobs_lock:
            _scrape_jobs[job_id]["finished_at"] = _time.time()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return JSONResponse({
        "status": "started",
        "job_id": job_id,
    })


@app.get("/api/structure-check/schedule", response_class=JSONResponse)
async def get_structure_check_schedule():
    """構造チェックの自動スケジュール状態を返す。"""
    from pathlib import Path
    last_check_path = Path(BASE_DIR) / "data" / "local" / "meta" / "structure" / "last_check.json"
    last_check_file = None
    if last_check_path.exists():
        try:
            last_check_file = json.loads(last_check_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    with _scheduler_lock:
        state = dict(_scheduler_state)

    return JSONResponse({
        "schedule": {
            "hour_jst": STRUCTURE_CHECK_HOUR_JST,
            "minute_jst": STRUCTURE_CHECK_MINUTE_JST,
            "description": f"毎朝 {STRUCTURE_CHECK_HOUR_JST:02d}:{STRUCTURE_CHECK_MINUTE_JST:02d} JST",
        },
        "scheduler": state,
        "last_check_file": last_check_file,
    })


@app.get("/api/admin/cron-jobs", response_class=JSONResponse)
async def get_cron_jobs():
    """定期実行ジョブ一覧とその状態を返す。"""
    import os as _os
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    _JST = _tz(_td(hours=9))

    disk_interval = float(_os.environ.get("DISK_CACHE_CLEANUP_INTERVAL_SEC", "86400"))
    queue_interval = float(_os.environ.get("SCRAPE_QUEUE_HOURLY_MAINTENANCE_SEC", "3600"))
    logs_h = int(_os.environ.get("LOGS_RETENTION_HOUR_JST", "12"))
    logs_m = int(_os.environ.get("LOGS_RETENTION_MINUTE_JST", "0"))
    logs_days = int(_os.environ.get("LOGS_RETENTION_DAYS", "7"))
    shutuba_h = int(_os.environ.get("DAILY_SHUTUBA_HOUR_JST", "7"))
    shutuba_m = int(_os.environ.get("DAILY_SHUTUBA_MINUTE_JST", "0"))
    shutuba_days = int(_os.environ.get("DAILY_SHUTUBA_DAYS_AHEAD", "14"))

    with _scheduler_lock:
        structure_state = dict(_scheduler_state)
    with _disk_cache_cleanup_lock:
        disk_state = dict(_disk_cache_cleanup_state)
    with _queue_maintain_lock:
        queue_state = dict(_queue_maintain_state)
    with _logs_retention_lock:
        logs_state = dict(_logs_retention_state)
    with _daily_shutuba_lock:
        shutuba_state = dict(_daily_shutuba_state)
    # マルチワーカー対応: ファイルの状態を優先して上書き（他プロセスが書いた可能性）
    _file_shutuba = _read_cron_state("daily_shutuba")
    if _file_shutuba:
        # last_run はより新しい方を採用
        _mem_lr = shutuba_state.get("last_run") or ""
        _fil_lr = _file_shutuba.get("last_run") or ""
        if _fil_lr >= _mem_lr and _fil_lr:
            shutuba_state["last_run"] = _fil_lr
            if _file_shutuba.get("last_result") is not None:
                shutuba_state["last_result"] = _file_shutuba["last_result"]
            if _file_shutuba.get("run_count") is not None:
                shutuba_state["run_count"] = _file_shutuba["run_count"]

    from src.scraper.job_queue import read_queue_hourly_maintain_state
    queue_file_state = read_queue_hourly_maintain_state()
    if queue_file_state.get("available") and not queue_state.get("last_run"):
        queue_state["last_run"] = queue_file_state.get("written_at")
        queue_state["last_result"] = queue_file_state

    jobs = [
        {
            "id": "structure_check",
            "name": "構造チェック",
            "description": "HTML構造の異常検知・自動再パース",
            "schedule": f"毎朝 {STRUCTURE_CHECK_HOUR_JST:02d}:{STRUCTURE_CHECK_MINUTE_JST:02d} JST",
            "interval_sec": None,
            "trigger_endpoint": "/api/structure-check",
            **structure_state,
        },
        {
            "id": "disk_cache_cleanup",
            "name": "ディスクキャッシュクリーンアップ",
            "description": f"data/cache の古いファイルを削除（max_age={int(disk_interval//3600)}h）",
            "schedule": f"{int(disk_interval//3600)}時間ごと",
            "interval_sec": int(disk_interval),
            "trigger_endpoint": "/api/admin/cron-jobs/disk-cache-cleanup/trigger",
            **disk_state,
        },
        {
            "id": "queue_maintain",
            "name": "キュー定期メンテ",
            "description": "失敗ジョブを待機に戻し、完了レコードを削除",
            "schedule": f"{int(queue_interval//60)}分ごと",
            "interval_sec": int(queue_interval),
            "trigger_endpoint": "/api/admin/cron-jobs/queue-maintain/trigger",
            **queue_state,
        },
        {
            "id": "logs_retention",
            "name": "ログ保持期限切れ削除",
            "description": f"logs/*.log のうち {logs_days}日超を削除",
            "schedule": f"毎日 {logs_h:02d}:{logs_m:02d} JST",
            "interval_sec": None,
            "trigger_endpoint": "/api/admin/cron-jobs/logs-retention/trigger",
            **logs_state,
        },
        {
            "id": "daily_shutuba",
            "name": "出馬表 毎日自動取得",
            "description": f"今日〜+{shutuba_days}日の race_lists を走査し race_shutuba をキュー投入",
            "schedule": f"毎日 {shutuba_h:02d}:{shutuba_m:02d} JST",
            "interval_sec": None,
            "trigger_endpoint": "/api/admin/cron-jobs/daily-shutuba/trigger",
            **shutuba_state,
        },
    ]
    return JSONResponse({"jobs": jobs})


@app.post("/api/admin/cron-jobs/disk-cache-cleanup/trigger", response_class=JSONResponse)
async def trigger_disk_cache_cleanup():
    """ディスクキャッシュクリーンアップを即時実行（バックグラウンド）。"""
    with _disk_cache_cleanup_lock:
        if _disk_cache_cleanup_state.get("running"):
            return JSONResponse({"status": "already_running"})
        _disk_cache_cleanup_state["running"] = True

    def _run():
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _JST = _tz(_td(hours=9))
        try:
            st = _get_storage()
            r = st.cleanup_disk_cache()
            sn = st.cleanup_snapshot_files(max_age_seconds=86400)
            with _disk_cache_cleanup_lock:
                _disk_cache_cleanup_state["last_run"] = _dt.now(_JST).isoformat()
                _disk_cache_cleanup_state["last_result"] = {**r, "snapshot_removed": sn}
                _disk_cache_cleanup_state["run_count"] += 1
        except Exception as e:
            with _disk_cache_cleanup_lock:
                _disk_cache_cleanup_state["last_run"] = _dt.now(_JST).isoformat()
                _disk_cache_cleanup_state["last_result"] = {"error": str(e)}
        finally:
            with _disk_cache_cleanup_lock:
                _disk_cache_cleanup_state["running"] = False

    threading.Thread(target=_run, daemon=True, name="disk-cache-cleanup-manual").start()
    return JSONResponse({"status": "started"})


@app.post("/api/admin/cron-jobs/queue-maintain/trigger", response_class=JSONResponse)
async def trigger_queue_maintain():
    """キュー定期メンテを即時実行（バックグラウンド）。"""
    with _queue_maintain_lock:
        if _queue_maintain_state.get("running"):
            return JSONResponse({"status": "already_running"})
        _queue_maintain_state["running"] = True

    def _run():
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _JST = _tz(_td(hours=9))
        try:
            from src.scraper.job_queue import run_hourly_queue_maintenance
            r = run_hourly_queue_maintenance()
            with _queue_maintain_lock:
                _queue_maintain_state["last_run"] = _dt.now(_JST).isoformat()
                _queue_maintain_state["last_result"] = r
                _queue_maintain_state["run_count"] += 1
        except Exception as e:
            with _queue_maintain_lock:
                _queue_maintain_state["last_run"] = _dt.now(_JST).isoformat()
                _queue_maintain_state["last_result"] = {"error": str(e)}
        finally:
            with _queue_maintain_lock:
                _queue_maintain_state["running"] = False

    threading.Thread(target=_run, daemon=True, name="queue-maintain-manual").start()
    return JSONResponse({"status": "started"})


@app.post("/api/admin/cron-jobs/logs-retention/trigger", response_class=JSONResponse)
async def trigger_logs_retention():
    """ログ保持期限切れ削除を即時実行（バックグラウンド）。"""
    with _logs_retention_lock:
        if _logs_retention_state.get("running"):
            return JSONResponse({"status": "already_running"})
        _logs_retention_state["running"] = True

    def _run():
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _JST = _tz(_td(hours=9))
        try:
            from src.utils.log_retention import run_log_retention_once
            r = run_log_retention_once(BASE_DIR)
            with _logs_retention_lock:
                _logs_retention_state["last_run"] = _dt.now(_JST).isoformat()
                _logs_retention_state["last_result"] = r
                _logs_retention_state["run_count"] += 1
        except Exception as e:
            with _logs_retention_lock:
                _logs_retention_state["last_run"] = _dt.now(_JST).isoformat()
                _logs_retention_state["last_result"] = {"error": str(e)}
        finally:
            with _logs_retention_lock:
                _logs_retention_state["running"] = False

    threading.Thread(target=_run, daemon=True, name="logs-retention-manual").start()
    return JSONResponse({"status": "started"})


@app.post("/api/admin/cron-jobs/daily-shutuba/trigger", response_class=JSONResponse)
async def trigger_daily_shutuba():
    """出馬表 毎日自動取得を即時実行（バックグラウンド）。"""
    import os as _os
    from datetime import date as _date, datetime as _dt, timedelta as _td, timezone as _tz
    with _daily_shutuba_lock:
        if _daily_shutuba_state.get("running"):
            return JSONResponse({"status": "already_running"})
        _daily_shutuba_state["running"] = True

    def _run():
        from src.scraper.job_queue import PRIORITY_URGENT_PEDIGREE_5GEN
        _JST = _tz(_td(hours=9))
        try:
            days_ahead = int(_os.environ.get("DAILY_SHUTUBA_DAYS_AHEAD", "14"))
            today = _date.today()
            end = today + _td(days=days_ahead)
            body = {
                "start_date": today.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "tasks": ["race_shutuba"],
                "smart_skip": True,
                "dry_run": False,
                "limit": 500,
                "jra_only": True,
                "priority": PRIORITY_URGENT_PEDIGREE_5GEN,
            }
            result = _sync_enqueue_period_race_tasks(body)
            if result.get("created", 0):
                _kick_scrape_queue_worker()
            with _daily_shutuba_lock:
                _daily_shutuba_state["last_run"] = _dt.now(_JST).isoformat()
                _daily_shutuba_state["last_result"] = result
                _daily_shutuba_state["run_count"] += 1
                _snap = dict(_daily_shutuba_state)
            _write_cron_state("daily_shutuba", {
                "last_run": _snap["last_run"],
                "last_result": _snap["last_result"],
                "run_count": _snap["run_count"],
            })
        except Exception as e:
            with _daily_shutuba_lock:
                _daily_shutuba_state["last_run"] = _dt.now(_JST).isoformat()
                _daily_shutuba_state["last_result"] = {"error": str(e)}
            _write_cron_state("daily_shutuba", {
                "last_run": _daily_shutuba_state["last_run"],
                "last_result": _daily_shutuba_state["last_result"],
            })
        finally:
            with _daily_shutuba_lock:
                _daily_shutuba_state["running"] = False

    threading.Thread(target=_run, daemon=True, name="daily-shutuba-manual").start()
    return JSONResponse({"status": "started"})


# ---------------------------------------------------------------------------
# Auto-scrape external cron jobs
# ---------------------------------------------------------------------------

_AUTO_SCRAPE_TASKS: dict[str, dict] = {
    "daily-race-lists": {
        "name": "デイリーレース一覧",
        "description": "今後14日分の開催日ごとに番組表を取得・更新",
        "schedule": "毎日 07:00 JST",
        "tags": ["race_lists", "開催番組表", "14日先まで"],
    },
    "catchup-missing": {
        "name": "欠損補完",
        "description": "カレンダーに存在するが race_lists が未取得の過去開催日を補完",
        "schedule": "毎日 09:00 JST",
        "tags": ["race_lists", "過去欠損分"],
    },
    "raceday-runner": {
        "name": "開催日常駐",
        "description": "各レース発走15分前に出馬表・直前オッズ・SmartRC等を逐次取得（開催日のみ起動）",
        "schedule": "土日 07:30 JST",
        "tags": ["race_card", "出馬表", "馬番・斤量・騎手", "odds", "直前オッズ", "smartrc_race", "SmartRC指数"],
    },
    "raceday-result-runner": {
        "name": "速報結果常駐",
        "description": "各レース発走15分後に速報結果を逐次取得する開催日常駐プロセス",
        "schedule": "土日 07:30 JST",
        "tags": ["result_on_time", "速報結果"],
    },
    "raceday-eve": {
        "name": "前日準備",
        "description": "翌日が開催日の場合に出馬表・馬柱・追い切り・SmartRCを18:00に取得し追走難度を事前計算",
        "schedule": "毎日 18:00 JST",
        "tags": [
            "race_shutuba",
            "出馬表",
            "shutuba_past",
            "馬柱(近走成績)",
            "oikiri",
            "追い切り・調教",
            "smartrc_race",
            "SmartRC指数",
            "tracking_difficulty",
        ],
    },
    "raceday-evening": {
        "name": "夕方結果取得",
        "description": "当日の全レース終了後に速報結果・確定オッズ・SmartRC指数を取得",
        "schedule": "土日 18:00 JST",
        "tags": ["result_on_time", "速報結果", "odds", "確定オッズ", "pair_odds", "確定2連複/馬連", "SmartRC指数"],
    },
    "weekly-update": {
        "name": "週次更新",
        "description": "先週の全開催日について公式DBから確定版レース結果・オッズを再取得",
        "schedule": "金曜 17:30 JST",
        "tags": ["race_result", "確定レース結果", "odds", "確定オッズ", "pair_odds", "確定2連複/馬連", "SmartRC指数"],
    },
    "jra-baba-morning": {
        "name": "JRA馬場情報",
        "description": "JRA公式馬場ページからクッション値・含水率・馬場状態を朝取得（変更検知方式）",
        "schedule": "毎10分 05:00-09:00 JST",
        "tags": ["jra_baba", "クッション値", "含水率", "馬場状態"],
    },
}

_AUTO_SCRAPE_STATUS_FILE = os.path.join(BASE_DIR, "data", "meta", "auto_scrape_status.json")
_AUTO_SCRAPE_PYTHON = "/home/hirokiakataoka/miniconda3/bin/python3"


def _auto_scrape_is_running(task: str) -> bool:
    import subprocess as _sp
    try:
        result = _sp.run(
            ["pgrep", "-f", f"auto_scrape.*--task.*{task}"],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _load_auto_scrape_status() -> dict:
    try:
        with open(_AUTO_SCRAPE_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/api/admin/system-stats", response_class=JSONResponse)
async def get_system_stats():
    """VPS のリソース使用状況（CPU・メモリ・ディスク・プロセス・ネットワーク）を返す。"""
    import psutil, time, shutil
    from pathlib import Path

    def _b2mb(b): return round(b / 1024 / 1024, 1)
    def _b2gb(b): return round(b / 1024 / 1024 / 1024, 2)

    # ── CPU ──
    cpu_pct  = psutil.cpu_percent(interval=0.5)
    cpu_count_logical  = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    load1, load5, load15 = psutil.getloadavg()

    # ── メモリ ──
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    mem = {
        "total_mb":     _b2mb(vm.total),
        "used_mb":      _b2mb(vm.used),
        "available_mb": _b2mb(vm.available),
        "free_mb":      _b2mb(vm.free),
        "percent":      vm.percent,
        "buffers_mb":   _b2mb(getattr(vm, "buffers", 0)),
        "cached_mb":    _b2mb(getattr(vm, "cached", 0)),
        "swap_total_mb": _b2mb(sw.total),
        "swap_used_mb":  _b2mb(sw.used),
        "swap_percent":  sw.percent,
    }

    # ── ディスク ──
    partitions = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
            partitions.append({
                "device":     part.device,
                "mountpoint": part.mountpoint,
                "fstype":     part.fstype,
                "total_gb":   _b2gb(usage.total),
                "used_gb":    _b2gb(usage.used),
                "free_gb":    _b2gb(usage.free),
                "percent":    usage.percent,
            })
        except PermissionError:
            pass

    # アプリのデータディレクトリ個別集計
    data_dirs = {}
    app_root = Path(BASE_DIR)
    for name, rel in [
        ("data/local",          "data/local"),
        ("data/cache",          "data/cache"),
        ("data/queue",          "data/queue"),
        ("data/page_reference", "data/page_reference"),
        ("data/features",       "data/features"),
    ]:
        p = app_root / rel
        if p.exists():
            try:
                total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                data_dirs[name] = _b2mb(total)
            except Exception:
                data_dirs[name] = None

    # ── プロセス ──
    top_procs = []
    for proc in sorted(psutil.process_iter(["pid","name","cmdline","cpu_percent","memory_info","status"]),
                       key=lambda p: p.info.get("memory_info") and p.info["memory_info"].rss or 0,
                       reverse=True)[:8]:
        try:
            cmd = " ".join((proc.info["cmdline"] or []))[:80] or proc.info["name"]
            top_procs.append({
                "pid":    proc.info["pid"],
                "name":   proc.info["name"],
                "cmd":    cmd,
                "cpu_pct": proc.cpu_percent(),
                "mem_mb": _b2mb(proc.info["memory_info"].rss if proc.info["memory_info"] else 0),
                "status": proc.info["status"],
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # ── ネットワーク ──
    net = psutil.net_io_counters()
    net_stats = {
        "bytes_sent_mb":  _b2mb(net.bytes_sent),
        "bytes_recv_mb":  _b2mb(net.bytes_recv),
        "packets_sent":   net.packets_sent,
        "packets_recv":   net.packets_recv,
        "errin":          net.errin,
        "errout":         net.errout,
    }

    # ── アップタイム ──
    boot_time = psutil.boot_time()
    uptime_sec = int(time.time() - boot_time)
    uptime_str = f"{uptime_sec//3600}h {(uptime_sec%3600)//60}m"

    # ── キュー scrape_queue.json ──
    queue_file = Path(BASE_DIR) / "data" / "queue" / "scrape_queue.json"
    queue_file_mb = _b2mb(queue_file.stat().st_size) if queue_file.exists() else 0

    return JSONResponse({
        "timestamp": time.time(),
        "uptime_str": uptime_str,
        "cpu": {
            "percent": cpu_pct,
            "count_logical": cpu_count_logical,
            "count_physical": cpu_count_physical,
            "load_avg": {"1m": round(load1,2), "5m": round(load5,2), "15m": round(load15,2)},
        },
        "memory": mem,
        "disk": {
            "partitions": partitions,
            "app_dirs_mb": data_dirs,
            "queue_file_mb": queue_file_mb,
        },
        "processes": top_procs,
        "network": net_stats,
    })


@app.get("/api/admin/auto-scrape-status", response_class=JSONResponse)
async def get_auto_scrape_status():
    """外部 cron で実行される auto_scrape ジョブの一覧と状態を返す。"""
    status_data = _load_auto_scrape_status()
    jobs = []
    for task_id, meta in _AUTO_SCRAPE_TASKS.items():
        task_status = status_data.get(task_id, {})
        last_run = task_status.get("last_run")
        last_result = {k: v for k, v in task_status.items() if k != "last_run"} if task_status else None
        if last_result == {}:
            last_result = None
        jobs.append({
            "id": task_id,
            "name": meta["name"],
            "description": meta["description"],
            "schedule": meta["schedule"],
            "tags": meta.get("tags", []),
            "running": _auto_scrape_is_running(task_id),
            "last_run": last_run,
            "next_run": None,
            "last_result": last_result,
            "run_count": 0,
            "trigger_endpoint": f"/api/admin/auto-scrape/{task_id}/trigger",
        })
    return JSONResponse({"jobs": jobs})


@app.post("/api/admin/auto-scrape/{task}/trigger", response_class=JSONResponse)
async def trigger_auto_scrape(task: str):
    """auto_scrape タスクを即時起動する（非同期サブプロセス）。"""
    import subprocess as _sp
    if task not in _AUTO_SCRAPE_TASKS:
        return JSONResponse({"status": "error", "message": f"Unknown task: {task}"}, status_code=400)
    if _auto_scrape_is_running(task):
        return JSONResponse({"status": "already_running"})
    _sp.Popen(
        [_AUTO_SCRAPE_PYTHON, "-m", "src.scraper.auto_scrape", "--task", task],
        cwd=BASE_DIR,
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )
    return JSONResponse({"status": "started"})


@app.get("/api/structure-fingerprints", response_class=JSONResponse)
async def get_structure_fingerprints():
    """保存済みの全カテゴリのフィンガープリントを返す。"""
    from pathlib import Path
    fp_dir = Path(BASE_DIR) / "data" / "local" / "meta" / "structure"
    result = {}
    if fp_dir.exists():
        for fp_file in fp_dir.glob("*.json"):
            if fp_file.stem in ("last_check", "report"):
                continue
            try:
                result[fp_file.stem] = json.loads(fp_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    return JSONResponse(result)


def _recommendation(prob: float) -> str:
    if prob >= 0.7:
        return "◎ 本命"
    elif prob >= 0.55:
        return "○ 対抗"
    elif prob >= 0.4:
        return "▲ 単穴"
    elif prob >= 0.25:
        return "△ 連下"
    return "☆ 穴馬"


# ── モデル学習 ─────────────────────────────────────


_training_job: dict[str, Any] = {"running": False}


@app.post("/api/train", response_class=JSONResponse)
async def trigger_training(background_tasks: BackgroundTasks):
    """
    モデル学習をバックグラウンドで実行する。
    大衆指標排除ポリシーに基づく LightGBM モデルを MLflow に登録。
    """
    if _training_job.get("running"):
        return JSONResponse({
            "status": "already_running",
            "started_at": _training_job.get("started_at", ""),
        })

    _training_job["running"] = True
    _training_job["started_at"] = datetime.now().isoformat()
    _training_job["result"] = None
    _training_job["error"] = None

    background_tasks.add_task(_run_training)

    return JSONResponse({
        "status": "started",
        "started_at": _training_job["started_at"],
    })


def _run_training():
    try:
        from src.pipeline.models.trainer import ModelTrainer
        import yaml
        _cfg = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")) as f:
                _cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
        trainer = ModelTrainer(mlflow_tracking_uri="http://localhost:5000", config=_cfg)
        result = trainer.train()
        _training_job["result"] = result
        _training_job["error"] = result.get("error")
    except Exception as e:
        _training_job["error"] = str(e)
        logger.error("学習ジョブ失敗: %s", e)
    finally:
        _training_job["running"] = False
        _training_job["finished_at"] = datetime.now().isoformat()


@app.get("/api/train/status", response_class=JSONResponse)
async def get_training_status():
    """学習ジョブの現在の状態を返す。"""
    return JSONResponse({
        "running": _training_job.get("running", False),
        "started_at": _training_job.get("started_at", ""),
        "finished_at": _training_job.get("finished_at", ""),
        "result": _training_job.get("result"),
        "error": _training_job.get("error"),
    })


# ── アンサンブル学習 ──

_ensemble_job: dict[str, Any] = {"running": False}


@app.post("/api/train/ensemble", response_class=JSONResponse)
async def trigger_ensemble_training(background_tasks: BackgroundTasks):
    """
    アンサンブル学習 (LightGBM + XGBoost + CatBoost + NN) をバックグラウンドで実行する。
    """
    if _ensemble_job.get("running"):
        return JSONResponse({
            "status": "already_running",
            "started_at": _ensemble_job.get("started_at", ""),
        })

    _ensemble_job["running"] = True
    _ensemble_job["started_at"] = datetime.now().isoformat()
    _ensemble_job["result"] = None
    _ensemble_job["error"] = None

    background_tasks.add_task(_run_ensemble_training)

    return JSONResponse({
        "status": "started",
        "started_at": _ensemble_job["started_at"],
    })


def _run_ensemble_training():
    try:
        from src.pipeline.models.ensemble_trainer import EnsembleTrainer
        import yaml
        _cfg = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")) as f:
                _cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
        trainer = EnsembleTrainer(
            mlflow_tracking_uri="http://localhost:5000", config=_cfg,
        )
        result = trainer.train()
        _ensemble_job["result"] = _serialize_metrics(result)
        _ensemble_job["error"] = result.get("error")
    except Exception as e:
        _ensemble_job["error"] = str(e)
        logger.error("アンサンブル学習ジョブ失敗: %s", e, exc_info=True)
    finally:
        _ensemble_job["running"] = False
        _ensemble_job["finished_at"] = datetime.now().isoformat()


def _serialize_metrics(result: dict) -> dict:
    """numpy 型を JSON シリアライズ可能な型に変換する。"""
    import numpy as np
    out = {}
    for k, v in result.items():
        if isinstance(v, (np.integer, np.int64)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialize_metrics(v)
        elif isinstance(v, list):
            out[k] = [
                _serialize_metrics(i) if isinstance(i, dict)
                else int(i) if isinstance(i, (np.integer,))
                else float(i) if isinstance(i, (np.floating,))
                else i
                for i in v
            ]
        else:
            out[k] = v
    return out


@app.get("/api/train/ensemble/status", response_class=JSONResponse)
async def get_ensemble_training_status():
    """アンサンブル学習ジョブの現在の状態を返す。"""
    return JSONResponse({
        "running": _ensemble_job.get("running", False),
        "started_at": _ensemble_job.get("started_at", ""),
        "finished_at": _ensemble_job.get("finished_at", ""),
        "result": _ensemble_job.get("result"),
        "error": _ensemble_job.get("error"),
    })


@app.get("/api/model/info", response_class=JSONResponse)
async def get_model_info():
    """MLflow に登録されているモデル情報を返す。"""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.MlflowClient()

        from src.pipeline.models.trainer import ModelTrainer
        model_name = ModelTrainer.MODEL_NAME

        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception:
            versions = []

        if not versions:
            return JSONResponse({"registered": False, "model_name": model_name})

        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

        run = client.get_run(latest.run_id)
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags

        return JSONResponse({
            "registered": True,
            "model_name": model_name,
            "latest_version": int(latest.version),
            "run_id": latest.run_id,
            "created_at": str(latest.creation_timestamp),
            "status": latest.status,
            "metrics": metrics,
            "params": params,
            "tags": {k: v for k, v in tags.items() if not k.startswith("mlflow.")},
            "public_indicators_excluded": tags.get("public_indicators") == "excluded",
        })
    except Exception as e:
        return JSONResponse({"registered": False, "error": str(e)})


# ── オッズ予測 API ──────────────────────────────────────

_odds_training_job: dict = {}


@app.post("/api/odds/train", response_class=JSONResponse)
async def train_odds_model(background_tasks: BackgroundTasks):
    """オッズ予測モデルの学習をバックグラウンドで開始する。"""
    if _odds_training_job.get("running"):
        return JSONResponse({"status": "already_running"})

    _odds_training_job["running"] = True
    _odds_training_job["started_at"] = datetime.now().isoformat()
    _odds_training_job["result"] = None
    _odds_training_job["error"] = None
    _odds_training_job["finished_at"] = None

    background_tasks.add_task(_run_odds_training)
    return JSONResponse({"status": "started"})


def _run_odds_training():
    try:
        from src.pipeline.models.final_odds_progress import load_status_file
        from src.pipeline.models.final_odds_trainer import FinalOddsTrainer

        trainer = FinalOddsTrainer()
        result = trainer.train(_get_storage())
        _odds_training_job["result"] = result
        _odds_training_job["error"] = result.get("error")
        _odds_training_job["progress"] = load_status_file()
    except Exception as e:
        _odds_training_job["error"] = str(e)
        try:
            from src.pipeline.models.final_odds_progress import load_status_file

            _odds_training_job["progress"] = load_status_file()
        except Exception:
            pass
    finally:
        _odds_training_job["running"] = False
        _odds_training_job["finished_at"] = datetime.now().isoformat()


@app.get("/api/odds/train/status", response_class=JSONResponse)
async def get_odds_training_status():
    """オッズ予測モデルの学習状態を返す。"""
    progress = {}
    try:
        from src.pipeline.models.final_odds_progress import load_status_file

        progress = load_status_file()
    except Exception:
        pass
    running = _odds_training_job.get("running", False) or progress.get("running", False)
    return JSONResponse({
        "running": running,
        "started_at": _odds_training_job.get("started_at", ""),
        "finished_at": _odds_training_job.get("finished_at", ""),
        "result": _odds_training_job.get("result"),
        "error": _odds_training_job.get("error") or progress.get("error"),
        "progress": progress,
    })


@app.post("/api/odds/snapshot/{race_id}", response_class=JSONResponse)
async def record_odds_snapshot(request: Request, race_id: str):
    """指定レースの現在オッズを取得し、推移履歴に記録する。"""
    if not is_developer(request):
        return JSONResponse({"error": "管理者権限が必要です"}, status_code=403)
    scraper = _get_runner()
    odds = scraper.scrape_odds(race_id, skip_existing=False)
    if not odds or not odds.get("entries"):
        return JSONResponse({"error": "オッズ取得失敗"}, status_code=404)

    from src.pipeline.models.odds_predictor import OddsTrajectoryTracker
    tracker = OddsTrajectoryTracker()
    live_map = {}
    for e in odds.get("entries", []):
        hn = e.get("horse_number", 0)
        if hn:
            live_map[hn] = e
    tracker.record_snapshot(race_id, odds)

    return JSONResponse({
        "status": "recorded",
        "race_id": race_id,
        "n_entries": len(odds.get("entries", [])),
    })


@app.get("/api/odds/history/{race_id}", response_class=JSONResponse)
async def get_odds_history(race_id: str):
    """指定レースのオッズ推移履歴を返す。"""
    from src.pipeline.models.odds_predictor import OddsTrajectoryTracker, ODDS_HISTORY_DIR
    history_path = ODDS_HISTORY_DIR / f"{race_id}.json"
    try:
        exists = history_path.exists()
    except OSError:
        return JSONResponse({"snapshots": [], "message": "履歴なし"})
    if not exists:
        return JSONResponse({"snapshots": [], "message": "履歴なし"})
    try:
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        return JSONResponse({"race_id": race_id, "n_snapshots": len(history), "snapshots": history})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/odds/predict/{race_id}", response_class=JSONResponse)
async def get_predicted_odds_api(race_id: str):
    """指定レースの予測オッズを返す。"""
    try:
        from src.pipeline.features.feature_builder import build_race_features

        st = _get_storage()
        race_data = {}
        for cat in ["race_shutuba", "race_result", "race_card"]:
            d = st.load(cat, race_id)
            if d:
                race_data[cat] = d

        features_df = build_race_features(race_data)
        if features_df.empty:
            return JSONResponse({"error": "特徴量構築失敗"}, status_code=400)

        from src.pipeline.models.odds_predictor import get_predicted_odds
        result = get_predicted_odds(features_df, race_id)
        return JSONResponse({"race_id": race_id, "predicted_odds": result})
    except Exception as e:
        import traceback as tb
        return JSONResponse({"error": str(e), "trace": tb.format_exc()}, status_code=500)


# ── Composite Score シミュレーション API ──────────────────

_sim_job: dict = {}


@app.post("/api/simulation/run", response_class=JSONResponse)
async def run_composite_simulation(background_tasks: BackgroundTasks, max_races: int = 2000):
    """バックテストシミュレーションをバックグラウンドで実行する。"""
    if _sim_job.get("running"):
        return JSONResponse({"status": "already_running", "started_at": _sim_job.get("started_at")})

    _sim_job["running"] = True
    _sim_job["started_at"] = datetime.now().isoformat()
    _sim_job["result"] = None
    _sim_job["error"] = None
    _sim_job["finished_at"] = None

    background_tasks.add_task(_run_simulation, max_races)
    return JSONResponse({"status": "started", "max_races": max_races})


def _run_simulation(max_races: int):
    try:
        from src.pipeline.inference.composite_optimizer import CompositeOptimizer, SimulationConfig
        optimizer = CompositeOptimizer(SimulationConfig())
        result = optimizer.optimize(_get_storage(), max_races=max_races)
        _sim_job["result"] = {
            "prob_weight": result.prob_weight,
            "min_prob_honmei_ratio": result.min_prob_honmei_ratio,
            "min_prob_taikou_ratio": result.min_prob_taikou_ratio,
            "top_n_bet": result.top_n_bet,
            "best_score": result.best_score,
            "roi": result.roi,
            "hit_rate": result.hit_rate,
            "top3_capture": result.top3_capture,
            "sharpe_ratio": result.sharpe_ratio,
            "n_races": result.n_races,
            "n_bets": result.n_bets,
        }
    except Exception as e:
        import traceback as tb
        _sim_job["error"] = str(e)
        logger.error("シミュレーション失敗: %s\n%s", e, tb.format_exc())
    finally:
        _sim_job["running"] = False
        _sim_job["finished_at"] = datetime.now().isoformat()


@app.get("/api/simulation/status", response_class=JSONResponse)
async def get_simulation_status():
    """シミュレーションジョブの現在の状態を返す。"""
    return JSONResponse({
        "running": _sim_job.get("running", False),
        "started_at": _sim_job.get("started_at", ""),
        "finished_at": _sim_job.get("finished_at", ""),
        "result": _sim_job.get("result"),
        "error": _sim_job.get("error"),
    })


@app.get("/api/simulation/params", response_class=JSONResponse)
async def get_current_composite_params():
    """現在有効なcomposite scoreパラメータと、最適化結果の詳細を返す。"""
    from src.pipeline.inference.composite_optimizer import OPTIM_RESULT_PATH, get_composite_params

    params = get_composite_params()
    detail = None
    if OPTIM_RESULT_PATH.exists():
        try:
            with open(OPTIM_RESULT_PATH, encoding="utf-8") as f:
                detail = json.load(f)
        except Exception:
            pass

    return JSONResponse({
        "current_params": params,
        "is_optimized": detail is not None,
        "detail": detail,
    })


# ═══════════════════════════════════════════════════════
# Backfill 進捗 API
# ═══════════════════════════════════════════════════════

@app.get("/api/backfill/status", response_class=JSONResponse)
async def get_backfill_status():
    """過去データ取得 (Backfill) の進捗状況を返す。"""
    from src.scraper.backfill import BackfillProgress, _generate_race_dates, DEFAULT_START_YEAR
    from src.scraper.backfill import LOCK_DIR, _pid_alive

    progress = BackfillProgress()
    summary = progress.get_summary()

    now = datetime.now()
    total_possible = 0
    yearly: dict[str, dict] = {}
    for y in range(DEFAULT_START_YEAR, now.year + 1):
        dates = _generate_race_dates(y)
        total_possible += len(dates)
        fast_done = [d for d in dates if progress.is_date_done(d, "fast")]
        full_done = [d for d in dates if progress.is_date_done(d, "full")]
        yearly[str(y)] = {
            "total_dates": len(dates),
            "fast_done": len(fast_done),
            "full_done": len(full_done),
            "fast_pct": round(len(fast_done) / len(dates) * 100, 1) if dates else 0,
        }

    fast_total = summary.get("fast_dates", 0)
    pct = round(fast_total / total_possible * 100, 1) if total_possible else 0

    running_jobs: list[dict] = []
    if LOCK_DIR.exists():
        for lf in LOCK_DIR.glob("backfill_*.lock"):
            try:
                info = json.loads(lf.read_text())
                running_jobs.append({
                    "name": lf.stem,
                    "pid": info.get("pid"),
                    "started": info.get("started"),
                    "alive": _pid_alive(info.get("pid", 0)),
                })
            except Exception:
                pass

    return JSONResponse({
        "overall": {
            "fast_dates_done": fast_total,
            "total_dates": total_possible,
            "fast_pct": pct,
            "horses_done": summary.get("horses_done", 0),
            "last_updated": summary.get("last_updated", "—"),
        },
        "yearly": yearly,
        "running_jobs": running_jobs,
    })


@app.post("/api/backfill/start", response_class=JSONResponse)
async def start_backfill(request: Request):
    """Backfill ジョブをバックグラウンドで開始する。"""
    import subprocess

    body = await request.json()
    year = body.get("year")
    phase = body.get("phase", "auto")
    max_dates = body.get("max_dates", 5)

    cmd = [sys.executable, "-m", "src.scraper.backfill",
           "--phase", phase, "--max-dates", str(max_dates)]
    if year:
        cmd.extend(["--year", str(year)])

    log_suffix = f"_{year}" if year else ""
    log_path = f"logs/backfill{log_suffix}_{phase}.log"
    os.makedirs("logs", exist_ok=True)

    with open(log_path, "a") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )

    return JSONResponse({
        "status": "started",
        "pid": proc.pid,
        "year": year,
        "phase": phase,
        "max_dates": max_dates,
        "log": log_path,
    })


def _race_lists_backfill_lock_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "logs" / "race_lists_backfill.lock"


def _pid_alive_rlb(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _race_lists_backfill_conservative_env(env: dict[str, str]) -> None:
    """
    netkeiba 側のブロック・SSL 異常を減らすため、子プロセスだけ間隔を広げる。
    setdefault のため既に .env 等で指定されていれば上書きしない。
    """
    env.setdefault("NETKEIBA_BURST_WINDOW", "5")
    env.setdefault("NETKEIBA_BURST_COOLDOWN_MIN", "14")
    env.setdefault("NETKEIBA_BURST_COOLDOWN_MAX", "26")
    env.setdefault("NETKEIBA_SESSION_COOLDOWN_INTERVAL", "32")
    env.setdefault("NETKEIBA_SESSION_COOLDOWN_MIN", "50")
    env.setdefault("NETKEIBA_SESSION_COOLDOWN_MAX", "100")
    env.setdefault("NETKEIBA_SESSION_REFRESH_INTERVAL", "70")
    env.setdefault("NETKEIBA_THROTTLE_MIN", "4.5")
    env.setdefault("NETKEIBA_THROTTLE_MAX", "9.0")


@app.get("/api/race-lists-backfill/status", response_class=JSONResponse)
async def race_lists_backfill_status():
    """race_lists 開催日バックフィル subprocess の状態とログ末尾。"""
    from pathlib import Path

    base = Path(__file__).resolve().parents[2]
    lock_path = _race_lists_backfill_lock_path()
    out: dict[str, Any] = {
        "running": False,
        "pid": None,
        "started_at_jst": None,
        "log_path": None,
        "log_tail": [],
        "stale_lock_cleared": False,
    }
    if not lock_path.exists():
        return JSONResponse(out)

    try:
        info = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as e:
        out["lock_error"] = str(e)
        return JSONResponse(out)

    pid = int(info.get("pid") or 0)
    out["pid"] = pid or None
    out["started_at_jst"] = info.get("started_at_jst")
    lp = info.get("log_path")
    out["log_path"] = lp
    alive = _pid_alive_rlb(pid)
    out["running"] = alive

    if not alive and pid:
        try:
            lock_path.unlink()
            out["stale_lock_cleared"] = True
        except OSError:
            pass
        return JSONResponse(out)

    if lp:
        p = Path(lp)
        if not p.is_absolute():
            p = base / lp
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                out["log_tail"] = lines[-48:]
            except OSError as e:
                out["log_read_error"] = str(e)

    return JSONResponse(out)


@app.post("/api/race-lists-backfill/start", response_class=JSONResponse)
async def race_lists_backfill_start(request: Request):
    """scripts/data/backfill_race_lists_kaisai_since_2020.py をバックグラウンド起動。"""
    import subprocess
    from pathlib import Path

    base = Path(__file__).resolve().parents[2]
    lock_path = _race_lists_backfill_lock_path()
    base.joinpath("logs").mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        try:
            old = json.loads(lock_path.read_text(encoding="utf-8"))
            opid = int(old.get("pid") or 0)
            if opid and _pid_alive_rlb(opid):
                return JSONResponse(
                    {
                        "error": "既に race_lists バックフィルが実行中です",
                        "pid": opid,
                        "log_path": old.get("log_path"),
                    },
                    status_code=409,
                )
        except Exception:
            pass
        try:
            lock_path.unlink()
        except OSError:
            pass

    try:
        body = await request.json()
    except Exception:
        body = {}

    conservative = body.get("conservative")
    if conservative is None:
        conservative = True
    else:
        conservative = bool(conservative)

    start = (body.get("start") or "2024-01-01").strip()
    end = (body.get("end") or "").strip()
    if body.get("sleep") is not None:
        sleep_sec = float(body.get("sleep"))
    else:
        sleep_sec = 1.2 if conservative else 0.35
    force = bool(body.get("force"))
    dry_run = bool(body.get("dry_run"))
    accept_implausible = bool(body.get("accept_implausible"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path_rel = f"logs/backfill_race_lists_{ts}.log"
    log_abs = base / log_path_rel

    script = base / "scripts" / "data" / "backfill_race_lists_kaisai_since_2020.py"
    if not script.is_file():
        return JSONResponse(
            {"error": f"スクリプトが見つかりません: {script}"},
            status_code=500,
        )

    cmd = [sys.executable, str(script), "--sleep", str(sleep_sec)]
    if start:
        cmd.extend(["--start", start])
    if end:
        cmd.extend(["--end", end])
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    if accept_implausible:
        cmd.append("--accept-implausible")

    child_env = os.environ.copy()
    if conservative:
        _race_lists_backfill_conservative_env(child_env)

    with open(log_abs, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(base),
            env=child_env,
        )

    info = {
        "pid": proc.pid,
        "started_at_jst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": log_path_rel,
        "cmd": cmd,
        "conservative": conservative,
        "sleep": sleep_sec,
    }
    lock_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        (base / "logs" / "backfill_race_lists_latest.log").write_text(
            str(log_abs) + "\n", encoding="utf-8"
        )
    except OSError:
        pass

    return JSONResponse({"status": "started", **info})


@app.post("/api/race-lists-backfill/stop", response_class=JSONResponse)
async def race_lists_backfill_stop():
    """起動時に記録した PID に SIGTERM を送る。"""
    import signal
    from pathlib import Path

    base = Path(__file__).resolve().parents[2]
    lock_path = _race_lists_backfill_lock_path()
    if not lock_path.exists():
        return JSONResponse({"status": "idle", "message": "race_lists バックフィルの実行記録がありません"})

    try:
        info = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as e:
        return JSONResponse({"error": f"lock 読込失敗: {e}"}, status_code=500)

    pid = int(info.get("pid") or 0)
    if not pid:
        try:
            lock_path.unlink()
        except OSError:
            pass
        return JSONResponse({"status": "idle", "message": "PID が記録されていません"})

    if not _pid_alive_rlb(pid):
        try:
            lock_path.unlink()
        except OSError:
            pass
        return JSONResponse({"status": "already_stopped", "pid": pid})

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            lock_path.unlink()
        except OSError:
            pass
        return JSONResponse({"status": "already_stopped", "pid": pid})
    except Exception as e:
        return JSONResponse({"error": str(e), "pid": pid}, status_code=500)

    # lock はプロセス終了後、GET /status が stale として削除する
    return JSONResponse({"status": "signal_sent", "pid": pid, "signal": "SIGTERM"})


# ══════════════════════════════════════════════════════
# 馬券最適化 (Betting Optimizer)
# ══════════════════════════════════════════════════════


@app.get("/betting", response_class=HTMLResponse)
async def betting_page(request: Request):
    return templates.TemplateResponse("betting/betting.html", {
        "request": request,
        "current_page": "betting",
        "breadcrumbs": [],
    })


@app.post("/api/betting/optimize", response_class=JSONResponse)
async def api_betting_optimize(request: Request):
    """
    レース予測 + オッズ → 馬券ポートフォリオ最適化 (単勝/複勝/馬連/ワイド/馬単)

    Body: {
        "race_id": "...",
        "bankroll": 100000,
        "bet_types": ["tansho", "fukusho", "umaren", "wide"],
        "kelly_fraction": 0.25,
        "min_ev": 1.05,
    }
    """
    from src.pipeline.inference.betting import BettingOptimizer, BettingConfig

    body = await request.json()
    race_id = body.get("race_id", "")
    bankroll = body.get("bankroll", 100000)

    if not race_id:
        return JSONResponse({"error": "race_id is required"}, status_code=400)

    def _run():
        storage = _get_storage()
        runner = _get_runner()
        bet_types = body.get("bet_types", ["tansho", "fukusho", "umaren", "wide"])
        need_pair = any(t in bet_types for t in ("umaren", "wide", "umatan"))
        need_single = any(t in bet_types for t in ("tansho", "fukusho"))
        pair_odds = {}
        if need_pair:
            pair_odds = storage.load("race_pair_odds", race_id)
            if not pair_odds:
                pair_odds = runner.scrape_pair_odds(race_id, skip_existing=False)
            if not pair_odds:
                pair_odds = {}
        single_odds = None
        if need_single:
            single_odds = storage.load("race_odds", race_id)
            if not single_odds:
                single_odds = runner.scrape_odds(race_id, skip_existing=False)
        predictions = _get_race_predictions(race_id, storage)
        if not predictions:
            return None
        config = BettingConfig(
            bet_types=bet_types,
            kelly_fraction=body.get("kelly_fraction", 0.25),
            min_ev=body.get("min_ev", 1.05),
            min_prob=body.get("min_prob", 0.02),
            max_candidates=body.get("max_candidates", 15),
            top_n_for_pairs=body.get("top_n_for_pairs", 6),
        )
        optimizer = BettingOptimizer(config)
        return optimizer.optimize(
            predictions, pair_odds, bankroll, single_odds=single_odds,
        )

    portfolio = await asyncio.to_thread(_run)
    if portfolio is None:
        return JSONResponse(
            {"error": "予測結果がありません。先にモデル予測を実行してください"},
            status_code=404,
        )

    candidates_out = []
    for c in portfolio["candidates"]:
        candidates_out.append({
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
        })

    return JSONResponse({
        "race_id": race_id,
        "bankroll": bankroll,
        "candidates": candidates_out,
        "total_bet": portfolio["total_bet"],
        "expected_return": portfolio["expected_return"],
        "expected_roi": portfolio["expected_roi"],
        "remaining": portfolio["remaining"],
        "prob_distribution": portfolio["prob_distribution"],
    })


@app.get("/api/betting/pair-odds/{race_id}", response_class=JSONResponse)
async def api_pair_odds(race_id: str):
    """2連系オッズを取得する。GCSにない場合はスクレイピングを実行。"""
    def _load():
        storage = _get_storage()
        data = storage.load("race_pair_odds", race_id)
        if not data:
            runner = _get_runner()
            data = runner.scrape_pair_odds(race_id, skip_existing=False)
        return data
    data = await asyncio.to_thread(_load)
    if not data:
        return JSONResponse({"error": "odds not found"}, status_code=404)
    return JSONResponse(data)


def _get_race_predictions(race_id: str, storage) -> list[dict]:
    """race_id に対する予測結果を取得する。"""
    pred_data = storage.load("race_predictions", race_id)
    if pred_data and pred_data.get("predictions"):
        return pred_data["predictions"]

    try:
        import json as _json
        pred_path = os.path.join("data", "processed", "predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                all_preds = _json.load(f)
            for rp in all_preds.get("races", []):
                if rp.get("race_id") == race_id:
                    return rp.get("predictions", [])
    except Exception:
        pass

    shutuba = storage.load("race_shutuba", race_id)
    odds = storage.load("race_odds", race_id)
    if shutuba and shutuba.get("entries"):
        odds_map = {}
        if odds:
            for e in odds.get("entries", []):
                odds_map[e.get("horse_number")] = e

        preds = []
        for i, e in enumerate(shutuba["entries"]):
            hn = e.get("horse_number", i + 1)
            om = odds_map.get(hn, {})
            preds.append({
                "horse_number": hn,
                "horse_name": e.get("horse_name", f"#{hn}"),
                "horse_id": e.get("horse_id", ""),
                "pred_score": om.get("popularity", len(shutuba["entries"]) - i),
                "win_odds": om.get("win_odds", 0),
            })
        return preds

    return []


# ══════════════════════════════════════════════════════
#  血統ベクトル空間マップ (成績ベース)
# ══════════════════════════════════════════════════════

@app.get("/tracking-difficulty", response_class=HTMLResponse)
async def tracking_difficulty_page(request: Request):
    return templates.TemplateResponse("analysis/tracking_difficulty.html", {
        "request": request,
        "current_page": "tracking_difficulty",
        "breadcrumbs": [],
    })


@app.get("/ai-sla", response_class=HTMLResponse)
async def ai_sla_page(request: Request):
    return templates.TemplateResponse("analysis/ai_sla.html", {"request": request})


def _url_path_prefix_before_suffix(url_path: str, page_suffix: str) -> str:
    """
    ブラウザが実際に叩いたパスから、サブパス配信用のプレフィックスを推定する。
    例: /keiba/queue-status + /queue-status → /keiba
    root_path がプロキシで付かない環境でも、ページURLと API のベースを一致させる。
    """
    suf = page_suffix if page_suffix.startswith("/") else f"/{page_suffix}"
    suf = suf.rstrip("/") or suf
    p = (url_path or "/").rstrip("/") or "/"
    if p.endswith(suf):
        return p[: -len(suf)].rstrip("/")
    return ""


@app.get("/scrape", response_class=HTMLResponse)
async def scrape_management_page(request: Request):
    from src.scraper.queue_tasks import catalog_for_api

    return templates.TemplateResponse("admin/scrape.html", {
        "request": request,
        "current_page": "scrape",
        "breadcrumbs": [{"label": "ホーム", "url": "/"}, {"label": "スクレイピング管理"}],
        "task_catalog": catalog_for_api(),
        "is_dev": is_developer(request),
    })


@app.get("/scrape-control", response_class=HTMLResponse)
async def scrape_control_page(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/scrape?tab=control", status_code=302)


@app.get("/queue-status", response_class=HTMLResponse)
async def queue_status_page(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/scrape?tab=queue", status_code=302)


@app.get("/cron-jobs", response_class=HTMLResponse)
async def cron_jobs_page(request: Request):
    return templates.TemplateResponse(
        "admin/cron_jobs.html",
        {"request": request, "current_page": "cron_jobs"},
    )


# ── 開発者: 直近サーバーログ（logs/*.log 末尾） ──


def _admin_safe_log_basename(basename: str) -> str | None:
    """``logs/`` 直下の *.log のみ（パストラバーサル禁止）。"""
    import re as _re
    s = (basename or "").strip()
    if not s or _re.search(r"[/\\]", s) or ".." in s:
        return None
    if not _re.match(r"^[\w.\-]+\.log$", s, _re.IGNORECASE):
        return None
    return s


def _list_admin_log_files() -> list[dict[str, Any]]:
    from pathlib import Path
    d = Path(BASE_DIR) / "logs"
    if not d.is_dir():
        return []
    items: list[tuple[float, Any, int]] = []
    for p in d.glob("*.log"):
        try:
            st = p.stat()
            items.append((st.st_mtime, p, st.st_size))
        except OSError:
            continue
    items.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "name": p.name,
            "size_bytes": sz,
            "mtime": datetime.fromtimestamp(mtime).isoformat(),
        }
        for mtime, p, sz in items
    ]


def _read_admin_log_tail(
    basename: str, *, max_lines: int, max_read_bytes: int
) -> tuple[str | None, int, bool, str | None]:
    """
    戻り値: (内容 or None, 行数, 先頭欠落の可能性, エラー)
    """
    from pathlib import Path
    b = _admin_safe_log_basename(basename)
    if not b:
        return (None, 0, False, "不正なファイル名です")
    p = (Path(BASE_DIR) / "logs" / b).resolve()
    logs_dir = (Path(BASE_DIR) / "logs").resolve()
    if p.parent != logs_dir or not p.is_file():
        return (None, 0, False, "ファイルが存在しません")
    try:
        size = p.stat().st_size
    except OSError as e:
        return (None, 0, False, str(e))
    truncated = False
    to_read = min(size, max_read_bytes)
    if size > to_read:
        truncated = True
    try:
        with open(p, "rb") as f:
            if size > to_read:
                f.seek(size - to_read)
            raw = f.read()
    except OSError as e:
        return (None, 0, False, str(e))
    text = raw.decode("utf-8", errors="replace")
    if size > to_read and "\n" in text:
        text = text.split("\n", 1)[-1] or text
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated = True
    return ("\n".join(lines) + ("\n" if lines else ""), len(lines), truncated, None)


@app.get("/api/admin/server-logs", response_class=JSONResponse)
async def api_admin_server_logs(
    request: Request,
    file: str | None = Query(None, description="logs/ 下の .log ファイル名"),
    max_lines: int = Query(500, ge=10, le=5_000),
    max_kib: int = Query(2048, ge=64, le=8192, description="ファイル末尾から読む最大キロバイト"),
):
    """``logs/`` 内ログの末尾を返す。開発者セッション必須。"""
    if not is_developer(request):
        return JSONResponse({"ok": False, "error": "認証が必要です"}, status_code=401)
    max_read_bytes = max_kib * 1024
    found = _list_admin_log_files()
    if not file and found:
        file = found[0]["name"]
    elif not file:
        return JSONResponse({
            "ok": True,
            "file": None,
            "files": [],
            "content": "",
            "line_count": 0,
            "message": "logs/ に .log がありません（scripts/server/restart_server.sh 実行で logs/server_YYYYMMDD_HHMMSS.log が作られます）",
        })
    content, nlines, trunc, err = _read_admin_log_tail(
        file, max_lines=max_lines, max_read_bytes=max_read_bytes
    )
    if err:
        return JSONResponse({"ok": False, "error": err, "files": found}, status_code=400)
    return JSONResponse({
        "ok": True,
        "file": _admin_safe_log_basename(file),
        "files": found,
        "content": content,
        "line_count": nlines,
        "truncated": trunc,
        "max_lines": max_lines,
        "max_read_bytes": max_read_bytes,
    })


@app.get("/server-logs", response_class=HTMLResponse)
async def server_logs_page(request: Request):
    from_path = _url_path_prefix_before_suffix(request.url.path, "/server-logs")
    root = (request.scope.get("root_path") or "").rstrip("/")
    api_prefix = from_path or root
    return templates.TemplateResponse("admin/server_logs.html", {
        "request": request,
        "current_page": "server_logs",
        "breadcrumbs": [],
        "api_prefix": api_prefix,
        "is_dev": is_developer(request),
    })


@app.get("/scrape-upcoming", response_class=HTMLResponse)
async def scrape_upcoming_page(request: Request):
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/scrape?tab=upcoming", status_code=302)


@app.get("/bloodline-vector", response_class=HTMLResponse)
async def bloodline_vector_page(request: Request):
    """
    血統ベクトル空間 v2 (L2 メタクラスタ + 非主流グループ単位)
    src.research.pedigree.bloodline_vector_l2 で生成した
    groups_embeddings.json を読み込んでテンプレートに渡す。
    """
    import json as _json
    from pathlib import Path as _Path
    payload: dict = {"nodes": [], "similar_top": {}, "meta": {}}
    p = _Path("data/page_reference/bloodline_vector/v2_l2/groups_embeddings.json")
    if p.exists():
        try:
            payload = _json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            payload = {"nodes": [], "similar_top": {}, "meta": {"error": str(e)}}
    return templates.TemplateResponse("pedigree/bloodline_vector.html", {
        "request": request,
        "current_page": "bloodline_vector",
        "breadcrumbs": [],
        "groups_payload_json": _json.dumps(payload, ensure_ascii=False),
    })


# ══════════════════════════════════════════════════════
#  血統構造マップ (祖先共有ベース)
# ══════════════════════════════════════════════════════

@app.get("/pedigree-map", response_class=HTMLResponse)
async def pedigree_map_page(request: Request):
    return templates.TemplateResponse("pedigree/pedigree_map.html", {
        "request": request,
        "current_page": "pedigree_map",
        "breadcrumbs": [],
    })


@app.get("/note-aptitude-race", response_class=HTMLResponse)
async def note_aptitude_race_page(request: Request):
    """note 血統ナレッジのレース全頭3次元マップ（UI）。"""
    from_path = _url_path_prefix_before_suffix(request.url.path, "/note-aptitude-race")
    root = (request.scope.get("root_path") or "").rstrip("/")
    api_prefix = from_path or root
    return templates.TemplateResponse("race/note_aptitude_race.html", {
        "request": request,
        "current_page": "note_aptitude_race",
        "breadcrumbs": [],
        "api_prefix": api_prefix,
    })


@app.get("/api/pedigree-map", response_class=JSONResponse)
async def api_pedigree_map():
    data_path = os.path.join(
        BASE_DIR, "data", "research", "pedigree_similarity", "pedigree_similarity.json"
    )
    if not os.path.exists(data_path):
        return JSONResponse({"error": "データ未生成。python -m src.research.pedigree.pedigree_similarity を実行してください。"})
    with open(data_path, encoding="utf-8") as f:
        payload = json.load(f)

    # ── 各ノードに特殊適性タグを付与 ──────────────
    try:
        from src.api.bloodline_meta_cluster import (
            get_sire_tags_map, list_tag_definitions, TAG_CATEGORY_COLORS,
        )
        sire_tags_map = get_sire_tags_map(min_lift=1.0)
        if sire_tags_map and isinstance(payload.get("data"), list):
            for node in payload["data"]:
                hid = str(node.get("horse_id", ""))
                if hid in sire_tags_map:
                    node["tags"] = sire_tags_map[hid]
                else:
                    node["tags"] = []
        payload["tag_definitions"] = list_tag_definitions()
        payload["tag_category_colors"] = TAG_CATEGORY_COLORS
    except Exception as e:
        logger.warning("pedigree-map タグ付与失敗: %s", e)
    return JSONResponse(payload)


@app.get("/api/pedigree-map/tags", response_class=JSONResponse)
async def api_pedigree_map_tags(min_lift: float = 1.0):
    """pedigree-map ノードごとの特殊適性タグマップ (horse_id → tags)。"""
    try:
        from src.api.bloodline_meta_cluster import (
            get_sire_tags_map, list_tag_definitions, TAG_CATEGORY_COLORS,
        )
        return JSONResponse({
            "tags_by_horse": get_sire_tags_map(min_lift=min_lift),
            "tag_definitions": list_tag_definitions(),
            "tag_category_colors": TAG_CATEGORY_COLORS,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pedigree-map/cluster-hierarchy", response_class=JSONResponse)
async def api_pedigree_map_cluster_hierarchy():
    """L1 (系統) × L2 (適性) × L3 (細粒度) の階層構造を返す。

    pedigree-map のクラスタ階層タブで描画するためのデータ。
    """
    try:
        from src.api.bloodline_meta_cluster import get_cluster_hierarchy
        return JSONResponse(get_cluster_hierarchy())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pedigree-map/tags-full", response_class=JSONResponse)
async def api_pedigree_map_tags_full():
    """父系ツリー用 (緩い閾値): 全種牡馬 (~400頭) のタグマップ + タグ定義 + 色情報。

    ツリー UI 初期化時に一括取得し、各ノード描画時に horse_id でルックアップする。
    """
    try:
        from src.api.bloodline_meta_cluster import (
            get_sire_tags_full_map, list_tag_definitions, TAG_CATEGORY_COLORS,
        )
        m = get_sire_tags_full_map()
        return JSONResponse({
            "tags_by_horse": m,
            "n_stallions": len(m),
            "tag_definitions": list_tag_definitions(),
            "tag_category_colors": TAG_CATEGORY_COLORS,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stallion-sire-tree", response_class=JSONResponse)
async def api_stallion_sire_tree():
    """
    アンカーノード間の親子関係ツリー (50ノード・軽量版) を返す。
    research/build_stallion_lineage.py が生成する anchor_tree.json を提供。
    """
    anchor_tree_path = os.path.join(
        BASE_DIR, "data", "research", "pedigree_race_index", "anchor_tree.json"
    )
    if not os.path.exists(anchor_tree_path):
        return JSONResponse({
            "error": "ツリーデータ未生成。python -m src.research.pedigree.build_stallion_lineage を実行してください。"
        })
    with open(anchor_tree_path, encoding="utf-8") as f:
        return JSONResponse(json.load(f))


# ── 全種牡馬ツリー（lazily cached in-memory） ─────────────────────────────────
_full_tree_nodes_df:  "pd.DataFrame | None" = None
_full_tree_node_map:  "dict[str, dict] | None" = None   # horse_id → node dict (O(1) lookup)
_full_tree_children:  "dict[str, list[str]] | None" = None  # ソート済み
_full_tree_parents:   "dict[str, str] | None" = None
_full_tree_lock = threading.Lock()

# root_horse_id → main_group (L1 系統) マッピング
_root_to_main: "dict[str, str] | None" = None


def _load_root_to_main() -> "dict[str, str]":
    """各 root の代表 L1 系統マッピングをロード (cached)。"""
    global _root_to_main
    if _root_to_main is not None:
        return _root_to_main
    path = os.path.join(BASE_DIR, "data", "research", "bloodline_meta_cluster",
                         "root_to_main_group.json")
    if not os.path.exists(path):
        _root_to_main = {}
        return _root_to_main
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _root_to_main = {k: v.get("main_group", "非主流") for k, v in data.items()}
    except Exception:
        _root_to_main = {}
    return _root_to_main

# ── 該当馬×母父ペアテーブル（horse_bms.parquet）キャッシュ ──────────────────
_horse_bms_df:   "pd.DataFrame | None" = None
_horse_bms_lock = threading.Lock()

# non-000a/ 競走馬の血統表（≒10世代）に登場する種牡馬 horse_id セット。
# ツリー表示をこのセットに絞ることで「分析対象外の古祖先」を除外する。
_relevant_stallion_ids: "set[str] | None" = None
# _RELEVANT_IDS_PATH は _PED_RACE_IDX_DIR が定義された後に参照するため、
# 実際のパスは _load_relevant_stallion_ids() / _regenerate_relevant_stallion_ids() 内で計算する。

# ── ツリー再構築ジョブ状態 ────────────────────────────────────────────────────
_rebuild_state: "dict[str, Any]" = {
    "running":     False,
    "progress":    0.0,      # 0.0 - 1.0
    "message":     "",
    "started_at":  None,
    "finished_at": None,
    "error":       None,
    "stats":       None,     # 完了時の統計情報
}
_rebuild_lock = threading.Lock()


def _load_relevant_stallion_ids() -> "set[str]":
    """non-000a/ 競走馬の血統表（≒10世代）に登場する種牡馬 horse_id セットをロード。
    ファイルが存在しない場合は空セット（フィルタなし）を返す。"""
    global _relevant_stallion_ids
    if _relevant_stallion_ids is not None:
        return _relevant_stallion_ids
    _relevant_ids_path = os.path.join(_PED_RACE_IDX_DIR, "relevant_stallion_ids.json")
    try:
        with open(_relevant_ids_path, encoding="utf-8") as f:
            data = json.load(f)
        _relevant_stallion_ids = set(data.get("ids", []))
    except (OSError, json.JSONDecodeError):
        _relevant_stallion_ids = set()
    return _relevant_stallion_ids


def _load_horse_bms() -> "bool":
    """horse_bms.parquet を lazy ロードしてキャッシュする。"""
    global _horse_bms_df
    if _horse_bms_df is not None:
        return True
    with _horse_bms_lock:
        if _horse_bms_df is not None:
            return True
        bms_path = os.path.join(_PED_RACE_IDX_DIR, "horse_bms.parquet")
        if not os.path.exists(bms_path):
            return False
        import pandas as _pd
        df = _pd.read_parquet(bms_path)
        for col in ("horse_id", "sire_id", "bms_id", "sire_root_id", "bms_root_id"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        _horse_bms_df = df
    return True


def _load_full_tree() -> bool:
    """full_sire_tree_nodes.parquet + full_sire_tree.parquet をキャッシュロード。"""
    global _full_tree_nodes_df, _full_tree_node_map, _full_tree_children, _full_tree_parents
    import pandas as _pd

    with _full_tree_lock:
        if _full_tree_nodes_df is not None:
            return True
        nodes_path = os.path.join(_PED_RACE_IDX_DIR, "full_sire_tree_nodes.parquet")
        edges_path = os.path.join(_PED_RACE_IDX_DIR, "full_sire_tree.parquet")
        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            return False

        nodes_df = _pd.read_parquet(nodes_path)
        nodes_df["horse_id"] = nodes_df["horse_id"].astype(str)
        _full_tree_nodes_df = nodes_df

        # O(1) ルックアップ用辞書を構築（DataFrame スキャンを排除）
        node_map: dict[str, dict] = {}
        for r in nodes_df.itertuples(index=False):
            hid = str(r.horse_id)
            node_map[hid] = {
                "id":            hid,
                "name":          str(r.name),
                "n_children":    int(r.n_children),
                "n_descendants": int(r.n_descendants),
                "depth":         int(r.depth_from_root),
                "is_root":       bool(r.is_root),
            }
        _full_tree_node_map = node_map

        edges_df = _pd.read_parquet(edges_path, columns=["sire_id", "child_id"])
        children_raw: dict[str, list[str]] = {}
        parents:  dict[str, str] = {}
        for row in edges_df.itertuples(index=False):
            sid = str(row.sire_id)
            cid = str(row.child_id)
            children_raw.setdefault(sid, []).append(cid)
            parents[cid] = sid

        # relevant_stallion_ids をロード（non-000a/ 競走馬の血統表≒10世代に登場する種牡馬）
        relevant_ids = _load_relevant_stallion_ids()

        # 子リストを horse_id と名前でユニーク化（同名は子孫数が多い方を優先）→ アルファベット順
        # ・末端ノード（n_children == 0）は除外
        # ・子孫が 1 頭以下（後継未確立）は除外
        # ・relevant_ids に含まれない（分析対象外）は除外
        def _dedup_children(cids: list[str]) -> list[str]:
            seen_ids:   set[str] = set()
            best_by_name: dict[str, str] = {}  # name → best horse_id
            for cid in cids:
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                info = node_map.get(cid, {})
                # 種牡馬子孫がいない（▶ が存在しない）ノードはスキップ
                # n_descendants は牡牝混在カウントのため信頼性が低く、
                # n_children > 0（実際に▶が出る）を唯一の基準とする
                if info.get("n_children", 0) == 0:
                    continue
                # non-000a/ 競走馬の血統表に登場しない種牡馬はスキップ
                if relevant_ids and cid not in relevant_ids:
                    continue
                name = info.get("name", cid).lower()
                cur_best = best_by_name.get(name)
                if cur_best is None:
                    best_by_name[name] = cid
                else:
                    # 子孫数が多い方を採用
                    cur_desc = node_map.get(cur_best, {}).get("n_descendants", 0)
                    new_desc = info.get("n_descendants", 0)
                    if new_desc > cur_desc:
                        best_by_name[name] = cid
            return sorted(
                best_by_name.values(),
                key=lambda c: node_map.get(c, {}).get("name", "").lower()
            )

        children: dict[str, list[str]] = {
            sid: _dedup_children(cids)
            for sid, cids in children_raw.items()
        }
        _full_tree_children = children
        _full_tree_parents  = parents

        # n_children を「末端フィルタ後の実際の可視子ノード数」に更新
        # parquet の n_children は末端ノード（n_children==0）を含むため
        # フロントエンドの hasKids 判定が狂い「▶あるのに展開で空」になるのを防ぐ
        for hid in node_map:
            node_map[hid]["n_children"] = len(children.get(hid, []))

        # DataFrame の n_children もフィルタ後の値に揃える
        # （roots / search エンドポイントのフィルタに使われるため必須）
        updated_n_children = nodes_df["horse_id"].map(
            lambda h: len(children.get(h, []))
        )
        nodes_df = nodes_df.copy()
        nodes_df["n_children"] = updated_n_children
        _full_tree_nodes_df = nodes_df

        return True


def _node_json(hid: str, with_tags: bool = True) -> dict:
    """単一ノードの JSON 表現を返す（O(1) dict lookup）。

    with_tags=True (default) の場合、特殊適性タグ (is_top20) を `tags` キーで埋め込む。
    """
    if _full_tree_node_map is None:
        base = {"id": hid, "name": hid}
    else:
        base = dict(_full_tree_node_map.get(hid) or {
            "id": hid, "name": hid, "n_children": 0, "n_descendants": 0, "depth": -1, "is_root": False
        })
    if with_tags:
        try:
            from src.api.bloodline_meta_cluster import get_sire_tags_full_map
            tags_map = get_sire_tags_full_map()
            base["tags"] = tags_map.get(hid, [])
        except Exception:
            base["tags"] = []
    return base


def _regenerate_relevant_stallion_ids(
    progress_cb: "Callable[[str, float], None] | None" = None
) -> None:
    """non-000a/ 競走馬の血統表（≒10世代）に登場する種牡馬 ID セットを再計算して保存する。"""
    import json as _json
    import time as _time
    import pandas as _pd

    def _cb(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(msg, frac)

    ped_dir  = os.path.join(os.path.dirname(_PED_RACE_IDX_DIR), "local", "horse_pedigree_5gen")
    out_path = os.path.join(_PED_RACE_IDX_DIR, "relevant_stallion_ids.json")

    _cb("関連種牡馬: 競走馬祖先を収集中...", 0.90)
    t0 = _time.time()
    seed_ids: set = set()
    n_files = 0
    for dirpath, _dirs, files in os.walk(ped_dir):
        folder = os.path.basename(dirpath)
        if folder == "000a":
            continue
        for fname in files:
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(dirpath, fname), encoding="utf-8") as f:
                    rec = _json.load(f)
            except Exception:
                continue
            n_files += 1
            hid = str(rec.get("horse_id") or "").strip()
            if hid:
                seed_ids.add(hid)
            for a in rec.get("ancestors") or []:
                aid = str(a.get("horse_id") or "").strip()
                if aid:
                    seed_ids.add(aid)

    _cb("関連種牡馬: 父系チェーンを辿り中...", 0.93)
    edges_path = os.path.join(_PED_RACE_IDX_DIR, "full_sire_tree.parquet")
    edges_df = _pd.read_parquet(edges_path, columns=["sire_id", "child_id"])
    sire_of = dict(zip(edges_df["child_id"].astype(str), edges_df["sire_id"].astype(str)))

    relevant: set = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(5):
        nxt = {sire_of[h] for h in frontier if h in sire_of and sire_of[h] not in relevant}
        relevant |= nxt
        frontier = nxt
        if not nxt:
            break

    out_data = {
        "ids": sorted(relevant),
        "count": len(relevant),
        "generated_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": n_files,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(out_data, f, ensure_ascii=False)

    logger.info("relevant_stallion_ids 再生成: %d 件 (%.1fs)", len(relevant), _time.time() - t0)
    _cb(f"関連種牡馬セット更新完了: {len(relevant):,} 件", 0.95)


def _invalidate_full_tree_cache() -> None:
    """インメモリツリーキャッシュを破棄する（再構築後に呼ぶ）。"""
    global _full_tree_nodes_df, _full_tree_node_map, _full_tree_children, _full_tree_parents
    global _relevant_stallion_ids, _horse_bms_df
    with _full_tree_lock:
        _full_tree_nodes_df = None
        _full_tree_node_map = None
        _full_tree_children = None
        _full_tree_parents  = None
    _relevant_stallion_ids = None
    with _horse_bms_lock:
        _horse_bms_df = None


def _run_rebuild_job() -> None:
    """バックグラウンドスレッドで全種牡馬ツリーを再構築する。"""
    import datetime as _dt2
    with _rebuild_lock:
        _rebuild_state["running"]     = True
        _rebuild_state["progress"]    = 0.0
        _rebuild_state["message"]     = "起動中..."
        _rebuild_state["started_at"]  = _dt2.datetime.now().isoformat()
        _rebuild_state["finished_at"] = None
        _rebuild_state["error"]       = None
        _rebuild_state["stats"]       = None

    def _cb(msg: str, frac: float) -> None:
        with _rebuild_lock:
            _rebuild_state["message"]  = msg
            _rebuild_state["progress"] = round(frac, 3)

    try:
        from src.research.pedigree.build_full_sire_tree import build as _build_tree
        stats = _build_tree(progress_cb=_cb)

        # relevant_stallion_ids を再生成する
        _cb("関連種牡馬セットを再計算中...", 0.88)
        try:
            _regenerate_relevant_stallion_ids(_cb)
        except Exception as _re:
            logger.warning("relevant_stallion_ids 再生成失敗（無視）: %s", _re)

        # horse_bms.parquet（該当馬×母父ペアテーブル）を再生成する
        _cb("母父ペアテーブルを再計算中...", 0.93)
        try:
            from src.research.pedigree.build_horse_bms_index import build as _build_bms
            _bms_stats = _build_bms()
            logger.info(
                "horse_bms 再生成: %d 件 (%ss)",
                _bms_stats["n_rows"], _bms_stats["elapsed_sec"],
            )
        except Exception as _be:
            logger.warning("horse_bms 再生成失敗（無視）: %s", _be)

        _invalidate_full_tree_cache()       # キャッシュ破棄 → 次回アクセス時に再ロード
        with _rebuild_lock:
            _rebuild_state["stats"]       = stats
            _rebuild_state["progress"]    = 1.0
            _rebuild_state["message"]     = (
                f"完了: ノード {stats['n_nodes']:,} / エッジ {stats['n_edges']:,} / "
                f"ルート {stats['n_roots']:,}  ({stats['elapsed_sec']}s)"
            )
            _rebuild_state["finished_at"] = _dt2.datetime.now().isoformat()
    except Exception as e:
        logger.exception("ツリー再構築エラー")
        with _rebuild_lock:
            _rebuild_state["error"]       = str(e)
            _rebuild_state["message"]     = f"エラー: {e}"
            _rebuild_state["progress"]    = 0.0
    finally:
        with _rebuild_lock:
            _rebuild_state["running"] = False


@app.post("/api/stallion-sire-tree/rebuild", response_class=JSONResponse)
async def api_stallion_sire_tree_rebuild():
    """
    全種牡馬ツリーをバックグラウンドで再構築する。
    既に実行中の場合は 409 を返す。
    """
    with _rebuild_lock:
        if _rebuild_state["running"]:
            return JSONResponse(
                {"error": "既に再構築ジョブが実行中です。", "state": dict(_rebuild_state)},
                status_code=409,
            )

    t = threading.Thread(target=_run_rebuild_job, daemon=True, name="sire-tree-rebuild")
    t.start()
    return JSONResponse({"ok": True, "message": "再構築ジョブを開始しました。"})


@app.get("/api/stallion-sire-tree/rebuild/status", response_class=JSONResponse)
async def api_stallion_sire_tree_rebuild_status():
    """再構築ジョブの現在状態を返す。"""
    with _rebuild_lock:
        state = dict(_rebuild_state)

    # 既存ファイルの更新日時も付与
    import os as _os
    nodes_path = os.path.join(_PED_RACE_IDX_DIR, "full_sire_tree_nodes.parquet")
    edges_path = os.path.join(_PED_RACE_IDX_DIR, "full_sire_tree.parquet")
    file_info = {}
    for key, path in [("nodes", nodes_path), ("edges", edges_path)]:
        if _os.path.exists(path):
            st = _os.stat(path)
            import datetime as _dt3
            file_info[key] = {
                "updated_at": _dt3.datetime.fromtimestamp(st.st_mtime).isoformat(),
                "size_kb": round(st.st_size / 1024, 1),
            }
        else:
            file_info[key] = None

    return JSONResponse({**state, "files": file_info})


@app.get("/api/stallion-sire-tree/l1-groups", response_class=JSONResponse)
async def api_stallion_sire_tree_l1_groups():
    """父系ツリーの第一階層 = L1 系統血統 (4 大主流 + 非主流) を返す。

    各 L1 ノードは以下を持つ:
        id, name, color, icon, founder_name, n_roots, n_descendants_total
    """
    if not _load_full_tree():
        return JSONResponse({"error": "full_sire_tree データ未生成。"}, status_code=404)
    root_to_main = _load_root_to_main()
    if not root_to_main:
        return JSONResponse({"error": "root_to_main_group.json 未生成。"
                              "python -m src.research.pedigree.build_root_to_main_group を実行してください。"},
                              status_code=404)
    relevant_ids = _load_relevant_stallion_ids()
    base_mask = (
        (_full_tree_nodes_df["is_root"] == True) &
        (_full_tree_nodes_df["n_children"] > 0)
    )
    if relevant_ids:
        base_mask = base_mask & _full_tree_nodes_df["horse_id"].isin(relevant_ids)
    roots_df = _full_tree_nodes_df[base_mask].copy()
    # ── L1 メタ定義 (bloodline_meta_cluster と整合) ──
    L1_META = [
        {"id": "Turn-To系",         "color": "#3b82f6", "icon": "△",
         "founder_name": "Turn-to (1951)",
         "description": "Hail to Reason → Roberto / Halo → Sunday Silence。Deep Impact, Hearts Cry 系。"},
        {"id": "Native Dancer系",   "color": "#10b981", "icon": "○",
         "founder_name": "Native Dancer (1950)",
         "description": "Raise a Native → Mr. Prospector → Kingmambo, King Kamehameha 系。"},
        {"id": "Northern Dancer系", "color": "#f59e0b", "icon": "◇",
         "founder_name": "Northern Dancer (1961)",
         "description": "Sadler's Wells / Storm Bird / Nijinsky / Storm Cat 系。"},
        {"id": "Nasrullah系",       "color": "#ef4444", "icon": "□",
         "founder_name": "Nasrullah (1940)",
         "description": "Princely Gift / Bold Ruler / Never Bend。古典系統。"},
        {"id": "非主流",             "color": "#9ca3af", "icon": "✕",
         "founder_name": "(複数の派生に属さない群)",
         "description": "Monsun系, Wild Rush系, Macho Uno系 等の独立系統群。"},
    ]
    l1_to_roots: "dict[str, list[str]]" = {m["id"]: [] for m in L1_META}
    for _, r in roots_df.iterrows():
        hid = str(r["horse_id"])
        l1 = root_to_main.get(hid, "非主流")
        if l1 not in l1_to_roots:
            l1_to_roots[l1] = []
        l1_to_roots[l1].append(hid)

    out = []
    for meta in L1_META:
        ids = l1_to_roots.get(meta["id"], [])
        n_desc_total = int(_full_tree_nodes_df[_full_tree_nodes_df["horse_id"].isin(ids)]["n_descendants"].sum())
        out.append({
            **meta,
            "name": meta["id"],
            "n_roots": len(ids),
            "n_descendants_total": n_desc_total,
        })
    return JSONResponse({"l1_groups": out})


@app.get("/api/stallion-sire-tree/roots", response_class=JSONResponse)
async def api_stallion_sire_tree_roots(
    limit: int = 50, offset: int = 0, main_group: str | None = None,
    sort: str = "n_descendants",
):
    """父不明（ルート）ノードを返す。

    Args:
        main_group: 指定すると該当 L1 系統に属する root のみ返す
                    (Turn-To系 / Native Dancer系 / Northern Dancer系 / Nasrullah系 / 非主流)
        sort: "n_descendants" (デフォルト, 子孫数の多い祖先順) または "name" (アルファベット順)
    """
    if not _load_full_tree():
        return JSONResponse({"error": "full_sire_tree データ未生成。python -m src.research.pedigree.build_full_sire_tree を実行してください。"}, status_code=404)
    relevant_ids = _load_relevant_stallion_ids()
    base_mask = (
        (_full_tree_nodes_df["is_root"] == True) &
        (_full_tree_nodes_df["n_children"] > 0)
    )
    if relevant_ids:
        base_mask = base_mask & _full_tree_nodes_df["horse_id"].isin(relevant_ids)
    # L1 フィルタ
    if main_group:
        root_to_main = _load_root_to_main()
        target_ids = {hid for hid, mg in root_to_main.items() if mg == main_group}
        base_mask = base_mask & _full_tree_nodes_df["horse_id"].isin(target_ids)
    roots_df = _full_tree_nodes_df[base_mask].copy()
    if sort == "name":
        roots_df = roots_df.sort_values("name", ascending=True, key=lambda s: s.str.lower())
    else:
        roots_df = roots_df.sort_values(
            ["n_descendants", "name"], ascending=[False, True],
        )
    total = len(roots_df)
    page  = roots_df.iloc[offset : offset + limit]
    root_to_main_cached = _load_root_to_main()
    nodes = []
    for _, r in page.iterrows():
        hid = str(r["horse_id"])
        node = _node_json(hid)
        node["main_group"] = main_group or root_to_main_cached.get(hid, "非主流")
        nodes.append(node)
    return JSONResponse({
        "total": total, "offset": offset, "nodes": nodes,
        "main_group": main_group, "sort": sort,
    })


@app.get("/api/stallion-sire-tree/node/{horse_id}", response_class=JSONResponse)
async def api_stallion_sire_tree_node(horse_id: str, offset: int = 0, limit: int = 100):
    """
    指定ノードの情報＋直接の子リスト（アルファベット順、ページネーション対応）。
    limit=0 で全件返す。
    """
    if not _load_full_tree():
        return JSONResponse({"error": "full_sire_tree データ未生成。"}, status_code=404)
    info = _node_json(horse_id)
    # ロード時にアルファベット順でソート済み
    all_children_ids = _full_tree_children.get(horse_id, [])
    total_children = len(all_children_ids)
    if limit > 0:
        page_ids = all_children_ids[offset: offset + limit]
    else:
        page_ids = all_children_ids
    children = [_node_json(c) for c in page_ids]
    parent_id = _full_tree_parents.get(horse_id)
    parent = _node_json(parent_id) if parent_id else None
    return JSONResponse({
        "node": info,
        "children": children,
        "total_children": total_children,
        "offset": offset,
        "limit": limit,
        "parent": parent,
    })


@app.get("/api/stallion-sire-tree/bms-stats/{horse_id}", response_class=JSONResponse)
async def api_stallion_sire_tree_bms_stats(horse_id: str, top: int = 10):
    """
    指定した種牡馬の BMS 関連統計を返す。

    Response:
      as_sire  : この種牡馬を「父」に持つ馬の 母父 TOP-N
      as_bms   : この種牡馬を「母父」に持つ馬の 父 TOP-N
      total_as_sire : 産駒総数
      total_as_bms  : 母父としての出現総数
    """
    if not _load_horse_bms():
        return JSONResponse({"error": "horse_bms データ未生成。ツリー再構築を実行してください。"}, status_code=404)

    df = _horse_bms_df

    def _top_list(sub, id_col: str, name_col: str, root_col: str) -> list[dict]:
        if sub.empty:
            return []
        grp = (
            sub.groupby([id_col, name_col, root_col], dropna=False)
               .size()
               .reset_index(name="count")
               .sort_values("count", ascending=False)
               .head(top)
        )
        return [
            {
                "id":        str(r[id_col]),
                "name":      str(r[name_col]),
                "root_name": str(r[root_col]),
                "count":     int(r["count"]),
            }
            for _, r in grp.iterrows()
            if str(r[id_col]).strip()
        ]

    # 産駒（父 = horse_id）の 母父分布
    as_sire_sub   = df[df["sire_id"] == horse_id]
    as_sire_total = len(as_sire_sub)
    as_sire       = _top_list(as_sire_sub, "bms_id", "bms_name", "bms_root_name")

    # 母父（bms = horse_id）の 父分布
    as_bms_sub   = df[df["bms_id"] == horse_id]
    as_bms_total = len(as_bms_sub)
    as_bms       = _top_list(as_bms_sub, "sire_id", "sire_name", "sire_root_name")

    return JSONResponse({
        "horse_id":      horse_id,
        "total_as_sire": as_sire_total,
        "total_as_bms":  as_bms_total,
        "as_sire":       as_sire,
        "as_bms":        as_bms,
    })


@app.get("/api/stallion-sire-tree/search", response_class=JSONResponse)
async def api_stallion_sire_tree_search(q: str = "", limit: int = 20):
    """
    名前で馬を検索して一致ノードを返す。
    """
    if not _load_full_tree():
        return JSONResponse({"error": "full_sire_tree データ未生成。"}, status_code=404)
    if not q.strip():
        return JSONResponse({"nodes": []})
    q_lower = q.strip().lower()
    relevant_ids = _load_relevant_stallion_ids()
    # n_children > 0 のみ: ▶ が存在するノードだけ検索対象
    mask = (
        _full_tree_nodes_df["name"].str.lower().str.contains(q_lower, na=False) &
        (_full_tree_nodes_df["n_children"] > 0)
    )
    if relevant_ids:
        mask = mask & _full_tree_nodes_df["horse_id"].isin(relevant_ids)
    matched = _full_tree_nodes_df[mask].sort_values("n_descendants", ascending=False).head(limit)
    nodes = [_node_json(str(r["horse_id"])) for _, r in matched.iterrows()]
    return JSONResponse({"nodes": nodes, "total": int(mask.sum())})


@app.get("/api/pedigree/note-aptitude", response_class=JSONResponse)
async def api_pedigree_note_aptitude(
    sire: str = "",
    dam_sire: str = "",
    broodmare_line: str = "",
    distance_m: int = 0,
    venue: str = "",
    surface: str = "芝",
    going_heavy: int = 0,
):
    """
    note まとめベースの種牡馬適性ナレッジから、父・母父・（任意）牝系・条件の簡易スコアを返す。
    予測確率ではなく説明用ヒューリスティック（research/sire_aptitude_note.py）。
    """
    try:
        from src.research.pedigree.sire_aptitude_note import (
            compute_note_aptitude_features,
            load_bundle,
            predict_context_score,
            stallion_table_rows,
        )

        bm = (broodmare_line or "").strip() or None
        feats = compute_note_aptitude_features(
            sire, dam_sire, int(distance_m or 0), broodmare_line=bm
        )
        ctx = predict_context_score(
            sire,
            dam_sire,
            venue=venue,
            surface=surface,
            distance_m=int(distance_m or 0),
            going_heavy=bool(int(going_heavy or 0)),
            broodmare_line=bm,
        )
        meta = load_bundle().get("meta", {})
        return JSONResponse({
            "meta": meta,
            "features": feats,
            "context": {k: v for k, v in ctx.items() if k != "blended_axes"},
            "blended_axes": ctx.get("blended_axes", {}),
            "table_preview": stallion_table_rows()[:5],
        })
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/pedigree/note-aptitude/table", response_class=JSONResponse)
async def api_pedigree_note_aptitude_table():
    """種牡馬×軸・牝系×軸の適性表（JSON 全行）。"""
    try:
        from src.research.pedigree.sire_aptitude_note import (
            broodmare_table_rows,
            load_bundle,
            stallion_table_rows,
        )

        return JSONResponse({
            "meta": load_bundle().get("meta", {}),
            "axes": load_bundle().get("axes", []),
            "rows": stallion_table_rows(),
            "broodmare_rows": broodmare_table_rows(),
        })
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/pedigree/race-note-3d", response_class=JSONResponse)
async def api_pedigree_race_note_3d(
    race_id: str = "",
    axis_x: str = "ts",
    axis_y: str = "gear_change",
    axis_z: str = "ts_sustain",
    w_father: float = 0.30,
    w_mf_mf: float = 0.14,
    w_mf_mm_f: float = 0.14,
    w_mm_f: float = 0.14,
    w_mmm_f: float = 0.14,
    w_mmmm_f: float = 0.14,
):
    """
    種牡馬因子統計ベースの適性3Dマップ。

    6祖先（父・母父母父・母父母母父・母母父・母母母父・母母母母父）の
    実レースデータ統計を重み付きブレンドし、選択された3軸で散布図を構成。
    """
    rid = (race_id or "").strip()
    if not rid:
        return JSONResponse({"error": "race_id が必要です"}, status_code=400)
    try:
        from src.research.pedigree.sire_factor_race_map import build_race_sire_factor_map

        weights = {
            "father": w_father, "mf_mf": w_mf_mf, "mf_mm_f": w_mf_mm_f,
            "mm_f": w_mm_f, "mmm_f": w_mmm_f, "mmmm_f": w_mmmm_f,
        }

        storage = _get_storage()
        out = await asyncio.to_thread(
            build_race_sire_factor_map,
            storage,
            rid,
            weights=weights,
            axis_x=axis_x,
            axis_y=axis_y,
            axis_z=axis_z,
        )
        if not out.get("horses"):
            return JSONResponse({
                **out,
                "error": "出走データがありません。race_shutuba または race_result（＋horse_result）を取得してください。",
            }, status_code=404)
        return JSONResponse(out)
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/pedigree/race-note-3d-compare", response_class=JSONResponse)
async def api_pedigree_race_note_3d_compare(
    race_id: str = "",
    axis_x: str = "ts",
    axis_y: str = "gear_change",
    axis_z: str = "ts_sustain",
    w_father: float = 0.30,
    w_mf_mf: float = 0.14,
    w_mf_mm_f: float = 0.14,
    w_mm_f: float = 0.14,
    w_mmm_f: float = 0.14,
    w_mmmm_f: float = 0.14,
    same_cond_from: str = "2025-01-01",
    same_cond_top_n: int = 3,
    recent_weeks: int = 2,
    cmp_surface: str = "",
    cmp_distance: int = 0,
    cmp_track_good: str = "",
):
    """同条件上位馬 + 直近同コースの比較データを返す。"""
    rid = (race_id or "").strip()
    if not rid:
        return JSONResponse({"error": "race_id が必要です"}, status_code=400)
    try:
        from src.research.pedigree.sire_factor_race_map import build_comparison_data

        weights = {
            "father": w_father, "mf_mf": w_mf_mf, "mf_mm_f": w_mf_mm_f,
            "mm_f": w_mm_f, "mmm_f": w_mmm_f, "mmmm_f": w_mmmm_f,
        }
        storage = _get_storage()
        out = await asyncio.to_thread(
            build_comparison_data,
            storage, rid,
            weights=weights,
            axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
            same_cond_from=same_cond_from,
            same_cond_top_n=same_cond_top_n,
            recent_weeks=recent_weeks,
            override_surface=cmp_surface.strip(),
            override_distance=cmp_distance,
            override_track_good=cmp_track_good.strip(),
        )
        return JSONResponse(out)
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/pedigree/race-note-3d-v2", response_class=JSONResponse)
async def api_pedigree_race_note_3d_v2(race_id: str = ""):
    """
    血統メタクラスタベースのレース適性マップ v2。
    旧 6 祖先重みブレンドに代わり compute_blended_prior_v2 を使用する。
    """
    rid = (race_id or "").strip()
    if not rid:
        return JSONResponse({"error": "race_id が必要です"}, status_code=400)
    try:
        from src.research.pedigree.race_note_3d_v2 import build_race_note_v2

        storage = _get_storage()
        out = await asyncio.to_thread(build_race_note_v2, storage, rid)
        if not out.get("horses"):
            return JSONResponse(
                {**out, "no_horse_data": True, "message": "出走データ未取得。race_shutuba を先に取得してください。"},
            )
        return JSONResponse(out)
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.get("/api/pedigree/week-races", response_class=JSONResponse)
async def api_pedigree_week_races():
    """今週の開催（土・日）のレース一覧を返す。/note-aptitude-race のデフォルト選択肢。"""
    try:
        from src.research.pedigree.race_note_3d_v2 import get_week_races

        storage = _get_storage()
        races = await asyncio.to_thread(get_week_races, storage)
        return JSONResponse({"races": races})
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.post("/api/pedigree/rebuild-sire-factor-stats", response_class=JSONResponse)
async def api_rebuild_sire_factor_stats(request: Request):
    """種牡馬因子統計を再計算する。mode=fast(スナップショット) / full(個別GCS走査+スナップショット更新)。"""
    if not is_developer(request):
        return JSONResponse({"error": "開発者ログインが必要です"}, status_code=401)
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        mode = body.get("mode", "fast")

        from src.research.pedigree.sire_factor_stats import (
            build_sire_factor_stats,
            build_sire_factor_stats_fast,
            build_and_upload_snapshot,
            invalidate_cache,
            save_sire_factor_stats,
        )

        storage = _get_storage()

        def _run():
            if mode == "full":
                data = build_sire_factor_stats(storage)
                save_sire_factor_stats(data, storage=storage)
                build_and_upload_snapshot(storage)
            else:
                data = build_sire_factor_stats_fast(storage)
                save_sire_factor_stats(data, storage=storage)
            invalidate_cache()
            return data.get("meta", {})

        meta = await asyncio.to_thread(_run)
        return JSONResponse({"status": "ok", "mode": mode, "meta": meta})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/pedigree/tune-weights", response_class=JSONResponse)
async def api_pedigree_tune_weights(request: Request):
    """ウェイト自動チューニング。scope=global(デフォルト) / race(race_id指定)。"""
    if not is_developer(request):
        return JSONResponse({"error": "開発者ログインが必要です"}, status_code=401)
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        race_id = body.get("race_id", "")

        from src.research.pedigree.sire_factor_aptitude import optimize_weights

        storage = _get_storage()
        race_ids = [race_id] if race_id else None

        result = await asyncio.to_thread(
            optimize_weights, storage, race_ids=race_ids
        )
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/pedigree/race-ensure-5gen", response_class=JSONResponse)
async def api_pedigree_race_ensure_5gen(request: Request):
    """
    指定レースの出走馬について horse_pedigree_5gen の欠損を調べ、
    欠損があれば PRIORITY_URGENT 相当でスクレイピングキューに投入する。

    （5世代適性ブレンドは各馬の1件の血統JSONのみ使用するため、祖先個体ごとの別取得は行わない。）
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    rid = str(body.get("race_id") or "").strip()
    if not rid:
        return JSONResponse({"error": "race_id が必要です"}, status_code=400)
    try:
        from src.research.pedigree.race_pedigree_5gen_prefetch import start_race_pedigree_prefetch

        storage = _get_storage()
        out = await asyncio.to_thread(start_race_pedigree_prefetch, storage, rid)
        _kick_urgent_worker()
        _kick_scrape_queue_worker()
        return JSONResponse(out)
    except Exception as e:
        import traceback as _tb
        return JSONResponse(
            {"error": str(e), "traceback": _tb.format_exc()},
            status_code=500,
        )


@app.get("/api/pedigree/race-ensure-5gen/status", response_class=JSONResponse)
async def api_pedigree_race_ensure_5gen_status(session_id: str = ""):
    """race-ensure-5gen で発行した session_id の進捗。"""
    sid = (session_id or "").strip()
    if not sid:
        return JSONResponse({"error": "session_id が必要です"}, status_code=400)
    try:
        from src.research.pedigree.race_pedigree_5gen_prefetch import session_progress

        storage = _get_storage()
        return JSONResponse(session_progress(sid, storage))
    except Exception as e:
        import traceback as _tb
        return JSONResponse(
            {"error": str(e), "traceback": _tb.format_exc()},
            status_code=500,
        )


@app.post("/api/pedigree/race-ensure-5gen/cancel", response_class=JSONResponse)
async def api_pedigree_race_ensure_5gen_cancel(request: Request):
    """race-ensure-5gen セッションの pending ジョブをキューから外し、セッションを中止済みにする。"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    sid = str(body.get("session_id") or "").strip()
    if not sid:
        return JSONResponse({"error": "session_id が必要です"}, status_code=400)
    try:
        from src.research.pedigree.race_pedigree_5gen_prefetch import cancel_race_pedigree_prefetch_session

        out = await asyncio.to_thread(cancel_race_pedigree_prefetch_session, sid)
        if not out.get("ok"):
            err = str(out.get("error") or "")
            code = 404 if err == "session_not_found" else 400
            return JSONResponse(out, status_code=code)
        return JSONResponse(out)
    except Exception as e:
        import traceback as _tb
        return JSONResponse(
            {"error": str(e), "traceback": _tb.format_exc()},
            status_code=500,
        )


@app.post("/api/pedigree/batch-race-ensure-5gen", response_class=JSONResponse)
async def api_pedigree_batch_race_ensure_5gen(request: Request):
    """
    開催日範囲（YYYYMMDD）の race_lists に載る JRA 各レースについて
    race-ensure-5gen と同じ start_race_pedigree_prefetch を順に実行（queue-status 一括用）。
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    df = str(body.get("date_from") or "").strip()
    dt = str(body.get("date_to") or "").strip()
    dry_run = bool(body.get("dry_run"))
    try:
        max_races = int(body.get("max_races") or 500)
    except (TypeError, ValueError):
        max_races = 500
    if not df or not dt:
        return JSONResponse(
            {"error": "date_from と date_to（YYYYMMDD）が必要です"},
            status_code=400,
        )
    try:
        from src.research.pedigree.race_pedigree_5gen_prefetch import batch_race_pedigree_5gen_date_range

        storage = _get_storage()
        out = await asyncio.to_thread(
            batch_race_pedigree_5gen_date_range,
            storage,
            df,
            dt,
            dry_run=dry_run,
            max_races=max_races,
        )
        if not out.get("ok"):
            return JSONResponse(out, status_code=400)
        if not dry_run:
            _kick_urgent_worker()
            _kick_scrape_queue_worker()
        return JSONResponse(out)
    except Exception as e:
        import traceback as _tb
        return JSONResponse(
            {"error": str(e), "traceback": _tb.format_exc()},
            status_code=500,
        )


# ══════════════════════════════════════════════════════
#  血統 × 距離適性 研究ページ
# ══════════════════════════════════════════════════════


_bloodline_job: dict[str, Any] = {"running": False}


@app.get("/bloodline", response_class=HTMLResponse)
async def bloodline_page(request: Request):
    return templates.TemplateResponse("pedigree/bloodline.html", {
        "request": request,
        "current_page": "bloodline",
        "breadcrumbs": [],
    })


@app.post("/api/bloodline/analyze", response_class=JSONResponse)
async def api_bloodline_analyze(request: Request, background_tasks: BackgroundTasks):
    """血統分析をバックグラウンドで実行する。"""
    if _bloodline_job.get("running"):
        return JSONResponse({
            "status": "already_running",
            "started_at": _bloodline_job.get("started_at", ""),
        })

    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    years = body.get("years")
    source = body.get("source", "gcs")

    _bloodline_job["running"] = True
    _bloodline_job["started_at"] = datetime.now().isoformat()
    _bloodline_job["result"] = None
    _bloodline_job["error"] = None

    background_tasks.add_task(_run_bloodline_analysis, years, source)

    return JSONResponse({"status": "started", "started_at": _bloodline_job["started_at"]})


def _run_bloodline_analysis(years: list[str] | None, source: str):
    try:
        from src.research.pedigree.bloodline_distance import BloodlineDistanceAnalyzer

        _base = os.path.join(os.path.dirname(__file__), "..", "..")
        _out = os.path.join(_base, "data", "research", "bloodline")
        analyzer = BloodlineDistanceAnalyzer(output_dir=_out)

        if source == "csv":
            analyzer.load_from_csv(os.path.join(_base, "data", "features"))
        else:
            analyzer.load_from_gcs(years=years)

        if analyzer.df.empty:
            _bloodline_job["error"] = "データが空です"
            return

        analyzer.generate_report()

        df = analyzer.df
        _bloodline_job["result"] = {
            "n_records": len(df),
            "n_sires": int(df["sire"].nunique()),
            "n_dam_sires": int(df["dam_sire"].nunique()),
            "report_path": "data/research/bloodline/bloodline_distance_report.html",
        }
    except Exception as e:
        _bloodline_job["error"] = str(e)
        logger.error("血統分析失敗: %s", e, exc_info=True)
    finally:
        _bloodline_job["running"] = False
        _bloodline_job["finished_at"] = datetime.now().isoformat()


@app.get("/api/bloodline/status", response_class=JSONResponse)
async def api_bloodline_status():
    return JSONResponse({
        "running": _bloodline_job.get("running", False),
        "started_at": _bloodline_job.get("started_at", ""),
        "finished_at": _bloodline_job.get("finished_at", ""),
        "result": _bloodline_job.get("result"),
        "error": _bloodline_job.get("error"),
    })


def _bloodline_surface_dir(base: "Path", surface: str) -> "Path":
    from src.research.pedigree.bloodline_surface import (
        normalize_surface_key,
        surface_output_dir,
    )

    sk = normalize_surface_key(surface)
    if (base / "by_surface" / "_meta.json").exists():
        return surface_output_dir(base, sk)
    return base


def _bloodline_surface_context(base: "Path", surface: str) -> dict[str, Any]:
    from src.research.pedigree.bloodline_surface import (
        SURFACE_LABELS,
        load_surface_meta,
        normalize_surface_key,
        run_count_for_surface,
    )

    sk = normalize_surface_key(surface)
    meta = load_surface_meta(base)
    return {
        "surface": sk,
        "surface_label": SURFACE_LABELS[sk],
        "run_count": run_count_for_surface(meta, sk),
        "surfaces": meta.get("surfaces"),
    }


@app.get("/api/bloodline/surfaces", response_class=JSONResponse)
async def api_bloodline_surfaces():
    from src.config.data_paths import BLOODLINE_DIR, COURSE_BLOODLINE_DIR
    from src.research.pedigree.bloodline_surface import load_surface_meta

    meta = load_surface_meta(BLOODLINE_DIR)
    if not meta:
        meta = load_surface_meta(COURSE_BLOODLINE_DIR)
    return JSONResponse(meta or {"surfaces": {}, "total_run_count": 0})


@app.get("/api/bloodline/data/{analysis_type}", response_class=JSONResponse)
async def api_bloodline_data(
    analysis_type: str,
    surface: str = Query("turf", description="turf|dirt|jump"),
):
    """分析結果 CSV/JSON を読み込んで返す（芝・ダート・障害は surface で分離）。"""
    import csv as csv_mod
    from pathlib import Path

    def _csv_table(p: Path) -> dict[str, Any]:
        rows: list[dict[str, str]] = []
        if not p.exists():
            return {"columns": [], "rows": []}
        with open(p, encoding="utf-8-sig") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                rows.append(row)
        return {"columns": list(rows[0].keys()) if rows else [], "rows": rows}

    from src.config.data_paths import BLOODLINE_DIR
    base = BLOODLINE_DIR
    data_dir = _bloodline_surface_dir(base, surface)
    ctx = _bloodline_surface_context(base, surface)

    if analysis_type == "sire_distance":
        top3 = data_dir / "sire_distance_top3rate.csv"
        if top3.exists():
            return JSONResponse({
                "format": "metric_bundle",
                "min_cell_samples": 20,
                "min_roi_samples": 10,
                **ctx,
                "metrics": {
                    "top3_rate": _csv_table(top3),
                    "top2_rate": _csv_table(data_dir / "sire_distance_top2rate.csv"),
                    "win_rate": _csv_table(data_dir / "sire_distance_win_rate.csv"),
                    "sample_count": _csv_table(data_dir / "sire_distance_sample_count.csv"),
                    "win_roi": _csv_table(data_dir / "sire_distance_win_roi.csv"),
                    "place_roi": _csv_table(data_dir / "sire_distance_place_roi.csv"),
                    "roi_count": _csv_table(data_dir / "sire_distance_roi_count.csv"),
                },
            })

    if analysis_type == "sire_damsire_rate":
        top3 = data_dir / "sire_damsire_top3rate.csv"
        if top3.exists():
            return JSONResponse({
                "format": "metric_bundle",
                "min_cell_samples": 10,
                "min_roi_samples": 8,
                **ctx,
                "metrics": {
                    "top3_rate": _csv_table(top3),
                    "top2_rate": _csv_table(data_dir / "sire_damsire_top2rate.csv"),
                    "win_rate": _csv_table(data_dir / "sire_damsire_win_rate.csv"),
                    "sample_count": _csv_table(data_dir / "sire_damsire_sample_count.csv"),
                    "win_roi": _csv_table(data_dir / "sire_damsire_win_roi.csv"),
                    "place_roi": _csv_table(data_dir / "sire_damsire_place_roi.csv"),
                    "roi_count": _csv_table(data_dir / "sire_damsire_roi_count.csv"),
                },
            })

    file_map = {
        "sire_distance": data_dir / "sire_distance_top3rate.csv",
        "best_distance": data_dir / "sire_best_distance.csv",
        "sire_damsire_rate": data_dir / "sire_damsire_top3rate.csv",
        "sire_damsire_dist": data_dir / "sire_damsire_avg_distance.csv",
        "similarity": data_dir / "distance_bloodline_similarity.csv",
        "clusters": data_dir / "sire_clusters.csv",
        "cluster_summary": data_dir / "cluster_summary.csv",
        "predictive_power": data_dir / "pedigree_predictive_power.json",
    }

    path = file_map.get(analysis_type)
    if not path or not path.exists():
        return JSONResponse(
            {"error": f"データなし: {analysis_type} (surface={ctx['surface']})"},
            status_code=404,
        )

    if path.suffix == ".json":
        import json as json_mod
        payload = json_mod.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and analysis_type == "predictive_power":
            return JSONResponse({**ctx, **payload})
        return JSONResponse(payload)

    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
    return JSONResponse({
        **ctx,
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
    })


# ══════════════════════════════════════════════════════
#  血統メタクラスタリング（馬名 → クラスタ → 適性プロファイル）
#  notebooks/pedigree/bloodline_subgroup_analysis.ipynb §21〜§22 のWeb版
# ══════════════════════════════════════════════════════


@app.get("/bloodline-cluster", response_class=HTMLResponse)
async def bloodline_cluster_page(request: Request):
    return templates.TemplateResponse("pedigree/bloodline_cluster.html", {
        "request": request,
        "current_page": "bloodline_cluster",
        "breadcrumbs": [],
    })


@app.get("/api/bloodline-cluster/meta", response_class=JSONResponse)
async def api_bloodline_cluster_meta():
    """アーティファクトのメタ情報 (生成日時、エンティティ数、L2 数等)。"""
    from src.api.bloodline_meta_cluster import get_meta, is_ready
    if not is_ready():
        return JSONResponse(
            {
                "error": "アーティファクト未生成",
                "hint": "python -m src.research.pedigree.build_meta_cluster_artifacts を実行してください",
            },
            status_code=503,
        )
    return JSONResponse(get_meta())


@app.get("/api/bloodline-cluster/lookup", response_class=JSONResponse)
async def api_bloodline_cluster_lookup(name: str = Query(..., min_length=1, description="馬名 (部分一致対応)")):
    """馬名 → 主流 + L1_sub + L2 + 強み・弱みプロファイル。"""
    from src.api.bloodline_meta_cluster import analyze_horse
    return JSONResponse(analyze_horse(name))


@app.get("/api/bloodline-cluster/lookup-by-id", response_class=JSONResponse)
async def api_bloodline_cluster_lookup_id(horse_id: str = Query(..., description="netkeiba horse_id (例: 2015104961)")):
    """horse_id 直接指定で適性プロファイルを取得 (race テーブルに無い馬でも 5 代血統があれば動作)。"""
    from src.api.bloodline_meta_cluster import lookup_by_horse_id
    return JSONResponse(lookup_by_horse_id(horse_id))


@app.get("/api/bloodline-cluster/sire-info", response_class=JSONResponse)
async def api_bloodline_cluster_sire_info(stallion_id: str = Query(..., description="種牡馬 horse_id 自身")):
    """種牡馬 ID 自身を入力 → その種牡馬のクラスタ分類 + プロファイル + 特殊適性タグ。

    /pedigree-map のツリーから馬名 (=種牡馬) クリックされた際の遷移先で利用。
    """
    from src.api.bloodline_meta_cluster import analyze_stallion
    return JSONResponse(analyze_stallion(stallion_id))


@app.get("/api/bloodline-cluster/clusters", response_class=JSONResponse)
async def api_bloodline_cluster_clusters():
    """全 L2 メタクラスタの構成 (主流別) + プロファイル top7 強み/弱み。"""
    from src.api.bloodline_meta_cluster import list_clusters
    return JSONResponse(_scrub_nan({"clusters": list_clusters()}))


@app.get("/api/bloodline-cluster/suggest", response_class=JSONResponse)
async def api_bloodline_cluster_suggest(prefix: str = Query("", description="馬名の先頭/部分文字列"),
                                        limit: int = Query(15, ge=1, le=50)):
    """馬名オートサジェスト用。"""
    from src.api.bloodline_meta_cluster import list_horses
    return JSONResponse({"results": list_horses(prefix, limit=limit)})


@app.post("/api/bloodline-cluster/reload", response_class=JSONResponse)
async def api_bloodline_cluster_reload():
    """アーティファクトを再ロード (ノートブック再実行後の反映用)。"""
    from src.api.bloodline_meta_cluster import reload_artifacts, get_meta
    ok = reload_artifacts()
    return JSONResponse({"reloaded": ok, "meta": get_meta() if ok else None})


# ══════════════════════════════════════════════════════
#  血統メタクラスタリング: 開発者向け管理エンドポイント
#  ──────────────────────────────────────────────────────
#  アーティファクトの再生成 / リロード / 状態確認。
#  /api/admin/ 配下なので開発者ログイン必須 (auth.py の DEV_ONLY_API_PREFIXES)。
# ══════════════════════════════════════════════════════

# 各アーティファクトの定義 (UI 表示・rebuild 対象判定)
_BLOODLINE_ARTIFACTS = [
    # ── 順序: 軽い→重い (UI で上から順に並ぶ) ──
    {
        "key": "pair_lift",
        "label": "父×母父系 ペア lift プロファイル",
        "files": [
            "pair_lift_profiles_bms_root.parquet",
            "pair_lift_profiles_group.parquet",
            "pair_lift_profiles_gxg.parquet",
            "pair_lift_meta.json",
        ],
        "rebuilder": "src.research.pedigree.build_pair_lift_profiles",
        "estimated_seconds": 5,
        "description": (
            "父個体×母父系 (母父父系root, 227系統) / 父個体×母父5大系統 / 父5大系統×母父5大系統"
            " の 3 階層ペア lift。race_result_slim.parquet と horse_bms.parquet から集計。"
            " 統合 prior の pair_bms_root / pair_group / pair_gxg レイヤーで使用。"
            " 旧 pair_indiv (父個体×母父個体, n>=80) は件数不足のため 2026-05-15 に廃止。"
        ),
    },
    {
        "key": "role_lift",
        "label": "役割別 (F/MF/FF/MMF) lift プロファイル",
        "files": [
            "role_lift_profiles.parquet",
            "role_lift_meta.json",
            "role_lift_main_group_fallback.json",
        ],
        "rebuilder": "src.research.pedigree.build_role_lift_profiles",
        "estimated_seconds": 60,
        "description": (
            "F (父) / MF (母父) / FF (父父) / MMF (母母父) の役割別 lift。"
            " 5 代血統の祖先位置から該当する種牡馬 ID を取得し、産駒の条件別勝率を集計。"
            " 統合 prior の role_blend レイヤーで使用 (新着馬や個別ペアが薄い時の fallback)。"
        ),
    },
    {
        "key": "l2_names",
        "label": "L2 クラスタ自動命名 (l2_names.json)",
        "files": [
            "l2_names.json",
        ],
        "rebuilder": "src.research.pedigree.generate_l2_fine_names",
        "estimated_seconds": 10,
        "description": (
            "L2_fine クラスタの自動命名 (例「東京芝中距離主流型」) と説明文・色・アイコン・"
            "得意/苦手タグ・代表種牡馬を生成。l2_profiles.json の更新後に再生成すると、"
            "UI ラベルが新しいクラスタ特性に追従する。"
        ),
    },
    {
        "key": "ancestor_l2_index",
        "label": "祖先 → L2 マッピング (ancestor_to_l2 / ancestor_vectors / ancestor_positions_2d)",
        "files": [
            "ancestor_to_l2.json",
            "ancestor_vectors.parquet",
            "ancestor_positions_2d.json",
        ],
        "rebuilder": "src.research.pedigree.build_ancestor_l2_index",
        "estimated_seconds": 90,
        "description": (
            "拡張祖先 (主流133頭 + 約4500頭) を L2 クラスタにマッピングし、各祖先の"
            "条件別 lift ベクトルと 2D 位置を計算。馬の血統適性プロファイル合成で利用。"
        ),
    },
    {
        "key": "l2_fine_clusters",
        "label": "L2 fine クラスタリング (l2_profiles / l2_centroids / l2_super_groups / l2_similarity)",
        "files": [
            "l2_profiles.json",
            "l2_centroids.json",
            "l2_super_groups.json",
            "l2_similarity.json",
            "l2_positions_2d.json",
        ],
        "rebuilder": "src.research.pedigree.build_l2_fine_clusters",
        "estimated_seconds": 60,
        "description": (
            "133 主流種牡馬を 31 次元の条件別 lift で再クラスタリング (Ward 法)。"
            " L2 cluster (=11 個前後) と SuperGroup (=5 個) を再構築し、各クラスタの平均プロファイル・"
            "中心点・類似度行列・2D 座標 (PCA) を保存。L2 構造の再計算はここから。"
        ),
    },
    {
        "key": "meta_cluster_base",
        "label": "メタクラスタ基盤 (notebook 由来: unified / overall / sid_to_* / scaler / knn / meta)",
        "files": [
            "unified.parquet",
            "overall_raw.json",
            "sid_to_main.json",
            "sid_to_name.json",
            "scaler.json",
            "knn_data.json",
            "meta.json",
            "non_main_root_of.json",
            "non_main_group_labels.json",
        ],
        "rebuilder": "src.research.pedigree.build_meta_cluster_artifacts",
        "estimated_seconds": 600,
        "description": (
            "全パイプラインの「基盤」となるアーティファクト群 (notebook §21〜§22 を headless 実行)。"
            " 種牡馬統合データ (unified)、全体勝率 (overall_raw)、ID→名前/系統マップ、"
            " 標準化スケーラ、KNN データ、メタ情報を一括生成する。最も重い (約 10 分)。"
            " 通常はレース結果データに大きな更新があった時のみ実行。"
        ),
    },
]

_bl_rebuild_job: dict[str, Any] = {
    "running": False,
    "target": None,
    "started_at": None,
    "finished_at": None,
    "duration_sec": None,
    "result": None,
    "error": None,
    "log": [],
}


def _get_bl_artifact_status() -> list[dict[str, Any]]:
    """各アーティファクトの存在 / 最終更新日時 / サイズを返す。"""
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    art_dir = _Path("data/page_reference/note_aptitude_race")
    out = []
    for spec in _BLOODLINE_ARTIFACTS:
        files_info = []
        for fn in spec["files"]:
            p = art_dir / fn
            if p.exists():
                st = p.stat()
                files_info.append({
                    "name": fn,
                    "exists": True,
                    "size_bytes": int(st.st_size),
                    "mtime_iso": _dt.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                })
            else:
                files_info.append({"name": fn, "exists": False})
        all_exist = all(f["exists"] for f in files_info)
        latest_mtime = max((f.get("mtime_iso", "") for f in files_info if f["exists"]), default="")
        out.append({
            "key": spec["key"],
            "label": spec["label"],
            "description": spec["description"],
            "estimated_seconds": spec["estimated_seconds"],
            "rebuilder_available": spec["rebuilder"] is not None,
            "files": files_info,
            "all_files_exist": all_exist,
            "latest_mtime_iso": latest_mtime,
        })
    return out


@app.get("/api/admin/bloodline-cluster/artifact-status", response_class=JSONResponse)
async def api_admin_bl_artifact_status():
    """全アーティファクトの状態 (存在・最終更新日時・サイズ・説明)。

    開発者専用 (/api/admin/ プレフィックス)。
    """
    return JSONResponse({
        "artifacts": _get_bl_artifact_status(),
        "current_job": _bl_rebuild_job,
    })


def _run_bl_rebuild(target_key: str, rebuilder_module: str):
    """バックグラウンドでアーティファクト再生成 → キャッシュリロード。"""
    import importlib
    from datetime import datetime as _dt
    import time as _time
    t0 = _time.time()
    _bl_rebuild_job["log"] = [f"[{_dt.now().isoformat(timespec='seconds')}] start: {target_key} ({rebuilder_module})"]
    try:
        mod = importlib.import_module(rebuilder_module)
        if hasattr(mod, "build"):
            mod.build()
        elif hasattr(mod, "main"):
            mod.main()
        else:
            raise RuntimeError(f"{rebuilder_module} に build() / main() 関数が見つかりません")
        # 生成後はキャッシュリロード
        from src.api.bloodline_meta_cluster import reload_artifacts
        reload_ok = reload_artifacts()
        _bl_rebuild_job["log"].append(f"reload_artifacts -> {reload_ok}")
        _bl_rebuild_job["result"] = {"target": target_key, "reload": reload_ok}
        _bl_rebuild_job["error"] = None
    except Exception as e:
        _bl_rebuild_job["error"] = f"{type(e).__name__}: {e}"
        _bl_rebuild_job["log"].append(f"ERROR: {_bl_rebuild_job['error']}")
        logger.error("bloodline rebuild [%s] failed: %s", target_key, e, exc_info=True)
    finally:
        _bl_rebuild_job["running"] = False
        _bl_rebuild_job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _bl_rebuild_job["duration_sec"] = round(_time.time() - t0, 2)


@app.post("/api/admin/bloodline-cluster/rebuild/{target}", response_class=JSONResponse)
async def api_admin_bl_rebuild(target: str, background_tasks: BackgroundTasks):
    """指定アーティファクトを再生成 (バックグラウンド)。

    target: アーティファクト key (例: pair_lift / role_lift)
    開発者専用 (/api/admin/ プレフィックス)。
    """
    spec = next((s for s in _BLOODLINE_ARTIFACTS if s["key"] == target), None)
    if not spec:
        return JSONResponse(
            {"status": "error", "message": f"未知の target: {target}"},
            status_code=400,
        )
    if not spec["rebuilder"]:
        return JSONResponse(
            {
                "status": "no_rebuilder",
                "message": f"{spec['label']} はノートブック側で生成する必要があります。",
                "hint": "notebooks/pedigree/bloodline_subgroup_analysis.ipynb の §21〜§22 を実行してください。",
            },
            status_code=400,
        )
    if _bl_rebuild_job.get("running"):
        return JSONResponse(
            {
                "status": "already_running",
                "current_target": _bl_rebuild_job.get("target"),
                "started_at": _bl_rebuild_job.get("started_at"),
            },
            status_code=409,
        )
    _bl_rebuild_job.update({
        "running": True,
        "target": target,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "duration_sec": None,
        "result": None,
        "error": None,
        "log": [f"queued: {target}"],
    })
    background_tasks.add_task(_run_bl_rebuild, target, spec["rebuilder"])
    return JSONResponse({
        "status": "started",
        "target": target,
        "estimated_seconds": spec["estimated_seconds"],
        "started_at": _bl_rebuild_job["started_at"],
    })


@app.get("/api/admin/bloodline-cluster/job-status", response_class=JSONResponse)
async def api_admin_bl_job_status():
    """直近の rebuild ジョブのステータス。"""
    return JSONResponse(_bl_rebuild_job)


@app.post("/api/admin/bloodline-cluster/reload", response_class=JSONResponse)
async def api_admin_bl_reload():
    """キャッシュリロード (開発者向け、admin 配下)。"""
    from src.api.bloodline_meta_cluster import reload_artifacts, get_meta
    ok = reload_artifacts()
    return JSONResponse({
        "reloaded": ok,
        "meta": get_meta() if ok else None,
        "artifacts": _get_bl_artifact_status() if ok else None,
    })


@app.get("/api/bloodline-cluster/tags", response_class=JSONResponse)
async def api_bloodline_cluster_tags(
    tag_id: Optional[str] = Query(None, description="特定タグID (例: track_heavy / runstyle_nige)"),
    category: Optional[str] = Query(None, description="カテゴリ: 脚質 / 馬場 / ペース / 距離 / コース"),
    top_n: int = Query(30, ge=1, le=200, description="各タグの上位N (combo_score 降順)"),
):
    """特殊適性タグの一覧 + 各タグの上位種牡馬を返す。

    タグ判定は L2 メタクラスタ (総合プロファイル) と直交する。
    "is_top20" は「絶対勝率 >= 条件平均×1.10」 AND 「lift >= 1.15」 AND 「combo Top20%」。
    """
    from src.api.bloodline_meta_cluster import list_special_tags
    return JSONResponse(list_special_tags(tag_id=tag_id, category=category, top_n=top_n))


@app.get("/api/bloodline-cluster/stats", response_class=JSONResponse)
async def api_bloodline_cluster_stats(
    target_kind: str = Query("sire", description="sire | L2 | L1_sub | main | all"),
    target_id: Optional[str] = Query(None, description="種牡馬ID / L2番号 / L1_sub番号 / 主流名"),
    main_group: Optional[str] = Query(None, description="L1_sub 指定時の主流名"),
    date_from: Optional[str] = Query(None, description="期間開始 (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="期間終了 (YYYY-MM-DD)"),
    venues: Optional[str] = Query(None, description="競馬場 (カンマ区切り 例: 東京,中山)"),
    surfaces: Optional[str] = Query(None, description="路面 (カンマ区切り 例: 芝,ダート)"),
    dist_min: Optional[float] = Query(None, description="距離下限 (m)"),
    dist_max: Optional[float] = Query(None, description="距離上限 (m)"),
    track_conditions: Optional[str] = Query(None, description="馬場 (カンマ区切り 例: 良,稍重,重,不良)"),
    grades: Optional[str] = Query(None, description="グレード (カンマ区切り 例: G1,G2,G3,オープン)"),
    breakdown: Optional[str] = Query(None, description="内訳 sire | venue | surface | dist_cat | grade | L2 | L1_sub"),
    top_per_breakdown: int = Query(30, ge=1, le=200, description="内訳の上位N (n_records 降順)"),
):
    """期間 × 舞台条件 × 種牡馬/クラスタ で統計量 (勝率/連対/複勝/平均着/平均人気/単回/複回) を返す。

    例:
        - 種牡馬ロードカナロアの 2023-2025 東京芝:
          /api/bloodline-cluster/stats?target_kind=sire&target_id=2008103552&date_from=2023-01-01&date_to=2025-12-31&venues=東京&surfaces=芝
        - L2=2 全体の 2024 年:
          /api/bloodline-cluster/stats?target_kind=L2&target_id=2&date_from=2024-01-01
        - Native Dancer系 の L1_sub=1 を内訳=sire で:
          /api/bloodline-cluster/stats?target_kind=L1_sub&target_id=1&main_group=Native Dancer系&breakdown=sire
    """
    from src.api.bloodline_meta_cluster import compute_stats
    result = compute_stats(
        target_kind=target_kind,
        target_id=target_id,
        main_group=main_group,
        date_from=date_from,
        date_to=date_to,
        venues=venues,
        surfaces=surfaces,
        dist_min=dist_min,
        dist_max=dist_max,
        track_conditions=track_conditions,
        grades=grades,
        breakdown=breakdown,
        top_per_breakdown=top_per_breakdown,
    )
    return JSONResponse(result)


@app.get("/api/bloodline-cluster/sire-presence-stats", response_class=JSONResponse)
async def api_bloodline_cluster_sire_presence_stats(
    stallion_id: str = Query(..., description="種牡馬 horse_id"),
    date_from: Optional[str] = Query(None, description="期間開始 (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="期間終了 (YYYY-MM-DD)"),
    venues: Optional[str] = Query(None, description="競馬場 (カンマ区切り)"),
    surfaces: Optional[str] = Query(None, description="路面 (カンマ区切り)"),
    dist_min: Optional[float] = Query(None, description="距離下限 (m)"),
    dist_max: Optional[float] = Query(None, description="距離上限 (m)"),
    track_conditions: Optional[str] = Query(None, description="馬場 (カンマ区切り)"),
    grades: Optional[str] = Query(None, description="グレード (カンマ区切り)"),
):
    """指定種牡馬 X が血統 (直近10世代) のどの領域に登場するかで層別した成績を返す。

    レスポンス内訳:
        - ``sire``        : X = 父そのもの (1代直結) - 既存 sire 統計と一致
        - ``father``      : A = X が父系領域 (父・父父・父父父...) のどこかに登場 (10gen 以内)
        - ``mother``      : B = X が母系領域 (母父系 OR 母母系) のどこかに登場 (10gen 以内)
        - ``father_only`` / ``mother_only`` / ``both`` : A∩B 排他的分割
    """
    from src.api.bloodline_meta_cluster import compute_sire_presence_stats
    result = compute_sire_presence_stats(
        stallion_id=stallion_id,
        date_from=date_from,
        date_to=date_to,
        venues=venues,
        surfaces=surfaces,
        dist_min=dist_min,
        dist_max=dist_max,
        track_conditions=track_conditions,
        grades=grades,
    )
    return JSONResponse(result)


@app.get("/api/pedigree-map/condition-ranking", response_class=JSONResponse)
async def api_pedigree_map_condition_ranking(
    side: str = Query("sire", description="sire | bms | father | mother"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venues: Optional[str] = Query(None),
    surfaces: Optional[str] = Query(None),
    dist_min: Optional[float] = Query(None),
    dist_max: Optional[float] = Query(None),
    track_conditions: Optional[str] = Query(None),
    grades: Optional[str] = Query(None),
    pop_min: Optional[float] = Query(None),
    pop_max: Optional[float] = Query(None),
    odds_min: Optional[float] = Query(None, description="単勝オッズ下限"),
    odds_max: Optional[float] = Query(None, description="単勝オッズ上限"),
    min_n: int = Query(30, ge=1, le=10000),
    top: int = Query(30, ge=1, le=200),
    sort: str = Query("balanced", description="balanced | win_roi | place_roi | win_rate | place3_rate | n_records"),
):
    """馬券条件マッチ種牡馬ランキング (探索モードのメイン API)。"""
    from src.api.bloodline_meta_cluster import compute_condition_ranking
    return JSONResponse(compute_condition_ranking(
        side=side,
        date_from=date_from, date_to=date_to,
        venues=venues, surfaces=surfaces,
        dist_min=dist_min, dist_max=dist_max,
        track_conditions=track_conditions, grades=grades,
        pop_min=pop_min, pop_max=pop_max,
        odds_min=odds_min, odds_max=odds_max,
        min_n=min_n, top=top, sort=sort,
    ))


@app.get("/api/pedigree-map/progeny-under-condition", response_class=JSONResponse)
async def api_pedigree_map_progeny_under_condition(
    stallion_id: str = Query(..., description="種牡馬 / 祖先 horse_id"),
    side: str = Query("sire", description="sire | bms | father | mother"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venues: Optional[str] = Query(None),
    surfaces: Optional[str] = Query(None),
    dist_min: Optional[float] = Query(None),
    dist_max: Optional[float] = Query(None),
    track_conditions: Optional[str] = Query(None),
    grades: Optional[str] = Query(None),
    pop_min: Optional[float] = Query(None),
    pop_max: Optional[float] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    min_n: int = Query(3, ge=1, le=10000, description="馬ごとの最小出走数"),
    top: int = Query(12, ge=1, le=100),
    sort: str = Query("balanced"),
    sample_per_horse: int = Query(5, ge=0, le=30),
):
    """探索モード行展開: 該当条件における代表的な該当馬と各馬の条件下成績。"""
    from src.api.bloodline_meta_cluster import compute_progeny_under_condition
    return JSONResponse(compute_progeny_under_condition(
        stallion_id=stallion_id, side=side,
        date_from=date_from, date_to=date_to,
        venues=venues, surfaces=surfaces,
        dist_min=dist_min, dist_max=dist_max,
        track_conditions=track_conditions, grades=grades,
        pop_min=pop_min, pop_max=pop_max,
        odds_min=odds_min, odds_max=odds_max,
        min_n=min_n, top=top, sort=sort,
        sample_per_horse=sample_per_horse,
    ))


@app.get("/api/bloodline-cluster/sire-heatmap", response_class=JSONResponse)
async def api_bloodline_cluster_sire_heatmap(
    stallion_id: str = Query(...),
    axis_x: str = Query("venue"),
    axis_y: str = Query("dist_cat"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    surfaces: Optional[str] = Query(None),
    track_conditions: Optional[str] = Query(None),
    min_n_cell: int = Query(1, ge=0),
):
    """指定種牡馬 直結産駒の 2 軸ヒートマップ用集計。"""
    from src.api.bloodline_meta_cluster import compute_sire_heatmap
    return JSONResponse(compute_sire_heatmap(
        stallion_id=stallion_id,
        axis_x=axis_x, axis_y=axis_y,
        date_from=date_from, date_to=date_to,
        surfaces=surfaces, track_conditions=track_conditions,
        min_n_cell=min_n_cell,
    ))


@app.get("/api/bloodline-cluster/sire-summary-card", response_class=JSONResponse)
async def api_bloodline_cluster_sire_summary_card(
    stallion_id: str = Query(...),
    period: str = Query("all", description="all | 2y | 1y"),
):
    """サマリーカード用の集計 (上部ヘッダ用)。"""
    from src.api.bloodline_meta_cluster import compute_sire_summary_card
    return JSONResponse(compute_sire_summary_card(stallion_id=stallion_id, period=period))


@app.get("/api/bloodline-cluster/sire-best-conditions", response_class=JSONResponse)
async def api_bloodline_cluster_sire_best_conditions(
    stallion_id: str = Query(..., description="種牡馬 horse_id"),
    min_n: int = Query(30, ge=1, le=10000, description="最小出走数 (これ以下の条件は除外)"),
    top: int = Query(10, ge=1, le=100, description="各ランキングの上位 N"),
    date_from: Optional[str] = Query(None, description="期間開始 (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="期間終了 (YYYY-MM-DD)"),
):
    """指定種牡馬直結産駒 (sire) のベスト条件ランキング (複合条件 + 単軸別)。

    返却:
      - rankings.combo: 場×路面×距離区分 (勝率/複勝率/単回収/複回収の上位 N)
      - rankings.combo_full: 場×路面×距離×馬場×グレード (単回収/複回収の上位 N)
      - rankings.by_venue / by_surface / by_dist_cat / by_track_condition / by_grade: 単軸別の全行
    """
    from src.api.bloodline_meta_cluster import compute_sire_best_conditions
    result = await asyncio.to_thread(
        compute_sire_best_conditions,
        stallion_id=stallion_id,
        min_n=min_n,
        top=top,
        date_from=date_from,
        date_to=date_to,
    )
    return JSONResponse(result)


@app.get("/api/bloodline-cluster/sire-presence-horses", response_class=JSONResponse)
async def api_bloodline_cluster_sire_presence_horses(
    stallion_id: str = Query(..., description="種牡馬 horse_id"),
    side: str = Query("father", description="father (A=父系領域) | mother (B=母系領域) | sire (父=本馬) | both (両方)"),
    limit: int = Query(10, ge=0, le=2000, description="返却件数 (0=全件)"),
    offset: int = Query(0, ge=0, description="開始位置"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venues: Optional[str] = Query(None),
    surfaces: Optional[str] = Query(None),
    dist_min: Optional[float] = Query(None),
    dist_max: Optional[float] = Query(None),
    track_conditions: Optional[str] = Query(None),
    grades: Optional[str] = Query(None),
):
    """指定種牡馬を A/B/sire/both で含む該当馬一覧を、推定獲得賞金の多い順で返す。"""
    from src.api.bloodline_meta_cluster import compute_sire_presence_horses
    result = await asyncio.to_thread(
        compute_sire_presence_horses,
        stallion_id,
        side,
        limit=limit,
        offset=offset,
        date_from=date_from,
        date_to=date_to,
        venues=venues,
        surfaces=surfaces,
        dist_min=dist_min,
        dist_max=dist_max,
        track_conditions=track_conditions,
        grades=grades,
    )
    return JSONResponse(result)


# ── 競走馬の血統ベース適性プロファイル (Phase 1: 馬名 → L2_fine 所属度) ──
_horse_apt_calc = None


def _get_horse_apt_calc():
    """HorseAptitudeProfileCalc のシングルトン取得 (遅延初期化)。"""
    global _horse_apt_calc
    if _horse_apt_calc is None:
        from src.research.pedigree.horse_aptitude_profile import (
            HorseAptitudeProfileCalc,
        )
        _horse_apt_calc = HorseAptitudeProfileCalc()
    return _horse_apt_calc


@app.get("/api/bloodline-cluster/horse-aptitude", response_class=JSONResponse)
async def api_bloodline_cluster_horse_aptitude(
    horse_id: Optional[str] = Query(None, description="競走馬 horse_id (優先)"),
    horse_name: Optional[str] = Query(None, description="競走馬名"),
):
    """指定された競走馬の **血統ベース適性プロファイル** を返す。

    - 10 世代以内の祖先 (主流 133 種牡馬 + 拡張 ~4640 祖先) の L2_fine 所属を
      位置別重み付き集計し、11 個の L2 クラスタへの所属度ベクトルを算出。
    - 実成績 (オッズ < 30 倍のレコードのみ) からのベスト条件もシグナルとして付与。

    Returns: HorseAptitudeProfileCalc.compute() の dict (フォーマットはモジュールの
             docstring を参照)。
    """
    if not horse_id and not horse_name:
        return JSONResponse({"error": "horse_id or horse_name required"}, status_code=400)
    try:
        calc = _get_horse_apt_calc()
        result = calc.compute(horse_id=horse_id, horse_name=horse_name)
    except FileNotFoundError as e:
        return JSONResponse({"error": f"artifacts not found: {e}"}, status_code=500)
    if "error" in result:
        return JSONResponse(result, status_code=404)
    return JSONResponse(result)


@app.get("/api/bloodline-cluster/horse-name-suggest", response_class=JSONResponse)
async def api_bloodline_cluster_horse_name_suggest(
    q: str = Query(..., min_length=1, description="馬名検索クエリ (先頭一致)"),
    limit: int = Query(15, ge=1, le=50),
):
    """馬名のサジェスト (先頭一致) を返す。

    Returns: ``{"items": [{"horse_id": "...", "horse_name": "...", "sex": "牡", "n_races": 17, "last_date": "..."}]}``
    """
    try:
        calc = _get_horse_apt_calc()
    except FileNotFoundError as e:
        return JSONResponse({"items": [], "error": str(e)}, status_code=500)
    q_norm = q.strip()
    if not q_norm:
        return JSONResponse({"items": []})
    idx = calc.idx
    mask = idx["horse_name"].str.startswith(q_norm, na=False)
    if not mask.any():
        # フォールバック: 部分一致
        mask = idx["horse_name"].str.contains(q_norm, na=False, regex=False)
    hits = idx[mask].sort_values(["last_date"], ascending=False).head(limit)
    items = [
        {
            "horse_id": str(r["horse_id"]),
            "horse_name": str(r["horse_name"]),
            "sex": str(r.get("sex", "?")),
            "n_races": int(r.get("n_races", 0)),
            "last_date": str(r.get("last_date", "")),
        }
        for _, r in hits.iterrows()
    ]
    return JSONResponse({"items": items, "q": q_norm})


# ══════════════════════════════════════════════════════
#  コース特性 × 血統適性 研究ページ
# ══════════════════════════════════════════════════════

_course_bl_job: dict[str, Any] = {"running": False}


@app.get("/course-bloodline", include_in_schema=False)
async def course_bloodline_page_redirect():
    """旧 /course-bloodline ページは /bloodline (統合 Viewer) にリダイレクト。

    「コース特性 × 血統適性」と「血統 × 距離適性 研究」は 1 ページに統合された。
    既存 API (/api/course-bloodline/data/*, /api/course-profiles) はそのまま残置。
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/bloodline", status_code=308)


@app.get("/api/course-profiles", response_class=JSONResponse)
async def api_course_profiles():
    """コースプロファイル (ドメインナレッジ) を返す。"""
    import json as json_mod
    from src.config.data_paths import COURSE_BLOODLINE_DIR, COURSE_PROFILES_JSON
    p = COURSE_PROFILES_JSON
    if not p.exists():
        return JSONResponse({"error": "course_profiles.json が見つかりません"}, status_code=404)
    payload = json_mod.loads(p.read_text(encoding="utf-8"))
    hl = COURSE_BLOODLINE_DIR / "profile_course_roi_highlights.json"
    if hl.exists():
        try:
            payload["roi_highlights"] = json_mod.loads(hl.read_text(encoding="utf-8"))
        except Exception:
            pass
    return JSONResponse(payload)


@app.post("/api/course-bloodline/analyze", response_class=JSONResponse)
async def api_course_bloodline_analyze(request: Request, background_tasks: BackgroundTasks):
    if _course_bl_job.get("running"):
        return JSONResponse({
            "status": "already_running",
            "started_at": _course_bl_job.get("started_at", ""),
        })

    body = await request.json()
    years = body.get("years")

    _course_bl_job["running"] = True
    _course_bl_job["started_at"] = datetime.now().isoformat()
    _course_bl_job["result"] = None
    _course_bl_job["error"] = None

    background_tasks.add_task(_run_course_bl_analysis, years)
    return JSONResponse({"status": "started", "started_at": _course_bl_job["started_at"]})


def _run_course_bl_analysis(years: list[str] | None):
    try:
        from src.research.race.course_bloodline import CourseBloodlineAnalyzer
        _base = os.path.join(os.path.dirname(__file__), "..", "..")
        _out = os.path.join(_base, "data", "research", "course_bloodline")

        analyzer = CourseBloodlineAnalyzer(output_dir=_out)
        analyzer.load_from_gcs(years=years)

        if analyzer.df.empty:
            _course_bl_job["error"] = "データが空です"
            return

        analyzer.generate_report()
        df = analyzer.df
        result = {
            "n_records": len(df),
            "n_sires": int(df["sire"].nunique()),
            "n_venues": int(df["venue"].nunique()),
        }
        if "grass_type_est" in df.columns:
            result["grass_type_counts"] = df["grass_type_est"].value_counts().to_dict()
        if "draw_zone" in df.columns:
            result["draw_zone_counts"] = df["draw_zone"].value_counts().to_dict()
        if "fc_band" in df.columns:
            result["fc_band_counts"] = df["fc_band"].value_counts().to_dict()
        _course_bl_job["result"] = result
    except Exception as e:
        _course_bl_job["error"] = str(e)
        logger.error("コース×血統分析失敗: %s", e, exc_info=True)
    finally:
        _course_bl_job["running"] = False
        _course_bl_job["finished_at"] = datetime.now().isoformat()


@app.get("/api/course-bloodline/status", response_class=JSONResponse)
async def api_course_bl_status():
    return JSONResponse({
        "running": _course_bl_job.get("running", False),
        "started_at": _course_bl_job.get("started_at", ""),
        "finished_at": _course_bl_job.get("finished_at", ""),
        "result": _course_bl_job.get("result"),
        "error": _course_bl_job.get("error"),
    })


@app.get("/api/course-bloodline/surfaces", response_class=JSONResponse)
async def api_course_bl_surfaces():
    from src.config.data_paths import COURSE_BLOODLINE_DIR
    from src.research.pedigree.bloodline_surface import load_surface_meta

    return JSONResponse(load_surface_meta(COURSE_BLOODLINE_DIR) or {"surfaces": {}, "total_run_count": 0})


@app.get("/api/course-bloodline/data/{analysis_type}", response_class=JSONResponse)
async def api_course_bl_data(
    analysis_type: str,
    surface: str = Query("turf", description="turf|dirt|jump"),
):
    import csv as csv_mod
    from pathlib import Path as _P

    def _csv_tab(p: _P) -> dict[str, Any]:
        rows: list[dict[str, str]] = []
        if not p.exists():
            return {"columns": [], "rows": []}
        with open(p, encoding="utf-8-sig") as f:
            r = csv_mod.DictReader(f)
            for row in r:
                rows.append(row)
        return {"columns": list(rows[0].keys()) if rows else [], "rows": rows}

    from src.config.data_paths import COURSE_BLOODLINE_DIR
    base = COURSE_BLOODLINE_DIR
    ctx = _bloodline_surface_context(base, surface)

    if analysis_type == "profiles":
        path = base / "course_profiles_summary.csv"
        if not path.exists():
            return JSONResponse({"error": "データなし: profiles"}, status_code=404)
        rows = []
        with open(path, encoding="utf-8-sig") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                rows.append(row)
        return JSONResponse({
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
        })

    data_dir = _bloodline_surface_dir(base, surface)

    if analysis_type == "venue_sire":
        top3 = data_dir / "venue_sire_top3rate.csv"
        if top3.exists():
            return JSONResponse({
                "format": "metric_bundle",
                "min_cell_samples": 15,
                "min_roi_samples": 8,
                **ctx,
                "metrics": {
                    "top3_rate": _csv_tab(top3),
                    "top2_rate": _csv_tab(data_dir / "venue_sire_top2rate.csv"),
                    "win_rate": _csv_tab(data_dir / "venue_sire_win_rate.csv"),
                    "sample_count": _csv_tab(data_dir / "venue_sire_sample_count.csv"),
                    "win_roi": _csv_tab(data_dir / "venue_sire_win_roi.csv"),
                    "place_roi": _csv_tab(data_dir / "venue_sire_place_roi.csv"),
                    "roi_count": _csv_tab(data_dir / "venue_sire_roi_count.csv"),
                },
            })

    if analysis_type == "grass_summary" and surface != "turf":
        return JSONResponse({
            **ctx,
            "columns": [],
            "rows": [],
            "note": "芝種別分析は芝レースのみ対象です",
        })

    file_map = {
        "venue_sire": data_dir / "venue_sire_top3rate.csv",
        "trait_correlation": data_dir / "sire_trait_correlation.csv",
        "aptitude": data_dir / "sire_course_aptitude.csv",
        "optimal": data_dir / "sire_optimal_conditions.csv",
        "track_condition": data_dir / "track_condition_interaction.csv",
        "draw_bloodline": data_dir / "draw_bloodline_interaction.csv",
        "draw_summary": data_dir / "sire_draw_bias_summary.csv",
        "grass_type": data_dir / "grass_type_bloodline.csv",
        "grass_summary": data_dir / "sire_grass_type_summary.csv",
        "fc_draw": data_dir / "first_corner_draw_interaction.csv",
        "fc_draw_summary": data_dir / "first_corner_draw_summary.csv",
    }

    path = file_map.get(analysis_type)
    if not path or not path.exists():
        return JSONResponse(
            {"error": f"データなし: {analysis_type} (surface={ctx['surface']})"},
            status_code=404,
        )

    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
    return JSONResponse({
        **ctx,
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
    })


# ═══════════════════════════════════════════════════════════════
#  クッション値・含水率 API
# ═══════════════════════════════════════════════════════════════

_cushion_job: dict[str, Any] = {"running": False}


@app.get("/api/cushion/data")
async def api_cushion_data(
    year: int | None = None,
    venue_code: str | None = None,
    venue_name: str | None = None,
):
    from pathlib import Path as _P
    data_path = _P(os.path.dirname(__file__)).parent / "data" / "local" / "jra_baba" / "cushion_values.json"
    if not data_path.exists():
        return JSONResponse({"error": "クッション値データなし"}, status_code=404)

    records = json.loads(data_path.read_text(encoding="utf-8"))
    if year:
        records = [r for r in records if r.get("year") == year]
    if venue_code:
        records = [r for r in records if r.get("venue_code") == venue_code]
    if venue_name:
        records = [r for r in records if r.get("venue_name") == venue_name]

    return JSONResponse({"total": len(records), "records": records})


@app.get("/api/cushion/stats")
async def api_cushion_stats():
    from pathlib import Path as _P
    data_path = _P(os.path.dirname(__file__)).parent / "data" / "local" / "jra_baba" / "cushion_values.json"
    if not data_path.exists():
        return JSONResponse({"error": "データなし"}, status_code=404)

    records = json.loads(data_path.read_text(encoding="utf-8"))
    by_year: dict[int, int] = {}
    by_venue: dict[str, int] = {}
    for r in records:
        y = r.get("year", 0)
        by_year[y] = by_year.get(y, 0) + 1
        v = r.get("venue_name", "")
        by_venue[v] = by_venue.get(v, 0) + 1

    return JSONResponse({
        "total": len(records),
        "years": dict(sorted(by_year.items())),
        "venues": dict(sorted(by_venue.items(), key=lambda x: -x[1])),
    })


@app.post("/api/cushion/scrape")
async def api_cushion_scrape(request: Request, background_tasks: BackgroundTasks):
    if _cushion_job["running"]:
        return JSONResponse({"error": "既に実行中"}, status_code=409)

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    years = body.get("years")

    _cushion_job.update(running=True, error=None, result=None, started_at=_time.time())
    background_tasks.add_task(_run_cushion_scrape, years)
    return JSONResponse({"status": "started"})


def _run_cushion_scrape(years: list[int] | None):
    try:
        from src.scraper.jra_cushion import JRACushionScraper
        _base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "jra_baba")
        scraper = JRACushionScraper(output_dir=_base)
        stats = scraper.scrape(years=years)
        _cushion_job["result"] = stats
    except Exception as e:
        import traceback
        _cushion_job["error"] = str(e)
        traceback.print_exc()
    finally:
        _cushion_job["running"] = False


@app.get("/api/cushion/scrape/status")
async def api_cushion_scrape_status():
    return JSONResponse({
        "running": _cushion_job["running"],
        "error": _cushion_job.get("error"),
        "result": _cushion_job.get("result"),
    })


@app.post("/api/cushion/admin/sync-gcs", response_class=JSONResponse)
async def api_cushion_admin_sync_gcs(request: Request):
    """ローカル cushion_values.json を年別に GCS へ同期（開発者ログイン必須）。"""
    if not is_developer(request):
        return JSONResponse({"error": "開発者ログインが必要です"}, status_code=401)

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    also_full = bool(body.get("also_full"))
    dry_run = bool(body.get("dry_run"))

    from pathlib import Path

    from src.scraper.jra_cushion_storage import upload_cushion_values_to_gcs

    base = Path(__file__).resolve().parents[3]
    out = upload_cushion_values_to_gcs(
        also_full=also_full,
        dry_run=dry_run,
        base_dir=base,
    )
    if not out.get("ok"):
        return JSONResponse(out, status_code=400)
    return JSONResponse(out)


@app.post("/api/cushion/admin/sync-preprocessed", response_class=JSONResponse)
async def api_cushion_admin_sync_preprocessed(request: Request):
    """
    GCS preprocessed/cushion_data の日次 JSON を jra_cushion 年別 JSON にマージ（開発者のみ）。
    """
    if not is_developer(request):
        return JSONResponse({"error": "開発者ログインが必要です"}, status_code=401)

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    dry_run = bool(body.get("dry_run"))
    update_local_json = body.get("update_local_json", True)
    if isinstance(update_local_json, str):
        update_local_json = update_local_json.lower() in ("1", "true", "yes")

    years_raw = body.get("years")
    years_list: list[int] | None = None
    if years_raw is not None and str(years_raw).strip():
        years_list = []
        for part in str(years_raw).replace(" ", "").split(","):
            if not part:
                continue
            if part.isdigit() and len(part) == 4:
                years_list.append(int(part))
        if not years_list:
            years_list = None

    from pathlib import Path

    from src.scraper.jra_cushion_sync import sync_preprocessed_to_jra_cushion

    base = Path(__file__).resolve().parents[3]

    def _job():
        return sync_preprocessed_to_jra_cushion(
            years=years_list,
            dry_run=dry_run,
            update_local_json=bool(update_local_json),
            base_dir=base,
        )

    out = await asyncio.to_thread(_job)
    if not out.get("ok"):
        return JSONResponse(out, status_code=400)
    return JSONResponse(out)


# ── ライブ馬場情報取得 ──

_baba_live_job: dict[str, Any] = {"running": False}


@app.post("/api/cushion/live")
async def api_cushion_live(background_tasks: BackgroundTasks):
    """JRA公式馬場情報ページから最新のクッション値・含水率をライブ取得。"""
    if _baba_live_job["running"]:
        return JSONResponse({"error": "既に実行中"}, status_code=409)

    _baba_live_job.update(running=True, error=None, result=None,
                          started_at=datetime.now().isoformat())
    background_tasks.add_task(_run_baba_live_scrape)
    return JSONResponse({"status": "started"})


def _run_baba_live_scrape():
    try:
        from src.scraper.jra_baba_live import JRABabaLiveScraper
        _base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "jra_baba")
        scraper = JRABabaLiveScraper(output_dir=_base)
        records = scraper.scrape()
        _baba_live_job["result"] = {
            "records": len(records),
            "venues": list({r["venue_name"] for r in records}),
            "dates": sorted({r["date"] for r in records}),
        }
    except Exception as e:
        import traceback
        _baba_live_job["error"] = str(e)
        traceback.print_exc()
    finally:
        _baba_live_job["running"] = False
        _baba_live_job["finished_at"] = datetime.now().isoformat()


@app.get("/api/cushion/live/status")
async def api_cushion_live_status():
    from src.scraper.jra_baba_live import (
        _load_poll_schedule, _get_today_entry,
        _get_poll_windows, _in_any_window, _next_poll_time,
    )
    import datetime as _dt

    _base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "jra_baba")
    schedule = _load_poll_schedule(_base)

    today_entry = _get_today_entry(schedule)
    in_window = False
    today_windows = []
    today_venues = []
    if today_entry:
        today_windows = [{"start": s, "end": e} for s, e in _get_poll_windows(today_entry)]
        in_window = _in_any_window(_get_poll_windows(today_entry))
        today_venues = today_entry.get("venues", [])

    next_date, next_start, wait_sec = _next_poll_time(schedule)
    next_window = ""
    if wait_sec > 0:
        next_window = (_dt.datetime.now() + _dt.timedelta(seconds=wait_sec)).strftime("%m/%d %H:%M")

    return JSONResponse({
        "running": _baba_live_job["running"],
        "started_at": _baba_live_job.get("started_at", ""),
        "finished_at": _baba_live_job.get("finished_at", ""),
        "error": _baba_live_job.get("error"),
        "result": _baba_live_job.get("result"),
        "schedule": {
            "today_type": today_entry.get("type", "") if today_entry else None,
            "today_venues": today_venues,
            "today_windows": today_windows,
            "in_window": in_window,
            "next_poll_date": next_date,
            "next_poll_start": next_start,
            "next_poll_at": next_window,
            "wait_seconds": wait_sec,
        },
    })


@app.get("/api/cushion/live/check")
async def api_cushion_live_check():
    """軽量な更新チェック (フルスクレイプはしない)。"""
    from src.scraper.jra_baba_live import JRABabaLiveScraper
    _base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "jra_baba")
    scraper = JRABabaLiveScraper(output_dir=_base)
    has_new = scraper.has_new_data()
    return JSONResponse({"has_new_data": has_new})


@app.get("/api/cushion/schedule")
async def api_cushion_schedule(days: int = 14):
    """直近のポーリングスケジュールを返す。"""
    from src.scraper.jra_baba_live import _load_poll_schedule, _get_poll_windows
    import datetime as _dt

    _base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "jra_baba")
    schedule = _load_poll_schedule(_base)

    today = _dt.date.today()
    cutoff = (today + _dt.timedelta(days=days)).isoformat()
    today_str = today.isoformat()

    upcoming = []
    for e in schedule:
        if today_str <= e["date"] <= cutoff:
            windows = _get_poll_windows(e)
            y, m, d = map(int, e["date"].split("-"))
            wd = ["月", "火", "水", "木", "金", "土", "日"][_dt.date(y, m, d).weekday()]
            upcoming.append({
                "date": e["date"],
                "weekday": wd,
                "type": e.get("type", ""),
                "venues": e.get("venues", []),
                "windows": [{"start": s, "end": end} for s, end in windows],
                "is_today": e["date"] == today_str,
            })

    return JSONResponse({"schedule": upcoming, "days": days})


# ── 自動スクレイプ ステータス ──

@app.get("/api/auto-scrape/status")
async def api_auto_scrape_status():
    """自動スクレイプの実行ステータスを返す。"""
    from src.scraper.auto_scrape import _load_status, _load_race_calendar
    import datetime as _dt

    status = _load_status()
    calendar = _load_race_calendar()
    today = _dt.date.today()

    upcoming_race = None
    for d in calendar.get("race_days", []):
        if d["date"] >= today.isoformat():
            upcoming_race = d
            break

    next_race = None
    if upcoming_race:
        rd = upcoming_race["date"]
        next_race = {
            "date": rd,
            "venues": [v["venue"] for v in upcoming_race["venues"]],
            "days_until": (_dt.date.fromisoformat(rd) - today).days,
        }

    next_friday = today + _dt.timedelta(days=(4 - today.weekday()) % 7)
    if next_friday == today and _dt.datetime.now().hour >= 18:
        next_friday += _dt.timedelta(days=7)

    return JSONResponse({
        "tasks": status,
        "next_race_day": next_race,
        "next_weekly_update": next_friday.isoformat(),
    })


# ── バッチ手動実行 (dev-only) ──────────────────────────────────────

_manual_batch_job: dict[str, Any] = {
    "status": "idle",   # idle | running | done | error
    "task": None,
    "date": None,
    "started_at": None,
    "finished_at": None,
    "logs": [],
    "current_step": "",
    "result": None,
    "error": None,
}
_manual_batch_lock = threading.Lock()


def _run_manual_batch(task: str, date_str: str) -> None:
    """バッチタスクをバックグラウンドスレッドで実行する。"""
    import datetime as _dt
    from src.scraper.auto_scrape import (
        run_raceday_evening_for_date,
        run_raceday_eve_for_date,
        run_weekly_update_for_dates,
        run_catchup_for_dates,
        task_daily_race_lists,
        task_catchup_missing_dates,
        _load_race_calendar,
        _last_week_race_dates,
        _missing_past_race_dates,
    )

    import re as _re
    _STEP_SKIP = _re.compile(r'^[\s=\-*]+$')

    job = _manual_batch_job
    job["status"] = "running"
    job["started_at"] = _dt.datetime.now().isoformat()
    job["finished_at"] = None
    job["logs"] = []
    job["current_step"] = "起動中..."
    job["result"] = None
    job["error"] = None

    # ログをメモリに積み、意味ある行を current_step に反映する Handler
    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            job["logs"].append(self.format(record))
            msg = record.getMessage().strip()
            if msg and not _STEP_SKIP.match(msg):
                job["current_step"] = msg[:200]

    handler = _ListHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    _orig_root_level = root_logger.level
    if _orig_root_level == logging.NOTSET or _orig_root_level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        if task == "raceday-evening":
            result = run_raceday_evening_for_date(date_str)
        elif task == "raceday-eve":
            # date_str が空なら翌日を使う
            target = date_str.replace("-", "") if date_str else ""
            if not target:
                target = (_dt.date.today() + _dt.timedelta(days=1)).strftime("%Y%m%d")
            result = run_raceday_eve_for_date(target)
        elif task == "weekly-update":
            # date_str (金曜日) の10日前〜当日の開催日を解決
            job["current_step"] = "カレンダー読み込み中..."
            cal = _load_race_calendar()
            ref = _dt.date.fromisoformat(date_str)
            range_start = ref - _dt.timedelta(days=10)
            target_dates = [
                d["date"]
                for d in cal.get("race_days", [])
                if range_start.isoformat() <= d["date"] <= ref.isoformat()
            ]
            if not target_dates:
                target_dates = _last_week_race_dates(cal)
            result = run_weekly_update_for_dates(target_dates)
        elif task == "catchup-missing":
            job["current_step"] = "カレンダー読み込み中..."
            cal = _load_race_calendar()
            job["current_step"] = "欠損日チェック中..."
            missing = _missing_past_race_dates(cal)
            if missing:
                job["current_step"] = f"補完対象: {len(missing)}日 ({', '.join(missing)})"
                result = run_catchup_for_dates(missing)
            else:
                job["current_step"] = "欠損日なし"
                result = {"status": "skipped", "reason": "no-missing-dates"}
            _invalidate_race_list_caches()
        elif task == "daily-race-lists":
            task_daily_race_lists()
            _invalidate_race_list_caches()
            result = {"status": "ok", "note": "daily-race-lists completed"}
        else:
            result = {"status": "error", "reason": f"unknown task: {task}"}

        job["result"] = result
        job["status"] = "done" if result.get("status") != "error" else "error"
    except Exception as e:
        logger.error("[ManualBatch] %s / %s: %s", task, date_str, e, exc_info=True)
        job["error"] = str(e)
        job["status"] = "error"
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(_orig_root_level)
        job["finished_at"] = _dt.datetime.now().isoformat()


@app.post("/api/auto-scrape/run", response_class=JSONResponse)
async def api_auto_scrape_run(request: Request):
    """バッチタスクを手動実行する (dev-only)。
    Body: {"task": "raceday-evening"|"weekly-update"|"catchup-missing"|"daily-race-lists", "date": "YYYY-MM-DD"}
    """
    from src.api.auth import is_developer
    if not is_developer(request):
        return JSONResponse({"error": "unauthorized"}, status_code=403)

    body = await request.json()
    task = str(body.get("task", ""))
    date_str = str(body.get("date", ""))

    _ALLOWED_TASKS = (
        "raceday-eve", "raceday-evening", "weekly-update",
        "catchup-missing", "daily-race-lists",
    )
    if task not in _ALLOWED_TASKS:
        return JSONResponse({"error": f"unknown task: {task}"}, status_code=400)

    import re as _re
    if not _re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return JSONResponse({"error": "date must be YYYY-MM-DD"}, status_code=400)

    with _manual_batch_lock:
        if _manual_batch_job["status"] == "running":
            return JSONResponse({"status": "already_running",
                                 "task": _manual_batch_job["task"],
                                 "date": _manual_batch_job["date"]})
        _manual_batch_job["task"] = task
        _manual_batch_job["date"] = date_str

    t = threading.Thread(target=_run_manual_batch, args=(task, date_str), daemon=True)
    t.start()
    return JSONResponse({"status": "started", "task": task, "date": date_str})


@app.get("/api/auto-scrape/run-status", response_class=JSONResponse)
async def api_auto_scrape_run_status(request: Request):
    """手動バッチの実行状態を返す (dev-only)。"""
    from src.api.auth import is_developer
    if not is_developer(request):
        return JSONResponse({"error": "unauthorized"}, status_code=403)

    job = _manual_batch_job
    return JSONResponse({
        "status": job["status"],
        "task": job["task"],
        "date": job["date"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "log_count": len(job["logs"]),
        "logs": job["logs"][-200:],   # 最新 200 行
        "current_step": job.get("current_step", ""),
        "result": job["result"],
        "error": job["error"],
    })


@app.post("/api/admin/invalidate-race-list-caches", response_class=JSONResponse)
async def api_invalidate_race_list_caches(request: Request):
    """race_lists 関連インメモリキャッシュを即時クリア (dev-only)。daily-race-lists cron から呼ぶ。"""
    from src.api.auth import is_developer
    if not is_developer(request):
        return JSONResponse({"error": "unauthorized"}, status_code=403)
    _invalidate_race_list_caches()
    return JSONResponse({"status": "ok", "cleared": ["_RACE_LIST_STEMS_CACHE", "_scrape_dates_cache",
                                                      "_SCRAPE_DATES_RAW_CACHE", "_PICKER_SCRAPE_DATES_CACHE"]})



# ══════════════════════════════════════════════════════
#  馬場速度レベリング v2 (2着タイム z-score / 基準データ方式)
# ══════════════════════════════════════════════════════

_track_speed_job: dict[str, Any] = {
    "status": "idle",
    "progress": [],
    "error": None,
}


def _track_speed_engine():
    from pathlib import Path
    from src.research.race.track_speed_engine import TrackSpeedEngine
    eng = TrackSpeedEngine(str(Path(__file__).resolve().parents[2]))
    eng.load_baselines()
    return eng


@app.get("/track-speed", response_class=HTMLResponse)
async def track_speed_page(request: Request):
    return templates.TemplateResponse("analysis/track_speed.html", {
        "request": request,
        "current_page": "track_speed",
        "breadcrumbs": [],
    })


@app.get("/track-speed/dev", response_class=HTMLResponse)
async def track_speed_dev_page(request: Request):
    from src.api.auth import is_developer
    if not is_developer(request):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/login?next=/track-speed/dev", status_code=302)
    return templates.TemplateResponse("analysis/track_speed_dev.html", {
        "request": request,
        "current_page": "track_speed",
        "breadcrumbs": [{"label": "馬場速度", "url": "/track-speed"}, {"label": "Dev Docs"}],
    })


@app.get("/api/track-speed/meta", response_class=JSONResponse)
async def api_track_speed_meta():
    from src.research.race.track_speed_engine import (
        BASELINES_PATH, META_PATH, PACE_BASELINES_PATH,
    )
    meta: dict[str, Any] = {
        "baselines_ready": BASELINES_PATH.exists(),
        "pace_baselines_ready": PACE_BASELINES_PATH.exists(),
    }
    if META_PATH.exists():
        import json as _json
        meta.update(_json.loads(META_PATH.read_text(encoding="utf-8")))
    try:
        eng = _track_speed_engine()
        meta["dates_count"] = len(eng.list_dates())
        meta["venues"] = eng.list_venues()
    except Exception:
        meta["venues"] = []
    return JSONResponse(meta)


@app.get("/api/track-speed/dates", response_class=JSONResponse)
async def api_track_speed_dates(venue: Optional[str] = Query(None)):
    try:
        eng = _track_speed_engine()
        return JSONResponse({"dates": eng.list_dates(venue=venue)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/track-speed/venues", response_class=JSONResponse)
async def api_track_speed_venues(
    date: str = Query(..., description="YYYYMMDD or YYYY-MM-DD"),
):
    try:
        eng = _track_speed_engine()
        return JSONResponse({"venues": eng.list_venues_for_date(date)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/track-speed/day", response_class=JSONResponse)
async def api_track_speed_day(
    date: str = Query(..., description="YYYYMMDD or YYYY-MM-DD"),
    venue: Optional[str] = Query(None),
):
    try:
        eng = _track_speed_engine()
        return JSONResponse(eng.query_day(date, venue=venue))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/track-speed/status", response_class=JSONResponse)
async def track_speed_status():
    from src.research.race.track_speed_engine import BASELINES_PATH, META_PATH
    return JSONResponse({
        "status": _track_speed_job["status"],
        "progress": _track_speed_job["progress"][-30:],
        "error": _track_speed_job["error"],
        "baselines_ready": BASELINES_PATH.exists(),
        "assignments_ready": META_PATH.exists(),
    })


@app.post("/api/track-speed/rebuild-baselines", response_class=JSONResponse)
async def track_speed_rebuild_baselines():
    if _track_speed_job["status"] == "running":
        return JSONResponse({"status": "already_running"})
    threading.Thread(target=_run_track_speed_baselines, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.post("/api/track-speed/assign", response_class=JSONResponse)
async def track_speed_assign(
    date_from: str = Query("2026-01-01"),
    date_to: Optional[str] = Query(None),
):
    if _track_speed_job["status"] == "running":
        return JSONResponse({"status": "already_running"})
    threading.Thread(
        target=_run_track_speed_assign,
        args=(date_from, date_to),
        daemon=True,
    ).start()
    return JSONResponse({"status": "started"})


@app.get("/api/track-speed/validate-perf", response_class=JSONResponse)
async def api_validate_perf():
    """レースパフォーマンス指数のバリデーションを実行して結果を返す。"""
    try:
        from src.research.race.perf_index import run_validation
        result = run_validation()
        return JSONResponse(result)
    except Exception as e:
        logger.error("[ValidatePerf] %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/track-speed/race-horses", response_class=JSONResponse)
async def api_race_horses(
    race_id: str = Query(...),
    race_perf: float = Query(..., description="レース全体のperf_index（2着馬の基準値）"),
):
    """レースの全馬にperf_indexと速度水準ラベルを付けて返す。2着馬がrace_perfと同値。"""
    try:
        import pandas as _pd
        from src.scraper.local_tables import load_flat_df
        from src.research.race.track_speed_engine import format_race_time, RACES_DIR

        year = str(race_id)[:4]
        cols = ["race_id", "finish_position", "horse_number", "horse_name", "time_sec"]
        df = load_flat_df("race_result", years=[year], columns=cols, base_dir=BASE_DIR)
        race_df = df[df["race_id"] == race_id].copy()
        if race_df.empty:
            return JSONResponse({"horses": [], "error": "レースデータなし"})
        race_df = race_df[race_df["finish_position"] > 0].sort_values("finish_position")
        r2 = race_df[race_df["finish_position"] == 2]
        if r2.empty:
            return JSONResponse({"horses": [], "error": "2着データなし"})
        time_2nd = float(r2.iloc[0]["time_sec"])

        # レースメタデータをtrack_speedパーケットから取得（速度水準分類に必要）
        race_meta: dict = {}
        try:
            rp = RACES_DIR / f"races_{year}.parquet"
            if rp.exists():
                meta_cols = ["race_id", "venue", "layout", "surface", "distance",
                             "class_band", "cond_pool", "time_2nd_adj"]
                rm = _pd.read_parquet(rp, columns=meta_cols)
                rm = rm[rm["race_id"] == race_id]
                if not rm.empty:
                    race_meta = rm.iloc[0].to_dict()
        except Exception:
            pass

        # 速度水準ラベル分類の準備
        eng = _track_speed_engine() if race_meta else None
        time_2nd_adj = float(race_meta.get("time_2nd_adj") or 0)

        horses = []
        for _, row in race_df.iterrows():
            t = row["time_sec"]
            if t is None or (isinstance(t, float) and _pd.isna(t)) or float(t) <= 0:
                continue
            t = float(t)
            diff = round(t - time_2nd, 1)
            hp = round(race_perf - diff * 10, 1)

            # 各馬のペース補正済みタイム = 2着馬の補正値 + 2着からの差
            horse_class_level: str | None = None
            if eng and time_2nd_adj > 0:
                time_adj_horse = time_2nd_adj + (t - time_2nd)
                horse_class_level = eng._classify_class_level(
                    time_adj_horse,
                    str(race_meta.get("venue", "")),
                    str(race_meta.get("layout") or "-"),
                    str(race_meta.get("surface", "")),
                    int(race_meta.get("distance") or 0),
                    str(race_meta.get("cond_pool") or "firm_yielding"),
                    class_band=str(race_meta.get("class_band") or ""),
                )

            horses.append({
                "finish_position": int(row["finish_position"]),
                "horse_number": int(row["horse_number"]) if row["horse_number"] else 0,
                "horse_name": str(row["horse_name"] or ""),
                "time_sec": round(t, 1),
                "time_fmt": format_race_time(t),
                "time_diff": diff,
                "horse_perf": hp,
                "horse_class_level": horse_class_level,
            })
        return JSONResponse({"horses": horses, "race_perf": race_perf, "time_2nd": time_2nd})
    except Exception as e:
        logger.error("[RaceHorses] %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/track-speed/by-category", response_class=JSONResponse)
async def api_track_speed_by_category(
    venue: str = Query(..., description="競馬場名"),
    level: int = Query(..., ge=1, le=5, description="馬場速度レベル 1=超低速〜5=超高速"),
    surface: Optional[str] = Query(None, description="芝 or ダート"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    try:
        eng = _track_speed_engine()
        return JSONResponse(eng.query_by_category(venue, level, surface=surface, offset=offset, limit=limit))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _run_track_speed_baselines():
    job = _track_speed_job
    job["status"] = "running"
    job["progress"] = []
    job["error"] = None

    def log(msg: str):
        job["progress"].append(msg)
        logger.info("[TrackSpeed] %s", msg)

    try:
        from pathlib import Path
        from src.research.race.track_speed_engine import TrackSpeedEngine
        eng = TrackSpeedEngine(str(Path(__file__).resolve().parents[2]))
        eng.build_baselines(years=["2020", "2021", "2022", "2023", "2024", "2025"], progress_cb=log)
        from src.research.race.perf_index import invalidate_sigma_cache
        invalidate_sigma_cache()
        job["status"] = "done"
    except Exception as e:
        logger.error("[TrackSpeed] %s", e, exc_info=True)
        job["status"] = "error"
        job["error"] = str(e)


def _run_track_speed_assign(date_from: str, date_to: str | None):
    job = _track_speed_job
    job["status"] = "running"
    job["progress"] = []
    job["error"] = None

    def log(msg: str):
        job["progress"].append(msg)
        logger.info("[TrackSpeed] %s", msg)

    try:
        from pathlib import Path
        from src.research.race.track_speed_engine import TrackSpeedEngine
        eng = TrackSpeedEngine(str(Path(__file__).resolve().parents[2]))
        y0 = int(date_from[:4])
        y1 = int((date_to or date_from)[:4])
        eng.assign_races(
            years=[str(y) for y in range(y0, y1 + 1)],
            date_min=date_from.replace("-", ""),
            date_max=(date_to or "").replace("-", "") or None,
            progress_cb=log,
        )
        job["status"] = "done"
    except Exception as e:
        logger.error("[TrackSpeed] %s", e, exc_info=True)
        job["status"] = "error"
        job["error"] = str(e)



# ─── ミオスタチン遺伝子ページ ──────────────────────────


@app.get("/myostatin", response_class=HTMLResponse)
async def myostatin_page(request: Request):
    return templates.TemplateResponse(
        "pedigree/myostatin.html",
        {"request": request, "current_page": "myostatin"},
    )


@app.get("/api/myostatin", response_class=JSONResponse)
async def myostatin_data():
    import json as _json
    from pathlib import Path
    # app.py は src/api/app.py に位置するため、project root は parents[2]。
    # (parents[3] だと project の親 = キーノット外を指してしまい、KB が見つからない)
    kb_path = Path(__file__).resolve().parents[2] / "data" / "local" / "knowledge" / "myostatin_genes.json"
    if not kb_path.exists():
        return JSONResponse(
            {"error": "KB not found", "checked_path": str(kb_path)},
            status_code=404,
        )
    data = _json.loads(kb_path.read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.post("/api/myostatin/predict", response_class=JSONResponse)
async def myostatin_predict(request: Request):
    from src.research.genes.myostatin import MyostatinLookup
    body = await request.json()
    sire = body.get("sire", "")
    dam_sire = body.get("dam_sire", "")
    distance = body.get("distance", 0)
    if not sire:
        return JSONResponse({"error": "sire is required"}, status_code=400)
    mstn = MyostatinLookup()
    result = {
        "sire_info": mstn.get_sire_info(sire),
        "dam_sire_info": mstn.get_sire_info(dam_sire) if dam_sire else None,
        "offspring": mstn.predict_offspring(sire, dam_sire) if dam_sire else None,
        "features": mstn.offspring_features(sire, dam_sire, distance) if dam_sire else None,
    }
    return JSONResponse(result)


# ─── 成長曲線ページ ──────────────────────────


@app.post("/api/myostatin/recalculate", response_class=JSONResponse)
async def myostatin_recalculate():
    """
    全ての未確定馬のミオスタチン遺伝子型を血統から再計算する。
    メンデルの法則に基づいてアレル確率を計算し、JSONファイルを更新する。
    """
    import json
    import os
    from pathlib import Path

    try:
        # JSONファイルのパス (app.py は src/api/ に在り、project root は parents[2])
        json_path = Path(__file__).resolve().parents[2] / "data" / "local" / "knowledge" / "myostatin_genes.json"

        # 読み込み
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 名前→遺伝子型のマップを作成
        stallion_map = {}
        for s in data["stallions"]:
            stallion_map[s["name"]] = s
            if s.get("name_en"):
                stallion_map[s["name_en"]] = s

        def get_alleles(name):
            """種牡馬のアレル確率を取得"""
            if name in stallion_map:
                s = stallion_map[name]
                return s.get("allele_c", 0.5), s.get("allele_t", 0.5)
            return 0.5, 0.5  # デフォルト

        def calculate_offspring_genotype(sire_c, sire_t, dam_c, dam_t):
            """
            父と母のアレル確率から、子のCC/CT/TT確率を計算

            Args:
                sire_c: 父がCアレルを渡す確率
                sire_t: 父がTアレルを渡す確率
                dam_c: 母がCアレルを渡す確率
                dam_t: 母がTアレルを渡す確率

            Returns:
                (allele_c, allele_t, genotype, confidence)
            """
            # CC確率 = 父C × 母C
            prob_cc = sire_c * dam_c
            # TT確率 = 父T × 母T
            prob_tt = sire_t * dam_t
            # CT確率 = 残り
            prob_ct = 1.0 - prob_cc - prob_tt

            # 子のアレル確率
            child_c = prob_cc + prob_ct * 0.5
            child_t = prob_tt + prob_ct * 0.5

            # 遺伝子型を推定
            if prob_cc > 0.9:
                genotype = "CC"
                confidence = "highly_likely"
            elif prob_tt > 0.9:
                genotype = "TT"
                confidence = "highly_likely"
            elif prob_cc > 0.7:
                genotype = "C?"
                confidence = "estimated"
            elif prob_tt > 0.7:
                genotype = "?T"
                confidence = "estimated"
            elif prob_ct > 0.6:
                genotype = "CT"
                confidence = "estimated"
            elif child_c > 0.6:
                genotype = "C?"
                confidence = "estimated"
            elif child_t > 0.6:
                genotype = "?T"
                confidence = "estimated"
            else:
                genotype = "??"
                confidence = "inferred"

            return round(child_c, 3), round(child_t, 3), genotype, confidence

        # 再計算
        updates = []
        for i, stallion in enumerate(data["stallions"]):
            # 確定している場合はスキップ
            if stallion["confidence"] == "confirmed":
                continue

            # 父の情報
            sire_name = stallion.get("sire", "")
            if not sire_name:
                continue

            sire_c, sire_t = get_alleles(sire_name)

            # 母父の情報（母のアレル確率の推定に使用）
            dam_sire_name = stallion.get("dam_sire", "")
            if dam_sire_name:
                dam_sire_c, dam_sire_t = get_alleles(dam_sire_name)
                # 母のアレル確率は母父から推定（簡易版：母父のアレル確率を使用）
                dam_c, dam_t = dam_sire_c, dam_sire_t
            else:
                # 母父不明の場合はデフォルト
                dam_c, dam_t = 0.5, 0.5

            # 子のアレル確率を計算
            child_c, child_t, genotype, confidence = calculate_offspring_genotype(
                sire_c, sire_t, dam_c, dam_t
            )

            # 既存より確実性が高い場合のみ更新
            old_gt = stallion.get("genotype", "??")
            old_conf = stallion.get("confidence", "inferred")

            # 確実性のランク
            conf_rank = {"confirmed": 4, "highly_likely": 3, "estimated": 2, "inferred": 1}
            old_rank = conf_rank.get(old_conf, 0)
            new_rank = conf_rank.get(confidence, 0)

            # より確実な情報の場合、または同等で遺伝子型が変わる場合のみ更新
            if new_rank >= old_rank:
                data["stallions"][i]["genotype"] = genotype
                data["stallions"][i]["confidence"] = confidence
                data["stallions"][i]["allele_c"] = child_c
                data["stallions"][i]["allele_t"] = child_t

                reason = f"父{sire_name}(C={sire_c:.2f},T={sire_t:.2f})"
                if dam_sire_name:
                    reason += f" × 母父{dam_sire_name}(C={dam_sire_c:.2f},T={dam_sire_t:.2f})"

                data["stallions"][i]["source"] = f"確率計算による推測: {reason}"

                if old_gt != genotype or old_conf != confidence:
                    updates.append({
                        "name": stallion["name"],
                        "old": f"{old_gt} ({old_conf})",
                        "new": f"{genotype} ({confidence})",
                        "allele_c": child_c,
                        "allele_t": child_t
                    })

        # メタデータを更新
        data["_meta"]["version"] = "2.3.2"
        data["_meta"]["last_updated"] = "2026-03-24"

        # 保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return JSONResponse({
            "success": True,
            "updated_count": len(updates),
            "total_stallions": len(data["stallions"]),
            "updates": updates[:20],  # 最初の20件のみ返す
            "message": f"{len(updates)}頭の遺伝子型を再計算しました"
        })

    except Exception as e:
        import traceback
        return JSONResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)


@app.get("/growth-curve", response_class=HTMLResponse)
async def growth_curve_page(request: Request):
    return templates.TemplateResponse(
        "analysis/growth_curve.html",
        {"request": request, "current_page": "growth_curve"},
    )


def _enrich_race_with_speed_index(
    race: dict,
    horse_number: int,
    storage,
    queue=None,
    force_refresh: bool = False
) -> tuple[dict, bool]:
    """
    レースデータにタイム指数を追加する（ない場合はキューに追加）

    Args:
        race: レースデータ
        horse_number: 馬番
        storage: ストレージインスタンス
        queue: スクレイピングキュー（オプション）
        force_refresh: 強制的に再スクレイピングするか

    Returns:
        (タイム指数が追加されたレースデータ, キューに追加したか)
    """
    race_id = race.get("race_id")
    if not race_id:
        return race, False

    # force_refreshの場合、既存のタイム指数を無視
    if not force_refresh and (race.get("time_index") or race.get("speed_index")):
        return race, False

    try:
        # force_refreshの場合、既存のrace_indexデータを削除
        if force_refresh:
            try:
                import os
                year = race_id[:4]
                local_path = os.path.join(storage._local_dir, "race_index", year, f"{race_id}.json")
                if os.path.exists(local_path):
                    os.remove(local_path)
                    logger.info(f"古いrace_indexを削除: {race_id}")
            except Exception as e:
                logger.warning(f"race_index削除エラー: {e}")

        # race_indexデータを確認
        speed_data = storage.load("race_index", race_id)

        # データがない場合はキューに追加
        if not speed_data:
            if queue:
                # スクレイピングキューに追加（最優先）
                logger.info(f"タイム指数をキューに追加（最優先）: {race_id}")
                queue.add_job(
                    kind="race",
                    target_id=race_id,
                    task="race_index",
                    priority=1,  # 最優先
                    label=f"タイム指数: {race_id}"
                )
                return race, True
            else:
                logger.warning(f"キューが利用できません。タイム指数のスクレイピングをスキップ: {race_id}")
                return race, False

        # タイム指数データから該当馬の指数を抽出
        if speed_data and "entries" in speed_data:
            for entry in speed_data["entries"]:
                if entry.get("horse_number") == horse_number:
                    # タイム指数Mを優先的に取得
                    time_index = (
                        entry.get("time_index_m") or
                        entry.get("speed_max") or
                        entry.get("speed_avg")
                    )
                    if time_index and time_index > 0:
                        race["time_index"] = time_index
                        logger.info(f"タイム指数を取得: {race_id} 馬番{horse_number} = {time_index}")
                    break

    except Exception as e:
        logger.warning(f"タイム指数取得エラー (race_id={race_id}): {e}")

    return race, False


_growth_curve_cache: dict[str, tuple[float, dict]] = {}
_GROWTH_CURVE_CACHE_TTL = 120  # 2 minutes


@app.get("/api/growth-curve/{horse_id}", response_class=JSONResponse)
async def growth_curve_data(
    horse_id: str,
    fetch_speed_index: bool = True,
    force_refresh: bool = False,
    limit: Optional[int] = None
):
    """
    馬の成長曲線データを返す。

    Args:
        horse_id: 馬ID
        fetch_speed_index: タイム指数を取得するか（デフォルト: True）
        force_refresh: race_indexを強制的に再スクレイピングするか（デフォルト: False）
        limit: 表示する直近レース数（デフォルト: None = 全て）

    Returns:
        - horse_name: 馬名
        - races: 出走履歴（日付、馬体重、着順、レース間隔等）
        - stats: 統計情報
    """
    try:
        import time as _time_mod
        # キャッシュチェック（force_refreshは除く）
        if not force_refresh:
            _cache_key = f"{horse_id}:{limit}:{fetch_speed_index}"
            _cached = _growth_curve_cache.get(_cache_key)
            if _cached and (_time_mod.time() - _cached[0]) < _GROWTH_CURVE_CACHE_TTL:
                return JSONResponse(_cached[1])

        storage = _get_storage()

        # force_refreshの場合、既存のhorse_resultを削除して再スクレイピング
        if force_refresh:
            import os
            year = horse_id[:4]
            local_path = os.path.join(storage._local_dir, "horse_result", year, f"{horse_id}.json")
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.info(f"古いhorse_resultを削除: {horse_id}")

            # 再スクレイピングを実行
            logger.info(f"horse_resultを再スクレイピング: {horse_id}")
            from src.scraper.run import ScraperRunner
            runner = ScraperRunner(interval=1.0, auto_login=True)
            runner.storage = storage

            # ログイン状態を確認
            if runner.client._logged_in:
                logger.info(f"✓ ログイン済み")
            else:
                logger.warning(f"✗ ログインしていません。タイム指数は取得できません。")

            horse_data = runner.scrape_horse(horse_id, skip_existing=False, with_history=True)

            if not horse_data:
                return JSONResponse({"error": f"馬ID {horse_id} のスクレイピングに失敗しました"}, status_code=500)

            # タイム指数の取得状況をログ出力
            if horse_data.get("race_history"):
                races_with_index = sum(1 for r in horse_data["race_history"] if r.get("time_index", 0) > 0)
                total_races = len(horse_data["race_history"])
                logger.info(f"タイム指数取得: {races_with_index}/{total_races}レース")

                # 最初のレースのtime_indexをログ出力（デバッグ用）
                first_race = horse_data["race_history"][0]
                logger.info(f"最新レース: {first_race.get('race_name')} - time_index={first_race.get('time_index', 0)}")
        else:
            # 通常の取得
            horse_data = storage.load("horse_result", horse_id)
            if not horse_data:
                return JSONResponse({"error": f"馬ID {horse_id} のデータが見つかりません"}, status_code=404)

        horse_name = horse_data.get("horse_name", "不明")
        results_all = horse_data.get("race_history", horse_data.get("results", []))
        total_race_count = len(results_all)

        # limitを先に適用してGCSフェッチ対象を絞り込む
        results = sorted(results_all, key=lambda r: r.get("date", ""), reverse=True)
        if limit and limit > 0:
            results = results[:limit]

        # タイム指数を補完（必要に応じてキューに追加）
        scraping_status = {
            "required_races": 0,
            "completed_races": 0,
            "pending_races": 0,
        }

        # タイム指数のステータス確認（2024/1/1以降のみ対象）
        if fetch_speed_index:
            for race in results:
                if race.get("time_index") and race.get("time_index") > 0:
                    scraping_status["completed_races"] += 1
                elif race.get("date", "").replace("/", "-") >= "2024-01-01":
                    scraping_status["required_races"] += 1

            # force_refreshの場合は何もしない（horse_result再取得で完了）
            if not force_refresh and scraping_status["required_races"] > 0:
                from src.scraper.job_queue import ScrapeJobQueue
                from concurrent.futures import ThreadPoolExecutor as _GcPool
                queue = ScrapeJobQueue()

                need_idx = [
                    (i, race)
                    for i, race in enumerate(results)
                    if (not race.get("time_index") or race.get("time_index") == 0)
                    and race.get("horse_number") and race.get("race_id")
                    and race.get("date", "").replace("/", "-") >= "2024-01-01"
                ]
                if need_idx:
                    rids = list({r.get("race_id") for _, r in need_idx if r.get("race_id")})
                    idx_map: dict[str, dict] = {}
                    def _load_idx(rid):
                        return rid, storage.load("race_index", rid)
                    with _GcPool(max_workers=min(len(rids), 20)) as pool:
                        for rid, data in pool.map(_load_idx, rids):
                            if data:
                                idx_map[rid] = data

                    _jobs_to_add = []
                    for i, race in need_idx:
                        race_id = race.get("race_id")
                        horse_number = race.get("horse_number")
                        speed_data = idx_map.get(race_id)
                        if speed_data and "entries" in speed_data:
                            for entry in speed_data["entries"]:
                                if entry.get("horse_number") == horse_number:
                                    ti = (entry.get("time_index_m")
                                          or entry.get("speed_max")
                                          or entry.get("speed_avg"))
                                    if ti and ti > 0:
                                        race["time_index"] = ti
                                        results[i] = race
                                        scraping_status["completed_races"] += 1
                                        scraping_status["required_races"] -= 1
                                    break
                        elif not speed_data and queue:
                            _jobs_to_add.append({
                                "job_kind": "race",
                                "target_id": race_id,
                                "tasks": ["race_index"],
                                "priority": 1,
                                "job_label": f"タイム指数: {race_id}",
                            })
                    if _jobs_to_add:
                        queue.bulk_add_jobs(_jobs_to_add)
                        scraping_status["pending_races"] += len(_jobs_to_add)

        if not results:
            return JSONResponse({"error": "出走履歴が見つかりません"}, status_code=404)

        # 出走履歴を古い順（昇順）にソート（日付でソート）
        # レース間隔を正しく計算するため、古い順に処理する
        results_sorted = sorted(results, key=lambda r: r.get("date", ""), reverse=False)

        # 出走履歴を処理
        races = []
        prev_date = None
        weights = []
        ranks = []

        for i, race in enumerate(results_sorted):
            date_str = race.get("date", "")
            weight = race.get("weight")
            weight_diff = race.get("weight_change", race.get("weight_diff"))

            # 着順を数値化（finish_positionまたはrank）
            rank = race.get("finish_position", race.get("rank"))
            if rank and isinstance(rank, str):
                try:
                    rank = int(rank) if rank.isdigit() else None
                except:
                    rank = None
            elif rank == 0 or rank == -1:
                rank = None

            # レース間隔を計算
            interval_days = None
            if prev_date and date_str:
                try:
                    from datetime import datetime
                    curr_date = datetime.strptime(date_str.replace("/", "-"), "%Y-%m-%d")
                    prev_date_obj = datetime.strptime(prev_date.replace("/", "-"), "%Y-%m-%d")
                    interval_days = (curr_date - prev_date_obj).days
                except:
                    pass

            # タイム指数（簡易版：存在すれば使用）
            time_index = race.get("time_index") or race.get("speed_index")

            race_info = {
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
            }
            races.append(race_info)

            if weight:
                weights.append(weight)
            if rank:
                ranks.append(rank)

            prev_date = date_str

        # デビュー時馬体重（全レース中最古の馬体重あり出走）
        debut_weight = None
        debut_date = None
        for r in sorted(results_all, key=lambda x: x.get("date", "")):
            w = r.get("weight")
            if w:
                debut_weight = w
                debut_date = r.get("date")
                break

        # 統計情報
        stats = {
            "horse_name": horse_name,
            "total_races": len(races),
            "avg_weight": sum(weights) / len(weights) if weights else 0,
            "weight_range": [min(weights), max(weights)] if weights else [0, 0],
            "best_rank": min(ranks) if ranks else None,
            "avg_rank": sum(ranks) / len(ranks) if ranks else None,
        }

        # レースを新しい順（降順）に戻す（フロントエンド表示用）
        races.reverse()

        response = {
            **stats,
            "total_all_races": total_race_count,
            "debut_weight": debut_weight,
            "debut_date": debut_date,
            "races": races,
        }

        # スクレイピングステータスを追加（必要な場合のみ）
        if scraping_status["required_races"] > 0 or scraping_status["pending_races"] > 0:
            response["scraping_status"] = scraping_status

        # キャッシュ書き込み
        if not force_refresh:
            _growth_curve_cache[_cache_key] = (_time_mod.time(), response)
            if len(_growth_curve_cache) > 500:
                oldest = min(_growth_curve_cache, key=lambda k: _growth_curve_cache[k][0])
                del _growth_curve_cache[oldest]

        return JSONResponse(response)

    except Exception as e:
        logger.error(f"成長曲線データ取得エラー: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ひらがな/カタカナ変換ユーティリティ
def _hiragana_to_katakana(text: str) -> str:
    """ひらがなをカタカナに変換"""
    result = []
    for char in text:
        code = ord(char)
        if 0x3041 <= code <= 0x3096:  # ひらがな範囲
            result.append(chr(code + 0x60))  # カタカナに変換
        else:
            result.append(char)
    return ''.join(result)


def _katakana_to_hiragana(text: str) -> str:
    """カタカナをひらがなに変換"""
    result = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:  # カタカナ範囲
            result.append(chr(code - 0x60))  # ひらがなに変換
        else:
            result.append(char)
    return ''.join(result)


def _normalize_search_text(text: str) -> list:
    """検索用に正規化（ひらがな、カタカナ、元の文字列）"""
    text = text.strip()
    variants = [text]
    hiragana = _katakana_to_hiragana(text)
    katakana = _hiragana_to_katakana(text)
    if hiragana not in variants:
        variants.append(hiragana)
    if katakana not in variants:
        variants.append(katakana)
    return variants


def _is_kana_only_query(q: str) -> bool:
    """クエリがひらがな・カタカナ（長音記号等）のみか。漢字・英数字混じりなら False。"""
    s = unicodedata.normalize("NFKC", (q or "").strip())
    if not s:
        return False
    for c in s:
        if c.isspace():
            continue
        o = ord(c)
        if o in (0x30FC, 0xFF70):  # ー 長音
            continue
        if 0x3041 <= o <= 0x3096:  # ひらがな
            continue
        if 0x30A1 <= o <= 0x30F6:  # カタカナ
            continue
        if 0x31F0 <= o <= 0x31FF:  # 小書きカタカナ拡張
            continue
        return False
    return True


_horse_kks: Any = None


@functools.lru_cache(maxsize=50000)
def _horse_name_hiragana_reading_cached(horse_name: str) -> str:
    """
    馬名全体を pykakasi でひらがな列にしたもの（漢字馬名を「い」等のかな入力でヒットさせる）。
    pykakasi 未導入・失敗時は空文字。
    """
    if not horse_name:
        return ""
    try:
        import pykakasi

        global _horse_kks
        if _horse_kks is None:
            _horse_kks = pykakasi.kakasi()
        parts = _horse_kks.convert(horse_name)
        return "".join((p.get("hira") or "") for p in parts)
    except Exception:
        return ""


def _load_horse_name_index():
    """馬名インデックスをロード（mtime で無効化されるメモリキャッシュ）。"""
    from src.utils.horse_name_index import load_horse_name_index

    return load_horse_name_index(BASE_DIR)


@app.get("/api/horse-names/index-meta", response_class=JSONResponse)
async def horse_names_index_meta():
    """
    馬名インデックスの参照用メタ（パス・頭数・生成時刻）。
    全馬リストは返さない。
    """
    from src.utils.horse_name_index import horse_name_index_meta

    return JSONResponse(horse_name_index_meta(BASE_DIR))


@app.get("/api/horse-names/search", response_class=JSONResponse)
async def search_horse_names(q: str = "", limit: int = 20):
    """
    馬名を検索して候補を返す。

    Args:
        q: 検索クエリ（ひらがな、カタカナ、漢字対応）
        limit: 返す最大件数

    Returns:
        [{"horse_id": "2019105551", "horse_name": "イクイノックス", "name_en": "Equinox"}, ...]
    """
    if not q or len(q) < 1:
        return JSONResponse({"results": []})

    try:
        # インデックスをロード
        index_data = _load_horse_name_index()
        horses = index_data.get("horses", [])

        if not horses:
            return JSONResponse({"results": [], "error": "インデックスが空です"})

        q_nfkc = unicodedata.normalize("NFKC", q.strip())
        # 検索クエリを正規化（ひらがな、カタカナのバリエーション）
        query_variants = _normalize_search_text(q_nfkc.lower())
        use_reading = _is_kana_only_query(q_nfkc)

        results = []

        for horse in horses:
            if len(results) >= limit:
                break

            horse_name = horse.get("horse_name", "")
            name_en = horse.get("name_en", "")

            if not horse_name:
                continue

            # 検索マッチング（部分一致）
            search_targets = _normalize_search_text(horse_name.lower())
            if name_en:
                search_targets.append(name_en.lower())
            # かなのみのクエリ（例: 「い」）では、漢字馬名を pykakasi 読みでも照合する
            if use_reading:
                h_read = _horse_name_hiragana_reading_cached(horse_name)
                if h_read:
                    search_targets.extend(_normalize_search_text(h_read.lower()))

            # いずれかのバリアントがマッチするか確認
            matched = False
            for query_var in query_variants:
                for target in search_targets:
                    if query_var in target:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                results.append({
                    "horse_id": horse.get("horse_id"),
                    "horse_name": horse_name,
                    "name_en": name_en or "",
                })

        return JSONResponse({"results": results})

    except Exception as e:
        logger.error(f"馬名検索エラー: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════════
#  血統×コース条件 分析ページ  /pedigree-race-stats
# ══════════════════════════════════════════════════════════════════

_PED_RACE_IDX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "research", "pedigree_race_index")
_ped_race_slim_df: "pd.DataFrame | None" = None
_ped_cats_df: "pd.DataFrame | None" = None
_ped_race_meta: dict = {}
_stallion_lineage_df: "pd.DataFrame | None" = None
_ped_race_lock = threading.Lock()


def _load_ped_race_index() -> tuple["Any", "Any", dict]:
    """race_result_slim と horse_pedigree_cats を遅延ロード（メモリキャッシュ）。"""
    global _ped_race_slim_df, _ped_cats_df, _ped_race_meta
    import pandas as _pd
    with _ped_race_lock:
        if _ped_race_slim_df is not None:
            return _ped_race_slim_df, _ped_cats_df, _ped_race_meta
        slim_path = os.path.join(_PED_RACE_IDX_DIR, "race_result_slim.parquet")
        cats_path = os.path.join(_PED_RACE_IDX_DIR, "horse_pedigree_cats.parquet")
        meta_path = os.path.join(_PED_RACE_IDX_DIR, "meta.json")
        if not os.path.exists(slim_path) or not os.path.exists(cats_path):
            return None, None, {}
        _ped_race_slim_df = _pd.read_parquet(slim_path)
        _ped_cats_df = _pd.read_parquet(cats_path)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as _f:
                _ped_race_meta = json.loads(_f.read())
        return _ped_race_slim_df, _ped_cats_df, _ped_race_meta


def _load_stallion_lineage() -> "pd.DataFrame | None":
    """stallion_lineage.parquet を遅延ロード（メモリキャッシュ）。"""
    global _stallion_lineage_df
    import pandas as _pd
    with _ped_race_lock:
        if _stallion_lineage_df is not None:
            return _stallion_lineage_df
        lineage_path = os.path.join(_PED_RACE_IDX_DIR, "stallion_lineage.parquet")
        if not os.path.exists(lineage_path):
            return None
        cols = ["stallion_id", "anchor_name", "group_id", "depth_to_anchor",
                "main_group_id", "main_group_name", "sub_group_label"]
        import pyarrow.parquet as _pq
        pq_cols = _pq.read_schema(lineage_path).names
        load_cols = [c for c in cols if c in pq_cols]
        lin = _pd.read_parquet(lineage_path, columns=load_cols)
        lin["stallion_id"] = lin["stallion_id"].astype(str)
        lin["group_id"] = lin["group_id"].astype(int)
        if "depth_to_anchor" in lin.columns:
            lin["depth_to_anchor"] = lin["depth_to_anchor"].astype(int)
        if "main_group_id" not in lin.columns:
            lin["main_group_id"] = -1
        if "main_group_name" not in lin.columns:
            lin["main_group_name"] = ""
        if "sub_group_label" not in lin.columns:
            lin["sub_group_label"] = lin.get("anchor_name", "")
        # 重複があればfirst を使う
        lin = lin.drop_duplicates(subset=["stallion_id"])
        _stallion_lineage_df = lin
        return _stallion_lineage_df


@app.get("/pedigree-race-stats", response_class=HTMLResponse)
async def pedigree_race_stats_page(request: Request):
    return templates.TemplateResponse("pedigree/pedigree_race_stats.html", {
        "request": request,
        "current_page": "pedigree_race_stats",
    })


@app.get("/api/pedigree-race-stats/meta", response_class=JSONResponse)
async def api_pedigree_race_stats_meta():
    """フィルタ用のメタ情報（会場・クラス・距離範囲等）を返す。"""
    try:
        slim, cats, meta = _load_ped_race_index()
        if slim is None:
            return JSONResponse({
                "error": "インデックス未生成。python -m src.research.pedigree.build_pedigree_race_index を実行してください。"
            })
        return JSONResponse({
            "venues": meta.get("venues", []),
            "surfaces": meta.get("surfaces", []),
            "grades": [g for g in meta.get("grades", []) if g],
            "track_conditions": meta.get("track_conditions", []),
            "dist_min": int(slim["distance"].min()),
            "dist_max": int(slim["distance"].max()),
            "date_min": str(slim["date"].min()),
            "date_max": str(slim["date"].max()),
            "built_at": meta.get("built_at"),
            "race_rows": meta.get("race_rows"),
            "unique_horses": meta.get("unique_horses"),
        })
    except Exception as e:
        logger.error("pedigree-race-stats meta エラー: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pedigree-race-stats/query", response_class=JSONResponse)
async def api_pedigree_race_stats_query(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venues: Optional[str] = Query(None, description="カンマ区切り: 東京,中山"),
    surface: Optional[str] = Query(None, description="芝 / ダート / 障"),
    dist_min: Optional[int] = Query(None),
    dist_max: Optional[int] = Query(None),
    track_conditions: Optional[str] = Query(None, description="カンマ区切り: 良,稍重"),
    grades: Optional[str] = Query(None, description="カンマ区切り: G1,G2"),
    finish_min: Optional[int] = Query(None, description="着順下限（1着=1）"),
    finish_max: Optional[int] = Query(None, description="着順上限"),
    pop_min: Optional[int] = Query(None, description="人気下限（1番人気=1）"),
    pop_max: Optional[int] = Query(None, description="人気上限"),
    odds_min: Optional[float] = Query(None, description="単勝オッズ下限 (payoff 由来)"),
    odds_max: Optional[float] = Query(None, description="単勝オッズ上限 (payoff 由来)"),
    cat: Optional[str] = Query(None, description="1 / 2 / 3 / all"),
    gen_min: Optional[int] = Query(None, description="世代下限（1〜10）"),
    gen_max: Optional[int] = Query(None, description="世代上限（1〜10）"),
    count_mode: str = Query("unique", description="unique=ユニーク馬数 / appearances=出現回数（クロス込み）"),
    top_n: int = Query(50, description="上位N件"),
    exclude_trunk: bool = Query(False, description="根幹血統を除外（派生・非主流固有のみ表示）"),
):
    """
    条件で絞り込み後の血統カテゴリ別種牡馬カウント分布を返す（10世代対応）。

    count_mode:
      unique      - 各種牡馬が血統に含まれるユニーク馬数（同一馬内の重複は1カウント）
      appearances - 各種牡馬が血統ツリー上に出現した総回数（クロス = 重複出現も全てカウント）

    レスポンス形式:
    {
      "total_entries": 675,
      "unique_horses": 450,
      "cat1": [{"stallion_id": "...", "stallion_name": "...", "gen": 1, "count": 120}, ...],
      "cat2": [...],
      "cat3": [...],
    }
    """
    try:
        slim, cats, meta = _load_ped_race_index()
        if slim is None:
            return JSONResponse({
                "error": "インデックス未生成。python -m src.research.pedigree.build_pedigree_race_index を実行してください。"
            })

        df = slim.copy()

        # ── レースフィルタ ──
        if date_from:
            df = df[df["date"] >= date_from]
        if date_to:
            df = df[df["date"] <= date_to]
        if venues:
            vlist = [v.strip() for v in venues.split(",") if v.strip()]
            df = df[df["venue"].isin(vlist)]
        if surface:
            slist = [s.strip() for s in surface.split(",") if s.strip()]
            df = df[df["surface"].isin(slist)]
        if dist_min is not None:
            df = df[df["distance"] >= dist_min]
        if dist_max is not None:
            df = df[df["distance"] <= dist_max]
        if track_conditions:
            tclist = [t.strip() for t in track_conditions.split(",") if t.strip()]
            df = df[df["track_condition"].isin(tclist)]
        if grades:
            glist = [g.strip() for g in grades.split(",") if g.strip()]
            df = df[df["grade"].isin(glist)]

        # 成績計算は着順フィルタを適用しない全出走を母集団とする
        df_for_perf = df.copy()

        if finish_min is not None:
            df = df[df["finish_position"] >= finish_min]
        if finish_max is not None:
            df = df[df["finish_position"] <= finish_max]
        if pop_min is not None:
            df = df[df["popularity"] >= pop_min]
        if pop_max is not None:
            df = df[df["popularity"] <= pop_max]
        # 単勝オッズ範囲フィルタ (payoff JSON 由来の win_odds_real が真の値)
        if "win_odds_real" in df.columns:
            if odds_min is not None:
                df = df[df["win_odds_real"] >= float(odds_min)]
            if odds_max is not None:
                df = df[df["win_odds_real"] <= float(odds_max)]

        total_entries = int(len(df))
        unique_horses = int(df["horse_id"].nunique())

        if total_entries == 0:
            return JSONResponse({
                "total_entries": 0,
                "unique_horses": 0,
                "cat1": [], "cat2": [], "cat3": [],
            })

        # ── 血統カテゴリJOIN ──
        horse_ids = df["horse_id"].unique()
        ped_filtered = cats[cats["horse_id"].isin(horse_ids)].copy()

        # cat フィルタ
        cat_filter = None
        if cat and cat != "all":
            try:
                cat_filter = int(cat)
            except ValueError:
                pass
        if cat_filter:
            ped_filtered = ped_filtered[ped_filtered["cat"] == cat_filter]

        # 世代フィルタ
        if gen_min is not None:
            ped_filtered = ped_filtered[ped_filtered["gen"] >= gen_min]
        if gen_max is not None:
            ped_filtered = ped_filtered[ped_filtered["gen"] <= gen_max]

        # カウント集計（unique = ユニーク馬数 / appearances = 出現回数）
        use_appearances = (count_mode == "appearances")

        lineage_df = _load_stallion_lineage()

        # 根幹血統の stallion_id セットを事前構築
        _trunk_ids: "set[str] | None" = None
        if exclude_trunk and lineage_df is not None:
            import pandas as _pd_local
            lin = lineage_df
            mg_col = lin["main_group_id"] if "main_group_id" in lin.columns \
                else _pd_local.Series(-1, index=lin.index)
            trunk_mask = (
                lin["sub_group_label"].str.endswith("その他", na=False)
                | ((mg_col == 5) & (lin["sub_group_label"] == "非主流"))
            )
            _trunk_ids = set(lin.loc[trunk_mask, "stallion_id"].astype(str))

        # 成績集計用に「該当出走の rows」を最小限のカラムだけ保持（着順フィルタなし）
        df_perf_cols = ["horse_id", "finish_position"]
        for c in ("win_payout", "place_payout"):
            if c in df_for_perf.columns:
                df_perf_cols.append(c)
        df_perf = df_for_perf[df_perf_cols].copy()
        df_perf["horse_id"] = df_perf["horse_id"].astype(str)
        df_perf["finish_position"] = df_perf["finish_position"].astype(int)
        if "win_payout" not in df_perf.columns:
            df_perf["win_payout"] = 0
        if "place_payout" not in df_perf.columns:
            df_perf["place_payout"] = 0

        # 成績計算用血統インデックス（着順/人気/オッズフィルタ前の全出走馬ベース）
        _horse_ids_for_perf = set(df_for_perf["horse_id"].astype(str).unique())
        ped_for_perf = cats[cats["horse_id"].isin(_horse_ids_for_perf)].copy()
        if gen_min is not None:
            ped_for_perf = ped_for_perf[ped_for_perf["gen"] >= gen_min]
        if gen_max is not None:
            ped_for_perf = ped_for_perf[ped_for_perf["gen"] <= gen_max]

        def _aggregate_perf(horse_ids_set: set[str]) -> dict:
            """指定 horse_id 集合に対するレース成績統計をまとめて返す。

            返却:
              n: 出走数
              wins/p2/p3/p45/p6plus: 着順別カウント
              win_rate / top3_rate: 0-1 のレート
              win_recovery / place_recovery: 100円賭けあたり払戻% (= 平均払戻)
            """
            if not horse_ids_set:
                return {"n": 0}
            sub = df_perf[df_perf["horse_id"].isin(horse_ids_set)]
            # 除外/取消 (finish_position <= 0) を成績集計から除く
            sub = sub[sub["finish_position"] > 0]
            n = int(len(sub))
            if n == 0:
                return {"n": 0}
            fp = sub["finish_position"].to_numpy()
            wins = int((fp == 1).sum())
            p2 = int((fp == 2).sum())
            p3 = int((fp == 3).sum())
            p45 = int(((fp == 4) | (fp == 5)).sum())
            p6plus = int((fp >= 6).sum())
            top3 = wins + p2 + p3
            win_pay_sum = int(sub["win_payout"].sum())
            place_pay_sum = int(sub["place_payout"].sum())
            return {
                "n": n,
                "wins": wins, "p2": p2, "p3": p3, "p45": p45, "p6plus": p6plus,
                "win_rate": round(wins / n, 4),
                "top3_rate": round(top3 / n, 4),
                # 100 円賭けに対する平均払戻 (% 表記用に *1 = そのまま)
                "win_recovery": round(win_pay_sum / n, 1),
                "place_recovery": round(place_pay_sum / n, 1),
            }

        def _count_stallions(cat_val: int) -> list[dict]:
            subset = ped_filtered[ped_filtered["cat"] == cat_val].copy()
            if subset.empty:
                return []
            for col in ["stallion_id", "stallion_name"]:
                if hasattr(subset[col], "cat"):
                    subset[col] = subset[col].astype(str)

            # stallion_id が空文字の場合は stallion_name をキーとして使う
            key = subset["stallion_id"].where(
                subset["stallion_id"].str.len() > 0, subset["stallion_name"]
            )
            subset = subset.assign(_key=key)

            # キーごとに最短名（国名サフィックスなし版）を正規名として採用
            canon = (
                subset.groupby("_key", observed=True)["stallion_name"]
                .agg(lambda x: min(x, key=len))
                .rename("_canon")
            )
            subset = subset.join(canon, on="_key")

            if use_appearances:
                grp = (
                    subset.groupby(["_key", "_canon", "gen"], observed=True)
                    .size()
                    .reset_index(name="count")
                )
            else:
                grp = (
                    subset.groupby(["_key", "_canon", "gen"], observed=True)["horse_id"]
                    .nunique()
                    .reset_index(name="count")
                )
            grp = (
                grp.rename(columns={"_key": "stallion_id", "_canon": "stallion_name"})
                .sort_values("count", ascending=False)
            )
            # 根幹血統除外（top_n 適用前にフィルタして派生因子を確保）
            if _trunk_ids is not None:
                grp = grp[~grp["stallion_id"].isin(_trunk_ids)]
            grp = grp.head(top_n)
            grp["gen"] = grp["gen"].astype(int)

            # 成績 horse_id マッピング: top-N 種牡馬のみ・着順フィルタなし全出走馬ベース
            # （全 stallion に groupby-apply するとメモリ/速度問題が出るため top-N に絞る）
            top_sids = set(grp["stallion_id"].tolist())
            _perf_sub = ped_for_perf[ped_for_perf["cat"] == cat_val]
            key_to_horses: dict = {}
            if not _perf_sub.empty and top_sids:
                _sid   = _perf_sub["stallion_id"].astype(str)
                _sname = _perf_sub["stallion_name"].astype(str)
                _pkey  = _sid.where(_sid.str.len() > 0, _sname)
                _mask  = _pkey.isin(top_sids).values
                for k, h in zip(_pkey.values[_mask],
                                _perf_sub["horse_id"].astype(str).values[_mask]):
                    if k not in key_to_horses:
                        key_to_horses[k] = set()
                    key_to_horses[k].add(h)
            perf_records = []
            for sid in grp["stallion_id"]:
                perf_records.append(_aggregate_perf(key_to_horses.get(sid, set())))
            for k in ("n", "wins", "p2", "p3", "p45", "p6plus",
                     "win_rate", "top3_rate", "win_recovery", "place_recovery"):
                grp[k] = [r.get(k, 0) for r in perf_records]

            # 系統グループ情報を JOIN
            if lineage_df is not None:
                lin_cols = [c for c in ["stallion_id", "anchor_name", "group_id",
                                        "depth_to_anchor", "main_group_id",
                                        "main_group_name", "sub_group_label"]
                            if c in lineage_df.columns]
                grp = grp.merge(lineage_df[lin_cols], on="stallion_id", how="left")
                grp["group_id"]       = grp["group_id"].fillna(-1).astype(int)
                if "depth_to_anchor" in grp.columns:
                    grp["depth_to_anchor"] = grp["depth_to_anchor"].fillna(0).astype(int)
                else:
                    grp["depth_to_anchor"] = 0
                grp["anchor_name"]    = grp["anchor_name"].fillna("").astype(str)
                if "main_group_id" in grp.columns:
                    grp["main_group_id"]  = grp["main_group_id"].fillna(-1).astype(int)
                else:
                    grp["main_group_id"]  = -1
                if "main_group_name" in grp.columns:
                    grp["main_group_name"] = grp["main_group_name"].fillna("").astype(str)
                else:
                    grp["main_group_name"] = ""
                if "sub_group_label" in grp.columns:
                    grp["sub_group_label"] = grp["sub_group_label"].fillna("").astype(str)
                else:
                    grp["sub_group_label"] = ""
            else:
                grp["group_id"]        = -1
                grp["anchor_name"]     = ""
                grp["depth_to_anchor"] = 0
                grp["main_group_id"]   = -1
                grp["main_group_name"] = ""
                grp["sub_group_label"] = ""

            return grp.to_dict(orient="records")

        result: dict = {
            "total_entries": total_entries,
            "unique_horses": unique_horses,
        }
        if not cat_filter or cat_filter == 1:
            result["cat1"] = _count_stallions(1)
        if not cat_filter or cat_filter == 2:
            result["cat2"] = _count_stallions(2)
        if not cat_filter or cat_filter == 3:
            result["cat3"] = _count_stallions(3)

        return JSONResponse(result)

    except Exception as e:
        logger.error("pedigree-race-stats query エラー: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pedigree-race-stats/lineage-meta", response_class=JSONResponse)
async def api_pedigree_race_stats_lineage_meta():
    """
    系統グループのメタ情報を返す。
    {
      "groups": [{"group_id": 0, "anchor_name": "Pharos", "anchor_id": "...", "count": 7}, ...]
    }
    """
    try:
        meta_path = os.path.join(_PED_RACE_IDX_DIR, "stallion_lineage_meta.json")
        if not os.path.exists(meta_path):
            return JSONResponse({"groups": []})
        with open(meta_path, encoding="utf-8") as f:
            groups = json.loads(f.read())
        # group_id=-1 は除外; 新スキーマに合わせて不足フィールドを補完
        out = []
        for g in groups:
            if g.get("group_id", -1) < 0:
                continue
            out.append({
                "group_id":        g.get("group_id"),
                "main_group_id":   g.get("main_group_id", -1),
                "main_group_name": g.get("main_group_name", ""),
                "anchor_name":     g.get("anchor_name", ""),
                "count":           g.get("count", 0),
                "color":           g.get("color", "#888"),
                "main_color":      g.get("main_color", "#888"),
            })
        return JSONResponse({"groups": out})
    except Exception as e:
        logger.error("lineage-meta エラー: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
