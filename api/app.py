"""
FastAPI アプリケーション

予測結果の表示・エージェントステータス・追加要件API・MLflow モデル予測APIを提供する。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from contextlib import asynccontextmanager
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from api.auth import (
    is_developer, requires_auth, is_public_path,
    verify_password, create_session_response, clear_session_response,
    COOKIE_NAME,
)


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
_queue_worker_thread: threading.Thread | None = None
_queue_worker_stop = threading.Event()
# マルチワーカー時は flock で1プロセスだけがキューループを回す（JSON 競合防止）
_queue_runner_leader_fh = None

# ── data/cache 定期クリーンアップ（HybridStorage.cleanup_disk_cache） ──
_disk_cache_cleanup_thread: threading.Thread | None = None
_disk_cache_cleanup_stop = threading.Event()


def _queue_worker_loop():
    """スクレイピングキューを処理するワーカー"""
    from scraper.job_queue import ScrapeJobQueue

    _worker_log = logging.getLogger("queue.worker")
    _worker_log.setLevel(logging.INFO)
    if not _worker_log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        _worker_log.addHandler(_h)
    _worker_log.info("スクレイピングキューワーカー起動")

    queue = ScrapeJobQueue()

    while not _queue_worker_stop.is_set():
        try:
            next_job = queue.get_next_job()
            if not next_job:
                queue.requeue_stale_running_jobs(assume_lock_holder=False)
                next_job = queue.get_next_job()
            if next_job:
                # 別スレッド（API からの process_queue 等）が .scrape.lock 中だと
                # process_queue は即 return するため、待機しないと無限ループでログが爆発する
                if queue.is_locked():
                    if _queue_worker_stop.wait(timeout=2):
                        break
                    continue
                _worker_log.info(
                    "ジョブ処理開始: %s",
                    next_job.get("job_label")
                    or next_job.get("target_id")
                    or next_job.get("race_id")
                    or "?",
                )
                queue.process_queue()
            else:
                # ジョブがない場合は30秒待機
                if _queue_worker_stop.wait(timeout=30):
                    break
        except Exception as e:
            _worker_log.error("ワーカーエラー: %s", e, exc_info=True)
            if _queue_worker_stop.wait(timeout=10):
                break

    _worker_log.info("スクレイピングキューワーカー終了")


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
        _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
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
            from scraper.structure_monitor import run_daily_check
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
    from scraper.job_queue import QUEUE_DIR

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
    global _scheduler_thread, _queue_worker_thread, _queue_runner_leader_fh, _disk_cache_cleanup_thread

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

    # スクレイピングキューワーカー（マルチワーカー時はリーダー1プロセスのみ）
    if _try_acquire_queue_runner_leader():
        _queue_worker_stop.clear()
        _queue_worker_thread = threading.Thread(
            target=_queue_worker_loop, daemon=True, name="queue-worker"
        )
        _queue_worker_thread.start()
        logger.info("スクレイピングキューワーカー起動（当プロセスがリーダー）")
    else:
        _queue_worker_thread = None
        logger.info("スクレイピングキューワーカーは別ワーカーが担当（スキップ）")

    try:
        from scraper.queue_worker_log import ensure_queue_worker_log_handler

        ensure_queue_worker_log_handler()
    except Exception as _e:
        logger.warning("キューワーカー用ログバッファの初期化をスキップ: %s", _e)

    # キュー関連ルートの存在確認（古いプロセスが動いていると UI のタスク一覧が 404 になる）
    _queue_api_required = (
        "/api/scrape-queue/tasks",
        "/api/scrape-queue/add-job",
        "/api/scrape-queue/enqueue-period-horses",
        "/api/scrape-queue/enqueue-period-races",
        "/api/scrape-queue/worker-logs",
        "/api/scrape-queue/resume",
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

    yield

    # 終了処理
    _scheduler_stop.set()
    _queue_worker_stop.set()
    _disk_cache_cleanup_stop.set()

    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)
    if _queue_worker_thread:
        _queue_worker_thread.join(timeout=5)
    if _disk_cache_cleanup_thread:
        _disk_cache_cleanup_thread.join(timeout=5)
    if _queue_runner_leader_fh:
        try:
            _queue_runner_leader_fh.close()
        except OSError:
            pass
        _queue_runner_leader_fh = None


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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
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
        return create_session_response(redirect_to=next_url)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "current_page": "login",
        "breadcrumbs": [],
        "next_url": next_url,
        "error": "パスワードが正しくありません",
    })


@app.get("/logout")
async def logout():
    return clear_session_response(redirect_to="/")


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
        "gcs_healthy": storage._gcs_healthy,
        "gcs_backoff_remaining_s": max(0, round(
            storage._gcs_backoff - (_time.time() - storage._gcs_last_failure), 1
        )) if storage._gcs_last_failure > 0 else 0,
        "load_cache_size": len(storage._load_cache),
        "load_cache_max": storage._LOAD_CACHE_MAX,
        "load_cache_ttl_s": storage._LOAD_CACHE_TTL,
        "blob_list_cache_size": len(storage._blob_list_cache),
        "blob_list_cache_ttl_current_s": storage._BLOB_LIST_TTL_CURRENT,
        "blob_list_cache_ttl_past_s": storage._BLOB_LIST_TTL_PAST,
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
    from scraper.storage import HybridStorage
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
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        _log.addHandler(_h)
    interval = float(_os.environ.get("DISK_CACHE_CLEANUP_INTERVAL_SEC", "86400"))
    _log.info(
        "data/cache クリーンアップ: 初回即時、その後 %.0fs ごと (max_age=%s)",
        interval,
        _os.environ.get("DISK_CACHE_CLEANUP_MAX_AGE_SEC", "604800"),
    )
    while not _disk_cache_cleanup_stop.is_set():
        try:
            r = _get_storage().cleanup_disk_cache()
            _log.info("disk cache cleanup: %s", r)
        except Exception as e:
            _log.warning("disk cache cleanup 失敗: %s", e)
        if _disk_cache_cleanup_stop.wait(timeout=interval):
            break
    _log.info("data/cache クリーンアップスレッド終了")


def _kick_scrape_queue_worker() -> None:
    """ロックが空いていればキューワーカーを起動（最優先ジョブを含む）。"""
    try:
        from scraper.job_queue import ScrapeJobQueue

        q = ScrapeJobQueue()
        if not q.is_locked():
            asyncio.create_task(asyncio.to_thread(q.process_queue))
    except Exception as e:
        logger.warning("キューワーカー起動スキップ: %s", e)


_html_archive_lock = threading.Lock()


def _get_html_archive():
    """HtmlArchive のシングルトンインスタンスを返す。"""
    from scraper.html_archive import HtmlArchive
    if not hasattr(_get_html_archive, "_inst"):
        with _html_archive_lock:
            if not hasattr(_get_html_archive, "_inst"):
                _get_html_archive._inst = HtmlArchive()
    return _get_html_archive._inst


@app.get("/monitor", response_class=HTMLResponse)
async def scrape_monitor_page(request: Request):
    """スクレイピング状態のリアルタイムモニタリングボード"""
    return templates.TemplateResponse("monitor.html", {
        "request": request,
        "current_page": "monitor",
        "breadcrumbs": [],
    })


MONITOR_SOURCES = [
    "race_shutuba",
    "race_result",
    "race_index",
    "race_shutuba_past",
    "race_odds",
    "race_paddock",
    "race_barometer",
    "race_oikiri",
    "race_trainer_comment",
    "smartrc_race",
]

_STEEPLECHASE_NA_CATS = frozenset({
    "race_index", "race_barometer", "race_oikiri", "smartrc_race",
})

# 障害レースかどうかを race_name から判定
import re as _re
_STEEPLECHASE_PATTERN = _re.compile(r"障害|ジャンプ|ハードル|スティープル")


# ── レスポンスキャッシュ (stale-while-revalidate) ──────────────────

_status_cache: dict[str, tuple[float, dict]] = {}
_status_cache_lock = threading.Lock()
_STATUS_CACHE_TTL = 300       # fresh: 5 minutes
_STATUS_CACHE_STALE = 3600    # stale-while-revalidate: return stale up to 1 hour
_status_bg_refreshing: set[str] = set()
_status_bg_refreshing_lock = threading.Lock()

_horse_ids_cache: dict[str, list[str]] = {}
_horse_ids_cache_lock = threading.Lock()


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
            "sources": MONITOR_SOURCES + ["horse_result", "horse_pedigree"],
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
            "sources": MONITOR_SOURCES + ["horse_result", "horse_pedigree"],
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
        sources["horse_pedigree"] = {
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
        for hcat in ("horse_result", "horse_pedigree"):
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
        "sources": MONITOR_SOURCES + ["horse_result", "horse_pedigree"],
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
        from scraper.monitor_backlog import summarize_missing_dates

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
_RACE_LIST_STEMS_TTL = 600.0

# picker_past_days 応答の短時間キャッシュ
_PICKER_SCRAPE_DATES_CACHE: dict[str, Any] = {"key": None, "data": None, "ts": 0.0}
_PICKER_SCRAPE_DATES_TTL = 60.0

_NO_STORE_HEADERS = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}


def _cached_race_list_stems(storage) -> list[str]:
    """race_lists のキーを stem（通常 YYYYMMDD）の昇順で返す。5分キャッシュ。"""
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
        description="UI 用: 直近 N 日は data/local/race_lists のファイル存在だけ確認（全件 glob しない）",
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

            storage = _get_storage()
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            race_dir = Path(__file__).resolve().parent.parent / "data" / "local" / "race_lists"
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

        from scraper.monitor_future_eligible import include_date_in_data_viewer_race_list

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
async def get_race_list_for_date(date: str):
    """指定日付のレース一覧を会場別に返す。"""
    def _load():
        storage = _get_storage()
        data = storage.load("race_lists", date)
        if not data:
            return {"date": date, "venues": [], "races": []}
        raw = data.get("races", [])
        jra = [r for r in raw if r.get("race_id") and _is_jra_race(r["race_id"])]
        venues = sorted(set(r.get("venue", "") for r in jra if r.get("venue")))
        races = []
        for r in sorted(jra, key=lambda x: (x.get("venue", ""), x.get("round", 0))):
            races.append({
                "race_id": r["race_id"],
                "round": r.get("round", 0),
                "venue": r.get("venue", ""),
                "race_name": r.get("race_name", ""),
            })
        return {"date": date, "venues": venues, "races": races}
    result = await asyncio.to_thread(_load)
    return JSONResponse(result)


_scrape_summary_cache: dict = {"data": None, "ts": 0.0, "gen": 0}
_SCRAPE_SUMMARY_FILTER_GEN = 3  # フィルタ条件変更時にインクリメントしてキャッシュ無効化


_SUMMARY_SOURCES = [
    "race_shutuba", "race_result", "race_index", "race_shutuba_past",
    "race_odds", "race_paddock", "race_barometer", "race_oikiri",
    "race_trainer_comment", "smartrc_race",
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

    from scraper.monitor_future_eligible import include_date_in_monitor_summary

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

    smart_skip = body.get("smart_skip", True)
    if isinstance(smart_skip, str):
        smart_skip = smart_skip.strip().lower() not in ("0", "false", "no")

    from scraper.job_queue import ScrapeJobQueue

    queue = ScrapeJobQueue()
    created = requeued = duplicate = 0
    details: list[dict[str, Any]] = []
    for date in to_enqueue:
        result = queue.add_job({
            "job_kind": "date",
            "target_id": date,
            "tasks": ["date_all"],
            "smart_skip": bool(smart_skip),
        })
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
    "race_shutuba", "race_result", "race_index", "race_shutuba_past",
    "race_odds", "race_paddock", "race_barometer", "race_oikiri",
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
    local_path = Path(f"data/meta/person/{ptype}_{person_id}.json")
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
    return templates.TemplateResponse("data_viewer.html", {
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
                from scraper.run import ScraperRunner
                _get_runner._inst = ScraperRunner(
                    interval=1.0, cache=True, auto_login=True,
                )
    return _get_runner._inst


CATEGORY_TO_METHOD = {
    "race_shutuba":      "scrape_race_card",
    "race_result":       "scrape_race_result",
    "race_index":        "scrape_speed_index",
    "race_shutuba_past": "scrape_shutuba_past",
    "race_odds":         "scrape_odds",
    "race_paddock":      "scrape_paddock",
    "race_barometer":    "scrape_barometer",
    "race_oikiri":       "scrape_oikiri",
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
    """実行中・キュー待ち・完了済みのスクレイピングジョブ一覧と統計を返す。"""
    now = _time.time()
    with _scrape_jobs_lock:
        active = {k: v for k, v in _scrape_jobs.items()
                  if v["status"] in ("running", "queued")
                  or (v["finished_at"] and now - v["finished_at"] < 120)}
        running_count = sum(1 for j in _scrape_jobs.values()
                           if j["status"] == "running")
        queued_count = sum(1 for j in _scrape_jobs.values()
                          if j["status"] == "queued")
    runner = _get_runner() if hasattr(_get_runner, "_inst") else None
    req_count = runner.client.request_count if runner else 0
    return JSONResponse({
        "jobs": list(active.values()),
        "stats": {
            "running": running_count,
            "queued": queued_count,
            "max_concurrent": _MAX_CONCURRENT_SCRAPE,
            "slots_available": max(0, _MAX_CONCURRENT_SCRAPE - running_count),
            "total_requests": req_count,
        },
    })


# ═══════════════════════════════════════════════════════
# 未来のレース & スクレイピングキュー管理
# ═══════════════════════════════════════════════════════

# 既定ウィンドウ（start/end 未指定）の upcoming 応答を短時間キャッシュ（netkeiba 負荷・体感速度）
_UPCOMING_RACES_DEFAULT_CACHE: dict[str, Any] = {"payload": None, "ts": 0.0}
_UPCOMING_RACES_DEFAULT_TTL = 300.0


@app.get("/api/upcoming-races", response_class=JSONResponse)
async def get_upcoming_races(start_date: str = None, end_date: str = None):
    """
    未来のレース一覧を取得（JRA は使わない）。

    各日について data/local/race_lists/{YYYYMMDD}.json が存在し、
    件数が is_plausible_race_day_races（中止で 12 の倍数でない日も妥当）ならそれを優先
    （カレンダー取得との整合・ポーリング負荷軽減）。
    極端に少ない件数などのファイルは無視し netkeiba から再取得する。

    上記が無い場合は netkeiba トップと同じ経路（race_list_get_date_list + race_list_sub）で取得。

    Args:
        start_date: 開始日 (YYYY-MM-DD形式、未指定時は明日)
        end_date: 終了日 (YYYY-MM-DD形式、未指定時は7日後)
    """
    now = _time.time()
    default_window = start_date is None and end_date is None
    if default_window:
        uc = _UPCOMING_RACES_DEFAULT_CACHE
        if uc["payload"] is not None and (now - float(uc["ts"])) < _UPCOMING_RACES_DEFAULT_TTL:
            return JSONResponse(uc["payload"])

    def _fetch():
        from datetime import datetime, timedelta
        import json
        from pathlib import Path

        from scraper.client import NetkeibaClient
        from scraper.netkeiba_top_race_list import (
            fetch_races_for_kaisai_date,
            is_plausible_race_day_races,
        )
        from scraper.monitor_future_eligible import include_date_in_data_viewer_race_list

        BASE_DIR = Path(__file__).parent.parent
        RACE_LIST_DIR = BASE_DIR / "data" / "local" / "race_lists"

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
        payload = {"races": races, "start_date": start_date, "end_date": end_date}
        if default_window:
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

        from scraper.job_queue import ScrapeJobQueue

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
        from scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        # ロック無しで running だけ残る孤児は status 取得時に pending へ戻し、UI とワーカーが復帰できるようにする
        if not queue.is_locked():
            jobs_preview = queue.load_queue()
            if any(j.get("status") == "running" for j in jobs_preview) and not any(
                j.get("status") == "pending" for j in jobs_preview
            ):
                queue.requeue_stale_running_jobs(assume_lock_holder=False)

        status = queue.get_status()

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

        return JSONResponse(status)

    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.get("/api/scrape-queue/worker-logs", response_class=JSONResponse)
async def api_scrape_queue_worker_logs(
    after: int = Query(-1, description="増分: 前回の max_id。初回は -1"),
    limit: int = Query(300, ge=1, le=800),
):
    """
    キュー実行スレッド中の scraper.* / queue.* / urllib3 等のログ行（メモリリング）。
    """
    try:
        from scraper.queue_worker_log import ensure_queue_worker_log_handler, get_worker_logs

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
        from scraper.queue_worker_log import clear_worker_logs

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
    アクセス系エラーで自動停止したキューを再開する（一時停止フラグを解除）。
    ジョブの pending / failed は変更しない。
    """
    try:
        from scraper.scrape_access_pause import clear_access_pause, read_access_pause

        clear_access_pause()
        return JSONResponse({"status": "ok", "transport_pause": read_access_pause()})
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/api/scrape-queue/tasks", response_class=JSONResponse)
async def scrape_queue_task_catalog():
    """キューに載せられるタスク一覧（UI・API用）。"""
    try:
        from scraper.queue_tasks import catalog_for_api

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
        from scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        try:
            result = queue.add_job(body)
        except ValueError as ve:
            return JSONResponse({"error": str(ve)}, status_code=400)

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

    Body (任意): force_summary (bool|1), max_dates (default 500), smart_skip (default true)
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
        from scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
        queue.clear_completed()

        return JSONResponse({"status": "cleared"})

    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


def _sync_scrape_queue_stop_and_clear() -> dict[str, Any]:
    """
    実行中ワーカーが次ループでジョブを取らないよう一瞬一時停止を掛け、
    全ジョブ削除後に必ず一時停止を解除する（残すと get_next_job が常に None で固まる）。
    """
    from scraper.job_queue import ScrapeJobQueue
    from scraper.scrape_access_pause import (
        clear_access_pause,
        read_access_pause,
        write_access_pause,
    )

    write_access_pause(
        reason="ユーザー操作: キュー全消去処理中（完了後に自動解除されます）"
    )
    queue = ScrapeJobQueue()
    removed = queue.clear_all_jobs()
    if queue.lock_file.exists() and not queue.is_locked():
        try:
            queue.lock_file.unlink()
        except OSError:
            pass
    clear_access_pause()
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
        from scraper.job_queue import ScrapeJobQueue

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
        from scraper.job_queue import ScrapeJobQueue

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
        
        from scraper.job_queue import ScrapeJobQueue
        
        queue = ScrapeJobQueue()
        created = 0
        requeued = 0
        duplicate = 0
        processed = 0

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
    body: start_date?, end_date? (YYYYMMDD), limit (default 200), dry_run (bool)
    """
    from scraper.job_queue import ScrapeJobQueue
    from scraper.missing_races import find_missing_jra_races_for_queue

    start_date = body.get("start_date") or None
    end_date = body.get("end_date") or None
    limit = int(body.get("limit") or 200)
    limit = max(1, min(limit, 2000))
    dry_run = bool(body.get("dry_run"))

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
    body: start_date?, end_date? (YYYYMMDD), tasks (list[str]), limit, dry_run, jra_only?
    """
    from scraper.job_queue import ScrapeJobQueue
    from scraper.period_runners import enqueue_horse_tasks_for_race_period

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
        return {"error": "tasks に馬エンティティのタスクIDを1つ以上指定してください（例: horse_pedigree）"}

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
        )
    except ValueError as ve:
        return {"error": str(ve)}

    result["status"] = "success"
    return result


def _sync_enqueue_period_race_tasks(body: dict) -> dict:
    """
    race_lists 上の期間内 JRA レースに対し、レース単位タスクをキューへ。
    body: start_date?, end_date? (YYYYMMDD), tasks (list[str]), limit, dry_run, jra_only?
    """
    from scraper.job_queue import ScrapeJobQueue
    from scraper.period_runners import enqueue_race_tasks_for_race_period

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
        )
    except ValueError as ve:
        return {"error": str(ve)}

    result["status"] = "success"
    return result


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
        
        from scraper.missing_races import scrape_status_detail

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
        from scraper.client import NetkeibaClient
        from scraper.netkeiba_top_race_list import (
            fetch_races_for_kaisai_date,
            invalidate_race_list_cache,
        )

        BASE_DIR = Path(__file__).parent.parent
        RACE_LIST_DIR = BASE_DIR / "data" / "local" / "race_lists"
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
    return templates.TemplateResponse("race_detail.html", {
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
        data_availability = {}
        for cat in MONITOR_SOURCES:
            data_availability[cat] = {
                "json": result.get(cat) is not None,
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
async def run_race_prediction(race_id: str):
    """
    指定レースの AI 予測を実行し、結果を GCS に保存して返す。

    フロー:
      1. scrape_race_all で全データ取得 (smart_skip=True)
      2. feature_builder で特徴量テーブル構築
      3. モデル予測 (MLflow) or フォールバック (ヒューリスティック)
      4. 推奨印 (◎○▲△☆) 付与
      5. GCS に race_predictions/{race_id}.json として保存
    """
    def _run():
        import time as _t
        import traceback
        from pipeline.feature_builder import build_race_features

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
            from pipeline.race_day import RaceDayPipeline, PipelineConfig
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
            from pipeline.race_day import RaceDayPipeline
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
        from pipeline.odds_predictor import get_predicted_odds
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

        entries = _compute_composite_scores(base_entries, odds_map)
        # ソースを判定
        sources = set(v.get("source", "") for v in odds_map.values())
        odds_source = "/".join(sorted(sources)) if sources else "none"
        mark_method = f"composite ({odds_source})"

        feature_highlights = _extract_feature_highlights(features_df, entries)

        elapsed = round(_t.time() - t_start, 2)

        comp_params = _load_composite_params()
        from pipeline.composite_optimizer import OPTIM_RESULT_PATH
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


@app.get("/api/race/{race_id}/tracking-difficulty", response_class=JSONResponse)
async def api_tracking_difficulty(race_id: str):
    """レースの追走難度・ペース予想・位置取り予測を返す（キャッシュ済みデータ優先で高速応答）。"""
    def _run():
        from pipeline.tracking_difficulty import (
            predict_tracking_difficulty,
            predict_race_pace,
            predict_position_flow,
            build_horse_profile,
        )

        storage = _get_storage()

        race_data = {}
        for cat_key, storage_key in [
            ("race_shutuba", "race_shutuba"),
            ("race_result", "race_result"),
        ]:
            d = storage.load(storage_key, race_id)
            if d:
                race_data[cat_key] = d

        horses_data = {}
        entries = (
            (race_data.get("race_shutuba") or {}).get("entries")
            or (race_data.get("race_result") or {}).get("entries")
            or []
        )
        for e in entries:
            hid = e.get("horse_id", "")
            if hid:
                hr = storage.load("horse_result", hid)
                if hr and "race_history" in hr:
                    horses_data[hid] = hr["race_history"]

        if not race_data:
            runner = _get_runner()
            race_data = runner.scrape_race_all(race_id, smart_skip=True)
            horses_data = None

        # 追走難度の予測
        tracking_results = predict_tracking_difficulty(
            race_data,
            horse_histories=horses_data,
            storage=storage,
        )

        # 馬プロファイルの構築
        horse_profiles = {}
        for e in entries:
            hid = e.get("horse_id", "")
            hn = e.get("horse_number", 0)
            history = horses_data.get(hid, []) if horses_data else []
            horse_profiles[hn] = build_horse_profile(history)

        shutuba = race_data.get("race_shutuba") or race_data.get("race_card") or {}
        result_data = race_data.get("race_result") or {}
        race_info = {
            "distance": shutuba.get("distance", 0) or result_data.get("distance", 0),
            "surface": shutuba.get("surface", "") or result_data.get("surface", ""),
            "track_condition": (
                shutuba.get("track_condition", "")
                or result_data.get("track_condition", "良")
            ),
        }

        # ペース予想
        pace_prediction = predict_race_pace(entries, horse_profiles, race_info)

        # 位置取り予測
        position_flow = predict_position_flow(
            entries, horse_profiles, tracking_results, pace_prediction
        )

        return {
            "race_id": race_id,
            "race_name": shutuba.get("race_name", "") or result_data.get("race_name", ""),
            "venue": shutuba.get("venue", "") or result_data.get("venue", ""),
            "surface": race_info["surface"],
            "distance": race_info["distance"],
            "track_condition": race_info["track_condition"],
            "field_size": len(tracking_results),
            "pace_prediction": pace_prediction,
            "position_flow": position_flow,
            "entries": tracking_results,
        }

    try:
        result = await asyncio.to_thread(_run)
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status_code=500)


@app.post("/api/tracking-difficulty/train", response_class=JSONResponse)
async def api_train_tracking_difficulty(request: Request):
    """追走難度モデルの学習を実行する。"""
    def _train():
        from pipeline.tracking_difficulty import TrackingDifficultyTrainer
        trainer = TrackingDifficultyTrainer(mlflow_tracking_uri="mlruns")
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
        from pipeline.composite_optimizer import get_composite_params
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
    複勝率と期待値のトレードオフで印を割り当てる。

    composite = prob^α × EV^(1-α)    (幾何平均ベース)

    α, 閾値パラメータはシミュレーション最適化結果から自動読込する。
    最適化が未実行ならデフォルト値を使用。

    ◎○ には複勝率の最低閾値を設け、低確率の大穴だけに
    ◎が付く事態を防ぐ。
    """
    params = _load_composite_params()
    alpha = params.get("prob_weight", _DEFAULT_PROB_WEIGHT)
    honmei_ratio = params.get("min_prob_honmei_ratio", _DEFAULT_MIN_PROB_HONMEI_RATIO)
    taikou_ratio = params.get("min_prob_taikou_ratio", _DEFAULT_MIN_PROB_TAIKOU_RATIO)

    scored: list[dict] = []

    for e in entries:
        hn = e["horse_number"]
        prob = e.get("normalized_score", 0)
        odds_info = odds_map.get(hn, {})

        place_min = odds_info.get("predicted_place_odds_min") or odds_info.get("place_odds_min", 0) or 0
        place_max = odds_info.get("predicted_place_odds_max") or odds_info.get("place_odds_max", 0) or 0
        win_odds = odds_info.get("predicted_win_odds") or odds_info.get("win_odds", 0) or 0
        odds_confidence = odds_info.get("confidence", 1.0)
        odds_source = odds_info.get("source", "live")
        place_avg = (place_min + place_max) / 2 if (place_min and place_max) else 0

        if prob > 0 and place_avg > 0:
            ev = prob * place_avg
            composite = (prob ** alpha) * (ev ** (1 - alpha))
        elif prob > 0:
            ev = 0
            composite = prob
        else:
            ev = 0
            composite = 0

        scored.append({
            **e,
            "place_odds_min": round(place_min, 1) if place_min else None,
            "place_odds_max": round(place_max, 1) if place_max else None,
            "place_odds_avg": round(place_avg, 1) if place_avg else None,
            "win_odds": round(win_odds, 1) if win_odds else None,
            "odds_confidence": round(odds_confidence, 2),
            "odds_source": odds_source,
            "estimated_prob": round(prob, 4),
            "expected_value": round(ev, 3) if ev else None,
            "composite_score": round(composite, 4),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    n = len(scored)
    base_prob = 1.0 / max(n, 1)
    min_prob_honmei = base_prob * honmei_ratio
    min_prob_taikou = base_prob * taikou_ratio

    assigned: set[str] = set()
    for i, s in enumerate(scored):
        prob = s["estimated_prob"]
        if "◎ 本命" not in assigned and prob >= min_prob_honmei:
            s["recommendation"] = "◎ 本命"
            assigned.add("◎ 本命")
        elif "○ 対抗" not in assigned and prob >= min_prob_taikou:
            s["recommendation"] = "○ 対抗"
            assigned.add("○ 対抗")
        elif "▲ 単穴" not in assigned:
            s["recommendation"] = "▲ 単穴"
            assigned.add("▲ 単穴")
        elif i < max(5, n // 3):
            s["recommendation"] = "△ 連下"
        else:
            s["recommendation"] = "☆ 穴馬"
        s["composite_rank"] = i + 1

    # ◎○▲ が1つも付かなかった場合のフォールバック
    for mark in ["◎ 本命", "○ 対抗", "▲ 単穴"]:
        if mark not in assigned and scored:
            for s in scored:
                if s["recommendation"] not in assigned:
                    s["recommendation"] = mark
                    assigned.add(mark)
                    break

    return scored


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
    check_path = Path(BASE_DIR) / "data" / "meta" / "structure" / "last_check.json"
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
    report_path = Path(BASE_DIR) / "data" / "meta" / "structure" / "report.md"
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
            from scraper.structure_monitor import run_daily_check
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
    last_check_path = Path(BASE_DIR) / "data" / "meta" / "structure" / "last_check.json"
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


@app.get("/api/structure-fingerprints", response_class=JSONResponse)
async def get_structure_fingerprints():
    """保存済みの全カテゴリのフィンガープリントを返す。"""
    from pathlib import Path
    fp_dir = Path(BASE_DIR) / "data" / "meta" / "structure"
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
        from pipeline.trainer import ModelTrainer
        import yaml
        _cfg = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")) as f:
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
        from pipeline.ensemble_trainer import EnsembleTrainer
        import yaml
        _cfg = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")) as f:
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

        from pipeline.trainer import ModelTrainer
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
        from pipeline.odds_predictor import OddsPredictor
        predictor = OddsPredictor()
        result = predictor.train(_get_storage())
        _odds_training_job["result"] = result
        _odds_training_job["error"] = result.get("error")
    except Exception as e:
        _odds_training_job["error"] = str(e)
    finally:
        _odds_training_job["running"] = False
        _odds_training_job["finished_at"] = datetime.now().isoformat()


@app.get("/api/odds/train/status", response_class=JSONResponse)
async def get_odds_training_status():
    """オッズ予測モデルの学習状態を返す。"""
    return JSONResponse({
        "running": _odds_training_job.get("running", False),
        "started_at": _odds_training_job.get("started_at", ""),
        "finished_at": _odds_training_job.get("finished_at", ""),
        "result": _odds_training_job.get("result"),
        "error": _odds_training_job.get("error"),
    })


@app.post("/api/odds/snapshot/{race_id}", response_class=JSONResponse)
async def record_odds_snapshot(race_id: str):
    """指定レースの現在オッズを取得し、推移履歴に記録する。"""
    scraper = _get_runner()
    odds = scraper.scrape_odds(race_id, skip_existing=False)
    if not odds or not odds.get("entries"):
        return JSONResponse({"error": "オッズ取得失敗"}, status_code=404)

    from pipeline.odds_predictor import OddsTrajectoryTracker
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
    from pipeline.odds_predictor import OddsTrajectoryTracker, ODDS_HISTORY_DIR
    history_path = ODDS_HISTORY_DIR / f"{race_id}.json"
    if not history_path.exists():
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
        from pipeline.feature_builder import build_race_features

        st = _get_storage()
        race_data = {}
        for cat in ["race_shutuba", "race_result", "race_card"]:
            d = st.load(cat, race_id)
            if d:
                race_data[cat] = d

        features_df = build_race_features(race_data)
        if features_df.empty:
            return JSONResponse({"error": "特徴量構築失敗"}, status_code=400)

        from pipeline.odds_predictor import get_predicted_odds
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
        from pipeline.composite_optimizer import CompositeOptimizer, SimulationConfig
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
    from pipeline.composite_optimizer import OPTIM_RESULT_PATH, get_composite_params

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
    from scraper.backfill import BackfillProgress, _generate_race_dates, DEFAULT_START_YEAR
    from scraper.backfill import LOCK_DIR, _pid_alive

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

    cmd = [sys.executable, "-m", "scraper.backfill",
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
    return Path(__file__).resolve().parent.parent / "logs" / "race_lists_backfill.lock"


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

    base = Path(__file__).resolve().parent.parent
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
    """scripts/backfill_race_lists_kaisai_since_2020.py をバックグラウンド起動。"""
    import subprocess
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent
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

    script = base / "scripts" / "backfill_race_lists_kaisai_since_2020.py"
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

    base = Path(__file__).resolve().parent.parent
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
    return templates.TemplateResponse("betting.html", {
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
    from pipeline.betting import BettingOptimizer, BettingConfig

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
    return templates.TemplateResponse("tracking_difficulty.html", {
        "request": request,
        "current_page": "tracking_difficulty",
        "breadcrumbs": [],
    })


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


@app.get("/queue-status", response_class=HTMLResponse)
async def queue_status_page(request: Request):
    from scraper.queue_tasks import catalog_for_api

    from_path = _url_path_prefix_before_suffix(request.url.path, "/queue-status")
    root = (request.scope.get("root_path") or "").rstrip("/")
    api_prefix = from_path or root
    return templates.TemplateResponse("queue_status.html", {
        "request": request,
        "current_page": "queue_status",
        "breadcrumbs": [],
        "api_prefix": api_prefix,
        "task_catalog": catalog_for_api(),
    })


@app.get("/scrape-upcoming", response_class=HTMLResponse)
async def scrape_upcoming_page(request: Request):
    return templates.TemplateResponse("scrape_upcoming.html", {
        "request": request,
        "current_page": "scrape_upcoming",
        "breadcrumbs": [],
    })


@app.get("/bloodline-vector", response_class=HTMLResponse)
async def bloodline_vector_page(request: Request):
    return templates.TemplateResponse("bloodline_vector.html", {
        "request": request,
        "current_page": "bloodline_vector",
        "breadcrumbs": [],
    })


# ══════════════════════════════════════════════════════
#  血統構造マップ (祖先共有ベース)
# ══════════════════════════════════════════════════════

@app.get("/pedigree-map", response_class=HTMLResponse)
async def pedigree_map_page(request: Request):
    return templates.TemplateResponse("pedigree_map.html", {
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
    return templates.TemplateResponse("note_aptitude_race.html", {
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
        return JSONResponse({"error": "データ未生成。python -m research.pedigree_similarity を実行してください。"})
    with open(data_path, encoding="utf-8") as f:
        return JSONResponse(json.load(f))


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
        from research.sire_aptitude_note import (
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
        from research.sire_aptitude_note import (
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
    distance_m: int = 0,
    mode: str = "shallow",
):
    """
    指定 race_id の出走全頭を、sire_aptitude_note の血統ブレンドから
    3次元（パワー・欧州瞬発・TS素地）に写像し、フィールド内でクラスタ（最大4）分けする。

    mode: shallow（父・母父・牝系） | 5gen（5世代血統＋父／母経路重み、データ無し時は shallow）
    """
    rid = (race_id or "").strip()
    if not rid:
        return JSONResponse({"error": "race_id が必要です"}, status_code=400)
    try:
        from research.note_aptitude_race_map import build_race_note_aptitude_map

        bm = (mode or "shallow").strip().lower()
        if bm in ("5g", "fivegen", "full"):
            bm = "5gen"
        if bm not in ("shallow", "5gen"):
            return JSONResponse(
                {"error": "mode は shallow または 5gen です"},
                status_code=400,
            )

        storage = _get_storage()
        out = await asyncio.to_thread(
            build_race_note_aptitude_map,
            storage,
            rid,
            distance_m=int(distance_m or 0),
            blend_mode=bm,
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
        from research.race_pedigree_5gen_prefetch import start_race_pedigree_prefetch

        storage = _get_storage()
        out = await asyncio.to_thread(start_race_pedigree_prefetch, storage, rid)
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
        from research.race_pedigree_5gen_prefetch import session_progress

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
        from research.race_pedigree_5gen_prefetch import cancel_race_pedigree_prefetch_session

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
        from research.race_pedigree_5gen_prefetch import batch_race_pedigree_5gen_date_range

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
    return templates.TemplateResponse("bloodline.html", {
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
        from research.bloodline_distance import BloodlineDistanceAnalyzer

        _base = os.path.join(os.path.dirname(__file__), "..")
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


@app.get("/api/bloodline/data/{analysis_type}", response_class=JSONResponse)
async def api_bloodline_data(analysis_type: str):
    """分析結果 CSV/JSON を読み込んで返す。"""
    import csv as csv_mod
    from pathlib import Path
    base = Path(os.path.dirname(__file__)).parent / "data" / "research" / "bloodline"

    file_map = {
        "sire_distance": base / "sire_distance_top3rate.csv",
        "best_distance": base / "sire_best_distance.csv",
        "sire_damsire_rate": base / "sire_damsire_top3rate.csv",
        "sire_damsire_dist": base / "sire_damsire_avg_distance.csv",
        "similarity": base / "distance_bloodline_similarity.csv",
        "clusters": base / "sire_clusters.csv",
        "cluster_summary": base / "cluster_summary.csv",
        "predictive_power": base / "pedigree_predictive_power.json",
    }

    path = file_map.get(analysis_type)
    if not path or not path.exists():
        return JSONResponse({"error": f"データなし: {analysis_type}"}, status_code=404)

    if path.suffix == ".json":
        import json as json_mod
        return JSONResponse(json_mod.loads(path.read_text(encoding="utf-8")))

    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
    return JSONResponse({"columns": list(rows[0].keys()) if rows else [], "rows": rows})


# ══════════════════════════════════════════════════════
#  コース特性 × 血統適性 研究ページ
# ══════════════════════════════════════════════════════

_course_bl_job: dict[str, Any] = {"running": False}


@app.get("/course-bloodline", response_class=HTMLResponse)
async def course_bloodline_page(request: Request):
    return templates.TemplateResponse("course_bloodline.html", {
        "request": request,
        "current_page": "course_bloodline",
        "breadcrumbs": [],
    })


@app.get("/api/course-profiles", response_class=JSONResponse)
async def api_course_profiles():
    """コースプロファイル (ドメインナレッジ) を返す。"""
    import json as json_mod
    from pathlib import Path as _P
    p = _P(os.path.dirname(__file__)).parent / "data" / "knowledge" / "course_profiles.json"
    if not p.exists():
        return JSONResponse({"error": "course_profiles.json が見つかりません"}, status_code=404)
    return JSONResponse(json_mod.loads(p.read_text(encoding="utf-8")))


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
        from research.course_bloodline import CourseBloodlineAnalyzer
        _base = os.path.join(os.path.dirname(__file__), "..")
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


@app.get("/api/course-bloodline/data/{analysis_type}", response_class=JSONResponse)
async def api_course_bl_data(analysis_type: str):
    import csv as csv_mod
    from pathlib import Path as _P
    base = _P(os.path.dirname(__file__)).parent / "data" / "research" / "course_bloodline"

    file_map = {
        "venue_sire": base / "venue_sire_top3rate.csv",
        "trait_correlation": base / "sire_trait_correlation.csv",
        "aptitude": base / "sire_course_aptitude.csv",
        "optimal": base / "sire_optimal_conditions.csv",
        "track_condition": base / "track_condition_interaction.csv",
        "profiles": base / "course_profiles_summary.csv",
        "draw_bloodline": base / "draw_bloodline_interaction.csv",
        "draw_summary": base / "sire_draw_bias_summary.csv",
        "grass_type": base / "grass_type_bloodline.csv",
        "grass_summary": base / "sire_grass_type_summary.csv",
        "fc_draw": base / "first_corner_draw_interaction.csv",
        "fc_draw_summary": base / "first_corner_draw_summary.csv",
    }

    path = file_map.get(analysis_type)
    if not path or not path.exists():
        return JSONResponse({"error": f"データなし: {analysis_type}"}, status_code=404)

    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
    return JSONResponse({"columns": list(rows[0].keys()) if rows else [], "rows": rows})


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
    data_path = _P(os.path.dirname(__file__)).parent / "data" / "jra_baba" / "cushion_values.json"
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
    data_path = _P(os.path.dirname(__file__)).parent / "data" / "jra_baba" / "cushion_values.json"
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

    _cushion_job.update(running=True, error=None, result=None, started_at=time.time())
    background_tasks.add_task(_run_cushion_scrape, years)
    return JSONResponse({"status": "started"})


def _run_cushion_scrape(years: list[int] | None):
    try:
        from scraper.jra_cushion import JRACushionScraper
        _base = os.path.join(os.path.dirname(__file__), "..", "data", "jra_baba")
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

    from scraper.jra_cushion_storage import upload_cushion_values_to_gcs

    base = Path(__file__).resolve().parent.parent
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

    from scraper.jra_cushion_sync import sync_preprocessed_to_jra_cushion

    base = Path(__file__).resolve().parent.parent

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
        from scraper.jra_baba_live import JRABabaLiveScraper
        _base = os.path.join(os.path.dirname(__file__), "..", "data", "jra_baba")
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
    from scraper.jra_baba_live import (
        _load_poll_schedule, _get_today_entry,
        _get_poll_windows, _in_any_window, _next_poll_time,
    )
    import datetime as _dt

    _base = os.path.join(os.path.dirname(__file__), "..", "data", "jra_baba")
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
    from scraper.jra_baba_live import JRABabaLiveScraper
    _base = os.path.join(os.path.dirname(__file__), "..", "data", "jra_baba")
    scraper = JRABabaLiveScraper(output_dir=_base)
    has_new = scraper.has_new_data()
    return JSONResponse({"has_new_data": has_new})


@app.get("/api/cushion/schedule")
async def api_cushion_schedule(days: int = 14):
    """直近のポーリングスケジュールを返す。"""
    from scraper.jra_baba_live import _load_poll_schedule, _get_poll_windows
    import datetime as _dt

    _base = os.path.join(os.path.dirname(__file__), "..", "data", "jra_baba")
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
    from scraper.auto_scrape import _load_status, _load_race_calendar
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


# ══════════════════════════════════════════════════════
#  馬場速度レベリング (Track Speed Index)
# ══════════════════════════════════════════════════════

_track_speed_job: dict[str, Any] = {
    "status": "idle",
    "progress": [],
    "error": None,
    "data_available": False,
}
_track_speed_data: dict[str, Any] = {}
_track_speed_lock = threading.Lock()


@app.get("/track-speed", response_class=HTMLResponse)
async def track_speed_page(request: Request):
    return templates.TemplateResponse("track_speed.html", {
        "request": request,
        "current_page": "track_speed",
        "breadcrumbs": [],
    })


@app.get("/api/track-speed/status", response_class=JSONResponse)
async def track_speed_status():
    return JSONResponse({
        "status": _track_speed_job["status"],
        "progress": _track_speed_job["progress"][-30:],
        "error": _track_speed_job["error"],
        "data_available": _track_speed_job["data_available"],
    })


@app.get("/api/track-speed", response_class=JSONResponse)
async def track_speed_data():
    if not _track_speed_data:
        return JSONResponse({"error": "データなし。計算を実行してください。"})
    return JSONResponse(_track_speed_data)


@app.post("/api/track-speed/compute", response_class=JSONResponse)
async def track_speed_compute():
    if _track_speed_job["status"] == "running":
        return JSONResponse({"status": "already_running"})
    threading.Thread(target=_compute_track_speed, daemon=True).start()
    return JSONResponse({"status": "started"})


def _compute_track_speed():
    """
    TSI (Track Speed Index) 算出:
      1. 全レースの2着走破タイムを収集
      2. (馬場×距離×会場) ごとにベースライン統計を算出
      3. raw_z = (time_2nd - mean) / std
      4. 出走馬のレーティングから馬質補正 (field_adj) を算出
      5. TSI = raw_z - field_adj
      6. 日×会場×馬場 で集約 → 速度レベル判定
    """
    import statistics
    global _track_speed_data

    job = _track_speed_job
    job["status"] = "running"
    job["progress"] = []
    job["error"] = None

    def log(msg: str):
        job["progress"].append(msg)
        logger.info("[TrackSpeed] %s", msg)

    try:
        storage = _get_storage()
        race_list_dates = sorted(storage.list_keys("race_lists"), reverse=True)
        log(f"開催日数: {len(race_list_dates)}")

        all_races: list[dict] = []

        for i, date in enumerate(race_list_dates):
            rl = storage.load("race_lists", date)
            if not rl:
                continue
            races = rl.get("races", [])
            jra = [r for r in races if r.get("race_id") and _is_jra_race(r["race_id"])]

            for race_meta in jra:
                rid = race_meta["race_id"]
                result = storage.load("race_result", rid)
                if not result:
                    continue

                entries = result.get("entries", [])
                if len(entries) < 3:
                    continue

                surface = result.get("surface", "")
                distance = result.get("distance", 0)
                venue = result.get("venue", "")
                track_cond = result.get("track_condition", "")
                grade = result.get("grade", "")

                if not surface or not distance or not venue:
                    continue
                if surface == "障":
                    continue

                second_entry = None
                for e in entries:
                    fp = e.get("finish_position") or e.get("finish_order")
                    if fp and (fp == 2 or str(fp) == "2"):
                        second_entry = e
                        break
                if not second_entry:
                    sorted_by_order = [
                        e for e in entries
                        if e.get("time_sec") and e["time_sec"] > 0
                    ]
                    sorted_by_order.sort(key=lambda x: x["time_sec"])
                    if len(sorted_by_order) >= 2:
                        second_entry = sorted_by_order[1]

                if not second_entry or not second_entry.get("time_sec"):
                    continue
                time_2nd = second_entry["time_sec"]
                if time_2nd <= 0:
                    continue

                horse_times = [
                    e["time_sec"] for e in entries
                    if e.get("time_sec") and e["time_sec"] > 0
                ]
                field_mean = statistics.mean(horse_times) if horse_times else time_2nd

                all_races.append({
                    "date": date,
                    "race_id": rid,
                    "venue": venue,
                    "surface": surface,
                    "distance": distance,
                    "track_condition": track_cond,
                    "grade": grade,
                    "round": result.get("round", 0),
                    "time_2nd": round(time_2nd, 1),
                    "field_mean": round(field_mean, 2),
                    "n_runners": len(entries),
                })

            if (i + 1) % 20 == 0:
                log(f"レース読込: {i+1}/{len(race_list_dates)} 日 ({len(all_races)} R)")

        log(f"全レース収集完了: {len(all_races)} R")

        if not all_races:
            job["status"] = "error"
            job["error"] = "有効なレースデータがありません"
            return

        baselines: dict[str, list[float]] = {}
        for r in all_races:
            key = f"{r['surface']}_{r['distance']}_{r['venue']}"
            baselines.setdefault(key, []).append(r["time_2nd"])

        baseline_stats: dict[str, dict] = {}
        for key, times in baselines.items():
            if len(times) < 3:
                continue
            mu = statistics.mean(times)
            sd = statistics.stdev(times) if len(times) > 1 else 1.0
            if sd < 0.1:
                sd = 0.1
            baseline_stats[key] = {"mean": round(mu, 3), "std": round(sd, 3), "n": len(times)}

        log(f"ベースライン: {len(baseline_stats)} グループ")

        for r in all_races:
            key = f"{r['surface']}_{r['distance']}_{r['venue']}"
            bl = baseline_stats.get(key)
            if not bl:
                key_wide = f"{r['surface']}_{r['distance']}"
                fallback = [
                    v for k, v in baseline_stats.items()
                    if k.startswith(key_wide)
                ]
                if fallback:
                    mu = statistics.mean([f["mean"] for f in fallback])
                    sd = statistics.mean([f["std"] for f in fallback])
                    bl = {"mean": mu, "std": sd, "n": sum(f["n"] for f in fallback)}
            if not bl:
                r["raw_z"] = 0.0
                r["field_adj"] = 0.0
                r["tsi"] = 0.0
                r["baseline_mean"] = None
                continue

            raw_z = (r["time_2nd"] - bl["mean"]) / bl["std"]

            bl_mean_field = bl["mean"]
            field_adj = (r["field_mean"] - bl_mean_field) / bl["std"] * 0.3

            tsi = raw_z - field_adj
            tsi = max(-3.0, min(3.0, tsi))

            r["raw_z"] = round(raw_z, 3)
            r["field_adj"] = round(field_adj, 3)
            r["tsi"] = round(tsi, 3)
            r["baseline_mean"] = round(bl["mean"], 1)

        log("TSI 算出完了")

        daily: dict[str, dict] = {}
        for r in all_races:
            date = r["date"]
            venue = r["venue"]
            surface = r["surface"]
            daily.setdefault(date, {}).setdefault(venue, {}).setdefault(surface, {
                "races": [], "tsi_values": [], "track_conditions": set(),
            })
            bucket = daily[date][venue][surface]
            bucket["races"].append({
                "round": r["round"],
                "distance": r["distance"],
                "grade": r["grade"],
                "time_2nd": r["time_2nd"],
                "baseline_mean": r["baseline_mean"],
                "raw_z": r["raw_z"],
                "field_adj": r["field_adj"],
                "tsi": r["tsi"],
            })
            if r.get("tsi") is not None:
                bucket["tsi_values"].append(r["tsi"])
            if r["track_condition"]:
                bucket["track_conditions"].add(r["track_condition"])

        def _speed_label(tsi: float) -> tuple[str, int]:
            if tsi <= -1.5:
                return "超高速", 5
            if tsi <= -0.5:
                return "高速", 4
            if tsi < 0.5:
                return "標準", 3
            if tsi < 1.5:
                return "低速", 2
            return "超低速", 1

        output_daily: dict[str, dict] = {}
        total_races_with_z = 0
        days_computed = 0

        for date in sorted(daily.keys()):
            output_daily[date] = {}
            day_has = False
            for venue, surfaces in daily[date].items():
                output_daily[date][venue] = {}
                for surface, bucket in surfaces.items():
                    vals = bucket["tsi_values"]
                    if not vals:
                        continue
                    mean_tsi = statistics.mean(vals)
                    label, level = _speed_label(mean_tsi)
                    races_sorted = sorted(bucket["races"], key=lambda x: x["round"])

                    output_daily[date][venue][surface] = {
                        "tsi": round(mean_tsi, 3),
                        "label": label,
                        "level": level,
                        "sample_size": len(vals),
                        "track_conditions": sorted(bucket["track_conditions"]),
                        "races": races_sorted,
                    }
                    total_races_with_z += len(vals)
                    day_has = True
            if day_has:
                days_computed += 1

        _track_speed_data.clear()
        _track_speed_data.update({
            "daily": output_daily,
            "meta": {
                "stats": {
                    "days_computed": days_computed,
                    "races_with_z": total_races_with_z,
                    "horses_with_hsr": len(all_races),
                    "baseline_groups": len(baseline_stats),
                },
            },
        })

        job["data_available"] = True
        job["status"] = "done"
        log(f"完了: {days_computed} 日, {total_races_with_z} R")

    except Exception as e:
        logger.error("[TrackSpeed] %s", e, exc_info=True)
        job["status"] = "error"
        job["error"] = str(e)


# ─── ミオスタチン遺伝子ページ ──────────────────────────


@app.get("/myostatin", response_class=HTMLResponse)
async def myostatin_page(request: Request):
    return templates.TemplateResponse(
        "myostatin.html",
        {"request": request, "current_page": "myostatin"},
    )


@app.get("/api/myostatin", response_class=JSONResponse)
async def myostatin_data():
    import json as _json
    from pathlib import Path
    kb_path = Path(__file__).resolve().parent.parent / "data" / "knowledge" / "myostatin_genes.json"
    if not kb_path.exists():
        return JSONResponse({"error": "KB not found"}, status_code=404)
    data = _json.loads(kb_path.read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.post("/api/myostatin/predict", response_class=JSONResponse)
async def myostatin_predict(request: Request):
    from research.myostatin import MyostatinLookup
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
