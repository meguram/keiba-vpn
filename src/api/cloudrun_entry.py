"""Cloud Run 用のエントリ（``uvicorn src.api.cloudrun_entry:app``）。1 コンテナで FastAPI と Flask(/api/v1) と /ops を受ける。

ホストでは FastAPI(:fastapi) と Flask(:flask) が別プロセスで、Next.js の rewrites が振り分けている。Cloud Run は 1 ポートなので、
ここで /api/v1/* を Flask(WSGI) へ、/ops/* を運用 API へ、それ以外を FastAPI へ渡す。

意図して FastAPI の lifespan（取得キューのワーカー・各種スケジューラ・先読みスレッド）は動かさない:
Cloud Run は読み取り専用の API であり、取得・定期処理は VPS の実行役と Cloud Run Jobs が担当する（architecture_target.md）。
lifespan を動かさなくても FastAPI のルーティングは動く。app.py / flask_app.py は変更しない。

環境変数:
  KEIBA_ENV                必須（prod）。未設定なら起動を拒否（暗黙の prod を防ぐ）。
  KEIBA_RUN_LOCATION=gcp   Slack 通知の実行場所表示。
  KEIBA_PROXY_KEY_REQUIRED=1 / KEIBA_PROXY_KEY / KEIBA_PROXY_KEY_NEXT   共有ヘッダ認証（src/api/middleware/proxy_key.py）。
  KEIBA_MIRROR_PULL=1      予測ファイル・出馬表の状態を GCS（ops/mirror/）から取り込む常駐スレッドを起動（既定 1）。
  KEIBA_SERVING_STATE_PULL=1  起動時に GCS の serving_state（config/serving_state.json）を取り込む。
  KEIBA_HEARTBEAT_SECRET / _NEXT   VPS ハートビートの署名鍵。
  KEIBA_MIRROR_MODE        thread（既定。CPU 常時割り当て向け: 取り込みの常駐スレッド）/ request（リクエスト課金向け: CPU はリクエスト中だけなので、
                           起動処理が全体の取り込みまで済ませ、以降はリクエストをきっかけに前回から 1 分以上経っていれば直近分を取り込む）。
  KEIBA_WARMUP_ON_START=1  受付（ポートを開く）の前に、直近のレースの予測・詳細を 1 回呼んで、モデル・履歴の読み込みを済ませる。
                           新しいインスタンスの最初の予測が 30〜45 秒かかり、急増時にタイムアウトしていたため。失敗・時間切れでも起動は続ける。
  KEIBA_PRERENDER=1        応答の置き場だけ有効にする（要求されたときに作った応答を KEIBA_PRERENDER_MAX_AGE_SEC 秒使い回す。src/api/prerender.py）。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable

from src.config.deployment import require_explicit_env

logger = logging.getLogger("api.cloudrun")

# メモリ 2GiB に収めるための既定（コンテナの書き込み層もメモリ扱い）。環境変数で上書き可。
_DEFAULTS = {
    "KEIBA_RUN_LOCATION": "gcp",
    "TZ": "Asia/Tokyo",
    "PYTHONUNBUFFERED": "1",
    "LOAD_CACHE_MAX_ENTRIES": "2000",
    "LOAD_CACHE_TTL_SEC": "900",
    "DISK_L2_MIN_WEEKLY_ACCESSES": "1000000",   # ディスクの L2 キャッシュは書かない（メモリを食うため）
    "KEIBA_BLOODLINE_PRELOAD": "0",
    "LOGS_RETENTION_DISABLED": "1",
    "KEIBA_MIRROR_PULL": "1",                   # 予測ファイル・出馬表の状態を GCS から取り込む（src/ops/gcs_mirror.py）
}

Scope = dict[str, Any]
ASGIApp = Callable[[Scope, Callable, Callable], Awaitable[None]]


def _wsgi_bridge(wsgi_app: Any) -> ASGIApp:
    try:
        from a2wsgi import WSGIMiddleware  # type: ignore

        return WSGIMiddleware(wsgi_app)
    except ImportError:
        from starlette.middleware.wsgi import WSGIMiddleware  # type: ignore

        return WSGIMiddleware(wsgi_app)


WARMUP_TIMEOUT_SEC = 150.0     # Cloud Run の起動プローブ（TCP。既定 240 秒）に収める
MIRROR_WAIT_SEC = 60.0
MIRROR_FULL_WAIT_SEC = 90.0    # request モード: 起動の完了前に『全体』の取り込みまで待つ上限（超えても起動は続ける）
# 取り込みのきっかけにしないパス（画面を開いているだけで 60 秒ごとに来る状態確認・ヘルスチェック・運用の受信）。内容を返す要求だけが鮮度を要る
MIRROR_GATE_SKIP = ("/api/health", "/api/v1/health", "/api/maintenance/status")


def mirror_mode() -> str:
    return "request" if (os.environ.get("KEIBA_MIRROR_MODE") or "thread").strip().lower() == "request" else "thread"


def gate_applies(path: str) -> bool:
    return not (path == "/ops" or path.startswith("/ops/") or path in MIRROR_GATE_SKIP)


async def warm_sequence(get: Callable[[str], Awaitable[tuple[int, Any]]], today: str | None = None) -> list[str]:
    """直近の開催日のレース一覧 → 先頭レースの予測・詳細を呼ぶ。呼んだパスを返す（失敗しても続ける）。"""
    from datetime import datetime, timedelta, timezone

    today = today or datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
    called: list[str] = []

    async def call(path: str) -> Any:
        called.append(path)
        try:
            status, doc = await get(path)
            return doc if status == 200 else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("ウォームアップ %s に失敗: %s", path, exc)
            return None

    await call("/api/v1/health")
    doc = await call("/api/scrape-dates")
    raw = (doc or {}).get("dates", []) if isinstance(doc, dict) else (doc or [])
    dates = sorted({x if isinstance(x, str) else str(x.get("date")) for x in raw if x})
    past = [d for d in dates if d <= today]
    if not past:
        return called
    rl = await call(f"/api/race-list/{past[-1]}")
    races = [r["race_id"] for r in ((rl or {}).get("races") or []) if isinstance(r, dict) and r.get("race_id")]
    if races:
        await call(f"/api/race/{races[0]}/predictions")
        await call(f"/api/race/{races[0]}")
    return called


class Gateway:
    """パスで 3 つの ASGI アプリに振り分ける。lifespan は、起動時ウォームアップ（あれば）を済ませてから完了を返す。"""

    def __init__(self, fastapi_app: ASGIApp, flask_asgi: ASGIApp, ops_app: ASGIApp) -> None:
        self.fastapi_app, self.flask_asgi, self.ops_app = fastapi_app, flask_asgi, ops_app
        self.warmup: Callable[[], Awaitable[None]] | None = None
        self.refresher: Any = None          # request モード: gcs_mirror.RequestRefresher

    async def _gate(self) -> None:
        """前回の取り込みから間隔が空いていれば、このリクエストをきっかけに取り込みを始める。長く止まっていた後だけ、完了を少し待つ。"""
        r = self.refresher
        wait = r.should_wait()
        t = r.start_if_due()
        if t is not None and wait:
            import asyncio

            await asyncio.to_thread(t.join, r.wait_s)

    @staticmethod
    def route_for(path: str) -> str:
        if path == "/ops" or path.startswith("/ops/"):
            return "ops"
        if path == "/api/v1" or path.startswith("/api/v1/"):
            return "flask"
        return "fastapi"

    async def __call__(self, scope: Scope, receive: Callable, send: Callable) -> None:
        t = scope.get("type")
        if t == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    if self.warmup is not None:       # uvicorn は lifespan の完了後にポートを開く = 温まるまで要求が来ない
                        import asyncio

                        try:
                            await asyncio.wait_for(self.warmup(), timeout=WARMUP_TIMEOUT_SEC)
                        except Exception as exc:  # noqa: BLE001  時間切れ・失敗でも起動は続ける
                            logger.warning("起動時ウォームアップを打ち切り: %r", exc)
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        path = scope.get("path") or ""
        if self.refresher is not None and t == "http" and gate_applies(path):
            try:
                await self._gate()
            except Exception as exc:  # noqa: BLE001  取り込みの都合でリクエストを失敗させない
                logger.warning("取り込みの起動に失敗: %r", exc)
        target = {"ops": self.ops_app, "flask": self.flask_asgi, "fastapi": self.fastapi_app}[self.route_for(path)]
        await target(scope, receive, send)


def _pull_serving_state() -> None:
    if (os.environ.get("KEIBA_SERVING_STATE_PULL") or "").strip() != "1":
        return
    from src.scripts.ops.serving_state import pull_default_groups

    for r in pull_default_groups():
        logger.info("serving_state: %s", r)


def _start_mirror_pull() -> Any:
    """予測ファイル・出馬表の状態を GCS（ops/mirror/）から取り込む。起動時に全部、以降は 1 分ごとに直近分（src/ops/gcs_mirror.py）。
    thread モードは常駐スレッド（None を返す）。request モードは起動時に 1 回だけ取り込み、以降はリクエストをきっかけに取り込む ``RequestRefresher`` を返す。"""
    if (os.environ.get("KEIBA_MIRROR_PULL") or "").strip() != "1" or not os.environ.get("GCS_BUCKET"):
        return None         # バケットが無い環境（テスト・dev）では取り込まない
    from src.ops import gcs_mirror

    if mirror_mode() == "request":
        gcs_mirror.start_initial()
        return gcs_mirror.RequestRefresher()
    gcs_mirror.start_refresher()
    return None


def build_app() -> ASGIApp:
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, v)
    require_explicit_env()
    _pull_serving_state()
    refresher = _start_mirror_pull()

    from fastapi import FastAPI

    from src.api.app import app as fastapi_app
    from src.api.flask_app import create_app
    from src.api.middleware.proxy_key import DEFAULT_EXEMPT, ProxyKeyMiddleware
    from src.ops.heartbeat_api import build_router

    ops = FastAPI(title="keiba ops", docs_url=None, redoc_url=None, openapi_url=None)
    ops.include_router(build_router())
    gateway = Gateway(fastapi_app, _wsgi_bridge(create_app()), ops)
    gateway.refresher = refresher
    # /ops/heartbeat（POST）は共有ヘッダではなく HMAC 署名で認証する。/ops/heartbeat/status・/check は共有ヘッダが要る
    outer = ProxyKeyMiddleware(gateway, exempt_paths=(*DEFAULT_EXEMPT, "/ops/heartbeat"))
    warm = (os.environ.get("KEIBA_WARMUP_ON_START") or "").strip() == "1"
    if warm or refresher is not None:
        gateway.warmup = lambda: _startup(outer, warm=warm, wait_full=refresher is not None)
    return outer


async def _startup(outer: ASGIApp, *, warm: bool, wait_full: bool) -> None:
    """起動処理（CPU が必ず割り当てられる間に済ませる）。ウォームアップ → request モードなら全体の取り込みの完了も待つ。"""
    import asyncio

    if warm:
        await _warmup(outer)
    if wait_full:
        from src.ops.gcs_mirror import initial_full_pulled

        ok = await asyncio.to_thread(initial_full_pulled.wait, MIRROR_FULL_WAIT_SEC)
        logger.info("起動時の全体の取り込み: %s", "完了" if ok else "時間切れ（以降は読まれたときに取り込む）")


async def _warmup(outer: ASGIApp) -> None:
    """取り込み（直近分）の完了を待ち、自分自身に ASGI で要求を送って予測の初回コスト（モデル・履歴の読み込み）を先に払う。"""
    import asyncio

    import httpx

    t0 = asyncio.get_running_loop().time()
    if (os.environ.get("KEIBA_MIRROR_PULL") or "").strip() == "1" and os.environ.get("GCS_BUCKET"):
        from src.ops.gcs_mirror import initial_recent_pulled

        await asyncio.to_thread(initial_recent_pulled.wait, MIRROR_WAIT_SEC)
    headers = {"X-Keiba-Proxy-Key": os.environ.get("KEIBA_PROXY_KEY", "")}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer), base_url="http://warmup", headers=headers, timeout=120) as c:
        async def get(path: str) -> tuple[int, Any]:
            r = await c.get(path)
            return r.status_code, (r.json() if r.status_code == 200 else None)

        called = await warm_sequence(get)
    logger.info("起動時ウォームアップ完了: %d 件・%.1f 秒", len(called), asyncio.get_running_loop().time() - t0)


app = build_app() if os.environ.get("KEIBA_CLOUDRUN_LAZY") != "1" else None  # テストは KEIBA_CLOUDRUN_LAZY=1 で import だけする
