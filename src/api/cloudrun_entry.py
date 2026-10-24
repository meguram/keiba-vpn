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


class Gateway:
    """パスで 3 つの ASGI アプリに振り分ける。lifespan は何もせず完了を返す。"""

    def __init__(self, fastapi_app: ASGIApp, flask_asgi: ASGIApp, ops_app: ASGIApp) -> None:
        self.fastapi_app, self.flask_asgi, self.ops_app = fastapi_app, flask_asgi, ops_app

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
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        target = {"ops": self.ops_app, "flask": self.flask_asgi, "fastapi": self.fastapi_app}[self.route_for(scope.get("path") or "")]
        await target(scope, receive, send)


def _pull_serving_state() -> None:
    if (os.environ.get("KEIBA_SERVING_STATE_PULL") or "").strip() != "1":
        return
    from src.scripts.ops.serving_state import pull_default_groups

    for r in pull_default_groups():
        logger.info("serving_state: %s", r)


def _start_mirror_pull() -> None:
    """予測ファイル・出馬表の状態を GCS（ops/mirror/）から取り込む。起動時に全部、以降は 1 分ごとに直近分（src/ops/gcs_mirror.py）。"""
    if (os.environ.get("KEIBA_MIRROR_PULL") or "").strip() != "1":
        return
    from src.ops.gcs_mirror import start_refresher

    start_refresher()


def build_app() -> ASGIApp:
    for k, v in _DEFAULTS.items():
        os.environ.setdefault(k, v)
    require_explicit_env()
    _pull_serving_state()
    _start_mirror_pull()

    from fastapi import FastAPI

    from src.api.app import app as fastapi_app
    from src.api.flask_app import create_app
    from src.api.middleware.proxy_key import DEFAULT_EXEMPT, ProxyKeyMiddleware
    from src.ops.heartbeat_api import build_router

    ops = FastAPI(title="keiba ops", docs_url=None, redoc_url=None, openapi_url=None)
    ops.include_router(build_router())
    gateway = Gateway(fastapi_app, _wsgi_bridge(create_app()), ops)
    # /ops/heartbeat（POST）は共有ヘッダではなく HMAC 署名で認証する。/ops/heartbeat/status・/check は共有ヘッダが要る
    return ProxyKeyMiddleware(gateway, exempt_paths=(*DEFAULT_EXEMPT, "/ops/heartbeat"))


app = build_app() if os.environ.get("KEIBA_CLOUDRUN_LAZY") != "1" else None  # テストは KEIBA_CLOUDRUN_LAZY=1 で import だけする
