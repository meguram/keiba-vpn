"""メンテナンス時間帯（毎日 JST 03:00–04:00）のゲート。FastAPI のミドルウェアと Flask の before_request の両方に入れる。

判定の中身は ``src.utils.maintenance``（純粋関数）。ここは HTTP への接続だけ:
- メンテナンス中は一般ユーザー向けの経路を 503 + Retry-After + JSON {maintenance: true, ...} で返す（キャッシュさせない）。
- 除外（ヘルスチェック・ログイン・運用 API・静的資産）、バイパス（管理者ログイン・トークン・同一ホストの内部呼び出し）は通す。
- バックグラウンドの取得・予測生成・cron は止めない。機能は ``KEIBA_MAINTENANCE_WINDOW=1`` のときだけ働く（既定オフ）。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

from src.utils import maintenance as M

logger = logging.getLogger("api.maintenance")

_audit_seen: dict[str, float] = {}


def _audit_bypass(reason: str, path: str) -> None:
    """バイパスの利用を監査ログに残す（同じ理由は 1 分に 1 回まで）。"""
    now = time.monotonic()
    if now - _audit_seen.get(reason, -1e9) >= 60:
        _audit_seen[reason] = now
        logger.info("maintenance bypass: reason=%s path=%s", reason, path)


def _dev_bypass_off(env: dict[str, str] | None = None) -> bool:
    e = os.environ if env is None else env
    return (e.get("KEIBA_MAINTENANCE_DEV_BYPASS") or "").strip() in ("0", "false", "no", "off")


def status_for(at: str | None = None) -> dict:
    cfg = M.load_config()
    st = M.compute_status(cfg, M.now_for_request(None, at, cfg))
    M.audit_transition(st, logger)
    return st


# 状態は設定と時刻だけで決まり、要求者（会員・匿名・管理者）で変わらないので、VPS の nginx に共有で置かせる。
# Next.js の middleware が 1 画面ごとに（15 秒のメモリを越えるたび）聞きに来て、Cloud Run を起こし続けていた（2026-11-23 の計測で 1 日 1,100 件）。
# 保守でないとき 120 秒置くと最悪 1 日 720 回（＋取り直し）。保守の始まりの見え方は最大 2 分＋swr 遅れるが、API 自身が保守中は 503 を返すので安全。
# 保守中・予告中（残り秒数・開始までの秒数を返す。毎日 1 時間＋予告）は 15 秒にして、秒読みと始まり・終わりの見え方をずらさない。
STATUS_SHARED_S_MAXAGE = 120
STATUS_ACTIVE_S_MAXAGE = 15


def status_cache_control(at: str | None, st: dict | None = None) -> str:
    """?at=（試験用の時刻の指定）付きは置かない。それ以外は nginx にだけ置く（ブラウザは毎回取り直す）。"""
    if at:
        return "no-store, max-age=0"
    from src.api.http_cache_headers import shared_short_value

    st = st or {}
    counting = bool(st.get("active")) or st.get("phase") in ("active", "notice")
    return shared_short_value(STATUS_ACTIVE_S_MAXAGE if counting else STATUS_SHARED_S_MAXAGE)


def decide(path: str, *, remote_addr: str | None, headers: dict[str, str], is_admin: bool) -> tuple[dict | None, str | None]:
    """(止めるときの status, バイパス理由)。止めないときは status が None。"""
    cfg = M.load_config()
    if not M.is_enabled(cfg):
        return None, None
    st = M.compute_status(cfg, M.now_for_request(None, None, cfg))
    M.audit_transition(st, logger)
    if not st["active"] or M.is_exempt(path, cfg):
        return None, None
    why = M.bypass_reason(remote_addr=remote_addr, headers=headers, is_admin=is_admin, cfg=cfg)
    if why:
        _audit_bypass(why, path)
        return None, why
    return st, None


# ───────────────────────── FastAPI ─────────────────────────

def install_fastapi(app: Any, is_admin: Callable[[Any], bool]) -> None:
    """ミドルウェアと /api/maintenance/status を登録する。is_admin(request) はログイン済み管理者の判定。"""
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    class MaintenanceMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            path = request.url.path
            if not M.is_enabled():
                return await call_next(request)
            try:
                admin = False
                if not M.is_exempt(path):
                    admin = bool(is_admin(request)) if not _dev_bypass_off() else _cookie_admin(request)
                st, _ = decide(path, remote_addr=getattr(request.client, "host", None), headers=dict(request.headers), is_admin=admin)
            except Exception as exc:  # noqa: BLE001  判定の失敗でサービスを止めない
                logger.warning("maintenance 判定に失敗（通常動作に倒す）: %s", exc)
                return await call_next(request)
            if st is None:
                return await call_next(request)
            wants_json = path.startswith("/api/") or "application/json" in (request.headers.get("accept") or "")
            if wants_json:
                return JSONResponse(M.block_payload(st), status_code=503, headers=M.block_headers(st))
            return HTMLResponse(M.block_html(st), status_code=503, headers=M.block_headers(st))

    app.add_middleware(MaintenanceMiddleware)

    @app.get("/api/maintenance/status", response_class=JSONResponse)
    async def maintenance_status(at: str | None = None):
        st = status_for(at)
        return JSONResponse(st, headers={"Cache-Control": status_cache_control(at, st)})


def _cookie_admin(request: Any) -> bool:
    from src.api.auth import COOKIE_NAME, _verify_token

    return _verify_token(request.cookies.get(COOKIE_NAME, ""))


# ───────────────────────── Flask ─────────────────────────

def install_flask(app: Any, is_admin: Callable[[], bool]) -> None:
    """before_request と /api/v1/maintenance/status を登録する。is_admin() はログイン済み管理者の判定。"""
    from flask import jsonify, make_response, request

    @app.before_request
    def _maintenance_gate():
        path = request.path
        if not M.is_enabled() or M.is_exempt(path):
            return None
        try:
            st, _ = decide(path, remote_addr=request.remote_addr, headers=dict(request.headers), is_admin=bool(is_admin()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("maintenance 判定に失敗（通常動作に倒す）: %s", exc)
            return None
        if st is None:
            return None
        resp = make_response(jsonify(M.block_payload(st)), 503)
        for k, v in M.block_headers(st).items():
            resp.headers[k] = v
        return resp

    @app.get("/api/v1/maintenance/status")
    def _maintenance_status():
        at = request.args.get("at")
        st = status_for(at)
        resp = make_response(jsonify(st))
        resp.headers["Cache-Control"] = status_cache_control(at, st)
        return resp
