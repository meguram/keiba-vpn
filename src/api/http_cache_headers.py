"""API の GET 応答にブラウザ向けの短いキャッシュ指示（Cache-Control: private, max-age=N と Vary: Cookie）を付ける。

許可リストは config/http_cache.json。既定は private なので、会員と匿名の応答が混ざらない。
Cookie が変わる（ログイン・ログアウト）と別の応答として取り直される。
例外: ルールに ``s_maxage`` があり、リクエストに Cookie / Authorization が無い（会員区分で応答が変わらない匿名）ときだけ
``public, max-age=N, s-maxage=S`` にして CDN に置かせる（閲覧者が増えてもオリジンに届くのは S 秒に 1 回）。
さらに ``settled`` の対象（開催から grace_days 日以上たった確定済みのレース）は s-maxage を 30 日（1 か月）にする。
日付が分からないものは短いまま（置きすぎない）。データ更新のあとは VPS の keiba-cache-warm が上書きし、モデル切替のあとは purge_api_cache.sh で入れ替える。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

_CFG = Path(os.environ.get("KEIBA_HTTP_CACHE_CONFIG") or Path(__file__).resolve().parents[2] / "config" / "http_cache.json")
_cache: dict[str, Any] = {"at": 0.0, "cfg": None}
_JST = timezone(timedelta(hours=9))
_race_date_peek: Callable[[str], str] | None = None


def config() -> dict[str, Any]:
    now = time.monotonic()
    if _cache["cfg"] is not None and now - _cache["at"] < 30:
        return _cache["cfg"]
    try:
        cfg = json.loads(_CFG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {"enabled": False}
    _cache.update(at=now, cfg=cfg)
    return cfg


def max_age_for(path: str, method: str, status: int, has_cache_control: bool, cfg: dict[str, Any] | None = None) -> int | None:
    """付けるべき max-age(秒)。付けない場合は None。"""
    cfg = cfg if cfg is not None else config()
    if not cfg.get("enabled") or method.upper() != "GET" or status != 200 or has_cache_control:
        return None
    if any(path == n or path.startswith(n.rstrip("/") + "/") or path.startswith(n) for n in cfg.get("never") or []):
        return None
    for r in cfg.get("rules") or []:
        if path.startswith(r["prefix"]):
            ma = int(r.get("max_age") or 0)
            return ma if ma > 0 else None
    return None


def shared_s_maxage_for(path: str, cfg: dict[str, Any] | None = None) -> int | None:
    """CDN など共有キャッシュに置いてよい秒数（``s_maxage``）。ルールに無ければ None。"""
    cfg = cfg if cfg is not None else config()
    for r in cfg.get("rules") or []:
        if path.startswith(r["prefix"]):
            s = int(r.get("s_maxage") or 0)
            return s if s > 0 else None
    return None


def set_race_date_peek(fn: Callable[[str], str] | None) -> None:
    """race_id → 開催日(YYYYMMDD) を I/O なしで引く関数を登録する（索引が未構築なら空文字を返すこと）。"""
    global _race_date_peek
    _race_date_peek = fn


def _first_token(path: str, prefix: str) -> str:
    return path[len(prefix):].split("/", 1)[0]


def _token_date(token: str) -> "datetime | None":
    """token（8桁の開催日 / 12桁の race_id）の開催日。分からなければ None。"""
    if token.isdigit() and len(token) == 8:
        try:
            return datetime.strptime(token, "%Y%m%d")
        except ValueError:
            return None
    if token.isdigit() and len(token) == 12 and _race_date_peek is not None:
        try:
            d = _race_date_peek(token) or ""
        except Exception:  # noqa: BLE001  索引の覗き見は失敗しても日付不明に落とすだけ
            d = ""
        if len(d) == 8 and d.isdigit():
            return _token_date(d)
    return None


def _is_settled(token: str, today: "datetime | None", grace_days: int) -> bool:
    """token が 8桁の開催日 / 12桁の race_id のとき、開催が grace_days 日より前（確定済み）なら True。分からなければ False。"""
    limit = (today or datetime.now(_JST)) - timedelta(days=grace_days)
    d = _token_date(token)
    if d is not None:
        return d.date() <= limit.date()
    if token.isdigit() and len(token) == 12:
        return int(token[:4]) < limit.year      # 日付不明: 前の年のレースだけ確定扱い
    return False


def settled_s_maxage_for(path: str, cfg: dict[str, Any] | None = None, today: "datetime | None" = None) -> int | None:
    """共有キャッシュ(nginx)に長く置く秒数。対象外・日付不明なら None（呼び出し側は通常の s_maxage を使う）。

    - ``settled.prefixes`` のレース・開催日: 確定済み（開催から grace_days 日以上）は ``settled.s_maxage``（30 日）。
      開催日が分かる当日・未来・確定前は ``settled.upcoming_s_maxage``（6 時間。データ更新時は VPS の事前ロードが上書きするので、
      これは上書きが漏れたときの上限）。
    - ``long.prefixes``（馬のページなど）: ``long.s_maxage``（30 日。馬のデータの保存時に事前ロードが上書きする）。
    """
    cfg = cfg if cfg is not None else config()
    if not cfg.get("enabled"):
        return None
    if any(path == n or path.startswith(n.rstrip("/") + "/") or path.startswith(n) for n in cfg.get("never") or []):
        return None
    st = cfg.get("settled") or {}
    s = int(st.get("s_maxage") or 0)
    for pre in st.get("prefixes") or []:
        if path.startswith(pre):
            token = _first_token(path, pre)
            if s > 0 and _is_settled(token, today, int(st.get("grace_days", 3))):
                return s
            up = int(st.get("upcoming_s_maxage") or 0)
            return up if up > 0 and _token_date(token) is not None else None
    lg = cfg.get("long") or {}
    ls = int(lg.get("s_maxage") or 0)
    if ls > 0 and any(path.startswith(pre) for pre in lg.get("prefixes") or []):
        return ls
    return None


def is_anonymous_request(cookie_header: str, authorization: str, remote_addr: str | None, headers) -> bool:
    """会員区分で応答が変わらない（誰が見ても同じ）匿名リクエストか。Cookie・Authorization が一切無く、dev の自動ログインでもないときだけ True。"""
    if (cookie_header or "").strip() or (authorization or "").strip():
        return False
    from src.api.auth import dev_auto_login_applies

    return not dev_auto_login_applies(remote_addr, headers)


def cache_control_value(ma: int, s_maxage: int | None, anonymous: bool, settled: int | None = None) -> str:
    """匿名で ``s_maxage`` のあるパスだけ共有キャッシュ可（public）。それ以外は従来どおり private（会員の応答が他人に渡らない）。
    確定済みのレース（settled）は共有側だけ長く置く。ブラウザの max-age は短いまま。"""
    if anonymous and s_maxage:
        return f"public, max-age={ma}, s-maxage={max(s_maxage, settled or 0)}"
    return f"private, max-age={ma}"


def _vary_add(existing: str) -> str:
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if not any(p.lower() == "cookie" for p in parts):
        parts.append("Cookie")
    return ", ".join(parts)


def install_fastapi(app) -> None:
    from starlette.middleware.base import BaseHTTPMiddleware

    class _HttpCacheMW(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            from src.utils import cache_epoch

            cache_epoch.sync()
            resp = await call_next(request)
            ma = max_age_for(request.url.path, request.method, resp.status_code, "cache-control" in resp.headers)
            if ma:
                anon = is_anonymous_request(
                    request.headers.get("cookie", ""), request.headers.get("authorization", ""),
                    getattr(request.client, "host", None), request.headers,
                )
                path = request.url.path
                resp.headers["Cache-Control"] = cache_control_value(ma, shared_s_maxage_for(path), anon, settled_s_maxage_for(path))
                resp.headers["Vary"] = _vary_add(resp.headers.get("Vary", ""))
            return resp

    app.add_middleware(_HttpCacheMW)


def install_flask(app) -> None:
    from flask import request

    @app.before_request
    def _cache_epoch_sync():
        from src.utils import cache_epoch

        cache_epoch.sync()

    @app.after_request
    def _http_cache(response):
        ma = max_age_for(request.path, request.method, response.status_code, "Cache-Control" in response.headers)
        if ma:
            anon = is_anonymous_request(
                request.headers.get("Cookie", ""), request.headers.get("Authorization", ""), request.remote_addr, request.headers,
            )
            response.headers["Cache-Control"] = cache_control_value(
                ma, shared_s_maxage_for(request.path), anon, settled_s_maxage_for(request.path))
            response.headers["Vary"] = _vary_add(response.headers.get("Vary", ""))
        return response
