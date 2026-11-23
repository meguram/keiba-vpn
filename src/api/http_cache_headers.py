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
_settled_peek: Callable[[str], str] | None = None


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


def set_settled_peek(fn: Callable[[str], str] | None) -> None:
    """token（12桁の race_id / 8桁の開催日）の結果が出たと、このプロセスが知っているかを I/O なしで返す関数を登録する。
    返り値は "confirmed"（確定の結果）・"flash"（速報だけ）・""（知らない）。結果が出ていれば、開催からの日数（grace_days）を
    待たずに長く置く（_confirmed_s_maxage）。知らなければ日数の規則に任せる。"""
    global _settled_peek
    _settled_peek = fn


# 確定の印で長くしてよいパス: VPS の cache-warm がデータの保存の目印で取り直すものだけ（scripts/ops/vps/cache_warm.py の RACE_PATHS・date_paths）。
# それ以外（card・running-map・final-odds など）は、確定の後に結果の取り直し（ラップ・通過の追記など）で変わっても取り直されないので、日数の規則のまま。
_CONFIRMED_WARMED = {"/api/race/": ("", "/predictions"), "/api/race-list/": ("",)}


def _confirmed_s_maxage(path: str, pre: str, token: str, st: dict, today: "datetime | None") -> int | None:
    """確定の印がある token のパスの保持秒数。印が無い・対象外のパスは None。
    確定の結果があり、開催日が前日以前と分かれば settled.s_maxage（30 日）。それ以外（当日・日付不明・速報だけ）は upcoming_s_maxage（6 時間）。
    速報は一部の馬だけ・審議中の仮の着順のこともあり、訂正の目印が間引かれる（同じ race_id は 60 秒に 1 回）こともあるので、何日たっても 6 時間まで。"""
    rests = _CONFIRMED_WARMED.get(pre)
    if _settled_peek is None or rests is None or path[len(pre) + len(token):] not in rests:
        return None
    try:
        level = _settled_peek(token)
    except Exception:  # noqa: BLE001  覗き見は失敗しても日数の規則に落とすだけ
        return None
    if not level:
        return None
    d = _token_date(token)
    now = today or datetime.now(_JST)
    if level == "confirmed" and d is not None and d.date() < now.date():
        s = int(st.get("s_maxage") or 0)
        return s if s > 0 else None
    up = int(st.get("upcoming_s_maxage") or 0)
    return up if up > 0 else None


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
            cs = _confirmed_s_maxage(path, pre, token, st, today)
            if cs:
                return cs
            up = int(st.get("upcoming_s_maxage") or 0)
            # 当日・未来の 6 時間は開催日(8桁)のパスだけ。race_id(12桁)は、cache-warm が取り直さないパス（final-odds・result-status・
            # bundle・card・running-map など）まで前方一致で長く置かれるので、確定前は短い s_maxage のまま（race_id の索引は確定の判定にだけ使う）
            return up if up > 0 and len(token) == 8 and _token_date(token) is not None else None
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


def stale_while_revalidate_cap(cfg: dict[str, Any] | None = None) -> int:
    """共有キャッシュが期限切れの後に古いものを返してよい秒数の上限（``stale_while_revalidate``。無ければ 0 = 付けない）。"""
    cfg = cfg if cfg is not None else config()
    try:
        return max(0, int(cfg.get("stale_while_revalidate") or 0))
    except (TypeError, ValueError):
        return 0


def cache_control_value(ma: int, s_maxage: int | None, anonymous: bool, settled: int | None = None, swr_cap: int = 0) -> str:
    """匿名で ``s_maxage`` のあるパスだけ共有キャッシュ可（public）。それ以外は従来どおり private（会員の応答が他人に渡らない）。
    確定済みのレース（settled）は共有側だけ長く置く。ブラウザの max-age は短いまま。

    ``swr_cap`` > 0 なら ``stale-while-revalidate=min(swr_cap, s-maxage)`` を足す。nginx は期限切れの後この秒数だけ古いものを即座に返し、
    裏で 1 本だけ取り直す（proxy_cache_background_update）。古さの上限は s-maxage＋この秒数（当日のレース 30 秒なら 60 秒）。
    この秒数を過ぎた鍵に同時に来た要求は、nginx が束ねず、その本数だけ上流へ行く（群がりの見積もりは docs/operations/edge_fix_20261123.md）。"""
    if anonymous and s_maxage:
        sm = max(s_maxage, settled or 0)
        swr = min(swr_cap, sm) if swr_cap > 0 else 0
        return f"public, max-age={ma}, s-maxage={sm}" + (f", stale-while-revalidate={swr}" if swr else "")
    return f"private, max-age={ma}"


def shared_short_value(s_maxage: int, cfg: dict[str, Any] | None = None) -> str:
    """要求者で中身が変わらない匿名の応答（メンテナンスの状態・Cookie の無い auth/status）を、nginx にだけ s_maxage 秒置かせる見出し。
    ブラウザは毎回取り直す（max-age=0）。期限切れの後は stale-while-revalidate の秒数だけ古いものを返して裏で取り直す。"""
    swr = min(stale_while_revalidate_cap(cfg), s_maxage)
    return f"public, max-age=0, s-maxage={s_maxage}" + (f", stale-while-revalidate={swr}" if swr else "")


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
                resp.headers["Cache-Control"] = cache_control_value(
                    ma, shared_s_maxage_for(path), anon, settled_s_maxage_for(path), stale_while_revalidate_cap())
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
                ma, shared_s_maxage_for(request.path), anon, settled_s_maxage_for(request.path), stale_while_revalidate_cap())
            response.headers["Vary"] = _vary_add(response.headers.get("Vary", ""))
        return response
