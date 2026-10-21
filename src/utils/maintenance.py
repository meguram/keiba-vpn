"""毎日のメンテナンス時間帯（既定 JST 03:00–04:00）の判定と、API ミドルウェアが使う共通ロジック。

- 時刻判定は純粋関数（``compute_status``）。時計は引数で注入でき、ホストの OS 時刻（UTC）に依存しない（zoneinfo で JST を明示）。
- 設定は ``config/maintenance.json``。30 秒キャッシュで読み直すため、書き換えるだけで API 再起動なしに反映される（臨時メンテは override）。
- 有効化: ``KEIBA_MAINTENANCE_WINDOW=1``（既定オフ）。dev 環境は ``KEIBA_MAINTENANCE_DEV=1`` も必要（既定は無効）。
- ユーザー向けの表示だけを止める。取得・予測生成・cron・運用 API は止めない（``exempt`` と内部呼び出しの判定）。
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

logger = logging.getLogger("utils.maintenance")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "maintenance.json"
CACHE_TTL_SEC = 30.0

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "timezone": "Asia/Tokyo",
    "daily": {"start": "03:00", "end": "04:00"},
    "notice_minutes": 30,
    "message": "ただいまメンテナンス中です。終了までしばらくお待ちください。",
    "notice_message": "まもなくメンテナンスを行います。この間はサービスをご利用いただけません。",
    "override": {"start": None, "end": None, "message": None},
    "exempt_exact": ["/api/health", "/api/v1/health", "/api/maintenance/status", "/api/v1/maintenance/status", "/login", "/logout"],
    "exempt_prefixes": ["/api/auth/", "/api/v1/auth/", "/api/scrape-jobs", "/api/scrape-queue", "/api/auto-scrape", "/api/admin",
                        "/api/monitor", "/api/quality-check", "/api/backfill", "/api/race-lists-backfill", "/api/train",
                        "/static/", "/data/image/", "/_next/"],
    "bypass": {"header": "X-Keiba-Maintenance-Bypass", "token_env": "KEIBA_MAINTENANCE_BYPASS_TOKEN",
               "admin_login": True, "internal_loopback": True},
    "notify": {"slack_start_end": False},
}

_cache: dict[str, Any] = {"at": -1e9, "cfg": None, "path": None}


# ───────────────────────── 設定 ─────────────────────────

def _merge(base: dict, over: Mapping) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue
        out[k] = _merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | str | None = None, *, ttl: float | None = None, mono: Callable[[], float] = time.monotonic) -> dict:
    """設定を読む（ttl 秒キャッシュ）。壊れている/無いときは既定値で動く（メンテナンスを誤って有効にしない側）。"""
    p = str(path or os.environ.get("KEIBA_MAINTENANCE_CONFIG") or CONFIG_PATH)
    if ttl is None:
        try:
            ttl = float(os.environ.get("KEIBA_MAINTENANCE_CACHE_TTL", CACHE_TTL_SEC))  # 試験用に短くできる（既定 30 秒）
        except ValueError:
            ttl = CACHE_TTL_SEC
    now = mono()
    if _cache["cfg"] is not None and _cache["path"] == p and now - _cache["at"] < ttl:
        return _cache["cfg"]
    cfg = DEFAULT_CONFIG
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        cfg = _merge(DEFAULT_CONFIG, raw)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        logger.warning("maintenance 設定を読めない（既定値で動作）: %s", exc)
    # 実行時の臨時メンテナンス（反映ランナーが窓を延長するときなど）。git 管理の設定を書き換えず、この 1 ファイルの override だけが上書きする
    try:
        ro = json.loads(runtime_override_path().read_text(encoding="utf-8"))
        if isinstance(ro, dict) and isinstance(ro.get("override"), dict):
            cfg = _merge(cfg, {"override": ro["override"]})
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        logger.warning("maintenance 実行時 override を読めない（無視）: %s", exc)
    _cache.update(at=now, cfg=cfg, path=p)
    return cfg


def runtime_override_path() -> Path:
    return Path(os.environ.get("KEIBA_MAINTENANCE_OVERRIDE") or REPO_ROOT / "data" / "local" / "meta" / "maintenance_override.json")


def clear_cache() -> None:
    _cache.update(at=-1e9, cfg=None, path=None)


def keiba_env(env: Mapping[str, str] | None = None) -> str:
    e = os.environ if env is None else env
    v = (e.get("KEIBA_ENV") or "").strip().lower()
    if v in ("dev", "development", "local"):
        return "dev"
    return v or "prod"  # 未設定は prod 扱い（auth.py と同じ）


def is_enabled(cfg: Mapping | None = None, env: Mapping[str, str] | None = None) -> bool:
    """メンテナンス時間帯の機能が有効か。環境変数が既定オフ。dev は更に KEIBA_MAINTENANCE_DEV=1 が要る。"""
    e = os.environ if env is None else env
    if (e.get("KEIBA_MAINTENANCE_WINDOW") or "").strip() != "1":
        return False
    if keiba_env(e) == "dev" and (e.get("KEIBA_MAINTENANCE_DEV") or "").strip() != "1":
        return False
    cfg = cfg if cfg is not None else load_config()
    return bool(cfg.get("enabled", True))


# ───────────────────────── 時刻判定（純粋関数） ─────────────────────────

def _tz(cfg: Mapping) -> ZoneInfo:
    return ZoneInfo(str(cfg.get("timezone") or "Asia/Tokyo"))


def _hm(s: str) -> dtime:
    h, m = str(s).split(":")[:2]
    return dtime(int(h), int(m))


def _parse_dt(v: Any, tz: ZoneInfo) -> datetime | None:
    """override 用: ISO 8601（オフセット付き）か 'YYYY-MM-DD HH:MM'（設定のタイムゾーン）。"""
    if not v:
        return None
    s = str(v).strip().replace("T", " ")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d.replace(tzinfo=tz) if d.tzinfo is None else d.astimezone(tz)


def daily_windows(cfg: Mapping, now: datetime) -> list[tuple[datetime, datetime]]:
    """前日・当日・翌日の毎日の窓（日跨ぎ設定にも対応）。"""
    tz = _tz(cfg)
    d = cfg.get("daily") or {}
    s, e = _hm(d.get("start", "03:00")), _hm(d.get("end", "04:00"))
    out = []
    base = now.astimezone(tz).date()
    for off in (-1, 0, 1):
        day: date = base + timedelta(days=off)
        st = datetime.combine(day, s, tzinfo=tz)
        en = datetime.combine(day, e, tzinfo=tz)
        if en <= st:
            en += timedelta(days=1)
        out.append((st, en))
    return out


def all_windows(cfg: Mapping, now: datetime) -> list[tuple[datetime, datetime, str]]:
    tz = _tz(cfg)
    wins: list[tuple[datetime, datetime, str]] = [(a, b, "daily") for a, b in daily_windows(cfg, now)]
    ov = cfg.get("override") or {}
    a, b = _parse_dt(ov.get("start"), tz), _parse_dt(ov.get("end"), tz)
    if a and b and b > a:
        wins.append((a, b, "override"))
    return sorted(wins, key=lambda w: w[0])


def compute_status(cfg: Mapping | None = None, now: datetime | None = None, *, enabled: bool | None = None) -> dict:
    """現在の状態。phase: normal / notice（開始 notice_minutes 前から）/ active。時刻は秒で整数化。"""
    cfg = cfg if cfg is not None else load_config()
    tz = _tz(cfg)
    now = (now or datetime.now(tz)).astimezone(tz)
    en = is_enabled(cfg) if enabled is None else enabled
    notice = timedelta(minutes=int(cfg.get("notice_minutes", 30)))
    ov = cfg.get("override") or {}
    wins = all_windows(cfg, now)
    covering = [w for w in wins if w[0] <= now < w[1]]
    cur = max(covering, key=lambda w: w[1]) if covering else None   # 重なるときは終了が遅い方（延長の override を優先）
    nxt = next((w for w in wins if w[0] > now), None)
    base = {"enabled": bool(en), "now": now.isoformat(timespec="seconds"), "timezone": str(cfg.get("timezone") or "Asia/Tokyo"),
            "active": False, "phase": "normal", "message": "", "source": None, "window_start": None, "window_end": None,
            "starts_in_sec": None, "remaining_sec": None, "ends_at": None, "ends_at_label": None, "retry_after": 0}
    if not en:
        if nxt:
            base.update(window_start=nxt[0].isoformat(timespec="seconds"), window_end=nxt[1].isoformat(timespec="seconds"))
        return base
    if cur:
        a, b, src = cur
        rem = max(0, int((b - now).total_seconds()))
        msg = (ov.get("message") if src == "override" and ov.get("message") else cfg.get("message")) or ""
        base.update(active=True, phase="active", message=msg, source=src, window_start=a.isoformat(timespec="seconds"),
                    window_end=b.isoformat(timespec="seconds"), remaining_sec=rem, ends_at=b.isoformat(timespec="seconds"),
                    ends_at_label=b.strftime("%H:%M"), retry_after=max(1, rem))
        return base
    if nxt:
        a, b, src = nxt
        until = max(0, int((a - now).total_seconds()))
        phase = "notice" if a - now <= notice else "normal"
        msg = (ov.get("message") if src == "override" and ov.get("message") else cfg.get("notice_message")) if phase == "notice" else ""
        base.update(phase=phase, message=msg or "", source=src, window_start=a.isoformat(timespec="seconds"),
                    window_end=b.isoformat(timespec="seconds"), starts_in_sec=until, ends_at=b.isoformat(timespec="seconds"),
                    ends_at_label=b.strftime("%H:%M"))
    return base


def now_for_request(env: Mapping[str, str] | None = None, at: str | None = None, cfg: Mapping | None = None) -> datetime | None:
    """テスト用の時計の上書き。dev 環境のときだけ有効（KEIBA_MAINTENANCE_NOW 環境変数 / ?at= クエリ）。本番では常に None。"""
    e = os.environ if env is None else env
    if keiba_env(e) != "dev":
        return None
    v = at or e.get("KEIBA_MAINTENANCE_NOW")
    return _parse_dt(v, _tz(cfg or load_config())) if v else None


# ───────────────────────── 除外・バイパス ─────────────────────────

def is_exempt(path: str, cfg: Mapping | None = None) -> bool:
    """メンテナンス中でも通す経路（ヘルスチェック・ログイン・運用 API・静的資産）。"""
    cfg = cfg if cfg is not None else load_config()
    if path in (cfg.get("exempt_exact") or []):
        return True
    return any(path.startswith(p) for p in (cfg.get("exempt_prefixes") or []))


def _is_loopback(addr: str | None) -> bool:
    import ipaddress

    s = (addr or "").strip().strip("[]")
    if not s:
        return False
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def is_internal_call(remote_addr: str | None, headers: Mapping[str, str]) -> bool:
    """同じホストの cron・ランナー・監視からの直接呼び出し（ループバック、かつプロキシ転送のヘッダー無し）。
    ブラウザのアクセスは Next.js のプロキシ経由で X-Forwarded-For が付くため対象外。"""
    if not _is_loopback(remote_addr):
        return False
    low = {k.lower(): v for k, v in headers.items()}
    return not (low.get("x-forwarded-for") or low.get("forwarded") or low.get("x-forwarded-host"))


def bypass_reason(*, remote_addr: str | None, headers: Mapping[str, str], is_admin: bool, cfg: Mapping | None = None,
                  env: Mapping[str, str] | None = None) -> str | None:
    """バイパスする理由（監査ログ用）。無ければ None。管理者の判定は呼び出し側（Cookie 検証）が渡す。"""
    cfg = cfg if cfg is not None else load_config()
    e = os.environ if env is None else env
    b = cfg.get("bypass") or {}
    low = {k.lower(): v for k, v in headers.items()}
    token = (e.get(str(b.get("token_env") or "KEIBA_MAINTENANCE_BYPASS_TOKEN")) or "").strip()
    given = (low.get(str(b.get("header") or "X-Keiba-Maintenance-Bypass").lower()) or "").strip()
    if token and given and hmac.compare_digest(token.encode(), given.encode()):
        return "token"
    if b.get("internal_loopback", True) and is_internal_call(remote_addr, headers):
        return "internal"
    if b.get("admin_login", True) and is_admin:
        return "admin"
    return None


# ───────────────────────── 応答（503） ─────────────────────────

def block_payload(status: Mapping) -> dict:
    return {"maintenance": True, "message": status.get("message") or DEFAULT_CONFIG["message"], "ends_at": status.get("ends_at"),
            "ends_at_label": status.get("ends_at_label"), "remaining_sec": status.get("remaining_sec"),
            "timezone": status.get("timezone")}


def block_headers(status: Mapping) -> dict[str, str]:
    return {"Retry-After": str(int(status.get("retry_after") or 60)), "Cache-Control": "no-store, max-age=0",
            "X-Keiba-Maintenance": "1", "X-Robots-Tag": "noindex"}


MAINTENANCE_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>メンテナンス中</title>
<style>body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0a0e17;color:#e6edf3;font-family:system-ui,'Noto Sans JP',sans-serif}}
main{{max-width:30rem;padding:2rem;text-align:center}}h1{{font-size:1.4rem}}p{{color:#9aa7b8;line-height:1.7}}b{{color:#e6edf3}}</style></head>
<body><main role="status" aria-live="polite"><h1>メンテナンス中です</h1><p>{message}</p>
<p>終了予定: <b>{ends}</b>（日本時間）</p></main></body></html>"""


def block_html(status: Mapping) -> str:
    import html

    return MAINTENANCE_HTML.format(message=html.escape(str(status.get("message") or "")), ends=html.escape(str(status.get("ends_at_label") or "")))


# ───────────────────────── 監査ログ ─────────────────────────

_last_phase: dict[str, str | None] = {"phase": None}


def audit_transition(status: Mapping, log: logging.Logger | None = None) -> str | None:
    """phase が変わったときだけ 1 行ログ（開始・終了・事前告知）。プロセスの起動直後の初回は記録のみ。"""
    log = log or logger
    prev, cur = _last_phase["phase"], str(status.get("phase"))
    _last_phase["phase"] = cur
    if prev is None or prev == cur:
        return None
    event = {"active": "start", "notice": "notice", "normal": "end" if prev == "active" else "notice_end"}[cur]
    log.info("maintenance %s: phase %s -> %s (ends_at=%s source=%s)", event, prev, cur, status.get("ends_at"), status.get("source"))
    return event
