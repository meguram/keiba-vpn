"""ホーム用: stg / prod × VPS / GCP の 4 マスの状態を 1 枚にする。ホストを見ない（GCS の ``ops/`` と Cloud Run の URL への HTTP 確認だけ）。

- VPS（prod）  ハートビートの鮮度・nginx / 画面 / cloudflared の応答・メモリ・ディスク。
- VPS（stg）   同じハートビートの ``stg_ui_http`` と ``keiba-ui-stg``（stg の画面を入れていなければ『未導入』）。
- GCP（prod）  Cloud Run ``keiba-api`` の本番（``KEIBA_API_URL_PROD`` の ``/api/health``）と、Jobs の最新の実行。
- GCP（stg）   Cloud Run ``keiba-api`` の stg タグ付きリビジョン（``KEIBA_API_URL_STG`` の ``/api/health``）。

API への確認は **人が画面を開いたときだけ**、結果を ``PROBE_TTL_S`` 使い回す。keiba-api は『CPU を常に割り当て』（インスタンス課金）なので、
定期的に叩くと落ちずに月 約 7,800 円になる（docs/operations/cost_recalc_20261028.md）。定期の確認は VPS のハートビート側でやり、API は叩かない。
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

PROBE_TTL_S = 600
LEVELS = ("green", "yellow", "red", "unknown")
_ORDER = {"green": 0, "unknown": 1, "yellow": 2, "red": 3}
_cache: dict[str, tuple[float, dict]] = {}

HttpGet = Callable[[str, float], tuple[int, int]]  # url, timeout → (HTTP コード（失敗は 0）, ミリ秒)


def http_get(url: str, timeout: float = 8.0) -> tuple[int, int]:
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "keiba-monitor/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = int(r.status)
    except urllib.error.HTTPError as e:
        code = int(e.code)
    except Exception:  # noqa: BLE001
        code = 0
    return code, round((time.monotonic() - t0) * 1000)


def worst(*levels: str) -> str:
    known = [lv for lv in levels if lv in _ORDER]
    return max(known, key=lambda x: _ORDER[x]) if known else "unknown"


def _item(name: str, value: Any, level: str = "green") -> dict:
    return {"name": name, "value": value, "level": level}


def _code_level(code: int | None) -> str:
    return "unknown" if code is None else "green" if code == 200 else "red"


def _age_text(sec: float | None) -> str:
    if sec is None:
        return "—"
    m = int(sec // 60)
    return f"{m} 分前" if m < 90 else f"{m // 60} 時間前"


def vps_cells(hb: dict) -> tuple[dict, dict]:
    """(prod, stg)。``hb`` は ``cloud_app.heartbeat_section`` の結果。"""
    age = hb.get("age_sec")
    never = age is None
    checks, units = hb.get("checks") or {}, hb.get("units") or {}
    if never:
        none = {"level": "red", "headline": "ハートビートが届いていません", "items": [], "hint": "VPS の keiba-heartbeat.timer と HEARTBEAT_URL（監視ポータル宛）を確認"}
        return dict(none), {**none, "level": "unknown", "headline": "未確認"}
    stale = any(a["key"] in ("stale", "no_heartbeat") for a in hb.get("alerts") or [])
    base = [_item("最終受信", _age_text(age), "red" if stale else "green")]
    if stale:
        prod = {"level": "red", "headline": "ハートビートが途絶えています", "items": base, "hint": "VPS・トンネル・ネットワークの停止の疑い"}
        return prod, {"level": "unknown", "headline": "VPS が応答していないため確認できません", "items": [], "hint": ""}
    items = base + [
        _item("画面 (Next.js 直)", checks.get("nextjs_http", "—"), _code_level(checks.get("nextjs_http"))),
        _item("nginx 経由", checks.get("nginx_http", "—"), _code_level(checks.get("nginx_http")) if "nginx_http" in checks else "unknown"),
    ]
    for n in ("nginx", "keiba-ui", "cloudflared-prod", "keiba-fetch-agent"):
        if n in units:
            st = units[n]
            items.append(_item(f"{n}（systemd）", st, "green" if st == "active" else "red" if st == "failed" else "yellow"))
    items.append(_item("空きメモリ", f"{hb['mem_available_mb']:.0f} MB" if hb.get("mem_available_mb") is not None else "—", "green"))
    items.append(_item("ディスク", f"{hb['disk_used_pct']:.0f}%" if hb.get("disk_used_pct") is not None else "—", "green" if (hb.get("disk_used_pct") or 0) <= 80 else "yellow"))
    lvl = worst(*(i["level"] for i in items if i["level"] != "unknown"), hb.get("level", "green"))
    prod = {"level": lvl, "headline": "正常" if lvl == "green" else "要確認", "items": items, "alerts": hb.get("alerts") or []}
    if "stg_ui_http" in checks or "keiba-ui-stg" in units:
        su = units.get("keiba-ui-stg")
        sitems = [_item("画面 (stg)", checks.get("stg_ui_http", "—"), _code_level(checks.get("stg_ui_http")) if "stg_ui_http" in checks else "unknown")]
        if su is not None:
            sitems.append(_item("keiba-ui-stg（systemd）", su, "green" if su == "active" else "red" if su == "failed" else "yellow"))
        sl = worst(*(i["level"] for i in sitems if i["level"] != "unknown"))
        stg = {"level": sl, "headline": "正常" if sl == "green" else "要確認", "items": sitems}
    else:
        stg = {"level": "unknown", "headline": "未導入", "items": [], "hint": "stg の画面を入れると表示されます（vps_05_stg_instance.sh）。ハートビートの送信側の更新も必要"}
    return prod, stg


def _probe(url: str | None, name: str, get: HttpGet, now: float, ttl: float = PROBE_TTL_S) -> dict:
    if not url:
        return {"state": "unconfigured", "env": name}
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return {**hit[1], "cached_age_s": round(now - hit[0])}
    code, ms = get(url.rstrip("/") + "/api/health", 8.0)
    res = {"state": "ok" if code == 200 else "down", "code": code, "ms": ms, "env": name}
    _cache[url] = (now, res)
    return {**res, "cached_age_s": 0}


def _gcp_cell(probe: dict, extra: list[dict] | None = None, tag: str = "") -> dict:
    if probe["state"] == "unconfigured":
        return {"level": "unknown", "headline": "接続先が未設定", "items": extra or [],
                "hint": f"監視ポータルの環境変数 {probe['env']}（Cloud Run の{tag}URL）を設定すると API の応答を確認します"}
    ok = probe["state"] == "ok"
    items = [_item("API /api/health", f"{probe['code'] or '応答なし'}（{probe['ms']} ms）", "green" if ok else "red"),
             _item("確認時刻", "今" if not probe.get("cached_age_s") else f"{probe['cached_age_s'] // 60} 分前（{PROBE_TTL_S // 60} 分は使い回し）", "green")] + (extra or [])
    lvl = worst(*(i["level"] for i in items if i["level"] != "unknown"))
    return {"level": lvl, "headline": "正常" if lvl == "green" else "要確認", "items": items}


def build(hb: dict, jobs: dict, *, env: dict[str, str] | None = None, get: HttpGet = http_get, now: float | None = None, probe: bool = True) -> dict:
    now = time.time() if now is None else now
    e = os.environ if env is None else env
    prod_vps, stg_vps = vps_cells(hb)
    if probe:
        p_prod, p_stg = _probe(e.get("KEIBA_API_URL_PROD"), "KEIBA_API_URL_PROD", get, now), _probe(e.get("KEIBA_API_URL_STG"), "KEIBA_API_URL_STG", get, now)
    else:
        p_prod = p_stg = {"state": "unconfigured", "env": "（確認を省略）"}
    ready = [r for r in jobs.get("rows", []) if r.get("ready")]
    bad = [r for r in ready if r["level"] in ("red", "yellow")]
    job_item = [_item("Jobs の最新の実行", f"{len(ready) - len(bad)}/{len(ready)} 正常" if ready else "準備中", "unknown" if not ready else jobs.get("level", "unknown"))]
    prod_gcp = _gcp_cell(p_prod, job_item, "本番の")
    stg_gcp = _gcp_cell(p_stg, None, "stg タグ付きリビジョンの")
    cells = {"prod": {"vps": prod_vps, "gcp": prod_gcp}, "stg": {"vps": stg_vps, "gcp": stg_gcp}}
    return {"cells": cells, "prod_level": worst(prod_vps["level"], prod_gcp["level"]), "stg_level": worst(stg_vps["level"], stg_gcp["level"]),
            "probed": probe}
