"""監視ポータルの『アクセス』: VPS が送ったアクセス集計（GCS の ``ops/traffic/bundle.json``）から、期間ごとの KPI・上位ページ・流入元・クローラーを作る。

ホストを見ない。受信は ``src/ops/ingest.py``（keiba-monitor が受ける）、集計は VPS の ``traffic_client.py``（nginx のログ → 日別）。
訪問者 = 同じ日の (IP, UA) の組の数（IP と UA は VPS の中で畳み、ここには来ない）。ページビュー = bot・API・静的ファイル・
クライアント遷移の先読み（``_rsc``）を除いた HTML の GET（200/304）。ログは約 14 日で消えるが、VPS が日別の集計を積むので推移は 90 日見られる。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))
OBJECT = "ops/traffic/bundle.json"
STALE_S = 45 * 60          # 15 分おきの送信が 3 回止まったら古いとみなす
SETUP_SCRIPT = "scripts/ops/vps/vps_07_traffic_timer.sh"


def read_bundle(io: Any) -> dict | None:
    try:
        raw = io.get_bytes(OBJECT)
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception:  # noqa: BLE001  壊れた 1 件で画面全体を落とさない
        return None


def _sum(days: list[dict], key: str) -> int:
    return sum(int(d.get(key) or 0) for d in days)


def _merge_counter(days: list[dict], key: str, n: int) -> list[list]:
    c: Counter = Counter()
    for d in days:
        for k, v in d.get(key) or []:
            c[k] += v
    return [[k, v] for k, v in c.most_common(n)]


def _merge_map(days: list[dict], key: str) -> dict[str, int]:
    c: Counter = Counter()
    for d in days:
        for k, v in (d.get(key) or {}).items():
            c[k] += v
    return dict(c)


def _pct(a: float, b: float) -> float | None:
    return round((a - b) / b * 100, 1) if b else None


def view(bundle: dict | None, now: datetime | None = None) -> dict:
    now = now or datetime.now(JST)
    if not bundle:
        return {"status": "not_collected", "hint": "まだ VPS から届いていません", "setup": SETUP_SCRIPT}
    ymd = lambda dt: dt.strftime("%Y%m%d")  # noqa: E731
    days: dict[str, dict] = bundle.get("days") or {}
    age = now.timestamp() - float(bundle.get("received_at") or 0) if bundle.get("received_at") else None
    base = {"collected_at": datetime.fromtimestamp(bundle["received_at"], JST).isoformat(timespec="seconds") if bundle.get("received_at") else None,
            "age_s": round(age) if age is not None else None, "stale": (age is None or age > STALE_S), "setup": SETUP_SCRIPT, "days_stored": len(days)}
    if not bundle.get("readable"):
        return {**base, "status": "unreadable", "reason": bundle.get("reason") or "ログを読めません"}
    today, yday = ymd(now), ymd(now - timedelta(days=1))
    cur7 = [days[d] for d in (ymd(now - timedelta(days=i)) for i in range(7)) if d in days]
    prev7 = [days[d] for d in (ymd(now - timedelta(days=i)) for i in range(7, 14)) if d in days]
    t, y = days.get(today) or {}, days.get(yday) or {}
    win = bundle.get("windows") or {}
    series = []
    for i in range(89, -1, -1):
        k = ymd(now - timedelta(days=i))
        d = days.get(k)
        series.append({"date": k, "page_views": d.get("page_views") if d else None, "visitors": d.get("visitors") if d else None,
                       "bots": sum((d.get("bot_kind") or {}).values()) if d else None, "errors": (d.get("status") or {}).get("5xx", 0) if d else None})
    bot_kind = _merge_map(cur7, "bot_kind")
    bot_status: dict[str, dict[str, int]] = {}
    for k, v in _merge_map(cur7, "bot_status").items():
        name, _, cls = k.rpartition(" ")
        bot_status.setdefault(name, {})[cls] = v
    crawlers = [{"name": n, "hits": h, "status": bot_status.get(n, {})} for n, h in _merge_counter(cur7, "bots", 15)]
    pv7, pv_prev = _sum(cur7, "page_views"), _sum(prev7, "page_views")
    return {
        **base, "status": "ok",
        "kpi": {
            "today": {"page_views": t.get("page_views"), "visitors": t.get("visitors") if t else None, "requests": t.get("requests")},
            "yesterday": {"page_views": y.get("page_views"), "visitors": y.get("visitors"), "requests": y.get("requests")},
            "d7": {"page_views": pv7, "visitors": win.get("7d"), "days": len(cur7), "pv_vs_prev_pct": _pct(pv7, pv_prev) if prev7 else None},
            "d14": {"visitors": win.get("14d")},
            "bots_7d": sum(bot_kind.values()), "api_7d": _sum(cur7, "api"), "errors_5xx_7d": _merge_map(cur7, "status").get("5xx", 0),
        },
        "series": series,
        "top_pages_7d": _merge_counter(cur7, "pages", 10), "top_pages_today": [list(x) for x in (t.get("pages") or [])[:10]],
        "top_patterns_7d": _merge_counter(cur7, "patterns", 10),
        "referrers_7d": _merge_counter(cur7, "referrers", 10), "referrer_kind_7d": _merge_map(cur7, "referrer_kind"),
        "devices_7d": _merge_map(cur7, "devices"), "hours_today": t.get("hours"), "hours_yesterday": y.get("hours"),
        "bot_kind_7d": bot_kind, "crawlers": crawlers,
        "err_pages_7d": _merge_counter(cur7, "err_pages", 10), "err5xx_7d": _merge_counter(cur7, "err5xx", 10),
        "seo_files_7d": _merge_map(cur7, "seo_files"),
    }


def summary(d: dict) -> dict:
    """ホームのカード用の短い要約。"""
    if d.get("status") != "ok":
        return {"status": d.get("status"), "reason": d.get("reason") or d.get("hint"), "setup": d.get("setup")}
    k = d["kpi"]
    return {"status": "ok", "stale": d.get("stale"), "age_s": d.get("age_s"), "today_pv": k["today"]["page_views"], "today_visitors": k["today"]["visitors"],
            "yesterday_pv": k["yesterday"]["page_views"], "d7_pv": k["d7"]["page_views"], "d7_visitors": k["d7"]["visitors"],
            "pv_vs_prev_pct": k["d7"]["pv_vs_prev_pct"], "top": (d["top_pages_7d"] or [[None, 0]])[0]}
