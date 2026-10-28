"""GCP（Cloud Run）で動く監視ポータル（読み取り専用）。ホストのローカルファイルを読まず、GCS の ``ops/`` の JSON だけを見る。

    python -m src.monitor.cloud_app            # 開発確認用（本番は gunicorn/uvicorn の WSGI として ``src.monitor.cloud_app:app``）

見るもの: VPS のハートビート（``ops/heartbeat/latest.json``）、Jobs の最新の実行（``ops/job_runs/<job>/latest.json``）、当日の異常
（``ops/alerts/<日>/``）、画面・取得役の配布（``ops/releases/<kind>/latest.json`` と ``applied/vps.json``）、切替判定
（``ops/cutover/status.json``）、成果物の一致（``ops/parity/<日>/``）。ホスト（研究用）の監視 ``src/monitor/app.py`` は別物で、運用の監視には使わない。
認証: Cloud Run は共有ヘッダ ``X-Keiba-Proxy-Key`` で守り（``KEIBA_PROXY_KEY_REQUIRED=1``）、前段に Cloudflare Access を置く。``/api/health`` だけ除外。
"""

from __future__ import annotations

import hmac
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))
LEVEL_ORDER = {"green": 0, "unknown": 1, "yellow": 2, "red": 3}


def _json(io: Any, key: str) -> Any:
    try:
        raw = io.get_bytes(key)
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception:  # noqa: BLE001  壊れた1件で画面全体を落とさない
        return None


def _worse(a: str, b: str) -> str:
    return a if LEVEL_ORDER.get(a, 1) >= LEVEL_ORDER.get(b, 1) else b


def heartbeat_section(io: Any, now: float) -> dict[str, Any]:
    from src.ops import heartbeat_schema as HS

    rec = _json(io, "ops/heartbeat/latest.json")
    alerts = HS.evaluate(rec or {}, now) if rec else [{"key": "no_heartbeat", "severity": "critical", "title": "VPS のハートビートが一度も届いていません", "detail": ""}]
    latest = (rec or {}).get("latest") or {}
    level = "red" if any(a["severity"] == "critical" for a in alerts) else "yellow" if alerts else "green"
    age = (now - float(rec["received_at"])) if rec and rec.get("received_at") else None
    return {"level": level, "age_sec": age, "alerts": alerts,
            "mem_available_mb": (latest.get("mem") or {}).get("available_mb"), "mem_used_mb": (latest.get("mem") or {}).get("used_mb"),
            "swap_used_mb": (latest.get("swap") or {}).get("used_mb"), "disk_used_pct": (latest.get("disk") or {}).get("root_used_pct"),
            "units": {n: u.get("active") for n, u in (latest.get("units") or {}).items()}, "checks": latest.get("checks") or {}}


def jobs_section(io: Any, now: datetime, job_ids: list[tuple[str, bool]]) -> dict[str, Any]:
    rows, level = [], "green"
    for jid, ready in job_ids:
        r = _json(io, f"ops/job_runs/{jid}/latest.json")
        if not r:
            rows.append({"job": jid, "ready": ready, "status": "未実行", "level": "unknown" if not ready else "yellow"})
            continue
        st = str(r.get("status") or "")
        lv = "green" if st in ("ok", "skipped") else "yellow" if st == "partial" else "red"
        rows.append({"job": jid, "ready": ready, "status": st, "reason": r.get("reason"), "started_at": r.get("started_at"), "duration_sec": r.get("duration_sec"),
                     "memory_peak_mib": r.get("memory_peak_mib"), "memory_limit_mib": r.get("memory_limit_mib"), "parallel": bool(r.get("parallel")), "level": lv})
        if ready:
            level = _worse(level, lv)
    return {"level": level, "rows": rows}


def alerts_section(io: Any, now: datetime, limit: int = 30) -> dict[str, Any]:
    day = now.strftime("%Y%m%d")
    items = []
    for key in sorted(io.list(f"ops/alerts/{day}/"))[-limit:]:
        a = _json(io, key)
        if a:
            items.append({"at": a.get("at"), "severity": a.get("severity"), "job": a.get("job"), "title": a.get("title"), "lines": a.get("lines") or []})
    crit = sum(1 for a in items if a["severity"] == "critical")
    return {"level": "red" if crit else "yellow" if items else "green", "count": len(items), "critical": crit, "items": list(reversed(items))}


def releases_section(io: Any) -> dict[str, Any]:
    rows, level = [], "green"
    for kind in ("ui", "fetch"):
        latest = _json(io, f"ops/releases/{kind}/latest.json") or {}
        applied = _json(io, f"ops/releases/{kind}/applied/vps.json") or {}
        in_sync = bool(latest) and applied.get("release_id") == latest.get("release_id")
        ok = str(applied.get("result", "")).lower() in ("ok", "applied", "success")
        lv = "green" if (in_sync and ok) else "yellow" if latest else "unknown"
        level = _worse(level, lv)
        rows.append({"kind": kind, "latest": latest.get("release_id"), "commit": (latest.get("commit") or "")[:8], "created_at": latest.get("created_at"),
                     "applied": applied.get("release_id"), "result": applied.get("result"), "in_sync": in_sync, "level": lv})
    return {"level": level, "rows": rows}


def cutover_section(io: Any) -> dict[str, Any]:
    st = _json(io, "ops/cutover/status.json")
    if not st:
        return {"level": "unknown", "summary": None}
    return {"level": "green", "summary": {k: st.get(k) for k in ("generated_at", "summary", "counts", "ready_to_cut") if k in st} or {"keys": sorted(st)[:8]}}


def parity_section(io: Any, now: datetime) -> dict[str, Any]:
    for back in range(0, 4):
        day = (now - timedelta(days=back)).strftime("%Y%m%d")
        keys = io.list(f"ops/parity/{day}/")
        if keys:
            rows = []
            for k in sorted(keys):
                r = _json(io, k) or {}
                rows.append({"job": k.rsplit("/", 1)[-1][:-5], "ok": bool(r.get("ok")), "status": r.get("status"), "problems": (r.get("problems") or [])[:3]})
            bad = [r for r in rows if not r["ok"]]
            return {"level": "yellow" if bad else "green", "date": day, "rows": rows}
    return {"level": "unknown", "date": None, "rows": []}


def job_ids_from_map() -> list[tuple[str, bool]]:
    from src.ops import job_map

    return [(j["id"], bool(j.get("ready"))) for j in job_map.load()["jobs"] if j.get("destination") == "gcp"]


def collect(io: Any, now_ts: float | None = None, job_ids: list[tuple[str, bool]] | None = None) -> dict[str, Any]:
    now_ts = now_ts if now_ts is not None else time.time()
    now = datetime.fromtimestamp(now_ts, JST)
    ids = job_ids if job_ids is not None else job_ids_from_map()
    sections = {
        "heartbeat": heartbeat_section(io, now_ts),
        "jobs": jobs_section(io, now, ids),
        "alerts": alerts_section(io, now),
        "releases": releases_section(io),
        "cutover": cutover_section(io),
        "parity": parity_section(io, now),
    }
    overall = "green"
    for s in sections.values():
        overall = _worse(overall, s["level"]) if s["level"] != "unknown" else overall
    return {"generated_at": now.isoformat(timespec="seconds"), "overall": overall, "sections": sections}


_PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>めぐ競馬 運用監視（GCP）</title>
<style>
body{background:#0d1117;color:#c9d1d9;font:14px system-ui,sans-serif;margin:0;padding:16px;max-width:1100px}
h1{font-size:18px}h2{font-size:15px;margin:22px 0 6px}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #30363d;padding:4px 8px;text-align:left;font-size:13px}
th{color:#8b949e}.green{color:#3fb950}.yellow{color:#d29922}.red{color:#f85149}.unknown{color:#8b949e}.dim{color:#8b949e}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;border:1px solid currentColor;font-size:12px}
</style></head><body>
<p><a href="/" style="color:#58a6ff">← ホーム</a></p>
<h1>運用監視（GCP） <span class="pill {{ d.overall }}">{{ d.overall }}</span> <span class="dim">{{ d.generated_at }}</span></h1>
{% set s = d.sections %}
<h2>VPS ハートビート <span class="pill {{ s.heartbeat.level }}">{{ s.heartbeat.level }}</span></h2>
<p class="dim">最終受信から {{ (s.heartbeat.age_sec // 60)|int if s.heartbeat.age_sec is not none else '—' }} 分 / 空きメモリ {{ s.heartbeat.mem_available_mb if s.heartbeat.mem_available_mb is not none else '—' }}MB /
swap {{ s.heartbeat.swap_used_mb if s.heartbeat.swap_used_mb is not none else '—' }}MB / ディスク {{ s.heartbeat.disk_used_pct if s.heartbeat.disk_used_pct is not none else '—' }}%</p>
{% for a in s.heartbeat.alerts %}<p class="{{ 'red' if a.severity == 'critical' else 'yellow' }}">● {{ a.title }} {{ a.detail }}</p>{% endfor %}
<h2>画面・取得役の配布 <span class="pill {{ s.releases.level }}">{{ s.releases.level }}</span></h2>
<table><thead><tr><th>種別</th><th>公開した版</th><th>commit</th><th>VPS が取り込んだ版</th><th>結果</th><th>一致</th></tr></thead><tbody>
{% for r in s.releases.rows %}<tr><td>{{ r.kind }}</td><td>{{ r.latest or '—' }}</td><td>{{ r.commit }}</td><td>{{ r.applied or '—' }}</td><td>{{ r.result or '—' }}</td><td class="{{ r.level }}">{{ '一致' if r.in_sync else '未反映' }}</td></tr>{% endfor %}
</tbody></table>
<h2>Jobs の最新の実行 <span class="pill {{ s.jobs.level }}">{{ s.jobs.level }}</span></h2>
<table><thead><tr><th>ジョブ</th><th>準備</th><th>状態</th><th>開始</th><th>所要(秒)</th><th>メモリ ピーク/上限(MiB)</th><th>並走</th><th>理由</th></tr></thead><tbody>
{% for r in s.jobs.rows %}<tr><td>{{ r.job }}</td><td>{{ 'ready' if r.ready else '準備中' }}</td><td class="{{ r.level }}">{{ r.status }}</td><td>{{ r.started_at or '—' }}</td><td>{{ r.duration_sec if r.duration_sec is not none else '—' }}</td>
<td>{{ r.memory_peak_mib or '—' }} / {{ r.memory_limit_mib or '—' }}</td><td>{{ '○' if r.parallel else '' }}</td><td class="dim">{{ r.reason or '' }}</td></tr>{% endfor %}
</tbody></table>
<h2>今日の異常通知 <span class="pill {{ s.alerts.level }}">{{ s.alerts.count }} 件（重大 {{ s.alerts.critical }}）</span></h2>
<table><thead><tr><th>時刻</th><th>重要度</th><th>ジョブ</th><th>内容</th></tr></thead><tbody>
{% for a in s.alerts['items'] %}<tr><td>{{ a.at }}</td><td class="{{ 'red' if a.severity == 'critical' else 'yellow' }}">{{ a.severity }}</td><td>{{ a.job or '' }}</td><td>{{ a.title }} <span class="dim">{{ a.lines|join(' / ') }}</span></td></tr>{% endfor %}
</tbody></table>
<h2>成果物の一致（並走） <span class="pill {{ s.parity.level }}">{{ s.parity.date or '—' }}</span></h2>
<table><thead><tr><th>ジョブ</th><th>一致</th><th>状態</th><th>問題</th></tr></thead><tbody>
{% for r in s.parity.rows %}<tr><td>{{ r.job }}</td><td class="{{ 'green' if r.ok else 'yellow' }}">{{ '○' if r.ok else '×' }}</td><td>{{ r.status or '' }}</td><td class="dim">{{ r.problems|join(' / ') }}</td></tr>{% endfor %}
</tbody></table>
<p class="dim">切替判定: {{ s.cutover.summary or '（まだ出ていません）' }}</p>
</body></html>"""


def _proxy_key_ok(headers: Any, env: dict[str, str] | None = None) -> bool:
    e = os.environ if env is None else env
    if (e.get("KEIBA_PROXY_KEY_REQUIRED") or "").strip().lower() not in ("1", "true", "yes", "on"):
        return True
    keys = [(e.get(n) or "").strip() for n in ("KEIBA_PROXY_KEY", "KEIBA_PROXY_KEY_NEXT")]
    keys = [k for k in keys if k]
    given = (headers.get("X-Keiba-Proxy-Key") or "").strip()
    return bool(keys and given and any(hmac.compare_digest(k.encode(), given.encode()) for k in keys))


# VPS からの定期送信は共有ヘッダ鍵ではなく HMAC 署名で認証する（VPS は Cloudflare Access を通らない）。ほかの経路は鍵が要る
SIGNED_POSTS = {"/ops/heartbeat", "/ops/traffic"}
_LEVEL_OF_PIPE = {"ok": "green", "wait": "green", "warn": "yellow", "ng": "red", "na": "unknown"}


def _default_notify(title: str, lines: list[str], severity: str = "warning") -> None:
    from src.utils.slack_notify import notify_event

    notify_event(title, lines, severity=severity, channel="alert")


def create_app(io: Any | None = None, now_fn: Any = time.time, job_ids: list[tuple[str, bool]] | None = None, *,
               hb_backend: Any = None, notify: Any = None, http_get: Any = None, env: dict[str, str] | None = None):
    from flask import Flask, jsonify, render_template, render_template_string, request

    from src.monitor import charts, environments, traffic
    from src.ops import ingest, pipeline_health, seo_audit

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = ingest.MAX_TRAFFIC_BYTES + 1024
    cache: dict[str, tuple[float, Any]] = {}

    def _io() -> Any:
        if io is not None:
            return io
        from src.ops.gcs_io import default_io

        return default_io()

    def _backend() -> Any:
        if hb_backend is not None:
            return hb_backend
        from src.ops import heartbeat_store

        return heartbeat_store.default_backend()

    def cached(key: str, ttl: float, fn: Any) -> Any:
        now = now_fn()
        hit = cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        val = fn()
        cache[key] = (now, val)
        return val

    def _now_dt() -> datetime:
        return datetime.fromtimestamp(now_fn(), JST)

    def pipeline_data() -> dict:
        def _build() -> dict:
            try:
                return pipeline_health.build(pipeline_health.GcsSources(_io()), _now_dt())
            except Exception as exc:  # noqa: BLE001  GCS の一時失敗で画面を落とさない
                n = _now_dt()
                return {"built_at": n.isoformat(timespec="seconds"), "days": [], "next": None, "today": n.strftime("%Y%m%d"), "gaps": [],
                        "overall": "na", "latest_date": None, "error": f"{type(exc).__name__}: {exc}"[:200]}
        return cached("pipeline", 120, _build)

    def traffic_view() -> dict:
        return cached("traffic", 60, lambda: traffic.view(traffic.read_bundle(_io()), _now_dt()))

    def seo_data() -> dict:
        return cached("seo", 300, lambda: seo_audit.read(_io()))

    def home_data(probe: bool = True) -> dict:
        ts = now_fn()
        hb, jobs = heartbeat_section(_io(), ts), jobs_section(_io(), _now_dt(), job_ids if job_ids is not None else job_ids_from_map())
        alerts = alerts_section(_io(), _now_dt())
        envs = environments.build(hb, jobs, env=env, get=http_get or environments.http_get, now=ts, probe=probe)
        pipe = pipeline_health.summary(pipeline_data())
        pl = _LEVEL_OF_PIPE.get(pipe.get("level") or "na", "unknown") if pipe.get("status") == "ok" else "unknown"
        overall = environments.worst(envs["prod_level"], alerts["level"], pl)
        return {"overall": overall, "env": envs, "pipe": pipe, "alerts": alerts, "traf": traffic.summary(traffic_view()), "seo": seo_data(),
                "probe_min": environments.PROBE_TTL_S // 60, "heartbeat": hb}

    def page(name: str, active: str, **ctx: Any):
        return render_template(f"portal/{name}.html", active=active, generated_at=_now_dt().strftime("%Y-%m-%d %H:%M JST"), **ctx)

    @app.before_request
    def _auth():
        if request.path == "/api/health":
            return None
        if request.method == "POST" and request.path in SIGNED_POSTS:
            return None  # 署名はハンドラ側で検証する
        if not _proxy_key_ok(request.headers):
            return jsonify({"error": "unauthorized", "reason": "proxy_key"}), 401
        return None

    @app.after_request
    def _no_store(resp):
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    @app.get("/api/health")
    def api_health():
        return jsonify({"status": "ok"})

    # ── VPS からの受信（HMAC 署名） ──
    @app.post("/ops/heartbeat")
    def ops_heartbeat():
        code, res = ingest.receive_heartbeat(request.headers, request.get_data(), _backend(), now=now_fn())
        return jsonify(res), code

    @app.post("/ops/traffic")
    def ops_traffic():
        code, res = ingest.receive_traffic(request.headers, request.get_data(), _io(), now=now_fn())
        if code == 200:
            cache.pop("traffic", None)
        return jsonify(res), code

    @app.post("/ops/heartbeat/check")
    def ops_heartbeat_check():
        return jsonify(ingest.run_check(_backend(), notify or _default_notify, now=now_fn()))

    # ── JSON ──
    @app.get("/api/overview")
    def api_overview():
        return jsonify(collect(_io(), now_fn(), job_ids))

    @app.get("/api/home")
    def api_home():
        d = home_data(probe=request.args.get("probe", "1") != "0")
        return jsonify({k: v for k, v in d.items() if k != "heartbeat"})

    @app.get("/api/pipeline")
    def api_pipeline():
        return jsonify(pipeline_data())

    @app.get("/api/traffic")
    def api_traffic():
        return jsonify(traffic_view())

    @app.get("/api/seo")
    def api_seo():
        return jsonify(seo_data())

    # ── 画面 ──
    @app.get("/")
    def index():
        d = home_data(probe=request.args.get("probe", "1") != "0")
        return page("home", "home", **d)

    @app.get("/pipeline")
    def pipeline_page():
        return page("pipeline", "pipeline", d=pipeline_data())

    @app.get("/traffic")
    def traffic_page():
        t = traffic_view()
        ctx: dict[str, Any] = {"t": t, "chart_trend": "", "chart_top": "", "chart_today": "", "chart_hours": ""}
        if t.get("status") == "ok":
            ser = t["series"]
            ctx["chart_trend"] = charts.combo([f"{x['date'][4:6]}/{x['date'][6:]}" for x in ser], [x["page_views"] for x in ser], [x["visitors"] for x in ser],
                                              bar_name="ページビュー", line_name="訪問者", w=1100, h=210, tick_every=10)
            ctx["chart_top"] = charts.hbars([(p, n) for p, n in t["top_pages_7d"]])
            ctx["chart_today"] = charts.hbars([(p, n) for p, n in t["top_pages_today"]])
            if t.get("hours_today"):
                ctx["chart_hours"] = charts.combo([f"{h}" for h in range(24)], t["hours_today"], t.get("hours_yesterday"), bar_name="今日", line_name="昨日",
                                                  w=360, h=150, tick_every=6)
        return page("traffic", "traffic", **ctx)

    @app.get("/seo")
    def seo_page():
        return page("seo", "seo", s=seo_data())

    @app.get("/ops")
    def ops_page():
        return render_template_string(_PAGE, d=collect(_io(), now_fn(), job_ids))

    return app


app = create_app()


def asgi_app():
    """Cloud Run 用の ASGI 入口（``uvicorn --factory src.monitor.cloud_app:asgi_app``）。Flask(WSGI) を包むだけ。"""
    try:
        from a2wsgi import WSGIMiddleware  # type: ignore
    except ImportError:
        from starlette.middleware.wsgi import WSGIMiddleware  # type: ignore
    return WSGIMiddleware(app)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8080")))
