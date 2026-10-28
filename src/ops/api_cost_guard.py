"""API 課金ガード: keiba-api の『課金される起動時間』を Cloud Monitoring から読み、月額の見込みと『夜間も起き続けている』を判定する。

keiba-api は『CPU を常に割り当て』（インスタンス課金）で、最後のリクエストから約 15 分は起動したままになる。定期的な ping（ヘルスチェック・cron・
外部の監視）が 15 分以内の間隔で入ると落ちずに月 数千〜1 万円超になる（2026-10-28 に実際に発生）。ping の種類を列挙しても漏れるので、
結果（課金される起動時間）そのものを見る: ``run.googleapis.com/container/billable_instance_time``（累積の秒、全リビジョン合計）。

- 判定は純粋関数 ``analyze``（時間ごとの秒数 → 月額の見込み・夜間の起動・異常の一覧）。
- 取得は ``fetch_hourly``（Monitoring API の REST。ADC。必要な権限は roles/monitoring.viewer）。取れなければ『確認できない』を返すだけで落ちない。
- ``run_check`` が結果を GCS（``ops/cost_guard/latest.json``）に置き、新しい異常・復旧だけを通知する（Scheduler が 30 分おきに呼ぶ）。
ホストを見ない（GCP の API と GCS だけ）。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.ops import heartbeat_schema as S

JST = timezone(timedelta(hours=9))
METRIC = "run.googleapis.com/container/billable_instance_time"
LATEST = "ops/cost_guard/latest.json"
STATE = "ops/cost_guard/state.json"
QUIET_JST = range(1, 5)          # 人が使わない時間帯（JST 1〜4 時台）
QUIET_ALIVE_S = 1800             # 1 時間のうち 30 分以上起動していたら『起き続けている』
API_BUDGET_YEN = 800             # API 分の月額の目安（全体の予算 2,000 円のうち。GCS が上限で約 1,200 円）
VCPU, GIB = 1.0, 2.0             # keiba-api の割り当て（deploy_cloud_run_api.sh）
_DEFAULTS = {"fx": 150.0, "vcpu_s": 1.8e-05, "gib_s": 2e-06, "free_vcpu": 240000.0, "free_gib": 450000.0}
CONFIG = Path(__file__).resolve().parents[2] / "config" / "cost_assumptions.json"


def prices() -> dict[str, float]:
    p = dict(_DEFAULTS)
    try:
        d = json.loads(CONFIG.read_text(encoding="utf-8"))
        p.update(fx=float(d["fx_jpy_per_usd"]), vcpu_s=float(d["prices"]["cloud_run_instance_vcpu_second"]["value"]),
                 gib_s=float(d["prices"]["cloud_run_instance_gib_second"]["value"]),
                 free_vcpu=float(d["free_tier"]["cloud_run_instance_vcpu_seconds"]["value"]),
                 free_gib=float(d["free_tier"]["cloud_run_instance_gib_seconds"]["value"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return p


def monthly_yen(instance_seconds_per_month: float, pr: dict[str, float], *, free: bool = True) -> float:
    vcpu, gib = instance_seconds_per_month * VCPU, instance_seconds_per_month * GIB
    if free:
        vcpu, gib = max(0.0, vcpu - pr["free_vcpu"]), max(0.0, gib - pr["free_gib"])
    return (vcpu * pr["vcpu_s"] + gib * pr["gib_s"]) * pr["fx"]


def analyze(hours: list[dict], now_ts: float, pr: dict[str, float] | None = None) -> dict:
    """``hours`` = [{"end": epoch秒, "sec": その 1 時間の課金される起動時間(全インスタンス合計)}]。"""
    pr = pr or prices()
    last24 = [h for h in hours if now_ts - 86400 < h["end"] <= now_ts + 60]
    day_s = sum(h["sec"] for h in last24)
    covered_h = len(last24)
    projected = monthly_yen(day_s / covered_h * 24 * 30, pr) if covered_h else None
    projected_nofree = monthly_yen(day_s / covered_h * 24 * 30, pr, free=False) if covered_h else None

    def quiet(h: dict) -> bool:
        mid = datetime.fromtimestamp(h["end"] - 1800, JST)
        return mid.hour in QUIET_JST

    quiet_alive = [h for h in last24 if quiet(h) and h["sec"] >= QUIET_ALIVE_S]
    avg_instances = day_s / (covered_h * 3600) if covered_h else None
    alerts: list[dict] = []
    if len(quiet_alive) >= 2:
        sev = "critical" if len(quiet_alive) >= 4 else "warning"
        alerts.append({"key": "api_alive_overnight", "severity": sev, "title": "keiba-api が夜間も起動し続けています",
                       "detail": f"人が使わない時間帯（JST 1〜4 時台）に {len(quiet_alive)} 時間分、30 分以上起動（平均 {avg_instances:.1f} 台）。"
                                 "定期的なリクエスト（ヘルスチェック・cron・外部の監視）が /api/ を叩いていないか確認"})
    if projected is not None and projected >= API_BUDGET_YEN:
        sev = "critical" if projected >= API_BUDGET_YEN * 2 else "warning"
        alerts.append({"key": "api_cost_projection", "severity": sev, "title": "keiba-api の月額の見込みが目安を超えています",
                       "detail": f"直近 24 時間の起動時間から月 約 {projected:,.0f} 円（無料枠なしなら約 {projected_nofree:,.0f} 円）。API 分の目安 {API_BUDGET_YEN} 円"})
    level = "red" if any(a["severity"] == "critical" for a in alerts) else "yellow" if alerts else "green"
    return {"level": level, "alerts": alerts, "covered_hours": covered_h, "avg_instances_24h": avg_instances, "instance_seconds_24h": day_s,
            "projected_month_yen": projected, "projected_month_yen_no_free": projected_nofree, "api_budget_yen": API_BUDGET_YEN,
            "quiet_alive_hours": len(quiet_alive),
            "series": [{"end": h["end"], "sec": round(h["sec"])} for h in hours[-48:]]}


def fetch_hourly(project: str, service: str, now_ts: float, hours: int = 48, session: Any = None) -> list[dict]:
    """Monitoring API（REST）から 1 時間ごとの合計秒を取る。session はテストで差し替える（``get(url, params=, timeout=)``）。"""
    if session is None:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/monitoring.read"])
        session = AuthorizedSession(creds)
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    params = {"filter": f'metric.type="{METRIC}" AND resource.labels.service_name="{service}"',
              "interval.startTime": iso(now_ts - hours * 3600), "interval.endTime": iso(now_ts),
              "aggregation.alignmentPeriod": "3600s", "aggregation.perSeriesAligner": "ALIGN_SUM", "aggregation.crossSeriesReducer": "REDUCE_SUM"}
    r = session.get(f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries", params=params, timeout=20)
    if r.status_code != 200:
        raise PermissionError(f"HTTP {r.status_code}") if r.status_code in (401, 403) else RuntimeError(f"HTTP {r.status_code}")
    out: list[dict] = []
    for ts in r.json().get("timeSeries", []):
        for p in ts.get("points", []):
            v = p.get("value", {})
            end = datetime.fromisoformat(p["interval"]["endTime"].replace("Z", "+00:00")).timestamp()
            out.append({"end": end, "sec": float(v.get("doubleValue", v.get("int64Value", 0)) or 0)})
    return sorted(out, key=lambda h: h["end"])


def evaluate(now_ts: float, *, project: str | None = None, service: str | None = None, fetch: Callable[..., list[dict]] = fetch_hourly) -> dict:
    project = project or os.environ.get("GCS_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
    service = service or os.environ.get("KEIBA_API_SERVICE") or "keiba-api"
    base = {"checked_at": datetime.fromtimestamp(now_ts, JST).isoformat(timespec="seconds"), "service": service}
    if not project:
        return {**base, "status": "unavailable", "level": "unknown", "alerts": [], "reason": "プロジェクト ID が未設定（GCS_PROJECT_ID）"}
    try:
        hours = fetch(project, service, now_ts)
    except PermissionError:
        return {**base, "status": "unavailable", "level": "unknown", "alerts": [],
                "reason": "Monitoring の読み取り権限がありません（keiba-monitor の SA に roles/monitoring.viewer が必要）"}
    except Exception as exc:  # noqa: BLE001  取れない日があっても画面と点検を落とさない
        return {**base, "status": "unavailable", "level": "unknown", "alerts": [], "reason": f"取得に失敗: {type(exc).__name__}"}
    if not hours:
        return {**base, "status": "unavailable", "level": "unknown", "alerts": [], "reason": "指標のデータがありません"}
    return {**base, "status": "ok", **analyze(hours, now_ts)}


def _get(io: Any, key: str) -> Any:
    try:
        raw = io.get_bytes(key)
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception:  # noqa: BLE001
        return None


def read_latest(io: Any) -> dict | None:
    return _get(io, LATEST)


def run_check(io: Any, notify: Callable[..., Any], *, now: float | None = None, **kw: Any) -> dict:
    """評価して GCS に置き、新しい異常・復旧だけを通知する（同じ異常は RENOTIFY_S まで再通知しない）。"""
    now = time.time() if now is None else now
    res = evaluate(now, **kw)
    io.put_bytes(LATEST, json.dumps(res, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    if res["status"] != "ok":
        return {"ok": False, "status": res["status"], "reason": res.get("reason"), "sent": [], "recovered": []}
    to_send, recovered, new_state = S.plan_notifications(res["alerts"], _get(io, STATE) or {}, now)
    io.put_bytes(STATE, json.dumps(new_state).encode("utf-8"), content_type="application/json")
    for a in to_send:
        notify(a["title"], [a["detail"]], a["severity"])
    if recovered:
        notify("keiba-api の課金の異常が解消しました", [", ".join(recovered)], "normal")
    return {"ok": True, "status": "ok", "level": res["level"], "sent": [a["key"] for a in to_send], "recovered": recovered,
            "projected_month_yen": res["projected_month_yen"]}
