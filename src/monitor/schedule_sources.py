"""定期実行ジョブの記録を集める（監視ポータル・毎朝の点検が使う。GCP の中で動く前提＝GCS の読み出しは転送料なし）。

- VPS: ``ops/heartbeat/vps/status.json``（VPS の見張り src/ops/sched_status.py が 15 分おきに書く）を 1 回読む。
- GCP: Cloud Scheduler の ``jobs.list`` 1 回と、Run Admin の executions の list（一覧にある Run Job ごとに 1 回）。どちらも読み取り無料。
  グループの中のジョブの結果は ``ops/job_runs/<job>/latest.json``（active なグループの分だけ）。
- ホストの cron は記録を GCP に送らないので判定しない（vps_alive などはホストの cron_status で見る）。

読めないときは例外を出さず、その場所を「読めない」として返す（画面は落とさない。schedule_rules がまとめて 1 件の注意にする）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

REGION = "asia-northeast1"
VPS_OBJECT = "ops/heartbeat/vps/status.json"


def _iso(v: Any) -> float | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _json(io: Any, key: str) -> Any:
    try:
        raw = io.get_bytes(key)
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception:  # noqa: BLE001
        return None


class GcpApi:
    """Scheduler・Run Admin の読み取り（REST）。session はテストで差し替える（``get(url, params=, timeout=)``）。"""

    def __init__(self, project: str | None = None, session: Any = None) -> None:
        self.project = project or os.environ.get("GCS_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
        self._session = session

    def _s(self) -> Any:
        if self._session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            creds, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform.read-only"])
            self.project = self.project or proj or ""
            self._session = AuthorizedSession(creds)
        return self._session

    def _get(self, url: str, params: dict | None = None) -> dict:
        r = self._s().get(url, params=params or {}, timeout=5)   # ホームの表示を長く止めない（失敗はまとめて「読めない」1 件）
        if r.status_code in (401, 403):
            raise PermissionError("権限がありません（roles/cloudscheduler.viewer・roles/run.viewer が必要）")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()

    def scheduler_jobs(self) -> list[dict[str, Any]]:
        d = self._get(f"https://cloudscheduler.googleapis.com/v1/projects/{self.project}/locations/{REGION}/jobs", {"pageSize": 100})
        return [{"name": str(j.get("name", "")).rsplit("/", 1)[-1], "schedule": j.get("schedule"), "tz": j.get("timeZone"), "state": j.get("state"),
                 "last_attempt": _iso(j.get("lastAttemptTime")), "status_code": (j.get("status") or {}).get("code")} for j in d.get("jobs") or []]

    def executions(self, run_job: str, limit: int = 20) -> list[dict[str, Any]]:
        d = self._get(f"https://run.googleapis.com/v2/projects/{self.project}/locations/{REGION}/jobs/{run_job}/executions", {"pageSize": limit})
        out = []
        for e in d.get("executions") or []:
            start = _iso(e.get("startTime")) or _iso(e.get("createTime"))
            end = _iso(e.get("completionTime"))
            failed = int(e.get("failedCount") or 0) + int(e.get("cancelledCount") or 0)
            out.append({"name": str(e.get("name", "")).rsplit("/", 1)[-1], "start": start, "end": end,
                        "ok": None if end is None else (failed == 0 and int(e.get("succeededCount") or 0) > 0), "code": None})
        return [x for x in out if x["start"] is not None]


def in_slots(job: dict[str, Any], executions: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
    """共用の Run Job の実行のうち、そのジョブの予定の窓（予定の 2 分前〜猶予）に始まったもの。予定の無いものはそのまま。"""
    from src.ops import scheduled_jobs as SJ

    cron = (job.get("schedule") or {}).get("cron")
    if not cron:
        return list(executions)
    grace = float(job.get("grace_min") or 30) * 60
    slots = SJ.slots_before(cron, now, n=200, days=8)
    return [e for e in executions if any(t - 120 <= e["start"] <= t + grace for t in slots)]


def gather(io: Any, reg: dict[str, Any], job_map: dict[str, Any], *, api: Any = None, now: float) -> dict[str, Any]:
    """``{"obs", "sources", "unregistered", "inner"}``（schedule_rules.evaluate の入力と、グループの中のジョブの結果）。"""
    obs: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    unregistered: list[dict[str, Any]] = []
    inner: dict[str, list[dict[str, Any]]] = {}
    jobs = reg["jobs"]

    # VPS
    vps = _json(io, VPS_OBJECT)
    if isinstance(vps, dict):
        sources["vps"] = {"seen_at": vps.get("written_at"), "as_of": vps.get("written_at")}
        if vps.get("error"):
            sources["vps"]["error"] = f"VPS の見張りが systemd を読めない: {str(vps['error'])[:120]}"
        for jid, o in (vps.get("jobs") or {}).items():
            obs[jid] = o
    else:
        sources["vps"] = {"seen_at": None}

    # GCP（API はポータル・毎朝のジョブ＝GCP の中でだけ読む。手元の開発・テストで認証を探しに行って待たない）
    if api is None:
        from src.config.gcp_guard import on_gcp

        api = GcpApi() if on_gcp() else None
    try:
        if api is None:
            raise RuntimeError("GCP の外では読まない（Cloud Run・Jobs の中で読む）")
        sched = api.scheduler_jobs()
        by_sched = {(j.get("ref") or {}).get("scheduler"): j for j in jobs if j["where"] == "gcp" and (j.get("ref") or {}).get("scheduler")}
        for s in sched:
            j = by_sched.get(s["name"])
            if j is None:
                unregistered.append({"name": s["name"], "schedule": s.get("schedule"), "state": s.get("state"), "where": "gcp"})
            elif j["status"] in ("retired", "paused") and s.get("state") == "ENABLED":
                unregistered.append({"name": f"{s['name']}（一覧では{'廃止' if j['status'] == 'retired' else '停止中'}なのに有効）",
                                     "schedule": s.get("schedule"), "state": s.get("state"), "where": "gcp"})
        # Run Job ごとの実行を、それを使う active なグループに割り当てる（使うものが無い Run Job の実行は想定外として扱う）
        run_jobs = sorted({(j.get("ref") or {}).get("run_job") for j in jobs if j["where"] == "gcp" and (j.get("ref") or {}).get("run_job")})
        for rj in run_jobs:
            ex = api.executions(rj)
            owners = [j for j in jobs if (j.get("ref") or {}).get("run_job") == rj and j["status"] == "active"]
            if owners:
                for j in owners:
                    # Run Job はサイズごとの共用。そのグループの予定の窓（予定〜猶予）に始まった実行だけを、そのグループの回とみなす
                    obs[j["id"]] = {"runs": [dict(e) for e in in_slots(j, ex, now)]}
            else:
                for j in jobs:
                    if (j.get("ref") or {}).get("run_job") == rj and j["kind"] == "run_job":
                        obs[j["id"]] = {"runs": ex}
                recent = [e for e in ex if now - e["start"] <= 24 * 3600]
                if recent and not any(j["kind"] == "run_job" and (j.get("ref") or {}).get("run_job") == rj for j in jobs):
                    unregistered.append({"name": f"{rj} の実行（直近 24 時間に {len(recent)} 回。使う定期実行が一覧に無い）", "schedule": None, "state": None, "where": "gcp"})
        sources["gcp"] = {"seen_at": now}
    except Exception as exc:  # noqa: BLE001
        sources["gcp"] = {"error": str(exc)[:160] or type(exc).__name__}

    # active なグループの中のジョブの結果（ops/job_runs/<job>/latest.json。状態の値は job_dispatch の BAD）
    from src.ops.job_dispatch import BAD
    groups = job_map.get("scheduler_groups") or {}
    for j in jobs:
        gid = (j.get("ref") or {}).get("group")
        if j["where"] != "gcp" or j["status"] != "active" or gid not in groups:
            continue
        rows = []
        for inner_id in groups[gid].get("jobs") or []:
            r = _json(io, f"ops/job_runs/{inner_id}/latest.json") or {}
            rows.append({"job": inner_id, "status": r.get("status"), "started_at": r.get("started_at"), "duration_sec": r.get("duration_sec"), "reason": r.get("reason")})
        inner[j["id"]] = rows
        runs = (obs.get(j["id"]) or {}).get("runs") or []
        if runs:
            # 実行（コンテナ）は成功でも、その回の中のジョブが失敗していれば失敗として数える（古い回の latest.json は見ない）
            last = sorted(runs, key=lambda r: r["start"], reverse=True)[0]
            hi = (last.get("end") or now) + 60
            bad = [r for r in rows if r["status"] in BAD and (_iso(r["started_at"]) or 0) >= last["start"] - 120 and (_iso(r["started_at"]) or 0) <= hi]
            if bad and last.get("end") is not None:
                last["ok"] = False
    return {"obs": obs, "sources": sources, "unregistered": unregistered, "inner": inner}


def build_view(io: Any, *, now: float, api: Any = None, reg: dict[str, Any] | None = None, job_map: dict[str, Any] | None = None) -> dict[str, Any]:
    """画面・毎朝の点検用: 一覧＋判定＋中身。"""
    from src.ops import schedule_rules as R
    from src.ops import scheduled_jobs as SJ

    reg = reg or SJ.load()
    if job_map is None:
        job_map = json.loads((SJ.REPO / "config" / "gcp_job_map.json").read_text(encoding="utf-8"))
    g = gather(io, reg, job_map, api=api, now=now)
    jobs = [j for j in reg["jobs"] if j["where"] in ("gcp", "vps")]
    res = R.evaluate(jobs, g["obs"], now, sources={k: v for k, v in g["sources"].items()}, unregistered=g["unregistered"])
    by_id = {r["id"]: r for r in res["rows"]}
    rows = []
    for j in reg["jobs"]:
        r = by_id.get(j["id"]) or {"level": "unknown", "label": "判定なし", "reasons": ["ホストの cron は記録を GCP に送らないので判定しない"], "last": None, "next_due": None, "estimated": False}
        runs = sorted(((g["obs"].get(j["id"]) or {}).get("runs") or []), key=lambda x: x["start"], reverse=True)[:10]
        rows.append({"job": j, **r, "runs": runs, "inner": g["inner"].get(j["id"]) or []})
    order = {"red": 0, "yellow": 1, "unknown": 2, "green": 3, "gray": 4}
    rows.sort(key=lambda x: ({"gcp": 0, "vps": 1, "host": 2}[x["job"]["where"]], order.get(x["level"], 9), x["job"]["id"]))
    return {"overall": res["overall"], "label": res["label"], "counts": res["counts"], "alerts": res["alerts"], "rows": rows, "sources": g["sources"],
            "result": res}
