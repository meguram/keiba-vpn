"""開催日・定期の仕事を前もって予約する（司令塔の役目）。

    python -m src.ops.reservations race-day --date YYYYMMDD [--dry-run]     # 翌開催日の全レース（前夜に 1 回。冪等）
    python -m src.ops.reservations recurring --horizon-hours 6 [--dry-run]  # VPS 実行役の定期タスク（毎時。冪等）

- **取得**（VPS の実行役が pull して 1 本ずつ実行）: ``gs://BUCKET/ops/fetch_tasks/pending/<task_id>.json``（``src/ops/fetch_task.py``）。
  各レースの T-60 / T-15（バンドル）と、``KEIBA_ODDS_LATE_SNAPSHOTS=1`` のときだけ T-10 / T-5 / T-2（単複のみ）、馬体重の検知（race-weight-check）。
  計画は ``fetch_plan.plan_day``（auto_scrape_queue と同じ時刻規則）を使う。
- **予測**（GCP）: Cloud Tasks の予約実行で Run Job（``predict-race``）を起動する。発走 45 分前のフォールバック（``predict-<race_id>``）。
  馬体重の検知後（VPS 実行役が ``ops/events/weights_announced/`` に置く）は ``events.py`` が即時の ``predict-<race_id>-w`` を積む。どちらも task_id が固定なので二重登録しても一度しか動かない。
- **定期**: job_map の VPS ジョブ（``schedule_jst``）を cron 式から展開して、今から ``horizon`` 時間先までのタスクを積む。
  ``ready=false`` のジョブは積まない（理由は出力に載せる）。

Cloud Tasks から Run Job を起動する形: ``POST https://run.googleapis.com/v2/projects/<P>/locations/<R>/jobs/<JOB>:run``
（本文 ``{"overrides": {"containerOverrides": [{"args": […]}]}}``、サービスアカウントの OAuth トークン）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.ops import fetch_task as FT
from src.ops import job_map

JST = ZoneInfo("Asia/Tokyo")
PREDICT_LEAD_MINUTES = 45
DEFAULT_QUEUE = "keiba-predict"
RUN_API = "https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/{job}:run"
OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


# ───────────────────────── 取得タスク ─────────────────────────

# 開催日のレース別タスク（T-60 / T-15 / 直前 T-10・T-5・T-2 / 馬体重の検知 race-weight-check）の計画は fetch_plan.plan_day が正（auto_scrape_queue と同じ時刻規則）。

def owned_tasks(env: dict[str, str] | None = None) -> set[str]:
    """VPS の実行役が担当するタスク（KEIBA_FETCH_AGENT_TASKS、無ければ config/fetch_owner.json の agent_tasks）。
    担当外のタスクを積むと誰も拾わず期限切れの failed が溜まるので、積むのは担当のものだけ（担当が空の間は何も積まない＝ホストの cron が全部やる）。"""
    e = os.environ if env is None else env
    raw = (e.get("KEIBA_FETCH_AGENT_TASKS") or "").strip()
    if raw:
        return {t.strip() for t in raw.split(",") if t.strip()}
    from src.ops.fetch_owner import agent_tasks

    return agent_tasks()


def recurring_tasks(doc: dict[str, Any], now: datetime, horizon_hours: float, owned: set[str] | None = None) -> tuple[list[FT.FetchTask], list[dict[str, str]]]:
    """VPS ジョブの schedule_jst を ``[now, now+horizon)`` に展開する。(タスク, 積まなかったジョブと理由)。"""
    owned = owned_tasks() if owned is None else owned
    end = now + timedelta(hours=horizon_hours)
    tasks: list[FT.FetchTask] = []
    skipped: list[dict[str, str]] = []
    for j in job_map.by_destination("vps", doc):
        if not j.get("schedule_jst"):
            continue
        if not j.get("enabled", True):
            continue
        if not j.get("ready"):
            skipped.append({"job": j["id"], "reason": "; ".join(j.get("blockers") or ["ready=false"])})
            continue
        if j["fetch_task"]["task"] not in owned:
            skipped.append({"job": j["id"], "reason": "実行役の担当ではない（fetch_owner.json の agent_tasks に無い。ホストの cron が実行する）"})
            continue
        for t in job_map.occurrences(j["schedule_jst"], now, end):
            kw: dict[str, Any] = {}
            if j.get("grace_seconds") is not None:
                kw["grace_seconds"] = int(j["grace_seconds"])
            if j.get("max_attempts") is not None:
                kw["max_attempts"] = int(j["max_attempts"])
            tasks.append(FT.make_task(j["fetch_task"]["task"], t, args=j["fetch_task"].get("args") or None,
                                      task_id=f"{j['id']}-{t.strftime('%Y%m%dT%H%M')}", **kw))
    return tasks, skipped


def shadow_tasks(real: list[FT.FetchTask], owned: set[str], env: dict[str, str] | None = None) -> list[FT.FetchTask]:
    """影モード（``KEIBA_FETCH_SHADOW_ENABLED=1`` のときだけ）で並走検証するタスク。無効なら空。"""
    from src.ops import fetch_shadow

    return fetch_shadow.shadow_tasks_from(real, fetch_shadow.load_config(), exclude=owned, env=env)


def enqueue_fetch(queue: FT.TaskQueue, tasks: list[FT.FetchTask]) -> dict[str, int]:
    res = {"enqueued": 0, "exists": 0, "error": 0}
    for t in tasks:
        try:
            res["enqueued" if queue.enqueue(t) else "exists"] += 1
        except Exception:  # noqa: BLE001
            res["error"] += 1
    return res


# ───────────────────────── Cloud Tasks → Run Job ─────────────────────────

def run_job_task_spec(*, project: str, region: str, run_job: str, args: list[str], schedule_time: datetime | None,
                      task_id: str, queue: str, service_account: str) -> dict[str, Any]:
    """Cloud Tasks に積む HTTP タスクの内容（クライアントに依存しない dict）。"""
    return {
        "queue": queue, "task_id": task_id, "schedule_time": schedule_time,
        "http_request": {
            "http_method": "POST",
            "url": RUN_API.format(project=project, region=region, job=run_job),
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"overrides": {"containerOverrides": [{"args": args}]}}).encode("utf-8"),
            "oauth_token": {"service_account_email": service_account, "scope": OAUTH_SCOPE},
        },
    }


def predict_run_job(doc: dict[str, Any] | None = None) -> str:
    """predict-race を動かす Run Job の名前（job_map の size から。例: keiba-dispatch-l）。"""
    d = doc or job_map.load()
    j = job_map.job_by_id("predict-race", d) or {}
    return d["runtime"]["run_jobs"][j.get("size", "l")]["name"]


def predict_args(race_id: str, date: str) -> list[str]:
    return ["-m", "src.ops.job_dispatch", "job", "predict-race", "--arg", f"race_id={race_id}", "--arg", f"date={date}"]


def predict_task_spec(race_id: str, date: str, run_at: datetime, *, suffix: str = "", env: dict[str, str] | None = None) -> dict[str, Any]:
    e = env if env is not None else os.environ
    project = (e.get("GCS_PROJECT_ID") or e.get("GCP_PROJECT_ID") or "").strip()
    region = (e.get("CLOUD_TASKS_LOCATION") or "asia-northeast1").strip()
    run_job = (e.get("KEIBA_DISPATCH_RUN_JOB") or "").strip() or predict_run_job()
    sa = (e.get("CLOUD_TASKS_OAUTH_SERVICE_ACCOUNT") or "").strip()
    if not (project and run_job and sa):
        raise ValueError("GCS_PROJECT_ID / CLOUD_TASKS_OAUTH_SERVICE_ACCOUNT が必要です（KEIBA_DISPATCH_RUN_JOB は省略時 job_map の predict-race のサイズ）")
    return run_job_task_spec(project=project, region=region, run_job=run_job, args=predict_args(race_id, date), schedule_time=run_at,
                             task_id=f"predict-{race_id}{suffix}", queue=(e.get("CLOUD_TASKS_QUEUE") or DEFAULT_QUEUE).strip(), service_account=sa)


def enqueue_run_job(spec: dict[str, Any], client: Any | None = None) -> str:
    """Cloud Tasks にタスクを作る。同名（AlreadyExists）は ``exists`` として扱う。戻り値: 'created' | 'exists'。"""
    from src.config.gcp_guard import assert_gcp_allowed

    assert_gcp_allowed("Cloud Tasks")
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import tasks_v2
    from google.protobuf import timestamp_pb2

    if client is None:
        from src.config.gcp_credentials import build_gcp_credentials

        client = tasks_v2.CloudTasksClient(credentials=build_gcp_credentials())
    e = os.environ
    project = (e.get("GCS_PROJECT_ID") or e.get("GCP_PROJECT_ID") or "").strip()
    region = (e.get("CLOUD_TASKS_LOCATION") or "asia-northeast1").strip()
    parent = client.queue_path(project, region, spec["queue"])
    hr = dict(spec["http_request"])
    hr["http_method"] = tasks_v2.HttpMethod.POST
    task: dict[str, Any] = {"http_request": hr, "name": client.task_path(project, region, spec["queue"], spec["task_id"])}
    if spec.get("schedule_time") is not None:
        ts = timestamp_pb2.Timestamp()
        when = spec["schedule_time"]
        ts.FromDatetime(when.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))
        task["schedule_time"] = ts
    try:
        client.create_task(request={"parent": parent, "task": task})
        return "created"
    except AlreadyExists:
        return "exists"


# ───────────────────────── 開催日 ─────────────────────────

def plan_race_day(rows: list[dict[str, Any]], date: str, now: datetime, *, late_snapshots: bool, predict_lead_minutes: int = PREDICT_LEAD_MINUTES,
                  build_events: Any = None) -> dict[str, Any]:
    """発走時刻表の行から、取得タスク（fetch_plan.plan_day）と予測タスクの計画を作る（副作用なし）。発走済みは除く。"""
    from src.ops import fetch_plan

    live = [r for r in rows if r["post_time"] > now]
    fetch = fetch_plan.plan_day(live, date, late=late_snapshots, weight=True, now=now, build_events=build_events)
    predicts = [{"race_id": str(r["race_id"]), "date": date, "run_at": max(r["post_time"] - timedelta(minutes=predict_lead_minutes), now + timedelta(seconds=10))}
                for r in live]
    return {"fetch": fetch, "predicts": predicts, "past": len(rows) - len(live), "races": len(rows)}


def reserve_race_day(date: str, storage: Any, queue: FT.TaskQueue, *, now: datetime | None = None, dry_run: bool = False,
                     enqueue_predict: Callable[[dict[str, Any]], str] | None = None, load_schedule: Callable[[Any, str], list[dict[str, Any]]] | None = None,
                     env: dict[str, str] | None = None, build_events: Any = None, owned: set[str] | None = None,
                     shadow_queue: FT.TaskQueue | None = None, predict_ready: bool = True) -> dict[str, Any]:
    """``predict_ready=False``（job_map の predict-race が ready=false）なら予測のタスクは積まない。積んでも起動した Job が
    『準備未完了』で何もせずに終わるだけで、開催日 1 日に約 36 回の起動が無駄になる。"""
    now = now or datetime.now(JST)
    e = env if env is not None else os.environ
    owned = owned_tasks(e) if owned is None else owned
    if load_schedule is None:
        from src.scraper.auto_scrape_queue import _fetch_race_schedule_storage as load_schedule
    rows = load_schedule(storage, date)
    late = e.get("KEIBA_ODDS_LATE_SNAPSHOTS", "") == "1"
    plan = plan_race_day(rows, date, now, late_snapshots=late, build_events=build_events)
    not_owned = sorted({t.task for t in plan["fetch"] if t.task not in owned})
    shadow_planned = shadow_tasks(plan["fetch"], owned, e)          # 影モード（ホストが取得を続けたまま並走検証）。担当済みは本番で実行するので除く
    plan["fetch"] = [t for t in plan["fetch"] if t.task in owned]
    out: dict[str, Any] = {"date": date, "races": plan["races"], "past": plan["past"], "fetch_tasks": len(plan["fetch"]), "predict_tasks": len(plan["predicts"]),
                           "late_snapshots": late, "dry_run": dry_run, "not_owned_tasks": not_owned,
                           "shadow_tasks": len(shadow_planned)}
    if dry_run:
        return out
    out["fetch"] = enqueue_fetch(queue, plan["fetch"])
    if shadow_queue is not None and shadow_planned:
        out["shadow"] = enqueue_fetch(shadow_queue, shadow_planned)
    pres: dict[str, Any] = {"created": 0, "exists": 0, "error": 0}
    if not predict_ready:
        out["predict"] = {**pres, "skipped": len(plan["predicts"]), "reason": "predict-race が ready=false"}
        return out
    for p in plan["predicts"]:
        try:
            spec = predict_task_spec(p["race_id"], p["date"], p["run_at"], env=e)
            pres[(enqueue_predict or enqueue_run_job)(spec)] += 1
        except Exception as exc:  # noqa: BLE001
            pres["error"] += 1
            out.setdefault("errors", []).append(f"{p['race_id']}: {type(exc).__name__}: {exc}")
    out["predict"] = pres
    return out


def reserve_recurring(queue: FT.TaskQueue, *, now: datetime | None = None, horizon_hours: float = 6, dry_run: bool = False,
                      doc: dict[str, Any] | None = None, owned: set[str] | None = None, shadow_queue: FT.TaskQueue | None = None,
                      env: dict[str, str] | None = None) -> dict[str, Any]:
    now = now or datetime.now(JST)
    doc = doc or job_map.load()
    owned_set = owned_tasks() if owned is None else owned
    tasks, skipped = recurring_tasks(doc, now, horizon_hours, owned_set)
    out: dict[str, Any] = {"horizon_hours": horizon_hours, "tasks": len(tasks), "skipped_jobs": skipped, "dry_run": dry_run}
    from src.ops import fetch_shadow

    shadow_planned: list[FT.FetchTask] = []
    if fetch_shadow.enabled(env):
        cands = set(fetch_shadow.load_config().task_names()) - set(owned_set)
        stasks, _ = recurring_tasks(doc, now, horizon_hours, cands)
        shadow_planned = shadow_tasks(stasks, set(owned_set), env)
    out["shadow_tasks"] = len(shadow_planned)
    if not dry_run:
        out["fetch"] = enqueue_fetch(queue, tasks)
        if shadow_queue is not None and shadow_planned:
            out["shadow"] = enqueue_fetch(shadow_queue, shadow_planned)
    return out


def _queue() -> FT.TaskQueue:
    return FT.TaskQueue(FT.GcsTaskStore(os.environ.get("GCS_BUCKET", "").strip()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("race-day")
    r.add_argument("--date", required=True)
    c = sub.add_parser("recurring")
    c.add_argument("--horizon-hours", type=float, default=6)
    for p in (r, c):
        p.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    queue = FT.TaskQueue(FT.MemoryTaskStore()) if ns.dry_run else _queue()
    from src.ops import fetch_shadow

    shadow_queue = FT.TaskQueue(fetch_shadow.PrefixedStore(queue.store)) if fetch_shadow.enabled() else None
    if ns.cmd == "race-day":
        from src.scraper.storage import HybridStorage

        from src.ops import job_map

        pr = next((j for j in job_map.load()["jobs"] if j["id"] == "predict-race"), {})
        res = reserve_race_day(ns.date, HybridStorage("."), queue, dry_run=ns.dry_run, shadow_queue=shadow_queue,
                               predict_ready=bool(pr.get("ready")))
    else:
        res = reserve_recurring(queue, horizon_hours=ns.horizon_hours, dry_run=ns.dry_run, shadow_queue=shadow_queue)
    print(json.dumps(res, ensure_ascii=False, default=str))
    bad = (res.get("fetch") or {}).get("error", 0) + (res.get("predict") or {}).get("error", 0) + (res.get("shadow") or {}).get("error", 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
