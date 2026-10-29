"""``config/gcp_job_map.json`` から、Cloud Run Jobs・Cloud Scheduler・Cloud Tasks キューを作る ``gcloud`` コマンドを生成する（表示するだけ。実行しない）。

    python -m src.ops.scheduler_gen queue|jobs|scheduler|all [--include-unready] [--parallel]
    環境変数: PROJECT_ID / REGION / GCS_BUCKET / IMAGE（省略時は placeholder。実行前にユーザーが実値を入れる）

- **Run Job は 3 つ**（サイズ s=512Mi・m=2Gi・l=4Gi）。全部同じイメージで、起動時の引数（``args``）で ``src.ops.job_dispatch`` のグループ／ジョブを選ぶ。
- **Scheduler はグループごとに 1 本**（``scheduler_groups``。6〜14 本に収める＝無料枠 3 本を超えた分は 1 本あたり約 15 円/月）。
  準備ができたジョブを 1 つも含まないグループは作らない（``--include-unready`` で全部）。
- 環境変数には必ず ``KEIBA_ENV=prod`` と ``KEIBA_RUN_LOCATION=gcp`` を入れる。通知の先頭・末尾に【PROD】/ 実行場所: GCP が付き、
  並走期間にホスト（【STG】/ ホストPC）の通知と見分けられる。``--parallel`` は ``KEIBA_PARALLEL_RUN=1``（影モード固定など）を足す。
- 作成は冪等（既にあれば update）。秘密は Secret Manager から渡す（値はここに出ない）。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any

from src.ops import job_map

SECRETS = ["SLACK_WEBHOOK_URL", "DEV_SECRET_KEY", "KEIBA_SHARED_LOGIN_ID"]      # 追加は SECRETS_EXTRA（例: SLACK_WEBHOOK_URL_ALERT）
# 有ると使う秘密（Secret Manager に無ければ渡さない。無い秘密を --set-secrets に書くと配備が失敗するため）。apply が存在を確かめて SECRETS_EXTRA に足す
OPTIONAL_SECRETS = ["SLACK_WEBHOOK_URL_PARALLEL"]      # 並走期間の prod 側の Slack（job_dispatch.parallel_wrap）
QUEUE = "keiba-predict"


def settings(env: dict[str, str] | None = None) -> dict[str, str]:
    e = os.environ if env is None else env
    project = e.get("PROJECT_ID", "YOUR_GCP_PROJECT_ID")
    region = e.get("REGION", "asia-northeast1")
    return {
        "project": project, "region": region, "bucket": e.get("GCS_BUCKET", "YOUR_BUCKET"),
        "image": e.get("IMAGE", f"{region}-docker.pkg.dev/{project}/keiba-vpn/jobs:latest"),
        "jobs_sa": e.get("RUN_SA_EMAIL", f"keiba-jobs@{project}.iam.gserviceaccount.com"),
        "sched_sa": e.get("SCHEDULER_SA_EMAIL", f"keiba-scheduler@{project}.iam.gserviceaccount.com"),
        "queue": e.get("CLOUD_TASKS_QUEUE", QUEUE),
        "secrets": ",".join(f"{n}={n}:latest" for n in SECRETS + [x for x in e.get("SECRETS_EXTRA", "").split(",") if x]),
        # 影モード（VPS の取得の並走検証）。1 のとき予約ジョブが影のタスクも積み、日次集計が合否を出す。VPS の実行役で影モードを始めてから 1 にする
        "fetch_shadow": "1" if e.get("FETCH_SHADOW", "").strip() == "1" else "0",
        # 課金エクスポートのテーブル（project.dataset.table）。課金アカウント ID を含むので公開リポジトリには置かず、変数で渡す
        "billing_table": e.get("GCP_BILLING_BQ_TABLE", "").strip(),
    }


def job_env(doc: dict[str, Any], st: dict[str, str], *, parallel: bool) -> dict[str, str]:
    env = dict(doc["runtime"]["env"])                      # KEIBA_ENV=prod, KEIBA_RUN_LOCATION=gcp, TZ
    env.update({"GCS_BUCKET": st["bucket"], "GCS_PROJECT_ID": st["project"], "KEIBA_SLACK_NOTIFY": "1",
                "CLOUD_TASKS_QUEUE": st["queue"], "CLOUD_TASKS_LOCATION": st["region"], "CLOUD_TASKS_OAUTH_SERVICE_ACCOUNT": st["sched_sa"],
                "KEIBA_DISPATCH_RUN_JOB": doc["runtime"]["run_jobs"]["l"]["name"]})
    if parallel:
        env["KEIBA_PARALLEL_RUN"] = "1"
    if st.get("fetch_shadow") == "1":
        env["KEIBA_FETCH_SHADOW_ENABLED"] = "1"
    if st.get("billing_table"):
        env["GCP_BILLING_BQ_TABLE"] = st["billing_table"]
    return env


def _kv(d: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in d.items())


def queue_commands(st: dict[str, str]) -> list[str]:
    return [
        f"gcloud tasks queues describe {st['queue']} --project {st['project']} --location {st['region']} >/dev/null 2>&1 || "
        f"gcloud tasks queues create {st['queue']} --project {st['project']} --location {st['region']} "
        f"--max-attempts 3 --min-backoff 10s --max-backoff 60s --max-dispatches-per-second 5 --max-concurrent-dispatches 3",
    ]


def run_job_commands(doc: dict[str, Any], st: dict[str, str], *, parallel: bool) -> list[str]:
    out = []
    env = _kv(job_env(doc, st, parallel=parallel))
    for size, rj in doc["runtime"]["run_jobs"].items():
        timeout = max([g["timeout_sec"] for g in doc["scheduler_groups"].values() if g["size"] == size] + [j.get("timeout_sec", 600) for j in doc["jobs"] if j.get("size") == size] + [600])
        out.append(
            f"gcloud run jobs deploy {rj['name']} --project {st['project']} --region {st['region']} --image {shlex.quote(st['image'])} "
            f"--memory {rj['memory']} --cpu {rj['cpu']} --task-timeout {timeout}s --max-retries 0 --parallelism 1 --tasks 1 "
            f"--service-account {st['jobs_sa']} --set-env-vars {shlex.quote(env)} --set-secrets {shlex.quote(st['secrets'])} "
            f"--command python3 --args=-m,src.ops.job_dispatch,list"
        )
    return out


def group_is_ready(doc: dict[str, Any], gid: str) -> bool:
    ids = doc["scheduler_groups"][gid]["jobs"]
    return any((j := job_map.job_by_id(i, doc)) and j.get("ready") and j.get("enabled", True) for i in ids)


def scheduler_commands(doc: dict[str, Any], st: dict[str, str], *, include_unready: bool = False) -> list[str]:
    out = []
    for gid, g in doc["scheduler_groups"].items():
        if not include_unready and not group_is_ready(doc, gid):
            out.append(f"# {gid}: 準備できたジョブが無いので作らない（{', '.join(g['jobs'])}）")
            continue
        rj = doc["runtime"]["run_jobs"][g["size"]]["name"]
        name = f"keiba-{gid}"
        uri = f"https://run.googleapis.com/v2/projects/{st['project']}/locations/{st['region']}/jobs/{rj}:run"
        body = json.dumps({"overrides": {"containerOverrides": [{"args": ["-m", "src.ops.job_dispatch", "group", gid]}]}}, separators=(",", ":"))
        common = (f"--project {st['project']} --location {st['region']} --schedule {shlex.quote(g['cron_jst'])} --time-zone Asia/Tokyo "
                  f"--uri {shlex.quote(uri)} --http-method POST --message-body {shlex.quote(body)} "
                  f"--oauth-service-account-email {st['sched_sa']} --oauth-token-scope https://www.googleapis.com/auth/cloud-platform "
                  f"--attempt-deadline 180s --description {shlex.quote(g['desc'][:200])}")
        out.append(f"# {gid}: {g['cron_jst']} JST → {rj} → {', '.join(g['jobs'])}")
        out.append(f"if gcloud scheduler jobs describe {name} --project {st['project']} --location {st['region']} >/dev/null 2>&1; then\n"
                   f"  gcloud scheduler jobs update http {name} --update-headers Content-Type=application/json {common}\n"
                   f"else\n  gcloud scheduler jobs create http {name} --headers Content-Type=application/json {common}\nfi")
    return out


def render(what: str, doc: dict[str, Any] | None = None, *, env: dict[str, str] | None = None, include_unready: bool = False, parallel: bool = False) -> str:
    doc = doc or job_map.load()
    st = settings(env)
    parts: list[str] = ["#!/usr/bin/env bash", "# 生成: python -m src.ops.scheduler_gen（config/gcp_job_map.json から）。内容を確認してから実行する。", "set -euo pipefail", ""]
    if what in ("queue", "all"):
        parts += ["# ── Cloud Tasks キュー（開催日の予測の予約）──", *queue_commands(st), ""]
    if what in ("jobs", "all"):
        parts += ["# ── Cloud Run Jobs（サイズ別の 3 つ。同じイメージ）──", *run_job_commands(doc, st, parallel=parallel), ""]
    if what in ("scheduler", "all"):
        parts += ["# ── Cloud Scheduler（グループごと）──", *scheduler_commands(doc, st, include_unready=include_unready), ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["queue", "jobs", "scheduler", "all"])
    ap.add_argument("--include-unready", action="store_true")
    ap.add_argument("--parallel", action="store_true", help="並走期間: KEIBA_PARALLEL_RUN=1 を環境変数に入れる")
    ns = ap.parse_args(argv)
    errs = job_map.validate(job_map.load())
    if errs:
        print("job_map の検証に失敗:\n" + "\n".join(errs), file=sys.stderr)
        return 3
    print(render(ns.what, include_unready=ns.include_unready, parallel=ns.parallel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
