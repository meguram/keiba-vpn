"""Scheduler/Run Job の gcloud コマンド生成、前夜の取得／計算の分離（raceday-eve-fetch / eve-precompute）、予測ジョブの薄いラッパー、異常通知。"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from src.ops import alerting, job_map, scheduler_gen as SG
from src.ops.gcs_io import MemoryIO, read_json
from src.ops.jobs import eve_precompute as EP, predict_race as PR

DOC = job_map.load()
ENV = {"PROJECT_ID": "proj", "GCS_BUCKET": "bkt", "REGION": "asia-northeast1"}


def test_generated_script_is_valid_bash_and_idempotent():
    text = SG.render("all", DOC, env=ENV, include_unready=True)
    p = Path("/tmp/_gen_test.sh")
    p.write_text(text)
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    assert text.count("gcloud run jobs deploy") == 3                      # s / m / l の 3 つだけ
    assert text.count("gcloud scheduler jobs update http") == len(DOC["scheduler_groups"])
    assert "scheduler jobs describe" in text and "queues describe" in text


def test_env_marks_prod_gcp_for_distinguishable_notifications():
    text = SG.render("jobs", DOC, env=ENV)
    assert text.count("KEIBA_ENV=prod,KEIBA_RUN_LOCATION=gcp") == 3
    assert "KEIBA_PARALLEL_RUN" not in text
    assert text.count("KEIBA_PARALLEL_RUN=1") == 0
    assert SG.render("jobs", DOC, env=ENV, parallel=True).count("KEIBA_PARALLEL_RUN=1") == 3


def test_scheduler_targets_run_api_with_group_args_and_service_account():
    text = SG.render("scheduler", DOC, env=ENV, include_unready=True)
    assert "https://run.googleapis.com/v2/projects/proj/locations/asia-northeast1/jobs/keiba-dispatch-l:run" in text
    assert "--oauth-service-account-email keiba-scheduler@proj.iam.gserviceaccount.com" in text
    bodies = re.findall(r"--message-body (\S+)", text)
    assert len(bodies) == 2 * len(DOC["scheduler_groups"])                # update と create の両方
    args = json.loads(shlex.split(bodies[0])[0])["overrides"]["containerOverrides"][0]["args"]
    assert args[:3] == ["-m", "src.ops.job_dispatch", "group"] and args[3] in DOC["scheduler_groups"]
    assert "--time-zone Asia/Tokyo" in text


def test_unready_groups_are_not_created_by_default():
    text = SG.render("scheduler", DOC, env=ENV)
    for gid in DOC["scheduler_groups"]:
        ready = SG.group_is_ready(DOC, gid)
        assert (f"keiba-{gid} " in text) == ready, gid
    assert "準備できたジョブが無いので作らない" in text


def test_run_job_memory_matches_sizes_and_secrets_not_inlined():
    text = SG.render("jobs", DOC, env=ENV)
    assert "--memory 512Mi" in text and "--memory 2Gi" in text and "--memory 8Gi" in text          # l は 8Gi（predict-race の実測ピーク 4.3GB のため）
    assert "--set-secrets SLACK_WEBHOOK_URL=SLACK_WEBHOOK_URL:latest" in text and "hooks.slack.com" not in text


def test_dockerfile_and_deploy_script_are_consistent():
    df = Path("deploy/docker/Dockerfile.jobs").read_text()
    assert 'ENTRYPOINT ["python3"]' in df and "src.ops.job_dispatch" in df and "KEIBA_ENV=prod" in df and "KEIBA_RUN_LOCATION=gcp" in df
    assert "playwright" in df and "mlflow" in df                         # ジョブに要らない依存を外している
    assert "members.json" in Path("deploy/docker/Dockerfile.jobs.dockerignore").read_text()
    sh = Path("scripts/gcp/deploy_cloud_run_jobs.sh").read_text()
    assert "--dry-run" in sh and "KEIBA_RUN_LOCATION=gcp" in sh
    r = subprocess.run(["bash", "scripts/gcp/deploy_cloud_run_jobs.sh", "show", "scheduler"], capture_output=True, text=True)
    assert r.returncode == 0 and "gcloud scheduler" in r.stdout


# ── 前夜の分離 ──

def test_raceday_eve_fetch_task_registered_and_allowlisted():
    from src.ops import fetch_task as FT

    src = Path("src/scraper/auto_scrape.py").read_text(encoding="utf-8")
    assert '"raceday-eve-fetch":      task_raceday_eve_fetch' in src
    assert "raceday-eve-fetch" in FT.AUTO_SCRAPE_MODES
    assert "KEIBA_EVE_FETCH_ONLY" in src and "KEIBA_EVE_FETCH_ONLY" in Path("src/scraper/auto_scrape_queue.py").read_text(encoding="utf-8")


def test_eve_precompute_runs_all_precomputes_for_race_ids():
    class St:
        def load(self, cat, key):
            assert (cat, key) == ("race_lists", "20261024")
            return {"races": [{"race_id": "202605040101"}, {"race_id": "202605040102"}, {"x": 1}]}

    seen = []

    def pre(date, ids):
        seen.append((date, ids))
        return {"ok": 2, "skip": 0, "fail": 0, "total": 2, "tracking_v2": {"ok": 2, "fail": 0}}, {"ok": 2, "skip": 0, "fail": 0, "total": 2}
    res = EP.run("20261024", St(), pre)
    assert res["status"] == "ok" and seen == [("20261024", ["202605040101", "202605040102"])]
    bad = EP.run("20261024", St(), lambda d, i: ({"ok": 1, "fail": 1}, {"ok": 2, "fail": 0}))
    assert bad["status"] == "error"
    assert EP.run("20261024", type("S", (), {"load": lambda s, c, k: None})(), pre)["status"] == "skipped"


def test_predict_race_wrapper_picks_row_and_uses_run_for_races():
    rows = [{"race_id": "202605040101", "post_time": "x"}, {"race_id": "202605040102"}]
    assert PR.pick_race(rows, "202605040102") == {"race_id": "202605040102"}
    assert PR.pick_race(rows, "202609999999") == {"race_id": "202609999999"}
    got = {}

    def fake_run(races, storage, ctx, weights_known=None):
        got.update(races=races, wk=weights_known)
        return [{"race_id": races[0]["race_id"], "status": "ok"}]
    import types
    import src.pipeline.inference.race_day_workflow as wf

    orig = wf.load_day_schedule
    wf.load_day_schedule = lambda st, d: rows
    try:
        ev = PR.run("202605040101", "20261024", storage=object(), ctx=object(), run_for_races=fake_run)
    finally:
        wf.load_day_schedule = orig
    assert ev[0]["status"] == "ok" and got["races"] == [rows[0]] and got["wk"] == {"202605040101": True}


def test_predict_race_cli_rejects_bad_race_id():
    assert PR.main(["--race-id", "123", "--date", "20261024"]) == 2


# ── 異常通知 ──

def test_alerting_records_and_sends_through_send_payload(monkeypatch):
    io, sent = MemoryIO(), []
    monkeypatch.setenv("KEIBA_SLACK_NOTIFY", "1")
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime(2026, 10, 24, 12, 0, 5, tzinfo=ZoneInfo("Asia/Tokyo"))
    res = alerting.notify("ジョブが失敗: a", ["rc=1"], severity="critical", job="a", io=io, now=now, send=lambda payload, channel: sent.append((payload, channel)) or {"ok": True})
    assert res["recorded"] and sent[0][1] == "alert" and "ジョブが失敗" in sent[0][0]["text"]
    rec = read_json(io, "ops/alerts/20261024/120005-a-critical.json")
    assert rec["job"] == "a" and rec["severity"] == "critical" and rec["slack_ok"] is True
    monkeypatch.delenv("KEIBA_SLACK_NOTIFY")
    sent.clear()
    alerting.notify("x", [], severity="warning", job="a", io=io, now=now, send=lambda *a, **k: sent.append(1))
    assert sent == []                                                       # KEIBA_SLACK_NOTIFY!=1 では送らない（記録は残る）


def test_no_direct_slack_calls_in_new_code():
    # Slack は必ず slack_notify.send_payload 経由（環境名・実行場所が付く）。webhook を直接叩く実装を入れない
    for f in list(Path("src/ops").rglob("*.py")) + [Path("src/scripts/ops/parity_check.py"), Path("src/scripts/ops/cron_cutover_status.py")]:
        t = f.read_text(encoding="utf-8")
        assert "hooks.slack.com" not in t and "requests.post" not in t, f
