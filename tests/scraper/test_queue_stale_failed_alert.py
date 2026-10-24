"""
check_stale_failed_jobs_and_notify / update_job_status の first_failed_at 連携の検証。

欠損データ（scraping job の失敗）が一定時間解消されない場合に Slack 通知するロジック
（docs/git_management/todo/scraping-queue.md 共通TODO 3番）のユニットテスト。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

os.environ.setdefault("GCS_BUCKET", "")
os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")


@pytest.fixture
def queue_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import src.scraper.job_queue as jq

    queue_file = tmp_path / "scrape_queue.json"
    alert_state = tmp_path / "queue_stale_failed_alert_state.json"
    monkeypatch.setattr(jq, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(jq, "QUEUE_STALE_FAILED_ALERT_STATE", alert_state)
    # job_queue.ScrapeJobQueue() は self.queue_file = QUEUE_FILE を __init__ で束縛するため
    # モジュール属性の monkeypatch だけで新規インスタンスにも反映される
    return jq, queue_file, alert_state


def _write_jobs(queue_file: Path, jobs: list[dict]) -> None:
    queue_file.write_text(
        json.dumps({"jobs": jobs, "updated_at": datetime.now().isoformat()}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_update_job_status_sets_first_failed_at_once(queue_paths):
    jq, queue_file, _ = queue_paths
    _write_jobs(queue_file, [{"job_id": "job-1", "status": "running"}])

    q = jq.ScrapeJobQueue()
    q.update_job_status("job-1", "failed", "boom")
    jobs = q.load_queue()
    first = jobs[0]["first_failed_at"]
    assert first

    # 2回目の失敗でも first_failed_at は上書きされない（最初の失敗時刻を保持する）
    jobs[0]["status"] = "running"
    _write_jobs(queue_file, jobs)
    q.update_job_status("job-1", "failed", "boom again")
    jobs2 = q.load_queue()
    assert jobs2[0]["first_failed_at"] == first


def test_update_job_status_completed_clears_first_failed_at(queue_paths):
    jq, queue_file, _ = queue_paths
    _write_jobs(
        queue_file,
        [{"job_id": "job-1", "status": "running", "first_failed_at": "2020-01-01T00:00:00"}],
    )
    q = jq.ScrapeJobQueue()
    q.update_job_status("job-1", "completed")
    jobs = q.load_queue()
    assert "first_failed_at" not in jobs[0]


def test_check_stale_failed_jobs_and_notify_alerts_past_threshold(queue_paths, monkeypatch):
    jq, queue_file, _ = queue_paths
    old = (datetime.now() - timedelta(hours=10)).isoformat()
    recent = (datetime.now() - timedelta(hours=1)).isoformat()
    _write_jobs(
        queue_file,
        [
            {
                "job_id": "stale-job",
                "status": "failed",
                "first_failed_at": old,
                "error": "timeout",
                "job_label": "20260101_01",
            },
            {
                "job_id": "fresh-job",
                "status": "failed",
                "first_failed_at": recent,
                "error": "timeout",
            },
            {
                "job_id": "pending-job",
                "status": "pending",
                "first_failed_at": old,
            },
        ],
    )

    calls = []
    monkeypatch.setattr(
        "src.utils.notify.notify_slack", lambda msg, **kw: calls.append(msg) or True
    )

    result = jq.check_stale_failed_jobs_and_notify(threshold_hours=6.0)
    assert result["ok"] is True
    assert result["alerted"] == 1
    assert result["alerted_job_ids"] == ["stale-job"]
    assert "stale-job" in result["stale_job_ids"]
    assert "fresh-job" not in result["stale_job_ids"]
    assert len(calls) == 1
    assert "stale-job" in calls[0]


def test_check_stale_failed_jobs_and_notify_does_not_renotify_within_window(queue_paths, monkeypatch):
    jq, queue_file, _ = queue_paths
    old = (datetime.now() - timedelta(hours=10)).isoformat()
    _write_jobs(
        queue_file,
        [{"job_id": "stale-job", "status": "failed", "first_failed_at": old, "error": "x"}],
    )

    calls = []
    monkeypatch.setattr(
        "src.utils.notify.notify_slack", lambda msg, **kw: calls.append(msg) or True
    )

    first = jq.check_stale_failed_jobs_and_notify(threshold_hours=6.0)
    assert first["alerted"] == 1

    second = jq.check_stale_failed_jobs_and_notify(threshold_hours=6.0)
    assert second["alerted"] == 0
    assert len(calls) == 1


def test_check_stale_failed_jobs_and_notify_disabled_when_threshold_zero(queue_paths, monkeypatch):
    jq, queue_file, _ = queue_paths
    old = (datetime.now() - timedelta(hours=100)).isoformat()
    _write_jobs(
        queue_file,
        [{"job_id": "stale-job", "status": "failed", "first_failed_at": old, "error": "x"}],
    )
    calls = []
    monkeypatch.setattr(
        "src.utils.notify.notify_slack", lambda msg, **kw: calls.append(msg) or True
    )

    result = jq.check_stale_failed_jobs_and_notify(threshold_hours=0)
    assert result.get("disabled") is True
    assert calls == []


if __name__ == "__main__":
    import unittest

    unittest.main()
