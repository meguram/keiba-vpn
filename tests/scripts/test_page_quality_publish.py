"""ページ品質チェックの要約（GCS へ上げる突き合わせ用）と VPS の取得タスク。"""

from __future__ import annotations

import json

from src.monitor import page_quality_check as pq
from src.ops import fetch_task as ft
from src.ops.gcs_io import MemoryIO
from src.scraper import auto_scrape


RESULT = {"summary_tag": "caution", "safe_count": 1, "caution_count": 1, "out_count": 0, "results": [
    {"category": "race_shutuba", "tag": "caution", "extra_keys": ["b", "a"], "elapsed_sec": 1.2, "validation_summary": {"top_missing": []}},
    {"category": "horse", "tag": "safe", "extra_keys": [], "elapsed_sec": 0.4, "validation_summary": {"top_missing": []}},
]}


def test_summary_drops_timings_and_is_stable():
    io = MemoryIO()
    assert pq.publish_summary(RESULT, io=io)
    doc = json.loads(io.get_bytes(pq.SUMMARY_KEY))
    assert doc == {"summary_tag": "caution", "results": [
        {"category": "horse", "tag": "safe", "extra_keys": [], "missing": []},
        {"category": "race_shutuba", "tag": "caution", "extra_keys": ["a", "b"], "missing": []}]}


def test_vps_task_is_registered_and_reports_out_as_error(monkeypatch):
    assert "page-quality-check" in ft.ALLOWED_TASKS and auto_scrape.TASKS["page-quality-check"] is auto_scrape.task_page_quality_check
    saved = {}
    monkeypatch.setattr(pq, "run_checks", lambda: {**RESULT, "summary_tag": "out", "results": [{"category": "odds", "tag": "out"}]})
    monkeypatch.setattr(pq, "save_result", lambda r: None)
    monkeypatch.setattr(pq, "publish_summary", lambda r: True)
    monkeypatch.setattr(auto_scrape, "_save_status", lambda task, r: saved.update({task: r}))
    auto_scrape.task_page_quality_check()
    assert saved["page-quality-check"]["status"] == "error" and "odds" in saved["page-quality-check"]["error"]
