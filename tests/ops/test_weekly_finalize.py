"""週次の後処理の受け渡し（VPS の weekly-update → 目印 → ホストの weekly-finalize）。docs/operations/vps_only_fetch.md の段 4 後半。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.ops import gcs_mirror as GM
from src.ops import host_postprocess as HP
from src.ops import weekly_finalize as WF

JST = timezone(timedelta(hours=9))
SAT = datetime(2026, 11, 14, 1, 0, tzinfo=JST)


class MemBackend:
    def __init__(self):
        self.objs: dict[str, bytes] = {}
        self.downloads: list[str] = []

    def list(self, prefix):  # 一覧はしない（呼ばれたら失敗）
        raise AssertionError("list は呼ばない")

    def download(self, key):
        self.downloads.append(key)
        if key not in self.objs:
            raise KeyError(key)
        return self.objs[key]

    def upload(self, key, data):
        self.objs[key] = data


def test_no_handoff_does_nothing(tmp_path):
    be, calls = MemBackend(), []
    res = WF.run(backend=be, finalize=lambda d, r: calls.append(d) or {}, now=SAT, done_file=tmp_path / "d.json")
    assert res == {"status": "skipped", "reason": "no-handoff"} and calls == []
    assert be.downloads == [GM.WEEKLY_HANDOFF_KEY]                    # 名前指定で 1 件だけ読む


def test_runs_finalize_once_per_handoff(tmp_path):
    be, calls = MemBackend(), []
    GM.publish_weekly_handoff(be, "20261113", ["2026-11-08", "2026-11-09"], now=SAT.timestamp() - 3600)
    fin = lambda d, r: calls.append(d) or {"status": "ok"}  # noqa: E731
    done = tmp_path / "d.json"
    assert WF.run(backend=be, finalize=fin, now=SAT, done_file=done)["status"] == "ok"
    assert calls == [["2026-11-08", "2026-11-09"]]
    assert WF.run(backend=be, finalize=fin, now=SAT + timedelta(hours=1), done_file=done)["reason"] == "already-done"
    assert len(calls) == 1
    # 同じ日に週次を回し直した（目印を書き直した）ら別の回として動く
    GM.publish_weekly_handoff(be, "20261113", ["2026-11-08", "2026-11-09"], now=SAT.timestamp() - 60)
    assert WF.run(backend=be, finalize=fin, now=SAT + timedelta(hours=2), done_file=done)["status"] == "ok"
    assert len(calls) == 2


def test_partial_or_exception_is_retried_up_to_max_tries(tmp_path):
    be, calls = MemBackend(), []
    GM.publish_weekly_handoff(be, "20261113", ["2026-11-08"], now=SAT.timestamp() - 3600)
    done = tmp_path / "d.json"

    def fin(d, r):
        calls.append(d)
        if len(calls) == 1:
            return {"status": "partial"}
        raise RuntimeError("PG down")

    assert WF.run(backend=be, finalize=fin, now=SAT, done_file=done)["status"] == "partial"
    assert WF.run(backend=be, finalize=fin, now=SAT, done_file=done)["status"] == "error"
    assert WF.run(backend=be, finalize=fin, now=SAT, done_file=done)["status"] == "error"
    res = WF.run(backend=be, finalize=fin, now=SAT, done_file=done)
    assert res["status"] == "error" and res["reason"].startswith("gave-up-after-3") and len(calls) == 3
    assert WF.run(backend=be, finalize=fin, now=SAT, done_file=done)["reason"] == "gave-up"      # 知らせは 1 回だけ


def test_main_exit_code_marks_partial_as_failure(monkeypatch):
    monkeypatch.setattr(WF, "run", lambda **k: {"status": "partial"})
    monkeypatch.setattr("src.utils.project_env.load_project_dotenv", lambda *a, **k: None)
    assert WF.main([]) == 1
    monkeypatch.setattr(WF, "run", lambda **k: {"status": "skipped"})
    assert WF.main([]) == 0


def test_stale_handoff_is_ignored(tmp_path):
    be, calls = MemBackend(), []
    GM.publish_weekly_handoff(be, "20261106", ["2026-11-01"], now=SAT.timestamp() - 8 * 86400)   # 前の週の目印
    res = WF.run(backend=be, finalize=lambda d, r: calls.append(d) or {}, now=SAT, done_file=tmp_path / "d.json")
    assert res["reason"] == "stale-handoff" and calls == []


def test_dry_run_does_not_finalize(tmp_path):
    be, calls = MemBackend(), []
    GM.publish_weekly_handoff(be, "20261113", ["2026-11-08"], now=SAT.timestamp())
    res = WF.run(backend=be, finalize=lambda d, r: calls.append(d) or {}, now=SAT, done_file=tmp_path / "d.json", dry_run=True)
    assert res["status"] == "dry-run" and calls == [] and not (tmp_path / "d.json").exists()


def test_auth_errors_are_raised():
    class Boom(MemBackend):
        def download(self, key):
            raise PermissionError("403")

    with pytest.raises(PermissionError):
        GM.read_weekly_handoff(Boom())


def test_host_job_runs_only_when_weekly_update_moved(monkeypatch, tmp_path):
    monkeypatch.setenv("KEIBA_HOST_POSTPROCESS_LOCK_DIR", str(tmp_path))
    calls = []
    run = lambda a, e: calls.append(a) or 0  # noqa: E731
    assert HP.run("weekly-finalize", owned=set(), now=SAT, runner=run) == 0 and calls == []
    assert HP.run("weekly-finalize", owned={"weekly-update"}, now=SAT, runner=run) == 0
    assert calls == [["-m", "src.ops.weekly_finalize"]]


def test_vps_finalize_writes_handoff_and_skips_postprocessing(monkeypatch):
    """VPS では finalize_weekly_update がめぐ指数などを動かさず、目印だけを書く。"""
    from src.scraper import auto_scrape as AS

    be = MemBackend()
    monkeypatch.setenv("KEIBA_RUN_LOCATION", "vps")
    monkeypatch.setattr(GM, "GcsBackend", lambda: be)
    monkeypatch.setattr(AS, "_trigger_megu_batch_for_dates", lambda *a, **k: pytest.fail("VPS でめぐ指数を動かした"))
    monkeypatch.setattr(AS, "_trigger_pg_sync_for_dates", lambda *a, **k: pytest.fail("VPS で PG 同期を動かした"))
    out = AS.finalize_weekly_update(["2026-11-08"], {"status": "ok"})
    assert out["post_processing"] == "host" and out["handoff"] == "ok" and out["status"] == "ok"
    doc = json.loads(be.objs[GM.WEEKLY_HANDOFF_KEY])
    assert doc["target_dates"] == ["2026-11-08"] and len(doc["run_date"]) == 8 and doc["written_at"] > 0


def test_vps_finalize_marks_error_when_handoff_fails(monkeypatch):
    from src.scraper import auto_scrape as AS

    class Boom(MemBackend):
        def upload(self, key, data):
            raise PermissionError("403")

    monkeypatch.setenv("KEIBA_RUN_LOCATION", "vps")
    monkeypatch.setattr(GM, "GcsBackend", lambda: Boom())
    out = AS.finalize_weekly_update(["2026-11-08"], {"status": "ok"})
    assert out["status"] == "error" and out["handoff"].startswith("error:")      # fetch_child がタスクを失敗にする
