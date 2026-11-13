"""VPS の中だけで取得する（2026-11）: ローカルのタスク表（FileTaskStore）・ストアの切り替え・予約の --no-predict・VPS のタイマー。"""

from __future__ import annotations

import configparser
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops import fetch_task as FT, reservations as R

JST = ZoneInfo("Asia/Tokyo")
T = Path("scripts/ops/vps/templates")


def test_file_store_create_is_exclusive_and_lists_by_prefix(tmp_path):
    s = FT.FileTaskStore(tmp_path)
    assert s.create("ops/fetch_tasks/pending/a.json", "A")
    assert not s.create("ops/fetch_tasks/pending/a.json", "B")               # 既にあれば作らない
    assert s.get("ops/fetch_tasks/pending/a.json")[0] == "A"
    assert s.create("ops/fetch_tasks/done/b.json", "x") and s.create("ops/events/w/1.json", "e")
    assert s.list("ops/fetch_tasks/pending/") == ["ops/fetch_tasks/pending/a.json"]
    assert s.list("ops/fetch_tasks/") == ["ops/fetch_tasks/done/b.json", "ops/fetch_tasks/pending/a.json"]
    assert s.list("ops/fetch_tasks/pend") == ["ops/fetch_tasks/pending/a.json"]
    assert s.list("nothing/") == []
    assert s.delete("ops/fetch_tasks/pending/a.json") and not s.delete("ops/fetch_tasks/pending/a.json")
    assert s.get("ops/fetch_tasks/pending/a.json") is None
    assert not list(tmp_path.rglob(".*.tmp"))                                  # 一時ファイルが残らない
    assert s.list("") == ["ops/events/w/1.json", "ops/fetch_tasks/done/b.json"] and s.list("op") == s.list("")
    with pytest.raises(FT.TaskError):
        s.get("../etc/passwd")


def test_file_store_create_race_has_one_winner(tmp_path):
    s, wins = FT.FileTaskStore(tmp_path), []
    ths = [threading.Thread(target=lambda i=i: wins.append(s.create("k/x.json", str(i)))) for i in range(16)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert wins.count(True) == 1


def test_queue_flow_on_file_store(tmp_path):
    q = FT.TaskQueue(FT.FileTaskStore(tmp_path))
    now = datetime(2026, 11, 14, 9, 0, tzinfo=JST)
    t = FT.make_task("entries-sync", now)
    assert q.enqueue(t) and not q.enqueue(t)
    due = q.due_pending(now)
    assert [x.task_id for x in due] == [t.task_id]
    c = q.claim(t.task_id, "vps-1", now)
    assert c is not None and q.claim(t.task_id, "vps-2", now) is None
    q.complete(c, {"rc": 0}, now)
    assert q.exists_anywhere(t.task_id) == "done" and q.due_pending(now) == []


def test_broken_task_file_does_not_stop_the_queue(tmp_path):
    s = FT.FileTaskStore(tmp_path)
    s.create("ops/fetch_tasks/pending/broken.json", "")                       # 電源断などで中身の無いファイル
    q = FT.TaskQueue(s)
    assert q.load("pending", "broken") is None and q.due_pending() == []


def test_task_store_from_env(tmp_path):
    s = FT.task_store_from_env({"KEIBA_FETCH_TASK_STORE": "file", "KEIBA_FETCH_TASK_DIR": str(tmp_path / "t")})
    assert isinstance(s, FT.FileTaskStore) and (tmp_path / "t").is_dir()
    with pytest.raises(FT.TaskError):
        FT.task_store_from_env({"KEIBA_FETCH_TASK_STORE": "s3"})


def test_resolve_date():
    now = datetime(2026, 11, 13, 20, 30, tzinfo=JST)
    assert R.resolve_date("today", now) == "20261113" and R.resolve_date("tomorrow", now) == "20261114"
    assert R.resolve_date("20261115", now) == "20261115"


def test_race_day_no_predict_reserves_only_fetch(tmp_path, monkeypatch):
    """--no-predict（VPS の中で動かすとき）: 取得のタスクだけ積み、予測の Cloud Tasks は積まない。"""
    q = FT.TaskQueue(FT.FileTaskStore(tmp_path))
    rows = [{"race_id": "202605050101", "post_time": datetime(2026, 11, 14, 10, 5, tzinfo=JST)}]
    called = []
    res = R.reserve_race_day("20261114", None, q, now=datetime(2026, 11, 13, 20, 30, tzinfo=JST), load_schedule=lambda s, d: rows,
                             enqueue_predict=called.append, env={}, owned={"race-odds-t60", "race-odds-t15", "race-weight-check", "race-result-t15"},
                             predict_ready=False)
    assert res["fetch"]["enqueued"] > 0 and called == [] and res["predict"]["skipped"] == 1
    assert q.list_ids("pending")


def _unit(name):
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(T / name, encoding="utf-8")
    return cp


def test_vps_reserve_units_use_local_store_and_fit_in_slice():
    for name in ("keiba-fetch-reserve.service", "keiba-fetch-raceday@.service"):
        sv = _unit(name)["Service"]
        # EnvironmentFile（fetch.env）の値は Environment= より優先されるので、ExecStart で file を強制する
        assert sv["ExecStart"].startswith("/usr/bin/env KEIBA_FETCH_TASK_STORE=file ") and sv["User"] == "keibafetch" and sv["Slice"] == "keiba-fetch.slice"
        assert sv["ProtectSystem"] == "strict" and sv["ReadWritePaths"] == "/var/lib/keiba-fetch"
        assert int(sv["MemoryMax"].rstrip("M")) <= 120                     # 取得役（約 90MB）と重なっても slice の 250M に収まる
    assert "--no-predict" in _unit("keiba-fetch-raceday@.service")["Service"]["ExecStart"]
    assert _unit("keiba-fetch-raceday-tomorrow.timer")["Timer"]["Unit"] == "keiba-fetch-raceday@tomorrow.service"
    assert _unit("keiba-fetch-raceday-tomorrow.timer")["Timer"]["Persistent"] == "false"     # 0 時を過ぎた取り戻しで翌々日を予約しない
    assert _unit("keiba-fetch-raceday-today.timer")["Timer"]["Unit"] == "keiba-fetch-raceday@today.service"
    assert _unit("keiba-fetch-reserve.timer")["Timer"]["OnCalendar"] == "*-*-* *:50:00"


def test_agent_with_file_store_never_touches_gcs_even_in_shadow_mode(tmp_path, monkeypatch, capsys):
    """2026-11-13: 担当が空でも、影モードの実行役が GCS のタスク表を毎分 6 回一覧していた（月 約 210 円）。
    タスク表をローカル（file）にすれば、影モードでも GCS のタスク表には一切触れない。"""
    from src.ops import fetch_agent as FA

    def no_gcs(*a, **k):
        raise AssertionError("file モードで GCS のタスク表を作った")

    monkeypatch.setattr(FT, "GcsTaskStore", no_gcs)
    for k, v in {"KEIBA_FETCH_TASK_STORE": "file", "KEIBA_FETCH_TASK_DIR": str(tmp_path / "tasks"), "KEIBA_FETCH_AGENT_ENABLED": "1",
                 "KEIBA_FETCH_AGENT_TASKS": "entries-sync", "KEIBA_FETCH_SHADOW_ENABLED": "1",
                 "KEIBA_FETCH_SHADOW_STATE_DIR": str(tmp_path / "shadow")}.items():
        monkeypatch.setenv(k, v)
    assert FA.main(["--dry-run"]) == 0
    assert (tmp_path / "tasks").is_dir()


def test_task_store_from_env_refuses_gcs_on_vps(monkeypatch):
    """取得役・予約・計画のどの入口でも、VPS では GCS のタスク表を作らない（2026-11-13、月 約 210 円の一覧）。"""
    monkeypatch.setattr(FT, "GcsTaskStore", lambda *a, **k: pytest.fail("GCS のタスク表を作った"))
    for env in ({"KEIBA_RUN_LOCATION": "vps"}, {"KEIBA_RUN_LOCATION": "VPS", "KEIBA_FETCH_TASK_STORE": "gcs"}):
        with pytest.raises(FT.TaskError):
            FT.task_store_from_env(env)
