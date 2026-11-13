"""取得の一覧（fetch_registry）: 心拍ファイルの読み書きと、取得の入口の分類の強制。"""

import json
import os
import re
from pathlib import Path

import pytest

from src.scraper import backfill_guard as bg
from src.scraper import fetch_registry as fr

ROOT = Path(__file__).resolve().parents[2]

# sh から起動する -m のモジュールのうち、外部からデータを取得しないもの。
# 新しい sh から新しいモジュールを起動するときは、取得なら fr.FETCH_MODULE_PATTERN に、取得でなければここに足す
NON_FETCH_MODULES = {
    "src.api.flask_app", "src.db.batch.stats_snapshot", "src.monitor.app", "src.ops.fetch_owner", "src.ops.gcs_mirror",
    "src.ops.cost_guard", "src.ops.cron_catchup", "src.ops.vps_alive", "src.ops.host_postprocess", "src.ops.job_owner", "src.ops.model_publish", "src.ops.release_manifest", "src.ops.release_publish", "src.ops.scheduler_gen",
    "src.ops.state_sync", "src.pipeline.betting.odds_drift", "src.pipeline.build_jockey_trainer_stats",
    "src.research.pedigree.build_ancestor_name_map", "src.research.pedigree.build_horse_pedigree_10gen",
    "src.research.pedigree.build_pedigree_10gen_3view_index", "src.research.pedigree.build_pedigree_10gen_index",
    "src.scraper.export_tables", "src.scraper.parse_audit", "src.scraper.parse_drift", "src.scraper.parse_drift_docs",
    "src.scripts.data.build_db_coverage_cache", "src.scripts.data.build_ml_warehouse", "src.scripts.data.build_px_state",
    "src.scripts.data.build_rb_state", "src.scripts.data.daily_results_notify", "src.scripts.data.dev_fixture",
    "src.scripts.data.drift_dump_features", "src.scripts.data.drift_monitor", "src.scripts.data.etl_stg_db",
    "src.scripts.data.update_run_axes",   # GCS の horse_lap を名前で読むだけ（netkeiba へは取りに行かない）
    "src.scripts.data.update_serving_state",   # 保存済みの出馬表・成績を読んで v4 の日次状態を作るだけ（netkeiba へは取りに行かない）
    "src.scripts.data.verify_past_values",   # 保存済みの race_result を名前で読んで値を検査するだけ（netkeiba へは取りに行かない）
    "src.scripts.data.frozen_tracking", "src.scripts.data.make_dev_mock", "src.scripts.data.odds_sec77_backfill",
    "src.scripts.data.precompute_race_analysis", "src.scripts.data.prediction_results", "src.scripts.data.pre_odds_table",
    "src.scripts.data.scrape_status_notify", "src.scripts.data.sync_horse_result_cache", "src.scripts.data.tp55_backfill",
    "src.scripts.data.tp65_prep", "src.scripts.data.tracking_v2_eval", "src.scripts.data.verify_past_backfill",
    "src.scripts.ops.cloudrun_smoke", "src.scripts.ops.collect_prod_status", "src.scripts.ops.env_parity_check",
    "src.scripts.ops.maintenance_notify", "src.scripts.ops.shared_login", "src.scripts.scraping.fetch_status",
    "src.scripts.scraping.migrate_row_data_to_unique_paths",
}


def _sh_modules() -> set[str]:
    mods = set()
    for sh in (ROOT / "scripts").rglob("*.sh"):
        body = fr._strip_comments(sh.read_text(encoding="utf-8", errors="replace"))
        mods |= set(re.findall(r"-m\s+(src\.[A-Za-z0-9_.]+)", body))
    return mods


def test_every_module_launched_from_scripts_is_classified():
    """sh から起動するモジュールは、取得（表示の対象）か取得でない（NON_FETCH_MODULES）のどちらかに分類されていること。
    新しい取得を足したのに fetch_status.sh に出ない、を防ぐ。"""
    mods = _sh_modules()
    assert "src.scripts.scraping.backfill_horse_training_past" in mods      # 抜き出しが動いていること
    unknown = sorted(m for m in mods if not fr.is_fetch_args(f"python -m {m}") and m not in NON_FETCH_MODULES)
    assert not unknown, f"取得なら fetch_registry.FETCH_MODULE_PATTERN に、取得でなければ NON_FETCH_MODULES に足す: {unknown}"
    wrong = sorted(m for m in NON_FETCH_MODULES & mods if fr.is_fetch_args(f"python -m {m}"))
    assert not wrong, f"取得でないのに取得の入口に当たる: {wrong}"


@pytest.mark.parametrize("args", [
    "/opt/conda/bin/python3 -m src.scraper.backfill --phase past",
    "python -m src.scraper.auto_scrape --task raceday-evening",
    "python -m src.scripts.scraping.backfill_horse_training_past --redo-truncated",
    "python -m src.scraper.course_position --from 2016",
    "python -m src.scraper.jma_weather",
    "python -m src.scripts.data.backfill_trifecta_odds",
    "python -m src.research.pedigree.sporthorse_births",
])
def test_fetch_entrypoints_match(args):
    assert fr.is_fetch_args(args)


@pytest.mark.parametrize("args", [
    "python -m src.scripts.scraping.fetch_status",
    "python -m src.scraper.backfill --report",
    "python -m src.scraper.backfill_guard",
    "python -m src.scripts.data.backfill_predictions --model v6e",     # 予測の補填（取得ではない）
    "python main.py --prod",
])
def test_non_fetch_do_not_match(args):
    assert not fr.is_fetch_args(args)


def test_registry_and_status_do_not_trigger_backfill_yield():
    """表示・登録のプロセスが、補填の譲りの相手（OTHER_FETCH_PATTERN）に当たらないこと（当たると補填が待ち続ける）。"""
    for args in ("python -m src.scripts.scraping.fetch_status", "python -m src.scraper.fetch_registry mark --pid 1",
                 "bash scripts/dev/fetch_status.sh"):
        assert not re.search(bg.OTHER_FETCH_PATTERN, args)


def test_launcher_detected_by_script_body(tmp_path):
    sh = tmp_path / "a.sh"
    sh.write_text("#!/bin/bash\n# python -m src.scraper.backfill はコメントなので数えない\necho hi\n")
    assert fr.launcher_script(f"bash {sh}") is None
    sh2 = tmp_path / "b.sh"
    sh2.write_text("while sleep 600; do :; done\nPYTHONPATH=. python -m src.scraper.backfill --phase past\n")
    assert fr.launcher_script(f"bash {sh2}") == str(sh2)
    assert fr.launcher_script("b.sh", cwd=str(tmp_path)) == "b.sh"
    assert fr.launcher_script(f"/bin/bash cron_run.sh x -- {sh2}") == str(sh2)     # 包みの sh の中も見る
    assert fr.launcher_script(f"python {sh2}") is None


@pytest.fixture
def reg(tmp_path, monkeypatch):
    d = tmp_path / "reg"
    monkeypatch.setenv("KEIBA_FETCH_REGISTRY_DIR", str(d))
    monkeypatch.setattr(fr, "REGISTRY_DIR", d)
    monkeypatch.setattr(fr, "_state", {})
    monkeypatch.setattr(fr, "_last_flush", 0.0)
    return d


def test_note_get_writes_heartbeat_throttled(reg, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(fr.time, "time", lambda: t[0])
    fr.note_get()
    e = json.loads(fr._file_for(os.getpid()).read_text())
    assert e["gets"] == 1 and e["state"] == "running" and e["start_ticks"] == fr.proc_start_ticks(os.getpid())
    assert (reg.stat().st_mode & 0o7777) == 0o1777                         # root と jovyan の両方が書ける
    for _ in range(10):
        fr.note_get()
    assert json.loads(fr._file_for(os.getpid()).read_text())["gets"] == 1   # 5 秒に 1 回だけ書く
    t[0] += 6
    fr.note_get()
    e = fr.live_entries()[os.getpid()]
    assert e["gets"] == 12 and fr.gets_within(e, 3600, t[0]) == 12


def test_set_state_and_progress(reg):
    fr.set_state("waiting", "取得スロット待ち")
    e = fr.live_entries()[os.getpid()]
    assert e["state"] == "waiting" and e["waiting_for"] == "取得スロット待ち"
    fr.note_get()                                                           # GET が出たら自動で実行中に戻る（すぐに書く）
    assert fr.live_entries()[os.getpid()]["state"] == "running"
    fr.progress(10, 100, unit="頭", label="テスト")
    e = fr.live_entries()[os.getpid()]
    assert e["state"] == "running" and e["label"] == "テスト"


def test_live_entries_skip_dead_reused_and_broken(reg):
    reg.mkdir(parents=True)
    (reg / "999999999-0.json").write_text(json.dumps({"pid": 999999999}))   # 死んだ pid
    fr._file_for(os.getpid()).write_text(json.dumps({"start_ticks": -1}))   # PID の使い回し（開始時刻が合わない）
    (reg / "1-0.json").write_text("{壊れた")
    assert fr.live_entries() == {}
    assert fr.mark(os.getpid(), "waiting", "待ち受け", "PID 1 の終了")
    e = fr.live_entries()[os.getpid()]
    assert e["marked"] and e["waiting_for"] == "PID 1 の終了"
    assert not fr.mark(999999999)


def test_never_raises_when_dir_unwritable(monkeypatch, tmp_path):
    f = tmp_path / "file"
    f.write_text("")
    monkeypatch.setenv("KEIBA_FETCH_REGISTRY_DIR", str(f / "sub"))
    monkeypatch.setattr(fr, "REGISTRY_DIR", f / "sub")                      # ファイルの下にディレクトリは作れない
    monkeypatch.setattr(fr, "_last_flush", 0.0)
    fr.note_get()
    fr.set_state("waiting", "x")
    fr.progress(1, 2)


def test_disabled_under_pytest_without_explicit_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("KEIBA_FETCH_REGISTRY_DIR", raising=False)
    monkeypatch.setattr(fr, "REGISTRY_DIR", tmp_path / "reg")
    monkeypatch.setattr(fr, "_last_flush", 0.0)
    fr.note_get()
    assert not (tmp_path / "reg").exists()


def test_cleanup_removes_own_dead_files_only(reg):
    reg.mkdir(parents=True)
    dead = reg / f"999999999-{os.geteuid()}.json"
    dead.write_text("{}")
    alive = reg / f"{os.getpid()}-{os.geteuid()}.json"
    alive.write_text("{}")
    fr._cleaned[0] = False
    fr.note_get()
    assert not dead.exists() and alive.exists()
    assert not list(reg.glob(".*.tmp"))                                     # 一時ファイルは残らない


def test_marked_launcher_becomes_running_when_child_fetches():
    from src.scripts.scraping import fetch_status as fs

    marked = {"state": "waiting", "marked": True, "label": "取り直し", "waiting_for": "PID 1 の終了"}
    rows = [{"pid": 50, "ppid": 1, "user": "root", "etime": "1", "args": "bash /tmp/x/redo.sh"},
            {"pid": 51, "ppid": 50, "user": "root", "etime": "1", "args": "python -m src.scripts.scraping.backfill_horse_training_past --redo-truncated"}]
    ts = fs.collect_tasks(rows=rows, entries={50: marked}, now_ts=1.0)
    assert len(ts) == 1 and ts[0]["state"] == "running" and ts[0]["label"] == "取り直し"


def test_launcher_doing_non_fetch_work_is_hidden(tmp_path):
    from src.scripts.scraping import fetch_status as fs

    sh = tmp_path / "a.sh"
    sh.write_text("python -m src.scraper.backfill --phase past\nbash scripts/cron/gcs_mirror_push.sh\n")
    base = {"user": "root", "etime": "1"}
    rows = [dict(base, pid=60, ppid=1, args=f"bash {sh}")]
    assert [t["state"] for t in fs.collect_tasks(rows=rows + [dict(base, pid=61, ppid=60, args="sleep 300")], entries={})] == ["waiting"]
    assert fs.collect_tasks(rows=rows + [dict(base, pid=62, ppid=60, args="git push origin")], entries={}) == []   # 取得の後の作業中


def test_relative_launcher_resolves_from_repo_root_when_cwd_unreadable():
    assert fr.launcher_script("bash scripts/cron/horse_training_past.sh", cwd=None) == "scripts/cron/horse_training_past.sh"
    assert fr.launcher_script("ssh host scripts/cron/horse_training_past.sh") is None


def test_line_continuation_in_script_is_joined(tmp_path):
    sh = tmp_path / "c.sh"
    sh.write_text("PYTHONPATH=. nice -n 10 python -m \\\n  src.scraper.backfill --phase past\n")
    assert fr.script_launches_fetch(sh)


def test_client_get_and_waits_write_heartbeat(monkeypatch):
    from src.scraper import client as client_mod
    from tests.scraper.test_access_guard import R, U, _client

    calls = []
    monkeypatch.setattr(client_mod, "_GLOBAL_MAX_SLOTS", 0)
    monkeypatch.setattr(fr, "note_get", lambda n=1: calls.append("get"))
    monkeypatch.setattr(fr, "set_state", lambda st, w=None: calls.append(st))
    c = _client(monkeypatch, [R(200)])
    c._get_with_backoff(U)
    assert calls == ["get"]
    busy = iter([True, True, False])
    assert bg.wait_for_quiet(max_wait_sec=100, poll=0, busy=lambda: next(busy), sleep=lambda s: None)
    assert calls[1:] == ["waiting", "running"]


def test_registry_failures_do_not_break_fetching(monkeypatch):
    from src.scraper import client as client_mod
    from tests.scraper.test_access_guard import R, U, _client

    def boom(*a, **k):
        raise OSError("disk")

    monkeypatch.setattr(client_mod, "_GLOBAL_MAX_SLOTS", 0)
    monkeypatch.setenv("KEIBA_FETCH_REGISTRY_DIR", "/nonexistent")
    monkeypatch.setattr(fr, "_write", boom)
    c = _client(monkeypatch, [R(200)])
    assert c._get_with_backoff(U).status_code == 200                       # 心拍が書けなくても取得は続く
    assert bg.wait_for_quiet(max_wait_sec=100, poll=0, busy=iter([True, False]).__next__, sleep=lambda s: None)


def test_process_that_reports_progress_is_shown_even_if_not_in_pattern():
    from src.scripts.scraping import fetch_status as fs

    rows = [{"pid": 70, "ppid": 1, "user": "jovyan", "etime": "1", "args": "python -c 'from src.scraper import fetch_registry'"}]
    ts = fs.collect_tasks(rows=rows, entries={70: {"label": "研究用の取得", "total": 10, "done": 1, "gets": 0}}, now_ts=1e9)
    assert [t["label"] for t in ts] == ["研究用の取得"]
    rows = [{"pid": 71, "ppid": 1, "user": "root", "etime": "1", "args": "python -m src.scraper.fetch_registry mark --pid 1"}]
    assert fs.collect_tasks(rows=rows, entries={}, now_ts=1e9) == []          # 登録の CLI そのものは出さない


def test_comment_backslash_does_not_swallow_next_line(tmp_path):
    sh = tmp_path / "d.sh"
    sh.write_text("# 説明 \\\npython -m src.scraper.backfill --phase past\n")
    assert fr.script_launches_fetch(sh)


def test_marked_running_without_child_is_running():
    from src.scripts.scraping import fetch_status as fs

    rows = [{"pid": 80, "ppid": 1, "user": "root", "etime": "1", "args": "bash /tmp/x/job.sh"}]
    ts = fs.collect_tasks(rows=rows, entries={80: {"marked": True, "state": "running", "label": "自前の取得"}}, now_ts=1.0)
    assert ts[0]["state"] == "running"


def test_backfill_guard_survives_missing_registry(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "src.scraper" and a and a[2] and "fetch_registry" in a[2]:
            raise ImportError("x")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    busy = iter([True, False])
    assert bg.wait_for_quiet(max_wait_sec=10, poll=0, busy=lambda: next(busy), sleep=lambda s: None)
