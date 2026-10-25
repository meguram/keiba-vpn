"""ディスパッチャ（許可リスト・メモリ予算・ガード・依存・記録・通知）。実プロセスは default_runner の 2 件だけ、他は偽の runner。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from src.ops import job_dispatch as D, job_map
from src.ops.gcs_io import MemoryIO, read_json, write_json

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 10, 24, 19, 10, tzinfo=JST)         # 土曜 19:10


def J(jid: str, **kw: Any) -> dict[str, Any]:
    base = {"id": jid, "destination": "gcp", "enabled": True, "ready": True, "blockers": [], "group": "g1", "steps": [{"argv": ["-m", "mod.a", "--d", "{today}"]}],
            "memory_mib": 512, "timeout_sec": 60, "state_in": [{"path": "data/in", "max_mb": 1}], "state_out": [{"path": "data/out", "max_mb": 1}], "when_jst": None}
    base.update(kw)
    return base


def make_doc(jobs: list[dict[str, Any]], cron: str = "10,40 19 * * *") -> dict[str, Any]:
    return {"runtime": {"env": {"KEIBA_ENV": "prod", "KEIBA_RUN_LOCATION": "gcp"}, "run_jobs": {"s": {"memory_mib": 512}, "m": {"memory_mib": 2048}, "l": {"memory_mib": 4096}}},
            "scheduler_groups": {"g1": {"cron_jst": cron, "size": "m", "jobs": [j["id"] for j in jobs], "timeout_sec": 600}},
            "models": {"mdl": {"env": "KEIBA_TEST_MODEL", "local_path": "x/mdl"}},
            "jobs": jobs}


class Rec:
    def __init__(self, rcs: dict[str, int] | None = None):
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.rcs = rcs or {}

    def __call__(self, argv, env, timeout, cwd):
        self.calls.append((argv, env))
        return self.rcs.get(argv[1], 0), [f"out {argv[1]}"]


def ctx(io, runner, tmp_path, **kw) -> D.Ctx:
    alerts: list[tuple] = []
    pushes: list[Any] = []
    pulls: list[Any] = []
    mirrors: list[Any] = []
    c = D.Ctx(io=io, root=tmp_path, now_fn=lambda: NOW, runner=runner, base_env={"KEIBA_ENV": "stg"}, memory_limit_mib=kw.pop("memory_limit_mib", 4096),
              alert=lambda *a, **k: alerts.append((a, k)),
              pull=lambda io_, entries, root, **k: pulls.append(entries) or {"downloaded": 0},
              push=lambda io_, entries, root, **k: pushes.append(entries) or {"uploaded": 0},
              fetch_model=kw.pop("fetch_model", lambda io_, name, d: tmp_path / "models" / name),
              mirror=kw.pop("mirror", lambda trees, **k: mirrors.append((list(trees), k)) or {"uploaded": 0}),
              peak_fn=lambda: 123, **kw)
    c.alerts, c.pushes, c.pulls, c.mirrors = alerts, pushes, pulls, mirrors     # type: ignore[attr-defined]
    return c


@pytest.fixture
def patch_doc(monkeypatch):
    def _p(jobs, cron="10,40 19 * * *"):
        doc = make_doc(jobs, cron)
        monkeypatch.setattr(job_map, "load", lambda path=None: doc)
        return doc
    return _p


def test_success_records_run_and_forces_prod_gcp(tmp_path, patch_doc):
    patch_doc([J("a")])
    io, r = MemoryIO(), Rec()
    c = ctx(io, r, tmp_path)
    assert D.run_group("g1", c) == 0
    argv, env = r.calls[0]
    assert argv == ["-m", "mod.a", "--d", "20261024"]
    assert env["KEIBA_ENV"] == "prod" and env["KEIBA_RUN_LOCATION"] == "gcp"        # 通知で【PROD】/ GCP と分かる
    rec = read_json(io, "ops/job_runs/a/latest.json")
    assert rec["status"] == "ok" and rec["scheduled_at"].startswith("2026-10-24T19:10") and rec["memory_peak_mib"] == 123
    assert c.pulls == [[{"path": "data/in", "max_mb": 1}]] and c.pushes == [[{"path": "data/out", "max_mb": 1}]]
    assert any(k.startswith("ops/job_runs/a/2026") for k in io.list("ops/job_runs/a/"))


def test_failure_alerts_and_does_not_push_state(tmp_path, patch_doc):
    patch_doc([J("a")])
    io, r = MemoryIO(), Rec({"mod.a": 1})
    c = ctx(io, r, tmp_path)
    assert D.run_group("g1", c) == 1
    assert read_json(io, "ops/job_runs/a/latest.json")["status"] == "failed"
    assert c.pushes == [] and len(c.alerts) == 1 and c.alerts[0][1]["severity"] == "critical"


def test_allow_fail_makes_partial_with_warning(tmp_path, patch_doc):
    patch_doc([J("a", steps=[{"argv": ["-m", "mod.x"], "allow_fail": True}, {"argv": ["-m", "mod.y"]}])])
    io, r = MemoryIO(), Rec({"mod.x": 3})
    c = ctx(io, r, tmp_path)
    assert D.run_group("g1", c) == 0
    assert read_json(io, "ops/job_runs/a/latest.json")["status"] == "partial"
    assert [a[1]["severity"] for a in c.alerts] == ["warning"] and len(c.pushes) == 1


def test_ok_codes(tmp_path, patch_doc):
    patch_doc([J("a", steps=[{"argv": ["-m", "mod.n"], "ok_codes": [0, 10]}])])
    c = ctx(MemoryIO(), Rec({"mod.n": 10}), tmp_path)
    assert D.run_group("g1", c) == 0


def test_dependency_failure_and_unrun_dependency(tmp_path, patch_doc):
    patch_doc([J("a"), J("b", depends=["a"], steps=[{"argv": ["-m", "mod.b"]}])])
    io, r = MemoryIO(), Rec({"mod.a": 1})
    assert D.run_group("g1", ctx(io, r, tmp_path)) == 1
    assert read_json(io, "ops/job_runs/b/latest.json")["status"] == "dep_failed"
    assert all(a[1] != "mod.b" for a, _ in [(c[0], 0) for c in r.calls])
    # 依存が準備未完了で選ばれていなければ、依存先はスキップ（失敗扱いにしない）
    patch_doc([J("a", ready=False, blockers=["x"]), J("b", depends=["a"], steps=[{"argv": ["-m", "mod.b"]}])])
    io2, r2 = MemoryIO(), Rec()
    assert D.run_group("g1", ctx(io2, r2, tmp_path)) == 0
    assert read_json(io2, "ops/job_runs/b/latest.json")["status"] == "skipped" and r2.calls == []


def test_memory_budget_refuses_oversized_job(tmp_path, patch_doc):
    patch_doc([J("a", memory_mib=3072)])
    io, r = MemoryIO(), Rec()
    c = ctx(io, r, tmp_path, memory_limit_mib=2048)
    assert D.run_group("g1", c) == 1
    assert read_json(io, "ops/job_runs/a/latest.json")["status"] == "refused" and r.calls == []


def test_unready_and_disabled_are_not_run_unless_allowed(tmp_path, patch_doc):
    patch_doc([J("a", ready=False, blockers=["サイズ未確認"]), J("b", enabled=False)])
    r = Rec()
    assert D.run_group("g1", ctx(MemoryIO(), r, tmp_path)) == 0 and r.calls == []
    r2 = Rec()
    D.run_group("g1", ctx(MemoryIO(), r2, tmp_path, allow_unready=True))
    assert [c[0][1] for c in r2.calls] == ["mod.a"]            # enabled=false は --allow-unready でも動かさない


def test_when_jst_filter_uses_group_tick(tmp_path, patch_doc):
    # tick は 19:10。*/15 は 0,15,30,45 分 → 19:10 では動かない
    patch_doc([J("a", when_jst="*/15 * * * *"), J("b", steps=[{"argv": ["-m", "mod.b"]}])])
    r = Rec()
    D.run_group("g1", ctx(MemoryIO(), r, tmp_path))
    assert [c[0][1] for c in r.calls] == ["mod.b"]


def test_race_day_guard_skips_and_runs(tmp_path, patch_doc):
    patch_doc([J("a", requires_race_day="tomorrow")])
    io, r = MemoryIO(), Rec()
    assert D.run_group("g1", ctx(io, r, tmp_path)) == 0
    assert read_json(io, "ops/job_runs/a/latest.json")["status"] == "skipped" and r.calls == []
    write_json(io, "state/data/page_reference/race_lists/20261025.json", {"races": [{"race_id": "202605040101"}]})
    assert D.run_group("g1", ctx(io, r, tmp_path)) == 0 and len(r.calls) == 1


def test_fetch_done_guard_waits_then_fails_after_deadline(tmp_path, patch_doc):
    patch_doc([J("a", requires_fetch_done="raceday-eve-fetch", fetch_deadline_jst="20:15")])
    io, r = MemoryIO(), Rec()
    assert D.run_group("g1", ctx(io, r, tmp_path)) == 0
    assert read_json(io, "ops/job_runs/a/latest.json")["status"] == "waiting" and r.calls == []
    late = ctx(io, r, tmp_path)
    late.now_fn = lambda: datetime(2026, 10, 24, 20, 20, tzinfo=JST)
    assert D.run_group("g1", late) == 1 and late.alerts
    io.put_bytes("ops/fetch_tasks/done/raceday-eve-fetch-20261024T1800.json", b"{}")
    assert D.run_group("g1", ctx(io, r, tmp_path)) == 0 and len(r.calls) == 1


def test_parallel_env_applies_only_in_parallel_run(tmp_path, patch_doc):
    patch_doc([J("a", parallel_env={"KEIBA_V3_SERVING": "shadow"})])
    r = Rec()
    c = ctx(MemoryIO(), r, tmp_path)
    D.run_group("g1", c)
    assert "KEIBA_V3_SERVING" not in r.calls[0][1]
    r2 = Rec()
    c2 = ctx(MemoryIO(), r2, tmp_path)
    c2.base_env["KEIBA_PARALLEL_RUN"] = "1"
    D.run_group("g1", c2)
    assert r2.calls[0][1]["KEIBA_V3_SERVING"] == "shadow"


def test_models_are_fetched_and_exposed(tmp_path, patch_doc):
    patch_doc([J("a", models=["mdl"], steps=[{"argv": ["-m", "mod.m", "--dir", "{model:mdl}"]}])])
    r = Rec()
    assert D.run_group("g1", ctx(MemoryIO(), r, tmp_path)) == 0
    argv, env = r.calls[0]
    assert argv[-1] == str(tmp_path / "models" / "mdl") and env["KEIBA_TEST_MODEL"] == argv[-1]


def test_dry_run_runs_nothing_and_writes_nothing(tmp_path, patch_doc):
    patch_doc([J("a", requires_race_day="tomorrow")])
    io, r = MemoryIO(), Rec()
    c = ctx(io, r, tmp_path, dry_run=True)
    assert D.run_group("g1", c) == 0 and r.calls == [] and io.list("") == []


def test_only_gcp_jobs_in_map_can_run(tmp_path, monkeypatch):
    doc = make_doc([J("h", destination="host", group=None)])
    doc["scheduler_groups"] = {"g1": {"cron_jst": "0 7 * * *", "size": "m", "jobs": ["h"], "timeout_sec": 60}}
    monkeypatch.setattr(job_map, "load", lambda path=None: doc)
    with pytest.raises(SystemExit):
        D.run_group("g1", ctx(MemoryIO(), Rec(), tmp_path))
    with pytest.raises(SystemExit):
        D.select(doc, ["nope"], ctx(MemoryIO(), Rec(), tmp_path), None)


def test_state_budget_error_fails_before_running(tmp_path, patch_doc):
    from src.ops.state_sync import StateBudgetError

    patch_doc([J("a")])
    io, r = MemoryIO(), Rec()
    c = ctx(io, r, tmp_path)

    def boom(*a, **k):
        raise StateBudgetError("too big")
    c.pull = boom
    assert D.run_group("g1", c) == 1 and r.calls == []
    assert "準備に失敗" in read_json(io, "ops/job_runs/a/latest.json")["reason"]


def test_params_and_render():
    p = D.build_params(datetime(2026, 1, 5, 7, 0, tzinfo=JST))       # 月曜・1 月
    assert p["today"] == "20260105" and p["tomorrow_iso"] == "2026-01-06" and p["years"] == "2025,2026" and p["prev_year"] == "2025"
    assert p["last_weekend_days"] == "20251228,20260103,20260104"      # 過去 8 日の土日（drift_monitor_weekly.sh と同じ）
    assert D.render("a-{today}-{model:m}", p, {"m": "/x"}) == "a-20260105-/x"
    with pytest.raises(KeyError):
        D.render("{nope}", p)


def test_default_runner_real_process_and_timeout():
    rc, tail = D.default_runner(["-c", "print('hello')"], {"PATH": "/usr/bin:/bin"}, 20, ".")
    assert rc == 0 and tail[-1] == "hello"
    rc, tail = D.default_runner(["-c", "import time; time.sleep(30)"], {"PATH": "/usr/bin:/bin"}, 1, ".")
    assert rc == 124 and "timeout" in tail[-1]


def test_container_memory_limit_env_override(monkeypatch):
    monkeypatch.setenv("KEIBA_DISPATCH_MEMORY_MIB", "2048")
    assert D.container_memory_limit_mib() == 2048


def test_mirror_out_pushes_to_mirror_except_in_parallel_run(tmp_path, patch_doc):
    patch_doc([J("a", mirror_out=["data/local/inference/predictions_v6"])])
    c = ctx(MemoryIO(), Rec(), tmp_path)
    assert D.run_group("g1", c) == 0
    assert c.mirrors[0][0] == ["data/local/inference/predictions_v6"] and c.mirrors[0][1]["since_epoch"] > 0
    c2 = ctx(MemoryIO(), Rec(), tmp_path)
    c2.base_env["KEIBA_PARALLEL_RUN"] = "1"
    assert D.run_group("g1", c2) == 0 and c2.mirrors == []                          # 並走期間はホストの出力を上書きしない
    c3 = ctx(MemoryIO(), Rec({"mod.a": 1}), tmp_path)
    D.run_group("g1", c3)
    assert c3.mirrors == []                                                         # 失敗したジョブの出力は上げない


def test_mirror_failure_fails_the_job(tmp_path, patch_doc):
    patch_doc([J("a", mirror_out=["data/local/inference/predictions_v6"])])

    def boom(trees, **k):
        raise RuntimeError("gcs down")

    c = ctx(MemoryIO(), Rec(), tmp_path, mirror=boom)
    assert D.run_group("g1", c) != 0


@pytest.mark.parametrize("argv", [
    ["job", "predict-race", "race_id=1", "date=2"],
    ["job", "--allow-unready", "predict-race", "race_id=1", "date=2"],
    ["job", "predict-race", "--arg", "race_id=1", "--arg", "date=2"],
])
def test_job_args_can_be_passed_positionally_for_gcloud(monkeypatch, argv):
    """gcloud run jobs execute --args は '--arg' を繰り返せない（--args の短縮と解釈される）ので、KEY=VALUE を位置引数でも渡せる。
    Python 3.11（Jobs のイメージ）の argparse は、位置引数の間にオプションを挟むと失敗する。オプションは job_id より前に置く（3.13 では挟んでも通るので、このテストは 3.11 でも通る形だけ）。"""
    calls = []
    monkeypatch.setattr(D, "run_single", lambda job_id, c, extra=None: calls.append((job_id, extra)) or 0)
    monkeypatch.setattr(job_map, "validate", lambda doc: [])
    monkeypatch.setattr("src.ops.gcs_io.default_io", lambda: MemoryIO())
    assert D.main(argv) == 0
    assert calls == [("predict-race", {"race_id": "1", "date": "2"})]
