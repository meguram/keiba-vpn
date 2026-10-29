"""影モード（VPS の取得をホストと並走で検証）: 書き込みの隔離・タスク表の名前空間・計画・上限と停止・日次レポート・切替の関門。オフライン。"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops import fetch_agent as fa
from src.ops import fetch_shadow as fs
from src.ops import fetch_task as ft
from src.ops import job_map, job_owner, shadow_eval, shadow_storage as ss
from src.ops.gcs_io import MemoryIO, write_json
from src.scripts.ops import cron_cutover_status as CS, cutover_plan as CP, shadow_report as SR

JST = ZoneInfo("Asia/Tokyo")
T0 = datetime(2026, 10, 24, 10, 0, tzinfo=JST)
RACE = {"race_id": "202605040101", "date": "20261024"}
ROOT = Path(__file__).resolve().parents[2]
ON = {"KEIBA_FETCH_SHADOW_ENABLED": "1"}


# ───────────────────────── 書き込みの隔離 ─────────────────────────

class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket_, self.name = bucket, name

    def upload_from_string(self, data, **kw):
        self.bucket_.writes.append(self.name)
        self.bucket_.objects[self.name] = data

    def download_as_text(self, **kw):
        if self.name not in self.bucket_.objects:
            raise FileNotFoundError(self.name)
        return self.bucket_.objects[self.name]

    def exists(self, **kw):
        return self.name in self.bucket_.objects

    def delete(self, **kw):
        self.bucket_.deleted.append(self.name)


class FakeBucket:
    def __init__(self, objects=None):
        self.objects, self.writes, self.deleted = dict(objects or {}), [], []

    def blob(self, name, *a, **k):
        return FakeBlob(self, name)

    def get_blob(self, name, *a, **k):
        return FakeBlob(self, name) if name in self.objects else None

    def list_blobs(self, *a, **k):
        return iter([FakeBlob(self, n) for n in sorted(self.objects)])

    def delete_blob(self, name):
        self.deleted.append(name)

    name = "bkt"


def test_shadow_blob_writes_only_under_shadow_and_reads_production():
    real = FakeBucket({"prod/a.json": "PROD"})
    b = ss.ShadowBucket(real, lambda: "shadow/20261024/")
    blob = b.blob("prod/a.json")
    assert blob.download_as_text() == "PROD"                         # 読みは本番
    blob.upload_from_string("SHADOW")
    assert real.writes == ["shadow/20261024/prod/a.json"] and real.objects["prod/a.json"] == "PROD"   # 書きは影（本番は無傷）


def test_shadow_bucket_denies_delete_copy_and_other_mutations():
    real = FakeBucket({"prod/a.json": "PROD"})
    b = ss.ShadowBucket(real, lambda: "shadow/20261024/")
    for call in (lambda: b.blob("prod/a.json").delete(), lambda: b.delete_blob("prod/a.json"), lambda: b.copy_blob,
                 lambda: b.blob("prod/a.json").patch(), lambda: b.blob("prod/a.json").rewrite, lambda: b.get_blob("prod/a.json").delete(),
                 lambda: list(b.list_blobs())[0].delete()):
        with pytest.raises(ss.ShadowViolation):
            call()
    assert real.deleted == [] and real.objects == {"prod/a.json": "PROD"}


def test_assert_shadow_name_rejects_non_shadow():
    with pytest.raises(ss.ShadowViolation):
        ss.assert_shadow_name("chuou/data/x.json")
    assert ss.assert_shadow_name("shadow/20261024/x.json")


def test_install_isolates_hybridstorage_save_from_production(monkeypatch, tmp_path):
    """本番と同じ HybridStorage.save を通しても、GCS への書き込みは shadow/ だけ（本番のオブジェクトは変わらない）。"""
    from google.cloud import storage as gcs_lib
    from src.scraper.storage import HybridStorage

    monkeypatch.setattr(ss, "_INSTALLED", False)
    for name in ("bucket", "get_bucket", "lookup_bucket", "create_bucket"):
        monkeypatch.setattr(gcs_lib.Client, name, getattr(gcs_lib.Client, name))
    for name in ("_get_bucket", "exists", "exists_gcs"):
        monkeypatch.setattr(HybridStorage, name, getattr(HybridStorage, name))
    monkeypatch.setenv("KEIBA_SCHEMA_STRICT", "0")
    monkeypatch.setattr("src.scraper.storage.gcp_forbidden", lambda: False)     # この PC の .env が dev でも、本番と同じ保存経路を通す

    real = FakeBucket()
    st = HybridStorage(base_dir=str(tmp_path), bucket_name="bkt")
    path = st._gcs_blob_path("race_odds", "202605040101")
    real.objects[path] = json.dumps({"entries": [], "phase": "t15"})
    st._gcs_available, st._gcs_bucket, st._gcs_retry = True, real, None
    ss.install()
    assert isinstance(st._get_bucket(), ss.ShadowBucket)
    assert st.save("race_odds", "202605040101", {"race_id": "202605040101", "entries": [{"horse_number": 1, "win_odds": 3.0}], "phase": "t60"})
    assert real.writes and all(w.startswith("shadow/") and w.endswith(path) for w in real.writes)
    assert json.loads(real.objects[path]) == {"entries": [], "phase": "t15"}       # 本番のオブジェクトは変わらない
    assert real.deleted == []


def test_install_makes_skip_check_look_at_shadow_but_reads_stay_on_production(monkeypatch, tmp_path):
    """影に無ければ取る: exists は影だけを見る（本番にあっても False）。load は本番を読む。影に書けば True になる。"""
    from google.cloud import storage as gcs_lib
    from src.scraper.storage import HybridStorage

    monkeypatch.setattr(ss, "_INSTALLED", False)
    for name in ("bucket", "get_bucket", "lookup_bucket", "create_bucket"):
        monkeypatch.setattr(gcs_lib.Client, name, getattr(gcs_lib.Client, name))
    for name in ("_get_bucket", "exists", "exists_gcs"):
        monkeypatch.setattr(HybridStorage, name, getattr(HybridStorage, name))
    monkeypatch.setenv("KEIBA_SCHEMA_STRICT", "0")
    monkeypatch.setattr("src.scraper.storage.gcp_forbidden", lambda: False)

    real = FakeBucket()
    st = HybridStorage(base_dir=str(tmp_path), bucket_name="bkt")
    path = st._gcs_blob_path("race_odds", "202605040101")
    real.objects[path] = json.dumps({"race_id": "202605040101", "entries": [], "phase": "t15"})
    st._gcs_available, st._gcs_bucket, st._gcs_retry = True, real, None
    ss.install()
    assert st.exists("race_odds", "202605040101") is False                    # 本番にあっても、影に無ければ取る
    assert st._get_bucket().blob(path).exists() is True                       # スキップ判定の外では exists は本番のまま
    assert json.loads(st._get_bucket().blob(path).download_as_text())["phase"] == "t15"     # 読みは本番
    assert st.save("race_odds", "202605040101", {"race_id": "202605040101", "entries": [{"horse_number": 1, "win_odds": 3.0}], "phase": "t60"})
    assert st.exists("race_odds", "202605040101") is True                     # 影に書いたので次からスキップ
    assert real.deleted == [] and all(w.startswith("shadow/") for w in real.writes)


# ───────────────────────── タスク表の名前空間 ─────────────────────────

def test_prefixed_store_keeps_shadow_and_production_queues_apart():
    inner = ft.MemoryTaskStore()
    real_q = ft.TaskQueue(inner)
    shadow_q = ft.TaskQueue(fs.PrefixedStore(inner))
    t = ft.make_task("race-odds-t60", T0, args=RACE)
    assert shadow_q.enqueue(t) and real_q.list_ids("pending") == [] and shadow_q.list_ids("pending") == [t.task_id]
    assert all(k.startswith("ops/fetch_tasks_shadow/") for k in inner.list(""))
    with pytest.raises(ValueError):
        fs.PrefixedStore(inner).create("ops/events/weights_announced/20261024/x.json", "{}")      # 本番のイベント等には触れない
    with pytest.raises(ValueError):
        fs.PrefixedStore(inner).get("chuou/data/x.json")


# ───────────────────────── 計画 ─────────────────────────

def _cfg():
    return fs.load_config()


def test_shadow_off_by_default_plans_nothing():
    real = [ft.make_task("race-odds-t60", T0, args=RACE)]
    assert fs.shadow_tasks_from(real, _cfg(), env={}) == []
    assert not fs.enabled({})


def test_shadow_plan_jitter_exclusion_and_allowlist():
    real = [ft.make_task("race-odds-t60", T0, args=RACE), ft.make_task("race-odds-t15", T0 + timedelta(minutes=45), args=RACE),
            ft.make_task("weekly-update", T0), ft.make_task("daily-race-lists", T0)]
    out = fs.shadow_tasks_from(real, _cfg(), exclude={"daily-race-lists"}, env=ON)
    assert {t.task for t in out} == {"race-odds-t60", "race-odds-t15"}              # 重い weekly-update も、本番担当の daily-race-lists も影にしない
    for t, r in zip(out, real):
        lo, hi = _cfg().jitter_range(t.task)
        shift = (ft.parse_ts(t.not_before) - ft.parse_ts(r.not_before)).total_seconds()
        assert lo <= shift <= hi and shift > 0                                      # 本番（ホスト）と同じ秒に叩かない
        assert t.task_id == r.task_id and ft.parse_ts(t.deadline) > ft.parse_ts(t.not_before) and t.max_attempts == 1
    assert [t.not_before for t in fs.shadow_tasks_from(real, _cfg(), env=ON)] == [t.not_before for t in fs.shadow_tasks_from(real, _cfg(), env=ON)]  # 冪等


def test_weight_polls_are_thinned():
    real = [ft.make_task("race-weight-check", T0 + timedelta(seconds=180 * i), args=RACE) for i in range(30)]
    n = len(fs.shadow_tasks_from(real, _cfg(), env=ON))
    assert 3 <= n <= 20 and n < 30


# ───────────────────────── 上限と停止 ─────────────────────────

def test_budget_blocks_over_limit_and_resets_next_day(tmp_path):
    cfg = fs.ShadowConfig(tasks=_cfg().tasks, max_requests_per_day=30)
    clk = {"t": T0}
    st = fs.ShadowState(cfg, tmp_path / "s.json", clock=lambda: clk["t"])
    assert st.try_spend(25) and not st.try_spend(10) and st.try_spend(5)
    clk["t"] = T0 + timedelta(days=1)
    assert st.try_spend(25)


def test_restriction_pauses_and_second_one_stops_for_the_day(tmp_path):
    cfg = fs.ShadowConfig(tasks=_cfg().tasks, access_pause_seconds=3600, max_restrictions_per_day=2)
    st = fs.ShadowState(cfg, tmp_path / "s.json", clock=lambda: T0)
    u1 = st.record_restriction()
    assert u1 == T0 + timedelta(seconds=3600) and st.paused_until() == u1
    u2 = st.record_restriction()
    assert u2 > T0 + timedelta(hours=10)                                           # 2 回目は翌朝まで止める


def _shadow_agent(tmp_path, runner, *, owned=frozenset()):
    store = ft.MemoryTaskStore()
    cfg = fs.ShadowConfig(tasks=_cfg().tasks, max_requests_per_day=100, access_pause_seconds=3600)
    state = fs.ShadowState(cfg, tmp_path / "s.json", clock=lambda: T0)
    notes = []
    ag = fs.build_shadow_agent(store, fa.AgentConfig(agent_id="vps-x", enabled=True), "python", owned=owned, cfg=cfg, state=state,
                               runner=runner, clock=lambda: T0, sleep=lambda s: None, notify=lambda task, until: notes.append((task, until)))
    return ag, store, state, notes


def test_shadow_agent_runs_without_touching_production_events_or_queue(tmp_path):
    calls = []

    def runner(t, timeout):
        calls.append(t.task_id)
        return fa.RunResult(rc=0, result={"errors": [], "announced": True, "ratio": 1.0})

    ag, store, state, notes = _shadow_agent(tmp_path, runner)
    t = ft.make_task("race-weight-check", T0, args=RACE)
    ag.q.enqueue(t)
    store.create("ops/events/weights_announced/20261024/202605040101.json", "{}")        # 本番では『発表済み』だが、影は比較のため取り直す
    assert ag.tick() == f"ran:{t.task_id}:done" and calls == [t.task_id]
    assert ag.cfg.agent_id == "vps-x-shadow"
    keys = store.list("")
    assert all(k.startswith("ops/fetch_tasks_shadow/") or k == "ops/events/weights_announced/20261024/202605040101.json" for k in keys)
    assert [k for k in keys if "/events/" in k] == ["ops/events/weights_announced/20261024/202605040101.json"]    # 影はイベントを新たに置かない
    assert ft.TaskQueue(store).list_ids("done") == []                                       # 本番のタスク表には何も出ない


def test_shadow_agent_skips_tasks_the_agent_owns_for_real(tmp_path):
    ag, *_ = _shadow_agent(tmp_path, lambda t, to: fa.RunResult(rc=0, result={"errors": []}), owned={"entries-sync"})
    assert "entries-sync" not in ag.cfg.tasks and "race-odds-t60" in ag.cfg.tasks


def test_shadow_agent_access_restriction_stops_notifies_and_blocks_next(tmp_path):
    def runner(t, timeout):
        return fa.RunResult(rc=3, result={"errors": ["access_restriction"], "access_restriction": True})

    ag, store, state, notes = _shadow_agent(tmp_path, runner)
    a, b = ft.make_task("race-odds-t60", T0, args=RACE), ft.make_task("race-odds-t60", T0, args={**RACE, "race_id": "202605040102"})
    ag.q.enqueue(a), ag.q.enqueue(b)
    assert ag.tick().endswith(":failed")
    assert notes and notes[0][0] == "race-odds-t60" and ag.q.load("failed", a.task_id).outcome["access_restriction"] is True
    assert ag.tick() == "paused"                                                             # 30 分以上 claim を止める
    fresh = fs.build_shadow_agent(store, fa.AgentConfig(agent_id="vps-x", enabled=True), "python", cfg=ag.state.cfg, state=state,
                                  runner=runner, clock=lambda: T0, notify=lambda *a: None)
    assert fresh.tick() == "paused"                                                          # 再起動しても止まったまま（状態ファイル）


def test_budget_exhausted_skips_without_fetching(tmp_path):
    calls = []
    inner = lambda t, to: (calls.append(t.task_id), fa.RunResult(rc=0, result={"errors": []}))[1]  # noqa: E731
    cfg = fs.ShadowConfig(tasks=_cfg().tasks, max_requests_per_day=5)
    state = fs.ShadowState(cfg, tmp_path / "s.json", clock=lambda: T0)
    r = fs.BudgetedRunner(inner, state, cfg)
    res = r(ft.make_task("race-odds-t15", T0, args=RACE), 60)                              # 見積り 12 > 上限 5
    assert res.result["skipped"] == "budget_exhausted" and calls == []


def test_pause_is_shared_between_production_and_shadow_agents(tmp_path):
    ag, store, *_ = _shadow_agent(tmp_path, lambda t, to: fa.RunResult(rc=0, result={"errors": []}))
    real = fa.FetchAgent(ft.TaskQueue(store), fa.AgentConfig(agent_id="vps-x", enabled=True, tasks=frozenset({"entries-sync"})),
                         lambda t, to: fa.RunResult(rc=0), clock=lambda: T0)
    real.shadow = ag
    ag._paused_until = T0 + timedelta(hours=1)
    real._share_pause()
    assert real._paused_until == ag._paused_until


# ───────────────────────── 比較と日次レポート ─────────────────────────

def _odds(phase="t15", scale=1.0, horses=(1, 2, 3, 4)):
    return {"phase": phase, "entries": [{"horse_number": n, "win_odds": round(2.0 * n * scale, 1)} for n in horses]}


P = {"odds_rtol": 0.15, "odds_horse_ok_ratio": 0.9, "min_success_rate": 0.95, "min_match_ratio": 0.95, "max_skipped_ratio": 0.1, "max_alerts": 0}


def test_compare_odds_tolerates_timing_and_uses_same_phase_snapshot():
    assert SR.compare_odds(_odds(scale=1.05), _odds(), 0.15, 0.9)[0]
    assert not SR.compare_odds(_odds(scale=1.5), _odds(), 0.15, 0.9)[0]
    assert not SR.compare_odds(_odds(horses=(1, 2, 3)), _odds(), 0.15, 0.9)[0]               # 出走馬の集合が違う
    newer = {**_odds("t10", 3.0), "snapshots": [{"phase": "t15", "entries": _odds()["entries"]}]}   # 本番は後続の phase が最新、同じ phase は履歴にある
    assert SR.compare_odds(_odds("t15", 1.02), newer, 0.15, 0.9)[0]


def test_compare_shutuba_ignores_unannounced_weight_but_not_identity():
    base = {"entries": [{"horse_id": "h1", "horse_number": 1, "bracket_number": 1, "horse_name": "A", "jockey_name": "J", "trainer_name": "T", "sex_age": "牡3", "weight": None}]}
    prod = json.loads(json.dumps(base)); prod["entries"][0]["weight"] = 480
    assert SR.compare_shutuba(base, prod)[0]
    wrong = json.loads(json.dumps(base)); wrong["entries"][0]["jockey_name"] = "X"
    assert not SR.compare_shutuba(wrong, prod)[0]
    a, b = json.loads(json.dumps(prod)), json.loads(json.dumps(prod)); a["entries"][0]["weight"] = 470
    assert not SR.compare_shutuba(a, b)[0]


def _put_task(io, state, task, nb, *, outcome=None, tid=None, args=None):
    tid = tid or f"{task}-{nb.strftime('%H%M')}-{len(io.list(''))}"
    write_json(io, f"ops/fetch_tasks_shadow/{state}/{tid}.json", {"task_id": tid, "task": task, "args": args or {}, "not_before": nb.isoformat(),
                                                                  "deadline": (nb + timedelta(minutes=10)).isoformat(), "outcome": outcome or {}})


def test_build_day_counts_success_compares_and_judges():
    io, cfg, day = MemoryIO(), _cfg(), "20261024"
    now = datetime(2026, 10, 25, 7, 30, tzinfo=JST)
    for i in range(10):
        _put_task(io, "done", "race-odds-t60", T0 + timedelta(minutes=i), outcome={"result": {"errors": []}}, args={"race_id": f"20260504{i:04d}"[:12], "date": day})
    path = "chuou/data/preprocessed/netkeiba/race_odds/2026/202605040001.json"
    write_json(io, f"shadow/{day}/{path}", _odds(scale=1.03))
    write_json(io, path, _odds())
    rep = SR.build_day(io, day, cfg, now=now)
    assert rep["planned"] == 10 and rep["done_ok"] == 10 and rep["success_rate"] == 1.0 and rep["is_raceday"] is True
    assert rep["compared"] == 1 and rep["matched"] == 1 and rep["ok"] is True and rep["per_task"]["race-odds-t60"]["ok"] is True
    # 本番と食い違う影・失敗が混ざると不合格
    write_json(io, f"shadow/{day}/{path}", _odds(scale=2.0))
    _put_task(io, "failed", "race-odds-t60", T0 + timedelta(minutes=30), outcome={"access_restriction": True})
    bad = SR.build_day(io, day, cfg, now=now)
    assert bad["ok"] is False and bad["access_restrictions"] == 1 and any("一致率" in r for r in bad["reasons"])


def test_overdue_pending_counts_as_failure_and_skip_is_reported():
    io, cfg, day = MemoryIO(), _cfg(), "20261024"
    now = datetime(2026, 10, 25, 7, 30, tzinfo=JST)
    _put_task(io, "pending", "entries-sync", T0)                                         # 期限を過ぎても実行されなかった
    _put_task(io, "done", "race-odds-t15", T0, outcome={"result": {"skipped": "budget_exhausted", "errors": []}})
    rep = SR.build_day(io, day, cfg, now=now)
    assert rep["failed"] == 1 and rep["skipped"] == 1 and rep["ok"] is False


# ───────────────────────── 切替の関門 ─────────────────────────

TODAY = datetime(2026, 10, 30, 9, 0, tzinfo=JST)


def good_report(date, *, race=False, tasks=("entries-sync",), ok=True):
    pt = {t: {"done_ok": 3, "ok": True, "success_rate": 1.0} for t in tasks}
    if race:
        pt["race-odds-t60"] = {"done_ok": 20, "ok": True, "success_rate": 1.0}
    return {"date": date, "ok": ok, "reasons": [] if ok else ["一致率 50%"], "is_raceday": race, "per_task": pt}


def vps_doc(task="entries-sync", **job_kw):
    j = {"id": "es", "destination": "vps", "enabled": True, "ready": True, "blockers": [], "origin": {"cron_job_id": "es_c"}, "fetch_task": {"task": task}, "parity": {"kind": "none"}, **job_kw}
    return {"cutover_defaults": {"min_consecutive_success": 1, "window_days": 14, "max_start_delay_sec_vps": 9999, "require_shadow": True}, "jobs": [j]}


def reports_for(**kw):
    days = shadow_eval.window_dates(TODAY, 5)
    return {d: good_report(d, race=(i == 2), **kw) for i, d in enumerate(days)}


def test_shadow_days_passes_only_with_five_good_days_including_a_raceday():
    doc, cfg = vps_doc(), _cfg()
    j = doc["jobs"][0]
    assert shadow_eval.evaluate_job(j, doc, reports_for(), TODAY, cfg)["ok"] is True
    missing = reports_for(); missing.pop(next(iter(missing)))
    assert shadow_eval.evaluate_job(j, doc, missing, TODAY, cfg)["ok"] is False                       # 1 日欠けた
    assert shadow_eval.evaluate_job(j, doc, {}, TODAY, cfg)["ok"] is False                            # 読めない・無い = 不合格
    badday = reports_for(); k = list(badday)[3]; badday[k] = good_report(k, ok=False)
    assert shadow_eval.evaluate_job(j, doc, badday, TODAY, cfg)["ok"] is False
    noraceday = {d: {**r, "is_raceday": False} for d, r in reports_for().items()}
    r = shadow_eval.evaluate_job(j, doc, noraceday, TODAY, cfg)
    assert r["ok"] is False and "開催日" in r["detail"]
    few = {d: good_report(d, tasks=("entries-sync",)) for d in reports_for()}
    for r_ in few.values():
        r_["per_task"]["entries-sync"]["done_ok"] = 0
    assert shadow_eval.evaluate_job(j, doc, few, TODAY, cfg)["ok"] is False                           # そのタスクが実行された日が足りない


def test_not_shadowable_task_blocks_unless_explicitly_exempted():
    doc = vps_doc(task="weekly-update")
    j = doc["jobs"][0]
    c = shadow_eval.evaluate_job(j, doc, reports_for(), TODAY, _cfg())
    assert c["ok"] is False and "対象外" in c["detail"]
    j["shadow"] = {"exempt": True, "reason": "ユーザー承認 2026-11-01"}
    assert shadow_eval.evaluate_job(j, doc, {}, TODAY, _cfg())["ok"] is True


def test_cutover_status_adds_shadow_days_check_and_fails_closed():
    doc = vps_doc()
    io = MemoryIO()
    j = doc["jobs"][0]
    write_json(io, "ops/fetch_tasks/done/es-1.json", {"task_id": "es-1", "task": "entries-sync", "not_before": (TODAY - timedelta(hours=2)).isoformat(),
                                                     "claimed_at": (TODAY - timedelta(hours=2)).isoformat(), "state": "done"})
    rec = CS.Records(io, shadow_cfg=_cfg())
    res = CS.evaluate_all(doc, rec, TODAY)
    row = res["rows"][0]
    assert row["state"] == "不合格" and any(c["name"] == "shadow_days" and c["ok"] is False for c in row["checks"])
    assert res["cut_units"]["es_c"]["ready_to_stop"] is False
    for d, r in reports_for().items():
        write_json(io, shadow_eval.REPORT_KEY.format(date=d), r)
    res = CS.evaluate_all(doc, CS.Records(io, shadow_cfg=_cfg()), TODAY)
    assert res["rows"][0]["state"] == "合格" and res["cut_units"]["es_c"]["ready_to_stop"] is True
    assert j["destination"] == "vps"


def test_cutover_plan_refuses_vps_unit_without_shadow_even_with_force(tmp_path, capsys):
    status_ok_except_shadow = {"rows": [{"job": "entries-sync", "checks": [{"name": "consecutive_success", "ok": True}]}],
                               "cut_units": {"entries-sync": {"ready_to_stop": True, "waiting_on": []}}}
    f = tmp_path / "status.json"
    f.write_text(json.dumps(status_ok_except_shadow))
    for extra in ([], ["--force"]):
        rc = CP.main(["make", "--unit", "entries-sync", "--status-file", str(f), "--out-dir", str(tmp_path / "o"), *extra])
        assert rc == 3
    assert not (tmp_path / "o").exists() or not list((tmp_path / "o").iterdir())
    assert CP.main(["make", "--unit", "entries-sync", "--force", "--out-dir", str(tmp_path / "o2")]) == 3        # 状態ファイル無しでも --force では作れない
    shadow_ok = {"rows": [{"job": "entries-sync", "checks": [{"name": "shadow_days", "ok": True, "detail": "ok"}]}],
                 "cut_units": {"entries-sync": {"ready_to_stop": True, "waiting_on": []}}}
    f.write_text(json.dumps(shadow_ok))
    assert CP.main(["make", "--unit", "entries-sync", "--status-file", str(f), "--out-dir", str(tmp_path / "o3")]) == 0
    s1 = (tmp_path / "o3" / "cutover_entries-sync_1_owner.sh").read_text()
    assert "src.ops.fetch_shadow can-switch entries-sync" in s1 and s1.index("can-switch") < s1.index("job_owner set")    # 何も変える前に関門


def test_gcp_units_are_not_affected_by_the_shadow_gate():
    ok, reasons = shadow_eval.status_gate(None, "odds-drift-weekly", job_map.load())
    assert ok and reasons == []


def test_job_owner_refuses_vps_unit_until_shadow_reports_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_FETCH_SHADOW_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("KEIBA_JOB_OWNER_FILE", str(tmp_path / "owner.json"))
    with pytest.raises(ValueError, match="影モード"):
        job_owner.set_owner("entries-sync", "vps")
    with pytest.raises(ValueError, match="影モード"):
        job_owner.set_owner("entries-sync", "vps", force=True)                                   # force でも外せない
    assert job_owner.main(["can-stop", "entries-sync"]) in (3, 6)
    # レポートが揃えば通る
    (tmp_path / "reports").mkdir()
    now = datetime.now(JST)
    for i, d in enumerate(shadow_eval.window_dates(now, 5)):
        rep = good_report(d, race=(i == 1), tasks=("entries-sync",))
        (tmp_path / "reports" / f"{d}.json").write_text(json.dumps(rep))
    job_owner.set_owner("entries-sync", "vps")
    assert job_owner.main(["can-stop", "entries-sync"]) == 0
    assert fs.main(["can-switch", "entries-sync"]) == 0


def test_job_owner_can_stop_blocks_with_exit_6_when_reports_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_FETCH_SHADOW_REPORT_DIR", str(tmp_path / "reports"))
    owner = tmp_path / "owner.json"
    owner.write_text(json.dumps({"owners": {"entries-sync": "vps"}}))
    monkeypatch.setenv("KEIBA_JOB_OWNER_FILE", str(owner))
    assert job_owner.main(["can-stop", "entries-sync"]) == 6
    assert fs.main(["can-switch", "entries-sync"]) == fs.BLOCKED_EXIT


def test_host_cron_switch_rejects_no_owner_check_on_real_crontab():
    env = {k: v for k, v in os.environ.items() if k != "CRONTAB_CMD"}
    r = subprocess.run(["bash", str(ROOT / "scripts/ops/cutover/host_cron_switch.sh"), "disable", "--job-id", "entries-sync", "--no-owner-check"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "検証用" in r.stderr


def test_default_job_map_requires_shadow_and_config_is_consistent():
    doc = job_map.load()
    assert doc["cutover_defaults"]["require_shadow"] is True
    cfg = fs.load_config()
    assert cfg.task_names() == frozenset(cfg.tasks)                                          # 設定のタスクはすべて allowlist にある
    assert cfg.passing["required_days"] >= 5 and cfg.access_pause_seconds >= 1800
    assert not (cfg.task_names() & {"weekly-update", "raceday-evening", "new-horse-pedigree", "catchup-missing"})   # 重い定期処理は対象外
    assert next(j for j in doc["jobs"] if j["id"] == "fetch-shadow-report")["group"] in doc["scheduler_groups"]


def test_fetch_child_refuses_tasks_outside_shadow_allowlist_and_installs_guard(monkeypatch, capsys):
    from src.ops import fetch_child as fc

    monkeypatch.setenv("KEIBA_FETCH_SHADOW", "1")
    assert fc.main(["auto", "weekly-update"]) == 4                                   # 重い定期処理は影では実行しない
    out = capsys.readouterr().out
    assert "shadow_not_allowed" in out and '"shadow": true' in out
    installed = []
    monkeypatch.setattr(ss, "install", lambda: installed.append(1))
    monkeypatch.setattr(fc, "run_auto", lambda mode: 0)
    assert fc.main(["auto", "entries-sync"]) == 0 and installed == [1]               # 対象のタスクは書き込みガードを入れてから実行
    monkeypatch.delenv("KEIBA_FETCH_SHADOW")
    installed.clear()
    assert fc.main(["auto", "weekly-update"]) == 0 and installed == []               # 通常の取得には一切影響しない
