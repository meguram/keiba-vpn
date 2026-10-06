"""stg のスクレイピング実行スクリプト（データチェックの計画 → キュー → アクセス制限で即終了 → 再開）のテスト。
実際のスクレイピング・GCS・netkeiba には一切触れない（フェイクのキュー／チェック／応答で検証）。"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.scraper import client as client_mod
from src.scraper import scrape_access_pause as pause_mod
from src.scraper.job_queue import ScrapeJobQueue
from src.scripts.scraping import scrape_from_health_plan as M
from tests.data_health.test_data_health import FakeStorage, _dt, _ids, _write_race_list, stg_env  # noqa: F401
from tests.data_health.test_validation import NO_ART, _settings, _world
from src.data_health.runner import run_health

NF = "<html>お探しのページが見つかりません</html>"


class Resp:
    def __init__(self, code, body="<html>x</html>"):
        self.status_code, self.text, self.content, self.headers = code, body, body.encode(), {"Content-Type": "text/html"}


class FakeBucket:
    def exists(self, timeout=None):
        return True


class FakeStorageOK:
    gcs_enabled = True

    def _get_bucket(self):
        return FakeBucket()


class FakeQueue:
    def __init__(self, on_process=None):
        self.added, self.paused, self.jobs = [], [], []
        self.locked, self.processed, self.on_process = False, 0, on_process
        self.requeued = []

    def is_locked(self):
        return self.locked

    def get_status(self):
        return {"pending": 0, "precheck": 0}

    def bulk_add_jobs(self, specs):
        self.added.append(list(specs))
        return {"created": len(specs)}

    def load_queue(self):
        return self.jobs

    def requeue_failed_jobs(self, job_ids=None, all_failed=False):
        self.requeued += list(job_ids or [])
        return len(job_ids or []), None

    def pause_queue_for_access_error(self, reason):
        self.paused.append(reason)
        pause_mod.write_access_pause(reason=reason)
        return 1

    def process_queue(self):
        self.processed += 1
        if self.on_process:
            self.on_process()


def S(kind="race", target="202605030211", tasks=("race_result",), *, status="missing", runner="学習PC（過去分の補完）", conf="confirmed",
      cats=("race_result",), overwrite=False, priority=0, date="20260101"):
    return {"job_kind": kind, "target_id": target, "tasks": list(tasks), "smart_skip": not overwrite, "overwrite": overwrite,
            "priority": priority, "reason": "テスト", "runner": runner, "status": status, "categories": list(cats),
            "confidence": conf, "date": date}


def report(specs, complete=False):
    comp = {"complete": complete, "scope": "2020-01-01 〜 2026-10-05", "levels": ["required"],
            "reasons": [] if complete else [{"code": "gap", "message": "race_result: 不足", "count": len(specs)}]}
    return {"plan": {"counts": {"jobs": len(specs), "by_runner": {}}, "specs": specs, "notes": []}, "completeness": comp}


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("netkeiba_id", "u@example.com")
    monkeypatch.setenv("netkeiba_pw", "secret")
    monkeypatch.setattr(pause_mod, "PAUSE_FILE", tmp_path / "pause.json")
    monkeypatch.setattr(client_mod, "_RESPONSE_OBSERVERS", [])
    return tmp_path


def go(argv, *, queue=None, reports=(), **kw):
    """run() を実行して (終了コード, 出力行, キュー, チェック回数) を返す。reports は呼ばれるたびに順に返すチェック結果。"""
    args = M.parse_args(list(argv))
    q = queue or FakeQueue()
    seq = list(reports)
    calls = []

    def check(a):
        calls.append(1)
        return seq[min(len(calls) - 1, len(seq) - 1)]

    lines: list[str] = []
    kw.setdefault("env", "stg")
    code = M.run(args, queue=q, check=check, storage=FakeStorageOK(), out=lines.append, sleep=lambda s: None,
                 confirm=kw.pop("confirm", None), probe=kw.pop("probe", None), **kw)
    return code, "\n".join(lines), q, len(calls)


# ── 事前確認 ────────────────────────────────────────────────────────────────

def test_preflight_refuses_dev_prod_missing_credentials_gcs_pause_and_running_worker(env):
    q = FakeQueue()
    p = lambda **k: M.preflight(k.pop("env", "stg"), allow_prod=k.pop("allow_prod", False), queue=k.pop("queue", q),  # noqa: E731
                                storage=k.pop("storage", FakeStorageOK()), environ=k.pop("environ", {"netkeiba_id": "a", "netkeiba_pw": "b"}))
    assert p() == []
    assert any("dev では実行できません" in m for _, m in p(env="dev"))
    assert any("prod では実行しません" in m for _, m in p(env="prod"))
    assert p(env="prod", allow_prod=True) == []
    assert any("認証情報" in m for _, m in p(environ={}))
    bad = type("S", (), {"gcs_enabled": False})()
    assert any("GCS に接続できません" in m for _, m in p(storage=bad))
    q.locked = True
    assert any("ワーカーが動いています" in m for _, m in p())
    q.locked = False
    pause_mod.write_access_pause(reason="block")
    assert any("アクセス一時停止中" in m for _, m in p())


def test_run_stops_at_preflight_without_touching_the_queue(env):
    code, out, q, n = go(["--execute"], reports=[report([S()])], env="dev")
    assert code == M.EXIT_PREFLIGHT and q.added == [] and q.processed == 0 and n == 0 and "dev では実行できません" in out


# ── 絞り込み・見積り ──────────────────────────────────────────────────────────

def test_select_and_estimate():
    specs = [S(target="202605030201", priority=0, date="20250101"), S(target="202605030202", priority=10, date="20260101"),
             S("date", "20250601", ("date_all",), runner="VPS cron（当日取得）", cats=("race_index",)),
             S(status="invalid", target="202605030203", overwrite=True), S(target="202605030204", conf="inferred"),
             S("horse", "2023100001", ("horse_profile", "horse_pedigree_5gen"), cats=("horse_result",))]
    ids = lambda sel: [s["target_id"] for s in sel]  # noqa: E731
    assert ids(M.select_specs(specs))[0] == "202605030202"                                  # 優先度の高い順
    assert set(ids(M.select_specs(specs, runner="vps"))) == {"20250601"}
    assert ids(M.select_specs(specs, status="invalid")) == ["202605030203"]
    assert set(ids(M.select_specs(specs, categories=["race_index"]))) == {"20250601"}
    assert "202605030204" not in ids(M.select_specs(specs, include_inferred=False))
    assert "202605030201" not in ids(M.select_specs(specs, since="2025-06-01"))            # 期間（spec の date）
    assert len(M.select_specs(specs, max_jobs=2)) == 2
    e = M.estimate(specs)
    assert e["jobs"] == 6 and e["overwrite"] == 1 and e["requests"] == 4 + M.DATE_ALL_REQUESTS + 4 and e["by_status"]["invalid"] == 1
    assert "上書き再取得" in M.format_estimate(e)


# ── ドライラン・実行・確認 ────────────────────────────────────────────────────

def test_dry_run_is_the_default_and_enqueues_nothing(env):
    code, out, q, _ = go([], reports=[report([S(), S(target="202605030212")])])
    assert code == M.EXIT_REMAINING and q.added == [] and q.processed == 0
    assert "ドライラン" in out and "202605030211" in out and "ジョブ 2 件" in out


def test_execute_runs_check_enqueue_process_and_final_check(env):
    specs = [S(), S(target="202605030212")]
    code, out, q, n = go(["--execute"], reports=[report(specs), report([], complete=True)])
    assert code == M.EXIT_COMPLETE and n == 2                                               # 実行前と実行後の 2 回チェック
    assert [s["target_id"] for s in q.added[0]] == ["202605030211", "202605030212"] and q.processed == 1
    assert "完全性: OK" in out


def test_overwrite_or_many_jobs_require_confirmation(env):
    specs = [S(status="invalid", overwrite=True)]
    asked = []
    code, out, q, _ = go(["--execute"], reports=[report(specs)], confirm=lambda m: (asked.append(m), False)[1])
    assert code == M.EXIT_NOT_CONFIRMED and q.added == [] and "同一バケット" in asked[0]      # 上書きは確認なしには実行しない
    code, _, q, _ = go(["--execute"], reports=[report(specs), report([], True)], confirm=lambda m: True)
    assert code == M.EXIT_COMPLETE and q.added[0][0]["overwrite"] is True
    code, _, q, _ = go(["--execute", "--yes"], reports=[report(specs), report([], True)])    # --yes で非対話
    assert code == M.EXIT_COMPLETE
    many = [S(target=f"20260503{i:04d}") for i in range(M.CONFIRM_ABOVE + 1)]
    code, _, q, _ = go(["--execute"], reports=[report(many)], confirm=lambda m: False)
    assert code == M.EXIT_NOT_CONFIRMED


def test_rounds_stop_when_there_is_no_progress(env):
    specs = [S(), S(target="202605030212")]
    code, out, q, _ = go(["--execute", "--rounds", "5"], reports=[report(specs), report(specs), report(specs)])
    assert code == M.EXIT_REMAINING and "進捗がありません" in out and q.processed == 1


def test_rounds_continue_while_progress_is_made(env):
    a = [S(target=f"20260503{i:04d}") for i in range(4)]
    code, out, q, _ = go(["--execute", "--rounds", "5"], reports=[report(a), report(a[:2]), report([], True)])
    assert code == M.EXIT_COMPLETE and q.processed == 2


def test_requeue_failed_jobs_for_the_same_dedupe_key(env):
    sp = S()
    q = FakeQueue()
    q.jobs = [{"job_id": "q1", "status": "failed", "dedupe_key": M.dedupe_key(sp)}, {"job_id": "q2", "status": "failed", "dedupe_key": "other"}]
    assert M.requeue_failed(q, [sp]) == 1 and q.requeued == ["q1"]
    code, out, q2, _ = go(["--execute", "--requeue-failed"], queue=q, reports=[report([sp]), report([], True)])
    assert code == M.EXIT_COMPLETE and "待機に戻した" in out


# ── アクセス制限: 検知 → 即終了 → その時点のチェック → 再開 ────────────────────────

def _trip(status=404, body="<html>Not Found</html>", url="https://db.netkeiba.com/race/202605030211/"):
    def fire():
        for obs in tuple(client_mod._RESPONSE_OBSERVERS):
            obs(url, Resp(status, body))
    return fire


def test_restriction_stops_immediately_checks_again_and_saves_resume_state(env):
    specs = [S(), S(target="202605030212")]
    after = [S(target="202605030212")]                                                      # 1 件は取得できた、という状況
    code, out, q, n = go(["--execute", "--max-jobs", "50", "--category", "race_result"], queue=FakeQueue(_trip()),
                         reports=[report(specs), report(after)])
    assert code == M.EXIT_BLOCKED and n == 2                                                 # 中断後にもう一度データチェック
    assert q.paused and "アクセス制限" in q.paused[0]                                         # 実行中ジョブを待機に戻し、一時停止フラグ
    assert pause_mod.read_access_pause()["active"] is True
    assert client_mod._RESPONSE_OBSERVERS == []                                              # 監視は解除されている
    st = json.loads(M.state_path().read_text(encoding="utf-8"))
    assert st["status"] == 404 and st["remaining"]["jobs"] == 1 and st["resume_command"].endswith("--resume")
    assert st["argv"] == ["--execute", "--max-jobs", "50", "--category", "race_result"]      # 再開で条件を引き継ぐ
    plan = json.loads((M.state_dir() / "scrape_plan.after_restriction.json").read_text(encoding="utf-8"))
    assert [s["target_id"] for s in plan["specs"]] == ["202605030212"]                        # 最新の状況に基づく、再開後に実行する計画
    assert "直ちに実行を終了" in out and "--resume" in out and "完全性: NG" in out


@pytest.mark.parametrize("fire", [_trip(500, "x"), _trip(403, "x"), _trip(429, "x"), _trip(400, "x"), _trip(200, "<html>アクセスが制限されています</html>")])
def test_other_errors_are_also_treated_as_restriction(env, fire):
    code, out, q, _ = go(["--execute"], queue=FakeQueue(fire), reports=[report([S()])])
    assert code == M.EXIT_BLOCKED and M.state_path().exists()


def test_transport_error_is_treated_as_restriction(env):
    import requests

    def fire():
        for obs in tuple(client_mod._RESPONSE_OBSERVERS):
            obs.on_error("https://db.netkeiba.com/x", requests.exceptions.ConnectTimeout("t"))
    code, out, q, _ = go(["--execute"], queue=FakeQueue(fire), reports=[report([S()])])
    assert code == M.EXIT_BLOCKED and "通信エラー" in out


def test_page_not_found_does_not_stop_the_run(env):
    q = FakeQueue(_trip(404, NF))
    code, out, q, _ = go(["--execute"], queue=q, reports=[report([S()]), report([], True)])
    assert code == M.EXIT_COMPLETE and not q.paused and "存在しないページとして見逃した" in out
    strict = FakeQueue(_trip(404, NF))
    code, *_ = go(["--execute", "--strict-not-found"], queue=strict, reports=[report([S()])])
    assert code == M.EXIT_BLOCKED                                                            # より敏感に: 存在しないページも制限扱い


def test_monitoring_can_be_tuned_or_disabled(env):
    code, *_ = go(["--execute", "--stop-on-status", ""], queue=FakeQueue(_trip(500, "x")), reports=[report([S()]), report([], True)])
    assert code == M.EXIT_COMPLETE
    code, *_ = go(["--execute", "--stop-on-status", "403"], queue=FakeQueue(_trip(500, "x")), reports=[report([S()]), report([], True)])
    assert code == M.EXIT_COMPLETE                                                           # 403 だけ監視 → 500 は対象外
    two = [None]

    def fire():
        for obs in tuple(client_mod._RESPONSE_OBSERVERS):
            obs("https://db.netkeiba.com/a", Resp(500, "x"))
            obs("https://db.netkeiba.com/a", Resp(200))                                     # 間に成功があればリセット
            obs("https://db.netkeiba.com/a", Resp(500, "x"))
    code, *_ = go(["--execute", "--stop-after", "2"], queue=FakeQueue(fire), reports=[report([S()]), report([], True)])
    assert code == M.EXIT_COMPLETE
    assert M.parse_statuses("any") is None and M.parse_statuses("404, 403") == {404, 403} and M.parse_statuses("") == set()
    with pytest.raises(ValueError):
        M.parse_statuses("abc")


def test_pause_flag_set_by_the_queue_itself_is_handled_like_a_restriction(env):
    """キュー側の 400 ブロック検知（pause フラグ）で process_queue が戻ってきた場合も、チェック→状態保存→終了する。"""
    q = FakeQueue(lambda: pause_mod.write_access_pause(reason="HTTP 400（ブロック疑い）"))
    code, out, q, n = go(["--execute"], queue=q, reports=[report([S()]), report([S()])])
    assert code == M.EXIT_BLOCKED and n == 2 and M.state_path().exists() and "HTTP 400" in out


def _restricted_state(env):
    code, *_ = go(["--execute", "--max-jobs", "7"], queue=FakeQueue(_trip()), reports=[report([S()]), report([S()])])
    assert code == M.EXIT_BLOCKED
    assert pause_mod.read_access_pause()["active"]


def test_resume_requires_cleared_pause_and_a_successful_probe(env):
    assert go(["--resume"], reports=[report([S()])])[0] == M.EXIT_PREFLIGHT                 # 再開する状態が無い
    _restricted_state(env)
    probed = []
    code, out, q, n = go(["--resume"], reports=[report([S()])], probe=lambda: (probed.append(1), (True, "ok"))[1])
    assert code == M.EXIT_BLOCKED and "--clear-pause" in out and probed == [] and n == 0    # 一時停止が残っていれば何もしない
    code, out, q, n = go(["--resume", "--clear-pause"], reports=[report([S()])], probe=lambda: (False, "HTTP 404"))
    assert code == M.EXIT_BLOCKED and "疎通確認: NG" in out and q.added == []               # まだ繋がらなければ実行しない
    assert pause_mod.read_access_pause()["active"] is False                                  # フラグは解除済み
    assert M.state_path().exists()                                                           # 状態は残る（まだ再開できていない）


def test_resume_replays_saved_conditions_and_resolves_the_state(env):
    _restricted_state(env)
    args = M.parse_args(["--resume", "--clear-pause", "--skip-probe", "--max-jobs", "3"])
    assert args.execute is True and args.max_jobs == 3 and args.resume is True                # 前回の --execute を引き継ぎ、今回の指定が優先
    code, out, q, n = go(["--resume", "--clear-pause", "--skip-probe", "--yes"], reports=[report([S()]), report([], True)])
    assert code == M.EXIT_COMPLETE and q.processed == 1 and "再開:" in out
    assert q.added[0] and not M.state_path().exists()                                        # 解決済みの状態へ退避
    assert list(M.state_dir().glob("access_restriction.resolved.*.json"))


def test_resume_after_a_second_restriction_keeps_a_fresh_state(env):
    _restricted_state(env)
    code, out, *_ = go(["--resume", "--clear-pause", "--skip-probe", "--yes"], queue=FakeQueue(_trip(503, "x")),
                       reports=[report([S()]), report([S()])])
    assert code == M.EXIT_BLOCKED and M.state_path().exists()                                # 再開後にまた制限 → また即終了して状態を更新


def test_run_log_records_rejections_and_outcome(env):
    code, out, q, _ = go(["--execute"], reports=[report([S()]), report([], True)])
    logs = list((M.state_dir() / "scrape_runs").glob("*.json"))
    rec = json.loads(logs[0].read_text(encoding="utf-8"))
    assert rec["exit_code"] == 0 and rec["rounds"][0]["enqueue"] == {"created": 1} and "schema_rejections" in rec["rounds"][0]


# ── チェック → キューの連携（実際のキュー正規化を通す）─────────────────────────────

def test_real_plan_specs_are_accepted_by_the_real_queue_normalizer(stg_env):
    """データチェックが作る spec（不足・スキーマ不適合・date_all・馬）が、ScrapeJobQueue にそのまま投入できる形式であること。"""
    day, _ = _world(stg_env, races=12)
    storage = _world(stg_env, bad={("race_odds", sorted(day)[0])})[1]
    storage.listing["race_index"]["2026"].discard(sorted(day)[1])
    rep = run_health(settings=_settings(), actual_env="stg", storage=storage, now=_dt(2026, 10, 20), since=date(2026, 1, 1),
                     until=date(2026, 10, 25), root=stg_env, infra=False, horses=False, ledger_dir=None)
    specs = rep["plan"]["specs"]
    assert {s["status"] for s in specs} >= {"missing", "invalid"}
    for s in specs:
        n = ScrapeJobQueue._normalize_incoming_job(None, s)                                  # ValueError なら連携不備
        assert n["job_kind"] == s["job_kind"] and n["tasks"] == sorted(set(s["tasks"]))
        assert (n["overwrite"], n["smart_skip"]) == ((True, False) if s["overwrite"] else (False, True))   # 上書き/スキップの意図が保たれる
    assert M.dedupe_key(specs[0]) == ScrapeJobQueue._normalize_incoming_job(None, specs[0])["dedupe_key"]
    sel = M.select_specs(specs, status="invalid")
    assert sel and all(s["overwrite"] for s in sel)


def _run_status(key: str = "stg") -> dict:
    from src.data_health import store

    p = store.base_dir(None) / "run_status.json"
    return json.loads(p.read_text(encoding="utf-8")).get(key, {}) if p.exists() else {}


def test_execute_reports_progress_to_the_dashboard_and_finishes_idle(env):
    code, _, _, _ = go(["--execute"], reports=[report([S()]), report([], complete=True)])
    st = _run_status()
    assert code == M.EXIT_COMPLETE and st["state"] == "idle" and st["phase"] == "完了"


def test_restriction_is_shown_as_interrupted_on_the_dashboard(env):
    go(["--execute"], queue=FakeQueue(_trip()), reports=[report([S()]), report([S()])])
    st = _run_status()
    assert st["state"] == "idle" and st["phase"] == "中断"


def test_dry_run_does_not_touch_the_dashboard_status(env):
    go([], reports=[report([S()])])
    assert _run_status() == {}
