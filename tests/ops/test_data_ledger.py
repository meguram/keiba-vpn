"""データの見た回数の台帳（src/ops/data_ledger.py）。出力先を変えても 2 回目の確かめが拒否されること等。"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from src.ops import data_ledger as L

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def led(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_LEDGER_SHARED", str(tmp_path / "shared" / "ledger.jsonl"))
    monkeypatch.setenv("KEIBA_LEDGER_GIT", str(tmp_path / "git" / "ledger.jsonl"))
    (tmp_path / "shared").mkdir()
    monkeypatch.delenv("KEIBA_LEDGER_MODE", raising=False)
    pol = json.loads(L.POLICY.read_text(encoding="utf-8"))               # 決まりの写し（enforce の系統を空にし、warn を前提に確かめる）
    pol["enforce_families"] = []
    (tmp_path / "policy.json").write_text(json.dumps(pol, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("KEIBA_LEDGER_POLICY", str(tmp_path / "policy.json"))
    return tmp_path


def _enforce(monkeypatch):
    monkeypatch.setenv("KEIBA_LEDGER_MODE", "enforce")


def _g(**kw):
    base = dict(tool="t.py", family="bet_cycles", version="abc123", period=("2025-01-01", "2025-12-31"), purpose="confirm")
    base.update(kw)
    return L.guard(**base)


def test_second_confirm_is_denied_even_if_output_dir_changes(led, monkeypatch, tmp_path):
    _enforce(monkeypatch)
    out1 = tmp_path / "out1"
    with _g() as tok:
        tok.finish(result_path=out1)
    # 出力先（result_path・環境変数）は鍵に入らない: 同じ対象・同じ期間の 2 回目は拒否
    monkeypatch.setenv("PTRAIN_OUT", str(tmp_path / "other"))
    with pytest.raises(L.LedgerDenied):
        _g()
    rows = L.load_rows()
    assert sorted(r["decision"] for r in rows if r.get("kind") == "start") == ["allowed", "denied"]   # 順によらない


def test_second_confirm_is_warned_in_warn_mode(led, capsys):
    _g().finish()
    tok = _g()
    assert tok.decision.decision == "warned"
    assert "confirm_again" in tok.decision.violations[0]
    assert "決まりの外" in capsys.readouterr().err


def test_shifted_window_overlapping_counts_as_same_period(led, monkeypatch):
    _enforce(monkeypatch)
    _g(period=("2025-01-01", "2025-06-30")).finish()
    with pytest.raises(L.LedgerDenied):
        _g(period=("2025-06-30", "2025-12-31"))          # 1 日だけ重なる
    _g(period=("2026-01-05", "2026-03-31")).finish()     # 重ならない（別の期間）は通る


def test_new_version_same_family_is_downgraded_to_eval_only(led, monkeypatch):
    _enforce(monkeypatch)
    _g(version="v1").finish()
    tok = _g(version="v2")
    assert tok.purpose == "eval_only" and tok.decision.decision == "downgraded"
    assert tok.summary()["use_for_decision"] is False
    assert tok.summary()["seen_family"] == 1


def test_eval_only_then_confirm_same_target_downgrades(led, monkeypatch):
    _enforce(monkeypatch)
    _g(purpose="eval_only", reason="下見ではなく報告のため").finish()
    assert _g().purpose == "eval_only"


def test_select_on_sealed_is_denied_and_dev_period_is_free(led, monkeypatch):
    _enforce(monkeypatch)
    with pytest.raises(L.LedgerDenied):
        _g(purpose="select", period=("2024-01-01", "2026-03-31"))
    tok = _g(purpose="select", period=("2016-01-01", "2021-12-31"))
    assert tok.id is None and tok.decision.decision == "skipped"          # 開発の期間は記録しない
    tok = _g(purpose="select", period=("2022-01-01", "2024-12-31"))
    assert tok.decision.decision == "allowed" and tok.id                   # 開発の検証は記録だけ


def test_eval_only_requires_reason_and_is_counted(led, monkeypatch):
    _enforce(monkeypatch)
    with pytest.raises(L.LedgerDenied):
        _g(purpose="eval_only")
    _g(purpose="eval_only", reason="報告の表のため").finish()
    _g(purpose="eval_only", reason="もう一度", version="zz").finish()
    c = L.tally(L.load_rows(), ("2025-01-01", "2025-12-31"), "bet_cycles")
    assert c["total"] == 2 and c["by_purpose"]["eval_only"] == 2


def test_forward_period_allows_repeated_eval_but_denies_select(led, monkeypatch):
    _enforce(monkeypatch)
    for _ in range(3):
        _g(family="frozen_tracking", purpose="forward", period=("2026-11-21", "2026-11-30")).finish()
    with pytest.raises(L.LedgerDenied):
        _g(family="frozen_tracking", purpose="select", period=("2026-11-21", "2026-11-30"))
    with pytest.raises(L.LedgerDenied):                                    # forward は封印の期間に使えない
        _g(family="frozen_tracking", purpose="forward", period=("2026-11-01", "2026-11-30"))


def test_another_worktree_sees_shared_copy(led, monkeypatch, tmp_path):
    """別の worktree（git の台帳が空）から呼んでも、共有の写しで拒否される。"""
    _enforce(monkeypatch)
    _g().finish()
    monkeypatch.setenv("KEIBA_LEDGER_GIT", str(tmp_path / "other_worktree" / "ledger.jsonl"))
    with pytest.raises(L.LedgerDenied):
        _g()


def test_git_copy_alone_still_counts(led, monkeypatch, tmp_path):
    """共有の写しが作り直されても、git の台帳に残った行で数える（ホストの故障に備える）。"""
    _enforce(monkeypatch)
    _g().finish()
    (tmp_path / "shared" / "ledger.jsonl").unlink()
    with pytest.raises(L.LedgerDenied):
        _g()


def test_enforce_without_shared_dir_is_denied(led, monkeypatch, tmp_path):
    _enforce(monkeypatch)
    monkeypatch.setenv("KEIBA_LEDGER_SHARED", str(tmp_path / "nowhere" / "ledger.jsonl"))
    with pytest.raises(L.LedgerDenied):
        _g()


def test_env_cannot_lower_enforce(led, monkeypatch, tmp_path):
    pol = json.loads(L.POLICY.read_text(encoding="utf-8"))
    pol["enforce_families"] = ["bet_cycles"]
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(pol), encoding="utf-8")
    monkeypatch.setenv("KEIBA_LEDGER_POLICY", str(p))
    monkeypatch.setenv("KEIBA_LEDGER_MODE", "warn")
    _g().finish()
    with pytest.raises(L.LedgerDenied):
        _g()
    assert _g(family="course").decision.mode == "warn"                     # 他の系統は warn のまま


def test_crash_rerun_allowed_but_finished_run_counts(led, monkeypatch):
    _enforce(monkeypatch)
    with pytest.raises(RuntimeError):
        with _g():
            raise RuntimeError("途中で落ちた")
    with _g() as tok:                                                     # 道具が書いた error で結果が無い: 再実行を許す
        assert tok.decision.decision == "allowed"
    with pytest.raises(L.LedgerDenied):
        _g()


def test_started_without_finish_counts(led, monkeypatch):
    """終わりの行が無い（強制終了など）は 1 回と数える。"""
    _enforce(monkeypatch)
    _g()
    with pytest.raises(L.LedgerDenied):
        _g()


def test_check_dates_outside_declared_period(led, monkeypatch, capsys):
    tok = _g(period=("2026-01-05", "2026-10-12"))
    tok.check_dates(["2026-01-05", "2026-10-12"])                         # 内側は通る
    tok.check_dates(["2026-01-05", "2026-11-08"])                         # warn: 記録して続ける
    assert "はみ出して" in capsys.readouterr().err
    assert [r["violation"] for r in L.load_rows() if r.get("kind") == "violation"] == ["range_outside"]
    _enforce(monkeypatch)
    tok2 = _g(version="x2", period=("2025-01-01", "2025-12-31"))
    with pytest.raises(L.LedgerDenied):
        tok2.check_dates(["2025-03-01", "2026-01-04"])
    tok3 = _g(version="x3", family="course", period=("2025-01-01", "2025-12-31"))
    tok3.check_race_ids(["202501010101", "202509010101"])
    with pytest.raises(L.LedgerDenied):
        tok3.check_race_ids(["202501010101", "202605010101"])


def test_frozen_dirty_is_violation(led, monkeypatch, tmp_path):
    _enforce(monkeypatch)
    f = tmp_path / "frozen.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(L.LedgerDenied, match="frozen_dirty"):
        _g(frozen_paths=[f])
    tracked = ROOT / "config/settings.yaml"                                # git 管理で変更の無いファイル
    tok = _g(version="clean", frozen_paths=[tracked])
    assert not any("frozen_dirty" in v for v in tok.decision.violations)


def test_concurrent_writes_do_not_break_lines(led):
    def w(i):
        for j in range(20):
            L.write_row({"id": f"c{i}-{j}", "kind": "note", "at": "2026-11-18T00:00:00+09:00", "x": "あ" * 200}, strict_shared=False)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(L.load_rows()) == 80


def test_check_append_only():
    base = '{"id":"a"}\n{"id":"b"}\n'
    assert L.check_append_only(base, base + '{"id":"c"}\n') == []
    assert L.check_append_only(base, '{"id":"c"}\n{"id":"b"}\n{"id":"a"}\n') == []   # union の並び替えは可
    assert L.check_append_only(base, '{"id":"a"}\n{"id":"B"}\n') == ['{"id":"b"}']


def test_union_merge_attribute():
    r = subprocess.run(["git", "-C", str(ROOT), "check-attr", "merge", "config/data_access/ledger.jsonl"], capture_output=True, text=True)
    assert r.stdout.strip().endswith("union")


def test_tests_never_write_real_ledgers(monkeypatch):
    monkeypatch.delenv("KEIBA_LEDGER_SHARED", raising=False)
    monkeypatch.delenv("KEIBA_LEDGER_GIT", raising=False)
    assert L.shared_path() != L.SHARED_DEFAULT and L.git_path() != L.GIT_LEDGER


def test_policy_periods_are_contiguous():
    from datetime import timedelta
    ps = L.load_policy(L.POLICY)["periods"]
    for a, b in zip(ps, ps[1:]):
        assert L._d(a["to"]) + timedelta(days=1) == L._d(b["from"])
    assert ps[0]["from"] is None and ps[-1]["to"] is None


def test_backfill_file_is_valid():
    src = ROOT / "config/data_access/backfill_20261118.json"
    if not src.exists():
        pytest.skip("過去の分の一覧がまだ無い")
    d = json.loads(src.read_text(encoding="utf-8"))
    ids = [e["id"] for e in d["entries"]]
    assert len(ids) == len(set(ids))
    for e in d["entries"]:
        assert e["purpose"] in L.PURPOSES and e["family"] and e["evidence"], e["id"]
        L._d(e["period"][0])


def test_reversed_period_is_error(led):
    with pytest.raises(ValueError):
        _g(period=("2026-01-01", "2025-01-01"))


def test_sys_exit_zero_inside_counts(led, monkeypatch):
    _enforce(monkeypatch)
    with pytest.raises(SystemExit):
        with _g():
            raise SystemExit(0)
    with pytest.raises(L.LedgerDenied):
        _g()


def test_keyboard_interrupt_counts(led, monkeypatch):
    _enforce(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        with _g():
            raise KeyboardInterrupt
    with pytest.raises(L.LedgerDenied):
        _g()


def test_unknown_family_is_violation(led, monkeypatch):
    _enforce(monkeypatch)
    with pytest.raises(L.LedgerDenied, match="unknown_family"):
        _g(family="bet_cycles2")


def test_fit_only_in_test_and_reproduce_needs_frozen(led, monkeypatch):
    _enforce(monkeypatch)
    assert _g(family="model", purpose="fit").decision.decision == "allowed"
    with pytest.raises(L.LedgerDenied, match="fit_outside"):
        _g(family="model", purpose="fit", period=("2025-01-01", "2026-03-31"))
    with pytest.raises(L.LedgerDenied, match="reproduce_unfrozen"):
        _g(purpose="reproduce", reason="再現")


def test_env_alone_cannot_redirect_outside_tests(monkeypatch):
    monkeypatch.setattr(L, "TESTING", False)
    monkeypatch.setenv("KEIBA_LEDGER_SHARED", "/tmp/empty.jsonl")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    assert L.shared_path() == L.SHARED_DEFAULT and L.git_path() == L.GIT_LEDGER


def test_broken_last_line_is_skipped(led):
    _g().finish()
    with open(L.shared_path(), "a", encoding="utf-8") as f:
        f.write('{"id": "half')
    assert len(L.load_rows()) == 2


def test_parallel_confirms_only_one_allowed(led, monkeypatch):
    _enforce(monkeypatch)
    res = []

    def run():
        try:
            _g(version="par")
            res.append("ok")
        except L.LedgerDenied:
            res.append("denied")
    ts = [threading.Thread(target=run) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(res) == ["denied", "denied", "denied", "ok"]


def test_rules_bet_opt_second_final_is_denied(monkeypatch):
    """rules_bet_opt の final（11/18 に 1 回流した）は、印を消しても台帳で 2 回目と判定される（enforce なら拒否）。"""
    import ast

    from src.research.course import bet_opt_search as S

    P = json.loads((ROOT / "config/rules_bet_opt/prereg_final.json").read_text(encoding="utf-8"))
    period = S.ledger_period(P)
    assert period == ("2025-01-01", "2026-12-31")                          # 実際に読む群の日付の最小〜最大
    monkeypatch.setenv("KEIBA_LEDGER_MODE", "enforce")
    d = L.decide(L._read_jsonl(L.GIT_LEDGER), family="rules_bet_opt", version="prereg_final.json", period=period,
                 purpose="confirm", reason="", policy=L.load_policy(L.POLICY))
    assert any(v.startswith("confirm_again") for v in d.violations) and d.decision == "denied"
    # 台帳の確かめが、印を立てる前（cmd_final の中）にある
    src = (ROOT / "src/research/course/bet_opt_search.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "cmd_final")
    calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert "LG.guard" in calls and "BO.MARK.write_text" not in calls          # 印は guard の後の _final_run で立てる
    assert "_final_run" in calls


def test_summary_with_violation_is_not_for_decision(led):
    _g().finish()
    tok = _g()                                                             # warn: confirm_again の違反つきで通る
    assert tok.summary()["use_for_decision"] is False


def test_amend_needs_approval_and_changes_count(led, monkeypatch):
    _enforce(monkeypatch)
    _g(version="ref").finish()
    first = next(r for r in L.load_rows() if r.get("kind") == "start")
    for bad in ("", " ", "自分で書いた"):
        with pytest.raises(ValueError):
            L.amend(ref=first["id"], purpose="eval_only", reason="参考の群", approved_by=bad)
    with pytest.raises(ValueError):                                        # 承認者と直す担当が同じ
        L.amend(ref=first["id"], purpose="eval_only", reason="r", approved_by="claude-orchestrator-handover",
                session="claude-orchestrator-handover")
    for p_ in ("reproduce", "fit", "select"):                              # confirm→eval_only 以外は直せない
        with pytest.raises(ValueError):
            L.amend(ref=first["id"], purpose=p_, reason="r", approved_by="claude-orchestrator-handover")
    with pytest.raises(L.LedgerDenied):
        _g(version="ref")
    L.amend(ref=first["id"], purpose="eval_only", reason="参考の群（判定に使わない）", approved_by="claude-orchestrator-handover",
            session="ledger")
    tok = _g(version="ref")                                                # 直した後も見た回数は減らない: eval_only の後の confirm → eval_only
    assert tok.purpose == "eval_only"
    with pytest.raises(ValueError):                                        # 同じ行の 2 回目の直し
        L.amend(ref=first["id"], purpose="eval_only", reason="x", approved_by="claude-orchestrator-handover", session="ledger")
    with pytest.raises(ValueError):
        L.amend(ref="no-such-id", purpose="eval_only", reason="x", approved_by="claude-orchestrator-handover", session="ledger")


def test_amend_locked_family_and_bad_rows_are_skipped(led, monkeypatch, capsys):
    _enforce(monkeypatch)
    with _g(family="prereg_rules", period=("2025-01-01", "2025-12-31")) as tok:
        pass
    with pytest.raises(ValueError, match="直せない"):
        L.amend(ref=tok.id, purpose="eval_only", reason="r", approved_by="claude-orchestrator-handover", session="ledger")
    # 手で足した壊れた直しの行（purpose なし・承認なし）は使わない（落ちない）
    L.write_row({"id": "x-amend", "kind": "amend", "ref": tok.id, "at": "2026-11-19T00:00:00+09:00"}, strict_shared=False)
    with pytest.raises(L.LedgerDenied):
        _g(family="prereg_rules", period=("2025-01-01", "2025-12-31"))
    assert "使わない" in capsys.readouterr().err


def test_amend_cli_and_show(led, capsys):
    _g(version="cli").finish()
    rid = next(r["id"] for r in L.load_rows() if r.get("kind") == "start")
    assert L.main(["amend", "--ref", rid, "--purpose", "eval_only", "--reason", "参考", "--approved-by", "claude-orchestrator-handover",
                   "--session", "ledger"]) == 0
    assert L.main(["amend", "--ref", "nope", "--purpose", "eval_only", "--reason", "r", "--approved-by", "ユーザー"]) == 2
    L.main(["show", "--from", "2025-01-01", "--to", "2025-12-31"])
    assert "直し: confirm→eval_only" in capsys.readouterr().out


def _guard_funcs(path: Path) -> set[str]:
    """ファイルの中で guard を呼ぶ最上位の関数の名前。"""
    import ast

    t = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for fn in t.body:
        if isinstance(fn, ast.FunctionDef):
            if any(isinstance(n, ast.Call) and ast.unparse(n.func).endswith(".guard") for n in ast.walk(fn)):
                out.add(fn.name)
    return out


# guard を持つ研究の CLI と、その中で guard を呼んでよい入口の関数
_GUARD_ENTRY = {
    "src/scripts/data/bet_cycles.py": {"cmd_final", "cmd_prereg_run", "cmd_tiers2"},
    "src/scripts/data/bet_forward_ledger.py": {"_main_inner"},
    "src/scripts/data/bet_age2_premium.py": {"_run_inner"},
    "src/scripts/data/bet_r2_combine.py": {"_main_inner"},
    "src/scripts/data/bet_r2_walkthrough.py": {"_build_inner"},
    "src/research/course/bet_opt_search.py": {"cmd_final"},
    "src/research/course/prereg_rules.py": {"cmd_run"},
    "src/research/course/upset_types.py": {"cmd_run"},
    "src/research/course/course_rule_stats.py": {"_rule_stats_inner"},   # めぐ分析の表示用の数え直し（eval_only・研究の CLI）
    "src/research/graded/pop_base.py": {"main"},                 # 同上（重賞の比べる相手）
    "src/research/bet_patterns/bet_patterns_run.py": {"guard", "cmd_select_extra", "cmd_evaluate_extra"},  # ◎ からの馬券のパターン（研究の CLI。各 cmd の中だけで呼ぶ）
    "src/research/course/megu_why.py": {"_collect_inner", "cmd_graded", "cmd_perm"},   # 100% を超えない理由の診断（eval_only・研究の CLI）
    "src/research/course/megu_oikiri.py": {"_select_inner", "_final_inner"},   # 追い切りの切り口の 1 回だけの評価（eval_only・研究の CLI）
    "src/research/megu_cycles/mc_data.py": {"cmd_build", "cmd_build_rich", "cmd_build_offset", "cmd_build_bag"},   # めぐサイクルの表（reproduce・研究の CLI）
    "src/research/megu_cycles/mc_eval.py": {"confirm", "eval_only_guard"},
    "src/research/rules_council/rc_frame.py": {"build"},          # ルールの会議の全走の表（T-60 の再現のずれの元を読む。eval_only・研究の CLI）
    "src/research/rules_council/t60_check.py": {"main"},          # 本物の T-60 と再現の帯の確かめ（オッズだけ。eval_only）
    "src/research/rules_council/rc2_real.py": {"guard"},          # RC-2 の段 S'（研究の CLI。各 cmd の中だけで呼ぶ）
    "src/research/rules_council/benter_diag.py": {"cmd_volume", "cmd_sstar", "cmd_depend"},   # RC-6 の着順なしの診断（eval_only。研究の CLI）
    "src/research/rules_council/forward_union.py": {"cmd_build", "cmd_ledger"},   # RC-2 の和集合の前向きの台帳（eval_only・forward。研究の CLI）
    "src/research/megu_cycles/mc_final.py": {"cmd_final", "cmd_drift"},     # めぐサイクルの最後の 1 回の確かめ・再現の楽観の測り（eval_only）   # めぐサイクルの 2022〜24 の確かめ（confirm）・割引きと最後の確かめ（eval_only）
}
_PROD = ("src/pipeline/inference/stage2_bets.py", "src/pipeline/inference/candidate_runner.py", "src/api/stage2_live.py")


def test_guard_is_only_in_research_cli_entrypoints():
    """enforce（拒否で SystemExit）にした系統の guard は、研究の CLI の入口からだけ呼ぶ。本番・当日の経路
    （候補の影のランナー・第 2 段の推奨 stage2_bets・API・cron）は、guard のある関数・モジュールを呼ばない。"""
    import ast

    have = set()
    for p in list((ROOT / "src").rglob("*.py")):
        rel = str(p.relative_to(ROOT))
        if rel != "src/ops/data_ledger.py" and _guard_funcs(p):
            have.add(rel)
            assert _guard_funcs(p) <= _GUARD_ENTRY.get(rel, set()), (rel, _guard_funcs(p))
    assert have == set(_GUARD_ENTRY), have ^ set(_GUARD_ENTRY)            # guard を呼ぶファイルを足したら、ここに入口を書く
    entry_names = set().union(*_GUARD_ENTRY.values())
    for f in _PROD:
        t = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        used = {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "bc"}
        used |= {a.name for n in ast.walk(t) if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("bet_cycles") for a in n.names}
        assert not (used & entry_names), (f, used & entry_names)
    mods = [Path(k).stem for k in _GUARD_ENTRY if not k.endswith("bet_cycles.py")]
    # 本番の経路（src の推論・API・ops、cron とサーバのシェル）は、bet_cycles 以外の guard を持つ CLI と台帳を参照しない
    for d in ("src/api", "src/pipeline", "src/ops", "scripts/cron", "scripts/ops", "scripts/server"):
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix not in (".py", ".sh") or str(p.relative_to(ROOT)) == "src/ops/data_ledger.py":
                continue
            t = p.read_text(encoding="utf-8", errors="ignore")
            assert "data_ledger" not in t, p
            assert not any(m in t for m in mods), (p, [m for m in mods if m in t])
            if p.suffix == ".sh":
                assert "bet_cycles" not in t, p


def test_importing_bet_cycles_does_not_load_ledger():
    r = subprocess.run([sys.executable, "-c", "import sys; import src.pipeline.inference.stage2_bets, src.scripts.data.bet_cycles, "
                        "src.pipeline.inference.candidate_runner, src.api.stage2_live; "
                        "print('src.ops.data_ledger' in sys.modules)"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr[-500:]
