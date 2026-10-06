"""data_health の設定（DATA_HEALTH_*）と環境別の結果管理のテスト。"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from src.data_health import store
from src.data_health.__main__ import main
from src.data_health.config import load_settings, parse_levels
from src.data_health.runner import JST, run_health
from src.data_health.spec import ARTIFACTS, RACE_CATEGORIES, level_overrides
from tests.data_health.test_data_health import FakeStorage, _dt, _ids, _write_race_list, stg_env  # noqa: F401


# ── 設定 ────────────────────────────────────────────────────────────────

def test_defaults_follow_profile():
    st = load_settings({}, profile="stg")
    assert st.env == "stg" and st.levels == {} and st.fail_on == "fail" and st.since is None


def test_env_vars_and_env_specific_levels_win():
    env = {
        "DATA_HEALTH_SINCE": "2024-01-01", "DATA_HEALTH_UNTIL": "2026-12-31", "DATA_HEALTH_FAIL_ON": "warn",
        "DATA_HEALTH_LEVELS": "race_barometer=required,C07=optional",
        "DATA_HEALTH_LEVELS_STG": "race_barometer=optional",      # 環境別が汎用より優先
        "DATA_HEALTH_LEVELS_PROD": "C07=required",                # 別環境の指定は無視される
        "DATA_HEALTH_SKIP": "infra.redis",
        "DATA_HEALTH_HORSE_RACES_MAX": "50",
    }
    st = load_settings(env, profile="stg")
    assert st.since == date(2024, 1, 1) and st.until == date(2026, 12, 31) and st.fail_on == "warn"
    assert st.levels == {"race_barometer": "optional", "C07": "optional", "infra.redis": "skip"}
    assert st.horse_races_max == 50


def test_profile_defaults_to_data_health_env_then_keiba_env(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "stg")
    assert load_settings({}).env == "stg"
    assert load_settings({"DATA_HEALTH_ENV": "prod"}).env == "prod"


@pytest.mark.parametrize("env", [
    {"DATA_HEALTH_SINCE": "2024/01/01"}, {"DATA_HEALTH_FAIL_ON": "boom"}, {"DATA_HEALTH_LEVELS": "race_odds"},
    {"DATA_HEALTH_LEVELS": "race_odds=critical"}, {"DATA_HEALTH_HORSE_RACES_MAX": "abc"}, {"DATA_HEALTH_HORSE_PAST_DAYS": "9999"},
])
def test_invalid_values_raise_clear_errors(env):
    with pytest.raises(ValueError):
        load_settings(env, profile="stg")


def test_unknown_ids_are_reported_not_applied():
    st = load_settings({"DATA_HEALTH_LEVELS": "race_oddz=required,race_odds=optional"}, profile="stg")
    assert st.levels == {"race_odds": "optional"}
    assert "race_oddz" in st.warnings[0]
    assert parse_levels("") == {}


def test_level_overrides_apply_and_restore():
    cat = next(c for c in RACE_CATEGORIES if c.name == "race_barometer")
    art = next(a for a in ARTIFACTS if a.id == "C07")
    before = (cat.level["stg"], art.level["stg"])
    with level_overrides("stg", {"race_barometer": "required", "C07": "skip"}):
        assert (cat.level["stg"], art.level["stg"]) == ("required", "skip")
        assert cat.level["prod"] != "required" or before[0] == "required"   # 他環境は変えない
    assert (cat.level["stg"], art.level["stg"]) == before


def test_override_changes_severity_in_report(stg_env):
    day = _ids([1], year="2026")
    _write_race_list(stg_env, "20261004", day)
    full = {c.name: {"2026": day} for c in RACE_CATEGORIES if c.name not in ("race_barometer",)}
    args = dict(storage=FakeStorage(full), now=_dt(2026, 10, 20), since=date(2026, 1, 1), until=date(2026, 10, 25),
                root=stg_env, infra=False, horses=False, actual_env="stg")
    base = run_health(env="stg", **args)
    assert next(f for f in base["findings"] if "race_barometer" in f["message"])["severity"] == "warn"
    cfg = load_settings({"DATA_HEALTH_LEVELS": "race_barometer=required"}, profile="stg")
    strict = run_health(settings=cfg, **args)
    assert next(f for f in strict["findings"] if "race_barometer" in f["message"])["severity"] == "fail"
    assert strict["settings"]["levels"] == {"race_barometer": "required"}
    skipped = run_health(settings=load_settings({"DATA_HEALTH_SKIP": "race_barometer"}, profile="stg"), **args)
    assert not any("race_barometer" in f["message"] for f in skipped["findings"])
    assert next(c for c in base["race_coverage"]["categories"] if c["name"] == "race_barometer")["level"] == "recommended"


def test_infra_override_downgrades_and_skips(monkeypatch):
    from src.data_health.checks import apply_overrides, check

    res = [check("infra.redis", "Redis", "fail", "x"), check("infra.disk", "disk", "warn", "y"), check("infra.db", "db", "ok", "z")]
    out = apply_overrides(res, {"infra.redis": "optional", "infra.disk": "skip", "infra.db": "required"})
    assert [c["status"] for c in out] == ["info", "skip", "ok"]


def test_profile_different_from_actual_skips_infra_and_plan(stg_env):
    day = _ids([1], year="2026")
    _write_race_list(stg_env, "20261004", day)
    rep = run_health(env="stg", actual_env="dev", storage=FakeStorage({"race_shutuba": {"2026": day}}), now=_dt(2026, 10, 20),
                     since=date(2026, 1, 1), until=date(2026, 10, 25), root=stg_env, infra=True, horses=False)
    assert rep["meta"]["key"] == "stg@dev"
    assert [c["status"] for c in rep["checks"]] == ["skip"]
    assert rep["plan"]["counts"]["jobs"] == 0 and rep["race_coverage"]["universe"]["by_confidence"]["inferred"] == 0


# ── 環境別の結果管理 ─────────────────────────────────────────────────────

def _report(stg_env, when, missing=False, **kw):
    day = _ids([1], year="2026")
    _write_race_list(stg_env, "20261004", day)
    listing = {c.name: {"2026": day} for c in RACE_CATEGORIES}
    if missing:
        listing["race_result"] = {"2026": set()}
    no_artifacts = load_settings({"DATA_HEALTH_SKIP": ",".join(a.id for a in ARTIFACTS)}, profile="stg")   # 派生データは対象外にして race だけ見る
    return run_health(settings=no_artifacts, actual_env="stg", storage=FakeStorage(listing), now=when, since=date(2026, 1, 1),
                      until=date(2026, 10, 25), root=stg_env, infra=False, horses=False, **kw)


def test_results_are_kept_per_env_with_history_and_diff(stg_env):
    out = stg_env / "health"
    r1 = _report(stg_env, _dt(2026, 10, 20, 9))
    assert r1["summary"]["overall"] == "ok"
    store.save(r1, out)
    r2 = _report(stg_env, _dt(2026, 10, 21, 9), missing=True)
    paths = store.save(r2, out)
    assert r2["diff"]["overall_before"] == "ok" and r2["diff"]["counts_delta"]["fail"] >= 1
    assert any("race_result" in m for m in r2["diff"]["new"])
    assert (out / "stg" / "latest.json").exists() and not (out / "dev").exists()
    hist = store.read_history(out / "stg")
    assert [h["overall"] for h in hist] == ["ok", "fail"]
    idx = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [e["key"] for e in idx] == ["stg"] and idx[0]["overall"] == "fail"
    assert "stg" in paths["index"].read_text(encoding="utf-8") and "推移" in paths["index"].read_text(encoding="utf-8")
    r3 = _report(stg_env, _dt(2026, 10, 22, 9))
    store.save(r3, out)
    assert any("race_result" in m for m in r3["diff"]["resolved"])


def test_import_merges_other_pc_result_and_keeps_newest(stg_env, tmp_path):
    out = tmp_path / "health"
    newer, older = _report(stg_env, _dt(2026, 10, 21, 9), missing=True), _report(stg_env, _dt(2026, 10, 20, 9))
    f_new, f_old = tmp_path / "new.json", tmp_path / "old.json"
    f_new.write_text(json.dumps(newer), encoding="utf-8")
    f_old.write_text(json.dumps(older), encoding="utf-8")
    assert store.import_report(f_new, out)["latest_updated"] is True
    res = store.import_report(f_old, out)
    assert res["latest_updated"] is False                       # 古い結果は最新を上書きしない
    assert json.loads((out / "stg" / "latest.json").read_text(encoding="utf-8"))["generated_at"] == newer["generated_at"]
    assert len(store.read_history(out / "stg")) == 2
    assert (out / "stg" / "latest.html").exists()
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        store.import_report(bad, out)


def test_envs_are_separated_by_key(stg_env, tmp_path):
    out = tmp_path / "health"
    r = _report(stg_env, _dt(2026, 10, 20, 9))
    store.save(r, out)
    dry = run_health(env="stg", actual_env="dev", storage=FakeStorage({}), now=_dt(2026, 10, 20), since=date(2026, 1, 1),
                     until=date(2026, 10, 25), root=stg_env, infra=False, horses=False)
    store.save(dry, out)
    keys = [e["key"] for e in store.collect_entries(out)]
    assert keys == ["stg", "stg@dev"]                           # dev PC の dry-run が本物の stg を上書きしない


# ── CLI ─────────────────────────────────────────────────────────────────

def test_cli_end_to_end_in_dev(tmp_path, monkeypatch, capsys):
    from src.scripts.data import make_dev_mock

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pr"))
    monkeypatch.setenv("KEIBA_CALCULATED_DATA_DIR", str(tmp_path / "calc"))
    monkeypatch.setenv("DATA_HEALTH_OUT_DIR", str(tmp_path / "health"))
    monkeypatch.setenv("DATA_HEALTH_SKIP", "infra.db,infra.redis")
    monkeypatch.setenv("DATA_HEALTH_FAIL_ON", "warn")
    make_dev_mock.generate(date.today(), make_dev_mock.dev_mock_root())
    code = main(["--no-horses", "--quiet"])
    assert (tmp_path / "health" / "dev" / "latest.json").exists() and (tmp_path / "health" / "index.html").exists()
    rep = json.loads((tmp_path / "health" / "dev" / "latest.json").read_text(encoding="utf-8"))
    assert rep["settings"]["sources"]["out_dir"] == "env" and rep["settings"]["levels"]["infra.redis"] == "skip"
    assert next(c for c in rep["checks"] if c["id"] == "infra.redis")["status"] == "skip"
    assert code in (0, 2)
    # CLI 引数は環境変数より優先される
    main(["--no-horses", "--no-infra", "--quiet", "--out-dir", str(tmp_path / "other")])
    assert (tmp_path / "other" / "dev" / "latest.json").exists()
    assert main(["--index-only"]) == 0
    with pytest.raises(SystemExit):
        monkeypatch.setenv("DATA_HEALTH_FAIL_ON", "nope")
        main(["--quiet"])


# ── スキーマ確認（全環境で同一か・適合率）────────────────────────────────────

def test_schema_report_counts_conformance_and_flags_failures(stg_env):
    from src.data_health import schema_check

    ids = {"202605010101", "202605010102"}
    bad = {("race_odds", "202605010102")}
    storage = FakeStorage({"race_odds": {"2026": ids}}, bad=bad)
    present = {"race_odds": {"2026": {k: 1.0 + i for i, k in enumerate(sorted(ids))}}}
    rep = schema_check.schema_report("stg", storage, present, sample=10)
    r = rep["categories"]["race_odds"]
    assert (r["sampled"], r["passed"]) == (2, 1) and any("top_missing:race_id" in m for m in r["issues"])
    sev = {f["severity"] for f in schema_check.schema_findings("stg", rep) if "race_odds" in f["message"]}
    assert sev == {"warn"}                                                      # race_odds は recommended → warn
    assert rep["fingerprint"] == schema_check.schemas.schema_fingerprint()
    assert all(u["status"] != "missing" for u in rep["undefined"])


def test_report_carries_schema_fingerprint_and_index_flags_mismatch(stg_env, tmp_path):
    out = tmp_path / "health"
    a = _report(stg_env, _dt(2026, 10, 20, 9))
    assert a["schema"]["fingerprint"] == a["meta"]["schema_fingerprint"]
    store.save(a, out)
    other = json.loads(json.dumps(_report(stg_env, _dt(2026, 10, 21, 9))))
    other["env"], other["meta"] = "prod", {**other["meta"], "env": "prod", "key": "prod", "actual_env": "prod", "schema_fingerprint": "deadbeef0000"}
    f = tmp_path / "prod.json"
    f.write_text(json.dumps(other), encoding="utf-8")
    store.import_report(f, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "一致していません" in html and "deadbeef0000" in html


def test_schema_sample_zero_disables_download(stg_env):
    cfg = load_settings({"DATA_HEALTH_SCHEMA_SAMPLE": "0"}, profile="stg")
    rep = run_health(settings=cfg, actual_env="stg", storage=FakeStorage({}), now=_dt(2026, 10, 20), since=date(2026, 1, 1),
                     until=date(2026, 10, 25), root=stg_env, infra=False, horses=False)
    assert rep["schema"]["categories"] == {} and rep["schema"]["sample"] == 0
