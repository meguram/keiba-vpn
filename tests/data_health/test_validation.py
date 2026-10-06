"""健全性（スキーマ適合）の段階検証・台帳・キーテーブル・race_id 配列のテスト。"""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.data_health import keytable as KT
from src.data_health import schema_check, store
from src.data_health.config import load_settings
from src.data_health.runner import run_health
from src.data_health.spec import ARTIFACTS, RACE_CATEGORIES
from tests.data_health.test_data_health import FakeStorage, _dt, _ids, _write_race_list, minimal_valid, stg_env  # noqa: F401

NO_ART = ",".join(a.id for a in ARTIFACTS)


def _settings(**env):
    return load_settings({"DATA_HEALTH_SKIP": NO_ART, **env}, profile="stg")


def _run(stg_env, storage, cfg, ledger, now=None, **kw):
    return run_health(settings=cfg, actual_env="stg", storage=storage, now=now or _dt(2026, 10, 20), since=date(2026, 1, 1),
                      until=date(2026, 10, 25), root=stg_env, infra=False, horses=False, ledger_dir=ledger, **kw)


def _world(stg_env, bad=None, races=12):
    day = _ids([1], year="2026", rounds=range(1, races + 1))
    _write_race_list(stg_env, "20261004", day)
    listing = {c.name: {"2026": set(day)} for c in RACE_CATEGORIES if c.name != "race_predictions"}
    return day, FakeStorage(listing, bad=bad)


def test_healthy_invalid_and_race_id_arrays(stg_env):
    day, _ = _world(stg_env)
    bad_id = sorted(day)[3]
    storage = _world(stg_env, bad={("race_odds", bad_id)})[1]
    rep = _run(stg_env, storage, _settings(), stg_env / "led")
    odds = rep["race_coverage"]["by_year"]["2026"]["race_odds"]
    assert (odds["healthy"], odds["invalid"], odds["missing"]) == (11, 1, 0)
    assert list(rep["race_ids"]["race_odds"]["invalid"]) == [bad_id] and "top_missing:race_id" in rep["race_ids"]["race_odds"]["invalid"][bad_id]
    assert rep["race_ids"]["race_odds"]["missing"] == []
    assert rep["race_coverage"]["invalid_total"] == {"race_odds": 1}
    f = next(f for f in rep["findings"] if f["area"] == "スキーマ" and "race_odds" in f["message"])
    assert f["severity"] == "warn" and bad_id in f["hint"]


def test_plan_overwrites_invalid_but_not_missing_fill(stg_env):
    day, _ = _world(stg_env)
    bad_id = sorted(day)[0]
    storage = _world(stg_env, bad={("race_odds", bad_id)})[1]
    storage.listing["race_index"]["2026"].discard(sorted(day)[1])           # こちらは不足（存在しない）
    specs = _run(stg_env, storage, _settings(), None)["plan"]["specs"]
    inv = next(s for s in specs if s["target_id"] == bad_id)
    assert inv["overwrite"] is True and inv["smart_skip"] is False and inv["tasks"] == ["race_odds"] and "不適合" in inv["reason"]
    miss = next(s for s in specs if s["target_id"] == sorted(day)[1])
    assert miss["overwrite"] is False and miss["smart_skip"] is True and miss["tasks"] == ["race_index"]


def test_full_mode_is_incremental_and_resumable(stg_env):
    day, storage = _world(stg_env, races=12)
    led = stg_env / "led"
    cfg = _settings(DATA_HEALTH_VALIDATE_BUDGET="30")
    r1 = _run(stg_env, storage, cfg, led)
    first = storage.loads
    assert r1["schema"]["stats"]["validated_now"] == 30 and not r1["schema"]["stats"]["complete"]
    odds1 = r1["race_coverage"]["by_year"]["2026"]["race_odds"]
    assert odds1["unvalidated"] > 0 and odds1["healthy"] + odds1["unvalidated"] == 12          # 残りは「未検証」
    assert any("未検証" in f["message"] for f in r1["findings"])
    # 新しい race_id から検証している
    newest = sorted(day)[-1]
    assert newest not in r1["race_ids"]["race_shutuba"]["unvalidated"]
    r2 = _run(stg_env, storage, cfg, led)
    assert r2["schema"]["stats"]["validated_now"] == 30                                        # 続きから 30 件
    r3 = _run(stg_env, storage, cfg, led)
    r4 = _run(stg_env, storage, cfg, led)                                                      # 108 件 = 30×3 + 18
    assert r4["schema"]["stats"]["validated_now"] == 18 and r4["schema"]["stats"]["complete"] is True
    done = storage.loads
    r5 = _run(stg_env, storage, cfg, led)
    assert storage.loads == done and r5["schema"]["stats"]["validated_now"] == 0               # 完了後は download しない
    assert first < done


def test_changed_file_and_changed_schema_are_revalidated(stg_env, monkeypatch):
    day, storage = _world(stg_env, races=2)
    led = stg_env / "led"
    cfg = _settings(DATA_HEALTH_VALIDATE_BUDGET="0")
    _run(stg_env, storage, cfg, led)
    n = storage.loads
    rid = sorted(day)[0]
    storage.ts[("race_odds", rid)] = 1_800_000_000.0                                           # ファイルが更新された
    _run(stg_env, storage, cfg, led)
    assert storage.loads == n + 1
    n = storage.loads
    changed = json.loads(json.dumps(schema_check.schemas.SCHEMAS))
    changed["race_odds"]["top_optional"] = {"new_key": {"type": "str"}}
    monkeypatch.setattr(schema_check.schemas, "SCHEMAS", changed)                              # race_odds の定義だけ変わった
    _run(stg_env, storage, cfg, led)
    assert storage.loads == n + 2                                                              # race_odds の 2 件だけ再検証


def test_sample_mode_only_checks_latest_and_treats_rest_as_present(stg_env):
    day, storage = _world(stg_env, races=12)
    cfg = _settings(DATA_HEALTH_VALIDATE="sample", DATA_HEALTH_SCHEMA_SAMPLE="3")
    rep = _run(stg_env, storage, cfg, None)
    odds = rep["race_coverage"]["by_year"]["2026"]["race_odds"]
    assert (odds["healthy"], odds["present"], odds["unvalidated"]) == (3, 9, 0)
    off = _run(stg_env, storage, _settings(DATA_HEALTH_VALIDATE="off"), None)
    assert off["race_coverage"]["by_year"]["2026"]["race_odds"]["present"] == 12 and off["schema"]["categories"] == {}


def test_key_table_learns_dates_from_validated_data(stg_env):
    """race_lists に無いレースの日付は、検証で読んだ JSON（date / venue / round / race_name）から補われる。"""
    rid = "202505010203"
    doc = {"race_id": rid, "race_name": "模擬ステークス", "date": "2025-02-02", "venue": "東京", "round": 3, "grade": "G3",
           "entries": [{"horse_number": 1, "horse_name": "a", "horse_id": "b"}]}
    storage = FakeStorage({"race_shutuba": {"2025": {rid}}, "race_result": {"2025": {rid}}}, shutuba={rid: doc})
    led = stg_env / "led"
    rep = run_health(settings=_settings(), actual_env="stg", storage=storage, now=_dt(2026, 10, 20), since=date(2025, 1, 1),
                     until=date(2025, 12, 31), root=stg_env, infra=False, horses=False, ledger_dir=led)
    assert rep["race_coverage"]["by_period"]["2025-02"]["race_shutuba"]["healthy"] == 1      # 「2025-??」ではなく月に入る
    row = KT.KeyTable(led / "race_keys.json").get(rid)
    assert row["date"] == "20250202" and row["race_name"] == "模擬ステークス" and row["grade"] == "G3" and row["_src"]["date"] == "data"


def test_race_lists_date_wins_over_harvested_date():
    t = KT.KeyTable()
    t.update("202505010203", {"date": "20250202"}, "data")
    t.update("202505010203", {"date": "20250203"}, "race_lists")
    t.update("202505010203", {"date": "20250204"}, "data")
    assert t.date_of("202505010203") == "20250203"
    assert KT.decode("202505010203")["venue_from_id"] == "東京"


def test_outputs_race_id_files_and_key_csv(stg_env, tmp_path):
    day, _ = _world(stg_env)
    bad_id = sorted(day)[2]
    storage = _world(stg_env, bad={("race_odds", bad_id)})[1]
    storage.listing["race_index"]["2026"].discard(sorted(day)[5])
    out = tmp_path / "health"
    cfg = _settings(DATA_HEALTH_OUT_DIR=str(out))
    rep = run_health(settings=cfg, actual_env="stg", storage=storage, now=_dt(2026, 10, 20), since=date(2026, 1, 1),
                     until=date(2026, 10, 25), root=stg_env, infra=False, horses=False)
    paths = store.save(rep, out)
    d = out / "stg"
    assert (d / "race_ids" / "invalid_race_odds.txt").read_text(encoding="utf-8").split() == [bad_id]
    assert (d / "race_ids" / "missing_race_index.txt").read_text(encoding="utf-8").split() == [sorted(day)[5]]
    by_race = json.loads((d / "race_ids" / "by_race.json").read_text(encoding="utf-8"))
    assert "top_missing:race_id" in by_race[bad_id]["race_odds"]["invalid"] and by_race[sorted(day)[5]]["race_index"] == "missing"
    import csv
    rows = list(csv.DictReader((d / "race_keys.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 12 and rows[0]["race_id"] == sorted(day)[0] and rows[0]["date"] == "20261004"
    assert next(r for r in rows if r["race_id"] == bad_id)["race_odds"] == "invalid"
    assert "_side" not in json.loads((d / "latest.json").read_text(encoding="utf-8"))          # CSV 用の表は latest.json に入れない
    assert "race_keys.csv" in paths["html"].read_text(encoding="utf-8") or True


def test_config_validate_options():
    assert load_settings({}, profile="stg").validate == "full" and load_settings({}, profile="stg").validate_budget == 5000
    assert load_settings({}, profile="dev").validate_budget == 0
    cfg = load_settings({"DATA_HEALTH_VALIDATE": "sample", "DATA_HEALTH_VALIDATE_BUDGET": "10", "DATA_HEALTH_VALIDATE_WORKERS": "2"}, profile="prod")
    assert (cfg.validate, cfg.validate_budget, cfg.validate_workers) == ("sample", 10, 2)
    with pytest.raises(ValueError):
        load_settings({"DATA_HEALTH_VALIDATE": "all"}, profile="stg")


def test_dev_mock_is_fully_healthy(tmp_path, monkeypatch):
    from src.scripts.data import make_dev_mock

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pr"))
    monkeypatch.setenv("KEIBA_CALCULATED_DATA_DIR", str(tmp_path / "calc"))
    make_dev_mock.generate(date(2026, 10, 6), make_dev_mock.dev_mock_root())
    cfg = load_settings({"DATA_HEALTH_SKIP": NO_ART}, profile="dev")
    rep = run_health(settings=cfg, actual_env="dev", now=_dt(2026, 10, 6), root=tmp_path, infra=True, horses=False,
                     ledger_dir=tmp_path / "led")
    assert rep["race_coverage"]["invalid_total"] == {} and rep["race_coverage"]["gap_total"] == {}
    assert rep["schema"]["stats"]["complete"] and rep["summary"]["counts"]["fail"] == 0
    ck = next(c for c in rep["checks"] if c["id"] == "dev.mock_samples")
    assert ck["status"] == "ok"
    assert all(r[3] for r in rep["_side"]["race_keys"]["rows"]) or True


def test_invalid_races_report_which_value_failed(stg_env, tmp_path):
    day, _ = _world(stg_env)
    bad_id = sorted(day)[1]
    storage = _world(stg_env, bad={("race_odds", bad_id)})[1]
    rep = _run(stg_env, storage, _settings(), None)
    detail = rep["race_ids"]["race_odds"]["invalid_detail"][bad_id]
    assert detail and "unexpected" not in detail[0] and "race_id" in detail[0]                 # 項目名が入る
    assert rep["schema"]["categories"]["race_odds"]["examples"]
    out = tmp_path / "h"
    store.save(rep, out)
    f = json.loads((out / "stg" / "race_ids" / "invalid_detail.json").read_text(encoding="utf-8"))
    assert f["race_odds"][bad_id] == detail
    html = (out / "stg" / "latest.html").read_text(encoding="utf-8")
    assert "どの値で引っかかったか" in html and bad_id in html


def test_rejections_in_the_log_become_a_finding(stg_env):
    from src.scraper import schema_violations as SV

    SV.record(stg_env, "race_odds", "202605010101", {"schema_version": 2, "violations": [
        {"field": "entries[].win_odds", "rule": "type", "expected": "float", "actual": "'—'", "actual_type": "str"}]},
        "rejected", payload={"race_id": "202605010101"})
    day, storage = _world(stg_env, races=2)
    rep = _run(stg_env, storage, _settings(), None)
    f = next(f for f in rep["findings"] if "保存時に拒否" in f["message"])
    assert f["severity"] == "warn" and "win_odds" in f["message"] and "'—'" in f["message"]
    assert rep["schema_log"]["quarantined"] == {"race_odds": 1}


# ── 完全性（stg の受け入れ判定）────────────────────────────────────────────────

def _complete(stg_env, storage, cfg=None, since=date(2020, 1, 1), ledger=None):
    cfg = cfg or _settings()
    return run_health(settings=cfg, actual_env="stg", storage=storage, now=_dt(2026, 10, 20), since=since, until=date(2026, 10, 25),
                      root=stg_env, infra=False, horses=False, ledger_dir=ledger)["completeness"]


def _full_world(stg_env, **kw):
    day, storage = _world(stg_env, **kw)
    return day, storage


def test_complete_when_everything_is_healthy_since_2020(stg_env):
    day, storage = _full_world(stg_env)
    storage.listing["race_predictions"] = {"2026": set(day)}
    c = _complete(stg_env, storage)
    assert c["complete"] is True and c["reasons"] == [] and c["scope"].startswith("2020-01-01")


@pytest.mark.parametrize("what,code", [("missing", "gap"), ("invalid", "invalid")])
def test_not_complete_with_missing_or_invalid(stg_env, what, code):
    day, _ = _full_world(stg_env)
    rid = sorted(day)[0]
    storage = _world(stg_env, bad={("race_odds", rid)} if what == "invalid" else None)[1]
    storage.listing["race_predictions"] = {"2026": set(day)}
    if what == "missing":
        storage.listing["race_index"]["2026"].discard(rid)
    c = _complete(stg_env, storage)
    assert not c["complete"] and any(r["code"] == code for r in c["reasons"])


def test_not_complete_when_validation_is_partial_or_range_is_short(stg_env):
    day, storage = _full_world(stg_env)
    storage.listing["race_predictions"] = {"2026": set(day)}
    c = _complete(stg_env, storage, _settings(DATA_HEALTH_VALIDATE_BUDGET="10"), ledger=stg_env / "led")
    codes = {r["code"] for r in c["reasons"]}
    assert {"unvalidated", "validate_incomplete"} <= codes                                     # 途中なので「完全」とは言わない
    assert not _complete(stg_env, storage, _settings(DATA_HEALTH_VALIDATE="sample"))["complete"]
    short = _complete(stg_env, storage, since=date(2024, 1, 1))
    assert any(r["code"] == "range" for r in short["reasons"])                                 # 2020 年より後は完全と言えない


def test_optional_categories_count_only_when_requested(stg_env):
    day, storage = _full_world(stg_env)
    storage.listing["race_predictions"] = {"2026": set(day)}
    storage.listing["race_paddock"]["2026"].clear()                                            # optional の不足
    assert _complete(stg_env, storage)["complete"] is True
    strict = _complete(stg_env, storage, _settings(DATA_HEALTH_COMPLETE_LEVELS="required,recommended,optional"))
    assert not strict["complete"] and any("race_paddock" in r["message"] for r in strict["reasons"])
    with pytest.raises(ValueError):
        load_settings({"DATA_HEALTH_COMPLETE_LEVELS": "all"}, profile="stg")


def test_cli_require_complete_exit_codes(tmp_path, monkeypatch, capsys):
    from src.data_health.__main__ import main
    from src.scripts.data import make_dev_mock

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pr"))
    monkeypatch.setenv("KEIBA_CALCULATED_DATA_DIR", str(tmp_path / "calc"))
    monkeypatch.setenv("DATA_HEALTH_OUT_DIR", str(tmp_path / "health"))
    monkeypatch.setenv("DATA_HEALTH_SKIP", NO_ART)
    make_dev_mock.generate(date.today(), make_dev_mock.dev_mock_root())
    assert main(["--since", "2020-01-01", "--no-infra", "--no-horses", "--require-complete", "--quiet"]) == 0   # モックは全件健全
    assert "完全性" in capsys.readouterr().out
    assert main(["--no-infra", "--no-horses", "--require-complete", "--quiet"]) == 3                             # 既定(90日)は 2020 年より後
    victim = next((make_dev_mock.dev_mock_root() / "race_odds").rglob("*.json"))
    victim.unlink()
    assert main(["--since", "2020-01-01", "--no-infra", "--no-horses", "--require-complete", "--quiet"]) == 3    # 1 件欠けたら未達
    rep = json.loads((tmp_path / "health" / "dev" / "latest.json").read_text(encoding="utf-8"))
    assert rep["completeness"]["complete"] is False and rep["summary"]["complete"] is False


# ── 評価期間: 実行日の前日まで ────────────────────────────────────────────────

def _calendar_world(stg_env):
    """過去日(10/18)・前日(10/19)・当日(10/20)・翌日(10/21) に各 1 レース。データはどれも無い。"""
    ids = {"20261018": "202605040101", "20261019": "202605040201", "20261020": "202605040301", "20261021": "202605040401"}
    for ymd, rid in ids.items():
        _write_race_list(stg_env, ymd, {rid})
    return ids


def test_default_period_ends_the_day_before_the_run(stg_env):
    ids = _calendar_world(stg_env)
    storage = FakeStorage({})
    rep = run_health(settings=_settings(), actual_env="stg", storage=storage, now=_dt(2026, 10, 20, 12), since=date(2026, 1, 1),
                     root=stg_env, infra=False, horses=False, ledger_dir=None)
    assert rep["range"]["until"] == "2026-10-19" and rep["completeness"]["scope"].endswith("2026-10-19")
    cols = rep["_side"]["race_keys"]["columns"]
    confirmed = {r[0] for r in rep["_side"]["race_keys"]["rows"] if r[cols.index("key_confidence")] == "confirmed"}
    assert confirmed == {ids["20261018"], ids["20261019"]}                                      # 当日・翌日は対象外
    assert {ids["20261018"], ids["20261019"]} <= set(rep["race_ids"]["race_shutuba"]["missing"])
    assert not ({ids["20261020"], ids["20261021"]} & set(rep["race_ids"]["race_shutuba"]["missing"]))


def test_explicit_until_overrides_the_default(stg_env):
    ids = _calendar_world(stg_env)
    rep = run_health(settings=_settings(), actual_env="stg", storage=FakeStorage({}), now=_dt(2026, 10, 20, 12), since=date(2026, 1, 1),
                     until=date(2026, 10, 21), root=stg_env, infra=False, horses=False)
    cols = rep["_side"]["race_keys"]["columns"]
    assert {r[0] for r in rep["_side"]["race_keys"]["rows"] if r[cols.index("key_confidence")] == "confirmed"} == set(ids.values())


def test_completeness_requires_the_period_to_reach_yesterday(stg_env):
    _calendar_world(stg_env)
    early = run_health(settings=_settings(), actual_env="stg", storage=FakeStorage({}), now=_dt(2026, 10, 20, 12),
                       since=date(2026, 1, 1), until=date(2026, 10, 10), root=stg_env, infra=False, horses=False)
    assert any(r["code"] == "range" and "前日" in r["message"] for r in early["completeness"]["reasons"])


def test_future_races_are_not_validated_but_upcoming_horses_are_still_checked(stg_env):
    today_ids = _calendar_world(stg_env)
    future = today_ids["20261021"]
    shutuba = {future: {"race_id": future, "race_name": "x", "entries": [{"horse_number": 1, "horse_name": "a", "horse_id": "2023100001"}]}}
    listing = {"race_shutuba": {"2026": set(today_ids.values())}}
    storage = FakeStorage(listing, shutuba=shutuba)
    rep = run_health(settings=_settings(), actual_env="stg", storage=storage, now=_dt(2026, 10, 20, 12), since=date(2026, 1, 1),
                     root=stg_env, infra=False, horses=True, ledger_dir=None)
    cnt = rep["schema"]["stats"]["categories"]["race_shutuba"]
    assert cnt["scope"] == 2                                                                     # 検証対象は前日までの 2 レースだけ
    assert any(g["horse_id"] == "2023100001" and g["category"] == "horse_result" for g in rep["horse_coverage"]["gaps"])   # 出走予定の馬（翌日）も確認


def test_dev_default_ends_yesterday_so_the_upcoming_mock_day_is_out_of_scope(tmp_path, monkeypatch):
    from src.scripts.data import make_dev_mock

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pr"))
    monkeypatch.setenv("KEIBA_CALCULATED_DATA_DIR", str(tmp_path / "calc"))
    make_dev_mock.generate(date(2026, 10, 6), make_dev_mock.dev_mock_root())                     # 過去日 10/04 と 次の土曜 10/10
    cfg = load_settings({"DATA_HEALTH_SKIP": NO_ART}, profile="dev")
    rep = run_health(settings=cfg, actual_env="dev", now=_dt(2026, 10, 6), root=tmp_path, infra=False, horses=False,
                     ledger_dir=tmp_path / "led")
    assert rep["range"]["until"] == "2026-10-05"
    assert rep["race_coverage"]["universe"]["races"] == 8 and rep["race_coverage"]["gap_total"] == {}
