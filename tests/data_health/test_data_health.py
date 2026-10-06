"""データ存在チェック（src.data_health）のテスト。GCS には触れず、フェイクストレージと dev モックで検証する。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

from src.data_health import coverage as V
from src.data_health.checks import check_artifact
from src.data_health.plan import build_plan
from src.data_health.runner import JST, run_health
from src.data_health.spec import ARTIFACTS, ArtifactSpec


def _dt(y, m, d, hh=12):
    return datetime(y, m, d, hh, tzinfo=JST)


def _sample_value(field: str, spec: dict, key: str):
    t, pat = spec.get("type", "any"), spec.get("pattern", "")
    if field == "race_id":
        return key
    if pat:
        return "2025-01-01" if r"\d{4}-" in pat else "20250101" if r"\d{8}" in pat else "1"
    return {"str": "x", "int": max(spec.get("min", 1), 1), "float": 1.0, "list": [], "dict": {}, "bool": True}.get(t, "x")


def minimal_valid(category: str, key: str) -> dict:
    """スキーマの必須項目だけを満たす最小のデータ（フェイクストレージ用）。"""
    from src.scraper import schemas

    sch = schemas.SCHEMAS[category]
    doc = {f: _sample_value(f, sp, key) for f, sp in sch.get("top_required", {}).items()}
    if sch.get("entry_required") is not None and "entry_required" in sch:
        doc[sch.get("entry_list_key", "entries")] = [{f: _sample_value(f, sp, key) for f, sp in sch["entry_required"].items()}]
    return doc


class FakeStorage:
    """HybridStorage の list / load だけを模す。load はスキーマに適合する最小データを返す（``bad`` のキーは不適合データ）。"""

    def __init__(self, listing: dict[str, dict[str, set[str]]], shutuba: dict[str, dict] | None = None,
                 bad: set[tuple[str, str]] | None = None):
        self.listing = listing
        self.shutuba = shutuba or {}
        self.bad = bad or set()
        self.gcs_enabled = True
        self.loads = 0
        self.ts: dict[tuple[str, str], float] = {}          # 更新時刻の上書き（ファイルが更新された状況を作る）

    def invalidate_blob_cache(self, *a, **k):
        pass

    def batch_list_blobs(self, category, year):
        return {k: self.ts.get((category, k), 1_700_000_000.0) for k in self.listing.get(category, {}).get(year, set())}

    def load(self, category, key, bypass_cache=False):
        self.loads += 1
        if category == "race_shutuba" and key in self.shutuba:
            return self.shutuba[key]
        if (category, key) in self.bad:
            return {"unexpected": True}
        from src.scraper import schemas

        return minimal_valid(category, key) if category in schemas.SCHEMAS else None


def _ids(kai_days, place="05", year="2025", kai=1, rounds=range(1, 13)):
    return {f"{year}{place}{kai:02d}{d:02d}{r:02d}" for d in kai_days for r in rounds}


# ── 期限 ────────────────────────────────────────────────────────────────

def test_due_time_follows_sla():
    sun, sat = date(2026, 10, 4), date(2026, 10, 10)
    assert V.due_time("weekly", sun).date() == date(2026, 10, 9)      # 日曜の結果 → 次の金曜
    assert V.due_time("weekly", sat).date() == date(2026, 10, 16)     # 土曜の結果 → 翌週金曜
    assert V.due_time("shutuba", sat).date() == date(2026, 10, 9)     # 出馬表は前日
    assert V.due_time("dayof", sat).date() == sat


def test_classify_pending_vs_missing():
    spec = next(c for c in V.RACE_CATEGORIES if c.name == "race_result")
    race = V.Race("202605030201", "20261004", "confirmed")
    assert V.classify(spec, race, set(), set(), _dt(2026, 10, 6)) == "pending"     # 金曜前は待機
    assert V.classify(spec, race, set(), set(), _dt(2026, 10, 10)) == "missing"    # 金曜を過ぎたら不足
    assert V.classify(spec, race, {"202605030201"}, set(), _dt(2026, 10, 6)) == "present"
    assert V.classify(spec, race, set(), {"202605030201"}, _dt(2026, 10, 10)) == "na"


# ── 連番からの推定 ──────────────────────────────────────────────────────────

def test_infer_missing_fills_gap_days_only_between_observed():
    observed = _ids([1, 3])
    inferred = V.infer_missing(observed)
    assert inferred == _ids([2])                       # 2日目が丸ごと欠損
    assert "202505010101" not in inferred


def test_infer_missing_detects_missing_rounds_and_leading_days():
    observed = _ids([1, 2]) - {"202505010205"}         # 2日目の5Rだけ欠け
    assert V.infer_missing(observed) == {"202505010205"}
    assert V.infer_missing(_ids([2])) == _ids([1])     # 開催回は1日目から始まる


# ── 全体（stg 相当）─────────────────────────────────────────────────────────

@pytest.fixture()
def stg_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pr"))
    monkeypatch.setenv("KEIBA_CALCULATED_DATA_DIR", str(tmp_path / "calc"))
    return tmp_path


def _write_race_list(tmp_path, ymd, ids):
    d = tmp_path / "pr" / "race_lists"
    d.mkdir(parents=True, exist_ok=True)
    races = [{"race_id": r, "round": int(r[10:12]), "venue": "東京", "race_name": "x"} for r in sorted(ids)]
    (d / f"{ymd}.json").write_text(json.dumps({"date": ymd, "races": races}), encoding="utf-8")


def test_stg_reports_missing_periods_and_builds_plan(stg_env):
    day1, day2, day4 = _ids([1]), _ids([2]), _ids([4])          # 3日目は race_lists にもデータにも無い
    _write_race_list(stg_env, "20250201", day1)
    _write_race_list(stg_env, "20250202", day2)
    listing = {
        "race_shutuba": {"2025": day1 | day2 | day4},
        "race_result": {"2025": day1 | (day2 - {"202505010205"}) | (day4 - {r for r in day4 if int(r[10:12]) <= 8})},
    }
    for c in V.RACE_CATEGORIES:            # 他カテゴリは全件あるものとして、結果と推定分だけを欠損にする
        listing.setdefault(c.name, {"2025": day1 | day2 | day4})
    rep = run_health(env="stg", storage=FakeStorage(listing), now=_dt(2026, 10, 6), since=date(2025, 1, 1),
                     until=date(2025, 12, 31), root=stg_env, infra=False, horses=False)
    rc = rep["race_coverage"]
    assert rc["universe"]["by_confidence"] == {"confirmed": 24, "present": 12, "inferred": 12}
    res = rc["by_year"]["2025"]["race_result"]
    assert (res["healthy"], res["missing"]) == (12 + 11 + 4, 1 + 8 + 12)
    assert rc["by_period"]["2025-02"]["race_result"]["missing"] == 1          # 日付が分かる欠損
    assert rc["by_period"]["2025-??"]["race_result"]["missing"] == 8 + 12     # 日付不明（4日目の8R＋推定した3日目）
    assert rc["gap_total"]["race_result"] == 21 and rc["gap_total"]["race_shutuba"] == 12
    assert rep["summary"]["overall"] == "fail"                                 # race_result は stg で required

    specs = rep["plan"]["specs"]
    assert any(s["job_kind"] == "race" and s["target_id"] == "202505010205" and "race_result" in s["tasks"] for s in specs)
    inferred_day = [s for s in specs if s["target_id"].startswith("2025050103")]
    assert len(inferred_day) == 12 and all("推定" in s["reason"] for s in inferred_day)
    assert all(s["smart_skip"] is True and s["overwrite"] is False for s in specs)
    assert all(s["runner"].startswith("学習PC") for s in specs)              # 2025年分は過去 → 学習PC


def test_date_all_when_many_races_missing_on_a_date(stg_env):
    day = _ids([1])
    _write_race_list(stg_env, "20250201", day)
    rep = run_health(env="stg", storage=FakeStorage({"race_shutuba": {"2025": day}}), now=_dt(2026, 10, 6),
                     since=date(2025, 1, 1), until=date(2025, 12, 31), root=stg_env, infra=False, horses=False)
    spec = [s for s in rep["plan"]["specs"] if s["job_kind"] == "date"]
    assert len(spec) == 1 and spec[0]["target_id"] == "20250201" and spec[0]["tasks"] == ["date_all"]


def test_complete_data_is_ok_and_pending_is_not_a_gap(stg_env):
    day = _ids([1], year="2026")
    _write_race_list(stg_env, "20261004", day)
    full = {c.name: {"2026": day} for c in V.RACE_CATEGORIES if c.name != "race_predictions"}
    full["race_predictions"] = {"2026": day}
    rep = run_health(env="stg", storage=FakeStorage(full), now=_dt(2026, 10, 6), since=date(2026, 1, 1),
                     until=date(2026, 10, 20), root=stg_env, infra=False, horses=False)
    assert rep["race_coverage"]["gap_total"] == {}
    assert rep["plan"]["counts"]["jobs"] == 0


def test_pending_results_for_recent_race_are_not_missing(stg_env):
    day = _ids([1], year="2026")
    _write_race_list(stg_env, "20261004", day)
    rep = run_health(env="stg", storage=FakeStorage({"race_shutuba": {"2026": day}}), now=_dt(2026, 10, 6),
                     since=date(2026, 1, 1), until=date(2026, 10, 20), root=stg_env, infra=False, horses=False)
    res = rep["race_coverage"]["by_year"]["2026"]["race_result"]
    assert res["pending"] == 12 and res["missing"] == 0            # 金曜(10/9)までは待機


def test_horse_gaps_from_shutuba_window(stg_env):
    day = _ids([1], year="2026", rounds=[1])                     # 1 レースだけ（他のレースの出走馬が混ざらないように）
    _write_race_list(stg_env, "20261010", day)
    rid = sorted(day)[0]
    shutuba = {rid: {"entries": [{"horse_id": "2023100001"}, {"horse_id": "2023100002"}]}}
    listing = {"race_shutuba": {"2026": day}, "horse_result": {"2023": {"2023100001"}}}
    rep = run_health(env="stg", storage=FakeStorage(listing, shutuba), now=_dt(2026, 10, 6), since=date(2026, 1, 1),
                     until=date(2026, 10, 20), root=stg_env, infra=False, horses=True)
    hc = rep["horse_coverage"]
    assert hc["horses"] == 2
    hr = next(c for c in hc["categories"] if c["name"] == "horse_result")
    assert (hr["expected"], hr["missing"]) == (2, 1)
    assert any(s["job_kind"] == "horse" and s["target_id"] == "2023100002" for s in rep["plan"]["specs"])


# ── 派生データ ──────────────────────────────────────────────────────────────

def test_artifact_per_year_reports_missing_years(tmp_path):
    spec = ArtifactSpec("T", "t", "g", ("data/local/features/race_tbl/{Y}/*.parquet",), {"stg": "required"}, per_year=True)
    (tmp_path / "data/local/features/race_tbl/2024").mkdir(parents=True)
    (tmp_path / "data/local/features/race_tbl/2024/a.parquet").write_text("x")
    r = check_artifact(spec, "stg", ["2024", "2025"], tmp_path)
    assert r["status"] == "fail" and r["missing_years"] == ["2025"]
    assert check_artifact(spec, "dev", ["2024"], tmp_path)["status"] == "skip"


def test_artifact_finds_both_feature_roots(tmp_path):
    spec = ArtifactSpec("T", "t", "g", ("{F}/horse_tbl/**/*.parquet",), {"stg": "recommended"})
    (tmp_path / "data/features/horse_tbl").mkdir(parents=True)
    (tmp_path / "data/features/horse_tbl/x.parquet").write_text("x")
    r = check_artifact(spec, "stg", [], tmp_path)
    assert r["status"] == "ok" and "data/features" in r["detail"]


def test_every_artifact_defines_all_envs():
    for a in ARTIFACTS:
        assert set(a.level) == {"dev", "stg", "prod"}, a.id


# ── dev（モック）──────────────────────────────────────────────────────────

def test_dev_health_on_mock_passes_and_detects_removed_file(tmp_path, monkeypatch):
    from src.scripts.data import make_dev_mock

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_PAGE_REFERENCE_DIR", str(tmp_path / "pr"))
    monkeypatch.setenv("KEIBA_CALCULATED_DATA_DIR", str(tmp_path / "calc"))
    today = date(2026, 10, 6)
    make_dev_mock.generate(today, make_dev_mock.dev_mock_root())
    now = _dt(2026, 10, 6)
    (tmp_path / "config").mkdir()
    for name in ("settings.yaml", "megu_par_time_v2_meta.json", "megu_predict_condition_weights.json", "megu_predict_params.json"):
        (tmp_path / "config" / name).write_text("x")                 # dev で required な設定ファイル
    rep = run_health(now=now, root=tmp_path, infra=True)
    assert rep["env"] == "dev"
    assert rep["summary"]["counts"]["fail"] == 0
    assert rep["race_coverage"]["gap_total"] == {}
    assert rep["plan"]["counts"]["jobs"] == 0                      # dev はスクレイピング計画を出さない
    assert next(c for c in rep["checks"] if c["id"] == "dev.gcp_blocked")["status"] == "ok"

    victim = next((tmp_path / "mock" / "race_shutuba").rglob("*.json"))
    victim.unlink()
    rep2 = run_health(now=now, root=tmp_path, infra=False)
    assert rep2["race_coverage"]["gap_total"]["race_shutuba"] == 1
    assert rep2["summary"]["overall"] == "fail"                    # dev でもモック欠損(required)は FAIL


def test_report_outputs_are_written(tmp_path):
    from src.data_health.report import render_html, render_text, write_outputs

    rep = run_health(env="stg", storage=FakeStorage({}), now=_dt(2026, 10, 6), since=date(2026, 1, 1),
                     until=date(2026, 10, 20), root=tmp_path, infra=False, horses=False)
    paths = write_outputs(rep, tmp_path / "out")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["env"] == "stg"
    assert "<table" in render_html(rep) and "stg" in render_text(rep)
    assert paths["plan"].exists() and paths["html"].exists()
