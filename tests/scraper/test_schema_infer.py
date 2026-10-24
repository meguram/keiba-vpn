"""スキーマ定義（schema_defs.json）と、実データからの再構成（schema_infer）のテスト。"""

from __future__ import annotations

import json

import pytest

from src.scraper import schema_infer as I
from src.scraper import schemas
from src.scraper.storage import HybridStorage


# ── 定義ファイル（全環境で同一・git 管理）────────────────────────────────────

def test_defs_file_is_the_single_source():
    defs = json.loads(schemas._DEFS_PATH.read_text(encoding="utf-8"))
    assert defs["categories"] == schemas.SCHEMAS and defs["schema_version"] == schemas.SCHEMA_VERSION
    assert len(schemas.schema_fingerprint()) == 12 and schemas.schema_fingerprint() == schemas.schema_fingerprint()


def test_every_category_has_schema_or_a_reason():
    """新カテゴリを足したのに定義も理由も無い、を CI で止める。"""
    cats = set(HybridStorage.CATEGORY_MAP)
    assert not (cats - set(schemas.SCHEMAS) - set(schemas.NO_SCHEMA_REASON)), "スキーマも no_schema_reason も無いカテゴリ"
    assert not (set(schemas.SCHEMAS) & set(schemas.NO_SCHEMA_REASON)), "スキーマ化したら no_schema_reason から外す"
    assert all(v["status"] in ("pending", "unused", "not_json") for v in schemas.NO_SCHEMA_REASON.values())


def test_fingerprint_changes_when_definition_changes(monkeypatch):
    before = schemas.schema_fingerprint()
    changed = json.loads(json.dumps(schemas.SCHEMAS))
    changed["race_odds"]["top_required"]["extra"] = {"type": "str"}
    monkeypatch.setattr(schemas, "SCHEMAS", changed)
    assert schemas.schema_fingerprint() != before


def test_observed_profiles_match_known_categories():
    for p in I.OBSERVED_DIR.glob("*.json"):
        prof = json.loads(p.read_text(encoding="utf-8"))
        assert prof["category"] == p.stem and prof["records"] >= 1


def test_advisory_schema_does_not_block_save(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_SCHEMA_STRICT", "1")
    schemas_copy = json.loads(json.dumps(schemas.SCHEMAS))
    schemas_copy["race_predictions"] = {"top_required": {"race_id": {"type": "str"}}, "advisory": True}
    schemas_copy["race_odds"]["advisory"] = False
    monkeypatch.setattr(schemas, "SCHEMAS", schemas_copy)
    st = HybridStorage(base_dir=str(tmp_path))
    bad = {"nope": 1}
    assert st.save("race_predictions", "202605010101", bad) is True            # advisory: 保存される
    assert bad["_meta"]["scrape_validation_status"] == "schema_failed"          # が、不合格は記録される
    with pytest.raises(schemas.SchemaValidationError):
        st.save("race_odds", "202605010101", {"nope": 1})                       # 厳格カテゴリは従来どおり止まる


# ── 観測プロファイル ──────────────────────────────────────────────────────────

def _rec(i, **over):
    d = {"race_id": f"2025050101{i % 12 + 1:02d}", "race_name": "x", "venue": "東京",
         "entries": [{"horse_number": n, "horse_name": f"h{n}", "horse_id": f"20200000{n:02d}", "odds": 1.5 * n} for n in range(1, 4)]}
    d.update(over)
    return d


def _profile(records, category="race_odds"):
    prof = I.new_profile(category)
    for i, r in enumerate(records):
        I.add_record(prof, r, f"k{i:03d}")
    return prof


def test_profile_counts_types_patterns_and_lists():
    prof = _profile([_rec(1), _rec(2, race_name=None, extra_flag=True)])
    top = prof["top"]
    assert prof["records"] == 2 and top["race_id"]["patterns"] == {"digits12": 2}
    assert top["race_name"]["null"] == 1 and top["extra_flag"]["types"] == {"bool": 1}
    f = prof["lists"]["entries"]["fields"]
    assert prof["lists"]["entries"]["items"] == 6 and f["odds"]["types"] == {"float": 6} and f["horse_number"]["max"] == 3
    assert prof["keys"] == {"min": "k000", "max": "k001"}


def test_derive_requires_enough_records_then_separates_required_and_optional():
    few = I.derive_schema(_profile([_rec(i) for i in range(5)]))
    assert few["top_required"] == {}                                             # 根拠不足 → すべて任意
    recs = [_rec(i) for i in range(40)]
    for i in range(0, 40, 4):                                                     # 25% にしか無いキー
        recs[i]["sometimes"] = "a"
    sch = I.derive_schema(_profile(recs))
    assert set(sch["top_required"]) == {"race_id", "race_name", "venue", "entries"}
    assert sch["top_optional"]["sometimes"]["type"] == "str"
    assert sch["top_required"]["race_id"] == {"type": "str", "non_empty": True, "pattern": r"^\d{12}$"}
    assert sch["top_required"]["entries"]["min_length"] == 1
    assert sch["entry_required"]["odds"] == {"type": "float"} and sch["entry_required"]["horse_id"]["pattern"] == r"^\d{10}$"


def test_null_values_lower_presence_and_mixed_types_become_any():
    recs = [_rec(i, race_name=None if i % 2 else "x", mixed=1 if i % 2 else "s") for i in range(40)]
    sch = I.derive_schema(_profile(recs))
    assert "race_name" in sch["top_optional"] and sch["top_required"]["mixed"]["type"] == "any"


def test_merge_profiles_adds_counts():
    a, b = _profile([_rec(i) for i in range(3)]), _profile([_rec(i) for i in range(4)])
    m = I.merge_profiles(a, b)
    assert m["records"] == 7 and m["top"]["venue"]["n"] == 7 and m["lists"]["entries"]["items"] == 21
    assert m["conformance"]["validated"] == 7


# ── 既存定義との照合・反映 ────────────────────────────────────────────────────

def _existing():
    return {"top_required": {"race_id": {"type": "str"}, "venue": {"type": "str"}, "gone": {"type": "str"}},
            "top_optional": {"race_name": {"type": "str"}}, "entry_required": {"horse_number": {"type": "int"}},
            "entry_optional": {}}


def test_reconcile_is_conservative_by_default_and_reports_the_rest():
    recs = [_rec(i) for i in range(40)]
    prof = _profile(recs)
    new, ch = I.reconcile(_existing(), I.derive_schema(prof), prof, promote=False, demote=False)
    kinds = {(c["kind"], c["detail"].split(" ")[0].rstrip(":")) for c in ch}
    assert ("add_optional", "top.entries") in kinds                              # 未定義のキーは任意として追加
    assert ("promote_candidate", "top.race_name") in kinds and "race_name" in new["top_optional"]   # 既定では昇格しない
    assert ("never_observed", "top.gone") in kinds and "gone" in new["top_required"]              # 自動では消さない
    assert ("add_optional", "entry.horse_name") in kinds
    new2, ch2 = I.reconcile(_existing(), I.derive_schema(prof), prof, promote=True, demote=True)
    assert "race_name" in new2["top_required"]


def test_reconcile_reports_type_conflicts_and_demotion():
    recs = [_rec(i, venue=123) for i in range(40)]
    for i in range(0, 40, 2):
        recs[i].pop("race_id")
    prof = _profile(recs)
    ex = _existing()
    new, ch = I.reconcile(ex, I.derive_schema(prof), prof, promote=False, demote=True)
    assert any(c["kind"] == "type_conflict" and "venue" in c["detail"] for c in ch)
    assert any(c["kind"] == "demote" and "race_id" in c["detail"] for c in ch) and "race_id" in new["top_optional"]


def test_new_category_is_added_as_advisory():
    prof = _profile([_rec(i) for i in range(40)], category="brand_new")
    new, ch = I.reconcile(None, I.derive_schema(prof), prof, promote=False, demote=False)
    assert new["advisory"] is True and ch[0]["kind"] == "new_category"


def test_apply_writes_defs_with_provenance_and_keeps_handwritten_constraints(tmp_path, monkeypatch):
    defs = tmp_path / "defs.json"
    defs.write_text(json.dumps({"schema_version": 2, "categories": {"race_odds": {
        "top_required": {"race_id": {"type": "str", "pattern": "^\\d{12}$"}}, "top_optional": {}}}}), encoding="utf-8")
    monkeypatch.setattr(I, "DEFS_PATH", defs)
    prof = _profile([_rec(i) for i in range(3)])
    rep = I.apply_profiles({"race_odds": prof})
    out = json.loads(defs.read_text(encoding="utf-8"))["categories"]["race_odds"]
    assert out["top_required"]["race_id"]["pattern"] == "^\\d{12}$"                # 手書きの制約は保つ
    assert "venue" in out["top_optional"] and out["_provenance"]["samples"] == 3 and out["_provenance"]["source"] == "observed"
    assert any(c["kind"] == "add_optional" for c in rep["race_odds"])


# ── データの出所 ─────────────────────────────────────────────────────────────

def test_mock_data_is_never_used_for_inference(tmp_path):
    (tmp_path / "race_odds" / "2025").mkdir(parents=True)
    real = {"race_id": "202505010101", "entries": [{"horse_number": 1}]}
    (tmp_path / "race_odds/2025/202505010101.json").write_text(json.dumps(real), encoding="utf-8")
    (tmp_path / "race_odds/2025/202505010102.json").write_text(json.dumps({**real, "_meta": {"dev_mock": True}}), encoding="utf-8")
    got = list(I.iter_dir(tmp_path, ["race_odds"], 10))
    assert [k for _, k, _ in got] == ["202505010101"]


def test_collect_from_dir_merges_into_profile_dir(tmp_path):
    for y in ("2024", "2025"):
        (tmp_path / "src" / "race_odds" / y).mkdir(parents=True)
        for n in range(3):
            (tmp_path / "src" / "race_odds" / y / f"{y}0501010{n}.json").write_text(
                json.dumps({"race_id": f"{y}0501010{n}", "entries": [{"horse_number": 1, "win_odds": 2.0}]}), encoding="utf-8")
    out = tmp_path / "obs"
    recs = list(I.iter_dir(tmp_path / "src", ["race_odds"], 100))
    I.collect(recs, source="dir", directory=out)
    p = I.collect(iter(recs), source="dir", directory=out)["race_odds"]       # 2 回目は加算
    assert p["records"] == 12 and p["conformance"]["validated"] == 12 and len(p["runs"]) == 2


def test_spread_picks_evenly():
    assert I._spread(list(range(100)), 4) == [0, 25, 50, 75] and I._spread([1, 2], 5) == [1, 2]


# ── 違反の記録（どの値で引っかかったか）─────────────────────────────────────────

def test_violations_carry_the_offending_values():
    data = {"race_id": "202505010101", "entries": [
        {"horse_number": 1, "horse_name": "a", "horse_id": "x", "bracket_number": 1},
        {"horse_number": "２", "horse_name": "b", "horse_id": "y", "bracket_number": 2},
        {"horse_name": "c", "horse_id": "z", "bracket_number": 3}], "race_name": "n", "date": "2026/10/01", "venue": "東京"}
    rep = schemas.validate("race_result", data)
    by = {(v["field"], v["rule"]): v for v in rep["violations"]}
    d = by[("date", "pattern")]
    assert d["actual"] == "'2026/10/01'" and d["actual_type"] == "str" and "日付" not in d["expected"]
    t = by[("entries[].horse_number", "type")]
    assert (t["actual"], t["index"], t["where"]["horse_id"]) == ("'２'", 1, "y")                  # 何番目の要素か・どの馬か
    m = by[("entries[].horse_number", "missing")]
    assert m["actual"] == "<キー無し>" and m["index"] == 2 and m["where"]["horse_name"] == "c"
    assert "entries[1].horse_number" in schemas.describe_violation(t)
    assert "'２'" in str(schemas.SchemaValidationError("race_result", "k", rep))


def test_long_values_and_many_violations_are_capped():
    big = {"race_id": "x" * 500, "entries": [{"horse_number": "bad"} for _ in range(200)]}
    rep = schemas.validate("race_odds", big)
    assert len(rep["violations"]) <= schemas.MAX_VIOLATIONS and rep.get("violations_truncated") is True
    assert all(len(v["actual"]) <= schemas.MAX_VALUE_CHARS + 1 for v in rep["violations"])


def test_save_rejection_records_values_and_quarantines_then_resolves(tmp_path, monkeypatch):
    from src.scraper import schema_violations as SV

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_SCHEMA_STRICT", "1")
    st = HybridStorage(base_dir=str(tmp_path))
    bad = {"race_id": "202505010101", "entries": [{"horse_number": "x"}]}
    with pytest.raises(schemas.SchemaValidationError) as ei:
        st.save("race_odds", "202505010101", bad)
    assert "'x'" in str(ei.value)                                                       # 例外メッセージにも値
    log = list(SV.read_log(tmp_path))
    assert len(log) == 1 and log[0]["decision"] == "rejected" and log[0]["key"] == "202505010101"
    assert any(v["field"] == "entries[].horse_number" and v["actual"] == "'x'" for v in log[0]["violations"])
    q = tmp_path / "data/local/quarantine/race_odds/202505010101.json"
    assert json.loads(q.read_text(encoding="utf-8"))["data"]["entries"][0]["horse_number"] == "x"     # 拒否したデータ本体
    assert st.exists("race_odds", "202505010101") is False                                             # 保存はされていない
    good = {"race_id": "202505010101", "entries": [{"horse_number": 1}]}
    assert st.save("race_odds", "202505010101", good) is True
    assert not q.exists()                                                                              # 後で合格したら隔離を解除
    s = SV.summarize(tmp_path)
    g = s["groups"][0]
    assert (g["category"], g["field"], g["rule"], g["count"]) == ("race_odds", "entries[].horse_number", "type", 1)
    assert g["values"] == [("str 'x'", 1)]


def test_advisory_and_lenient_saves_keep_values_in_meta_and_log(tmp_path, monkeypatch):
    from src.scraper import schema_violations as SV

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    st = HybridStorage(base_dir=str(tmp_path))
    monkeypatch.setenv("KEIBA_SCHEMA_STRICT", "0")
    d = {"race_id": "202505010102", "entries": [{"horse_number": "y"}]}
    assert st.save("race_odds", "202505010102", d) is True                                              # lenient: 保存される
    assert d["_meta"]["schema_validation"]["violations"][0]["actual"] == "'y'"                           # データ自体にも残る
    assert [r["decision"] for r in SV.read_log(tmp_path)] == ["saved_lenient"]
    assert not (tmp_path / "data/local/quarantine").exists()                                             # 保存したので隔離しない


def test_recording_failure_never_breaks_saving(tmp_path, monkeypatch):
    from src.scraper import schema_violations as SV

    monkeypatch.setenv("KEIBA_ENV", "dev")
    monkeypatch.setenv("KEIBA_DEV_MOCK_DIR", str(tmp_path / "mock"))
    monkeypatch.setenv("KEIBA_SCHEMA_STRICT", "1")
    (tmp_path / "data").write_text("ファイルがあるのでディレクトリを作れない", encoding="utf-8")
    SV.record(tmp_path, "race_odds", "k", {"violations": []}, "rejected", payload={})              # 例外にならない
    monkeypatch.setenv("KEIBA_SCHEMA_VIOLATION_LOG", "0")
    monkeypatch.setenv("KEIBA_SCHEMA_QUARANTINE", "0")
    SV.record(tmp_path, "race_odds", "k", {"violations": []}, "rejected", payload={})


def test_summary_cli_shows_values(tmp_path, capsys):
    from src.scraper import schema_violations as SV

    for i in range(3):
        SV.record(tmp_path, "race_odds", f"k{i}", {"schema_version": 2, "violations": [
            {"field": "entries[].win_odds", "rule": "type", "expected": "float", "actual": "'—'", "actual_type": "str", "index": 0}]},
            "rejected", payload={"race_id": f"k{i}"})
    assert SV.main(["--base-dir", str(tmp_path), "summary"]) == 0
    out = capsys.readouterr().out
    assert "entries[].win_odds" in out and "3 回 / 3 key" in out and "3 × str '—'" in out and "隔離中" in out
    assert SV.main(["--base-dir", str(tmp_path), "show", "race_odds", "k1"]) == 0
    assert "entries[0].win_odds" in capsys.readouterr().out
