"""めぐ分析（出馬表 + 分析モジュール）。騎手×コースは、開催日より前の結果だけで、ID をキーに数えること。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.race_analysis import runs as runs_mod
from src.research.race_analysis.base import ModuleOutput
from src.research.race_analysis.running_map_module import RunningMapModule
from src.research.race_analysis.service import analyze_race


@pytest.fixture(autouse=True)
def _no_running_map(monkeypatch):
    """走法マップは全期間の出走記録を読むので、ここ（表の分析のテスト）では対象外にする。つなぎ込みは test_running_map_module で見る。"""
    monkeypatch.setattr(RunningMapModule, "applicable", lambda self, ctx: False)


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key):
        return self.data.get((cat, key))


def _row(rid, date, venue, surface, dist, fin, jid, bracket=1, horse="hx", passing="3-3", field=10):
    return {"race_id": rid, "date": date, "venue": venue, "surface": surface, "distance": dist, "finish_position": fin, "jockey_id": jid,
            "bracket_number": bracket, "horse_id": horse, "passing_order": passing, "field_size": field}


@pytest.fixture()
def flat_dir(tmp_path, monkeypatch):
    rows = [
        _row("A1", "2026-01-05", "東京", "芝", 1600, 1, "J1"),
        _row("A2", "2026-01-12", "東京", "芝", 1600, 2, "J1"),
        _row("A3", "2026-01-19", "東京", "芝", 1600, 5, "J1"),
        _row("A4", "2026-02-01", "東京", "芝", 2000, 3, "J1"),        # 同競馬場・同芝ダ（距離違い）
        _row("A5", "2026-02-08", "中山", "芝", 1600, 1, "J1"),        # 同芝ダ（別の競馬場）
        _row("A6", "2026-02-15", "東京", "ダート", 1600, 1, "J1"),     # 芝ダが違う → 数えない
        _row("A7", "2026-03-01", "東京", "芝", 1600, 1, "J2"),
        _row("FUT", "2026-10-31", "東京", "芝", 1600, 1, "J1"),        # 本レース当日の結果は入れない（リーク防止）
        _row("LATER", "2026-11-01", "東京", "芝", 1600, 1, "J1"),
    ]
    d = tmp_path / "2026"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    monkeypatch.setattr(runs_mod, "FLAT_DIR", tmp_path)
    monkeypatch.setattr(runs_mod, "SIRE_MAP_PATH", tmp_path / "no_such_sire_map.parquet")
    runs_mod._cache.update(sig=None, df=None, sire_sig=None, sire_df=None)
    return tmp_path


def _storage(entries, **card):
    base = {"race_name": "テスト特別", "date": "2026-10-31", "venue": "東京", "round": 11, "surface": "芝", "distance": 1600, "entries": entries}
    base.update(card)
    return FakeStorage({("race_shutuba", "202605040111"): base})


def test_counts_only_before_race_date_by_jockey_id(flat_dir):
    st = _storage([
        {"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_name": "甲", "jockey_id": "J1"},
        {"horse_number": 2, "horse_name": "ニ", "horse_id": "h2", "jockey_name": "乙", "jockey_id": "J2"},
        {"horse_number": 3, "horse_name": "サン", "horse_id": "h3", "jockey_name": "丙", "jockey_id": "J9"},
    ])
    d = analyze_race(st, "202605040111")
    assert d["status"] == "ok" and d["provisional_entries"] is False
    m = d["modules"][0]
    assert m["id"] == "jockey_course" and m["status"] == "ok" and m["data_until"] == "2026-03-01"
    # 出典と期間は、実際に集計に使った範囲（本レース当日以降は含めない）
    assert m["source"] == {"provider": "netkeiba.com", "dataset": "レース結果", "period_from": "2026-01-05", "period_to": "2026-03-01", "races": 7, "rows": 7}
    a = {e["horse_number"]: e["analysis"]["jockey_course"] for e in d["entries"]}
    j1 = a[1]
    assert j1["course_rec"] == [1, 1, 0, 1] and j1["course_n"] == 3                  # A1 A2 A3（FUT・LATER は除外）
    assert j1["course_win"] == pytest.approx(1 / 3) and j1["course_top2"] == pytest.approx(2 / 3) and j1["course_top3"] == pytest.approx(2 / 3)
    assert j1["venue_surface_n"] == 4 and j1["venue_surface_rec"] == [1, 1, 1, 1]    # + A4
    assert "surface_n" not in j1                                                     # 全場の群は出さない（別の競馬場の A5・ダートの A6 は数えない）
    assert a[2]["course_n"] == 1 and a[2]["course_win"] == 1.0                       # 別の騎手は混ざらない
    assert a[3]["course_n"] == 0 and a[3]["course_win"] is None                       # 実績が無い騎手は率を作り出さない


def test_groups_and_heat_scales(flat_dir):
    st = _storage([{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_id": "J1"}])
    m = analyze_race(st, "202605040111")["modules"][0]
    assert [g["key"] for g in m["groups"]] == ["course", "venue_surface"]
    by = {c["key"]: c for c in m["columns"]}
    # 色の基準はその条件の全騎手の平均。同コース（東京 芝1600m）は本レース前に A1〜A3・A7 の 4 走で、1着は A1 と A7 の 2 回
    sc = by["course_win"]["scale"]
    assert sc["mid"] == pytest.approx(2 / 4) and sc["lo"] == pytest.approx(0.5 * 0.25) and sc["hi"] == 1.0
    assert by["course_top3"]["scale"]["mid"] == pytest.approx(3 / 4)
    assert by["course_rec"].get("scale") is None                                      # 成績（record）には色を付けない


def test_missing_jockey_id_gives_empty_cells_not_a_crash(flat_dir):
    st = _storage([{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_name": None, "jockey_id": ""}])
    e = analyze_race(st, "202605040111")["entries"][0]["analysis"]["jockey_course"]
    assert e["course_rec"] is None and e["course_n"] == 0 and e["course_win"] is None


def test_provisional_entries_without_horse_numbers_keep_order(flat_dir):
    st = _storage([
        {"horse_number": 0, "horse_name": "ロク", "horse_id": "h6", "jockey_id": "J1"},
        {"horse_number": 0, "horse_name": "シチ", "horse_id": "h7", "jockey_id": "J2"},
    ])
    d = analyze_race(st, "202605040111")
    assert d["provisional_entries"] is True and [e["horse_name"] for e in d["entries"]] == ["ロク", "シチ"]
    assert all(e["horse_number"] is None for e in d["entries"])


def test_no_entries(flat_dir):
    d = analyze_race(FakeStorage({}), "202605040101")
    assert d["status"] == "no_entries" and d["entries"] == [] and d["modules"] == []


def test_a_failing_module_does_not_break_the_others(flat_dir):
    class Boom:
        id, label, description = "boom", "壊れる分析", ""

        def applicable(self, ctx):
            return True

        def compute(self, ctx) -> ModuleOutput:
            raise RuntimeError("x")

    from src.research.race_analysis.registry import MODULES

    st = _storage([{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_id": "J1"}])
    d = analyze_race(st, "202605040111", modules=(Boom(), *MODULES))
    assert [(m["id"], m["status"]) for m in d["modules"]][:2] == [("boom", "error"), ("jockey_course", "ok")]
    assert "boom" not in d["entries"][0]["analysis"] and "jockey_course" in d["entries"][0]["analysis"]


def test_module_not_applicable_when_course_unknown(flat_dir):
    st = _storage([{"horse_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_id": "J1"}], surface=None, distance=None)
    d = analyze_race(st, "202605040111")
    st = {m["id"]: m["status"] for m in d["modules"]}
    assert st["jockey_course"] == st["gate_course"] == st["sire_course"] == "not_applicable"      # コース条件が要る分析
    assert st["past_position"] == "ok"                                                           # 過去の位置取りはコースに依らない


def test_analysis_endpoint_is_member_only():
    from src.api.tier import member_only_path

    assert member_only_path("/api/v1/races/202605040111/analysis")
    assert not member_only_path("/api/v1/races/202605040111/analysis/extra")


def test_gate_course_uses_bracket_and_averages_over_all_horses(tmp_path, monkeypatch):
    rows = [
        _row("G1", "2026-01-05", "東京", "芝", 1600, 1, "J1", bracket=1),
        _row("G1", "2026-01-05", "東京", "芝", 1600, 2, "J2", bracket=2),
        _row("G1", "2026-01-05", "東京", "芝", 1600, 3, "J3", bracket=8),
        _row("G2", "2026-01-12", "東京", "芝", 1600, 1, "J1", bracket=1),
        _row("G2", "2026-01-12", "東京", "芝", 1600, 5, "J2", bracket=8),
        _row("G3", "2026-01-19", "東京", "芝", 2000, 4, "J1", bracket=1),     # 距離違い
        _row("G4", "2026-10-31", "東京", "芝", 1600, 1, "J1", bracket=1),     # 当日 → 数えない
    ]
    d = tmp_path / "2026"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    monkeypatch.setattr(runs_mod, "FLAT_DIR", tmp_path)
    monkeypatch.setattr(runs_mod, "SIRE_MAP_PATH", tmp_path / "none.parquet")
    runs_mod._cache.update(sig=None, df=None, sire_sig=None, sire_df=None)
    st = _storage([
        {"horse_number": 1, "bracket_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_id": "J1"},
        {"horse_number": 2, "bracket_number": 8, "horse_name": "ハチ", "horse_id": "h2", "jockey_id": "J2"},
        {"horse_number": 3, "bracket_number": 0, "horse_name": "未", "horse_id": "h3", "jockey_id": "J3"},
    ])
    d = analyze_race(st, "202605040111")
    m = next(x for x in d["modules"] if x["id"] == "gate_course")
    assert m["status"] == "ok" and "枠順がまだ確定していない" in " ".join(m["notes"])
    a = {e["horse_name"]: e["analysis"]["gate_course"] for e in d["entries"]}
    assert a["イチ"]["course_rec"] == [2, 0, 0, 0] and a["イチ"]["venue_surface_rec"] == [2, 0, 0, 1]   # 1枠: G1,G2 は1着、G3(2000m)は4着
    assert a["ハチ"]["course_rec"] == [0, 0, 1, 1]                                                       # 8枠: 3着と5着
    assert a["未"]["course_rec"] is None and a["未"]["course_win"] is None                               # 枠未確定は作り出さない
    win = next(c for c in m["columns"] if c["key"] == "course_win")
    assert win["scale"]["mid"] == pytest.approx(2 / 5)                                                    # 色の基準はこのコースの全馬の平均（5 走中 1着 2 回）


def test_sire_course_joins_sire_by_horse_id(tmp_path, monkeypatch):
    rows = [
        _row("S1", "2026-01-05", "東京", "芝", 1600, 1, "J1", horse="a1"),
        _row("S1", "2026-01-05", "東京", "芝", 1600, 4, "J2", horse="a2"),
        _row("S2", "2026-01-12", "東京", "芝", 1600, 2, "J1", horse="b1"),
        _row("S3", "2026-01-19", "東京", "芝", 1600, 9, "J1", horse="z9"),   # 父が分からない馬 → どの種牡馬にも数えない
    ]
    d = tmp_path / "2026"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    sire = pd.DataFrame([
        {"horse_id": "a1", "sire_id": "SIRE_A", "sire_name": "父A", "dam_sire_id": "D1", "dam_sire_name": "母父1"},
        {"horse_id": "a2", "sire_id": "SIRE_B", "sire_name": "父B", "dam_sire_id": "D1", "dam_sire_name": "母父1"},
        {"horse_id": "b1", "sire_id": "SIRE_A", "sire_name": "父A", "dam_sire_id": "D2", "dam_sire_name": "母父2"},
        {"horse_id": "e1", "sire_id": "SIRE_A", "sire_name": "父A", "dam_sire_id": "D1", "dam_sire_name": "母父1"},   # 今回の出走馬
    ])
    sire_path = tmp_path / "horse_sire_map.parquet"
    sire.to_parquet(sire_path)
    monkeypatch.setattr(runs_mod, "FLAT_DIR", tmp_path)
    monkeypatch.setattr(runs_mod, "SIRE_MAP_PATH", sire_path)
    runs_mod._cache.update(sig=None, df=None, sire_sig=None, sire_df=None)
    st = _storage([
        {"horse_number": 1, "bracket_number": 1, "horse_name": "今回A", "horse_id": "e1", "jockey_id": "J1"},
        {"horse_number": 2, "bracket_number": 2, "horse_name": "父不明", "horse_id": "unknown", "jockey_id": "J2"},
    ])
    d = analyze_race(st, "202605040111")
    m = next(x for x in d["modules"] if x["id"] == "sire_course")
    assert m["status"] == "ok" and m["key_header"] == "父（種牡馬）"
    a = {e["horse_name"]: e["analysis"]["sire_course"] for e in d["entries"]}
    assert a["今回A"]["_key"] == "父A" and a["今回A"]["sire_rec"] == [1, 1, 0, 0]       # 父A の産駒: S1 で 1着、S2 で 2着
    assert a["父不明"]["sire_rec"] is None and a["父不明"].get("_key") is None
    assert "父を特定できない" in " ".join(m["notes"])


def test_sire_course_excludes_the_horses_own_runs(tmp_path, monkeypatch):
    rows = [
        _row("S1", "2026-01-05", "東京", "芝", 1600, 1, "J1", horse="e1"),   # 今回の出走馬自身の過去走 → 自分の行では数えない
        _row("S2", "2026-01-12", "東京", "芝", 1600, 3, "J1", horse="e1"),
        _row("S3", "2026-01-19", "東京", "芝", 1600, 8, "J2", horse="a1"),   # 同じ父の別の馬
    ]
    d = tmp_path / "2026"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    sire = pd.DataFrame([
        {"horse_id": h, "sire_id": "SIRE_A", "sire_name": "父A", "dam_sire_id": "D1", "dam_sire_name": "母父1"} for h in ("e1", "a1")
    ])
    sire_path = tmp_path / "horse_sire_map.parquet"
    sire.to_parquet(sire_path)
    monkeypatch.setattr(runs_mod, "FLAT_DIR", tmp_path)
    monkeypatch.setattr(runs_mod, "SIRE_MAP_PATH", sire_path)
    runs_mod._cache.update(sig=None, df=None, sire_sig=None, sire_df=None)
    st = _storage([
        {"horse_number": 1, "bracket_number": 1, "horse_name": "自身あり", "horse_id": "e1", "jockey_id": "J1"},
        {"horse_number": 2, "bracket_number": 2, "horse_name": "兄弟", "horse_id": "a1", "jockey_id": "J2"},
    ])
    d = analyze_race(st, "202605040111")
    a = {e["horse_name"]: e["analysis"]["sire_course"] for e in d["entries"]}
    assert a["自身あり"]["sire_rec"] == [0, 0, 0, 1] and a["自身あり"]["sire_n"] == 1     # 自分の 2 走（1着・3着）を除き、a1 の 1 走だけ
    assert a["兄弟"]["sire_rec"] == [1, 0, 1, 0] and a["兄弟"]["sire_top3"] == 1.0         # a1 の行は e1 の 2 走だけ
    j = {e["horse_name"]: e["analysis"]["jockey_course"] for e in d["entries"]}
    assert j["自身あり"]["course_n"] == 0                                                  # J1 の騎乗は e1 の分だけ → 除くと 0


@pytest.mark.parametrize("raw, shown", [
    ("マジェスティックウォリアーMajestic Warrior(米)", "マジェスティックウォリアー(米)"),
    ("ノーザンダンサーNorthern Dancer", "ノーザンダンサー"),
    ("Gun Runner (米)", "Gun Runner (米)"),
    ("オメガパフューム", "オメガパフューム"),
])
def test_sire_name_drops_english_name_after_kana(raw, shown):
    from src.research.race_analysis.course_stats import display_sire_name

    assert display_sire_name(raw) == shown


def test_build_horse_sire_map_parses_ancestors_and_is_incremental(tmp_path):
    import json

    from src.scripts.data import build_horse_sire_map as B

    ped = tmp_path / "ped"
    (ped / "2022").mkdir(parents=True)

    def doc(hid, sire_id):
        return {"horse_id": hid, "sire": "x", "dam_sire": "y", "ancestors": [
            {"generation": 1, "position": 0, "name": "父名", "horse_id": sire_id},
            {"generation": 1, "position": 1, "name": "母名", "horse_id": "m"},
            {"generation": 2, "position": 2, "name": "母父名", "horse_id": "dsire"}]}

    (ped / "2022" / "2022100001.json").write_text(json.dumps(doc("2022100001", "S1")), encoding="utf-8")
    (ped / "2022" / "2022100002.json").write_text(json.dumps({"horse_id": "2022100002", "ancestors": []}), encoding="utf-8")   # 父が取れない
    out = tmp_path / "map.parquet"
    r1 = B.build(out, ped, {"2022100001", "2022100002", "2022100003"})            # 3 頭目は血統 JSON 無し
    assert r1["added"] == 1 and r1["no_pedigree"] == 2 and r1["total"] == 1
    m = pd.read_parquet(out)
    assert m.iloc[0].to_dict() == {"horse_id": "2022100001", "sire_id": "S1", "sire_name": "父名", "dam_sire_id": "dsire", "dam_sire_name": "母父名", "sire_line_id": "父名系", "sire_line_name": "父名系"}
    (ped / "2022" / "2022100003.json").write_text(json.dumps(doc("2022100003", "S2")), encoding="utf-8")
    r2 = B.build(out, ped, {"2022100001", "2022100002", "2022100003"})            # 既存の行は読み直さない。後から出来た血統を拾う
    assert r2["total"] == 2 and r2["read"] == 2


def test_line_of_walks_paternal_chain_to_nearest_head_without_other_bucket():
    from src.research.race_analysis.sire_lines import line_of

    def chain(*pairs):
        return [{"generation": g, "position": 0, "horse_id": i, "name": n} for g, (i, n) in enumerate(pairs, start=1)]

    # 最初に当たる系統の祖でまとまる
    assert line_of(chain(("2002100816", "ディープインパクト"), ("000a00033a", "サンデーサイレンスSunday Silence(米)"), ("000a0012bf", "Halo"))) == ("000a00033a", "サンデーサイレンス系")
    # 孫系統（ロードカナロア → キングカメハメハ → キングマンボ）は近い祖が優先
    assert line_of(chain(("2008103552", "ロードカナロア"), ("2001103460", "キングカメハメハ"), ("000a001d7e", "Kingmambo(米)")))[1] == "キングカメハメハ系"
    # 一覧に無い系統は『その他』にせず、わかる範囲でいちばん古い父系の祖の系統
    assert line_of(chain(("x1", "父"), ("x2", "ハンプトンHampton"), ("x3", "Bay Ronald(英)"))) == ("x3", "Bay Ronald系")
    assert line_of(chain(("x1", "父"), ("x2", "ハンプトンHampton")))[1] == "ハンプトン系"
    assert line_of([]) is None
    # 5代で決まらないときは、いちばん古い祖自身の血統をたどって系統の祖に当たる（途中で当たれば『その他』にしない）
    deeper = {"x3": [{"generation": 1, "position": 0, "horse_id": "x4", "name": "A"}, {"generation": 2, "position": 0, "horse_id": "000a000e04", "name": "Northern Dancer"}]}
    assert line_of(chain(("x1", "父"), ("x3", "Vice Regent")), lookup=deeper.get) == ("000a000e04", "ノーザンダンサー系")
    # 母側（position != 0）の祖は見ない
    assert line_of([{"generation": 1, "position": 0, "horse_id": "x", "name": "父"}, {"generation": 2, "position": 1, "horse_id": "000a00033a", "name": "母方"}]) == ("x", "父系")


def test_sire_tab_shows_line_next_to_sire(tmp_path, monkeypatch):
    from src.scraper import sire_line_store as S

    monkeypatch.setenv("KEIBA_SIRE_LINE_MAP", str(tmp_path / "sl.json"))
    S.record_html("", "x")      # 台帳はまだ空・完了の印なし

    rows = [
        _row("S1", "2026-01-05", "東京", "芝", 1600, 1, "J1", horse="a1"),
        _row("S1", "2026-01-05", "東京", "芝", 1600, 4, "J2", horse="a2"),
        _row("S2", "2026-01-12", "東京", "芝", 1600, 2, "J1", horse="b1"),
    ]
    d = tmp_path / "2026"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    sire = pd.DataFrame([
        {"horse_id": "a1", "sire_id": "SIRE_A", "sire_name": "父A", "dam_sire_id": "D1", "dam_sire_name": "m", "sire_line_id": "000a00033a", "sire_line_name": "サンデーサイレンス系"},
        {"horse_id": "a2", "sire_id": "SIRE_B", "sire_name": "父B", "dam_sire_id": "D1", "dam_sire_name": "m", "sire_line_id": "000a0012bf", "sire_line_name": "ヘイロー系"},
        {"horse_id": "b1", "sire_id": "SIRE_C", "sire_name": "父C", "dam_sire_id": "D2", "dam_sire_name": "m", "sire_line_id": "000a00033a", "sire_line_name": "サンデーサイレンス系"},   # 父は違うが同じ系統
        {"horse_id": "e1", "sire_id": "SIRE_D", "sire_name": "父D", "dam_sire_id": "D1", "dam_sire_name": "m", "sire_line_id": "000a00033a", "sire_line_name": "サンデーサイレンス系"},
    ])
    sire_path = tmp_path / "horse_sire_map.parquet"
    sire.to_parquet(sire_path)
    monkeypatch.setattr(runs_mod, "FLAT_DIR", tmp_path)
    monkeypatch.setattr(runs_mod, "SIRE_MAP_PATH", sire_path)
    runs_mod._cache.update(sig=None, df=None, sire_sig=None, sire_df=None)
    st = _storage([{"horse_number": 1, "bracket_number": 1, "horse_name": "今回D", "horse_id": "e1", "jockey_id": "J1"}])
    d0 = analyze_race(st, "202605040111")
    m0 = next(x for x in d0["modules"] if x["id"] == "sire_course")
    assert [g["key"] for g in m0["groups"]] == ["sire"]                  # 系統の取得が完全に終わるまで、系統の群は出さない
    assert "line_rec" not in d0["entries"][0]["analysis"]["sire_course"]
    S.record_html('<table class="blood_table"><tr><td rowspan="16"><a href="/horse/z/">z</a></td></tr></table>', "x")
    S.set_complete()
    d = analyze_race(st, "202605040111")
    m = next(x for x in d["modules"] if x["id"] == "sire_course")
    assert [g["key"] for g in m["groups"]] == ["sire", "line"]            # 種牡馬と系統を同じタブに並べる。同競馬場・同芝ダの群は無い
    a = d["entries"][0]["analysis"]["sire_course"]
    assert a["_key"] == "父D" and a["sire_rec"] == [0, 0, 0, 0] and a["sire_n"] == 0       # 父D 自身の産駒はまだ走っていない
    assert a["line_name"] == "サンデーサイレンス系" and a["line_rec"] == [1, 1, 0, 0]       # 父A・父C の産駒（父Bはヘイロー系なので入らない）


def test_sire_line_store_reads_label_and_overrides_derived_line(tmp_path, monkeypatch):
    from src.scraper import sire_line_store as S
    from src.scripts.data import build_horse_sire_map as B

    html = """<table class="blood_table"><tr><td rowspan="16"><a href="/horse/2007103265/">トゥザグローリー</a><br>Mr. Prospector系</td>
    <td rowspan="8"><a href="/horse/2001103460/">キングカメハメハ</a></td></tr></table>"""
    assert S.lines_from_html(html) == {"2007103265": ("トゥザグローリー", "Mr. Prospector系")}
    path = tmp_path / "sl.json"
    assert S.record_html(html, "H1", path) == 1
    S.mark_missing("S9", "H2", path)
    assert S.labels(path) == {"2007103265": "Mr. Prospector系"}            # 表記なしで残したものはラベルに入らない
    assert "S9" in S.load(path)                                          # ただし取得済みとして覚える

    m = pd.DataFrame([
        {"horse_id": "a", "sire_id": "2007103265", "sire_line_id": "キングカメハメハ系", "sire_line_name": "キングカメハメハ系"},
        {"horse_id": "b", "sire_id": "other", "sire_line_id": "ロベルト系", "sire_line_name": "ロベルト系"},
    ])
    out = B.apply_netkeiba_lines(m, S.labels(path))
    assert list(out["sire_line_name"]) == ["Mr. Prospector系", "ロベルト系"] and list(out["sire_line_id"]) == ["Mr. Prospector系", "ロベルト系"]


def test_fetch_sire_lines_picks_most_common_sires_first_and_skips_done():
    from src.scripts.scraping.fetch_sire_lines import pick_targets

    sm = pd.DataFrame({"horse_id": ["a", "b", "c", "d"], "sire_id": ["S1", "S1", "S2", "S3"]})
    assert pick_targets(sm, set()) == [("S1", "b"), ("S2", "c"), ("S3", "d")]
    assert pick_targets(sm, {"S1"}) == [("S2", "c"), ("S3", "d")]


def test_store_complete_marker_survives_rewrites(tmp_path):
    from src.scraper import sire_line_store as S

    p = tmp_path / "sl.json"
    assert S.is_complete(p) is False
    S.mark_missing("A", "h", p)
    S.set_complete(p)
    assert S.is_complete(p) is True
    S.mark_missing("B", "h", p)                                  # 書き直しても印は残る
    assert S.is_complete(p) is True and set(S.load(p)) == {"A", "B"}


def test_ancestor_sire_children_picks_child_of_each_male_ancestor():
    from src.scraper.sire_line_store import ancestor_sire_children

    rec = {"horse_id": "H", "ancestors": [
        {"generation": 1, "position": 0, "horse_id": "A", "sex": "牡"},
        {"generation": 1, "position": 1, "horse_id": "B", "sex": "牝"},
        {"generation": 2, "position": 0, "horse_id": "C", "sex": "牡"},
        {"generation": 2, "position": 2, "horse_id": "D", "sex": "牡"},
        {"generation": 3, "position": 4, "horse_id": "E", "sex": "牡"},
        {"generation": 6, "position": 0, "horse_id": "Z", "sex": "牡"},
        {"generation": 2, "position": 1, "horse_id": "I", "sex": "牝"},
        {"generation": 3, "position": 2, "horse_id": "F", "sex": ""},
        {"generation": 3, "position": 3, "horse_id": "G", "sex": ""},
    ]}
    assert ancestor_sire_children(rec) == {"A": "H", "C": "A", "D": "B", "E": "D", "F": "I"}


def test_store_merge_and_diff_for_parallel_ledgers(tmp_path):
    from src.scraper import sire_line_store as S

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    S.mark_missing("2007103265", "h", a)                                           # 手元は『表記なし』
    S.mark_missing("ONLY_A", "h", a)
    S.record_html('<table class="blood_table"><tr><td rowspan="16"><a href="/horse/2007103265/">x</a><br>ロベルト系</td></tr></table>', "h2", b)
    S.mark_missing("ONLY_B", "h", b)
    assert S.diff(S.load(a), S.load(b)) == {"only_a": ["ONLY_A"], "only_b": ["ONLY_B"], "line": ["2007103265"]}
    assert S.merge(S.load(b), a) == 2                                      # X は表記ありが勝ち、ONLY_B が増える
    assert S.labels(a) == {"2007103265": "ロベルト系"} and set(S.load(a)) == {"2007103265", "ONLY_A", "ONLY_B"}
    assert S.merge(S.load(b), a) == 0                                      # 2 回足しても同じ
    S.merge({"2007103265": {"name": None, "line": None, "from": "z", "at": "2099-01-01"}}, a)
    assert S.labels(a) == {"2007103265": "ロベルト系"}                              # 後から来た『表記なし』で消えない


def _writer(path, prefix):
    from src.scraper import sire_line_store as S

    for i in range(25):
        S.mark_missing(f"{prefix}{i}", "h", path)


def test_store_concurrent_writers_lose_nothing(tmp_path):
    import multiprocessing as mp

    from src.scraper import sire_line_store as S

    p = tmp_path / "sl.json"
    ps = [mp.Process(target=_writer, args=(p, c)) for c in "ABCD"]
    [x.start() for x in ps]
    [x.join() for x in ps]
    assert len(S.load(p)) == 100


def _ped(father_id, line):
    cell = f'<a href="/horse/{father_id}/">父</a>' + (f"<br>{line}" if line else "")
    return f'<table class="blood_table"><tr><td rowspan="32">{cell}</td></tr></table>'


def test_fetch_and_record_distinguishes_no_data_from_errors(tmp_path):
    from src.scraper import sire_line_store as S

    p, waits = tmp_path / "sl.json", []
    sleep = waits.append

    def pages(*items):
        it = iter(items)

        def fetch(url):
            x = next(it)
            if isinstance(x, Exception):
                raise x
            return x
        return fetch

    # 系統あり
    assert S.fetch_and_record(pages(_ped("1111111111", "ロベルト系")), "u", "1111111111", "c1", p, sleep=sleep) == "line"
    # ページは正常で系統の表記だけ無い → 理由つきで残す（取り直さない）
    assert S.fetch_and_record(pages(_ped("2222222222", "")), "u", "2222222222", "c2", p, sleep=sleep) == "no_line"
    assert S.load(p)["2222222222"]["reason"] == "no_line" and S.load(p)["2222222222"]["line"] is None
    # 父が別の馬 → 理由つきで残す
    assert S.fetch_and_record(pages(_ped("9999999999", "")), "u", "3333333333", "c3", p, sleep=sleep) == "father_mismatch"
    assert waits == []
    # 通信エラーは 2 回目で成功すれば台帳に正しく残り、間を 1 回置く
    assert S.fetch_and_record(pages(OSError("boom"), _ped("4444444444", "ノーザンダンサー系")), "u", "4444444444", "c4", p, sleep=sleep) == "line"
    assert len(waits) == 1
    # 血統表の無い応答（メンテ画面など）も取り直し、3 回とも駄目なら台帳に残さない
    waits.clear()
    assert S.fetch_and_record(pages("<html>maintenance</html>", OSError("x"), "<html></html>"), "u", "5555555555", "c5", p, sleep=sleep) == "failed"
    assert len(waits) == 2 and "5555555555" not in S.load(p)


def test_running_map_module_returns_a_figure_keyed_by_entry(monkeypatch):
    from src.research.race import running_map as rm
    from src.research.race_analysis.base import RaceContext

    seen = {}

    def fake(entries, race_id, conditions):
        seen.update(conditions=conditions, n=len(entries))
        return {"status": "ok", "axes": [{"key": "ts"}], "groups": [{"id": "A", "label": "高TS", "horse_numbers": [1]}],
                "horses": [{"horse_id": "h1", "horse_number": 1, "horse_name": "甲", "x": 0.1, "y": 0.2, "group": "A",
                            "percentiles": {"ts": 80.0}, "n_runs": 5, "estimated": False, "own_weight": 0.6,
                            "last_run_date": "2026-10-01", "values": {"ts": 55.0}}],
                "unmapped": [{"horse_id": "h2", "horse_name": "乙", "horse_number": 2, "reason": "過去の出走も父のデータもありません"}],
                "fidelity": {"explained": 0.8}, "data_until": "2026-10-25",
                "extent": {"n": 24, "x": [-2, 2], "y": [-3, 3]},
                "aptitude": {"course": "東京 芝 1600m", "n_runs": 400, "n_races": 30, "lift": [1.0], "sig": "2"}}
    monkeypatch.setattr(rm, "build_running_map", fake)
    ctx = RaceContext(race_id="202605040811", date="2026/10/31", venue="東京", surface="芝", distance=1600, track_condition="稍",
                      entries=[{"key": "h1", "horse_id": "h1", "horse_number": 1}, {"key": "h2", "horse_id": "h2", "horse_number": 2},
                               {"key": "idx2", "horse_id": None, "horse_number": 3}])
    out = RunningMapModule.__new__(RunningMapModule).compute(ctx)
    assert seen == {"conditions": {"date": "2026-10-31", "venue": "東京", "surface": "芝", "distance": 1600, "track_condition": "稍重"}, "n": 2}
    assert out.view == "map" and out.columns == [] and out.data_until == "2026-10-25"
    h = out.figure["horses"][0]
    assert h["key"] == "h1" and "values" not in h and out.figure["unmapped"][0]["key"] == "h2"
    assert out.figure["extent"]["n"] == 24 and out.figure["aptitude"]["course"] == "東京 芝 1600m"
