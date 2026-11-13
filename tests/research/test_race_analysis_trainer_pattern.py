import json

import pytest

from src.research.race_analysis import trainer_pattern as TP
from src.research.race_analysis.base import RaceContext


def _row(n, heads, c1, c2, c3, w, p, base):
    k = len(base)
    return [n, heads, c1, c2, c3, w, p, base, [c1] + [0] * (k - 1), [c1 + c2 + c3] + [0] * (k - 1), [w] + [0.0] * (k - 1), [p] + [0.0] * (k - 1)]


TABLE = {"v": 1, "y0": 2020, "y1": 2025, "max_odds": 50.0, "edges": [2, 3, 5, 10, 20],
         "trainers": {"t1": {"n": "調教師1", "p": {"栗東坂路|A2": _row(10, 8, 2, 1, 1, 9.0, 8.0, [4, 3, 1, 1, 1, 0]),
                                                  "栗東坂路|B1": _row(6, 5, 0, 1, 1, 3.0, 4.0, [2, 2, 1, 1, 0, 0]),
                                                  "美浦坂路|X": _row(4, 4, 1, 0, 0, 2.0, 1.0, [0, 1, 1, 1, 1, 0])}}}}


def test_levels_sum_trainer_and_place_and_pick_pattern():
    lv = TP.levels_for(TABLE, "t1", "栗東坂路|A2")
    assert lv["pattern"][0] == 10 and lv["place"][0] == 16 and lv["trainer"][0] == 20
    assert lv["place"][2:5] == [2, 2, 2] and lv["trainer"][2:5] == [3, 2, 2]
    assert lv["place"][5] == 12.0 and lv["trainer"][6] == 13.0
    assert lv["place"][7] == [6, 5, 2, 2, 1, 0] and lv["trainer"][7] == [6, 6, 3, 3, 2, 0]
    assert lv["place"][1] is None   # 頭数は足せない


def test_levels_without_pattern_or_unknown_trainer():
    lv = TP.levels_for(TABLE, "t1", None)
    assert lv["pattern"] is None and lv["place"] is None and lv["trainer"][0] == 20
    assert TP.levels_for(TABLE, "zz", "栗東坂路|A2") is None
    assert TP.levels_for(TABLE, "t1", "函館ウッド|W52.0-53.9")["pattern"] is None   # 表に無いパターン（調教師は有る）


def _record(hid):
    def e(day, laps, ri="2026/03/15  阪神1R 3歳未勝利"):
        return {"race_info": ri, "date": day, "course": "栗坂", "track_condition": "良", "lap_times": laps, "position": "", "leg_color": "馬也", "evaluation": "", "rank": "B"}
    return {"horse_id": hid, "entries": [e("2026-03-11", [53.1, 38.4, 24.9, 12.6]),      # 水: ベスト
                                         e("2026-03-09", [56.0, 40.0, 26.0, 13.0]),      # 月: 遅い
                                         e("2026-03-07", [51.0, 37.0, 24.0, 12.0]),      # 前週の土曜: 候補外
                                         e("2026-03-18", [50.0, 36.0, 23.0, 11.5], "2026/03/22  阪神1R 3歳未勝利")]}   # 別レース向け


@pytest.fixture
def host(tmp_path, monkeypatch):
    def path(hid):
        return tmp_path / f"{hid}.json" if hid else None
    monkeypatch.setattr(TP, "_training_path", path)
    (tmp_path / "h1.json").write_text(json.dumps(_record("h1")), encoding="utf-8")
    return tmp_path


def test_patterns_of_uses_best_weekday_workout_of_race_week_only(host):
    out = TP.patterns_of(["h1", "h2"], "2026-03-15")
    assert out["h2"] is None and out["h1"].startswith("栗東坂路|")
    assert TP.patterns_of(["h1"], "2026-03-22")["h1"].startswith("栗東坂路|") and TP.patterns_of(["h1"], "2026-04-05")["h1"] is None


def test_training_sig_changes_when_the_file_changes(host):
    a = TP.training_sig([{"horse_id": "h1"}, {"horse_id": "h2"}])
    assert a[0][1] is not None and a[1][1] is None
    (host / "h1.json").write_text(json.dumps(_record("h1")) + " ", encoding="utf-8")
    assert TP.training_sig([{"horse_id": "h1"}])[0] != a[0]


def test_applicable_only_after_the_table_period_and_with_training_files(host, monkeypatch):
    monkeypatch.setattr(TP, "load_table", lambda: TABLE)
    m = TP.TrainerPatternModule()
    ent = [{"horse_id": "h1", "key": "h1"}]
    assert m.applicable(RaceContext("r", "2026-03-15", "阪神", "芝", 1600, ent))
    assert not m.applicable(RaceContext("r", "2025-12-28", "阪神", "芝", 1600, ent))     # 表の期間内（未来の結果を含む）
    assert not m.applicable(RaceContext("r", "2026-03-15", "阪神", "芝", 1600, [{"horse_id": "h2", "key": "h2"}]))   # 調教の記録が無い環境


def test_compute_attaches_levels_per_horse(host, monkeypatch):
    monkeypatch.setattr(TP, "load_table", lambda: TABLE)
    wk = {"h1": {"st": "pattern", "last": None, "why": None}, "h2": {"st": "unfetched", "last": None, "why": None}}
    monkeypatch.setattr(TP, "workouts_of", lambda ids, d: ({"h1": "栗東坂路|A2", "h2": None}, wk))
    ent = [{"horse_id": "h1", "key": "h1", "horse_number": 1, "horse_name": "A", "trainer_id": "t1", "trainer_name": "調教師1"},
           {"horse_id": "h2", "key": "h2", "horse_number": 2, "horse_name": "B", "trainer_id": "zz", "trainer_name": "不明"}]
    out = TP.TrainerPatternModule().compute(RaceContext("r", "2026-03-15", "阪神", "芝", 1600, ent))
    assert out.view == "trainer_pattern" and out.figure["edges"] == [2, 3, 5, 10, 20]
    h1, h2 = out.figure["horses"]
    assert h1["levels"]["pattern"][0] == 10 and h2["levels"] is None and h2["pat"] is None
    assert h1["wk"]["st"] == "pattern" and h2["wk"]["st"] == "unfetched"
    json.dumps(out.figure)   # そのまま JSON にできる


def _rec_with(entries, scraped="2026-03-10 00:00:00"):
    def e(day, laps, course="栗坂", ri="2026/03/15  阪神1R 3歳未勝利"):
        return {"race_info": ri, "date": day, "course": course, "track_condition": "良", "lap_times": laps, "position": "", "leg_color": "馬也", "evaluation": "", "rank": "B"}
    return {"horse_id": "h9", "entries": [e(*x) for x in entries], "_meta": {"scraped_at_jst": scraped}}


@pytest.mark.parametrize("entries,scraped,st,why", [
    ([("2026-03-11", [53.1, 38.4, 24.9, 12.6])], "2026-03-14 18:00:00", "pattern", None),
    ([("2026-03-11", [None, None, 24.9, 12.6])], "2026-03-14 18:00:00", "out", "no_4f"),        # 平日だが 4F の時計なし（終いだけ）
    ([("2026-03-14", [53.1, 38.4, 24.9, 12.6])], "2026-03-14 18:00:00", "out", "weekend"),      # レース週の土曜だけ
    ([("2026-03-04", [53.1, 38.4, 24.9, 12.6])], "2026-03-14 18:00:00", "out", "no_week"),      # 前の週だけ
    ([("2026-03-08", [53.1, 38.4, 24.9, 12.6])], "2026-03-14 18:00:00", "out", "no_week"),      # 前の週の日曜だけ（レース週の土日ではない）
    ([("2026-03-18", [50.0, 36.0, 23.0, 11.5], "栗坂", "2026/03/22  阪神1R 3歳未勝利")], "2026-03-10 00:00:00", "unfetched", None),   # このレースの分をまだ取っていない
    ([("2026-03-18", [50.0, 36.0, 23.0, 11.5], "栗坂", "2026/03/22  阪神1R 3歳未勝利")], "2026-03-16 00:00:00", "none", None),   # レースの後に取っても無い
    ([("2026-03-18", [50.0, 36.0, 23.0, 11.5], "栗坂", "2026/03/22  阪神1R 3歳未勝利")], "2026-03-14 18:00:00", "none", None),   # 前日夕方に取って無い
    ([("2026-03-18", [50.0, 36.0, 23.0, 11.5], "栗坂", "2026/03/22  阪神1R 3歳未勝利")], "2026-03-13 18:00:00", "unfetched", None),   # 2 日前の取得
    ([("2026-03-18", [50.0, 36.0, 23.0, 11.5], "栗坂", "2026/03/22  阪神1R 3歳未勝利")], "2026-03-14 09:00:00", "unfetched", None),   # 前日でも朝（最終追い切りが載る前）
])
def test_workouts_of_tells_why_a_horse_has_no_pattern(tmp_path, monkeypatch, entries, scraped, st, why):
    monkeypatch.setattr(TP, "_training_path", lambda hid: tmp_path / f"{hid}.json")
    (tmp_path / "h9.json").write_text(json.dumps(_rec_with(entries, scraped)), encoding="utf-8")
    pats, wk = TP.workouts_of(["h9", "h0"], "2026-03-15")
    assert wk["h9"]["st"] == st and wk["h9"]["why"] == why
    assert (pats["h9"] is not None) == (st == "pattern")
    assert wk["h0"] == {"st": "nofile", "last": None, "why": None}   # 調教ページが無い
    if st in ("pattern", "out"):
        assert wk["h9"]["last"]["place"] == "栗東坂路" and wk["h9"]["last"]["date"] == entries[-1][0]
    json.dumps(wk)


def _row(day, t4):
    return {"train_date": day, "course": "栗坂", "t4": t4}


def test_wk_status_cls_when_weekday_4f_exists_but_no_pattern():
    """レース週の平日に 4F はあるが、パターンに区分できない（pat が None）"""
    w = TP._wk_status({"_meta": {"scraped_at_jst": "2026-03-14 18:00:00"}}, [_row("2026-03-11", 52.0)], "2026-03-15", None)
    assert w["st"] == "out" and w["why"] == "cls" and w["last"]["t4"] == 52.0


@pytest.mark.parametrize("day,t4,why", [
    ("2026-03-11", None, "no_4f"),                 # 月曜のレース（3/16）: レース週は前の週の月〜金（3/9〜3/13）
    ("2026-03-14", 52.0, "weekend"),               # 前の土曜（3/14）はレース週の土日
    ("2026-03-15", 52.0, "weekend"),
    ("2026-03-06", 52.0, "no_week"),               # その前の週
])
def test_wk_status_monday_race_uses_previous_week(day, t4, why):
    w = TP._wk_status({"_meta": {"scraped_at_jst": "2026-03-15 18:00:00"}}, [_row(day, t4)], "2026-03-16", None)
    assert w["st"] == "out" and w["why"] == why


def test_wk_status_uses_epoch_when_jst_time_is_missing():
    from datetime import datetime, timedelta, timezone

    ts = datetime(2026, 3, 14, 19, 0, tzinfo=timezone(timedelta(hours=9))).timestamp()
    assert TP._wk_status({"_meta": {"scraped_at": ts}}, [], "2026-03-15", None)["st"] == "none"       # 前日 19:00 の取得 → 記録なし
    assert TP._wk_status({"_meta": {"scraped_at": ts - 86400}}, [], "2026-03-15", None)["st"] == "unfetched"
    assert TP._wk_status({}, [], "2026-03-15", None)["st"] == "unfetched"                            # 時刻が全く無い古い記録は「未取得」のまま
