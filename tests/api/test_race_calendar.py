from src.api.race_calendar import build_month, day_summary, grade_of, stem_range


def _r(rid, venue, name, rnd, icon=None):
    return {"race_id": rid, "venue": venue, "race_name": name, "round": rnd, "list_grade_icon": icon}


def test_grade_from_icon_and_name():
    assert grade_of(_r("202608040711", "京都", "秋華賞", 11, "Icon_GradeType1")) == "G1"
    assert grade_of(_r("x", "京都", "サウジRC", 11, "Icon_GradeType3")) == "G3"
    assert grade_of(_r("x", "東京", "東京HJ", 8, "Icon_GradeType11")) == "J.G2"
    assert grade_of(_r("x", "京都", "宇治川特別", 9, "Icon_GradeType17")) is None      # 条件戦
    assert grade_of(_r("x", "京都", "セントウルS(G2)", 11)) == "G2"                      # 古い一覧: 名前から
    assert grade_of(_r("x", "京都", "3歳未勝利", 1)) is None


def test_day_summary_orders_venues_and_grades_by_code():
    rs = [
        _r("202605040711", "東京", "第45回アルゼンチン共和国杯(G2)", 11),
        _r("202604040711", "新潟", "新潟記念(G3)", 11),
        _r("202608040701", "京都", "2歳未勝利", 1),
    ]
    s = day_summary("20261011", rs)
    assert s["venues"] == ["新潟", "東京", "京都"] and s["n"] == 3
    assert [g["name"] for g in s["grades"]] == ["新潟記念", "アルゼンチン共和国杯"]


def test_build_month_skips_future_other_months_and_non_race_days():
    data = {"20261010": {"races": [_r("202605040101", "東京", "未勝利", 1)]}, "20261011": {"races": []},
            "20261024": {"races": [_r("202605050101", "東京", "未勝利", 1)]}, "20260926": {"races": [_r("202604030101", "新潟", "未勝利", 1)]}}
    days = build_month("202610", list(data), data.get, lambda s, races, meta: bool(races), "20261017")
    assert [d["date"] for d in days] == ["20261010"]                    # 10/11 は開催なし、10/24 は未来、9 月は別の月


def test_stem_range_clips_to_today_and_ignores_garbage():
    r = stem_range(["20200105", "20261227", "20261017", "abc", "2026"], "20261018")
    assert r == {"min": "20200105", "max": "20261017"}
    assert stem_range([], "20261018") == {"min": None, "max": None}
