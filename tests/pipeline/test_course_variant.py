import pandas as pd

from src.pipeline.features import course_variant as cv


def test_unique_and_ambiguous():
    assert cv.resolve_course_type("06", "芝", 2200) == "外"            # 中山芝2200 は外回りだけ
    assert cv.resolve_course_type("06", "ダート", 1800) == ""           # ダートは区別なし
    assert cv.resolve_course_type("08", "芝", 1600) is None             # 京都芝1600 は内外の両方がある
    assert cv.resolve_course_type("08", "芝", 1600, parsed_available=True) == "内"   # 無印=内（外は「外」が付く）
    assert cv.resolve_course_type("08", "芝", 1600, parsed="外") == "外"
    assert cv.resolve_course_type("99", "芝", 1600) is None


def test_add_course_variant_series():
    df = pd.DataFrame({"venue_code": ["06", "08", "05"], "surface": ["芝", "芝", "ダ"], "distance": [2200, 1600, 1600], "course_type": ["", "", ""]})
    got = cv.add_course_variant(df)
    assert got.iloc[0] == "外" and pd.isna(got.iloc[1])
