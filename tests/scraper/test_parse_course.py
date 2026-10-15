import pytest

from src.scraper.parsers import parse_course


@pytest.mark.parametrize("text,expected", [
    ("芝右1600m / 天候 : 晴", ("芝", "右", 1600)),
    ("ダ右1200m / 天候 : 曇", ("ダート", "右", 1200)),
    ("ダート右1200m", ("ダート", "右", 1200)),
    ("芝直1000m / 天候", ("芝", "直", 1000)),
    ("芝直線1000m / 天候", ("芝", "直", 1000)),
    ("芝右 外1200m / 天候 : 曇 / 芝 : 稍重", ("芝", "右", 1200)),
    ("芝左 内2000m", ("芝", "左", 2000)),
    ("芝右 内2周 2000m", ("芝", "右", 2000)),
    ("芝右 外・内 3170m", ("芝", "右", 3170)),
    ("ダ左1400m", ("ダート", "左", 1400)),
    ("障芝 右 3000m", ("障", "右", 3000)),
    ("障2860m", ("障", "", 2860)),
])
def test_parse_course_formats(text, expected):
    assert parse_course(text) == expected


def test_no_course_returns_none():
    assert parse_course("天候 : 晴 / 発走 : 11:35") is None
    assert parse_course("") is None
