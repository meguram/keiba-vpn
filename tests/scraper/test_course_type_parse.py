"""内回り/外回り（course_type）の抽出が、出馬表（旧書式）と結果ページ（新書式）の両方で効くこと。"""

import pytest

from src.scraper.parsers import RaceCardParser, parse_course_type


@pytest.mark.parametrize("text,expected", [
    ("12:50発走 / 芝2400m (右 外 B) / 天候:曇 / 馬場:良", "外"),
    ("14:55発走 / 芝1600m (右 外 B) / 天候:曇 / 馬場:良", "外"),
    ("12:15発走 / 芝1600m (右 B) / 天候:曇 / 馬場:良", ""),
    ("12:50発走 / ダ1400m (右) / 天候:曇 / 馬場:良", ""),
    ("10:00発走 / 芝2000m (左 内) / 天候:晴", "内"),
    ("10:00発走 / 芝3000m (右 外-内) / 天候:晴", "外-内"),
    ("10:00発走 / 芝2500m (右 内2周) / 天候:晴", "内2周"),
    ("芝右 外2200m / 天候 : 曇 / 芝 : 良 / 発走 : 10:20", "外"),
    ("芝右 内2周2000m / 天候 : 晴", "内2周"),
    ("障芝 外-内2890m / 天候 : 晴", "外-内"),
    ("ダ右1400m / 天候 : 晴 / ダート : 良 / 発走 : 10:45", ""),
    ("芝右1600m / 天候 : 晴", ""),
    ("", ""),
])
def test_parse_course_type(text, expected):
    assert parse_course_type(text) == expected


def _card_html(data1: str) -> str:
    return ("<html><head><title>テスト 2026年10月17日 3R</title></head><body>"
            "<div class='RaceName'>3歳未勝利</div>"
            f"<div class='RaceData01'>{data1}</div>"
            "<div class='RaceData02'>2回 京都 3日目 サラ系３歳 未勝利 馬齢 9頭</div></body></html>")


@pytest.mark.parametrize("data1,course_type,direction,surface,distance", [
    ("13:20発走 / 芝2400m (右 外 B) / 天候:曇 / 馬場:良", "外", "右", "芝", 2400),
    ("12:15発走 / 芝1600m (右 B) / 天候:曇 / 馬場:良", "", "右", "芝", 1600),
    ("芝右 外2200m / 天候 : 曇 / 馬場 : 良 / 発走 : 10:20", "外", "右", "芝", 2200),
])
def test_card_parser_course_type(data1, course_type, direction, surface, distance):
    info = RaceCardParser().parse(_card_html(data1), "202608020307")
    assert info["course_type"] == course_type
    assert info["direction"] == direction
    assert info["surface"] == surface and info["distance"] == distance
