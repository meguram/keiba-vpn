"""race.netkeiba.com の RaceData01 の 2026-10 下旬からの書き方（天候はアイコンの title、馬場は Item04）から、馬場・天候を取れること。

10/31 以降の速報（race_result_on_time）で馬場・天候が空になった（「天候:」「馬場:」の文字が無くなったため）。
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from src.scraper.parsers import RaceCardParser, RaceResultOnTimeParser, weather_track_from_racedata

FIX = Path(__file__).resolve().parent / "fixtures" / "race_netkeiba_result_202608050301_racedata_202611.html"


def test_on_time_parser_reads_icon_weather_and_item04_track():
    d = RaceResultOnTimeParser().parse(FIX.read_text(encoding="utf-8"), race_id="202608050301")
    assert d["weather"] == "曇" and d["track_condition"] == "良"
    assert d["surface"] == "ダート" and d["distance"] == 1400 and d["start_time"] == "10:01"


def test_old_text_format_still_wins():
    html = '<div class="RaceData01">10:05発走 / 芝1600m(左) / 天候:晴 / 馬場:稍 <span class="Item04"> 良</span></div>'
    assert RaceResultOnTimeParser()._parse_race_info(BeautifulSoup(html, "html.parser"))["track_condition"] == "稍"


def test_helper_ignores_non_condition_text():
    tag = BeautifulSoup('<div class="RaceData01"><span class="Item04">/ 直線：弱向風</span></div>', "html.parser").div
    assert weather_track_from_racedata(tag) == ("", "")
    tag = BeautifulSoup('<div class="RaceData01"><span class="Icon_Weather01" aria-label="雨"></span><span class="Item04">/ 不良 </span></div>',
                        "html.parser").div
    assert weather_track_from_racedata(tag) == ("雨", "不")                  # これまでの速報の略称にそろえる
    tag = BeautifulSoup('<div class="RaceData01"><span class="Icon_Weather01" title="晴れのち曇り"></span><span class="Item04">稍<b>重</b></span></div>',
                        "html.parser").div
    assert weather_track_from_racedata(tag) == ("", "稍")                    # 許さない天候は入れない。分かれた文字もつなげて読む
    tag = BeautifulSoup('<div class="RaceData01"><span class="Item04">芝:良 ダ:稍</span></div>', "html.parser").div
    assert weather_track_from_racedata(tag) == ("", "良")                    # 2 つ並ぶときは先頭（旧形式の正規表現と同じ）
    assert weather_track_from_racedata(None) == ("", "")


def test_card_parser_uses_the_same_fallback():
    info = RaceCardParser()._parse_card_info(BeautifulSoup(FIX.read_text(encoding="utf-8"), "html.parser"))
    assert info["weather"] == "曇" and info["track_condition"] == "良"
