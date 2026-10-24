"""着順セルの正規化と、解析失敗の手がかり(finish_anomalies)のテスト。"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from src.scraper.finish_status import UNPARSED, finish_anomalies, normalize_finish
from src.scraper.parsers import RaceResultParser
from tests.scraper.test_result_parser_columns import HEAD_EXPANDED, _cells, _table


@pytest.mark.parametrize("raw,expected", [
    ("1", (1, "finished")), ("12", (12, "finished")), ("１０", (10, "finished")), (" 3 ", (3, "finished")),
    ("5(降)", (5, "finished")),
    ("取", (-1, "scratched")), ("除", (-1, "excluded")), ("中", (-1, "dnf")), ("失", (-1, "disqualified")),
    ("降", (-1, UNPARSED)), ("", (-1, UNPARSED)), (None, (-1, UNPARSED)), ("---", (-1, UNPARSED)), ("同着", (-1, UNPARSED)),
])
def test_normalize_finish(raw, expected):
    assert normalize_finish(raw) == expected


def test_legit_exception_and_parse_failure_are_distinguishable():
    # どちらも position は -1 だが、status で区別できる（従来はどちらも -1 だけだった）
    assert normalize_finish("中")[1] == "dnf" and normalize_finish("？？")[1] == UNPARSED


def test_anomalies_flag_unranked_finisher_and_rank_gap():
    entries = [
        {"horse_number": 1, "finish_position": 1, "finish_status": "finished", "time_sec": 60.0},
        {"horse_number": 2, "finish_position": 3, "finish_status": "finished", "time_sec": 61.0},   # 2 着が欠番
        {"horse_number": 3, "finish_position": -1, "finish_status": "unparsed", "time_sec": 60.5},   # 完走したのに着順を読めていない
        {"horse_number": 4, "finish_position": -1, "finish_status": "dnf", "time_sec": 0.0},          # 競走中止(正当)
    ]
    a = finish_anomalies(entries)
    assert a == {"unparsed": [3], "timed_unranked": [3], "rank_gap": [3]}
    assert finish_anomalies(entries[:1] + entries[3:]) == {"unparsed": [], "timed_unranked": [], "rank_gap": []}


def test_parser_keeps_raw_token_and_status():
    for raw, pos, status in (("1", 1, "finished"), ("中", -1, "dnf"), ("降", -1, "unparsed")):
        cells = _cells("3-3-2", "34.1"); cells[0] = raw
        e = RaceResultParser()._parse_entries(BeautifulSoup(_table(HEAD_EXPANDED, cells), "html.parser"))[0]
        assert (e["finish_position"], e["finish_status"], e["finish_position_raw"]) == (pos, status, raw)
