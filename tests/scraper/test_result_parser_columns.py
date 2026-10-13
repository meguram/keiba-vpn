from bs4 import BeautifulSoup

from src.scraper.parsers import RaceResultParser

HEAD_EXPANDED = ["着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差", "ﾀｲﾑ指数", "ｽﾀｰﾄ指数", "追走指数",
                 "上がり指数", "備考", "通過", "上り", "単勝", "人気", "馬体重", "調教師"]


def _table(head, cells):
    th = "".join(f"<th>{h}</th>" for h in head)
    td = "".join(f"<td>{c}</td>" for c in cells)
    return f'<table class="race_table_01"><tr>{th}</tr><tr>{td}</tr></table>'


def _cells(passing, last_3f):
    return ["1", "1", "7", '<a href="/horse/2020100001/">テスト</a>', "牡3", "56", '<a href="/jockey/01234/">騎手</a>',
            "1:34.5", "", "85", "90", "88", "92", "80", passing, last_3f, "3.5", "1", "480(+2)",
            '<a href="/trainer/05678/">調教師</a>']


def _parse(html):
    return RaceResultParser()._parse_entries(BeautifulSoup(html, "html.parser"))


def test_columns_follow_header_even_when_premium_cells_are_blank():
    e = _parse(_table(HEAD_EXPANDED, _cells("", "")))[0]
    assert e["passing_order"] == ""            # 空欄は空欄のまま（指数列の値を読まない）
    assert e["odds"] == 3.5 and e["popularity"] == 1
    assert e["weight"] == 480 and e["weight_change"] == 2


def test_columns_follow_header_with_values():
    e = _parse(_table(HEAD_EXPANDED, _cells("3-3-2", "34.1")))[0]
    assert e["passing_order"] == "3-3-2" and e["last_3f"] == 34.1


def test_falls_back_to_layout_guess_without_header():
    html = '<table class="race_table_01"><tr><td>x</td></tr><tr>' + "".join(f"<td>{c}</td>" for c in _cells("3-3-2", "34.1")) + "</tr></table>"
    assert len(_parse(html)) == 1  # ヘッダーが無くても従来の推定で動く


def test_round_is_derived_from_race_id():
    r = RaceResultParser().parse("<html><body></body></html>", race_id="202509040410")
    assert r["round"] == 10
