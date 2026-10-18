from src.scraper import payoff_repair as pr


def test_is_concatenated():
    assert pr.is_concatenated({"複勝": {"numbers": "341", "payout": "120110140"}})
    assert not pr.is_concatenated({"複勝": [{"numbers": "3", "payout": "120", "popularity": "1"}]})
    assert not pr.is_concatenated({"単勝": {"numbers": "3", "payout": "480"}})


def test_unique_split_fukusho_and_wide():
    payoff = {"複勝": {"numbers": "341", "payout": "120110140"}, "ワイド": {"numbers": "3 - 41 - 31 - 4", "payout": "170370220"}}
    fin = {3: 1, 4: 2, 1: 3, 2: 4, 5: 5}
    rows = pr.repair_race(payoff, fin, [1, 2, 3, 4, 5], {}, {})
    f = {r["combo"]: r["payout"] for r in rows if r["bet_type"] == "fukusho"}
    w = {r["combo"]: r["payout"] for r in rows if r["bet_type"] == "wide"}
    assert f == {"3": 120, "4": 110, "1": 140}
    assert w == {"3-4": 170, "1-3": 370, "1-4": 220}
    assert not any(r["ambiguous"] for r in rows)


def test_ambiguity_resolved_by_odds():
    # 馬番 1,12,11 の複勝 "11211" のように分け方が複数あるとき、複勝オッズの幅で 1 つに絞る
    payoff = {"複勝": {"numbers": "11211", "payout": "1202001,100"}}
    fin = {1: 1, 12: 2, 11: 3, 2: 4}
    odds = {1: (1.1, 1.3), 12: (1.9, 2.1), 11: (10.0, 12.0)}
    rows = pr.repair_race(payoff, fin, [1, 2, 11, 12], odds, {})
    got = {r["combo"]: r["payout"] for r in rows}
    assert got == {"1": 120, "12": 200, "11": 1100}
    assert not any(r["ambiguous"] for r in rows)


def test_parser_splits_multirow_cells_with_br():
    from bs4 import BeautifulSoup

    from src.scraper.parsers import RaceResultParser

    html = ('<table class="pay_table_01"><tr><th>単勝</th><td>14</td><td class="txt_r">270</td><td class="txt_r">6</td></tr>'
            '<tr><th>複勝</th><td>14<br/>13<br/>3</td><td class="txt_r">270<br/>220<br/>170</td><td class="txt_r">6<br/>4<br/>1</td></tr>'
            '<tr><th>ワイド</th><td>13 - 14<br/>3 - 14<br/>3 - 13</td><td class="txt_r">920<br/>630<br/>490</td><td class="txt_r">11<br/>5<br/>2</td></tr></table>')
    pay = RaceResultParser()._parse_payoff(BeautifulSoup(html, "html.parser"))
    assert [x["numbers"] for x in pay["複勝"]] == ["14", "13", "3"] and [x["payout"] for x in pay["複勝"]] == ["270", "220", "170"]
    assert pay["ワイド"][1] == {"numbers": "3 - 14", "payout": "630", "popularity": "5"}
