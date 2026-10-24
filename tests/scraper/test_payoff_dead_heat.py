from bs4 import BeautifulSoup

from src.pipeline.betting import winning_combos as W
from src.scraper.parsers import RaceResultParser

# 3 着同着（2 と 3）の払戻表（202405010309 の形）
HTML = (
    '<table class="pay_table_01">'
    '<tr><th>単勝</th><td>7</td><td class="txt_r">370</td><td class="txt_r">2</td></tr>'
    '<tr><th>複勝</th><td>7<br/>4<br/>2<br/>3</td><td class="txt_r">110<br/>110<br/>110<br/>200</td><td class="txt_r">2<br/>1<br/>3<br/>6</td></tr>'
    '<tr><th>馬連</th><td>4 - 7</td><td class="txt_r">390</td><td class="txt_r">2</td></tr>'
    '<tr><th>ワイド</th><td>4 - 7<br/>2 - 7<br/>3 - 7<br/>2 - 4<br/>3 - 4</td><td class="txt_r">120<br/>150<br/>460<br/>120<br/>340</td><td class="txt_r">1<br/>3<br/>12<br/>2<br/>10</td></tr>'
    '<tr><th>馬単</th><td>7 → 4</td><td class="txt_r">760</td><td class="txt_r">3</td></tr>'
    '<tr><th>三連複</th><td>2 - 4 - 7<br/>3 - 4 - 7</td><td class="txt_r">210<br/>790</td><td class="txt_r">1<br/>8</td></tr>'
    '<tr><th>三連単</th><td>7 → 4 → 2<br/>7 → 4 → 3</td><td class="txt_r">1,060<br/>4,050</td><td class="txt_r">3<br/>28</td></tr>'
    "</table>"
)


def test_dead_heat_payoff_rows_match_expected_combos():
    pay = RaceResultParser()._parse_payoff(BeautifulSoup(HTML, "html.parser"))
    got = W.normalize_payoff(pay)
    groups = W.finish_groups({7: 1, 4: 2, 2: 3, 3: 3, 1: 5, 5: 6, 8: 7, 6: 8})
    for bt in ("tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"):
        assert set(got[bt]) == W.winning_combos(groups, bt, n_runners=8), bt
    assert got["sanrentan"][(7, 4, 3)] == 4050
    assert len(got["wide"]) == 5 and (2, 3) not in got["wide"]
