"""結果ページのヘッダ（実ページから抜き出した断片）を、表記ごとにパースできること。"""

import pytest

from src.scraper.parsers import RaceResultParser

TPL = """<html><body><div class="data_intro"><dl class="racedata fc"><dt>4 R</dt><dd><h1>{name}</h1>
<p><diary_snap_cut><span>{course}&nbsp;/&nbsp;
天候 : 曇&nbsp;/&nbsp;
{surf} : 稍重&nbsp;&nbsp;/&nbsp;
発走 : 11:35</span></diary_snap_cut></p></dd></dl>
<p class="smalltxt">2026年09月27日 4回中山9日目&nbsp;{name}&nbsp;&nbsp;(混)[指](馬齢)</p></div>
<table class="race_table_01"></table></body></html>"""


@pytest.mark.parametrize("course,surf,expected", [
    ("芝右 外1200m", "芝", ("芝", "右", 1200)),
    ("芝右1600m", "芝", ("芝", "右", 1600)),
    ("芝左 内2周 2000m", "芝", ("芝", "左", 2000)),
    ("ダ右1800m", "ダート", ("ダート", "右", 1800)),
    ("芝直線1000m", "芝", ("芝", "直", 1000)),
    ("障芝 右 外・内 3170m", "芝", ("障", "右", 3170)),
    ("障芝 外-内2890m", "芝", ("障", "", 2890)),
])
def test_header_variants(course, surf, expected):
    d = RaceResultParser().parse(TPL.format(name="2歳新馬", course=course, surf=surf), race_id="202606040904")
    assert (d["surface"], d["direction"], d["distance"]) == expected
    assert d["weather"] == "曇" and d["track_condition"] == "稍重" and d["start_time"] == "11:35"
    assert d["date"] == "2026-09-27" and d["round"] == 4


@pytest.mark.parametrize("name,grade", [
    ("4歳以上500万下", "1勝"), ("4歳以上1000万下", "2勝"), ("3歳以上1600万下", "3勝"),
    ("3歳以上1勝クラス", "1勝"), ("フェアウェルS(3勝)", "3勝"), ("2歳未勝利", "未勝利"),
])
def test_old_class_names_map_to_current_grade(name, grade):
    d = RaceResultParser().parse(TPL.format(name=name, course="芝右1600m", surf="芝"), race_id="201906040904")
    assert d["grade"] == grade


def test_old_class_name_wins_over_page_wide_text():
    html = TPL.format(name="4歳以上500万下", course="芝右1600m", surf="芝").replace("</body>", "<div>次のレース オープン 未勝利</div></body>")
    assert RaceResultParser().parse(html, race_id="201906040904")["grade"] == "1勝"
