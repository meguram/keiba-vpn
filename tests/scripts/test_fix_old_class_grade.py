from src.scraper.parsers import RaceResultParser
from src.scripts.maintenance.fix_old_class_grade import corrected_grade


def test_class_from_name():
    f = RaceResultParser.class_from_name
    assert f("4歳以上500万下") == "1勝" and f("4歳以上1000万下") == "2勝" and f("3歳以上1600万下") == "3勝"
    assert f("3歳以上1勝クラス") == "1勝" and f("東村山特別(2勝)") == "2勝" and f("4歳以上3勝クラス") == "3勝"
    assert f("2歳未勝利") == "未勝利" and f("2歳新馬") == "新馬"
    assert f("札幌日経オープン(L)") == "" and f("日本ダービー") == "" and f("") == ""      # 名前から決められないものは空


def test_corrected_grade():
    assert corrected_grade({"grade": "その他", "race_name": "4歳以上500万下"}) == "1勝"
    assert corrected_grade({"grade": "OP", "race_name": "2歳新馬"}) == "新馬"
    assert corrected_grade({"grade": "2勝", "race_name": "3歳1勝クラス"}) == "1勝"
    assert corrected_grade({"grade": "1勝", "race_name": "4歳以上500万下"}) is None      # すでに正しければ触らない
    assert corrected_grade({"grade": "L", "race_name": "札幌日経オープン(L)"}) is None      # オープン・重賞は触らない
    assert corrected_grade({"grade": "その他"}) is None
