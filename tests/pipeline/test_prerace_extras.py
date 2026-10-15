from src.pipeline.features import prerace_extras as pe


def test_extract_race_merges_oikiri_and_comment_by_horse():
    oik = {"entries": [{"horse_id": "h1", "evaluation": "A"}, {"horse_id": "h2", "evaluation": "d"}, {"horse_id": "h3", "evaluation": ""}]}
    com = {"entries": [{"horse_id": "h1", "comment": "状態は良化していて仕上がりも上々。期待したい。"}, {"horse_id": "h4", "comment": "まだ太い。ひと叩きしたい"}]}
    rows = {r["horse_id"]: r for r in pe.extract_race("r", oik, com)}
    assert rows["h1"]["oikiri_eval"] == 4.0 and rows["h1"]["comment_pos"] >= 3 and rows["h1"]["has_comment"] == 1.0
    assert rows["h2"]["oikiri_eval"] == 1.0 and "has_comment" not in rows["h2"]
    assert rows["h3"]["oikiri_eval"] is None
    assert rows["h4"]["comment_neg"] >= 3 and "oikiri_eval" not in rows["h4"]


def test_result_words_are_flagged():
    assert pe.has_result_words("内容のある競馬で2着に好走") and not pe.has_result_words("仕上がりは上々")


def test_handles_missing_docs():
    assert pe.extract_race("r", None, None) == []
