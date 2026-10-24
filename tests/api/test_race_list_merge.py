from src.api.app import merge_db_and_list_races


def _r(rid, date=None):
    return {"race_id": rid, "date": date} if date else {"race_id": rid}


def test_db_missing_races_are_filled_from_list_of_same_date():
    db = [_r("202608040601"), _r("202608040602")]
    lst = [_r("202608040601", "20261017"), _r("202608040602", "20261017"), _r("202608040609", "20261017")]
    out = merge_db_and_list_races(db, lst, "20261017")
    assert [r["race_id"] for r in out] == ["202608040601", "202608040602", "202608040609"]


def test_list_rows_of_other_dates_are_not_added():
    db = [_r("202608040601")]
    lst = [_r("202608040601", "20261017"), _r("202608030101", "20261010"), _r("202608030102")]
    assert [r["race_id"] for r in merge_db_and_list_races(db, lst, "20261017")] == ["202608040601"]


def test_without_db_the_list_is_used():
    lst = [_r("202608040601", "20261017")]
    assert merge_db_and_list_races([], lst, "20261017") == lst
