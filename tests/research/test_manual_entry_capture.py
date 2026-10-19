import csv

from tests.research.test_manual_entry_server import _setup


def test_capture_matches_current_horse_and_saves(tmp_path):
    s, d = _setup(tmp_path)
    assert s.capture("https://sporthorse-data.com/pedigree/beta-1", "Beta Pedigree", "30-Jan-1980")["ok"] is False     # 表示中は alpha
    r = s.capture("https://sporthorse-data.com/pedigree/alpha-1", "Alpha | SporthorseData", "30-Jan-1970")
    assert r["ok"] is True
    rows = list(csv.reader(open(d / "birth_manual.csv", encoding="utf-8")))
    assert rows[-1][:5] == ["a", "Alpha", "1970", "1", "30"] and rows[-1][5].endswith("/alpha-1")


def test_capture_rejects_wrong_year_and_none(tmp_path):
    s, d = _setup(tmp_path)
    assert s.capture("https://sporthorse-data.com/pedigree/alpha", "Alpha", "30-Jan-1999")["ok"] is False             # 生年違い
    assert s.capture_none("https://sporthorse-data.com/pedigree/alpha", "Alpha")["ok"] is True
    assert [r["horse_id"] for r in s.pending()][:1] == ["b"]
    # スラッグが違ってもタイトルの馬名で一致すれば受け付ける
    assert s.capture("https://sporthorse-data.com/pedigree/bt-99", "Beta - Thoroughbred", "03-Mar-1980")["ok"] is True
