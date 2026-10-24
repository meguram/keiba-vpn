import json
from datetime import date

import pytest

from src.research.pedigree import dormello_activity as da


def test_phase_matches_published_examples():
    # Halo 1969 -> ディープインパクト(2001 受胎) / サンデーサイレンス 1986 / ディープインパクト 2002 -> 2016 受胎
    assert da.phase(2001, 1969) == 0
    assert da.phase(2001, 1986) == 7
    assert da.phase(2016, 2002) == 6


def test_zone_is_not_resolved_without_manual():
    assert da.effective_value(0) is None
    assert da.effective_value(0, 8) == 8
    assert da.effective_value(5) == 5
    assert da.effective_value(5, 8) == 5


def test_conception_year():
    assert da.conception_year(2017) == 2016
    assert da.conception_year(2017, date(2017, 1, 5)) == 2016
    assert da.conception_year(2017, date(2017, 12, 28)) == 2017


def test_birth_year_of():
    assert da.birth_year_of("2017104238", {}) == 2017
    assert da.birth_year_of("000a001025", {}) is None
    assert da.birth_year_of("000a001025", {"000a001025": "1969-03-24"}) == 1969


def _ped(tmp_path, hid, ancestors):
    d = tmp_path / "ped" / hid[:4]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{hid}.json").write_text(json.dumps({"horse_id": hid, "ancestors": ancestors}), encoding="utf-8")


def test_build_review_keeps_manual_values(tmp_path):
    # 対象馬 2017 生まれ（受胎 2016）、父 = 2008 生まれ -> 差 8 -> ゾーン、母(位置奇数)は対象外
    _ped(tmp_path, "2017100001", [
        {"generation": 1, "position": 0, "name": "父", "horse_id": "2008100001"},
        {"generation": 1, "position": 1, "name": "母", "horse_id": "2009100002"},
        {"generation": 2, "position": 0, "name": "父父", "horse_id": "000a000001"},
    ])
    review = tmp_path / "review.csv"
    stats = da.build_review([tmp_path / "ped"], {}, review)
    assert stats["zone_rows"] == 1 and stats["pairs_with_birth_year"] == 1
    rows = review.read_text(encoding="utf-8").splitlines()
    assert rows[1].startswith("2008100001,父,2008,2016,1,1,,")

    review.write_text(rows[0] + "\n" + "2008100001,父,2008,2016,1,1,8,手入力\n", encoding="utf-8")
    da.build_review([tmp_path / "ped"], {}, review)
    assert da.load_manual(review) == {("2008100001", 2016): (8, "手入力")}


def test_manual_value_out_of_range(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text(",".join(da.REVIEW_COLUMNS) + "\nx,n,2000,2008,1,1,9,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        da.load_manual(p)
