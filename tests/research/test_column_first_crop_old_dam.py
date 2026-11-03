"""めぐコラム『初年度産駒と高齢出産』の母の年齢・何年目の産駒か（src/research/columns/first_crop_old_dam.py）。"""

from __future__ import annotations

import pandas as pd

from src.research.columns import first_crop_old_dam as fc


def test_dam_age_and_crop_from_ids():
    ped = pd.DataFrame({
        "horse_id": ["2018100001", "2019100002", "2020100003", "2014100004", "2020100005"],
        "sire_id": ["S1", "S1", "S1", "S2", "S2"],
        "sire": "x", "dam": "y",
        "dam_id": ["2008100001", "2000100002", "000a00ffff", "2005100003", "1999100004"],
    })
    runs = pd.DataFrame({"horse_id": ped["horse_id"], "odds": 2.0, "win": 0.0})
    d = fc.frame(runs, ped).set_index("horse_id")
    assert d.loc["2018100001", "dage"] == 10 and d.loc["2018100001", "dage_bin"] == "10歳"
    assert d.loc["2019100002", "dage_bin"] == "19歳" and d.loc["2020100005", "dage_bin"] == "20歳〜"
    assert pd.isna(d.loc["2020100003", "dage"])                        # 輸入された母（生年が分からない）は除く
    assert d.loc["2018100001", "crop_bin"].startswith("1年目") and d.loc["2020100003", "crop"] == 3
    assert pd.isna(d.loc["2020100005", "crop"])                        # 2014 年から産駒のいる父は初年度が分からない


def test_clustered_interval_is_wider_when_runs_repeat():
    v = pd.Series([1.0, 1.0, 0.0, 0.0] * 25)
    one = fc.clustered(v, pd.Series(range(100)))
    same = fc.clustered(v, pd.Series([i // 2 for i in range(100)]))      # 同じ馬が 2 回ずつ同じ結果
    assert one[0] == same[0] == 0.5 and same[3] > one[3]
