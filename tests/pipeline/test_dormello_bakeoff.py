from src.scripts.data import dormello_bakeoff as db


def _males():
    # (世代, 位置, 祖先ID): 父、父父、母父、父父父、母父父...
    return [(1, 0, "S"), (2, 0, "SS"), (2, 2, "DS"), (3, 0, "SSS"), (3, 2, "SDS"), (3, 4, "DSS"), (3, 6, "DDS")]


def test_activity_features_basic():
    births = {"S": "2008", "SS": "2000", "DS": "2005", "SSS": "1990", "SDS": "1995", "DSS": "1999", "DDS": "2004"}
    f = db.activity_features(_males(), 2016, births, 8)
    # 父 (2016-2008)%8=0 -> ゾーン、父父 16%8=0、母父 11%8=3、SSS 26%8=2、SDS 21%8=5、DSS 17%8=1、DDS 12%8=4
    assert f["act_sire"] == 0 and f["act_sire_zero"] == 1.0
    assert f["act_dsire"] == 3
    assert f["act_max3"] == 5 and f["act_max3_gen"] == 3 and f["act_max3_is_sire"] == 0.0
    assert f["act_n_zero3"] == 2 and f["act_n_known3"] == 7


def test_activity_ties_prefer_younger_generation_and_sire():
    births = {"S": "2010", "SS": "2008", "DS": "2008"}
    f = db.activity_features([(1, 0, "S"), (2, 0, "SS"), (2, 2, "DS")], 2018, births, 8)
    # 父 8%8=0、SS 10%8=2、DS 10%8=2（同値）-> 世代は同じ 2 で最小、父ではない
    assert f["act_max3"] == 2 and f["act_max3_gen"] == 2 and f["act_max3_is_sire"] == 0.0
    f2 = db.activity_features([(1, 0, "S"), (2, 0, "SS")], 2018, {"S": "2012", "SS": "2008"}, 8)
    # 父 6、SS 2 -> 最大は父
    assert f2["act_max3"] == 6 and f2["act_max3_gen"] == 1 and f2["act_max3_is_sire"] == 1.0


def test_unknown_birth_is_skipped():
    f = db.activity_features([(1, 0, "S"), (2, 0, "X")], 2018, {"S": "2012"}, 8)
    assert f["act_n_known3"] == 1 and f["act_max3"] == 6
    g = db.activity_features([(1, 0, "X")], 2018, {}, 8)
    assert g["act_sire"] is None and "act_max3" not in g


def test_placebo_cycle_changes_phase():
    births = {"S": "2008"}
    assert db.activity_features([(1, 0, "S")], 2018, births, 8)["act_sire"] == 2
    assert db.activity_features([(1, 0, "S")], 2018, births, 7)["act_sire"] == 3
    assert db.activity_features([(1, 0, "S")], 2018, births, 9)["act_sire"] == 1
