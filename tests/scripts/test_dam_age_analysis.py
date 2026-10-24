import pandas as pd

from src.scripts.data import dam_age_analysis as D


def _lineage(rows):
    return pd.DataFrame(rows, columns=["horse_id", "sire_id", "dam_id", "src"])


def test_foal_table_first_foal_and_ages():
    L = _lineage([
        ("2019100001", "2005100001", "2015100001", "subject"),   # 母 2015 生まれの最初の産駒（既知の前の産駒なし）→ 推定の初仔
        ("2021100002", "2005100001", "2015100001", "subject"),   # 同じ母の 2 番目 → 確実に 2 仔目以降
        ("2020100003", "2005100001", "2005100009", "subject"),   # 母 2005 生まれ。前の産駒が無くても古い世代は不明
        ("2019100004", "2005100001", "2016100001", "subject"),   # 母 3 歳で出産 → 確実に初仔
    ])
    T = D.foal_table(L).set_index("horse_id")
    assert T.loc["2019100001", "dam_first_foal"] == 1.0 and T.loc["2019100001", "dam_first_foal_conf"] == "推定"
    assert T.loc["2021100002", "dam_first_foal"] == 0.0 and T.loc["2021100002", "dam_first_foal_conf"] == "確実"
    assert T.loc["2021100002", "parity_min"] == 2 and T.loc["2021100002", "dam_age_at_foaling"] == 6
    assert pd.isna(T.loc["2020100003", "dam_first_foal"]) and T.loc["2020100003", "dam_first_foal_conf"] == "不明"
    assert T.loc["2019100004", "dam_first_foal_conf"] == "確実" and T.loc["2019100004", "dam_age_at_foaling"] == 3


def test_sire_stud_year_only_for_reliable_new_sires():
    rows = []
    # 新しい種牡馬（2010 生まれ）: 初年度産駒 2018、2 年目 2019
    rows += [("2018100001", "2010100001", "2012100001", "subject"), ("2019100002", "2010100001", "2013100001", "subject")]
    # 既存の種牡馬: 2012 から産駒
    rows += [("2012100003", "2000100001", "2005100001", "subject"), ("2020100004", "2000100001", "2014100001", "subject")]
    # 生年が分からない（外国の）種牡馬は、初産駒が 2019 でも k を決めない
    rows += [("2019100005", "000a000001", "2014100002", "subject")]
    T = D.foal_table(_lineage(rows)).set_index("horse_id")
    assert T.loc["2018100001", "sire_year_group"] == "1" and T.loc["2019100002", "sire_year_group"] == "2"
    assert T.loc["2020100004", "sire_year_group"] == "既存"
    assert pd.isna(T.loc["2019100005", "sire_year_group"]) or T.loc["2019100005", "sire_year_group"] is None


def test_age_band_edges():
    assert D.age_band(6) == "≤6" and D.age_band(7) == "7-11" and D.age_band(15) == "15-17" and D.age_band(18) == "18+"
    assert D.age_band(float("nan")) is None
