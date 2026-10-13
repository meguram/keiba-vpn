import pandas as pd
import pytest

from src.pipeline.models.time_split import (
    TRAINING_YEARS,
    restrict_to_training_years,
    yearly_expanding_folds,
)


def _df():
    rows = []
    for year in range(2019, 2027):
        for month in range(1, 13):
            rows.append(
                {
                    "race_id": f"{year}01010{month:02d}01",
                    "date": pd.Timestamp(year=year, month=month, day=15),
                }
            )
    return pd.DataFrame(rows)


def test_2026_and_earlier_than_2020_are_not_training_years():
    out = restrict_to_training_years(_df())
    years = set(out["race_id"].str[:4])
    assert years == set(TRAINING_YEARS)
    assert "2026" not in years and "2019" not in years


def test_folds_have_no_overlap_and_respect_gap():
    df = restrict_to_training_years(_df())
    folds = list(yearly_expanding_folds(df))
    assert [f.valid_year for f in folds] == ["2022", "2023", "2024"]
    for f in folds:
        train, valid = df.loc[f.train_index], df.loc[f.valid_index]
        assert set(train["race_id"]).isdisjoint(valid["race_id"])
        assert train["date"].max() < f.train_end <= f.valid_start
        assert (valid["race_id"].str[:4] == f.valid_year).all()
        assert (train["race_id"].str[:4] >= "2020").all()


def test_purge_and_embargo_drop_rows_just_before_valid():
    df = restrict_to_training_years(_df())
    boundary = pd.DataFrame(
        {
            "race_id": ["202301010101", "202301010102"],
            "date": [pd.Timestamp("2022-12-25"), pd.Timestamp("2022-12-20")],
        }
    )
    df = pd.concat([df, boundary], ignore_index=True)
    fold = next(f for f in yearly_expanding_folds(df) if f.valid_year == "2023")
    train_ids = set(df.loc[fold.train_index, "race_id"])
    assert "202301010101" not in train_ids  # 7+3 日以内は除外
    assert "202301010102" in train_ids  # 12 日前は残る


def test_valid_year_outside_training_years_is_rejected():
    with pytest.raises(ValueError):
        list(yearly_expanding_folds(restrict_to_training_years(_df()), valid_years=("2026",)))


def test_missing_date_is_rejected():
    df = restrict_to_training_years(_df())
    df.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError):
        list(yearly_expanding_folds(df))
