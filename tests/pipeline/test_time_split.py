import pandas as pd
import pytest

from src.pipeline.models.time_split import (
    TRAINING_YEARS,
    loyo_folds,
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


def test_loyo_folds_exclude_valid_year_and_gap_but_keep_both_sides():
    df = _df()
    df = df[df["race_id"].str[:4].between("2020", "2025")].reset_index(drop=True)
    df.loc[len(df)] = {"race_id": "202101010901", "date": pd.Timestamp("2020-12-28")}     # 検証年の前 10 日以内
    folds = list(loyo_folds(df, valid_years=["2021"]))
    assert [f.valid_year for f in folds] == ["2021"]
    f = folds[0]
    train, valid = df.loc[f.train_index], df.loc[f.valid_index]
    assert (valid["race_id"].str[:4] == "2021").all()
    assert not (train["race_id"].str[:4] == "2021").any()
    assert pd.Timestamp("2020-12-28") not in set(train["date"])
    assert set(train["date"].dt.year) == {2020, 2022, 2023, 2024, 2025}


def test_loyo_folds_rejects_empty_valid_year():
    with pytest.raises(ValueError):
        list(loyo_folds(_df(), valid_years=["2010"]))


def test_tp05_folds_logs_loyo_years_only_when_set(monkeypatch, capsys):
    from src.pipeline.models.tp05_data import _folds

    df = _df()
    df = df[df["race_id"].str[:4].between("2020", "2025")].reset_index(drop=True)
    monkeypatch.delenv("OOF_LOYO_YEARS", raising=False)
    _folds(df)
    assert "OOF_LOYO_YEARS" not in capsys.readouterr().out
    monkeypatch.setenv("OOF_LOYO_YEARS", "2021,2022")
    folds = _folds(df)
    assert [f.valid_year for f in folds] == ["2021", "2022"]
    assert "OOF_LOYO_YEARS=2021,2022" in capsys.readouterr().out
