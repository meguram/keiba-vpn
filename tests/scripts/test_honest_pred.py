import numpy as np
import pandas as pd
import pytest

from src.scripts.data import honest_pred as hp


def test_parse_years():
    assert hp.parse_years(None) is None
    assert hp.parse_years("") is None
    assert hp.parse_years("2020-2025") == [2020, 2021, 2022, 2023, 2024, 2025]
    assert hp.parse_years("2020, 2022-2023") == [2020, 2022, 2023]
    with pytest.raises(ValueError):
        hp.parse_years("2025-2020")


def test_caveat_marks_only_suspect_years():
    assert hp._caveat("venue", "東京", [2020, 2021]) == ""
    assert hp._caveat("venue", "東京", [2017, 2020]) == hp.SUSPECT_NOTE
    assert hp._caveat("year", 2018, [2017, 2018, 2020]) == hp.SUSPECT_NOTE
    assert hp._caveat("year", 2020, [2017, 2018, 2020]) == ""
    assert hp._caveat("era", "2017-19(低)", [2017, 2020]) == hp.SUSPECT_NOTE
    assert hp._caveat("era", "2022-24", [2017, 2022]) == ""


def _synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    rows = []
    for y in (2016, 2017, 2020, 2021):
        for r in range(40):
            n = 8
            w = rng.integers(0, n)
            p = rng.dirichlet(np.ones(n))
            for h in range(n):
                rows.append({"race_id": f"{y}{r:08d}", "horse_id": f"h{h}", "date": pd.Timestamp(y, 6, 1 + r % 28), "year": y,
                             "finish_position": 1 if h == w else 2 + (h > w) * 0 + h % 5 + 1, "p_win": p[h], "p_top2": min(1, 2 * p[h]),
                             "p_top3": min(1, 3 * p[h]), "source": "x", "reliability": "高",
                             "venue": "東京" if r % 2 else "中山", "surface": "芝", "distance": 1600, "grade": "未勝利",
                             "race_class": "a", "track_condition": "良", "field_size": n})
    d = pd.DataFrame(rows)
    out = tmp_path / "out"
    out.mkdir()
    d[[c for c in d.columns if c not in ("venue", "surface", "distance", "grade", "race_class", "track_condition", "field_size")]].to_parquet(out / "honest_pred_2016_2025.parquet")
    (tmp_path / "features").mkdir()
    d[["race_id", "horse_id", "venue", "surface", "distance", "grade", "race_class", "track_condition", "field_size"]].to_parquet(tmp_path / "features/train_table.parquet")
    monkeypatch.setattr(hp, "OUT", out)
    monkeypatch.setattr(hp, "BASE", tmp_path)
    return out


def test_analyze_years_writes_separate_table_with_caveats(tmp_path, monkeypatch):
    out = _synthetic(tmp_path, monkeypatch)
    hp.analyze(min_races=10, boots=20)
    hp.analyze(min_races=10, boots=20, years=[2020, 2021])
    full, sub = pd.read_csv(out / "stage_accuracy.csv"), pd.read_csv(out / "stage_accuracy_2020-2021.csv")
    assert full["caveat"].fillna("").str.contains("漏れ疑い").any() and not sub["caveat"].fillna("").str.contains("漏れ疑い").any()
    assert set(sub.loc[sub["dim"] == "year", "value"]) == {"2020", "2021"}
    assert set(full.loc[full["dim"] == "year", "value"]) == {"2017", "2020", "2021"}   # 2016 は既定で除外
    assert "漏れ疑い" in (out / "NOTES.txt").read_text(encoding="utf-8")
