"""tp-35: ラップ種別の分類、前週土日の定義、特徴量が『日付が厳密に前』の結果だけから作られていること（bit 単位の一致）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import tp35_lap_groups as L


@pytest.mark.parametrize("laps,expect", [
    ((12.5, 13.1), "A1"), ((13.1, 12.8), "B1"), ((12.1, 12.5), "A2"), ((12.6, 12.2), "B2"), ((11.9, 12.5), "A3"), ((12.7, 11.8), "B3"),
    ((11.4, 11.8), "A3"), ((11.9, 11.5), "B3"), ((12.3, 11.9), "B3"), ((13.5, 13.0), "X"), ((13.0, 13.4), "X"), ((np.nan, 12.0), "NA"),
])
def test_classify_book_examples_and_edges(laps, expect):
    # 資料の例（15.0-14.0-13.1-12.5 は A1 など）は (終い1F, その前の1F) = laps
    assert L.classify_lap([laps[0]], [laps[1]])[0] == expect


def test_slope_l2_is_exact_and_wood_is_approximate():
    laps = pd.DataFrame([
        {"horse_id": "a", "train_date": "2024-01-10", "race_date": "2024-01-14", "course": "栗坂", "kind": "slope", "center": "栗", "t1": 12.5, "t2": 25.0, "t3": 38.1, "t4": 52.0, "t5": np.nan, "t6": np.nan, "days_before": 4},
        {"horse_id": "b", "train_date": "2024-01-10", "race_date": "2024-01-14", "course": "CW", "kind": "wood", "center": "栗", "t1": 12.0, "t2": np.nan, "t3": 37.0, "t4": 50.0, "t5": 63.0, "t6": 78.0, "days_before": 4},
    ])
    p = L.prepare_laps(laps)
    s, w = p.iloc[0], p.iloc[1]
    assert s["l2"] == 12.5 and s["l2_approx"] == 0
    assert w["l2"] == 12.5 and w["l2_approx"] == 1                                                             # ウッド: (t3 - t1) / 2


@pytest.mark.parametrize("race_day,prev_sat_sun", [("2024-06-15", ("2024-06-08", "2024-06-09")), ("2024-06-16", ("2024-06-08", "2024-06-09")), ("2024-06-17", ("2024-06-08", "2024-06-09"))])
def test_previous_weekend_definition(race_day, prev_sat_sun):
    # 土曜開催は 7・6 日前、日曜開催は 8・7 日前、月曜開催（前の週に数える）は 9・8 日前
    rows = []
    for off in range(1, 12):
        td = (pd.Timestamp(race_day) - pd.Timedelta(days=off)).strftime("%Y-%m-%d")
        rows.append({"horse_id": "h", "train_date": td, "race_date": race_day, "course": "栗坂", "kind": "slope", "center": "栗", "t1": 12.5, "t2": 25.0, "t3": 38.0, "t4": 52.0, "t5": np.nan, "t6": np.nan, "days_before": off})
    p = L.prepare_laps(pd.DataFrame(rows))
    got = sorted(p.loc[p["is_prev_weekend"].eq(1), "train_date"].tolist())
    assert got == sorted(prev_sat_sun)


def _toy(seed: int = 0, n_days: int = 70):
    rng = np.random.default_rng(seed)
    days = [d for d in pd.date_range("2024-01-06", periods=n_days * 7, freq="D") if d.weekday() in (5, 6)][: n_days]
    runs, laps = [], []
    horses = [f"h{i}" for i in range(40)]
    trainers = [f"t{i}" for i in range(6)]
    for di, d in enumerate(days):
        for r in range(4):
            rid = f"{d:%Y%m%d}{r:02d}"
            field = list(rng.choice(horses, size=10, replace=False))
            fin = rng.permutation(10) + 1
            for h, f in zip(field, fin):
                runs.append({"race_id": rid, "horse_id": h, "date": d.strftime("%Y-%m-%d"), "trainer_id": trainers[int(h[1:]) % 6], "finish_position": int(f), "field_size": 10,
                             "venue_code": "05" if r % 2 else "06", "surface": "芝" if r < 2 else "ダ", "distance": [1200, 1600, 2000, 1800][r]})
                for off in (2, 5, 8, 9):
                    td = (d - pd.Timedelta(days=off)).strftime("%Y-%m-%d")
                    t1 = float(np.round(rng.uniform(11.0, 14.5), 1))
                    t2 = float(np.round(t1 + rng.uniform(11.0, 14.0), 1))
                    laps.append({"horse_id": h, "train_date": td, "race_date": d.strftime("%Y-%m-%d"), "course": "栗坂", "kind": "slope", "center": "栗", "t1": t1, "t2": t2, "t3": t2 + 13.0, "t4": t2 + 26.5,
                                 "t5": np.nan, "t6": np.nan, "days_before": off})
    return pd.DataFrame(runs), pd.DataFrame(laps)


def _returns(runs: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    win = np.where(runs["finish_position"].to_numpy() == 1, rng.uniform(2, 20, len(runs)), 0.0)
    place = np.where(runs["finish_position"].to_numpy() <= 3, rng.uniform(1, 5, len(runs)), 0.0)
    return pd.DataFrame({"win_ret": win, "place_ret": place})


def test_features_use_only_strictly_earlier_results_bitwise():
    runs, laps = _toy()
    base = L.build_features(runs, laps, _returns(runs, 1))
    cut = pd.Timestamp(runs["date"].iloc[-1]) - pd.Timedelta(days=21)
    late = pd.to_datetime(runs["date"]) >= cut
    # 切り替え日以降のレースの結果（着順と払戻）を入れ替える → 切り替え日より前の行、および切り替え日当日の行の特徴量は 1 ビットも変わらない
    runs2 = runs.copy()
    rng = np.random.default_rng(7)
    for rid, ix in runs2[late].groupby("race_id").groups.items():
        runs2.loc[ix, "finish_position"] = rng.permutation(runs2.loc[ix, "finish_position"].to_numpy())
    alt = L.build_features(runs2, laps, _returns(runs2, 1))
    day = pd.to_datetime(runs["date"])
    same_or_before = (day <= cut).to_numpy()
    n_nonnull = 0
    for g, f in base.items():
        cols = [c for c in f.columns if c not in ("race_id", "horse_id")]
        a, b = f.loc[same_or_before, cols].to_numpy(dtype=float), alt[g].loc[same_or_before, cols].to_numpy(dtype=float)
        assert np.array_equal(a, b, equal_nan=True), f"{g}: 結果を変えると過去/当日の特徴量が変わる"
        n_nonnull += int(np.isfinite(a).sum())
    assert n_nonnull > 1000                                                                           # テストが空振り（全て欠損）でない
    # 切り替え日より後の行は変わる（結果が入る）はずなので、検査が効いていることも確かめる
    changed = any(not np.array_equal(f.loc[~same_or_before].drop(columns=["race_id", "horse_id"]).to_numpy(dtype=float), alt[g].loc[~same_or_before].drop(columns=["race_id", "horse_id"]).to_numpy(dtype=float), equal_nan=True) for g, f in base.items())
    assert changed
