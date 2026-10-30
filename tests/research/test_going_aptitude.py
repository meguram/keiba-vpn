"""芝道悪適性・ダートタフ適正（src/research/race/going_aptitude.py）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.race import going_aptitude as ga


def _runs(rows: list[tuple]) -> pd.DataFrame:
    """rows: (horse_id, date, surface, x, finish_position)。頭数は 10 固定。"""
    df = pd.DataFrame(rows, columns=["horse_id", "date", "surface", "x", "finish_position"])
    df["date"] = pd.to_datetime(df["date"])
    df["field_size"] = 10
    df["perf"] = -(df["finish_position"] - 1) / 9
    return df


def _population(surface: str, n: int = 40) -> list[tuple]:
    """判定可能な馬 n 頭（パワー走・非パワー走が各 3 走）。差が馬ごとに違うよう着順を散らす。"""
    power_x, normal_x = (9.0, 10.0) if surface == "芝" else (5.0, 8.0)
    rows = []
    for i in range(n):
        h = f"H{i}"
        for k in range(3):
            rows.append((h, f"2025-01-{k + 1:02d}", surface, power_x, 1 + (i + k) % 10))
            rows.append((h, f"2025-02-{k + 1:02d}", surface, normal_x, 1 + (i * 3 + k) % 10))
    return rows


def test_power_flag_direction_and_na():
    s = pd.Series(["芝", "芝", "ダート", "ダート", "芝"])
    x = pd.Series([9.2, 9.3, 6.4, 6.5, np.nan])
    f = ga.power_flag(s, x)
    assert list(f.iloc[:4]) == [True, False, True, False]
    assert pd.isna(f.iloc[4])


def test_undeterminable_is_nan_not_zero():
    rows = _population("芝")
    rows += [("NEW", "2025-01-01", "芝", 9.0, 1)]                               # パワー走 1 走だけ
    rows += [("ONE", f"2025-01-{k + 1:02d}", "芝", 10.0, 3) for k in range(5)]   # 非パワー走だけ
    out = ga.going_scores(_runs(rows), ["H0", "NEW", "ONE", "NONE"], "2026-01-01").set_index("horse_id")
    assert np.isfinite(out.loc["H0", "going_turf"])
    for h in ("NEW", "ONE", "NONE"):
        assert np.isnan(out.loc[h, "going_turf"])
    assert out.loc["NONE", "n_power_going_turf"] == 0
    assert out.loc["ONE", "n_normal_going_turf"] == 5
    assert not out.loc["NONE", "low_going_turf"]        # 判定不可は「低信頼」ではなく不明
    assert out.loc["H0", "low_going_turf"]             # 各3走は m_eff=1.5 で低信頼


def test_turf_and_dirt_are_separate():
    out = ga.going_scores(_runs(_population("芝")), ["H0"], "2026-01-01").set_index("horse_id")
    assert np.isfinite(out.loc["H0", "going_turf"])
    assert np.isnan(out.loc["H0", "going_dirt"])        # ダートの走が無ければ不明（芝の値を流用しない）


def test_only_runs_before_as_of():
    rows = _population("ダート")
    runs = _runs(rows)
    assert np.isnan(ga.going_scores(runs, ["H0"], "2025-01-01").set_index("horse_id").loc["H0", "going_dirt"])
    assert np.isnan(ga.going_scores(runs, ["H0"], "2025-02-01").set_index("horse_id").loc["H0", "going_dirt"])   # パワー走しか無い
    assert np.isfinite(ga.going_scores(runs, ["H0"], "2025-03-01").set_index("horse_id").loc["H0", "going_dirt"])


def test_too_few_determinable_horses_all_nan():
    out = ga.going_scores(_runs(_population("芝", n=10)), ["H0", "H1"], "2026-01-01")
    assert out["going_turf"].isna().all()


def test_runs_without_moisture_are_not_counted():
    rows = _population("芝")
    rows += [("H0", f"2025-03-{k + 1:02d}", "芝", np.nan, 1) for k in range(5)]
    out = ga.going_scores(_runs(rows), ["H0"], "2026-01-01").set_index("horse_id")
    assert out.loc["H0", "n_power_going_turf"] == 3
    assert out.loc["H0", "n_normal_going_turf"] == 3
