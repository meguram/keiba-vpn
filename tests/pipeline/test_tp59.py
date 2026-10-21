"""tp-59: 騎手・調教師の回収率分析の部品（レース単位ブートストラップ、as-of の関係者別成績、上積み特徴量の先読みなし）。"""

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import tp59_lib as L


def _frame(n_races=200, per_race=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        w = rng.integers(per_race)
        for h in range(per_race):
            odds = float(rng.choice([2.0, 5.0, 10.0, 30.0]))
            rows.append({"race_id": f"r{r}", "year": 2022 + r % 3, "jockey_id": f"j{h % 4}", "win": float(h == w), "top2": float(h == w), "top3": float(h == w),
                         "win_odds": odds, "ret_win": 100 * odds if h == w else 0.0, "ret_place": 0.0, "q_win": 1.0 / per_race})
    return pd.DataFrame(rows)


def test_boot_roi_pooled_matches_direct():
    d = _frame()
    r = L.boot_roi(d["race_id"].to_numpy(), d["ret_win"].to_numpy(), n_boot=200)
    assert r["roi"] == pytest.approx(100 * d["ret_win"].sum() / (100 * len(d)))
    assert r["lo"] <= r["roi"] <= r["hi"]
    assert r["races"] == 200


def test_race_equal_roi_averages_races():
    race = np.array(["a", "a", "b"])
    ret = np.array([300.0, 0.0, 100.0])
    r = L.race_equal_roi(race, ret, n_boot=50)
    assert r["roi"] == pytest.approx(125.0)


def test_boot_diff_zero_for_identical_sets():
    d = _frame()
    r = L.boot_diff(d["race_id"].to_numpy(), d["ret_win"].to_numpy(), d["race_id"].to_numpy(), d["ret_win"].to_numpy(), n_boot=100)
    assert r["diff"] == pytest.approx(0.0)
    assert r["lo"] <= 0.0 <= r["hi"]


def test_asof_table_never_uses_current_or_future_year():
    d = _frame()
    ty = L.entity_year_table(d, ["jockey_id"])
    a = L.asof_table(ty, ["jockey_id"], 2023)
    direct = d[d["year"] < 2023].groupby("jockey_id")["n" if "n" in d else "win"].size()
    assert a.set_index("jockey_id")["n"].to_dict() == direct.to_dict()
    assert a["n"].sum() == (d["year"] < 2023).sum()


def test_reliability_is_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    rows = []
    for j in range(60):
        ret = rng.choice([0.0, 500.0], size=400, p=[0.8, 0.2])
        rows.append({"n": 400.0, "ret_win": ret.sum(), "ret2_win": (ret**2).sum()})
    t = pd.DataFrame(rows)
    assert L.reliability(t, "win", 100)["reliability"] < 0.35


def test_reality_check_detects_real_edge_but_not_noise():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 100, size=(3000, 30))
    assert L.reality_check(noise, n_boot=300)["p_value"] > 0.05
    edge = noise.copy()
    edge[:, 3] += 40
    assert L.reality_check(edge, n_boot=300)["p_value"] < 0.05


def test_max_drawdown_orders_by_date():
    r = L.max_drawdown(np.array(["2022-01-01", "2022-01-02", "2022-01-03"]), np.array(["a", "b", "c"]), np.array([100.0, -300.0, 50.0]))
    assert r["max_drawdown_yen"] == pytest.approx(300.0)
    assert r["final_profit_yen"] == pytest.approx(-150.0)


def test_add_excess_uses_only_earlier_years_for_band_expectation():
    d = _frame()
    d["jockey_id"] = d["jockey_id"].astype(str)
    out = L.add_excess(d)
    y = 2024
    prior = d[d["year"] < y].assign(b=L.odds_band(d[d["year"] < y]["win_odds"])).groupby("b")["ret_win"].mean()
    row = out[out["year"] == y].iloc[0]
    assert row["ex_win"] == pytest.approx(row["ret_win"] - prior[L.odds_band(pd.Series([row["win_odds"]])).iloc[0]])


def test_payoff_win_parses_single_and_dead_heat():
    from src.scripts.data import tp59_track as T

    assert T.payoff_win('{"単勝": {"numbers": "7", "payout": "380", "popularity": "3"}}') == {7: 380.0}
    assert T.payoff_win('{"単勝": [{"numbers": "7", "payout": "380"}, {"numbers": "9", "payout": "540"}]}') == {7: 380.0, 9: 540.0}
    assert T.payoff_win("not json") == {}


def test_residual_flags_use_only_years_up_to_cutoff():
    from src.scripts.data import tp59_track as T

    rows = []
    for y, w in ((2024, 1.0), (2025, 0.0), (2026, 0.0)):
        for i in range(10):
            rows.append({"year": y, "jockey_id": "j1", "trainer_id": "t1", "win": w if i == 0 else 0.0, "q_win": 0.05})
    m = pd.DataFrame(rows)
    assert T.residual_flags(m, 2024)["jockey_pos"] == ["j1"]          # 2024 年: 1 勝 > 市場の見込み 0.5
    assert T.residual_flags(m, 2026)["jockey_pos"] == []              # 2026 年まで含めると 1 勝 < 1.5


def test_family_rank_ignores_current_year_results():
    from src.scripts.data import tp59_rules as R

    d = _frame(n_races=300)
    d["trainer_id"] = d["jockey_id"]
    d["venue_code"] = "05"
    d["surface"], d["dist_band"], d["cls"], d["win_odds"] = "芝", "d_mile", "c1_maiden", d["win_odds"]
    d["ret_place"] = d["ret_win"] * 0.5
    d = L.add_excess(d)
    d = R.add_keys(d)
    base = R.family_rank(d, "j", 2024, "win", "roi", 30)
    d2 = d.copy()
    d2.loc[d2["year"] == 2024, "ret_win"] = 0.0
    d2.loc[d2["year"] == 2024, "ex_win"] = -999.0
    alt = R.family_rank(d2, "j", 2024, "win", "roi", 30)
    pd.testing.assert_series_equal(base, alt)


def test_rs_features_are_asof_and_missing_before_oof_years(tmp_path, monkeypatch):
    from src.scripts.data import tp59_feat as F

    dates = pd.to_datetime(["2021-12-01", "2022-01-05", "2022-01-12", "2022-01-12", "2022-01-19"])
    df = pd.DataFrame({"race_id": [f"r{i}" for i in range(5)], "horse_id": [f"h{i}" for i in range(5)], "date": dates,
                       "jockey_id": ["j1"] * 5, "trainer_id": ["t1"] * 5, "finish_position": [1, 1, 5, 2, 1]})
    oof = pd.DataFrame({"race_id": df["race_id"][1:], "horse_id": df["horse_id"][1:], "year": 2022, "p_win": [0.1, 0.1, 0.1, 0.1], "p_top3": [0.3, 0.3, 0.3, 0.3]})
    f = tmp_path / "oof.parquet"
    oof.to_parquet(f)
    monkeypatch.setattr(F, "BASE_OOF", f)
    r = F.rs_features(df, k=1.0)
    assert np.isnan(r.loc[0, "rs_j_win"])                       # OOF のある年より前の行は欠損
    assert r.loc[1, "rs_j_win"] == pytest.approx(0.0)           # 最初の日は履歴なし
    assert r.loc[2, "rs_j_win"] == pytest.approx(r.loc[3, "rs_j_win"])   # 同日の結果は使わない
    assert r.loc[2, "rs_j_win"] == pytest.approx((1 - 0.1) / (1 + 1.0))
    assert r.loc[4, "rs_j_win"] == pytest.approx((0.9 - 0.1 - 0.1) / 4.0)
