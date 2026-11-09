"""persona1_search の枠組みの確認（作り物の表。GCS・ローカルの大きな表は読まない）。"""
import numpy as np
import pandas as pd

from src.scripts.data import persona1_search as P


def _frame(n_races=600, informative=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        k = 10
        true = rng.dirichlet(np.ones(k) * 3)
        win = rng.choice(k, p=true)
        q = true * rng.uniform(0.85, 1.15, k)      # 市場は真の確率にほぼ等しい（控除率 20%）
        q = q / q.sum()
        odds = 0.8 / q
        pm = true if informative else rng.dirichlet(np.ones(k) * 3)
        for h in range(k):
            rows.append({"race_id": f"R{r:04d}", "date": f"{2022 + r * 4 // n_races}-{1 + (r * 4 // 1) % 12:02d}-{1 + (r * 7) % 28:02d}", "horse_number": h + 1, "win_odds": odds[h],
                         "place_odds_min": max(1.0, odds[h] / 3), "place_odds_max": max(1.1, odds[h] / 2.5), "p_win": pm[h], "p_top3": min(0.99, pm[h] * 3),
                         "ret_win": 100 * odds[h] if h == win else 0.0, "ret_place": 0.0})
    return pd.DataFrame(rows)


def test_yen_units_are_normalised_to_multiples(tmp_path):
    """100 円あたりの円（例 450）で入っている払戻は、倍率（4.5）に直る。直さないと回収率が 100 倍に出る。"""
    d = _frame(n_races=300)
    d["ret_place"] = np.where(d["ret_win"] > 0, 150.0, 0.0)          # 複勝も円で入れる
    p = tmp_path / "f.parquet"
    d.to_parquet(p)
    r = P.run(p, 2023, 2025, tmp_path / "out")
    assert r["K"] == len(P.candidates())
    grid = pd.read_csv(tmp_path / "out" / "grid_2023_2025_s7.csv")
    assert grid["test_roi"].max() < 20        # 倍率なら高くても数倍。円のままなら数百になる
    assert "caveat" in r                       # 確定オッズを使った上限の値、という留保が結果に入る


def test_uninformative_model_gets_no_confirmation_of_edge(tmp_path):
    d = _frame(n_races=800, informative=False, seed=3)
    p = tmp_path / "f.parquet"
    d.to_parquet(p)
    r = P.run(p, 2023, 2025, tmp_path / "out")
    p_rc = r["reality_check"]["p"]
    assert p_rc is None or p_rc > 0.05


def test_fit_ab_ignores_other_years():
    """a, b の選択は渡した表（前の年）だけで決まる。テスト年の行を足しても変わらない、は呼び出し側の分け方で保証するので、
    ここでは『同じ表なら同じ値』と、列の順序に依存しないことを確かめる。"""
    d = _frame(n_races=200, seed=5)
    d["q_win"] = (1 / d["win_odds"]) / (1 / d["win_odds"]).groupby(d["race_id"]).transform("sum")
    d["win"] = (d["ret_win"] > 0).astype(int)
    assert P.fit_ab(d) == P.fit_ab(d.sample(frac=1.0, random_state=1).sort_values("race_id"))


def test_roi_lower_matches_hand_calculation():
    pr = pd.DataFrame({"pl": [1.0, -1.0] * 200, "n": [1, 1] * 200}, index=[f"r{i}" for i in range(400)])
    roi, lb, n = P.roi_lower(pr)
    assert n == 400 and abs(roi - 1.0) < 1e-12            # 損益ゼロ → 回収率 100%
    se = np.sqrt((pr["pl"] ** 2).sum()) / 400               # 残差 = pl（平均 0）
    assert abs(lb - (1 - P.Z_SELECT * se)) < 1e-12


def test_place_requires_eight_runners():
    d = pd.DataFrame({"race_id": ["a"] * 7 + ["b"] * 8, "win_odds": 5.0, "place_odds_min": 2.0, "place_odds_max": 3.0, "n_run": [7] * 7 + [8] * 8,
                      "p_win": 0.2, "p_top3": 0.9, "pw_b": 0.2, "p3_b": 0.9})
    ok = P.select(d, ("place", "model", 1.0, None))
    assert not ok[:7].any() and ok[7:].all()                # 7 頭以下は 2 着までしか払われないので買わない


def test_race_profit_counts_bets_and_profit():
    d = pd.DataFrame({"race_id": ["a", "a", "b"], "win_odds": [2.0, 10.0, 3.0], "place_odds_min": [1.2, 2.0, 1.5], "place_odds_max": [1.4, 3.0, 1.8],
                      "p_win": [0.6, 0.2, 0.5], "p_top3": [0.9, 0.5, 0.8], "pw_b": [0.6, 0.2, 0.5], "p3_b": [0.9, 0.5, 0.8],
                      "ret_win": [2.0, 0.0, 0.0], "ret_place": [1.3, 0.0, 1.6]})
    pr = P.race_profit(d, ("win", "model", 1.1, None))      # EV: 1.2, 2.0, 1.5 → 全部買う
    assert pr.loc["a", "n"] == 2 and abs(pr.loc["a", "pl"] - ((2.0 - 1) + (0.0 - 1))) < 1e-9
    assert pr.loc["b", "n"] == 1 and abs(pr.loc["b", "pl"] - (-1.0)) < 1e-9


def test_decision_odds_noise_changes_only_decision_odds(tmp_path):
    """判断用のオッズだけが動き、払戻は確定のまま。市場の確率はレース内で合計 1。"""
    d = _frame(n_races=120, seed=9)
    pre = d[["race_id", "horse_number"]].copy()
    pre["phase"] = "t05"
    pre["win_odds"] = d["win_odds"] * np.exp(np.random.default_rng(0).normal(-0.1, 0.2, len(d)))
    pre["place_odds_min"] = d["place_odds_min"] * 0.95
    fin = d[["race_id", "horse_number", "win_odds", "place_odds_min"]]
    pre.to_parquet(tmp_path / "pre.parquet")
    fin.to_parquet(tmp_path / "fin.parquet")
    d["q_win"] = 0.0
    out = P.decision_odds_noise(d, "t05", np.random.default_rng(1), tmp_path / "pre.parquet", tmp_path / "fin.parquet")
    assert (out["ret_win"].to_numpy() == d["ret_win"].to_numpy()).all()
    assert not np.allclose(out["win_odds"], d["win_odds"])
    assert np.allclose(out.groupby("race_id")["q_win"].sum(), 1.0)


def test_decision_odds_noise_keeps_takeout_per_race(tmp_path):
    """レース内の 1/オッズ の合計（控除率）は、再現後も元の確定オッズの値に保たれる（馬ごとに独立に引くと崩れる）。"""
    d = _frame(n_races=150, seed=11)
    d["place_odds_max"] = d["place_odds_max"].where(d["horse_number"] > 1, 0)       # 複勝オッズの無い馬は 0 のまま
    pre = d[["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max"]].copy()
    pre["phase"] = "t05"
    rng = np.random.default_rng(2)
    pre["win_odds"] = d["win_odds"] * np.exp(rng.normal(-0.1, 0.4, len(d)))
    pre["place_odds_min"] = d["place_odds_min"] * np.exp(rng.normal(-0.05, 0.2, len(d)))
    pre.to_parquet(tmp_path / "pre.parquet")
    d[["race_id", "horse_number", "win_odds", "place_odds_min"]].to_parquet(tmp_path / "fin.parquet")
    d["q_win"] = 0.0
    out = P.decision_odds_noise(d, "t05", np.random.default_rng(3), tmp_path / "pre.parquet", tmp_path / "fin.parquet")
    before = (1 / d["win_odds"]).groupby(d["race_id"]).sum()
    after = (1 / out["win_odds"]).groupby(out["race_id"]).sum()
    assert np.allclose(before, after, rtol=0.02)           # 下限 1.0 の丸めで少しだけずれる
    assert (out.loc[d["place_odds_max"] == 0, "place_odds_max"] == 0).all()
    has = out["place_odds_max"] > 0
    assert (out.loc[has, "place_odds_max"] >= out.loc[has, "place_odds_min"]).all()
