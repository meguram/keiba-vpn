"""ドリフト診断（drift_diagnose / drift_diagnose_features / drift_mask_experiment の純関数部分）の合成データテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.data import drift_diagnose as D
from src.scripts.data import drift_diagnose_features as DF
from src.scripts.data import drift_mask_experiment as ME


def _eval_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2024, 2026):
        for month in (2, 7):
            for r in range(40):
                n = 10
                pm = rng.dirichlet(np.ones(n) * 2)
                sharp = 0.6 if (year == 2026 and month == 7) else 1.0
                lg = sharp * np.log(pm)
                pw = np.exp(lg) / np.exp(lg).sum()
                win = rng.choice(n, p=pm)
                for i in range(n):
                    rows.append({"race_id": f"{year}{month:02d}{r:04d}", "horse_id": f"h{i}", "year": year, "month": month, "date": pd.Timestamp(year, month, 5),
                                 "src": "forward" if year == 2026 else "oof", "p_win": pw[i], "p_top2": min(2 * pw[i], 1), "p_top3": min(3 * pw[i], 1),
                                 "p_mkt": pm[i], "y_win": float(i == win), "y_top2": float(i <= 1), "y_top3": float(i <= 2),
                                 "surface": "芝" if i % 2 else "ダート", "popularity": i + 1})
    return pd.DataFrame(rows)


def test_monthly_and_period_summary_flag_the_flat_month():
    df = _eval_frame()
    mo = {m["ym"]: m for m in D.monthly(df)}
    assert mo["2026-07"]["sd_ratio"] < mo["2026-02"]["sd_ratio"]
    assert mo["2026-07"]["gap"] > mo["2026-02"]["gap"]
    ps = D.period_summary(df)
    assert ps["JunOct"]["diff"] > ps["JanMay"]["diff"]


def test_calibration_reports_recalibration_share():
    df = _eval_frame()
    cal = D.calibration(df)
    assert set(cal) == {"base", "26_JanMay", "26_JunOct"}
    assert cal["26_JunOct"]["model"]["ece"] >= 0
    assert abs(cal["base"]["gap_raw"] - cal["base"]["gap_after_recalibration"]) < 0.05


def test_strata_decomposition_sums():
    df = _eval_frame()
    t = D.strata_table(df.assign(n_prior=3, age=3.0, distance=1600, field_size=10, track_condition="良", venue="東京", grade="1勝", race_class="x"), (6, 10))
    for col, v in t.items():
        assert abs(v["mix"] + v["within"] - v["total"]) < 1e-9, col


def test_feature_families_and_adversarial_detect_shift():
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"race_id": [f"a{i // 10}" for i in range(400)], "horse_id": "h", "tl_x": rng.normal(0, 1, 400), "f2": rng.normal(0, 1, 400)})
    b = pd.DataFrame({"race_id": [f"b{i // 10}" for i in range(400)], "horse_id": "h", "tl_x": rng.normal(1.5, 1, 400), "f2": rng.normal(0, 1, 400)})
    assert DF.family("tl_x") == "TL(tl_)" and DF.family("mtl_t_rank_norm") == "AUX(mtl_)"
    res = DF.adversarial(a, b, ["tl_x", "f2"])
    assert res["auc_mean"] > 0.7 and res["top_features"][0]["feature"] == "tl_x"
    same = DF.adversarial(a, a.assign(race_id=lambda x: "z" + x["race_id"]), ["tl_x", "f2"])
    assert same["auc_mean"] < 0.65


def test_swap_in_aux_oof_and_mask_groups():
    from src.scripts.data import drift_dump_features as DD

    X = pd.DataFrame({"race_id": ["r1", "r1"], "horse_id": ["a", "b"], "mtl_t_rank_norm": [0.1, 0.2], "other": [1.0, 2.0]})
    aux = pd.DataFrame({"race_id": ["r1"], "horse_id": ["a"], "mtl_t_rank_norm": [9.0]})
    out = DD.swap_in_aux_oof(X.copy(), aux)
    assert out.loc[0, "mtl_t_rank_norm"] == 9.0 and np.isnan(out.loc[1, "mtl_t_rank_norm"]) and out["other"].tolist() == [1.0, 2.0]
    g = ME.groups_of(["tl_a", "wc_b", "oikiri_eval", "has_comment", "comment_len", "x"])
    assert g["training_tl_wc"] == ["tl_a", "wc_b"] and set(g["oikiri_comment"]) == {"oikiri_eval", "has_comment", "comment_len"}
    assert set(g["both"]) == {"tl_a", "wc_b", "oikiri_eval", "has_comment", "comment_len"}


def test_pick_days_and_spread():
    from src.scripts.data import drift_dump_features as DD

    assert DD.spread(list("abcdefghij"), 4) == ["a", "c", "f", "h"]
    assert DD.spread(["a", "b"], 5) == ["a", "b"]
    assert DD.parse_months("1-3,7") == [1, 2, 3, 7]


def test_benchmark_asof_stats_exclude_same_day():
    from src.scripts.data import drift_benchmark as B

    df = pd.DataFrame({"horse_id": ["a", "a", "a", "b"], "date": pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-12", "2026-01-12"]),
                       "y_win": [1.0, 0.0, 1.0, 0.0], "y_top3": [1.0, 1.0, 1.0, 0.0], "pct": [0.1, 0.5, 0.2, 0.9]})
    s = B.asof_stats(df, "horse_id", "h")
    assert s["h_n"].tolist() == [0.0, 1.0, 1.0, 0.0]            # 同日の別レースは含めない。初回は 0 件
    assert s["h_win"].iloc[1] > s["h_win"].iloc[0]                 # 前走で勝った馬は勝率の推定が上がる
