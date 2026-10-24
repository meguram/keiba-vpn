"""tp-25b: ペース特徴量の定義とリークの確認。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import tp25b_pace as P


def _entries(n_races=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        d = pd.Timestamp("2022-01-01") + pd.Timedelta(days=7 * r)
        fin = rng.permutation(6) + 1
        for h in range(6):
            rows.append({"race_id": f"R{r:03d}", "horse_id": f"h{h}", "date": d, "finish_position": int(fin[h]), "field_size": 6, "l3f_dev": float(rng.normal())})
    return pd.DataFrame(rows)


def _actual(df, seed=1):
    rng = np.random.default_rng(seed)
    ids = df["race_id"].unique()
    return pd.DataFrame({"race_id": ids, "pf_z": rng.normal(size=len(ids)), "bal_z": rng.normal(size=len(ids))})


def test_slopes_ignore_current_and_future_races():
    df, act = _entries(), _actual(_entries())
    a = P.pace_slopes(df, act)
    # 後半のレースの結果とペースを書き換えても、前半の行の特徴量は変わらない
    df2, act2 = df.copy(), act.copy()
    late = df2["date"] >= pd.Timestamp("2022-06-01")
    df2.loc[late, "finish_position"] = (df2.loc[late, "finish_position"] % 6) + 1
    df2.loc[late, "l3f_dev"] = 9.0
    act2.loc[act2["race_id"].isin(df2.loc[late, "race_id"]), ["pf_z", "bal_z"]] = 5.0
    b = P.pace_slopes(df2, act2)
    early = (df["date"] < pd.Timestamp("2022-06-01")).to_numpy()
    pd.testing.assert_frame_equal(a[early], b[early])
    assert (a.loc[df["date"] == df["date"].min(), "sl_n"] == 0).all()
    assert a.loc[df["date"] == df["date"].min()].drop(columns="sl_n").isna().all().all()


def test_slope_recovers_sign_with_shrinkage():
    rng = np.random.default_rng(3)
    rows, acts = [], []
    for r in range(60):
        x = rng.normal()
        acts.append({"race_id": f"R{r}", "pf_z": x, "bal_z": 0.0})
        rows.append({"race_id": f"R{r}", "horse_id": "a", "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=7 * r), "finish_position": 1,
                     "field_size": 8, "l3f_dev": 0.5 * x + 0.01 * rng.normal()})
    df = pd.DataFrame(rows)
    s = P.pace_slopes(df, pd.DataFrame(acts))
    last = s["sl_l3f_pf_z"].iloc[-1]
    assert 0.3 < last < 0.5            # 真の傾き 0.5 より少し縮む


def test_race_targets_z_uses_fit_period_only():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        rows.append({"race_id": f"r{i}", "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i * 2), "venue_code": "05", "surface": "芝", "distance": 1600,
                     "pace_first": 35 + rng.normal(), "pace_second": 35.0, "pace_bal": rng.normal(), "rpci": 50 + rng.normal()})
    races = pd.DataFrame(rows)
    lab = pd.DataFrame({"race_id": ["r0", "r0"], "pos_early": [0.1, 0.9], "pos_late": [0.3, 0.5], "is_hurdle": [False, False]})
    t = P.race_targets(races, lab)
    fit = races["date"] <= P.FIT_END
    assert abs(t.loc[fit.to_numpy(), "t_pf_z"].mean()) < 0.1
    assert abs(t.loc[0, "t_churn"] - 0.3) < 1e-9 and t["t_churn"].iloc[1:].isna().all()


def test_interactions_have_expected_columns():
    df = _entries(3)
    df["horse_number"] = np.tile(np.arange(1, 7), 3)
    df["a_style"] = np.linspace(0.1, 0.9, len(df)) % 1
    df["f2_nfront"], df["f2_lone_front"] = 2.0, 0.0
    df["geo_first_curve"] = 300.0
    pred = pd.DataFrame({"race_id": df["race_id"].unique(), **{c: 0.5 for c in P.PRED_COLS}})
    sl = pd.DataFrame({"sl_res_pf_z": 0.1, "sl_n": 3}, index=df.index)
    out = P.interaction_features(df, pred, sl)
    assert set(P.PRED_COLS) <= set(out.columns) and "ex_res_pf_z" in out.columns and "ix_gate_x_curve" in out.columns
    assert np.allclose(out["ex_res_pf_z"], 0.05)
