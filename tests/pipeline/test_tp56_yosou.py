"""tp-56 の特徴量の先読み検査: 日付を切って作り直しても、それ以前の走の値が完全に一致する。"""

import numpy as np
import pandas as pd

from src.pipeline.features import tp56_yosou as ty


def _runs(n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    horses = [f"h{i:03d}" for i in range(60)]
    for d in range(n_days):
        date = (pd.Timestamp("2022-01-01") + pd.Timedelta(days=d * 7)).strftime("%Y-%m-%d")
        for v in ("05", "09"):
            for r in range(1, 5):
                rid = f"2022{v}01{d + 1:02d}{r:02d}"
                n = int(rng.integers(8, 15))
                hs = rng.choice(horses, n, replace=False)
                fin = rng.permutation(n) + 1
                ord1 = rng.integers(1, n + 1, n)
                l3 = 34 + rng.normal(0, 0.6, n)
                t3, s3 = 35.5 + rng.normal(0, 0.8), 35.0 + rng.normal(0, 0.8)
                for k in range(n):
                    rows.append({"race_id": rid, "date": date, "venue": v, "surface": "芝" if r % 2 else "ダ", "distance": 1200 + 200 * (r % 3),
                                 "field_size": n, "horse_id": hs[k], "horse_number": k + 1, "bracket_number": 1 + k * 8 // n,
                                 "finish_position": float(fin[k]), "popularity": float(rng.integers(1, n + 1)),
                                 "last_3f": float(l3[k]), "passing_order": f"{ord1[k]}-{ord1[k]}", "pace": '{"first_half_3f": %.1f, "second_half_3f": %.1f}' % (t3, s3)})
    return pd.DataFrame(rows)


def _flags(df, seed=1):
    rng = np.random.default_rng(seed)
    f = df[["race_id", "horse_id"]].copy()
    f["trouble_rem"] = (rng.random(len(f)) < 0.05).astype(float)
    f["has_remarks"] = 1.0
    return f


def test_truncation_gives_identical_past_features():
    runs = ty.prepare_runs(_runs())
    flags = _flags(runs)
    full = ty.build_features(runs, flags)
    cut = "2022-06-30"
    part = ty.build_features(runs[runs["date"] <= cut].copy(), flags)
    a = full[full["date"] <= cut].reset_index(drop=True)
    b = part.reset_index(drop=True)
    assert len(a) == len(b) and (a["race_id"].to_numpy() == b["race_id"].to_numpy()).all()
    cols = [c for c in a.columns if c not in ("race_id", "horse_id", "date", "venue", "surface")]
    for c in cols:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), c
        assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=0, atol=1e-9), c


def test_future_result_changes_do_not_move_past_values():
    runs = ty.prepare_runs(_runs())
    base = ty.build_features(runs, None)
    r2 = runs.copy()
    late = r2["date"] >= "2022-08-01"
    r2.loc[late, "finish_position"] = r2.loc[late, "field_size"] + 1 - r2.loc[late, "finish_position"]
    r2.loc[late, "last_3f"] = r2.loc[late, "last_3f"] + 1.0
    alt = ty.build_features(r2, None)
    a = base[base["date"] < "2022-08-01"].reset_index(drop=True)
    b = alt[alt["date"] < "2022-08-01"].reset_index(drop=True)
    for c in ["R_fb", "R_kb", "R_chalk", "R_attr", "front_t", "kick_t", "stam_t", "fit_rate", "elo_g", "elo_c", "elo_rel_c", "elo_pw_c"]:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), c
        assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=0, atol=1e-9), c


def test_feature_ranges_and_types():
    f = ty.build_features(ty.prepare_runs(_runs()), _flags(ty.prepare_runs(_runs())))
    assert set(f["tp"].unique()) <= set(range(7))
    assert f["fit_best"].dropna().isin([0.0, 1.0]).all()
    assert (f["elo_pw_c"].dropna().between(0, 1)).all()
    assert f.groupby("race_id")["fit_rank"].min().dropna().eq(1.0).all()
