"""tp-84: 予想ペース × 馬のタイプ・感応度の特徴量の検査（先読みなし・向き・被覆）。"""

import numpy as np
import pandas as pd

from src.pipeline.features import tp84_pacefit as pf
from tests.pipeline.test_tp42_racelevel import _synth

SD = {"F5": 1.0, "RPCI": 2.0}


def _inputs(seed=0):
    d = _synth(seed)
    rng = np.random.default_rng(seed + 100)
    n = len(d)
    d["fp"] = rng.uniform(0, 1, n)
    d["pos_first"] = rng.uniform(0, 1, n)
    d["l3f_rank"] = rng.uniform(0, 1, n)
    d["date"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(d["dnum"] - d["dnum"].min(), unit="D")
    rid = d["race_id"].unique()
    pace = pd.DataFrame({"race_id": rid, **{c: rng.normal(0, 1, len(rid)) for c in pf.PP}})
    real = pd.DataFrame({"race_id": rid, "d_F5": rng.normal(0, 1, len(rid)), "d_RPCI": rng.normal(0, 2, len(rid))})
    return d, pace, real


def test_shapes_and_groups():
    d, pace, real = _inputs()
    g = pf.build(d, pace, real, SD)
    assert set(g) == {"PF_A", "PF_B", "PF_ALL"}
    assert all(len(t) == len(d) for t in g.values())
    assert list(g["PF_ALL"].columns) == pf.KEY + [c for c in g["PF_A"].columns if c not in pf.KEY] + [c for c in g["PF_B"].columns if c not in pf.KEY]
    # 実装と同じく left で結合して行の順を保つ（inner の結合は pandas の版によってキーごとに並び替わる）
    assert np.allclose(g["PF_A"]["pp_RPCI"], d.merge(pace, on="race_id", how="left")["pp_RPCI"])


def test_no_lookahead_results_on_or_after_the_race_day_do_not_change_earlier_rows():
    d, pace, real = _inputs(1)
    base = pf.build(d, pace, real, SD)["PF_ALL"]
    cut = float(np.sort(d["dnum"].unique())[len(d["dnum"].unique()) // 2])
    rng = np.random.default_rng(9)
    d2, real2 = d.copy(), real.copy()
    late = d2["dnum"] >= cut
    for c in ("fp", "pos_first", "l3f_rank"):
        d2.loc[late, c] = rng.uniform(0, 1, late.sum())
    late_races = d.loc[late, "race_id"].unique()
    k = real2["race_id"].isin(late_races)
    real2.loc[k, ["d_F5", "d_RPCI"]] = rng.normal(5, 5, (k.sum(), 2))
    alt = pf.build(d2, pace, real2, SD)["PF_ALL"]
    keep = (d["dnum"] <= cut).to_numpy()
    a, b = base.drop(columns=pf.KEY).loc[keep].to_numpy(float), alt.drop(columns=pf.KEY).loc[keep].to_numpy(float)
    both = np.isnan(a) & np.isnan(b)
    assert np.allclose(np.where(both, 0, a), np.where(both, 0, b), atol=1e-9, rtol=0)


def test_slope_picks_up_a_pace_sensitive_horse():
    n = 8
    R = np.tile(np.array([0.2, 0.2, -0.2, -0.2, 0.2, -0.2, 0.2, -0.2]), (1, 1))[:, :n]
    X = np.where(R > 0, 1.0, -1.0)                    # ペースが速い走で上振れ → 速いペースが向く馬
    b, kern, kcon = pf._slope_and_kernel(R, X, np.array([1.0]))
    b2, kern2, _ = pf._slope_and_kernel(R, X, np.array([-1.0]))
    assert b[0] > 0.05 and kern[0] > 0 > kern2[0]
    assert kcon[0] > 0 > _[0]


def test_unseen_horse_gets_nan_style_and_zero_shrunk_values():
    d, pace, real = _inputs(2)
    g = pf.build(d, pace, real, SD)["PF_ALL"]
    first = d.groupby("horse_id").cumcount().to_numpy() == 0
    assert g.loc[first, "sty_rel"].isna().all()
    assert (g.loc[first, "kern_RPCI"].abs() < 1e-12).all() and (g.loc[first, "sens_RPCI"].abs() < 1e-12).all()
