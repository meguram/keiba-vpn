"""axis_variant_eval の補助関数の検算（合成データ）。"""
import numpy as np
import pandas as pd

from src.scripts.data import axis_variant_eval as av


def test_make_sustain_short_keeps_only_short_runs() -> None:
    s = pd.Series([50.0, 51.0, 52.0, 53.0])
    d = pd.Series([1200, 1400, 1600, 2000])
    out = av.make_sustain_short(s, d)
    assert out.tolist()[:2] == [50.0, 51.0] and out.iloc[2:].isna().all()


def test_residualize_removes_linear_part_and_keeps_nan() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2 * x + 1 + rng.normal(scale=0.1, size=500)
    y[3] = np.nan
    r = av.residualize(y, x[:, None], np.ones(500, bool))
    assert np.isnan(r[3]) and np.nanstd(r) < 0.2
    assert abs(np.corrcoef(r[np.isfinite(r)], x[np.isfinite(r)])[0, 1]) < 0.05


def test_peak_asof_uses_only_previous_runs() -> None:
    d = pd.DataFrame({"horse_id": ["a"] * 4 + ["b"] * 2, "ts": [40.0, 60.0, 70.0, 10.0, 55.0, 45.0]})
    out = av.peak_asof(d)
    assert np.isnan(out[0]) and np.isnan(out[4])                 # 最初の走は前が無い
    assert abs(out[1] - 45.0) < 1e-9 and abs(out[2] - 50.0) < 1e-9     # 1 走だけなら本番の _peak が仮の値 50 を足す（40 と 50 の平均）
    assert abs(out[3] - 65.0) < 1e-9                              # 60 と 70
    assert abs(out[5] - 52.5) < 1e-9


def test_knot_variants_keep_anchors() -> None:
    cur = [[52.0, 3.0], [58.0, 4.0], [62.0, 5.0], [64.0, 5.5], [71.0, 10.0]]
    k = av.knot_variants(cur)
    assert k["cur"] == cur and [5.5 in (y for _x, y in k["no55"])] == [False]
    assert k["lin"] == [[52.0, 3.0], [71.0, 10.0]]
