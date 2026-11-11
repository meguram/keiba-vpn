import numpy as np

from src.scripts.data import final_week_lap_split as S


def test_band_boundaries_are_11_or_less_12s_13_or_more():
    assert list(S.band(np.array([11.9, 12.0, 12.9, 13.0]))) == ["11台以下", "12台", "12台", "13以上"]


def test_gap_bins_put_equal_laps_in_zero_and_split_on_half_and_one_second():
    g = S.gapb(np.array([0.0, 0.1, 0.5, 0.6, 1.0, 1.1]))
    assert [str(v) for v in g] == ["差0", "差0.1-0.5", "差0.1-0.5", "差0.6-1.0", "差0.6-1.0", "差1.1以上"]
