import numpy as np

from src.scripts.data import ptrain_wind as pw


def test_bearing_and_wind_components():
    assert abs(pw.bearing(np.array([0.0]), np.array([1.0]))[0] - 0.0) < 1e-9          # 北へ進む
    assert abs(pw.bearing(np.array([1.0]), np.array([0.0]))[0] - 90.0) < 1e-9         # 東へ進む
    head, cross = pw.wind_components(5.0, 90.0, np.array([90.0]))                        # 東へ進み、東から吹く = 向かい風
    assert abs(head[0] - 5.0) < 1e-9 and abs(cross[0]) < 1e-9
    head, cross = pw.wind_components(5.0, 270.0, np.array([90.0]))                       # 西から = 追い風
    assert abs(head[0] + 5.0) < 1e-9
    head, cross = pw.wind_components(4.0, 180.0, np.array([90.0]))                       # 東へ進み南から = 右から
    assert abs(head[0]) < 1e-9 and cross[0] > 3.9


def test_race_wind_features_segments():
    h = np.full(251, 90.0)                      # 2000m を東へ直進
    f = pw.race_wind_features(5.0, 90.0, h, 2000)
    assert abs(f["wind_head_all"] - 5.0) < 1e-9 and abs(f["wind_head_home"] - 5.0) < 1e-9 and f["wind_cross_abs"] < 1e-9
