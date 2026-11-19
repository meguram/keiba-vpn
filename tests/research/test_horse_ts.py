"""horse_ts.course_ts: x を 0.5%・99.5% 点で打ち切ってから偏差値にする。外れ値 1 走でコースの広がりが崩れないこと。"""
import numpy as np
import pandas as pd

from src.research.race import horse_ts as ht


def _frame(n_big: int = 800, n_small: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for venue, dist, n in (("京都", 1600, n_big), ("中山", 1800, n_small)):
        x = rng.normal(11.5, 0.4, n)
        if venue == "京都":
            x[0] = 29.7          # 歩く走
        for i in range(n):
            rows.append({"race_id": f"{venue}{i // 16}", "horse_id": f"{venue}{i}", "venue": venue, "surface": "芝", "distance": dist,
                         "peak": float(x[i]), "offset": 0.0})
    return ht.course_ts(pd.DataFrame(rows)).reset_index(drop=True)


def test_outlier_is_clipped_and_does_not_inflate_sd():
    h = _frame()
    kyoto = h[h["venue"] == "京都"]
    assert kyoto["x"].max() == 29.7                      # x は打ち切る前の値のまま
    assert kyoto["x_w"].max() < 14                       # 打ち切った後の値
    body = kyoto[kyoto["x"] < 20]
    assert 9 < body["ts"].std() < 11


def test_small_course_is_clipped_on_both_sides_by_band_z():
    h = _frame()
    small = h[h["venue"] == "中山"]
    assert (small["x_w"] < small["x"]).any() and (small["x_w"] > small["x"]).any()
    assert (small["x_w"] != small["x"]).mean() < 0.03    # 切れるのは 1% 前後


def test_order_and_untouched_columns():
    h = _frame()
    assert (h["ts"].rank(ascending=False).corr(h["x"].rank())) > 0.99    # 速いほど ts が高い（順位はほぼ保たれる）
    assert h["ts"].notna().all()
