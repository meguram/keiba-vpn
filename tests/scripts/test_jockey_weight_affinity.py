"""騎手 × 馬体重帯の列（src/scripts/data/jockey_weight_affinity.py）が、その日より前の日のレースだけで作られること。"""

import numpy as np
import pandas as pd

from src.scripts.data import jockey_weight_affinity as jw


def _rows():
    d = pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r3", "r3", "r4"],
        "horse_id": ["h1", "h2", "h3", "h4", "h5", "h6"],
        "date": pd.to_datetime(["2019-01-05", "2019-01-05", "2019-01-05", "2019-01-12", "2019-01-12", "2019-01-19"]),
        "jockey_id": ["A", "B", "A", "A", "A", "A"],
        "weight": [430, 500, 435, 436, 530, 438],
        "finish_position": [1, 2, 3, 5, 1, 2],
        "e3": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "q": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    })
    d["band"] = jw.band_of(d["weight"])
    d["y1"] = (d["finish_position"] == 1).astype(float)
    d["y3"] = (d["finish_position"] <= 3).astype(float)
    d["done"] = True
    return d


def test_band_of():
    b = jw.band_of(pd.Series([430, 439, 440, 479, 480, 519, 520, 700, None]))
    assert list(b[:7]) == ["~439", "~439", "440-479", "440-479", "480-519", "480-519", "520~"]
    assert b[7] is None and b[8] is None


def test_asof_excludes_same_day_and_future():
    d = _rows()
    f = jw.build_features(d, "band")
    # 初日（1/5）の行は前の日が無いので騎乗数 0
    assert (f.loc[:2, "jwb_n"] == 0).all()
    # 1/12 の A・〜439 の行（h4）は、1/5 の A・〜439 の 2 走（h1 勝ち・h3 3 着）だけを数える
    assert f.loc[3, "jwb_n"] == 2
    # 1/12 の A・520〜 の行（h5）は、前の日に A・520〜 の騎乗が無い
    assert f.loc[4, "jwb_n"] == 0
    # 1/19 の A・〜439 は 1/5 の 2 走と 1/12 の 1 走（h4）
    assert f.loc[5, "jwb_n"] == 3


def test_future_rows_do_not_change_past_features():
    d = _rows()
    f1 = jw.build_features(d, "band")
    d2 = d.copy()
    d2.loc[5, "finish_position"] = 1                 # 最後の日の結果を変えても、それより前の行の列は変わらない
    d2["y1"] = (d2["finish_position"] == 1).astype(float)
    d2["y3"] = (d2["finish_position"] <= 3).astype(float)
    f2 = jw.build_features(d2, "band")
    cols = ["jwb_n", "jwb_top3", "jwb_ae3"]
    np.testing.assert_allclose(f1.loc[:4, cols].to_numpy(), f2.loc[:4, cols].to_numpy())


def test_prev_band_counts_all_earlier_days():
    """引く帯の騎乗がその日に無いとき（前走の帯で引く版）でも、前の日までの騎乗を全部数える。"""
    d = _rows()
    extra = pd.DataFrame({"race_id": ["r5"], "horse_id": ["h7"], "date": pd.to_datetime(["2019-01-26"]), "jockey_id": ["A"],
                          "weight": [540], "finish_position": [4], "e3": [0.5], "q": [0.2]})
    extra["band"] = jw.band_of(extra["weight"])
    extra["y1"], extra["y3"], extra["done"] = 0.0, 0.0, True
    d = pd.concat([d, extra], ignore_index=True)
    d["band_lookup"] = d["band"]
    d.loc[6, "band_lookup"] = "~439"                 # 1/26 の行を「〜439」で引く（その日に A・〜439 の騎乗は無い）
    f = jw.build_features(d, "band_lookup")
    assert f.loc[6, "jwb_n"] == 4                    # 1/5 の 2 走・1/12・1/19


def test_genryo_flag():
    d = pd.DataFrame({"race_id": ["r"] * 5 + ["h"] * 3, "sex_age": ["牡3"] * 5 + ["牡4"] * 3,
                      "jockey_weight": [56, 56, 56, 53, 51, 57, 55, 52]})
    g = jw.genryo_flag(d)
    assert list(g[:4]) == [0.0, 0.0, 0.0, 1.0]
    assert np.isnan(g[4])                            # 5kg 軽いのは減量とみなさない
    assert np.isnan(g[5:]).all()                     # 最頻値の割合が 6 割未満（ハンデ戦など）は欠け


def test_cond_logit_recovers_beta():
    rng = np.random.default_rng(0)
    nr, k = 3000, 8
    race = np.repeat(np.arange(nr), k)
    z = rng.normal(size=nr * k)
    eta = 0.3 * z
    p = np.exp(eta).reshape(nr, k)
    p = p / p.sum(1, keepdims=True)
    w = np.array([rng.choice(k, p=pp) for pp in p])
    win = np.zeros(nr * k, int)
    win[np.arange(nr) * k + w] = 1
    b, se, _ = jw.cond_logit(race, np.zeros(nr * k), z[:, None], win)
    assert abs(b[0] - 0.3) < 3 * se[0]
