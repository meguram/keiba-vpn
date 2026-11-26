"""基準の走破時計の関数（src/pipeline/race_time）: 前半 600m の補間と、単調の制約。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.race_time.features import age_ord, first_lap_len, front_time_at
from src.pipeline.race_time.model import RaceTimeModel, ispline_basis


@pytest.mark.parametrize("dist,first", [(1000, 200), (1150, 150), (1200, 200), (1700, 100), (1900, 100), (2100, 100), (2500, 100), (1450, 50)])
def test_first_lap_length(dist, first):
    assert first_lap_len(dist) == first


def test_front_600m_on_each_distance():
    """結果表の first_half_3f は 1700・1900・2100m で 500m 分になる。ラップからは 600m の通過時計を正しく出す。"""
    assert front_time_at([12.3, 11.4, 11.2, 11.2, 11.1, 11.6], 1200) == pytest.approx(34.9)          # 200m 区間だけ
    laps1700 = [7.1, 11.0, 11.2, 12.4, 12.8, 12.6, 12.5, 12.4, 12.8]                               # 100m + 200m × 8
    assert front_time_at(laps1700, 1700) == pytest.approx(7.1 + 11.0 + 11.2 + 12.4 / 2)            # 500m の後の 100m は区間の半分
    laps1150 = [9.6, 11.0, 11.4, 11.8, 12.0, 12.5]                                                  # 150m + 200m × 5
    assert front_time_at(laps1150, 1150) == pytest.approx(9.6 + 11.0 + 11.4 + 11.8 * 50 / 200)
    laps1900 = [7.0, 11.2, 11.6, 12.4, 12.6, 12.5, 12.3, 12.2, 12.1, 12.6]                          # 100m + 200m × 9
    assert front_time_at(laps1900, 1900) == pytest.approx(7.0 + 11.2 + 11.6 + 12.4 / 2)
    laps2100 = [6.9, 11.3, 12.0] + [12.5] * 8                                                       # 100m + 200m × 10
    assert front_time_at(laps2100, 2100) == pytest.approx(6.9 + 11.3 + 12.0 + 12.5 / 2)
    assert front_time_at([12.0] * 5, 1200) is None                                                  # ラップの数が距離と合わない
    assert front_time_at(None, 1200) is None and front_time_at([12.0, "x"], 400) is None


def test_age_ord():
    assert age_ord("サラ系２歳 未勝利") == 0 and age_ord("サラ系３歳 １勝クラス") == 1
    assert age_ord("サラ系３歳以上 ２勝クラス") == 2 and age_ord("サラ系４歳以上 オープン") == 2 and age_ord("") is None


def test_ispline_basis_is_monotone_from_0_to_1():
    x = np.linspace(0, 10, 101)
    B = ispline_basis(x, [2.0, 5.0, 7.5], 0.0, 10.0)
    assert np.all(np.diff(B, axis=0) >= -1e-12) and np.allclose(B[0], 0) and np.allclose(B[-1], 1)


def _toy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    course = rng.choice(["東京|芝|1600", "中山|ダ|1800", "京都|芝|2000"], n)
    cls = rng.integers(1, 9, n)
    front = 35 + rng.normal(0, 0.8, n)
    field = rng.integers(6, 19, n)
    age = rng.integers(0, 3, n)
    base = pd.Series(course).map({"東京|芝|1600": 94.0, "中山|ダ|1800": 112.0, "京都|芝|2000": 120.0}).to_numpy()
    y = base - 0.25 * cls + 0.9 * (front - 35) - 0.3 * (age >= 1) + rng.standard_t(4, n) * 0.6       # 重い裾の外れ値も
    y[:10] += 15                                                                                    # 極端な外れ値（落馬など）
    return pd.DataFrame({"course": course, "class_ord": cls, "front600": front, "field_size": field, "age_ord": age,
                         "newcomer": 0, "y_top5": y, "date": "20240101", "venue": "x", "surf": "芝"})


def test_fit_recovers_directions_and_every_course_curve_is_monotone():
    m = RaceTimeModel(lam=20.0).fit(_toy())
    viol = m.monotone_violations()
    assert sum(sum(r.values()) for r in viol.values()) == 0                                         # 全コース × 全項で破れ 0
    xs = np.arange(1, 9, dtype=float)
    for c in m.courses:
        eff = m.term_curve(c, "cls", xs)
        assert eff[-1] - eff[0] == pytest.approx(-0.25 * 7, abs=0.6)                                # クラスの効きの大きさを取り戻す
    assert m.params["age"][0] > 0.1                                                                 # 3 歳は 2 歳より速い
    r = _toy(seed=1).pipe(lambda d: d.y_top5 - m.predict(d))
    assert float(np.median(np.abs(r))) < 0.7                                                        # Huber で外れ値に引っ張られない


def test_monotone_constraint_holds_even_when_the_data_go_the_other_way():
    d = _toy(seed=2)
    d["y_top5"] = d["y_top5"] + 0.5 * d["class_ord"]                                                # データはクラスが上がるほど遅い（逆向き）
    m = RaceTimeModel(lam=20.0).fit(d)
    assert sum(sum(r.values()) for r in m.monotone_violations().values()) == 0
    free = RaceTimeModel(lam=20.0, monotone=False).fit(d)
    assert sum(r["cls"] for r in free.monotone_violations().values()) > 0                           # 制約を外すと逆向きになる（比べの版）


def test_save_and_load_round_trip(tmp_path):
    d = _toy()
    m = RaceTimeModel(lam=20.0).fit(d)
    p = tmp_path / "m.json"
    m.save(p)
    m2 = RaceTimeModel.load(p)
    assert np.allclose(m.predict(d), m2.predict(d))


def test_v2_course_age_steps_stay_monotone_and_finer_knots():
    d = _toy(seed=3)
    m = RaceTimeModel(lam=5.0, spec="v2").fit(d)
    assert len(m.terms["cls"].knots) == 6 and len(m.terms["pace"].knots) == 9
    assert "B_age" in m.params and np.min(m.params["B_age"]) >= 0                                  # コースごとの年齢の段も古馬ほど速い
    assert sum(sum(r.values()) for r in m.monotone_violations().values()) == 0
    m2 = RaceTimeModel.from_json(m.to_json())
    assert m2.spec == "v2" and np.allclose(m.predict(d), m2.predict(d))
