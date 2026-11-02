"""騎手コラムの共通集計（src/research/columns/jockey_common.py）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.columns import jockey_common as jc


def test_bh_matches_hand_computation():
    q = jc.bh([0.01, 0.04, 0.03, float("nan"), 0.5])
    # 4 件: 0.01*4/1=0.04, 0.03*4/2=0.06, 0.04*4/3=0.0533 → 単調化で 0.0533, 0.5*4/4=0.5
    assert np.isclose(q[0], 0.04) and np.isclose(q[2], 0.04 * 4 / 3) and np.isclose(q[1], 0.04 * 4 / 3)
    assert np.isnan(q[3]) and np.isclose(q[4], 0.5)


def test_rank_cuts_are_5_20_50_20_5_percent():
    r = jc.rank_of(pd.Series(np.arange(100, 0, -1, dtype=float)))
    assert r.value_counts().to_dict() == {"S": 5, "A": 20, "B": 50, "C": 20, "D": 5}
    assert r.iloc[0] == "S" and r.iloc[-1] == "D"


def test_eb_shrinks_small_samples_more():
    rng = np.random.default_rng(0)
    rows = []
    for j, (n, p) in {"big": (4000, 0.4), "small": (40, 0.4), **{f"m{k}": (1000, 0.2) for k in range(20)}}.items():
        rows += [{"jockey_id": j, "top3": float(x)} for x in rng.random(n) < p]
    t = jc.eb_effect(pd.DataFrame(rows), "top3")
    # 同じ真の率でも、騎乗の少ない騎手は平均（約 0.2）に強く引き寄せられる
    assert t.at["big", "eb"] - t.at["small", "eb"] > 0.05 or t.at["small", "eb"] < t.at["small", "raw"]
    assert abs(t.at["big", "eb"] - t.at["big", "raw"]) < abs(t.at["small", "eb"] - t.at["small", "raw"])


def _rides(jid: str, level: str, n: int, wins: int, exp: float) -> pd.DataFrame:
    y = np.r_[np.ones(wins), np.zeros(n - wins)]
    return pd.DataFrame({"jockey_id": jid, "lv": level, "win": y, "exp_win": exp, "res_win": y - exp})


def test_two_axis_labels_need_both_axes_for_nigate():
    # 騎手 X: どこでも人気どおり（見込み 0.3）。長距離だけ見込みを大きく下回るが、全騎手の平均（約 0.08）よりは上
    parts = [_rides("X", "短", 2000, 600, 0.3), _rides("X", "長", 600, 120, 0.3)]
    # その他の騎手（平均を作る）
    parts += [_rides(f"o{k}", lv, 2000, 100, 0.05) for k in range(5) for lv in ("短", "長")]
    d = pd.concat(parts, ignore_index=True)
    t = jc.two_axis(d, ["X"], "lv", ["短", "長"], "win")
    long_ = t[t["level"] == "長"].iloc[0]
    assert long_["a"] == "up" and long_["b"] == "down"
    assert long_["short"] == "相対的に下" and long_["label"] == "平均より上だが、ほかの条件ほどではない"


def test_calibrate_is_unbiased_overall():
    rng = np.random.default_rng(1)
    n = 20000
    implied = rng.uniform(0.01, 0.5, n)
    d = pd.DataFrame({"implied": implied, "starters": rng.integers(6, 18, n), "win": (rng.random(n) < implied).astype(float)})
    d["top3"] = (rng.random(n) < np.minimum(1, implied * 2.5)).astype(float)
    c = jc.calibrate(d)
    assert abs(c["res_win"].mean()) < 1e-9 and abs(c["res_top3"].mean()) < 1e-9
    assert c["exp_win"].corr(c["implied"]) > 0.9
