"""rules-bet-opt の道具（src/research/course/bet_opt.py・bet_opt_search.py）の小さな確かめ。表は合成で、手元のデータを読まない。"""

import numpy as np
import pandas as pd

from src.research.course import bet_opt as BO
from src.research.course import bet_opt_search as S


def test_thirds_covers_window():
    assert BO.thirds(2016, 2024) == [(2016, 2018), (2019, 2021), (2022, 2024)]
    b = BO.thirds(2016, 2019)
    assert b[0][0] == 2016 and b[-1][1] == 2019 and len(b) == 3


def _field(n_races=30, n=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        o = np.sort(rng.uniform(1.5, 80, n))
        fin = rng.permutation(n) + 1
        for i in range(n):
            rows.append({"race_id": f"R{r:03d}", "hn": i + 1, "odds_final": o[i], "pop_final": i + 1, "fin": fin[i],
                         "ret_tan": o[i] if fin[i] == 1 else 0.0, "ret_fuku": 1.5 if fin[i] <= 3 else 0.0})
    return pd.DataFrame(rows)


def test_sim_t60_keeps_overround_and_rows():
    f = _field()
    pool = [np.linspace(-0.2, 0.4, 8), np.linspace(-0.1, 0.6, 8), np.zeros(7)]
    s = BO.sim_t60(f, pool, seed=1)
    m = f.merge(s, on=["race_id", "hn"])
    assert len(m) == len(f)
    a = m.groupby("race_id").apply(lambda g: (1 / g["odds_final"]).sum())
    b = m.groupby("race_id").apply(lambda g: (1 / g["pre"]).sum())
    assert np.allclose(a.to_numpy(), b.to_numpy())                 # 1/オッズの合計は元のレースに合わせる
    assert (m.groupby("race_id")["pre_pop"].min() == 1).all()


def test_sim_t60_common_random_numbers():
    """γ = 0 は重みが一様なので、同じ seed なら何度でも同じ。γ が小さいほど γ = 0 の版に近い（共通の乱数）。"""
    f = _field(n_races=60)
    pool = [np.linspace(-0.3, 0.3, 8) * k for k in (0.5, 1.0, 1.5, 2.0)]
    a = BO.sim_t60(f, pool, seed=3)
    b = BO.sim_t60(f, pool, seed=3)
    assert np.allclose(a["pre"], b["pre"])
    g = BO.sim_t60(f, pool, seed=3, gamma=1e-6)
    assert np.allclose(a["pre"], g["pre"])


def test_rw_adjust_monotone():
    rng = np.random.default_rng(0)
    W = rng.poisson(1.0, size=(300, 200)).astype(np.float32)
    good = {"cost": np.ones(200), "ret": np.r_[np.full(60, 4.0), np.zeros(140)]}       # 回収率 120%
    bad = {"cost": np.ones(200), "ret": np.r_[np.full(40, 2.0), np.zeros(160)]}        # 回収率 40%
    r = S.rw_adjust({"good": good, "bad": bad}, W, {"good", "bad"})
    assert r["good"]["p_rw"] <= r["bad"]["p_rw"]
    assert r["good"]["p_bonf"] >= r["good"]["p_raw"]


def test_judge_order_negative_first():
    """5 段は 否定 → 上から（prereg_final.json の order）。"""
    ok = {"bet_races": 200, "roi": 1.2, "roi_lo": 1.05, "roi_hi": 1.4, "roi_ex_top5": 1.1}
    assert S.judge(ok, (0.1, 0.02, 0.2), 100) == S.LEVELS[1]
    assert S.judge(ok | {"roi_lo": 0.95}, (0.1, 0.02, 0.2), 100) == S.LEVELS[2]
    assert S.judge(ok | {"roi": 0.9, "roi_ex_top5": 0.85}, (0.1, 0.02, 0.2), 100) == S.LEVELS[3]
    assert S.judge(ok, (0.1, -0.02, 0.2), 100) == S.LEVELS[4]
    assert S.judge(ok, (-0.01, -0.05, 0.03), 100) == S.LEVELS[0]           # 差の点推定 ≤ 0 は否定
    assert S.judge(ok | {"roi_hi": 0.95}, (0.1, 0.02, 0.2), 100) == S.LEVELS[0]
    assert S.judge(ok | {"bet_races": 10}, (0.1, 0.02, 0.2), 100).startswith("判定なし")


def test_cand_arrays_row_alignment():
    """◎ の行の順と、再現のオッズ・払戻が対応する（帯 ≤ 5 の複勝）。"""
    f = _field(n_races=12)
    hm = f[f["pop_final"] == 2].copy().reset_index(drop=True)
    hm["score"], hm["date"] = 1.0, "2020-01-01"
    pool = [np.zeros(8)]
    D = S.Draws(hm, f, pool, 0.0, "win", [[0, 5.0]], k=2)
    a = S.cand_arrays(hm, D, {"kind": "grid", "band": [0, 5.0], "strength": "all", "bet": "fuku"}, {"all": -np.inf})
    want = np.where(hm["odds_final"].to_numpy() <= 5.0, 1.0, 0.0)            # ずれ 0 なら T-60 = 確定
    assert np.allclose(a["cost"], np.tile(want, 2))
    assert np.allclose(a["ret"], np.tile(want * hm["ret_fuku"].to_numpy(), 2))
