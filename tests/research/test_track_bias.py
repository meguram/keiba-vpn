"""トラックバイアスの道具（src/research/track_bias）の確かめ。合成のレースだけで、手元のデータは読まない。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.track_bias import baseline as B
from src.research.track_bias import data as D
from src.research.track_bias import day as Y


def _race(rid, date, n=10, venue="東京", surf="芝", fins=None, odds=None, pos=None, rnd=1, cond="良", grade="1勝", cp=None):
    fins = fins if fins is not None else list(range(1, n + 1))
    odds = odds if odds is not None else [2.0 + i for i in range(n)]
    pos = pos if pos is not None else list(range(1, n + 1))
    return pd.DataFrame({
        "race_id": rid, "date": date, "venue": venue, "surface": surf + ("" if surf == "芝" else "ート"), "distance": 1600, "grade": grade,
        "track_condition": cond, "start_time": "10:00", "round": rnd, "finish_position": fins, "bracket_number": [(i // 2) + 1 for i in range(n)],
        "horse_number": list(range(1, n + 1)), "horse_name": [f"馬{i}" for i in range(n)], "horse_id": [f"{rid}_{i}" for i in range(n)],
        "passing_order": [f"{p}-{p}" for p in pos], "last_3f": 34.0, "odds": odds, "popularity": list(range(1, n + 1)), "corner_passing": cp,
    })


def test_harville_sums_to_three_and_monotone():
    p = np.array([0.4, 0.3, 0.2, 0.06, 0.04])
    q = D.harville_top3(p)
    assert q.sum() == pytest.approx(3.0)
    assert np.all(np.diff(q) <= 1e-12) and np.all((q >= p) & (q <= 1))
    assert D.harville_top3(np.array([0.5, 0.3, 0.2])).tolist() == [1, 1, 1]


def test_parse_order_text_inner_first():
    m = D.parse_order_text("(*5,2)-(3,4)1")
    assert m == {5: 0, 2: 1, 3: 0, 4: 1, 1: 0}
    lat = D.last_corner_lateral('[{"corner": 0, "label": "馬場指数", "order_text": "-10"}, {"corner": 3, "order_text": "1,2"}, {"corner": 4, "order_text": "(1,2,3)"}]')
    assert lat == {1: 0, 2: 1, 3: 2}


def test_clean_drops_non_runners_and_bad_races():
    a = _race("A", "2020-01-05")
    a.loc[9, ["finish_position", "passing_order"]] = [-1, ""]   # 印が無く通過順も無い（取消・除外） → 落とす
    b = _race("B", "2020-01-05", venue="中山")
    b.loc[0, "odds"] = np.nan                          # オッズの欠け → レースごと落とす
    c = _race("C", "2020-01-05", venue="京都")
    c["passing_order"] = ""                            # 通過順が無い（直線） → 落とす
    h = D.clean(pd.concat([a, b, c], ignore_index=True))
    assert set(h["race_id"]) == {"A"} and h["N"].iloc[0] == 9


def test_dnf_kept_with_y0_and_scratch_dropped():
    a = _race("A", "2020-01-05")
    a["finish_position_raw"] = a["finish_position"].astype(str)
    a["finish_status"] = "finished"
    a.loc[9, ["finish_position", "finish_position_raw", "finish_status"]] = [-1, "中", "dnf"]   # 中止 → 残す
    a.loc[8, ["finish_position", "finish_position_raw", "finish_status", "passing_order"]] = [-1, "取", "scratched", ""]  # 取消 → 落とす
    h = D.horse_values(D.clean(a))
    assert len(h) == 9 and h["N"].iloc[0] == 9
    d = h[h["horse_number"] == 10].iloc[0]
    assert d["y"] == 0 and d["fin"] == 99
    assert h["q"].sum() == pytest.approx(3.0) and h["y"].sum() == pytest.approx(3.0)


def test_dead_heat_third_split():
    r = _race("T", "2020-01-05", fins=[1, 2, 3, 3, 5, 6, 7, 8, 9, 10])
    h = D.horse_values(D.clean(r))
    assert h["y"].tolist()[:5] == [1, 1, 0.5, 0.5, 0] and h["y"].sum() == pytest.approx(3.0)
    r2 = _race("T2", "2020-01-05", fins=[1, 1, 3, 4, 5, 6, 7, 8, 9, 10])
    assert D.horse_values(D.clean(r2))["y"].sum() == pytest.approx(3.0)


def test_lateral_missing_horse_makes_b_nan():
    cp = '[{"corner": 4, "order_text": "(1,2,3)4,5,6,7,8,9"}]'          # 10 番が並びに無い
    R = D.race_values(D.horse_values(D.clean(_race("L", "2020-01-05", cp=cp))))
    assert np.isnan(R["inner_b"].iloc[0])
    cp2 = '[{"corner": "x"}, {"corner": 4, "order_text": "(1,2,3)4,5,6,7,8,9,10"}]'   # 角の番号が数字でなくても落ちない
    R2 = D.race_values(D.horse_values(D.clean(_race("L2", "2020-01-05", cp=cp2))))
    assert np.isfinite(R2["inner_b"].iloc[0])


def test_front_sign():
    n = 10
    front = _race("F", "2020-01-05", fins=list(range(1, n + 1)), pos=list(range(1, n + 1)), odds=[10.0] * n)   # 前の馬が上位
    back = _race("K", "2020-01-05", venue="中山", fins=list(range(n, 0, -1)), pos=list(range(1, n + 1)), odds=[10.0] * n)  # 後ろの馬が上位
    R = D.race_values(D.horse_values(D.clean(pd.concat([front, back], ignore_index=True)))).set_index("race_id")
    assert R.loc["F", "front_early"] > 0.5 and R.loc["K", "front_early"] < -0.5
    # 逃げた人気馬が負けて後方の穴馬が来ると、同じ着順の並びでも人気どおりより大きく負
    upset = _race("U", "2020-01-05", venue="京都", fins=list(range(n, 0, -1)), pos=list(range(1, n + 1)), odds=[1.5 + 3 * i for i in range(n)])
    Ru = D.race_values(D.horse_values(D.clean(upset))).set_index("race_id")
    assert Ru.loc["U", "front_early"] < R.loc["K", "front_early"]


def test_inner_c_removes_front_part():
    """内枠ほど前に行き、前が残っただけのレースでは、案 A は内有利に出るが、案 C はほぼ 0。"""
    n = 12
    r = _race("I", "2020-01-05", n=n, fins=list(range(1, n + 1)), pos=list(range(1, n + 1)), odds=[12.0] * n)
    R = D.race_values(D.horse_values(D.clean(r))).iloc[0]
    assert R["inner_a"] > 0.5 and abs(R["inner_c"]) < 1e-9


def _many(days=40, venue="東京"):
    rng = np.random.default_rng(0)
    parts = []
    for k in range(days):
        date = (pd.Timestamp("2018-01-06") + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d")
        for j in range(4):
            n = 10
            parts.append(_race(f"{k}_{j}", date, n=n, venue=venue, fins=list(rng.permutation(n) + 1), rnd=j + 1))
    return pd.concat(parts, ignore_index=True)


def test_baseline_is_asof():
    R = D.race_values(D.horse_values(D.clean(_many())))
    e1 = B.expect(R, "front_early", 50)
    # 最後の日の値を大きく変えても、それより前の日の E は変わらない（その日の E も、その日自身を使わない）
    R2 = R.copy()
    last = R2["date"] == R2["date"].max()
    R2.loc[last, "front_early"] += 100.0
    e2 = B.expect(R2, "front_early", 50)
    assert np.allclose(e1, e2)
    first = (R["date"] == R["date"].min()).to_numpy()
    assert np.allclose(e1[first], 0.0)                  # 最初の日は履歴なし


def test_levels_and_label():
    assert [Y.level(z) for z in (-2.5, -1.2, 0.3, 1.0, 2.0)] == [-2, -1, 0, 1, 2]
    assert Y.label(1, 1, 6) == "内有利・前残り"
    assert Y.label(0, 0, 6) == "中立"
    assert Y.label(-2, 0, 6) == "強い差し有利"
    assert Y.label(2, -1, 2) == Y.NO_JUDGE


def test_day_pipeline_runs_and_contrib_sums_to_z():
    R = D.race_values(D.horse_values(D.clean(_many())))
    ks = {c: 50 for c in D.METRICS + D.AUX}
    R = B.add_baseline(R, D.METRICS + D.AUX, ks)
    days = Y.day_raw(R)
    days = days[days["date"] > days["date"].min()]      # 最初の日は分散が無い
    cal = {"metrics": {c: {"芝": {"mu": 0.1, "sd": 1.3}} for c in D.METRICS + D.AUX}}
    d = Y.apply(days, cal)
    assert set(d["label"]) <= set(Y.FRONT_WORDS.values()) | set(Y.INNER_WORDS.values()) | {"中立", Y.NO_JUDGE} | {
        f"{i}・{fr}" for i in Y.INNER_WORDS.values() for fr in Y.FRONT_WORDS.values()}
    c = Y.race_contrib(R, d, cal)
    one = d.iloc[5]
    s = c[(c["date"] == one["date"]) & (c["venue"] == one["venue"])]["zc_front_early"].sum()
    assert s - 0.1 / 1.3 == pytest.approx(one["z_front_early"])     # 寄与の和 − μ/s = 日の z


def test_variance_nan_race_left_out_of_day():
    R = D.race_values(D.horse_values(D.clean(_many(days=3))))
    R = B.add_baseline(R, D.METRICS + D.AUX, {c: 50 for c in D.METRICS + D.AUX})
    R.loc[R.index[-1], "V_front_early"] = np.nan
    d = Y.day_raw(R)
    last = d[d["date"] == d["date"].max()].iloc[0]
    assert last["n_front_early"] == 3 and last["n_races"] == 4
