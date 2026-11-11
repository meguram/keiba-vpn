"""レース環境 ENV の先読みの検査: 後の結果を書き換えても、前のレースの特徴量は変わらない。"""

import numpy as np
import pandas as pd

from src.pipeline.features.race_env.build import build_env_race


def _rs(seed=0, days=60):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = pd.Timestamp("2021-06-01")   # 2021 年の分が「前の年」として偏りの期待値の当てはめに使われる
    for di in range(days * 2):
        date = d0 + pd.Timedelta(days=7 * (di // 2) + (di % 2))
        for v in ("05", "06"):
            for s in ("芝", "ダート"):
                for r in range(1, 9):
                    rows.append(dict(race_id=f"{date:%Y%m%d}{v}{s == '芝'}{r:02d}", date=date, venue_code=v, surface=s, distance=int(rng.choice([1200, 1600])),
                                     direction="右", grade="1勝", round=r, field_size=int(rng.integers(8, 17)),
                                     top2_time=float(rng.normal(96, 1.5)), top2_l3f=float(rng.normal(35, 0.8)), front_idx=float(rng.uniform(0.2, 0.8)),
                                     pace_diff=float(rng.normal(0, 1))))
    d = pd.DataFrame(rows)
    d["top2_front"] = d["top2_time"] - d["top2_l3f"]
    return d


def _perturb(rs, mask):
    x = rs.copy()
    for c in ("top2_time", "top2_l3f", "top2_front", "front_idx", "pace_diff"):
        x.loc[mask, c] = x.loc[mask, c] + 50.0
    return x


def _same(a, b, ids):
    assert a.set_index("race_id").loc[ids, "env_bias_bias_front_day"].notna().sum() >= 2, "偏りの列が欠損だらけで、検査が自明になっている"
    a = a.set_index("race_id").loc[ids].sort_index()
    b = b.set_index("race_id").loc[ids].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_later_days_do_not_change_earlier_features():
    rs = _rs()
    cut = rs["date"].sort_values().iloc[int(len(rs) * 0.8)]
    base = build_env_race(rs)
    pert = build_env_race(_perturb(rs, rs["date"] > cut))
    ids = rs.loc[rs["date"] <= cut, "race_id"]
    _same(base, pert, ids)


def test_same_day_later_rounds_do_not_change_earlier_rounds():
    rs = _rs(seed=1)
    day = rs["date"].sort_values().iloc[int(len(rs) * 0.8)]
    sel = (rs["date"] == day) & (rs["venue_code"] == "05") & (rs["surface"] == "芝")
    base = build_env_race(rs)
    pert = build_env_race(_perturb(rs, sel & (rs["round"] >= 4)))
    ids = rs.loc[sel & (rs["round"] <= 4), "race_id"]   # 4R 自身の特徴量も「3R まで」なので変わらない
    _same(base, pert, ids)


def test_other_venue_same_day_is_not_used_and_perturbing_one_race_changes_only_later_ones():
    rs = _rs(seed=2)
    day = rs["date"].sort_values().iloc[int(len(rs) * 0.8)]
    one = (rs["date"] == day) & (rs["venue_code"] == "05") & (rs["surface"] == "芝") & (rs["round"] == 2)
    base = build_env_race(rs)
    pert = build_env_race(_perturb(rs, one))
    other = rs.loc[(rs["date"] == day) & ~((rs["venue_code"] == "05") & (rs["surface"] == "芝")), "race_id"]
    _same(base, pert, other)                      # 同日の他の場・面は変わらない
    after = rs.loc[(rs["date"] > day) & (rs["venue_code"] == "05") & (rs["surface"] == "芝"), "race_id"]
    assert not base.set_index("race_id").loc[after].equals(pert.set_index("race_id").loc[after])   # 翌日以降には効く（検査自体が空でない）


def test_prior_window_does_not_reach_across_a_long_break():
    from src.pipeline.features.race_env.asof import prior_and_day

    def row(date, v, rid):
        return dict(race_id=rid, date=pd.Timestamp(date), venue_code="05", surface="芝", round=1, x=v)

    rs = pd.DataFrame([row("2022-01-08", 100.0, "a"), row("2022-06-04", 1.0, "b"), row("2022-06-05", 2.0, "c"), row("2022-06-11", 3.0, "d")])
    out = prior_and_day(rs, ["x"], "t").set_index("race_id")
    assert np.isnan(out.loc["a", "t_x_prior"])
    assert np.isnan(out.loc["b", "t_x_prior"])        # 開催が空いた後の 1 日目は、前の開催の値を使わない
    assert out.loc["c", "t_x_prior"] == 1.0           # 2 日目は 1 日目だけ（100 が混ざらない）
    assert out.loc["d", "t_x_prior"] == 1.5           # 3 日目は 1・2 日目の平均
