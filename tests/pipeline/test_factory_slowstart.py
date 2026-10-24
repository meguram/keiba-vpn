"""G6 出遅れ率: 先読み検査（将来の結果を変えても、その日以前の行の特徴量が変わらない）と定義の確認。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import factory_slowstart as fs


def _toy(n_days: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=7 * d)
        for r in range(3):
            hs = rng.choice(12, size=8, replace=False)
            for k, h in enumerate(hs):
                rows.append({"race_id": f"R{d:03d}{r}", "horse_id": f"h{h}", "date": date, "jockey_id": f"j{rng.integers(0, 4)}", "trainer_id": f"t{rng.integers(0, 3)}",
                             "venue_code": "05", "surface": "芝" if r % 2 else "ダ", "distance": 1600 + 400 * (r % 2), "horse_number": k + 1, "field_size": 8,
                             "bracket_number": k // 2 + 1, "sex_age": "牡4", "weight": 480, "finish_position": int(rng.integers(1, 9))})
    frame = pd.DataFrame(rows)
    lab = frame[["horse_id", "date"]].drop_duplicates(["horse_id", "date"]).copy()
    lab["slow_b"] = (rng.random(len(lab)) < 0.1).astype(float)
    parents = pd.DataFrame({"horse_id": [f"h{i}" for i in range(12)], "sire_id": [f"s{i % 3}" for i in range(12)], "dam_sire_id": [f"d{i % 2}" for i in range(12)]})
    return frame, lab, parents


def _all(frame, lab, parents):
    f = fs.prepare(frame)
    g = fs.build_groups(f, lab, "slow_b", parents)
    X = pd.concat(list(g.values()) + [fs.with_damage(f, lab, "slow_b")], axis=1)
    return X


def test_no_leak_when_future_labels_change():
    frame, lab, parents = _toy()
    cut = pd.Timestamp("2022-05-01")
    a = _all(frame, lab, parents)
    lab2 = lab.copy()
    lab2.loc[lab2["date"] >= cut, "slow_b"] = 1.0 - lab2.loc[lab2["date"] >= cut, "slow_b"]
    frame2 = frame.copy()
    late = frame2["date"] >= cut
    frame2.loc[late, "finish_position"] = (frame2.loc[late, "finish_position"] % 8) + 1
    b = _all(frame2, lab2, parents)
    early = (frame["date"] <= cut).to_numpy()
    pd.testing.assert_frame_equal(a[early].reset_index(drop=True), b[early].reset_index(drop=True), check_exact=False, atol=1e-9)


def test_horse_rate_counts_only_earlier_races():
    frame, lab, _ = _toy()
    f = fs.prepare(frame)
    g = fs.horse_windows(f, lab, "slow_b", 0.1, "ss")
    h = f["horse_id"].iloc[0]
    for i in np.flatnonzero((f["horse_id"] == h).to_numpy())[:6]:
        past = lab[(lab["horse_id"] == h) & (lab["date"] < f["date"].iloc[i])]
        assert g["ss_h_n"].iloc[i] == len(past)


def test_proxy_flags_a_slow_start_and_uses_only_prior_runs():
    rows = [{"horse_id": "a", "rdate": f"2019{m:02d}01", "ten": 12.0 + 0.01 * (m % 3), "c1_rel": 0.2, "trackkind": 1, "range": 1600} for m in range(1, 12)]
    rows += [{"horse_id": "a", "rdate": "20200101", "ten": 12.0, "c1_rel": 0.2, "trackkind": 1, "range": 1600},
             {"horse_id": "a", "rdate": "20200201", "ten": 14.5, "c1_rel": 0.8, "trackkind": 1, "range": 1600}]
    rows += [{"horse_id": f"b{i}", "rdate": f"2018{(i % 12) + 1:02d}15", "ten": 12.0 + 0.1 * ((i * 7) % 10 - 5), "c1_rel": 0.5, "trackkind": 1, "range": 1600} for i in range(200)]
    c = pd.DataFrame(rows)
    lab = fs.proxy_labels(c)
    a = lab[lab["horse_id"] == "a"].set_index("date")
    assert a.loc[pd.Timestamp("2020-02-01"), "slow_a"] == 1.0
    assert a.loc[pd.Timestamp("2020-01-01"), "slow_a"] == 0.0
    c2 = c.copy()
    c2.loc[(c2["horse_id"] == "a") & (c2["rdate"] == "20200201"), "ten"] = 12.0           # 未来の値を変えても、過去の resid は変わらない
    a2 = fs.proxy_labels(c2).query("horse_id == 'a'").set_index("date")
    assert np.isclose(a.loc[pd.Timestamp("2020-01-01"), "resid"], a2.loc[pd.Timestamp("2020-01-01"), "resid"])
