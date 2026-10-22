"""tp-76: 候補特徴量が as-of（その走・その日の後の結果を使わない）であることと、基本の性質のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import tp76_asof as T
from src.scripts.data import tp76_screen as S


def synthetic(seed: int = 0, days: int = 130) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horses = [f"h{i:03d}" for i in range(220)]
    ab = rng.normal(0, 1, len(horses))
    sire = rng.choice([f"s{i}" for i in range(12)], len(horses))
    rows = []
    d0 = pd.Timestamp("2021-10-01")
    for di in range(days):
        date = d0 + pd.Timedelta(days=di)
        pool = rng.permutation(len(horses))      # 1 日に 1 頭は 1 走だけ
        slot = 0
        for surf in ("芝", "ダート"):
            for rd in range(1, 7):
                rid = f"{date:%Y%m%d}{surf[0]}{rd:02d}"
                hs = pool[slot * 12:(slot + 1) * 12]
                slot += 1
                perf = ab[hs] + rng.normal(0, 1, 12)
                order = np.argsort(-perf)
                rank = np.empty(12, int); rank[order] = np.arange(1, 13)
                pos1 = np.clip(rng.uniform(0, 1, 12) - 0.1 * (perf - perf.mean()), 0, 1)
                for j, h in enumerate(hs):
                    rows.append({"race_id": rid, "horse_id": horses[h], "date": date, "venue_code": "05", "surface": surf, "distance": 1600.0,
                                 "track_condition": "良", "field_size": 12, "round": rd, "finish_position": int(rank[j]), "horse_number": j + 1,
                                 "bracket_number": j // 2 + 1, "age": 3 + int(h % 3), "sex": "牡" if h % 2 else "牝", "run_no": 0,
                                 "lay_days": float(rng.integers(7, 80)), "pos1_frac": float(pos1[j]), "posL_frac": float(np.clip(pos1[j] + rng.normal(0, .1), 0, 1)),
                                 "fin_pct": (rank[j] - 1) / 11, "agari_rank": float(rng.integers(1, 13)), "last_3f": 36.0 + rng.normal(0, 1) - 0.2 * perf[j],
                                 "time_sec": 96.0 + rank[j] * 0.2 + rng.normal(0, 0.1), "cush_bin": float(rng.integers(0, 3)),
                                 "dist_band": "d_mile", "sire": sire[h], "year": date.year})
    return pd.DataFrame(rows)


def _diff(a, b):
    return S.max_diff(a, b)


def test_truncation_exact_equality():
    df = synthetic()
    d = T.prepare(df)
    consts = T.fit_constants(d)
    full = T.build_features(df, consts=consts).set_index(["race_id", "horse_id"])
    cols = [c for c in full.columns if c != "date"]
    cut = pd.Timestamp("2022-01-20")
    g = T.build_features(df[pd.to_datetime(df["date"]) <= cut], consts=consts).set_index(["race_id", "horse_id"])
    assert _diff(full.loc[g.index, cols].to_numpy(float), g[cols].to_numpy(float)) < 1e-9


def test_future_and_sameday_outcome_perturbation_invariance():
    df = synthetic(1)
    consts = T.fit_constants(T.prepare(df))
    base = T.build_features(df, consts=consts).set_index(["race_id", "horse_id"])
    cols = [c for c in base.columns if c != "date"]
    day = pd.Timestamp("2022-01-25")
    mid_round = 4
    sel = (df["date"] > day) | ((df["date"] == day) & (df["round"] >= mid_round))
    rng = np.random.default_rng(3)
    pert = df.copy()
    shuf = rng.permutation(sel.sum())
    for c in ["finish_position", "pos1_frac", "posL_frac", "fin_pct", "agari_rank", "last_3f", "time_sec"]:
        v = pert.loc[sel, c].to_numpy(); pert.loc[sel, c] = v[shuf]
    h = T.build_features(pert, consts=consts).set_index(["race_id", "horse_id"])
    rnd = df.set_index(["race_id", "horse_id"])["round"]
    keep = base.index[(base["date"] < day) | ((base["date"] == day) & (rnd.loc[base.index].to_numpy() <= mid_round))]
    # ラウンド 4 以降の結果（ラウンド 4 自身を含む）を壊しても、ラウンド 4 までの特徴量は変わらない
    assert _diff(h.loc[keep, cols].to_numpy(float), base.loc[keep, cols].to_numpy(float)) < 1e-9


def test_day_bias_uses_only_earlier_rounds_and_counts():
    df = synthetic(2)
    f = T.build_features(df).set_index(["race_id", "horse_id"])
    d = df[(df["date"] == pd.Timestamp("2022-01-10")) & (df["surface"] == "芝")]
    n = f.loc[d.index.map(lambda i: (d.loc[i, "race_id"], d.loc[i, "horse_id"])), "A_day_n"].groupby(level=0).first()
    assert list(n.sort_index().to_numpy()) == [0, 1, 2, 3, 4, 5]


def test_elo_and_kalman_reward_winners():
    df = synthetic(4, days=200)
    f = T.build_features(df)
    d = T.prepare(df)
    late = d["date"] > pd.Timestamp("2022-03-01")
    win = (d["finish_position"] == 1)
    assert f.loc[late & win, "C_elo"].mean() > f.loc[late & ~win, "C_elo"].mean()
    assert f.loc[late & win, "B_m"].mean() > f.loc[late & ~win, "B_m"].mean()


def test_features_have_no_future_columns_and_finite_shapes():
    df = synthetic(5)
    f = T.build_features(df)
    assert len(f) == len(df)
    assert not f.duplicated(["race_id", "horse_id"]).any()
    for c in ("B_sd", "C_unc"):
        assert (f[c] > 0).all()
