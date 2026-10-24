"""tp-37: 調教師ごとの時計の標準化・交互作用・本数/間隔/前走比・追い切り騎手・短評が『日付が厳密に前』の情報だけから作られていること（bit 単位）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import tp37_workout as W

KINDS = [("slope", "栗", "栗坂"), ("wood", "栗", "CW"), ("slope", "美", "美坂"), ("poly", "美", "美P")]
LEG = ["馬也", "強め", "一杯", "仕掛", "G強"]
EVAL = ["順調", "動き軽快", "まずまず", "目立たず"]


def _toy(seed: int = 0, n_days: int = 80):
    rng = np.random.default_rng(seed)
    days = [d for d in pd.date_range("2023-01-07", periods=n_days * 7, freq="D") if d.weekday() in (5, 6)][:n_days]
    horses = [f"h{i}" for i in range(120)]
    trainers = [f"t{i}" for i in range(24)]
    runs, laps = [], []
    for d in days:
        for r in range(4):
            rid = f"{d:%Y%m%d}{r:02d}"
            field = list(rng.choice(horses, size=10, replace=False))
            fin = rng.permutation(10) + 1
            for h, f in zip(field, fin):
                runs.append({"race_id": rid, "horse_id": h, "date": d.strftime("%Y-%m-%d"), "trainer_id": trainers[int(h[1:]) % 24], "finish_position": int(f), "field_size": 10,
                             "venue_code": "05" if r % 2 else "06", "surface": "芝" if r < 2 else "ダ", "distance": [1200, 1600, 2000, 1800][r],
                             "a_style": float(rng.uniform()), "grade": ["未勝利", "1勝", "", "OP"][r]})
                for off in (1, 2, 5, 8, 9, 15):
                    td = (d - pd.Timedelta(days=off)).strftime("%Y-%m-%d")
                    kind, center, course = KINDS[int(rng.integers(0, len(KINDS)))]
                    t1 = float(np.round(rng.uniform(11.0, 14.5), 1))
                    t2 = float(np.round(t1 + rng.uniform(11.0, 14.0), 1))
                    t3 = float(np.round(t2 + 13.0, 1))
                    laps.append({"horse_id": h, "train_date": td, "race_date": d.strftime("%Y-%m-%d"), "course": course, "kind": kind, "center": center, "cond": ["良", "稍", "重"][int(rng.integers(0, 3))],
                                 "rider": ["助手", "師", "見習", "", "黛", "丹内"][int(rng.integers(0, 6))], "position": float(rng.integers(1, 9)), "leg_color": LEG[int(rng.integers(0, len(LEG)))],
                                 "rank": "ABCD"[int(rng.integers(0, 4))], "evaluation": EVAL[int(rng.integers(0, len(EVAL)))],
                                 "t1": t1, "t2": t2, "t3": t3, "t4": float(np.round(t3 + 13.5, 1)), "t5": np.nan, "t6": np.nan, "days_before": off,
                                 "accel": float(np.round(t1 - (t3 - t1) / 2, 2))})
    return pd.DataFrame(runs), pd.DataFrame(laps)


def _returns(runs: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"win_ret": np.where(runs["finish_position"].to_numpy() == 1, rng.uniform(2, 20, len(runs)), 0.0),
                         "place_ret": np.where(runs["finish_position"].to_numpy() <= 3, rng.uniform(1, 5, len(runs)), 0.0)})


def test_prior_sums_match_bruteforce_and_exclude_same_day():
    rng = np.random.default_rng(1)
    n = 3000
    code = rng.integers(-1, 7, n)
    day = rng.integers(0, 60, n)
    V = np.column_stack([np.ones(n), rng.normal(size=n) * 0.37])
    got = W.prior_sums(code, day, V)
    for i in rng.choice(n, 200, replace=False):
        if code[i] < 0:
            assert (got[i] == 0).all()
            continue
        m = (code == code[i]) & (day < day[i])
        assert got[i, 0] == m.sum()                                              # 件数は厳密に一致（同日・自分は含まない）
        assert abs(got[i, 1] - V[m, 1].sum()) < 1e-12


def test_prior_sums_bitwise_independent_of_future_rows():
    rng = np.random.default_rng(2)
    n = 4000
    code = rng.integers(0, 5, n)
    day = rng.integers(0, 80, n)
    V = np.column_stack([np.ones(n), rng.normal(size=n)])
    a = W.prior_sums(code, day, V)
    V2 = V.copy()
    late = day >= 50
    V2[late, 1] = rng.normal(size=late.sum()) * 1000                              # 50 日目以降の値をまるごと入れ替える
    b = W.prior_sums(code, day, V2)
    assert np.array_equal(a[day <= 50], b[day <= 50])                             # 50 日目までの行は 1 ビットも変わらない
    assert not np.array_equal(a[day > 60], b[day > 60])                           # 検査が効いている


def test_standardized_times_are_centered_and_trainer_habit_is_removed():
    runs, laps = _toy(0)
    Wk = W.add_standardized(W.prepare_workouts(laps, runs), "2023-04-01")
    z = Wk["z0_t4"].dropna()
    assert len(z) > 5000 and abs(z.mean()) < 0.15 and 0.7 < z.std() < 1.3
    assert Wk["ztr_t4"].notna().sum() > 3000
    assert (Wk.loc[Wk["ztr_t4"].notna(), "trn_n"] >= W.MIN_TRAINER).all()          # 調教師の過去が少ないうちは出さない
    assert Wk["clu"].max() >= 1 and Wk["clu"].min() >= -1                           # クラスタが付く


def test_trainer_with_slow_habit_is_standardized_toward_zero():
    # 調教師 A は常に 1 秒遅い。標準化後（調教師比）の平均は、標準化前より 0 に近い
    runs, laps = _toy(3)
    trainer = laps["horse_id"].map(lambda h: "t0" if int(h[1:]) % 24 == 0 else "t1")
    laps["t4"] = laps["t4"] + np.where(laps["horse_id"].map(lambda h: int(h[1:]) % 24 == 0), 2.0, 0.0)
    Wk = W.add_standardized(W.prepare_workouts(laps, runs), "2023-04-01")
    late = Wk[Wk["train_day"] > int(np.datetime64("2023-06-01", "D").astype(np.int64))]
    slow = late[late["trainer_id"] == "t0"]
    assert slow["z0_t4"].mean() > 0.5
    assert abs(slow["ztr_t4"].mean()) < abs(slow["z0_t4"].mean()) * 0.5
    del trainer


def test_features_use_only_strictly_earlier_information_bitwise():
    runs, laps = _toy()
    fit_before = "2023-04-01"
    base = W.build_features(runs, laps, _returns(runs, 1), fit_before)
    cut = pd.Timestamp(runs["date"].iloc[-1]) - pd.Timedelta(days=35)
    runs2, laps2 = runs.copy(), laps.copy()
    rng = np.random.default_rng(7)
    late_run = pd.to_datetime(runs["date"]) >= cut
    for _, ix in runs2[late_run].groupby("race_id").groups.items():
        runs2.loc[ix, "finish_position"] = rng.permutation(runs2.loc[ix, "finish_position"].to_numpy())
    late_w = pd.to_datetime(laps["train_date"]) >= cut                                   # 切り替え日以降の追い切りの時計・脚色・短評・騎手を入れ替える
    n = int(late_w.sum())
    for c in ("t1", "t2", "t3", "t4", "accel"):
        laps2.loc[late_w, c] = laps2.loc[late_w, c].to_numpy() + rng.normal(size=n) * 3.0
    laps2.loc[late_w, "leg_color"] = rng.choice(LEG, n)
    laps2.loc[late_w, "evaluation"] = rng.choice(EVAL, n)
    laps2.loc[late_w, "rider"] = rng.choice(["助手", "師", "黛", "丹内"], n)
    alt = W.build_features(runs2, laps2, _returns(runs2, 1), fit_before)
    day = pd.to_datetime(runs["date"])
    same_or_before = (day <= cut).to_numpy()                                              # 切り替え日当日の行も変わらない（当日の結果・時計は入らない）
    nonnull = 0
    for g, f in base.items():
        if g == "KEYS":
            continue
        cols = [c for c in f.columns if c not in ("race_id", "horse_id")]
        a, b = f.loc[same_or_before, cols].to_numpy(dtype=float), alt[g].loc[same_or_before, cols].to_numpy(dtype=float)
        assert np.array_equal(a, b, equal_nan=True), f"{g}: 未来の結果・時計を変えると過去/当日の特徴量が変わる"
        nonnull += int(np.isfinite(a).sum())
    assert nonnull > 5000                                                                 # 空振り（全て欠損）でない
    changed = [g for g, f in base.items() if g != "KEYS" and not np.array_equal(f.loc[~same_or_before].drop(columns=["race_id", "horse_id"]).to_numpy(dtype=float),
                                                              alt[g].loc[~same_or_before].drop(columns=["race_id", "horse_id"]).to_numpy(dtype=float), equal_nan=True)]
    assert len(changed) >= 4                                                              # 切り替え日より後の行は変わる（検査が効いている）


def test_groups_have_expected_columns_and_rows():
    runs, laps = _toy()
    out = W.build_features(runs, laps, _returns(runs, 1), "2023-04-01")
    assert len(out) == 20 and "KEYS" in out
    for g, f in out.items():
        assert len(f) == len(runs) and (f["race_id"].to_numpy() == runs["race_id"].astype(str).to_numpy()).all(), g
        assert f.shape[1] >= 5, g
    assert {"f_ztr_t4", "f_ztrs_t4", "m_ztr_t4"} <= set(out["TRZ"].columns)
    assert {"f_clu", "f_zh_t4"} <= set(out["CLZ"].columns)


PHRASES = ["追走して先着と動きの良さは目立っていた。好仕上がり。", "平凡な動きで時計も詰まってこない。疑問。", "手応え劣勢も最後まで食らいついた。", "同入で順調。上々の伸び。", "遅れたが力強い。初戦から。", "地味で物足りない。叩いてから。"]


def _comments(laps: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = laps[rng.random(len(laps)) < 0.35][["horse_id", "train_date", "race_date"]].copy()
    pick["comment"] = rng.choice(PHRASES, len(pick))
    return pick.reset_index(drop=True)


def test_text_flags_extract_partner_results_and_sentiment():
    f = W.text_flags(pd.Series(["追走して先着と動きの良さは目立っていた。好仕上がり。", "平凡な動きで物足りない", "手応え劣勢も食らいついた", None]))
    assert f.loc[0, "sensen"] == 1 and f.loc[0, "tsuiso"] == 1 and f.loc[0, "net"] > 0
    assert f.loc[1, "net"] < 0 and f.loc[1, "sensen"] == 0
    assert f.loc[2, "ressei"] == 1 and f.loc[2, "awase"] == 1
    assert f.loc[3, ["pos", "neg", "sensen"]].sum() == 0


def test_text_groups_use_only_strictly_earlier_information_bitwise():
    runs, laps = _toy()
    com = _comments(laps)
    fit_before = "2023-04-01"
    base = W.build_features(runs, laps, _returns(runs, 1), fit_before, comments=com)
    assert {"D7_text_flags", "D7_text_lift", "D8_text_ctx"} <= set(base)
    cut = pd.Timestamp(runs["date"].iloc[-1]) - pd.Timedelta(days=35)
    runs2, com2 = runs.copy(), com.copy()
    rng = np.random.default_rng(9)
    for _, ix in runs2[pd.to_datetime(runs["date"]) >= cut].groupby("race_id").groups.items():
        runs2.loc[ix, "finish_position"] = rng.permutation(runs2.loc[ix, "finish_position"].to_numpy())
    late = (pd.to_datetime(com["train_date"]) >= cut).to_numpy()
    com2.loc[late, "comment"] = rng.choice(PHRASES, int(late.sum()))
    alt = W.build_features(runs2, laps, _returns(runs2, 1), fit_before, comments=com2)
    same_or_before = (pd.to_datetime(runs["date"]) <= cut).to_numpy()
    n = 0
    for g in ("D7_text_flags", "D7_text_lift", "D8_text_ctx"):
        cols = [c for c in base[g].columns if c not in ("race_id", "horse_id")]
        a, b = base[g].loc[same_or_before, cols].to_numpy(dtype=float), alt[g].loc[same_or_before, cols].to_numpy(dtype=float)
        assert np.array_equal(a, b, equal_nan=True), g
        n += int(np.isfinite(a).sum())
        assert not np.array_equal(base[g].loc[~same_or_before, cols].to_numpy(dtype=float), alt[g].loc[~same_or_before, cols].to_numpy(dtype=float), equal_nan=True), g
    assert n > 1000
