import numpy as np
import pandas as pd

from src.pipeline.features.acq_groups import nonjra_features, sibling_features


def _world(seed=0, n_horses=30, n_runs=400):
    rng = np.random.default_rng(seed)
    horses = [f"20{18 + i % 4}1{i:05d}" for i in range(n_horses)]
    dams = [f"d{i % 8}" for i in range(n_horses)]
    sires = [f"s{i % 3}" for i in range(n_horses)]
    ped = pd.DataFrame({"horse_id": horses, "sire_id": sires, "dam_id": dams})
    h = rng.integers(0, n_horses, n_runs)
    runs = pd.DataFrame({
        "horse_id": [horses[i] for i in h],
        "race_id": [f"r{j}" for j in range(n_runs)],
        "date": pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 80, n_runs), unit="D"),
        "is_jra": rng.integers(0, 2, n_runs),
        "field_size": rng.integers(8, 16, n_runs),
        "finish_position": rng.integers(1, 8, n_runs),
        "time_sec": 100.0,
    })
    keys = runs[["race_id", "horse_id", "date"]].copy()
    return runs, ped, keys


def test_nonjra_matches_brute_force_and_excludes_same_day():
    runs, ped, keys = _world()
    f = nonjra_features(runs, keys)
    for i in range(0, len(keys), 7):
        k = keys.iloc[i]
        prev = runs[(runs.horse_id == k.horse_id) & (runs.date < k.date)]
        assert f["nj_all_n"].iloc[i] == len(prev)
        assert f["nj_prior_n"].iloc[i] == (prev.is_jra == 0).sum()
        assert f["nj_jra_n"].iloc[i] == (prev.is_jra == 1).sum()
        if len(prev):
            assert f["nj_days_last_any"].iloc[i] == (k.date - prev.date.max()).days
        else:
            assert np.isnan(f["nj_days_last_any"].iloc[i])


def test_sibling_matches_brute_force_and_excludes_self():
    runs, ped, keys = _world(1)
    f = sibling_features(runs, ped, keys)
    jr = runs[runs.is_jra == 1].merge(ped, on="horse_id")
    for i in range(0, len(keys), 5):
        k = keys.iloc[i]
        dam = ped.loc[ped.horse_id == k.horse_id, "dam_id"].iloc[0]
        prev = jr[(jr.dam_id == dam) & (jr.horse_id != k.horse_id) & (jr.date < k.date)]
        assert f["sib_runs"].iloc[i] == len(prev)
        assert f["sib_n_ran"].iloc[i] == prev.horse_id.nunique()
        assert f["sib_n_winners"].iloc[i] == prev[prev.finish_position == 1].horse_id.nunique()


def test_future_runs_do_not_change_past_rows():
    runs, ped, keys = _world(2)
    cut = pd.Timestamp("2021-02-15")
    a = sibling_features(runs, ped, keys)
    b = sibling_features(runs[runs.date < cut], ped, keys)
    m = (keys.date < cut).to_numpy()
    pd.testing.assert_frame_equal(a[m].reset_index(drop=True), b[m].reset_index(drop=True))
    c = nonjra_features(runs, keys)
    d = nonjra_features(runs[runs.date < cut], keys)
    pd.testing.assert_frame_equal(c[m].reset_index(drop=True), d[m].reset_index(drop=True))


def test_partner_note_parse_and_window():
    from src.pipeline.features.acq_groups import parse_partner_note, partner_features

    p = parse_partner_note("中アニーアイル一杯と５Ｆ併せで０秒４先着 / 内ベルウッド馬也と併せ０秒１遅れ / 外フロイデ馬也と併入")
    assert [x["score"] for x in p] == [1, -1, 0]
    assert [x["margin"] for x in p] == [0.4, -0.1, 0.0]
    assert parse_partner_note("") == []
    work = pd.DataFrame({
        "horse_id": ["h"] * 3,
        "train_date": ["2024-01-01", "2024-01-20", "2024-02-10"],
        "partner_notes": ["外A馬也と併入", "外B馬也と０秒３先着", "外C馬也と０秒５遅れ"],
    })
    keys = pd.DataFrame({"race_id": ["r"], "horse_id": ["h"], "date": ["2024-02-10"]})
    f = partner_features(work, keys)
    assert f["pt_n_work"].iloc[0] == 1 and f["pt_ahead_rate"].iloc[0] == 1.0  # 当日(02-10)は含まず、35 日窓は 01-06 以降 → 01-20 だけ
