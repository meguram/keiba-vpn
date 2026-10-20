import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm
from src.scripts.data import tp55_eval as ev


def _synth(n=700, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pressure = rng.normal()
        dist = int(rng.choice([1200, 1600, 1700, 2000]))
        nl = lf.n_laps_of(dist)
        d0 = lf.first_lap_meters(dist)
        base = np.array([d0 / 200 * 6.0 if k == 0 else 12.0 for k in range(nl)])
        fast = pressure * 0.5                          # 圧力が高いほど前半が速く、後半が遅い（ハイペース）
        laps = base.copy()
        laps[:3] -= fast
        laps[-3:] += fast
        laps += rng.normal(0, 0.15, nl)
        rows.append({
            "race_id": f"R{i:05d}", "date": f"{2020 + i * 6 // n}-{1 + i % 12:02d}-{1 + i % 27:02d}", "venue": rng.choice(["東京", "中山"]), "surface": "芝",
            "distance": dist, "direction": "左", "grade": "", "race_class": "未勝利", "weather": "晴", "track_condition": "良", "field_size": 16,
            "round_no": 1 + i % 12, "kai": 1, "nichime": 1, "month": 1, "n_laps": nl, "d0": d0, "press": pressure + rng.normal(0, 0.3),
            "e1": 0.1, "unknown_ratio": 0.0, "n_entrants": 16, "laps_json": json.dumps([round(x, 1) for x in laps]),
            "course_key": f"x|{dist}", "sd_n": 0,
        })
    df = pd.DataFrame(rows)
    df["year"] = df["date"].str[:4].astype(int)
    return df


def test_feature_columns_drop_groups():
    n_all, c_all = lfm.feature_columns(())
    n2, c2 = lfm.feature_columns(("ten",))
    assert "ten_min" in n_all and "ten_min" not in n2
    n3, c3 = lfm.feature_columns(("course", "cond"))
    assert "distance" not in n3 and "venue" not in c3 and "weather" not in c3


def test_fit_predict_and_evaluate_synthetic():
    df = _synth()
    train, test = df[df["year"] < 2025], df[df["year"] == 2025]
    m = lfm.LapForecast(scenario="A", threads=2, n_oof=2).fit(train)
    out = m.predict(test)
    assert out["proba"].shape == (len(test), 3) and np.allclose(out["proba"].sum(axis=1), 1.0)
    res, pred = ev.evaluate(m, train, test, n_draw=100)
    c = res["classify"]
    assert c["logloss"] < c["logloss_prior"]                         # 圧力の特徴量でシナリオが当たる
    assert res["agg"]["first3f"]["mixture_mae"] < res["agg"]["first3f"]["par_mae"]
    assert res["coverage"]["lap0"]["coverage80"] is not None
    d = lfm.mixture_summary(out, 0, m.scenario_names())
    assert abs(sum(s["prob"] for s in d["scenarios"]) - 1.0) < 1e-6 and d["n_laps"] == len(d["expected_laps"])


def test_scenario_b_labels_sorted_by_pace():
    df = _synth()
    m = lfm.LapForecast(scenario="B", k_cluster=3, threads=2, n_oof=2).fit(df[df["year"] < 2025])
    c = m.cen_sorted
    assert (np.diff(c[:, 2] - c[:, 0]) >= -1e-9).all()               # 『後半-前半』の小さい（スロー寄り）順に並ぶ


def test_save_load_roundtrip(tmp_path):
    df = _synth(400)
    m = lfm.LapForecast(scenario="A", threads=2, n_oof=2).fit(df[df["year"] < 2025])
    m.save(tmp_path / "m.pkl")
    m2 = lfm.LapForecast.load(tmp_path / "m.pkl")
    a = m.predict(df[df["year"] == 2025])["proba"]
    b = m2.predict(df[df["year"] == 2025])["proba"]
    assert np.allclose(a, b)
