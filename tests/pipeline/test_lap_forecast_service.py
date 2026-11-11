import json

import numpy as np
import pytest

from src.pipeline.inference import lap_forecast_service as svc
from src.pipeline.models import lap_forecast as lfm


def _out(n_laps=6, K=3):
    return {"par": np.array([[12.0, 11.0, 11.5, 11.8, 11.7, 12.0] + [np.nan] * (lfm.MAXJ - 6)]), "n_laps": np.array([n_laps]),
            "proba": np.array([[0.2, 0.5, 0.3]]),
            "mu_dev": np.stack([np.array([[+0.3, +0.2, +0.2, -0.1, -0.3, -0.4] + [np.nan] * (lfm.MAXJ - 6)]),
                                np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [np.nan] * (lfm.MAXJ - 6)]),
                                np.array([[-0.4, -0.3, -0.2, 0.2, 0.4, 0.6] + [np.nan] * (lfm.MAXJ - 6)])], axis=1),
            "sd": np.full((K, lfm.MAXJ), 0.2)}


def test_lap_meters():
    assert svc.lap_meters(1700) == [100] + [200] * 8
    assert svc.lap_meters(1200) == [200] * 6


def test_mixture_summary_shape_and_probabilities():
    s = lfm.mixture_summary(_out(), 0, ["スロー", "ミドル", "ハイ"], n_draw=200, seed=1)
    assert s["n_laps"] == 6 and len(s["expected_laps"]) == 6 and len(s["band80_lo"]) == 6
    assert abs(sum(x["prob"] for x in s["scenarios"]) - 1.0) < 1e-3
    assert s["likely_scenario"] == "ミドル" and s["likely_prob"] == 0.5
    # 期待ラップは確率の重み付き平均、前半3Fは最初の3区間の和
    exp0 = 0.2 * (12.0 + 0.3) + 0.5 * 12.0 + 0.3 * (12.0 - 0.4)
    assert abs(s["expected_laps"][0] - exp0) < 0.01
    assert abs(s["scenarios"][0]["first3f"] - (12.3 + 11.2 + 11.7)) < 0.02
    assert all(lo <= hi for lo, hi in zip(s["band80_lo"], s["band80_hi"]))


def test_attach_is_noop_when_disabled_or_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(svc.ENV_DIR, str(tmp_path))
    resp = {"predictions": []}
    monkeypatch.delenv(svc.ENV_FLAG, raising=False)
    assert svc.attach_lap_forecast(resp, "R1", "2026-10-17") is resp
    monkeypatch.setenv(svc.ENV_FLAG, "1")
    assert svc.attach_lap_forecast(resp, "R1", "2026-10-17") is resp          # 保存なし
    doc = svc.make_doc("R1", "2026-10-17", 1200, lfm.mixture_summary(_out(), 0, ["a", "b", "c"], n_draw=50), "A_noten", [12.0] * 6)
    svc.save_forecast(doc)
    out = svc.attach_lap_forecast(resp, "R1", "2026-10-17")
    assert out["lap_forecast"]["race_id"] == "R1" and out["lap_forecast"]["lap_meters"] == [200] * 6 and "predictions" in out
    assert json.loads(svc.path_for("R1", "2026-10-17").read_text(encoding="utf-8"))["version"] == svc.VERSION


def test_forecast_frame_uses_full_or_noten_model_by_ten_availability():
    from src.scripts.data import tp55_backfill as bf
    from tests.pipeline.test_tp55_model import _synth

    df = _synth(500)
    train, test = df[df["year"] < 2025], df[df["year"] == 2025].copy().reset_index(drop=True)
    m_full = lfm.LapForecast(scenario="A", threads=2, n_oof=2).fit(train)
    m_no = lfm.LapForecast(scenario="A", threads=2, n_oof=2, drop_groups=("ten",)).fit(train)
    test["ten_cnt"] = [5 if i % 2 == 0 else 0 for i in range(len(test))]
    docs = bf.forecast_frame(test, {"A_full": m_full, "A_noten": m_no}, {"A_full": m_full.scenario_names(), "A_noten": m_no.scenario_names()})
    assert len(docs) == len(test)
    models = {d["race_id"]: d["model"] for d in docs}
    assert {models[r] for r, c in zip(test["race_id"], test["ten_cnt"]) if c > 0} == {"A_full"}
    assert {models[r] for r, c in zip(test["race_id"], test["ten_cnt"]) if c == 0} == {"A_noten"}
    d = docs[0]
    assert abs(sum(s["prob"] for s in d["scenarios"]) - 1.0) < 1e-3 and d["lap_meters"][0] in (100, 200) and len(d["expected_laps"]) == d["n_laps"]


def _train_lapd(tmp_path):
    import lightgbm as lgb
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"e1": rng.normal(size=n), "venue": pd.Categorical(rng.choice(["中山", "東京"], n), categories=["中山", "東京"])})
    y = X["e1"] * 2 + (X["venue"] == "東京") * 1.0
    (tmp_path / "lapd").mkdir()
    for t in ("ld_f3", "ld_q"):
        for s in (0, 1):
            lgb.train({"objective": "regression", "verbose": -1, "seed": s, "min_data_in_leaf": 10}, lgb.Dataset(X, label=y), num_boost_round=30).save_model(str(tmp_path / "lapd" / f"{t}_s{s}.txt"))
    (tmp_path / "lapd_meta.json").write_text(json.dumps({"num": ["e1"], "cats": ["venue"], "levels": {"venue": ["中山", "東京"]}, "seeds": [0, 1], "targets": ["ld_f3", "ld_q"]}), encoding="utf-8")
    return X, y


def test_lapd_model_predicts_in_training_scale_and_handles_unseen_category(tmp_path):
    import pandas as pd

    X, y = _train_lapd(tmp_path)
    m = svc.LapdModel(tmp_path)
    out = m.predict(pd.DataFrame({"e1": X["e1"].to_numpy(), "venue": X["venue"].astype(str).to_numpy()}))
    assert np.corrcoef(out["ld_f3"], y)[0, 1] > 0.95
    odd = m.predict(pd.DataFrame({"e1": [0.0], "venue": ["札幌"]}))   # 学習に無い場は欠損扱いで落ちない
    assert np.isfinite(odd["ld_f3"][0])


def test_make_doc_saves_lapd_and_serving_reads_it():
    from src.pipeline.features import serving_v6 as sv

    s = lfm.mixture_summary(_out(), 0, ["スロー", "ミドル", "ハイ"], n_draw=50, seed=1)
    par = [12.0, 11.0, 11.5, 11.8, 11.7, 12.0]
    doc = svc.make_doc("r1", "2025-04-05", 1200, s, "A_full", par, {"ld_f3": np.float64(0.123456), "ld_l3": float("nan"), "ld_q": -0.5})
    assert doc["lapd"] == {"ld_f3": 0.1235, "ld_l3": None, "ld_q": -0.5}
    cols = sv.lapd_race_columns(doc)
    assert cols["ld_f3"] == 0.1235 and np.isnan(cols["ld_l3"])
    assert "lapd" not in svc.make_doc("r1", "2025-04-05", 1200, s, "A_full", par)
    assert all(np.isnan(v) for v in sv.lapd_race_columns({}).values())


class _FakeLapd:
    def predict(self, sub):
        return {"ld_f3": sub["distance"].to_numpy(float), "ld_l3": -sub["distance"].to_numpy(float), "ld_q": np.zeros(len(sub))}


def test_forecast_frame_attaches_lapd_to_the_right_race_after_split(tmp_path):
    import pandas as pd

    from src.pipeline.features import serving_v6 as sv
    from src.scripts.data import tp55_backfill as bf
    from tests.pipeline.test_tp55_model import _synth

    df = _synth(500)
    train, test = df[df["year"] < 2025], df[df["year"] == 2025].copy().reset_index(drop=True)
    m_full = lfm.LapForecast(scenario="A", threads=2, n_oof=2).fit(train)
    m_no = lfm.LapForecast(scenario="A", threads=2, n_oof=2, drop_groups=("ten",)).fit(train)
    test["ten_cnt"] = [5 if i % 2 == 0 else 0 for i in range(len(test))]
    names = {"A_full": m_full.scenario_names(), "A_noten": m_no.scenario_names()}
    docs = bf.forecast_frame(test, {"A_full": m_full, "A_noten": m_no}, names, _FakeLapd())
    dist = dict(zip(test["race_id"].astype(str), test["distance"]))
    assert all(d["lapd"]["ld_f3"] == float(dist[d["race_id"]]) and d["lapd"]["ld_l3"] == -float(dist[d["race_id"]]) for d in docs)
    # JSON 保存 → 学習側と同じ名前の列で読める。lapd が無い日だけなら黙らず止まる
    old = svc.make_doc("OLD", "2025-04-05", 1200, lfm.mixture_summary(_out(), 0, ["a", "b", "c"], n_draw=50), "A_full", [12.0] * 6)
    new = {**old, "race_id": "NEW", "lapd": {"ld_f3": 1.5, "ld_l3": -2.0, "ld_q": 0.25}}
    for d in (old, new):
        p = tmp_path / "lap_forecast" / "2025-04-05" / f"{d['race_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d), encoding="utf-8")
    base = pd.DataFrame({"race_id": ["NEW", "OLD"], "horse_id": ["h1", "h2"], "date": ["2025-04-05"] * 2})
    out = sv.stage_forecast_columns(base, ["ld_f3", "ld_l3", "ld_q", "lp_p_slow"], tmp_path)
    assert out.loc[0, ["ld_f3", "ld_l3", "ld_q"]].tolist() == [1.5, -2.0, 0.25] and out.loc[1, ["ld_f3", "ld_l3", "ld_q"]].isna().all()
    only_old = base[base["race_id"] == "OLD"].reset_index(drop=True)
    with pytest.raises(KeyError, match="lapd"):
        sv.stage_forecast_columns(only_old, ["ld_f3"], tmp_path)
