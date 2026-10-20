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
