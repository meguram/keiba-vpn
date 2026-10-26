"""推奨馬券の想定オッズ(§77)を作れたかの記録と、モデルの公開漏れの検出。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from src.pipeline.inference import odds_sec77 as os77
from src.pipeline.inference import weight_triggered_predict as wtp

ROOT = Path(__file__).resolve().parents[2]


def _card():
    return pl.DataFrame({"race_id": ["202606010101", "202606010101", "202606010102"], "horse_id": ["a", "b", "c"], "horse_number": [1, 2, 1]})


def test_status_is_no_model_and_error_logged_when_model_missing(monkeypatch, caplog):
    monkeypatch.setenv("KEIBA_ODDS_SEC77", "1")
    with caplog.at_level("ERROR"):
        st = wtp._save_sec77_odds(SimpleNamespace(sec77=None), _card(), None, None, None, ["x"])
    assert st == {"202606010101": "no_model", "202606010102": "no_model"}
    assert any("モデルが無い" in r.getMessage() for r in caplog.records)


def test_status_is_disabled_when_switched_off(monkeypatch):
    monkeypatch.setenv("KEIBA_ODDS_SEC77", "0")
    st = wtp._save_sec77_odds(SimpleNamespace(sec77=None), _card(), None, None, None, ["x"])
    assert set(st.values()) == {"disabled"} and not os77.enabled()


def test_status_is_error_for_every_race_when_computation_fails(monkeypatch):
    class Boom:
        def features(self, *a, **k):
            raise RuntimeError("boom")

    st = wtp._save_sec77_odds(SimpleNamespace(sec77=Boom(), v6=None), _card(), None, None, None, ["x"])
    assert st == {"202606010101": "error", "202606010102": "error"}


def test_every_job_that_predicts_ships_the_sec77_model():
    jm = json.loads((ROOT / "config/gcp_job_map.json").read_text(encoding="utf-8"))
    comp = jm["models"]["odds_sec77"]
    assert comp["env"] == "ODDS_SEC77_DIR" and comp["local_path"].endswith("odds_pred/serving")
    users = [j for j in jm["jobs"] if "bundle_v6e" in (j.get("models") or [])]
    assert users and all("odds_sec77" in j["models"] for j in users)
