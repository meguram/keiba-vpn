"""想定オッズの係数の置き場の代わり（ホストの作業ツリーには config/ の写しが無い）。"""
import json

from src.pipeline.inference import target_bets as TB

CO = {"poly2": {k: [0.0, 1.0, 0.0] for k in ("umaren", "wide", "umatan", "sanrenpuku", "sanrentan")}}


def test_falls_back_to_modeling_dir_when_config_copy_is_missing(tmp_path, monkeypatch):
    fb = tmp_path / "fb.json"
    fb.write_text(json.dumps(CO), encoding="utf-8")
    monkeypatch.delenv("KEIBA_ODDS_PRED_CONFIG", raising=False)
    monkeypatch.setattr(TB, "ODDS_PRED_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(TB, "ODDS_PRED_FALLBACK", fb)
    TB.reload()
    assert TB.odds_ready()
    monkeypatch.setattr(TB, "ODDS_PRED_FALLBACK", tmp_path / "missing2.json")
    TB.reload()
    assert not TB.odds_ready()


def test_explicit_env_path_does_not_fall_back(tmp_path, monkeypatch):
    fb = tmp_path / "fb.json"
    fb.write_text(json.dumps(CO), encoding="utf-8")
    monkeypatch.setenv("KEIBA_ODDS_PRED_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setattr(TB, "ODDS_PRED_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(TB, "ODDS_PRED_FALLBACK", fb)
    TB.reload()
    assert not TB.odds_ready()



def test_partial_coefficients_are_not_ready(tmp_path, monkeypatch):
    fb = tmp_path / "fb.json"
    fb.write_text(json.dumps({"poly2": {"sanrenpuku": [0.0, 1.0, 0.0]}}), encoding="utf-8")
    monkeypatch.delenv("KEIBA_ODDS_PRED_CONFIG", raising=False)
    monkeypatch.setattr(TB, "ODDS_PRED_PATH", fb)
    TB.reload()
    assert not TB.odds_ready()
    TB.reload()
