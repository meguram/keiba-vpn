"""ローカルに残した予測（shadow / バックフィル）を画面向けの応答にする処理。優先順位・注記・期待値の計算。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.api import local_predictions as LP


class FakeStorage:
    def __init__(self, data):
        self.data = data

    def load(self, cat, key, *a, **k):
        return self.data.get((cat, key))


def _probs():
    return pd.DataFrame({
        "race_id": ["R1"] * 3, "horse_id": ["h1", "h2", "h3"],
        "p_win": [0.50, 0.30, 0.20], "p_top2": [0.75, 0.60, 0.40], "p_top3": [0.90, 0.80, 0.55],
    })


def _storage():
    return FakeStorage({
        ("race_shutuba", "R1"): {"date": "2026-10-18", "race_name": "テスト", "venue": "東京", "round": 11, "entries": [
            {"horse_number": 1, "horse_id": "h1", "horse_name": "イチ", "weight": 480, "weight_change": 2, "bracket_number": 1},
            {"horse_number": 2, "horse_id": "h2", "horse_name": "ニ", "weight": 0},
            {"horse_number": 3, "horse_id": "h3", "horse_name": "サン", "weight": 500, "weight_change": 0},
        ]},
        ("race_odds", "R1"): {"entries": [
            {"horse_number": 1, "win_odds": 3.0, "place_odds_min": 1.2, "place_odds_max": 1.6, "popularity": 1},
            {"horse_number": 2, "win_odds": 4.0, "place_odds_min": 1.5, "place_odds_max": 2.1, "popularity": 2},
        ]},
    })


def _put(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_shadow_wins_over_backfill_and_missing_returns_none(tmp_path):
    assert LP.find_local_prediction("R1", "20261018", tmp_path) is None
    _put(tmp_path / "predictions_backfill" / "bundle_v4_2025" / "20261018" / "R1.parquet", _probs())
    p, src, label = LP.find_local_prediction("R1", "20261018", tmp_path)
    assert src == "backfill" and label == "bundle_v4_2025"
    _put(tmp_path / "predictions_v3" / "20261018" / "R1.parquet", _probs())
    assert LP.find_local_prediction("R1", "20261018", tmp_path)[1] == "shadow"
    assert LP.find_local_prediction("R1", "", tmp_path) is None and LP.find_local_prediction("R1", "2026", tmp_path) is None


def test_payload_has_note_source_and_ev_from_model_probabilities():
    d = LP.build_payload("R1", _storage().load("race_shutuba", "R1"), _storage().load("race_odds", "R1"), _probs(), "backfill", "bundle_v4_2025")
    assert d["has_prediction"] is True and d["status"] == "success" and d["prediction_source"] == "backfill"
    assert "バックフィル" in d["prediction_note"] and "確定した予測ではありません" in d["prediction_note"]
    assert d["model_type"] == "ensemble_v4" and d["model_version"] == "bundle_v4_2025"
    by = {p["horse_number"]: p for p in d["predictions"]}
    assert by[1]["win_prob"] == 0.5 and by[1]["top3_prob"] == 0.9 and by[1]["pred_rank"] == 1
    assert by[1]["ev_win"] == pytest.approx(0.5 * 3.0)                                  # 勝率 × 単勝オッズ
    assert by[1]["ev_place"] == pytest.approx(0.9 * (1.2 + 1.6) / 2)                    # 複勝率 × 複勝オッズ平均
    assert by[3]["win_odds"] is None and by[3]["ev_win"] is None                        # オッズが無い馬は期待値も作らない
    assert by[2]["weight"] is None and by[1]["weight_change"] == 2
    assert "pred_score" in by[1] and "composite_score" in by[1]                         # race/[id] 画面が予測ありと判定する項目
    assert {p["composite_rank"] for p in d["predictions"]} == {1, 2, 3}
    assert any(p["mark_type"] == "honmei" for p in d["predictions"]) and all("buy_tier" in p for p in d["predictions"])


def test_load_prefers_shadow_label_and_handles_broken_files(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_V3_BUNDLE", "/x/final_v3/bundle_2025")
    _put(tmp_path / "predictions_v3" / "20261018" / "R1.parquet", _probs())
    d = LP.load_local_prediction_payload("R1", _storage(), inf=tmp_path)
    assert d["prediction_source"] == "shadow" and d["model_version"] == "bundle_2025" and d["model_type"] == "ensemble_v3"
    assert "検証用" in d["prediction_note"]
    _put(tmp_path / "predictions_v3" / "20261018" / "R1.parquet", _probs().drop(columns=["p_top3"]))
    assert LP.load_local_prediction_payload("R1", _storage(), inf=tmp_path) is None     # 列が足りない → 従来の応答へ
    assert LP.load_local_prediction_payload("R9", _storage(), inf=tmp_path) is None


def test_route_returns_local_prediction_only_when_gcs_has_none(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.api.app as appmod

    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path))
    st = _storage()
    _put(tmp_path / "predictions_backfill" / "bundle_v4_2025" / "20261018" / "R1.parquet", _probs())
    monkeypatch.setattr(appmod, "_get_storage", lambda: st)
    c = TestClient(appmod.app, raise_server_exceptions=False)
    r = c.get("/api/race/R1/predictions")
    assert r.status_code == 200 and r.json()["prediction_source"] == "backfill" and r.json()["has_prediction"] is True
    st.data[("race_predictions", "R1")] = {"race_id": "R1", "status": "success", "predictions": [{"horse_number": 1}], "_meta": {"scraped_at": 1}}
    r2 = c.get("/api/race/R1/predictions")                                             # GCS の予測があればそちらが優先
    assert "prediction_source" not in r2.json() and r2.json()["predictions"][0]["horse_number"] == 1


def test_eval_backfill_logloss_and_top1_hit():
    from src.scripts.data.eval_backfill_predictions import evaluate

    preds = pd.DataFrame({"race_id": ["R1"] * 3 + ["R2"] * 3, "horse_id": list("abcabc"),
                          "p_win": [0.6, 0.3, 0.1, 0.5, 0.3, 0.2], "p_top2": [0.8, 0.6, 0.3, 0.7, 0.6, 0.4], "p_top3": [0.9, 0.8, 0.5, 0.85, 0.8, 0.55]})
    fin = pd.DataFrame({"race_id": ["R1"] * 3 + ["R2"] * 3, "horse_id": list("abcabc"), "finish_position": [1, 2, 3, 2, 1, 3]})
    r = evaluate(preds, fin, bins=2)
    assert r["races"] == 2 and r["rows"] == 6 and r["top1_hit"] == 0.5
    assert r["win"]["logloss"] < r["win"]["uniform_logloss"] + 0.2 and r["top3"]["rate"] == 1.0


def test_shadow_label_comes_from_the_day_table_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_V3_BUNDLE", "/x/final_v4/bundle_v4_2025")
    _put(tmp_path / "predictions_v3" / "20261018" / "R1.parquet", _probs())
    assert LP.find_local_prediction("R1", "20261018", tmp_path)[2] == "bundle_v4_2025"     # 対応表が無ければ現在の設定
    (tmp_path / "predictions_backfill").mkdir(parents=True, exist_ok=True)
    (tmp_path / "predictions_backfill" / "_shadow_models.json").write_text('{"20261018": "bundle_2025"}', encoding="utf-8")
    assert LP.find_local_prediction("R1", "20261018", tmp_path)[2] == "bundle_2025"
    d = LP.load_local_prediction_payload("R1", _storage(), inf=tmp_path)
    assert d["model_version"] == "bundle_2025" and d["model_type"] == "ensemble_v3"
