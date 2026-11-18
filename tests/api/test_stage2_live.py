"""候補の影のランナーの出力を API で使う stage2_live: ファイルがあれば新しい表示・無ければ今の表示（target・v6e）、止めの目印、開始日、
target と同じ形の応答（見送りの隠し方・推奨度・印）、予測の表示だけ候補にする（推奨・成績の記録は v6e）。ミラーの木。外部接続なし。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.api import local_predictions as LP
from src.api import stage2_live as S2
from src.api.v1 import delegates
from src.ops import gcs_mirror as gm

RID, DAY = "202605050101", "20261121"
CARD = {"race_id": RID, "date": "2026-11-21", "grade": "未勝利", "entries": [{"horse_number": 3, "horse_name": "テスト馬"}, {"horse_number": 5, "horse_name": "相手"}]}


class _Storage:
    def load(self, category, key, **_):
        return CARD if category == "race_shutuba" else None


def _decision(status="buy"):
    d = {"strategy": "stage2_r2", "race_id": RID, "frozen_hash": "70ba29fa7b55", "status": status, "field": 12, "c_class": "未勝利",
         "r_max": 0.8, "q_r_max": 4, "as_of": "2026-11-21T09:55:00+09:00", "odds_t60_at": "2026-11-21T09:00:00+09:00",
         "horses": [{"horse_number": 3, "p_win": 0.12, "q": 0.05, "r": 0.87, "pop": 6, "odds": 15.8}]}
    if status == "buy":
        d.update({"skip_reason": None, "honmei": d["horses"][0], "bets": [{"bet_type": "tansho", "combo": [3], "yen": 100}],
                  "horse_marks": {"3": "◎", "5": "○"}})
    else:
        d.update({"skip_reason": "r_max", "skip_label": "見送り（市場とのずれが小さい）", "bets": [], "horse_marks": {}})
    return d


@pytest.fixture
def inf(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path))
    monkeypatch.delenv("KEIBA_STAGE2_START", raising=False)
    monkeypatch.setattr("src.scraper.storage.HybridStorage", lambda *a, **k: _Storage())
    from src.pipeline.inference import target_bets as TB

    monkeypatch.delenv("KEIBA_BET_STRATEGY", raising=False)                      # tests/conftest.py が legacy に固定しているのを、本番の既定（target）に戻す
    monkeypatch.setattr(TB, "available", lambda: TB.strategy() != "legacy")     # 係数のファイルが無くても target の経路を通す。legacy に戻すと第 2 段も出ない
    return tmp_path


def _put(inf, d, day=DAY):
    p = inf / "bets_stage2" / day / f"{RID}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def test_no_file_means_current_path(inf):
    assert delegates._optimize_stage2({"race_id": RID}, "full") is None


def test_buy_is_target_shaped(inf):
    _put(inf, _decision("buy"))
    out, code = delegates.optimize_betting({"race_id": RID}, view="full")
    assert code == 200 and out["strategy"] == "target" and out["sub_strategy"] == "stage2_r2" and out["trial"] is True
    assert out["status"] == "bet" and out["level"] == 1 and out["race_tier"]["level"] == 1
    b = out["bets"][0]
    assert b["combo"] == [3] and b["yen"] == 100 and b["legs"][0]["horse_name"] == "テスト馬" and b["est_multiple"] == 1.0
    assert out["horse_marks"] == {"3": "honmei"}
    assert out["stage2"]["frozen_hash"] == "70ba29fa7b55" and "試行中" in out["note"]
    assert not delegates.is_skipped_payload(out)


def test_skip_keeps_reason_and_hides_bets(inf):
    _put(inf, _decision("skip"))
    out, code = delegates.optimize_betting({"race_id": RID}, view="paid")
    assert code == 200 and out["skipped"] is True and out["bets"] == [] and out["level"] == 0
    assert out["stage2"]["skip_label"] == "見送り（市場とのずれが小さい）" and "horse_marks" not in out


def test_disabled_mark_goes_back(inf):
    _put(inf, _decision("buy"))
    (inf / "bets_stage2" / "_disabled.json").write_text("{}", encoding="utf-8")
    assert delegates._optimize_stage2({"race_id": RID}, "full") is None


def test_days_before_start_are_ignored(inf, monkeypatch):
    _put(inf, _decision("buy"))
    monkeypatch.setenv("KEIBA_STAGE2_START", "20261122")
    assert delegates._optimize_stage2({"race_id": RID}, "full") is None


def test_broken_file_falls_back(inf):
    p = inf / "bets_stage2" / DAY / f"{RID}.json"
    p.parent.mkdir(parents=True)
    p.write_text("{broken", encoding="utf-8")
    assert delegates._optimize_stage2({"race_id": RID}, "full") is None


def _pred(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"race_id": [RID], "horse_id": ["h"], "p_win": [1.0], "p_top2": [1.0], "p_top3": [1.0]}).to_parquet(path)


def test_candidate_only_for_display(inf):
    _pred(inf / "predictions_v6" / DAY / f"{RID}.parquet")
    _pred(inf / "predictions_v6c" / DAY / f"{RID}.parquet")
    assert LP.find_primary_prediction(RID, DAY, inf)[1] == "candidate"
    assert LP.find_primary_prediction(RID, DAY, inf, candidate=False)[1] == "shadow"     # 推奨（target）と成績の記録は v6e
    (inf / "predictions_v6c" / "_disabled.json").write_text("{}", encoding="utf-8")
    assert LP.find_primary_prediction(RID, DAY, inf)[1] == "shadow"


def test_mirror_has_trees_and_windows():
    for t in ("predictions_v6c", "predictions_stage2", "bets_stage2"):
        assert f"data/local/inference/{t}" in gm.TREES and f"data/local/inference/{t}" in gm.WINDOWED_DIRS


def test_push_notes_stage2_files_locally(tmp_path):
    gm._note_stage2_push(tmp_path, ["data/local/inference/bets_stage2/20261121/R1.json", "data/local/inference/predictions_v6/20261121/R1.parquet",
                                    "data/local/inference/bets_stage2/_disabled.json"])
    rows = [json.loads(x) for x in (tmp_path / "data/local/inference/launch_watch/push_20261121.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "pushed" and rows[0]["race_ids"] == ["R1"]
    gm._note_stage2_push(tmp_path, ["data/local/inference/predictions_v6/20261121/R2.parquet"])     # 判定が無い push は記録しない
    assert len((tmp_path / "data/local/inference/launch_watch/push_20261121.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_disabled_mark_off_means_enabled(inf):
    _put(inf, _decision("buy"))
    (inf / "bets_stage2" / "_disabled.json").write_text('{"disabled": false}', encoding="utf-8")
    assert delegates._optimize_stage2({"race_id": RID}, "full") is not None


def test_members_do_not_get_market_odds(inf):
    _put(inf, _decision("buy"))
    out, _ = delegates.optimize_betting({"race_id": RID}, view="paid")
    s = out["stage2"]
    assert "horses" not in s and set(s["honmei"]) == {"horse_number"} and "odds" not in out["bets"][0]
    assert out["bets"][0]["hit_rate_pct"] is None and out["bets"][0]["ev_line_multiple"] is None
    full, _ = delegates.optimize_betting({"race_id": RID}, view="full")
    assert full["stage2"]["horses"] and full["stage2"]["honmei"]["odds"] == 15.8 and full["bets"][0]["odds"] == 15.8


def test_legacy_strategy_ignores_stage2(inf, monkeypatch):
    _put(inf, _decision("buy"))
    monkeypatch.setenv("KEIBA_BET_STRATEGY", "legacy")
    assert delegates._optimize_betting.__name__ and not delegates.optimize_betting({"race_id": RID}, view="full")[0].get("sub_strategy")


@pytest.mark.parametrize("grade", ["1勝", "G1", "OP", ""])
def test_other_classes_keep_current_target(inf, monkeypatch, grade):
    """未勝利・新馬以外のレースは、判定ファイルがあっても今の target（推奨度・印も）のまま。"""
    _put(inf, {**_decision("skip"), "skip_reason": "class", "c_class": "1勝"})
    monkeypatch.setitem(CARD, "grade", grade)
    assert delegates._optimize_stage2({"race_id": RID}, "full") is None


def test_newcomer_and_missing_material_still_shown(inf, monkeypatch):
    monkeypatch.setitem(CARD, "grade", "新馬")
    _put(inf, {"strategy": "stage2_r2", "race_id": RID, "frozen_hash": "70ba29fa7b55", "status": "skip", "skip_reason": "no_odds_t60",
               "skip_label": "見送り（T-60 オッズ未着）", "bets": [], "horse_marks": {}})
    out, _ = delegates.optimize_betting({"race_id": RID}, view="paid")
    assert out["sub_strategy"] == "stage2_r2" and out["skipped"] and out["stage2"]["skip_label"] == "見送り（T-60 オッズ未着）"



def test_dev_only_mark_hides_from_members(inf):
    _put(inf, _decision("buy"))
    (inf / "bets_stage2" / "_dev_only.json").write_text('{"disabled": true}', encoding="utf-8")
    assert delegates._optimize_stage2({"race_id": RID}, "paid") is None          # 会員には今の target
    assert delegates._optimize_stage2({"race_id": RID}, "full") is not None      # 開発者には第 2 段
