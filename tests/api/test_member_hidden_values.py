"""会員（開発者以外）にも見せない値（ユーザーの判断 2026-11-25）。研究の期間の回収率・サンプル買い目の期待値・Kelly。外部接続なし。"""

from src.api.tier import ROI_NOTICE, strip_bet_ev, strip_research_roi


def test_strip_research_roi_drops_columns_values_and_sentences():
    mod = {"columns": [{"key": "cs_win"}, {"key": "cs_win_roi"}, {"key": "band_place_roi"}],
           "cells": {"h1": {"cs_win": 0.1, "cs_win_roi": 1.4, "band_place_roi": 0.9}},
           "notes": ["位置取り=初角の通過順位。", "検証では回収率が 81.3% でした。条件は 2016〜2024 年で選んだ。", "単勝の払戻率は年で動く。"],
           "rules": [{"label": "x", "B_honmei": {"races": 10, "roi_t": 1.47, "roi_f": 0.9, "rec": [1, 2, 3, 4]}}]}
    out = strip_research_roi({"modules": [mod]})["modules"][0]
    assert [c["key"] for c in out["columns"]] == ["cs_win"] and out["cells"]["h1"] == {"cs_win": 0.1}
    assert out["notes"][0] == "位置取り=初角の通過順位。" and "回収率" not in "".join(out["notes"][:-1]) and out["notes"][-1] == ROI_NOTICE
    assert out["rules"][0]["B_honmei"] == {"races": 10, "rec": [1, 2, 3, 4]}
    assert mod["columns"][1]["key"] == "cs_win_roi"                 # 入力は書き換えない


def test_strip_bet_ev_keeps_bets_and_virtual_results():
    payload = {"view": "paid", "race_ev": 2.4, "portfolio_ev": 1.3, "expected_payout_yen": 5000, "evidence": "検証: 2022〜24年の回収率1.26",
               "skip_reason": "レースの期待値が水準（2.00）に届きません（このレース 1.42）。", "race_tier": {"level": 2, "label": "注目"},
               "bets": [{"bet_type": "wide", "combo": [1, 2], "ev_line_multiple": 3.1, "legs": [{"horse_number": 1, "mark": ""}]}],
               "plans": [{"id": "all", "race_ev": 2.4, "bets": []}],
               "result": {"invested": 300, "returned": 450, "roi": 1.5}}
    out = strip_bet_ev(payload)
    flat = repr(out)
    assert not any(k in flat for k in ("race_ev", "portfolio_ev", "expected_payout_yen", "ev_line_multiple", "evidence", "1.42"))
    assert "期待値" not in out["skip_reason"] and out["skip_reason"]
    assert out["race_tier"]["label"] == "注目" and out["bets"][0]["combo"] == [1, 2]
    assert out["result"] == {"invested": 300, "returned": 450, "roi": 1.5}           # 仮想購入の結果（実際の記録）は残す
