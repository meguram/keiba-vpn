"""公開許可リスト（config/public_fields.json）: 公開モードで B/C/O 分類が一般（匿名・会員）に漏れないこと。外部接続なし。"""

from __future__ import annotations

import itertools
import json

import pytest

from src.api import tier
from src.utils import public_policy as P
from tests.api.test_tiered_access import TOP

DETAIL = {
    "race_id": "202608020311", "race_name": "R", "venue": "京都", "round": 11, "date": "20261024", "start_time": "15:40",
    "surface": "芝", "distance": 2000, "grade": "G1", "track_condition": "良", "direction": "右", "weather": "晴",
    "race_shutuba": {"race_id": "x", "entries": [{"horse_number": 1, "horse_name": "A", "bracket_number": 1, "sex_age": "牡4", "jockey_name": "J",
                                                  "jockey_weight": 56.0, "trainer_name": "T", "weight": 480, "weight_change": 2,
                                                  "odds": 3.0, "popularity": 1, "horse_id": "h", "jockey_id": "j", "trainer_id": "t", "sire": "s"}]},
    "race_result": {"race_id": "x", "payoff": {"tansho": [[1, 300]]}, "entries": [{"horse_number": 1, "horse_name": "A", "finish_position": 1,
                                                                                 "odds": 3.0, "popularity": 1, "horse_id": "h", "time_sec": 120.1}]},
    "race_odds": {"entries": []}, "race_pair_odds": {}, "race_index": {}, "race_paddock": {}, "race_barometer": {},
    "race_trainer_comment": {}, "smartrc_race": {}, "race_shutuba_past": {}, "race_oikiri": {}, "horses": {"h": {"name": "A"}},
    "unknown_new_key": {"x": 1}, "data_availability": {"race_odds": True},
}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("KEIBA_PUBLIC_MODE", raising=False)
    P.reload()
    yield
    P.reload()


def _set(monkeypatch, tmp_path, mode="public", **release):
    cfg = json.loads(P.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["mode"] = mode
    cfg["release"].update(minimal_identifiers_public=False, b_facts_public=False, odds_derived_public=False, actual_odds_public=False)      # 実ファイルの公開許可に依存させない
    cfg["release"].update(release)
    f = tmp_path / "pf.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(P, "CONFIG_PATH", f)
    P.reload()
    return cfg


def _keys(out):
    return set(out) | {k for e in out.get("predictions", []) for k in e}


def test_private_mode_is_unchanged():
    assert P.is_public_mode() is False
    assert tier.redact_predictions_member(TOP) is TOP and tier.redact_race_detail_member(DETAIL) is DETAIL
    assert "win_prob" in _keys(tier.redact_predictions(TOP))


def test_public_mode_default_only_class_a(monkeypatch, tmp_path):
    cfg = _set(monkeypatch, tmp_path)
    p = cfg["predictions"]
    for role, fn in (("anonymous", tier.redact_predictions), ("member", tier.redact_predictions_member)):
        out = fn(TOP)
        keys = _keys(out)
        banned = set(p["entry_O"]) | set(p["entry_B"]) | set(p["entry_C"]) | set(p["top_O"]) | set(p["top_identifiers"])
        assert not (banned & keys), (role, banned & keys)
        assert out["access"]["public_mode"] is True and "B" in out["access"]["withheld"] and "O" in out["access"]["withheld"]
        assert {"win_prob", "top3_prob", "show_prob"} <= keys
    assert "mark_type" in _keys(tier.redact_predictions_member(TOP)) and "mark_type" not in _keys(tier.redact_predictions(TOP))


def test_odds_cannot_be_recovered_from_public_fields(monkeypatch, tmp_path):
    """期待値と勝率が両方出るとオッズを逆算できる。オッズ許可が無い限り、期待値・買い判断は会員にも出さない。"""
    _set(monkeypatch, tmp_path)
    keys = _keys(tier.redact_predictions_member(TOP))
    assert not ({"ev_win", "ev_place", "expected_value", "buy_tier", "win_odds", "popularity"} & keys)


def test_release_switches(monkeypatch, tmp_path):
    _set(monkeypatch, tmp_path, minimal_identifiers_public=True)
    k = _keys(tier.redact_predictions(TOP))
    assert {"horse_number", "horse_name"} <= k and "jockey_name" not in k and "race_name" in k
    _set(monkeypatch, tmp_path, b_facts_public=True)
    k = _keys(tier.redact_predictions(TOP))
    assert {"jockey_name", "weight", "sex_age"} <= k and "horse_id" not in k
    _set(monkeypatch, tmp_path, odds_derived_public=True)
    m, g = _keys(tier.redact_predictions_member(TOP)), _keys(tier.redact_predictions(TOP))
    assert "ev_win_cal" in m and "ev_win_cal" not in g                       # 較正後の期待値は会員に出る
    assert not ({"ev_win", "ev_place", "ev_top2", "expected_value", "win_odds", "place_odds_min", "place_odds_max", "popularity",
                 "top2_odds_est", "odds_source", "bet_suggestion"} & m)      # 実オッズ（と、確率から逆算できる生の期待値）は出さない
    _set(monkeypatch, tmp_path, odds_derived_public=True, actual_odds_public=True)
    assert {"ev_win", "win_odds", "popularity", "bet_suggestion"} <= _keys(tier.redact_predictions_member(TOP))     # 許可が取れたら出せる
    assert "win_odds" not in _keys(tier.redact_predictions(TOP))                                                     # ゲストには常に出ない


def test_class_c_never_leaks_under_any_release_combination(monkeypatch, tmp_path):
    cfg = _set(monkeypatch, tmp_path)
    c_entry, c_keys = set(cfg["predictions"]["entry_C"]), set(cfg["race_detail"]["C_keys"])
    for ident, b, o in itertools.product((False, True), repeat=3):
        _set(monkeypatch, tmp_path, minimal_identifiers_public=ident, b_facts_public=b, odds_derived_public=o)
        for fn in (tier.redact_predictions, tier.redact_predictions_member):
            assert not (c_entry & _keys(fn(TOP)))
        for fn in (tier.redact_race_detail, tier.redact_race_detail_member):
            out = fn(DETAIL)
            assert not (c_keys & set(out)) and "unknown_new_key" not in out and "race_odds" not in out
            ids = {k for part in ("race_shutuba", "race_result") for e in (out.get(part) or {}).get("entries", []) for k in e}
            assert not ({"horse_id", "jockey_id", "trainer_id", "sire", "odds", "popularity"} & ids)


def test_race_detail_public_mode_levels(monkeypatch, tmp_path):
    _set(monkeypatch, tmp_path)
    out = tier.redact_race_detail(DETAIL)
    assert set(out) == {"race_id", "access"}                      # 既定: 識別のレース ID だけ
    _set(monkeypatch, tmp_path, minimal_identifiers_public=True)
    assert "race_name" in tier.redact_race_detail(DETAIL) and "race_shutuba" not in tier.redact_race_detail(DETAIL)
    _set(monkeypatch, tmp_path, b_facts_public=True)
    out = tier.redact_race_detail(DETAIL)
    assert out["race_shutuba"]["entries"][0]["horse_name"] == "A" and "payoff" not in out["race_result"]
    assert "horses" not in out                                    # 馬の詳細（取得元固有の ID つき）は出さない
    _set(monkeypatch, tmp_path, b_facts_public=True, odds_derived_public=True)
    assert "payoff" not in tier.redact_race_detail(DETAIL)["race_result"]     # 確定払戻は確定オッズそのものなので、実オッズの許可まで出さない
    _set(monkeypatch, tmp_path, b_facts_public=True, odds_derived_public=True, actual_odds_public=True)
    assert "payoff" in tier.redact_race_detail(DETAIL)["race_result"]


def test_owner_only_paths(monkeypatch, tmp_path):
    assert P.owner_only_path("/api/race/1/entry/2/card") is False  # private
    _set(monkeypatch, tmp_path)
    assert P.owner_only_path("/api/race/1/entry/2/card") and P.owner_only_path("/api/horse/h/detail")
    assert P.owner_only_path("/api/v1/betting/simulate") and P.owner_only_path("/api/prediction-results")
    # 有料ゾーンの推奨馬券（推奨度の段階だけの出力）は会員に出す。有料ゾーンを無効にすれば開発者専用に戻る
    assert P.owner_only_path("/api/v1/betting/optimize") is False and P.owner_only_path("/api/v1/betting/race-values") is False
    monkeypatch.setenv("KEIBA_PAID_ZONE", "0")
    assert P.owner_only_path("/api/v1/betting/optimize") is True
    monkeypatch.delenv("KEIBA_PAID_ZONE")
    assert P.owner_only_path("/api/race/1/predictions") is False   # 落として返す
    _set(monkeypatch, tmp_path, odds_derived_public=True)
    assert P.owner_only_path("/api/v1/betting/optimize") is False and P.owner_only_path("/api/horse/h/detail")


def test_unreadable_config_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "CONFIG_PATH", tmp_path / "missing.json")
    P.reload()
    assert P.is_public_mode() is True and P.entry_allowed_fields("member") <= {"win_prob", "top3_prob", "show_prob", "pred_rank", "top2_prob", "place_prob"} | set()


def test_env_overrides_mode(monkeypatch):
    monkeypatch.setenv("KEIBA_PUBLIC_MODE", "public")
    assert P.is_public_mode() is True
    monkeypatch.setenv("KEIBA_PUBLIC_MODE", "private")
    assert P.is_public_mode() is False


def test_fastapi_gate_blocks_owner_paths_in_public_mode(monkeypatch, tmp_path):
    import os

    os.environ.setdefault("GCS_BUCKET", "")
    os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")
    from fastapi.testclient import TestClient

    from src.api.app import app

    _set(monkeypatch, tmp_path)
    c = TestClient(app)
    r = c.get("/api/race/202608020311/entry/2020100001/card")
    assert r.status_code == 403 and r.json()["reason"] == "not_public_yet"
