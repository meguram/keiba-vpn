"""印 v2（ユーザー定義）とディスパッチャ。既定は v1 のまま・内部名は互換。"""

import json

import pytest

from src.utils import race_probabilities as RP


def _field(n=12, seed=3):
    import random

    random.seed(seed)
    raw = [random.random() + 0.05 for _ in range(n)]
    tot = sum(raw)
    es = []
    for i, w in enumerate(raw):
        pw = w / tot
        es.append({"horse_number": i + 1, "win_prob": pw, "top2_prob": min(pw * 2, 0.95), "top3_prob": min(pw * 3, 0.97),
                   "win_odds": round(0.8 / pw, 1), "ev_win": 0.8 + (i % 4) * 0.1, "ev_place": 0.85 + (i % 3) * 0.1, "composite_score": pw})
    return es


def _marks(es):
    return {e["horse_number"]: e["mark_type"] for e in es}


def test_default_is_v1(monkeypatch):
    monkeypatch.delenv("KEIBA_MARK_RULE", raising=False)
    monkeypatch.setenv("KEIBA_MARK_RULE_CONFIG", "/nonexistent.json")
    a, b = _field(), _field()
    RP.assign_mece_marks(a)
    RP.assign_mece_marks_v1(b)
    assert _marks(a) == _marks(b)


def test_env_switches_to_v2(monkeypatch):
    monkeypatch.setenv("KEIBA_MARK_RULE", "v2")
    a, b = _field(), _field()
    RP.assign_mece_marks(a)
    RP.assign_mece_marks_v2(b, RP.mark_rule_config().get("v2"))
    assert _marks(a) == _marks(b)


def test_config_file_version(monkeypatch, tmp_path):
    monkeypatch.delenv("KEIBA_MARK_RULE", raising=False)
    p = tmp_path / "mr.json"
    p.write_text(json.dumps({"version": 2, "v2": {"anchor": {"limits": [[0, 2]]}}}))
    monkeypatch.setenv("KEIBA_MARK_RULE_CONFIG", str(p))
    assert RP.mark_rule_version() == 2
    es = _field(14)
    RP.assign_mece_marks(es)
    assert sum(1 for e in es if e["mark_type"] == "anchor") == 2


def test_v2_structure_and_names():
    es = _field()
    RP.assign_mece_marks_v2(es)
    counts = {}
    for e in es:
        counts[e["mark_type"]] = counts.get(e["mark_type"], 0) + 1
    assert set(counts) <= {"honmei", "pair", "show_val", "star", "anchor", "none"}
    assert counts["honmei"] == 1 and counts.get("pair", 0) == 1 and counts.get("show_val", 0) <= 1 and counts.get("star", 0) <= 1


def test_v2_honmei_is_max_ev_among_win_qualified():
    es = _field()
    RP.assign_mece_marks_v2(es, {"honmei": {"win_mult": 1.0, "topk": 3}})
    top3 = sorted(es, key=lambda e: -e["win_prob"])[:3]
    qual = [e for e in top3 if e["win_prob"] >= 1 / len(es)]
    want = max(qual or top3[:1], key=lambda e: (e["ev_win"], e["win_prob"]))
    assert [e for e in es if e["mark_type"] == "honmei"][0] is want


def test_v2_star_has_no_odds_ceiling_but_needs_place_floor(monkeypatch):
    monkeypatch.setattr(RP, "star_v1_config", lambda: {})          # 従来の判定（生の期待値・上限なし）の確認
    es = _field(12)
    for e in es:
        e["mark_type"] = "none"
    longshot = es[0]
    longshot.update({"win_odds": 80.0, "ev_win": 3.0, "top3_prob": 0.30})
    es[1].update({"win_odds": 60.0, "ev_win": 5.0, "top3_prob": 0.01})  # EV は高いが複勝率が下限未満
    RP.assign_mece_marks_v2(es, {"star": {"odds_min": 10.0, "odds_max": None, "top3_floor_mult": 1.0, "ev": "win"}})
    assert longshot["mark_type"] == "star" and es[1]["mark_type"] != "star"


def test_v2_anchor_count_by_field_size():
    for n, want in ((14, 3), (10, 3), (6, 2), (4, 1)):
        es = _field(n)
        RP.assign_mece_marks_v2(es)
        assert sum(1 for e in es if e["mark_type"] == "anchor") <= want


def test_v2_one_mark_per_horse_and_empty():
    RP.assign_mece_marks_v2([])
    es = _field(9)
    RP.assign_mece_marks_v2(es)
    assert all(isinstance(e["mark_type"], str) for e in es)


@pytest.mark.parametrize("rule", ["win", "top2", "blend"])
def test_v2_pair_rules_pick_other_than_honmei(rule):
    es = _field()
    RP.assign_mece_marks_v2(es, {"pair": {"rule": rule}})
    h = [e for e in es if e["mark_type"] == "honmei"][0]
    p = [e for e in es if e["mark_type"] == "pair"][0]
    assert h is not p
