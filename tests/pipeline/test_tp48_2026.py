import json

import numpy as np

from src.scripts.data import tp48_2026 as t


def test_rules_unique_and_hash_stable():
    ids = [r["id"] for r in t.RULES]
    pols = [r["policy"] for r in t.RULES]
    assert len(set(ids)) == len(ids) and len(set(pols)) == len(pols)
    p = t.rules_payload("2026-10-20 13:00:00")
    assert t.rules_hash(p) == t.rules_hash(t.rules_payload("2030-01-01 00:00:00"))      # 固定時刻は規則のハッシュに入れない


def test_rule_names_exist_in_tp48_policies():
    from src.scripts.data import tp48_eval as te
    from src.scripts.data import tp36_single as t36

    ev = np.full((3, 50), np.nan)
    ev[:, :5] = [[2.5, 2.0, 1.5, 1.2, 1.0]] * 3
    py = np.zeros((3, 50))
    store: dict = {}
    for bt in ("fukusho", "wide", "umaren"):
        for sch in ("model", "mixed"):
            te.policy_rows(f"{bt}|{sch}", ev, py, np.array([2024, 2025, 2026]), np.ones(3, bool), bt, store, [], [])
    assert t36.KMAX == 50
    missing = [r["policy"] for r in t.RULES if r["policy"] not in store]
    assert not missing


def test_evaluate_policy_roi_and_without_top3():
    stake = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 0.0])
    ret = np.array([0.0, 0.0, 500.0, 300.0, 200.0, 0.0])
    r = t.evaluate_policy(stake, ret, np.ones(6, bool), n_boot=50)
    assert r["bets"] == 5 and abs(r["roi"] - 2.0) < 1e-9 and abs(r["hit_race_rate"] - 0.6) < 1e-9
    assert abs(r["roi_without_top3"] - 0.0) < 1e-9          # 上位 3 的中を除くと残りは 0 円
    none = t.evaluate_policy(stake, ret, np.zeros(6, bool))
    assert none["roi"] is None and none["bets"] == 0


def test_section_is_idempotent(tmp_path, monkeypatch):
    res = {"hash": "abc", "registered_at": "2026-10-20 13:00:00", "races": {"2024": 10, "2025": 10, "2026": 10},
           "rules": [{"id": "F1", "policy": "fukusho|model|ev>=2.0|top5", "note": "n", **{k: {"roi": 0.9, "lo": 0.8, "hi": 1.0, "bets": 50, "races_bet": 40, "hit_race_rate": 0.3, "roi_without_top3": 0.7}
                                                                                    for k in ("y2024", "y2025", "design", "y2026")}}],
           "reality_check_2026": {"t_max": 1.0, "p_value": 0.5, "n_strategies": 1, "best": "F1"}}
    ch = tmp_path / "c.html"
    ch.write_text("<html><body><p>x</p></body></html>", encoding="utf-8")
    monkeypatch.setattr(t, "CHAPTER", ch)
    j = tmp_path / "e.json"
    j.write_text(json.dumps(res), encoding="utf-8")

    class A:
        json = str(j)

    t.cmd_report(A)
    t.cmd_report(A)
    txt = ch.read_text(encoding="utf-8")
    assert txt.count(t.MARK_BEGIN) == 1 and "区間下限が 100% を超えた規則は無かった" in txt
