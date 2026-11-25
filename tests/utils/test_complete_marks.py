"""画面の印の最後の段（src/utils/race_probabilities.complete_marks）。rules-council R0-10・依頼 2 の追記（2026-11-25）:
◎○▲ は AI の勝率の 1・2・3 位、★△消は印のルールのまま、1 頭 1 印、◎○▲★ を各 1 頭（頭数が 4 未満なら上から）。"""

from src.utils.race_probabilities import complete_marks, marks_complete


def _e(no, win, top3, mt="none", ev=None, **kw):
    return {"horse_number": no, "win_prob": win, "top3_prob": top3, "mark_type": mt, "ev_win": ev, **kw}


def _marks(rows):
    return {r["horse_number"]: r["mark_type"] for r in rows}


def test_top3_by_win_prob_overrides_rule_honmei():
    """印のルールの ◎（勝率の閾値を超えた中で期待値が最大）は使わない。◎○▲ は勝率の 1・2・3 位。"""
    rows = [_e(1, 0.30, 0.6, "pair"), _e(2, 0.20, 0.5), _e(3, 0.12, 0.45, "honmei", ev=1.6), _e(4, 0.15, 0.4, "show_val"),
            _e(5, 0.08, 0.3, "anchor"), _e(6, 0.05, 0.2, "kesi")]
    out = complete_marks(rows)
    m = _marks(out)
    assert (m[1], m[2], m[4]) == ("honmei", "pair", "show_val")
    assert m[3] == "star" and m[5] == "anchor" and m[6] == "kesi" and marks_complete(out)   # 元の ◎ は無印に戻り、★ の埋めで期待値の順に選ばれる
    assert rows[0]["mark_type"] == "pair"                                                   # 入力は書き換えない


def test_rule_star_kept_unless_in_top3():
    rows = [_e(1, 0.3, 0.6), _e(2, 0.2, 0.5), _e(3, 0.15, 0.4), _e(4, 0.08, 0.3, "star", ev=1.4), _e(5, 0.06, 0.2, ev=2.0)]
    assert _marks(complete_marks(rows))[4] == "star"
    rows2 = [_e(1, 0.3, 0.6, "star", ev=1.2), _e(2, 0.2, 0.5), _e(3, 0.15, 0.4), _e(4, 0.08, 0.3, ev=1.5), _e(5, 0.06, 0.2, ev=0.8)]
    m = _marks(complete_marks(rows2))
    assert m[1] == "honmei" and m[4] == "star"                                              # 上位 3 頭の ★ は ◎ になり、★ は埋め直す


def test_every_race_gets_all_four_once():
    rows = [_e(i, w, t) for i, (w, t) in enumerate([(0.12, 0.35), (0.11, 0.4), (0.10, 0.3), (0.09, 0.28), (0.08, 0.2)], 1)]
    out = complete_marks(rows)
    assert marks_complete(out) and _marks(out)[1] == "honmei"
    two_stars = complete_marks([_e(1, 0.3, 0.6), _e(2, 0.2, 0.5), _e(3, 0.1, 0.4), _e(4, 0.08, 0.3, "star", ev=1.1), _e(5, 0.05, 0.2, "star", ev=1.5)])
    assert _marks(two_stars)[5] == "star" and _marks(two_stars)[4] == "anchor" and marks_complete(two_stars)


def test_scratched_or_unpredicted_horses_get_no_mark():
    """取消・除外・確率の無い馬には印を付けず、頭数にも数えない（実際の 3 頭立てで ★ を求めない）。"""
    rows = [_e(1, 0.5, 0.9), _e(2, 0.3, 0.8), _e(3, 0.2, 0.7), {"horse_number": 4, "win_prob": None, "mark_type": "none"},
            {"horse_number": None, "win_prob": 0.4, "mark_type": "none"}]
    out = complete_marks(rows)
    assert [r["mark_type"] for r in out[3:]] == ["none", "none"]
    assert sorted(r["mark_type"] for r in out[:3]) == ["honmei", "pair", "show_val"] and marks_complete(out)


def test_small_field_and_kesi_in_top3():
    rows = [_e(1, 0.5, 0.9, "kesi", kesi_reason="x"), _e(2, 0.3, 0.8, "anchor"), _e(3, 0.2, 0.7)]
    out = complete_marks(rows)
    assert _marks(out) == {1: "honmei", 2: "pair", 3: "show_val"} and "kesi_reason" not in out[0] and marks_complete(out)
    two = complete_marks([_e(1, 0.6, 0.9, "star"), _e(2, 0.4, 0.8)])
    assert _marks(two) == {1: "honmei", 2: "pair"} and marks_complete(two)


def test_changed_marks_drop_old_recommendation():
    rows = [_e(1, 0.35, 0.7, "pair", recommendation="○ 買い目の2番手"), _e(2, 0.2, 0.5, "honmei", recommendation="◎"), _e(3, 0.1, 0.3), _e(4, 0.05, 0.2)]
    out = complete_marks(rows)
    assert "recommendation" not in out[0] and "recommendation" not in out[1]


def test_marks_complete_detects_gaps():
    assert not marks_complete([_e(1, 0.3, 0.6, "pair"), _e(2, 0.2, 0.5, "show_val"), _e(3, 0.1, 0.4, "star"), _e(4, 0.1, 0.3)])
    assert marks_complete([])
