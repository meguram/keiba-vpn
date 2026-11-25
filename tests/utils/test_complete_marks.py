"""印を過不足なく（◎○▲★ を各 1 頭、1 頭 1 印。ユーザーの方針 2026-11-25）。src/utils/race_probabilities.complete_marks。"""

from src.utils.race_probabilities import complete_marks, marks_complete


def _e(no, win, top3, mt="none", ev=None, **kw):
    return {"horse_number": no, "win_prob": win, "top3_prob": top3, "mark_type": mt, "ev_win": ev, **kw}


def _marks(rows):
    return {r["horse_number"]: r["mark_type"] for r in rows}


def test_missing_honmei_takes_the_pair_horse_and_pair_is_refilled():
    """◎ が閾値で付かず勝率 1 位が ○ のレース: ◎ は ○ の馬に繰り上げ、○ は次の馬で埋める（◎ が ○ より弱くならない）。"""
    rows = [_e(1, 0.30, 0.6, "pair"), _e(2, 0.20, 0.5, "show_val"), _e(3, 0.15, 0.45, "anchor"), _e(4, 0.10, 0.3), _e(5, 0.05, 0.2, "kesi")]
    out = complete_marks(rows)
    m = _marks(out)
    assert m[1] == "honmei" and m[5] == "kesi" and marks_complete(out)
    assert sorted(v for v in m.values() if v in ("honmei", "pair", "show_val", "star")) == ["honmei", "pair", "show_val", "star"]
    assert rows[0]["mark_type"] == "pair"                      # 入力は書き換えない


def test_exactly_one_each_and_duplicates_are_demoted():
    rows = [_e(1, 0.3, 0.6, "honmei", ev=1.2), _e(2, 0.25, 0.6, "honmei", ev=0.9), _e(3, 0.2, 0.5), _e(4, 0.1, 0.4), _e(5, 0.05, 0.2)]
    out = complete_marks(rows)
    m = _marks(out)
    assert m[1] == "honmei" and m[2] != "honmei" and marks_complete(out)


def test_bet_marks_race_keeps_bet_marks_and_fills_from_unmarked():
    """買い目からの印のレース（keep_assigned）: 買い目の ◎・▲・△ は書き換えず、足りない ○・★ は無印の馬から埋める（△ は無印がいない時だけ）。"""
    rows = [_e(1, 0.3, 0.6, "honmei", recommendation="◎ 買い目の軸"), _e(2, 0.2, 0.5, "show_val"), _e(3, 0.15, 0.4, "anchor", ev=1.4),
            _e(4, 0.08, 0.3, ev=0.8), _e(5, 0.06, 0.25, ev=0.9), _e(6, 0.02, 0.1, "kesi", ev=2.0)]
    out = complete_marks(rows, keep_assigned=True)
    m = _marks(out)
    assert (m[1], m[2], m[3], m[6]) == ("honmei", "show_val", "anchor", "kesi") and marks_complete(out)
    assert {m[4], m[5]} == {"pair", "star"} and out[0]["recommendation"] == "◎ 買い目の軸"


def test_honmei_fill_uses_win_prob_not_ev():
    """◎ を埋めるときは勝率の順（期待値の高い人気薄の ★ を ◎ に上げない）。"""
    rows = [_e(1, 0.35, 0.7, "pair", ev=0.8), _e(2, 0.20, 0.5, "show_val", ev=0.9), _e(3, 0.04, 0.2, "star", ev=1.6, win_odds=40),
            _e(4, 0.10, 0.3, ev=0.7), _e(5, 0.05, 0.2, ev=0.6)]
    out = complete_marks(rows)
    m = _marks(out)
    assert m[1] == "honmei" and m[3] == "star" and marks_complete(out)
    assert "recommendation" not in out[0]


def test_scratched_or_unpredicted_horses_get_no_mark():
    """取消・除外・確率の無い馬には印を付けず、頭数にも数えない（実際の 3 頭立てで ★ を求めない）。"""
    rows = [_e(1, 0.5, 0.9), _e(2, 0.3, 0.8), _e(3, 0.2, 0.7), {"horse_number": 4, "win_prob": None, "mark_type": "none"},
            {"horse_number": None, "win_prob": 0.4, "mark_type": "none"}]
    out = complete_marks(rows)
    assert [r["mark_type"] for r in out[3:]] == ["none", "none"]
    assert sorted(r["mark_type"] for r in out[:3]) == ["honmei", "pair", "show_val"] and marks_complete(out)


def test_skip_race_without_any_mark_gets_all_four():
    """見送り・閾値で何も付かないレースでも ◎○▲★ を付ける（◎・○ は勝率、★ は期待値、▲ は複勝率の順）。"""
    rows = [_e(i, w, t) for i, (w, t) in enumerate([(0.12, 0.35), (0.11, 0.4), (0.10, 0.3), (0.09, 0.28), (0.08, 0.2)], 1)]
    out = complete_marks(rows)
    assert marks_complete(out) and _marks(out)[1] == "honmei"


def test_small_field_marks_from_the_top_and_kesi_only_when_needed():
    rows = [_e(1, 0.5, 0.9, "kesi"), _e(2, 0.3, 0.8, "anchor"), _e(3, 0.2, 0.7)]
    out = complete_marks(rows)
    m = _marks(out)
    assert sorted(m.values()) == ["honmei", "pair", "show_val"] and marks_complete(out)      # 3 頭: ◎○▲ だけ（★ は無し）
    two = complete_marks([_e(1, 0.6, 0.9, "star"), _e(2, 0.4, 0.8)])
    assert sorted(_marks(two).values()) == ["honmei", "pair"] and marks_complete(two)


def test_marks_complete_detects_gaps():
    assert not marks_complete([_e(1, 0.3, 0.6, "pair"), _e(2, 0.2, 0.5, "show_val"), _e(3, 0.1, 0.4, "star"), _e(4, 0.1, 0.3)])
    assert marks_complete([])
