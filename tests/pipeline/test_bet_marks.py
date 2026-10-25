"""推奨馬券を正として、印を買い目から決め直す（bet_marks）。"""

from src.pipeline.inference import bet_marks as BM


def _ents():
    return [{"horse_number": n, "win_prob": p, "mark_type": m, "recommendation": "x"} for n, p, m in
            [(1, 0.30, "honmei"), (2, 0.20, "pair"), (3, 0.15, "show_val"), (4, 0.10, "star"), (5, 0.08, "anchor"), (6, 0.02, "kesi"), (7, 0.15, "none")]]


def test_stake_weights_and_ordered_bets():
    w = BM.horse_stakes([{"bet_type": "umaren", "combo": [3, 5], "yen": 600}, {"bet_type": "umatan", "combo": [7, 3], "yen": 300}])
    assert w == {3: 300 + 100, 5: 300, 7: 200}


def test_marks_follow_stake_order_and_ties_break_by_win_prob():
    bets = [{"bet_type": "sanrenpuku", "combo": [3, 5, 7], "yen": 900}, {"bet_type": "wide", "combo": [5, 7], "yen": 200}]
    marks = BM.aligned_mark_types(_ents(), bets)
    assert marks == {7: "honmei", 5: "pair", 3: "show_val"}          # 5 と 7 は購入額 400 で同じ → 勝率の高い 7 が先。3 は 300


def test_more_than_three_horses_get_anchor():
    bets = [{"bet_type": "sanrenpuku", "combo": [1, 2, 3], "yen": 300}, {"bet_type": "sanrenpuku", "combo": [1, 4, 5], "yen": 200}]
    marks = BM.aligned_mark_types(_ents(), bets)
    assert marks[1] == "honmei" and marks[4] == marks[5] == "anchor" and set(marks) == {1, 2, 3, 4, 5}


def test_no_bets_keeps_original_marks():
    assert BM.aligned_mark_types(_ents(), []) is None
    assert BM.aligned_mark_types(_ents(), None) is None


def test_apply_drops_unbought_marks_but_keeps_kesi():
    out = BM.apply_to_entries(_ents(), {3: "honmei", 7: "pair"})
    by = {e["horse_number"]: e for e in out}
    assert by[3]["mark_type"] == "honmei" and by[3]["recommendation"].startswith("◎")
    assert by[1]["mark_type"] == "none" and by[1]["recommendation"] == ""         # 従来の◎でも買い目に無ければ印なし
    assert by[4]["mark_type"] == "none"                                           # ★も同様
    assert by[6]["mark_type"] == "kesi" and by[6]["recommendation"] == "x"        # 消印は残す


def test_relabel_legs_updates_all_plans():
    pf = {"bets": [{"legs": [{"horse_number": 3, "mark": "▲"}, {"horse_number": 9, "mark": "△"}]}],
          "plans": [{"bets": [{"legs": [{"horse_number": 3, "mark": "▲"}]}]}]}
    BM.relabel_legs(pf, {3: "honmei"})
    assert pf["bets"][0]["legs"][0]["mark"] == "◎" and pf["bets"][0]["legs"][1]["mark"] == ""
    assert pf["plans"][0]["bets"][0]["legs"][0]["mark"] == "◎"
