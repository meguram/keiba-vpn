"""事前登録の閾値のルール（src/research/course/prereg_rules.py）。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.research.course import prereg_rules as PR


def _toy():
    rows = []
    for r in range(4):
        for h in range(1, 5):
            rows.append({"race_id": f"R{r}", "horse_id": f"H{h}", "p_race_id": f"R{r - 1}" if r else None, "fin": h, "jw": 55 + (h == 1),
                         "wt": 460 + 10 * h, "jockey_id": f"J{h}" if r % 2 == 0 else f"J{h + (h == 2)}", "grade": "G3", "p_grade": "G3",
                         "p_fin": h, "p_margin": 0.0 if h == 1 else 0.1 * h, "p_jw": 55, "p_l3rank": h * 3, "p_pos1": h, "p_pos4": h,
                         "age": 3 + h, "interval": 14 * h, "dist": 1600, "p_dist": 1400 + 100 * (h % 3), "hn": h, "bracket_number": h})
    return pd.DataFrame(rows)


def test_prev_lookup_and_masks():
    d = _toy()
    pv = PR.prev_lookup(d)
    assert np.isnan(pv["prev_wt"].iloc[0]) and pv["prev_wt"].iloc[4] == 470                  # 前走の行（R0・H1）の体重
    assert pv["prev_win_jw"].iloc[4] == 56                                                  # 前走の勝ち馬の斤量
    M = PR.rule_masks(d, pv)
    app, flag = M["U20"]
    assert not app[0] and app[4] and not flag[4]                                            # 前走を引けない馬は判定の外・470kg は ⭕
    app, flag = M["G2a"]
    assert app[5] and flag[5] and not flag[4]                                                # R1 の H2 は乗り替わり、H1 は同じ騎手
    app, flag = M["U15"]
    assert flag[0] and not flag[2]                                                           # 4 歳は ⭕ でない・6 歳は ⭕


def test_mh_simple():
    st = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    flag = np.array([1, 1, 0, 0, 1, 1, 0, 0], bool)
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1], float)
    assert abs(PR.mh(st, flag, y) - (-0.75)) < 1e-9                                         # 層ごとの差 −1・−0.5 を等しい重みで


def test_judge_controls_and_checks():
    C = {"q_max": 0.05, "effect_min_pt": {"全馬": 1.0}, "breadth": {"share": 0.75}}
    good = {"id": "U01", "side": "graded", "layer": "全馬", "control": False, "rd": -0.05, "se": 0.01,
            "periods": {"2017〜21": -0.05, "2022〜24": -0.04, "2020〜24": -0.04}, "rd2024_lo": -0.1, "breadth": {}}
    ctl = {"id": "N1", "side": "graded", "layer": "全馬", "control": True, "rd": 0.0, "se": 0.01, "periods": {}, "breadth": {}}
    r = PR.judge([dict(good), dict(ctl)], C)
    assert r["adopted"] == ["U01/graded/全馬"] and not r["null_broken"]
    r2 = PR.judge([dict(good), dict(ctl, rd=0.05)], C)
    assert r2["null_broken"] and r2["adopted"] == []                                        # 負の対照が通れば、全体を使わない
    r3 = PR.judge([dict(good, periods={"2017〜21": -0.05, "2022〜24": -0.01, "2020〜24": -0.02}), dict(ctl)], C)
    assert r3["adopted"] == [] and not r3["rows"][0]["checks"]["3 最近の持続"]              # 最近で半分未満は採らない


def test_run_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "MARK", tmp_path / "m.json")
    monkeypatch.setattr(PR, "OUTP", tmp_path)
    PR.MARK.write_text(json.dumps({"finished_at": "x"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="1 回だけ"):
        PR.cmd_run()                                                                      # 流し終えたら 2 回目は流さない
    PR.MARK.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="1 回だけ"):
        PR.cmd_run()                                                                      # 途中で落ちて印だけ残っても流し直さない
    PR.MARK.unlink()
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="1 回だけ"):
        PR.cmd_run()                                                                      # 結果のファイルがあっても流さない
    (tmp_path / "result.json").unlink()
    monkeypatch.setattr(PR, "_git_clean", lambda paths: False)
    with pytest.raises(SystemExit, match="commit"):
        PR.cmd_run()


def test_config_matches_code():
    C = json.loads(PR.CONFIG.read_text(encoding="utf-8"))
    d = _toy()
    M = PR.rule_masks(d, PR.prev_lookup(d))
    assert {r["id"] for r in C["rules"]} | {n["id"] for n in C["negative_controls"]} == set(M)   # 登録の ID とコードの式が一対一
    n = sum(2 if r["side"] == "both" else 1 for r in C["rules"]) * len(C["layers"])
    assert n == 22                                                                            # BH の分母（§13.2）


def test_rule_boundaries():
    d = _toy()
    d.loc[4, ["grade", "p_grade", "p_fin", "p_margin", "p_jw"]] = ["G1", "G3", 2, 0.15, 56]     # 勝ち馬（56kg）と同じ斤量、差 0.15 → ⭕ でない
    d.loc[5, ["grade", "p_grade", "p_fin", "p_margin", "p_jw"]] = ["G1", "G3", 2, 0.15, 57]     # 1kg 重い → 直した差 0.05 → ⭕
    d.loc[6, ["grade", "p_grade", "p_fin"]] = ["G1", "OP", 1]                                     # 前走が重賞でない → ⭕ でない
    d.loc[7, ["grade", "p_l3rank", "p_pos1", "p_pos4"]] = ["G1", 9, 1, 1]                         # 上がり 9 位でも逃げ → U04 は ⭕
    M = PR.rule_masks(d, PR.prev_lookup(d))
    app, flag = M["U01"]
    assert app[4] and flag[4] and app[5] and not flag[5] and app[6] and flag[6]
    app, flag = M["U04"]
    assert app[7] and not flag[7]
    d2 = _toy()
    d2.loc[8, ["grade", "p_l3rank", "p_pos1", "p_pos4"]] = ["G1", 9, np.nan, np.nan]          # 通過順位が欠ける → 判定の外
    app, _ = PR.rule_masks(d2, PR.prev_lookup(d2))["U04"]
    assert not app[8]


def test_boot_empty_and_prev_from_full():
    assert len(PR.boot(np.array([], int), np.array([], bool), np.array([]), np.array([], str), 5, 1)) == 0
    d = _toy()
    pv_full = PR.prev_lookup(d)
    pv_cut = PR.prev_lookup(d[d["race_id"] != "R0"].reset_index(drop=True))
    assert not np.isnan(pv_full["prev_wt"].iloc[4]) and np.isnan(pv_cut["prev_wt"].iloc[0])     # 前走の表は絞る前の全体から引く
