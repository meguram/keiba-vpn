"""重賞のルールベース予想家の消しの版（src/research/graded/kill.py）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.graded import kill as K


def _S(**over):
    """馬 × 型の行を 1 つ（主 12 走・副・全重賞の件数つき）。"""
    r = {"row": 1, "type": "A04", "level": "a", "layer": "主", "x": -0.3, "all_sub_n": 1000.0,
         "main_n": 12.0, "main_w1": 0.0, "main_w2": 0.0, "main_w3": 0.0, "main_e3": 3.0, "main_ed": 6.0,
         "sub_n": 5.0, "sub_w1": 0.0, "sub_w2": 0.0, "sub_w3": 0.0, "sub_e3": 1.0, "sub_ed": 5.0,
         "sub4_n": 50.0, "sub4_w1": 0.0, "sub4_w2": 0.0, "sub4_w3": 0.0, "sub4_e3": 10.0, "sub4_ed": 30.0,
         "glob_n": 80.0, "glob_w1": 1.0, "glob_w2": 0.0, "glob_w3": 0.0, "glob_e3": 16.0, "glob_ed": 50.0}
    r.update(over)
    return pd.DataFrame([r])


def test_versions_grid_and_names_are_unique():
    vs = K.all_versions()
    assert len(vs) == 2 * 3 * 3 * 2 * 2 * 2 + 2
    names = [K.version_name(p) for p in vs]
    assert len(set(names)) == len(names)


def test_stat_hits_needs_runs_races_and_negative_x():
    assert K.stat_hits(_S(), "3着内0", 10, "主だけ")[0]
    assert not K.stat_hits(_S(), "3着内0", 20, "主だけ")[0]                 # 走数が足りない
    assert not K.stat_hits(_S(main_ed=2.0), "3着内0", 10, "主だけ")[0]      # 主の異なる回が 3 未満
    assert K.stat_hits(_S(main_n=40.0, main_ed=3.0), "3着内0", 40, "主だけ")[0]   # 主は回の数の下限が 3（n の下限によらない）
    assert not K.stat_hits(_S(main_n=0.0, main_ed=0.0, all_sub_n=100.0, sub4_ed=4.0), "3着内0", 10, "主＋副")[0]   # 副は max(3, 10/2)=5 回
    assert not K.stat_hits(_S(x=0.0), "3着内0", 10, "主だけ")[0]            # x が 0（NO_INFO など）は消さない
    assert not K.stat_hits(_S(main_w3=1.0), "3着内0", 10, "主だけ")[0]
    # 副は全体が 300 走以上なら OP 以上（5 走で足りない）、未満なら 3 勝クラス以上（50 走・3着内 0）
    assert not K.stat_hits(_S(main_n=0.0, main_ed=0.0), "3着内0", 10, "主＋副")[0]
    assert K.stat_hits(_S(main_n=0.0, main_ed=0.0, all_sub_n=100.0), "3着内0", 10, "主＋副")[0]
    # 全重賞は 3着内が 1 あるので 3着内0 では当たらず、上限の基準なら（80 走で 1 回、期待 0.2）当たる
    assert not K.stat_hits(_S(main_n=0.0, main_ed=0.0), "3着内0", 40, "全部")[0]
    assert K.stat_hits(_S(main_n=0.0, main_ed=0.0), "上限が期待の半分以下", 40, "全部")[0]


def test_roi_hits_only_main_or_sub_layer_and_bad_direction():
    prim = {"patterns": [["E02", "4角 後方20%", "凡"]], "used": ["E02"], "params": {"layer": "主＋副"}}
    S = pd.concat([_S(type="E02", level="4角 後方20%"), _S(type="E02", level="4角 後方20%", layer="全体"),
                   _S(type="E02", level="4角 後方20%", x=0.2), _S(type="E02", level="別")], ignore_index=True)
    assert K.roi_hits(S, prim).tolist() == [True, False, False, False]


def _base():
    # 1 レース 6 頭。前日版の順位 = 馬番の順
    return pd.DataFrame({"race_id": "r1", "row": range(6), "hn": [1, 2, 3, 4, 5, 6], "rank": range(1, 7),
                         "y1": [0, 1, 0, 0, 0, 0], "y3": [1, 1, 0, 1, 0, 0], "field": 6})


def test_order_marks_keeps_or_skips_killed_honmei():
    b = _base()
    a = K.order_marks(b, {0, 2}, "そのまま").sort_values("rank")
    assert a["hn"].tolist() == [1, 2, 4, 5, 6, 3]          # ◎○ は点の上位のまま、相手は消しでない馬 → 消した馬（補い）
    c = K.order_marks(b, {0, 2}, "消しは◎○にしない").sort_values("rank")
    assert c["hn"].tolist() == [2, 4, 5, 6, 1, 3]


def test_race_metrics_kill_counts_costs_and_misses():
    b = K.order_marks(_base(), {0, 2}, "そのまま")
    pay = {("r1", "tansho"): {"2": 500}, ("r1", "fukusho"): {"1": 150, "2": 200, "4": 300},
           ("r1", "umaren"): {"1-2": 900}, ("r1", "wide"): {"1-2": 300, "1-4": 400, "2-4": 500}}
    r = K.race_metrics_kill(b, pay).iloc[0]
    assert r["ret_fukusho"] == 150 and r["ret_tansho"] == 0
    assert r["cost_umaren"] == 200 and r["ret_wide"] == 700           # ◎1 → ○2・▲4
    assert r["cost_wide_all"] == 400 and r["ret_wide_all"] == 700     # 消しでない 2・4・5・6
    assert r["cost_wide_3"] == 300 and r["cost_wide_5"] == 500
    assert r["n_killed"] == 2 and r["killed_top3"] == 1 and r["kill_miss_race"] == 1 and r["honmei_killed"] == 1
    assert np.isclose(r["killed_e3"], 1.0)
    s = K.summarize_bets(K.race_metrics_kill(b, pay), n_boot=50)
    assert s["roi_wide_all"]["est"] == 700 / 400


def test_race_with_no_left_horses_costs_nothing():
    b = K.order_marks(_base(), {1, 2, 3, 4, 5}, "そのまま")
    pay = {("r1", bt): {} for bt in ("tansho", "fukusho", "umaren", "wide")}
    r = K.race_metrics_kill(b, pay).iloc[0]
    assert r["cost_wide_all"] == 0 and r["cost_wide"] == 200


def test_choose_primary_uses_precision_not_payout():
    def v(name, nmin, killed, rate, honmei="そのまま"):
        pr = {k: {"killed": killed, "races": 100, "top3_rate": rate, "e3_rate": 0.2} for k in ("disc", "conf")}
        return {"name": name, "params": {"crit": "3着内0", "nmin": nmin, "layer": "主だけ", "hits": 1, "roi7": "入れない", "honmei": honmei},
                "precision": pr, "passes": K.passes(pr)}
    vers = [v("a", 10, 300, 0.12), v("b", 20, 200, 0.05), v("c", 40, 150, 0.08), v("d", 10, 900, 0.01, "消しは◎○にしない")]
    assert K.choose_primary(vers) == "b"                  # a は精度が足りない。b と c では消しの数が多い b


def test_order_marks_rewrites_marks_after_reordering():
    b = _base().assign(mark=["◎", "○", "▲", "△", "△", ""])
    a = K.order_marks(b, {2}, "そのまま").sort_values("rank")
    assert a["mark"].tolist() == ["◎", "○", "▲", "△", "△", ""] and a["hn"].tolist()[2] == 4


def test_constants_are_json_round_trip_stable():
    import json
    c = K.constants()
    assert json.loads(json.dumps(c)) == json.loads(json.dumps(K.constants())) and c["ed_min_check"] == [3, 3, 3, 5, 10, 20]


def test_kill_summary_and_handicapper_html_section(tmp_path):
    """export.kill_summary が主に見る版と全版の一覧を作り、graded_handicapper.html の検証のタブに消しの節の材料が載る。"""
    import json

    from src.research.graded.export import kill_summary
    from src.scripts.docs import build_graded_handicapper_html as OLD

    o = {"est": 0.8, "lo": 0.6, "hi": 1.0, "cost": 100.0}
    allm = {k: dict(o) for k in ["win", "place", "top3_in3"] + [f"roi_{b}" for b in K.BETS]}
    kill = {"races": 2, "killed_per_race": 1.0, "left_per_race": 5.0, "horses_per_race": 6.0, "killed": 2, "killed_top3_rate": 0.1,
            "killed_e3_rate": 0.2, "killed_win_rate": 0.0, "race_miss_rate": 0.5, "honmei_killed_rate": 0.0}
    per = {"2025-2026": {"all": allm, "kill": kill, "diff_vs_前日版": {k: {"est": 0.0, "lo": -0.1, "hi": 0.1} for k in allm}}}
    p = K.all_versions()[0]
    name = K.version_name(p)
    ev = {"evaluated_at": "t", "note": "3 回目", "span": [2025, 2026], "primary": name, "versions": {"前日版（消しなし）": per, name: per}}
    pk = {"frozen_at": "f", "n_versions": 1, "rules": {"stat": "x<0", "crit": {"a": "b"}, "marks": "m", "bets": "b"},
          "versions": [{"name": name, "params": p, "primary": True, "passes": True,
                        "precision": {"disc": {"top3_rate": 0.1, "e3_rate": 0.2, "killed_per_race": 1.0}, "conf": {"top3_rate": 0.1, "e3_rate": 0.2, "killed_per_race": 1.0}}}]}
    ks = kill_summary(ev, pk)
    assert ks["primary"] == name and len(ks["table"]) == 1 and ks["table"][0]["diff"]["roi_wide_all"] == [0.0, -0.1, 0.1]
    src = tmp_path / "r.json"
    src.write_text(json.dumps({"prereg": {"frozen_at": "x"}, "races": [], "eval_races": [], "kill": ks}, ensure_ascii=False), encoding="utf-8")
    h = OLD.build(src, tmp_path / "h.html").read_text(encoding="utf-8")
    import re
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', h, re.S).group(1).replace("<\\/", "</"))
    assert data["kill"]["primary"] == name and data["kill"]["table"][0]["name"] == name   # slim を通っても消しの材料が残る
    assert "function renderKill" in h and "if(D.kill)h+=renderKill()" in h
