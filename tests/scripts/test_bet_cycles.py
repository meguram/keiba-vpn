"""bet_cycles（推奨馬券の最適化サイクル）のテスト。合成データだけで動く（手元のデータ・GCS は読まない）。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from src.scripts.data import bet_cycles as bc

ROOT = Path(__file__).resolve().parents[2]


def _race(n=10, seed=0, fin=None):
    rng = np.random.default_rng(seed)
    pw = np.sort(rng.dirichlet(np.ones(n)))[::-1]
    pw, p3 = bc.normalize_probs(pw, None)
    odds = np.round(0.8 / pw, 1)
    pop = np.argsort(np.argsort(odds)) + 1.0
    hn = np.arange(1, n + 1)
    fin = np.arange(1, n + 1, dtype=float) if fin is None else np.asarray(fin, float)
    order = hn[np.argsort(fin)]
    a, b, c = (int(x) for x in order[:3])
    pay = {"tansho": {str(a): 500}, "fukusho": {str(a): 150, str(b): 200, str(c): 300},
           "umaren": {bc.combo_key("umaren", (a, b)): 1000}, "wide": {bc.combo_key("wide", (a, b)): 400, bc.combo_key("wide", (a, c)): 500,
                                                                     bc.combo_key("wide", (b, c)): 600},
           "umatan": {bc.combo_key("umatan", (a, b)): 2000}, "sanrenpuku": {bc.combo_key("sanrenpuku", (a, b, c)): 3000},
           "sanrentan": {bc.combo_key("sanrentan", (a, b, c)): 9000}}
    return bc.Race("202201010101", "2022-01-05", hn, pw, p3, odds, pop, fin, pay)


def test_combo_key_matches_payoff_format():
    assert bc.combo_key("umaren", (6, 2)) == "2-6"
    assert bc.combo_key("sanrenpuku", (14, 4, 13)) == "4-13-14"
    assert bc.combo_key("umatan", (6, 2)) == "6>2"
    assert bc.combo_key("sanrentan", (6, 2, 3)) == "6>2>3"
    assert bc.combo_key("tansho", (7,)) == "7"


def test_normalize_probs_sums():
    pw, p3 = bc.normalize_probs(np.array([0.5, 0.3, 0.1, 0.05, 0.05]), np.array([0.9, 0.8, 0.6, 0.4, 0.3]))
    assert pw.sum() == pytest.approx(1.0)
    assert p3.sum() == pytest.approx(3.0)
    assert p3.max() <= 1.0 + 1e-12
    _, q = bc.normalize_probs(np.array([0.5, 0.3, 0.2]), None)       # p_top3 が無ければ Harville で補う
    assert q.sum() == pytest.approx(3.0)


@pytest.mark.parametrize("typ,k,expect", [
    ("umaren", 3, 3), ("wide", 4, 4), ("umatan", 3, 3), ("umatan_multi", 3, 6), ("trio1", 4, 6), ("trio1", 5, 10),
    ("trio2", 3, 3), ("tri_first", 3, 6), ("tri_first", 4, 12), ("box_trio", 3, 4), ("box_umaren", 3, 6)])
def test_point_counts(typ, k, expect):
    """点数は betting_patterns.md §2 の式どおり（相手 k 頭）。"""
    assert len(bc.make_tickets(0, list(range(1, 8)), {"type": typ, "k": k})) == expect


def test_formation_counts():
    O = list(range(1, 8))
    assert len(bc.make_tickets(0, O, {"type": "trio_form", "m": 3})) == 7       # ◎-○▲-○▲△△△ = 1 + 2m
    assert len(bc.make_tickets(0, O, {"type": "tri_form", "m": 3})) == 8        # ◎→○▲→○▲△△△ = 2 + 2m


def test_honmei_always_in_every_ticket():
    for typ in ("tansho", "fukusho", "umaren", "wide", "umatan", "umatan_multi", "trio1", "trio2", "trio_form", "tri_first", "tri_form"):
        for bt, c, u in bc.make_tickets(5, [1, 2, 3, 4], {"type": typ, "k": 3}):
            assert 5 in c, (typ, c)


def test_check_marks_detects_each_rule():
    marks = {1: "◎", 2: "○", 3: "▲"}
    ok = [{"bet_type": "umaren", "combo": [1, 2], "yen": 200}, {"bet_type": "umaren", "combo": [1, 3], "yen": 100}]
    assert bc.check_marks(ok, marks) == []
    # R2・R3・R4: ○ の目に ◎ より多く張る（◎ 抜きの本線）
    bad = [{"bet_type": "umaren", "combo": [2, 3], "yen": 500}, {"bet_type": "umaren", "combo": [1, 2], "yen": 100}]
    v = bc.check_marks(bad, marks)
    assert {"R2", "R3", "R4"} <= set(v)
    # R1: 消の馬
    assert "R1" in bc.check_marks(ok, marks, kesi={3})
    # R6・R8: ○ 1 着の馬単を混ぜる
    v = bc.check_marks([{"bet_type": "umatan", "combo": [2, 1], "yen": 300}, {"bet_type": "umatan", "combo": [1, 3], "yen": 100}], marks)
    assert "R6" in v
    # R7: ◎ の単勝が無いのに ○ の単勝だけ
    assert "R7" in bc.check_marks([{"bet_type": "tansho", "combo": [2], "yen": 100}, {"bet_type": "wide", "combo": [1, 2], "yen": 100}], marks)
    # ▲ が ○ より厚い（印の順）
    v = bc.check_marks([{"bet_type": "wide", "combo": [1, 3], "yen": 300}, {"bet_type": "wide", "combo": [1, 2], "yen": 100}], marks)
    assert "R3" in v


def _all_specs():
    hon = [{"rule": "top"}, {"rule": "fav_maxodds", "fav_k": 3}, {"rule": "ana_ev", "fav_k": 3}, {"rule": "ana_p3", "fav_k": 4},
           {"rule": "ev_max"}]
    par = [{"mode": "p3", "n": 3}, {"mode": "fav", "n": 3, "fav_k": 3}, {"mode": "mix", "n": 2, "n_ana": 3, "fav_k": 3}, {"mode": "pw", "n": 4}]
    tks = [[{"type": t, "k": 3, "m": 2}] for t in ("tansho", "fukusho", "umaren", "wide", "umatan", "umatan_multi", "trio1", "trio2",
                                                    "trio_form", "tri_first", "tri_form")]
    tks += [[{"type": "tansho"}, {"type": "wide", "k": 3}], [{"type": "fukusho", "units": 2}, {"type": "trio1", "k": 4}]]
    for h, p, t, w in itertools.product(hon, par, tks, (1, 2, 3)):
        yield {"name": "x", "honmei": h, "partners": p, "tickets": t, "main_w": w}


def test_all_generated_strategies_respect_marks():
    """◎ 起点で組んだ買い方は、どの組み合わせでも印と配分が矛盾しない（R1〜R8 に違反しない）。"""
    races = [_race(n, s) for n, s in ((8, 1), (12, 2), (16, 3), (18, 4))]
    for spec in _all_specs():
        for r in races:
            f = bc.race_features(r)
            x = bc.play(r, f, spec)
            assert not x.get("violations"), (spec, x.get("violations"), x.get("bets"))
            if x["cost"]:
                assert x["cost"] <= bc.MAX_UNITS * bc.UNIT


def test_unit_weights_violation_is_caught():
    """◎ の単勝より相手の目を厚くする配分（units で ○ の目を厚く）は R2 か R3 で落ちる。"""
    r = _race(10, 5)
    spec = {"name": "bad", "honmei": {"rule": "top"}, "partners": {"mode": "p3", "n": 1},
            "tickets": [{"type": "tansho", "units": 1}, {"type": "umaren", "k": 1, "units": 1}]}
    bets = bc.play(r, bc.race_features(r), spec)["bets"]
    bets_bad = bets + [{"bet_type": "wide", "combo": [int(r.hn[1]), int(r.hn[2])], "yen": 900}]
    marks = {int(r.hn[0]): "◎", int(r.hn[1]): "○", int(r.hn[2]): "▲"}
    assert bc.check_marks(bets_bad, marks)


def test_settlement():
    r = _race(10, 0)          # 着順 = 馬番（1-2-3）。予測の並びは馬番とは別
    H = int(np.where(r.hn == 1)[0][0])
    O = [int(np.where(r.hn == 2)[0][0]), int(np.where(r.hn == 3)[0][0])]
    tk = bc.make_tickets(H, O, {"type": "wide"}) + bc.make_tickets(H, O, {"type": "tri_first"})
    bets = [{"bet_type": bt, "combo": [int(r.hn[i]) for i in c], "yen": u * 100} for bt, c, u in tk]
    ret = sum(r.pay[b["bet_type"]].get(bc.combo_key(b["bet_type"], b["combo"]), 0) for b in bets)
    assert ret == 400 + 500 + 9000


def test_metrics_required_fields_and_median():
    cost = np.array([100, 100, 0, 100, 200], float)
    ret = np.array([0, 300, 0, 0, 1000], float)
    dates = np.array(["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02", "2022-01-03"])
    m = bc.metrics(cost, ret, dates, bc.poisson_weights(5, 50))
    assert m["races"] == 5 and m["bet_races"] == 4 and m["bet_rate"] == pytest.approx(0.8)
    assert m["hit_rate"] == pytest.approx(0.5)
    assert m["roi"] == pytest.approx(1300 / 500)
    assert m["avg_return"] == pytest.approx(1300 / 4)
    assert m["median_return"] == pytest.approx(150)
    assert m["median_hit_return"] == pytest.approx(650)
    assert m["roi_ex_top1"] == pytest.approx(300 / 500)
    for k in ("roi_lo", "roi_hi", "daily_roi_sd", "max_drawdown", "avg_cost"):
        assert k in m


def test_decompose_adds_up():
    a = {"hit_rate": 0.2, "avg_hit_return": 900.0, "avg_cost": 200.0}
    b = {"hit_rate": 0.3, "avg_hit_return": 500.0, "avg_cost": 100.0}
    d = bc.decompose(a, b)
    roi_a, roi_b = 0.2 * 900 / 200, 0.3 * 500 / 100
    assert d["log_total"] == pytest.approx(np.log(roi_a / roi_b))


def test_qcuts_fit_on_train_only():
    import pandas as pd

    tr = pd.DataFrame({"p1": np.linspace(0, 1, 101)})
    cuts = bc.fit_qcuts(tr, cols=("p1",))
    ev = bc.apply_qcuts(pd.DataFrame({"p1": [0.05, 0.5, 0.95]}), cuts)
    assert ev["q_p1"].tolist() == [1, 3, 5]


def test_policy_skip_and_map():
    r = _race(10, 2)
    f = bc.race_features(r)
    f["shape"] = "chaos"
    pol = {"name": "p", "policy": {"by": "shape", "map": {"one": {"honmei": {"rule": "top"}, "tickets": [{"type": "tansho"}]}}, "default": "skip"}}
    assert bc.play(r, f, pol)["cost"] == 0
    f["shape"] = "one"
    assert bc.play(r, f, pol)["cost"] == 100


def test_cycle_configs_are_valid():
    """config/bet_cycles の各サイクルの買い方が展開でき、合成のレースで印の整合を守る。"""
    races = [_race(12, 7), _race(16, 8)]
    for p in sorted((ROOT / "config/bet_cycles").glob("cycle_*.json")):
        cyc = json.loads(p.read_text(encoding="utf-8"))
        for spec in bc.expand_variants(cyc):
            for r in races:
                f = bc.race_features(r)
                f.update({"shape": "one", "q_p1": 3, "q_ent": 3, "q_gap12": 3, "q_ev_max": 3, "fieldband": "10-13頭"})
                x = bc.play(r, f, spec)
                assert not x.get("violations"), (p.name, spec.get("name"), x.get("violations"))


def test_cap_units_keeps_upper_marks():
    """上限を超えたら、そこで止める（上位の目を削って下位を残すことはしない）。"""
    tk = [("umaren", (0, 1), 15), ("umaren", (0, 2), 6), ("umaren", (0, 3), 1)]
    assert bc.cap_units(tk) == [("umaren", (0, 1), 15)]


def test_r8_multi_exemption_requires_honmei_in_all():
    marks = {1: "◎", 2: "○", 3: "▲"}
    bets = [{"bet_type": "umatan", "combo": [1, 2], "yen": 300}, {"bet_type": "umatan", "combo": [2, 1], "yen": 100},
            {"bet_type": "umatan", "combo": [3, 2], "yen": 100}]
    assert "R8" in bc.check_marks(bets, marks)


def test_mixed_box_is_checked():
    """◎ 起点の券種とボックスを混ぜた買い方は、◎ なしの扱いにしない（整合の検査を受ける）。"""
    r = _race(10, 3)
    spec = {"name": "mix", "honmei": {"rule": "top"}, "partners": {"mode": "p3", "n": 3},
            "tickets": [{"type": "tansho"}, {"type": "box_umaren", "k": 3}]}
    x = bc.play(r, bc.race_features(r), spec)
    assert x["violations"]          # ボックスの ◎ 抜きの目が ◎ の持ち分を崩す


def test_reference_families_named():
    assert bc.family({"honmei": {"rule": "ev_max"}}).startswith("参考")
    assert bc.family({"tickets": [{"type": "box_trio"}]}).startswith("参考")
    assert not bc.family({"honmei": {"rule": "ana_ev", "max_rank": 8}, "partners": {"mode": "fav"}}).startswith("参考")


def _race_with_pop(pop, p2=None, seed=0, n=10):
    r = _race(n, seed)
    r.pop = np.asarray(pop, float)
    r.p2 = bc.p2_from(r.pw, r.p3, None) if p2 is None else np.asarray(p2, float)
    return r


def test_fav1_penalty():
    pop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                 # 予測 1 位が 1 番人気
    r = _race_with_pop(pop)
    f = bc.race_features(r)
    assert bc.pick_honmei(r, f, {"rule": "fav1_penalty", "lam": 0.0})[0] == 0
    i, _ = bc.pick_honmei(r, f, {"rule": "fav1_penalty", "lam": 1.0})
    assert r.pop[i] != 1 and i == 1                        # 2 番人気以下で勝率最大


def test_fav1_exception_paths():
    pop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = _race_with_pop(pop)
    f = bc.race_features(r)
    p2 = r.p2.copy()
    p2[0] = 0.95
    r.p2 = p2
    i, why = bc.pick_honmei(r, f, {"rule": "fav1_exc", "p2_min": 0.9})
    assert i == 0 and "例外" in why
    p2[0] = 0.5
    r.p2 = p2
    i, _ = bc.pick_honmei(r, f, {"rule": "fav1_exc", "p2_min": 0.9, "base": {"rule": "top"}})   # base が 1 番人気を選んでも替わる
    assert r.pop[i] != 1
    i, why = bc.pick_honmei(r, f, {"rule": "fav1_exc", "p2_min": 0.9, "base": {"rule": "ana_p3", "fav_k": 99, "fallback": "skip"}})
    assert i is None                                       # base が見送りならそのまま見送り


def test_p2_from_bounds():
    pw, p3 = bc.normalize_probs(np.array([0.4, 0.3, 0.15, 0.1, 0.05]), None)
    for raw in (None, np.array([0.7, np.nan, 0.3, 0.2, 0.1]), np.array([0.9, 0.8, 0.2, 0.05, 0.05])):
        p2 = bc.p2_from(pw, p3, raw)
        assert np.all(p2 >= pw - 1e-12) and np.all(p2 <= p3 + 1e-12)
        assert abs(p2.sum() - 2.0) < 0.05


def test_family_reference_through_fav1_exc():
    assert bc.family({"honmei": {"rule": "fav1_exc", "base": {"rule": "ana_ev", "fav_k": 1}}}).startswith("参考")
    assert not bc.family({"honmei": {"rule": "fav1_exc", "base": {"rule": "ana_ev", "fav_k": 1, "max_rank": 8}}}).startswith("参考")
    a = bc.family({"honmei": {"rule": "fav1_penalty", "lam": 1}})
    assert a == bc.family({"honmei": {"rule": "fav1_penalty", "lam": 0.5}})     # λ で系統が割れない


def test_choose_fav1_threshold_not_single_lucky_point():
    grid = [{"p2_min": 0.80, "n": 200, "rentai": 0.86}, {"p2_min": 0.825, "n": 150, "rentai": 0.83},
            {"p2_min": 0.85, "n": 100, "rentai": 0.875}, {"p2_min": 0.875, "n": 50, "rentai": 0.9}, {"p2_min": 0.9, "n": 10, "rentai": 0.5}]
    assert bc.choose_fav1_threshold(grid, 0.85, 30)["p2_min"] == 0.85


def test_eligible_and_rank():
    base = {"train": {"roi_ex_top5": 0.88}, "train_folds": {"2022": 0.9, "2023": 0.89}}
    row = {"train": {"bet_races": 500, "violation_races": 0, "hit_rate": 0.05, "roi_ex_top5": 1.1, "roi_lo": 1.0}, "train_folds": {"2022": 1.2, "2023": 1.0}}
    assert bc.eligible(row, base) and not bc.eligible(row, base, min_hit=0.08)
    assert bc.rank_key(row, "robust") == pytest.approx(1.0) and bc.rank_key(row, "roi_lo") == pytest.approx(1.0)
    row["train_folds"]["2023"] = 0.85
    assert not bc.eligible(row, base)


def test_report_renders_and_escapes():
    from src.scripts.data import bet_cycles_report as rp

    m = {"races": 100, "bet_races": 10, "bet_rate": 0.1, "hit_rate": 0.2, "roi": 1.1, "roi_lo": 0.9, "roi_hi": 1.3, "avg_return": 110,
         "median_return": 0, "median_hit_return": 500, "avg_cost": 100, "roi_ex_top5": 0.8, "daily_roi_sd": 0.5, "max_drawdown": 300,
         "max_drawdown_days": 2.0, "h_fav1_rate": 0.1}
    log = {"tag": "t", "cycles": [
        {"n": 1, "title": "<a&b>", "hypothesis": "x\ny", "change": "c", "selected": None, "baselines": [{"name": "B0", "train": m, "eval": m}], "drop": {}},
        {"n": 2, "title": "t", "hypothesis": "h", "change": "c", "baselines": [],
         "selected": {"name": "n<1>", "family": "f", "spec": {"honmei": {"rule": "fav1_exc", "p2_min": 0.85, "base": {"rule": "ana_ev", "fav_k": 1, "max_rank": 8}},
                                                               "partners": {"mode": "p3", "n": 2}, "tickets": [{"type": "wide"}]},
                      "train": m, "eval": {"races": 50, "bet_races": 0}, "train_folds": {"2022": 1.0}}, "eval_check": {"verdict": "上向き（点推定のみ）"}}]}
    h = rp.render_html(log)
    assert "&lt;a&amp;b&gt;" in h and "n&lt;1&gt;" in h and "<a&b>" not in h
    import re

    for tb in re.findall(r"<table>.*?</table>", h, flags=re.S):
        nth = len(re.findall(r"<th[ >]", tb.split("</thead>")[0]))
        for tr in re.findall(r"<tr>(.*?)</tr>", tb.split("<tbody>")[1], flags=re.S):
            assert len(re.findall(r"<td", tr)) == nth
    assert "2番人気以下で" in rp.spec_text(log["cycles"][1]["selected"]["spec"])
    md = rp.render_md_tables(log)
    assert md.startswith(rp.MD_BEGIN) and md.endswith(rp.MD_END) and "x y" in md


def test_parse_flat_payoff_seven_types():
    raw = json.dumps({"単勝": {"numbers": "4", "payout": "3,110", "popularity": "8"},
                      "複勝": [{"numbers": "4", "payout": "1,980", "popularity": "8"}, {"numbers": "2", "payout": "570", "popularity": "4"}],
                      "馬連": {"numbers": "4-2", "payout": "16,720", "popularity": "40"},
                      "ワイド": [{"numbers": "2-4", "payout": "3,370", "popularity": "30"}],
                      "馬単": {"numbers": "4→2", "payout": "40,000", "popularity": "90"},
                      "3連複": {"numbers": "8-4-2", "payout": "50,000", "popularity": "100"},
                      "三連単": {"numbers": "4-2-8", "payout": "300,000", "popularity": "500"}}, ensure_ascii=False)
    x = bc.parse_flat_payoff(raw)
    assert x["umaren"] == {"2-4": 16720} and x["umatan"] == {"4>2": 40000} and x["sanrenpuku"] == {"2-4-8": 50000}
    assert x["sanrentan"] == {"4>2>8": 300000} and x["fukusho"]["2"] == 570
    bad = json.loads(raw)
    bad["ワイド"] = {"numbers": "2-4", "payout": "3,370"}          # 連結された保存（popularity なし）は読めないとみなす
    assert bc.parse_flat_payoff(json.dumps(bad, ensure_ascii=False)) is None


def test_judge_level_rules():
    good = {"roi": 1.3, "roi_lo": 1.05, "roi_hi": 1.6, "roi_ex_top5": 1.1, "folds": {"2022": 1.2, "2023": 1.4}, "bet_races": 500}
    assert bc.judge_level(good, 0.9) == "得意"
    bad = {"roi": 0.7, "roi_lo": 0.5, "roi_hi": 0.85, "roi_ex_top5": 0.6, "folds": {"2022": 0.8, "2023": 0.6}, "bet_races": 500}
    assert bc.judge_level(bad, 0.9) == "不得意"
    mid = {"roi": 1.1, "roi_lo": 0.8, "roi_hi": 1.4, "roi_ex_top5": 0.95, "folds": {"2022": 1.3, "2023": 0.9}, "bet_races": 500}
    assert bc.judge_level(mid, 0.9) == "判定できない"
    assert bc.shrunk({"roi": 2.0, "bet_races": 100}, 1.0, 300) == pytest.approx(1.25)


def test_tiers_skip_bad_levels_and_order():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 4000
    feats = pd.DataFrame({ax: "x" for ax in bc.TIER_AXES}, index=range(n))
    feats["c_venue"] = rng.choice(["良場", "悪場", "並場"], n)
    cost = np.full(n, 100.0)
    p_hit = np.where(feats["c_venue"] == "良場", 0.4, np.where(feats["c_venue"] == "悪場", 0.05, 0.25))   # 全体では勝っている（1.17 前後）
    ret = np.where(rng.random(n) < p_hit, 500.0, 0.0)
    years = np.where(np.arange(n) % 2 == 0, 2022, 2023)
    dates = np.array([f"2022-01-{1 + i % 28:02d}" for i in range(n)])
    W = bc.poisson_weights(n, 200)
    mdl = bc.fit_tiers(cost, ret, feats, np.ones(n, bool), 0.9, dates, W, years)
    assert mdl["axes"]["c_venue"]["悪場"]["judge"] == "不得意"
    t = bc.assign_tiers(feats, mdl)
    assert (t[feats["c_venue"] == "悪場"] == 0).all()                 # 不得意は見送り
    assert t[feats["c_venue"] == "良場"].mean() > t[feats["c_venue"] == "並場"].mean()


def test_resolve_auto_only_p2_min(tmp_path):
    (tmp_path / "stage_fav1_exception.json").write_text(json.dumps({"chosen": {"p2_min": 0.85}}), encoding="utf-8")
    s = [{"name": "auto", "honmei": {"rule": "fav1_exc", "p2_min": "auto", "base": {"rule": "ana_ev", "note": "auto"}}}]
    out = bc.resolve_auto(s, tmp_path)
    assert out[0]["honmei"]["p2_min"] == 0.85 and out[0]["name"] == "auto" and out[0]["honmei"]["base"]["note"] == "auto"
    assert bc.resolve_auto([{"honmei": {"p2_min": 0.9}}], tmp_path / "none") == [{"honmei": {"p2_min": 0.9}}]


def test_parse_flat_payoff_rejects_concatenated_amount():
    raw = {"単勝": {"numbers": "4", "payout": "3,1903,470", "popularity": "8"}}
    for k in ("複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"):
        raw[k] = [{"numbers": "4" if k == "複勝" else ("4-2" if k in ("馬連", "ワイド", "馬単") else "4-2-8"), "payout": "100", "popularity": "1"}]
    assert bc.parse_flat_payoff(json.dumps(raw, ensure_ascii=False)) is None


def test_fav1_rule_ok():
    assert bc.fav1_rule_ok({"honmei": {"rule": "fav1_exc", "base": {"rule": "ana_ev", "fav_k": 1}}})
    assert not bc.fav1_rule_ok({"honmei": {"rule": "top"}})
    assert bc.fav1_rule_ok({"honmei": {"rule": "top"}, "skip": {"h_pop_min": 4}})
    assert not bc.fav1_rule_ok({"honmei": {"rule": "ana_ev", "fav_k": 3}})                       # 候補なしで予測 1 位に戻る
    assert bc.fav1_rule_ok({"honmei": {"rule": "ana_ev", "fav_k": 3, "fallback": "skip"}})
    assert not bc.fav1_rule_ok({"honmei": {"rule": "fav1_penalty", "lam": 0.5}})
    pol = {"policy": {"typing": {"kind": "feat", "col": "x"}, "map": {"1": "skip", "2": {"honmei": {"rule": "top"}}}}}
    assert not bc.fav1_rule_ok(pol)


def test_report_conclusion_stage_once_and_on_top():
    from src.scripts.data import bet_cycles_report as rp

    m = {"races": 10, "bet_races": 5, "bet_rate": 0.5, "hit_rate": 0.2, "roi": 1.0, "roi_lo": 0.9, "roi_hi": 1.1, "avg_return": 100, "median_return": 0,
         "median_hit_return": 500, "avg_cost": 100, "roi_ex_top5": 0.9, "daily_roi_sd": 0.1, "max_drawdown": 0, "max_drawdown_days": 0, "h_fav1_rate": 0}
    log = {"tag": "t", "cycles": [{"n": 1, "title": "t", "hypothesis": "h", "change": "c", "selected": None, "baselines": [{"name": "B0", "train": m, "eval": m}], "drop": {}}],
           "stages": {"stage_00_conclusion": {"title": "結論XYZ", "text": ["x"], "tables": []}, "stage_cycles14_20": {"title": "c1420", "text": [], "tables": []}}}
    h = rp.render_html(log)
    assert h.count("結論XYZ") == 1
    assert h.index("結論XYZ") < h.index("推移") and "14 / 20" not in h and "8 / 20" in h


def test_load_pred_where_and_r_rule(tmp_path):
    import pandas as pd

    d = pd.DataFrame({"race_id": ["1"] * 4, "horse_id": ["a", "b", "a", "b"], "p_win": [0.6, 0.4, 0.5, 0.5], "q": [0.5, 0.5, 0.5, 0.5],
                      "r": [0.2, -0.2, 0.0, 0.0], "part": ["oof", "oof", "test2025", "test2025"], "odds_phase": ["final"] * 4})
    f = tmp_path / "p.parquet"
    d.to_parquet(f)
    o = bc.load_pred({"path": str(f), "where": {"part": "oof", "odds_phase": "final"}})
    assert len(o) == 2 and set(o.columns) == {"race_id", "horse_id", "p_win", "q", "r"}
    r = _race(8, 1)
    r.r = np.array([0.1, -0.3, 0.5, 0.2, -0.1, 0.9, 0.0, 0.3])
    r.pop = np.arange(1, 9, dtype=float)
    f_ = bc.race_features(r)
    i, why = bc.pick_honmei(r, f_, {"rule": "r_max", "fav_k": 1, "p_min": 0.0})
    assert i == 5 and r.pop[i] != 1
    assert bc.fav1_rule_ok({"honmei": {"rule": "r_max", "fav_k": 1}})
    assert bc.fav1_rule_ok({"honmei": {"rule": "fav1_exc", "exc_by": "r", "r_min": "auto", "base": {"rule": "r_max", "fav_k": 1}}})
    assert "r_max" in f_ and f_["r_max"] == pytest.approx(float(r.r[r.pw >= 0.05].max()))   # 勝率 5% 以上の中の最大


def test_round2_cycle_configs_are_valid():
    """config/bet_cycles/r2 の各サイクルの買い方が展開でき、r と連対率のある合成のレースで印の整合を守る（auto は仮の値に置き換える）。"""
    races = [_race(12, 7), _race(16, 8)]
    for r in races:
        r.r = np.linspace(0.6, -0.6, len(r.pw))
        r.p2 = bc.p2_from(r.pw, r.p3, None)
    d = ROOT / "config/bet_cycles/r2"
    old = bc.CYC_DIR
    bc.CYC_DIR = d
    try:
        for p in sorted(d.glob("cycle_*.json")):
            cyc = json.loads(p.read_text(encoding="utf-8"))
            specs = json.loads(json.dumps(bc.expand_variants(cyc)).replace('"auto"', "0.5"))
            assert specs, p.name
            for spec in specs:
                for r in races:
                    f = bc.race_features(r)
                    f.update({"shape": "one", "q_p1": 3, "q_ent": 3, "q_gap12": 3, "q_ev_max": 3, "q_r_max": 3, "q_r_top": 3, "fieldband": "10-13頭"})
                    x = bc.play(r, f, spec)
                    assert not x.get("violations"), (p.name, spec.get("name"), x.get("violations"))
    finally:
        bc.CYC_DIR = old


def test_test_markers_block():
    r1 = ["bet_test2025_f96a5aeff33e.json"]
    assert bc.test_markers_block(1, "", r1)[0]                         # 第 1 回はもう見た
    assert bc.test_markers_block(1, "test", r1)[0]                     # group を付けても同じ回は見られない
    assert bc.test_markers_block(3, "test", r1)[0]                     # 回は 1・2 だけ
    assert bc.test_markers_block(2, "test", [])[0]                     # 第 2 回は第 1 回の後だけ
    assert not bc.test_markers_block(2, "test", r1)[0]
    assert bc.test_markers_block(2, "fwd", r1)[0]                      # 2026 年は test の後
    r2t = r1 + ["bet_test2025_r2_test_abcdefabcdef.json"]
    assert bc.test_markers_block(2, "test", r2t)[0] and bc.test_markers_block(2, "", r2t)[0]
    assert not bc.test_markers_block(2, "fwd", r2t)[0]
    assert bc.test_markers_block(2, "fwd", r2t + ["TEST2025_EVALUATED_r2_fwd_abcdefabcdef.json"])[0]


def test_exc_r_values_only_exception_level():
    spec = {"honmei": {"rule": "fav1_exc", "exc_by": "r", "r_min": 0.3, "base": {"rule": "r_max", "r_min": 0.1}}}
    assert bc._exc_r_values(spec) == [0.3]
    assert not bc.fav1_rule_ok({"honmei": {"rule": "fav1_exc", "exc_by": "r", "r_min": -9.0}})
