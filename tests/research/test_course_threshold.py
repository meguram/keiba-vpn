"""閾値のルールの版（src/research/course/threshold.py）。"""

import numpy as np
import pandas as pd
import pytest

from src.research.course import threshold as TH


def _frame(n_races=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        y = 2016 + r % 9
        for h in range(1, 11):
            strong = h <= 2
            rows.append({"race_id": f"{y}{r:04d}", "year": y, "surf": "芝", "dist": 1600, "course": "東京芝1600", "cls": 2, "track": "良",
                         "race_class": "1勝クラス", "rkey": None, "fin": h if strong else rng.integers(3, 11),
                         "p_fin": 1 if strong else 8, "p_margin": 0.1 if strong else 1.5, "p_l3rank": 1 if strong else 9, "p_pos4": 2, "p_pos4r": 0.2,
                         "interval": 28, "p_dist": 1600, "jw": 55, "p_jw": 55, "starts_prev": 5, "top3_prev": 3 if strong else 0,
                         "course_top3_prev": 0, "j_rate": 0.25, "t_rate": 0.1, "pay_ok": True, "ret_tan": 3.0 if h == 1 else 0,
                         "ret_fuku": 1.5 if (strong and h <= 3) else 0, "field": 10, "hn": h,
                         "pop_final": 3 + (h % 5) if strong else 1 + (h % 8), "odds_final": 5.0})
    return pd.DataFrame(rows)


def test_conditions_and_neighbors():
    d = _frame(4)
    C = TH.conditions(d)
    labels = [c["label"] for c in C]
    assert "前走の上がり 3 位以内" in labels and "中 2〜8 週" in labels
    R = TH.rules_of(C)
    assert all(len({C[i]["fam"] for i in r}) == len(r) for r in R)                   # 同じ種類は 1 個まで
    i3 = labels.index("前走 3 着以内")
    nb = TH.neighbors(C, (i3,))
    assert {C[x[0]]["label"] for x in nb} == {"前走 5 着以内", "前走 2 着以内"}       # 1 段ずらし
    d.loc[0, "p_pos4"] = 0
    C2 = TH.conditions(d)
    assert not C2[labels.index("前走の 4 角が前から 50% 以内")]["m"][0]             # 0 は不明として外す


def test_bh_and_wilson():
    q = TH.bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(q >= np.array([0.01, 0.04, 0.03, 0.5]) - 1e-12) and q[3] == 0.5
    lo, hi = TH.wilson(8, 10)
    assert lo < 0.8 < hi


def test_search_finds_strong_rule_and_guards_years():
    d = _frame(60)
    d["yr"] = d["year"]
    d["clsn"] = 2.0
    d["dband"] = "1401〜1800"
    d["group"] = "1勝・2勝"
    other = d.copy()
    other["course"] = "中山芝2000"                                                    # 似た単位（同じ芝 × 群のほかの距離帯）
    other["dist"] = 2000
    other["dband"] = "1801〜2200"
    other["race_id"] = "x" + other["race_id"]
    d = pd.concat([d, other], ignore_index=True)
    r = TH.search(d, "course", log=lambda *a: None)
    assert r["why"]["通った"] > 0, r["why"]
    u = r["units"]["東京芝1600|1勝・2勝"]
    assert u["rules"], "強い馬（前走 1 着・上がり 1 位）のルールが見つかる"
    best = u["rules"][0]
    assert best["disc"]["place"] > 0.75 and best["pass"]["0.75"] and best["why"]
    assert best["disc"]["ex"] > 0 and best["uplift"]["pt"] > 0 and best["trend"] and len(best["periods"]) == 3   # オッズから見込む値を超えた分・3 期間
    d2 = d.copy()
    d2.loc[0, "yr"] = 2025
    with pytest.raises(ValueError):
        TH.search(d2, "course", log=lambda *a: None)                                  # 2025 年の行があれば止める


def _prep(d):
    d = d.copy()
    d["yr"] = d["year"]
    d["clsn"] = pd.to_numeric(d["cls"])
    d["dband"] = "1401〜1800"
    d["group"] = d["cls"].map(TH.CR.group_of)
    return d


def test_breadth_excludes_own_course():
    """似た単位はこの単位のコースを外して比べる: ほかのコースで強さが無ければ、検査で落ちる。"""
    a = _frame(60)
    b = _frame(60, seed=1)
    b["course"] = "中山芝2000"
    b["dist"] = 2000
    b["race_id"] = "x" + b["race_id"]
    b["fin"] = np.where(b["p_fin"] == 1, 8 + (b["hn"] % 3), b["hn"] - 2).clip(1, 10)   # ほかの距離帯では、前走 1 着の馬がむしろ負ける
    d = _prep(pd.concat([a, b], ignore_index=True))
    d.loc[d["course"] == "中山芝2000", "dband"] = "1801〜2200"
    r = TH.search(d, "course", log=lambda *x: None)
    assert not r["units"]["東京芝1600|1勝・2勝"]["rules"]
    assert r["why"]["似た単位（探した層の外）"] > 0                                  # 似た単位（ほかの距離帯）で人気を超えないので落ちる


def test_pooled_counts_each_run_once():
    a = _frame(60)
    b = a.copy()
    b["course"] = "中山芝2000"
    b["race_id"] = "x" + b["race_id"]
    d = _prep(pd.concat([a, b], ignore_index=True))
    d.loc[d["course"] == "中山芝2000", "dband"] = "1801〜2200"                         # 似た単位は、ほかの距離帯で確かめる
    r = TH.search(d, "course", log=lambda *x: None)
    yr = d["yr"]
    strong = (d["p_fin"] == 1)
    for u in r["units"].values():
        assert len(u["rules"]) >= 2                                                     # 同じ走に当たるルールを複数選んでも
    assert r["pooled_selected"]["disc"]["n"] <= int((strong & (yr <= 2021)).sum())      # 全体の走は重ねて数えない


def test_graded_pool_upper_classes():
    d = _frame(60)
    d["cls"] = 4                                                                       # 3 勝クラス（寄せ先の母集団）
    g = d[d["race_id"].str.endswith("0")].copy()
    g["cls"] = 6
    g["rkey"] = "テスト記念"
    d.loc[g.index, ["cls", "rkey"]] = g[["cls", "rkey"]]
    d = _prep(d)
    P = TH.pools(d, "graded")
    assert set(P["target"][P["target"] != ""]) == {"テスト記念|東京芝1600"}
    assert (P["levels"][0][1] != "").sum() == len(d)                                     # 3 勝以上はすべて寄せ先に入る（全馬場）
    r = TH.search(d, "graded", log=lambda *x: None)
    assert "テスト記念|東京芝1600" in r["units"]


def test_layer_restricts_rows():
    d = _prep(_frame(30))
    r_all = TH.search(d, "course", "全馬", log=lambda *x: None)
    r_fav = TH.search(d, "course", "1 番人気", log=lambda *x: None)
    u = "東京芝1600|1勝・2勝"
    n_fav = int(((d["pop_final"] <= 1) & (d["yr"] <= 2021)).sum())
    assert r_fav["units"][u]["n_disc"] == n_fav < r_all["units"][u]["n_disc"]       # 層は人気で切る（人気は探す切り口ではない）


def test_in_layer_and_odds_expected():
    pop = np.array([1, 2, 3, np.nan])
    odds = np.array([1.8, 2.6, 5.0, np.nan])
    assert TH.in_layer(None, pop, odds).all()
    assert TH.in_layer(("pop", 2), pop, odds).tolist() == [True, True, False, False]
    assert TH.in_layer(("odds", 2.5), pop, odds).tolist() == [True, False, False, False]
    d = _frame(30)
    d["odds_final"] = np.where(d["p_fin"] == 1, 1.8, 20.0)                               # 強い馬はオッズも低い
    fit = np.ones(len(d), bool)
    e = TH.expected(d, fit)
    assert e[d["p_fin"].to_numpy() == 1].mean() > 0.9 > 0.3 > e[d["p_fin"].to_numpy() != 1].mean()   # オッズの帯で見込むので、強さは見込みに入る


def test_repro_and_uplift_disjoint():
    Y = np.array([[1, 1, 0, 0, 1, 2.0, 1.5, 0.5, 0.25]] * 4 + [[1, 0, 0, 0, 1, 0, 0, 0.5, 0.25]] * 4, float)
    m_rule = np.array([True] * 4 + [False] * 4)
    base = np.ones(8, bool)
    pre_in = [np.array([True] * 8), np.array([True, True, False, False] * 2)]
    r = TH._repro(Y, m_rule, base, pre_in)
    assert r["rule"]["place"] == 1.0 and abs(r["uplift"]["pt"] - 1.0) < 1e-9        # ルール対、層のほかの馬（重ならない 2 群）
    assert abs(r["base"]["n"] - 6.0) < 1e-9                                            # 20 通りの平均（ここでは 2 通り）
