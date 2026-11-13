"""父と母側の系統の影響（sire_dam_influence）: 適性の差の縮小・距離の傾き・系統の補完・Shapley の分け方・小系統の距離配置。"""

import numpy as np
import pandas as pd
import pytest

from src.research.pedigree import sire_dam_influence as S


def _runs():
    """h1: 芝で好走・ダートで凡走（芝 3 走・ダ 3 走）。h2: 芝しか走らない。h3: 距離が延びるほど好走。全員 2 勝クラスを 1 回は走る"""
    rows = []

    def add(h, i, sf, dist, pos, grade="2勝", cond="良"):
        rows.append({"race_id": f"r{h}{i}", "date": f"2024-01-{i + 1:02d}", "horse_id": h, "surface": sf, "distance": dist, "direction": "右",
                     "track_condition": cond, "venue": "東京", "grade": grade, "finish_position": pos, "field_size": 11,
                     "passing_order": "3-3", "last_3f": 34.0 + pos / 10, "sex_age": "牡4"})
    for i in range(3):
        add("h1", i, "芝", 1600, 1)
        add("h1", 10 + i, "ダート", 1600, 10)
        add("h2", i, "芝", 1600, 5)
    for i, (d, p) in enumerate(((1200, 11), (1600, 8), (2000, 4), (2400, 1))):
        add("h3", i, "芝", d, p)
    add("h4", 0, "芝", 1600, 1, grade="1勝")   # 2 勝クラス以上を走っていない → 外れる
    return pd.DataFrame(rows)


def test_prepare_runs_keeps_only_horses_that_ran_2win_class():
    r = S.prepare_runs(_runs())
    assert set(r.horse_id) == {"h1", "h2", "h3"}
    assert r.pct.between(0, 1).all()


def test_contrast_needs_both_conditions():
    r = S.prepare_runs(_runs())
    c = S.contrast(r, r.surface == "芝", r.surface == "ダート", "x", min_n=2)
    assert list(c.index) == ["h1"]                     # h2 は芝しか走らないので外れる
    raw = r[(r.horse_id == "h1") & (r.surface == "芝")].adj.mean() - r[(r.horse_id == "h1") & (r.surface == "ダート")].adj.mean()
    assert c["h1"] == pytest.approx(raw)                # 1 頭だけなら全体の平均 = 本人で、縮めても変わらない


def test_shrink_toward_mean_by_precision():
    d = pd.Series({"a": 0.4, "b": 0.4, "c": -0.2, "d": 0.0})
    v = pd.Series({"a": 0.001, "b": 0.5, "c": 0.01, "d": 0.01})
    s = S.shrink(d, v)
    m = d.mean()
    assert abs(s["a"] - 0.4) < abs(s["b"] - 0.4)        # 精度の高い a はほとんど動かない
    assert min(0.4, m) <= s["b"] <= max(0.4, m)          # b は平均の側へ寄る


def test_closing_rank_uses_all_runners_before_filtering():
    r = S.prepare_runs(_runs())
    # h4 は対象外だが、上がりの順位は全頭の中で付けている（h1 の芝のレースは単独なので 1.0）
    assert r.closing.between(0, 1).all()


def test_slope_sign_positive_for_horse_better_at_longer():
    r = S.prepare_runs(_runs())
    s = S.slope(r, r.surface == "芝", "x", min_n=4)
    assert s["h3"] > 0


def test_line_classifier_table_then_paternal_then_sons():
    lab = {"A": ("ディープ系", "サンデー系", "日本型サンデー系"), "B": ("Tサンデー系", "サンデー系", "日本型サンデー系"),
           "C": ("ダンチヒ系", "ノーザンダンサー系", "欧州型ノーザンダンサー系")}
    # SS は表に無い。息子 A・B は小系統が割れる → 大系統の名前。X は表に無いが父 C が表にある → C の系統。
    # Old は息子 SS（推定）から 2 回目に推定される（息子 1 頭なので小系統は大系統の名前）。
    # Y は表に無く、父 SS も表に無い → さかのぼった先の SS の推定を使う。
    # Z の息子は D1・D2（同じ小系統で 2 頭一致）→ その小系統。W の息子は D3 だけ → 大系統の名前
    lab.update({"D1": ("ダンチヒ系", "ノーザンダンサー系", "欧"), "D2": ("ダンチヒ系", "ノーザンダンサー系", "欧"), "D3": ("ダンチヒ系", "ノーザンダンサー系", "欧")})
    sire_of = {"A": "SS", "B": "SS", "X": "C", "SS": "Old", "Y": "SS", "D1": "Z", "D2": "Z", "D3": "W"}
    cl = S.line_classifier(lab, sire_of)
    assert cl("A") == (lab["A"], "表")
    assert cl("X") == (lab["C"], "父系をさかのぼり")
    assert cl("SS") == (("サンデー系", "サンデー系", "日本型サンデー系"), "息子から推定（2 頭）")   # 系統の分かる息子は A・B の 2 頭（Y は分からないので数えない）
    assert cl("Old")[0][0] == "サンデー系" and cl("Old")[1].startswith("息子から推定")
    assert cl("Y") == (("サンデー系", "サンデー系", "日本型サンデー系"), "父系の祖先を息子から推定（2 頭）")
    assert cl("Z")[0][0] == "ダンチヒ系"
    assert cl("W")[0][0] == "ノーザンダンサー系"
    assert cl("nobody") == (None, None)


def test_sire_edges_from_positions():
    anc = {(1, 0): {"horse_id": "s"}, (1, 1): {"horse_id": "d"}, (2, 0): {"horse_id": "ss"}, (2, 2): {"horse_id": "ds"}}
    e = S.sire_edges({"h": anc})
    assert e == {"h": "s", "s": "ss", "d": "ds"}


def test_shapley_sums_to_full_r2_and_finds_the_true_factor():
    rng = np.random.default_rng(0)
    n = 1200
    F = pd.DataFrame({"父": rng.choice(list("abcdef"), n), "母父": rng.choice(list("uvwxyz"), n)})
    eff = dict(zip("abcdef", [-1.0, -0.5, 0, 0.3, 0.6, 1.0]))
    y = F["父"].map(eff).to_numpy() + rng.normal(0, 1, n)
    y = (y - y.mean()) / y.std()
    folds = S._folds(n, 5, 0)
    sh, full, alone = S.shapley(F, y, folds)
    assert sum(sh.values()) == pytest.approx(full, abs=1e-9)
    assert sh["父"] > 0.15 and sh["母父"] < 0.02
    res = S.influence(pd.Series(y), F, None, nperm=5, nboot=0, rng=rng)
    assert res["shapley"]["父"] > res["null95"]["父"]
    assert res["null95"]["母父"] < 0.03                  # 並べ替えの帰無はほぼ 0


def test_line_map_nearest_and_dendrogram():
    rng = np.random.default_rng(1)
    rows, lines = [], []
    centers = {"L1": [1, 1], "L2": [1.1, 0.9], "L3": [-1, -1], "L4": [-1.1, -0.8]}
    for line, c in centers.items():
        for _ in range(60):
            rows.append(np.array(c) + rng.normal(0, 0.3, 2))
            lines.append(line)
    df = pd.DataFrame(rows, columns=["TS", "ギア"])
    F = pd.DataFrame({"父": lines})
    e = S.line_map(df, F, "父", ["TS", "ギア"], rng, nboot=5, min_n=30)
    assert sorted(e["lines"]) == ["L1", "L2", "L3", "L4"]
    assert e["near"]["L1"][0][0] == "L2" and e["near"]["L3"][0][0] == "L4"
    assert sorted(e["dendro_order"]) == sorted(e["lines"]) and len(e["linkage"]) == 3
    assert set(e["xy"]) == set(e["lines"])
