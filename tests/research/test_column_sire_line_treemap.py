"""めぐコラム『種牡馬系統による適性の違いはあるのか』の集計（src/research/columns/sire_line_treemap.py）。"""

from __future__ import annotations

import math

import numpy as np

from src.research.columns import sire_line_treemap as t
from src.scripts.scraping.trace_sire_line_origins import compact_record


def _rec(name, chain, lines):
    """出力（大きな版）の 1 頭分を作り、記事が読む小さな版にする。"""
    return compact_record({"name": name, "chain": chain, "lines": lines, "origin_line": lines[-1], "root": chain[-1][:2]})


ORIGIN = {
    # ヴァーミリアン型: 自身の表記が Raise a Native系、父は Mr. Prospector系（系統名が飛び飛びに出る）
    "V": _rec("ヴァーミリアン", [["V", "ヴァーミリアン", "Raise a Native系"], ["E", "エルコンドルパサー", "Mr. Prospector系"],
                              ["K", "Kingmambo (米)", "Mr. Prospector系"], ["M", "Mr. Prospector (米)", "Mr. Prospector系"],
                              ["R", "Raise a Native (米)", "Raise a Native系"]],
              ["Raise a Native系", "Mr. Prospector系", "Raise a Native系"]),
    "KK": _rec("キングカメハメハ", [["KK", "キングカメハメハ", "Mr. Prospector系"], ["K", "Kingmambo (米)", "Mr. Prospector系"],
                               ["M", "Mr. Prospector (米)", "Mr. Prospector系"], ["R", "Raise a Native (米)", "Raise a Native系"]],
               ["Mr. Prospector系", "Raise a Native系"]),
    "LC": _rec("ロードカナロア", [["LC", "ロードカナロア", "Mr. Prospector系"], ["KK", "キングカメハメハ", "Mr. Prospector系"],
                              ["K", "Kingmambo (米)", "Mr. Prospector系"], ["M", "Mr. Prospector (米)", "Mr. Prospector系"],
                              ["R", "Raise a Native (米)", "Raise a Native系"]],
               ["Mr. Prospector系", "Raise a Native系"]),
}


def test_line_path_skips_repeated_names():
    assert t.line_path(["Raise a Native系", "Mr. Prospector系", "Raise a Native系", "Native Dancer系"]) == \
        ["Native Dancer系", "Raise a Native系", "Mr. Prospector系"]
    assert t.line_path(["Halo系", None, "Turn-to系"]) == ["Turn-to系", "Halo系"]


def test_sire_path_branches_below_the_line_founder():
    assert ORIGIN["V"]["chain"] == [["E", "エルコンドルパサー"], ["K", "Kingmambo (米)"], ["M", "Mr. Prospector (米)"]]
    p = t.sire_path("V", ORIGIN["V"])
    assert [n for _, n, _ in p] == ["Raise a Native系", "Mr. Prospector系", "Kingmambo系", "エルコンドルパサー系", "ヴァーミリアン"]
    assert [k for _, _, k in p] == ["line", "line", "branch", "branch", "sire"]
    assert t.strip_country("Kingmambo (米)") == "Kingmambo"


def test_build_tree_puts_a_sire_who_is_also_an_ancestor_inside_his_branch():
    root = t.build_tree(ORIGIN, {"V", "KK", "LC"})
    by = {n.key: n for n in root.walk()}
    kk_branch = by["H:KK"]
    assert [c.name for c in kk_branch.children] == ["キングカメハメハ（自身）", "ロードカナロア"]
    km = by["H:K"]
    assert {c.key for c in km.children} == {"H:E", "H:KK"}
    leaves = sorted(s for n in root.walk() if n.leaf for s in n.sires)
    assert leaves == ["KK", "LC", "V"]                       # どの種牡馬も 1 回だけ


def _st(n: float, mean: float) -> np.ndarray:
    """全成分に、1 頭 1 走ずつの n 走・平均 mean の十分統計量を入れる。"""
    s = np.zeros((len(t.COMPONENTS), 6))
    s[:, 0] = n; s[:, 1] = n * mean; s[:, 2] = n * mean * mean
    s[:, 3] = n * mean * mean; s[:, 4] = n * mean; s[:, 5] = n
    return s


def test_prune_pools_small_sires_and_collapses_single_children():
    root = t.build_tree(ORIGIN, {"V", "KK", "LC"})
    st = {"V": _st(30, 0.5), "KK": _st(5000, 0.4), "LC": _st(40, 0.3)}
    hs = {"V": 5, "KK": 300, "LC": 12}
    t.aggregate(root, st, hs)
    names = {k: v["name"] for k, v in ORIGIN.items()}
    pr = t.prune(root, names)
    t.aggregate(pr, st, hs)
    leaves = [n for n in pr.walk() if n.leaf]
    # LC（12 頭）は MIN_HORSES 以上なので KK の branch の中に残る。V（5 頭）の枝は MIN_HORSES 未満なので葉になる
    assert sum(int(n.stats[0][0]) for n in leaves) == 5070
    assert all(len(n.children) != 1 for n in pr.walk() if n.kind != "root")


def test_metric_raw_mean_and_surface_weighted_difference():
    s = np.zeros((len(t.COMPONENTS), 6))
    i = t.COMPONENTS.index("dirt")
    s[i] = [4, 2, 2, 2, 2, 4]                                  # 2 頭 × 2 走、[1,1] と [0,0] → 平均 0.5
    m, v, n = t.metric_raw(s, t.METRICS[0])
    assert m == 0.5 and n == 4 and v > 0
    # 道悪: 芝だけに十分な出走があり、ダートは MIN_PART 未満 → 芝の差だけ
    for c, mean in [("heavy_t", 0.1), ("good_t", 0.0), ("heavy_d", 1.0), ("good_d", 0.0)]:
        k = 100 if c.endswith("_t") else 5
        s[t.COMPONENTS.index(c)] = [k, k * mean, k * mean * mean, k * mean * mean, k * mean, k]
    going = next(m for m in t.METRICS if m.key == "going")
    d, v, n = t.metric_raw(s, going)
    assert math.isclose(d, 0.1) and n == 100


def test_extra_metric_needs_min_runs_else_no_value():
    s = np.zeros((len(t.COMPONENTS), 6))
    ts = next(m for m in t.EXTRA_METRICS if m.key == "ts")
    i = t.COMPONENTS.index("ax_ts")
    s[i] = [3, 150, 7500, 7500, 150, 3]                         # 1 走だけの馬が 3 頭 → 分散が小さく出るが、値なし扱い（親へ寄せる）
    m, v, n = t.metric_raw(s, ts)
    assert m == 50 and n == 3 and math.isinf(v)
    s[i] = [30, 1500, 75000 + 30 * 4, 75000, 1500, 30]
    assert math.isfinite(t.metric_raw(s, ts)[1])


def test_label_pools_names_members_instead_of_count():
    root = t.Node("R", "全", "root", children=[
        t.Node("a+pool", "ほか 2 頭", "pool", sires=["x", "y"]),
        t.Node("b", "Foo系（3 頭）", "pool", sires=["p", "q", "r"])])
    names = {"x": "甲", "y": "乙", "p": "A", "q": "B", "r": "C"}
    t.label_pools(root, names, {"x": 3, "y": 7, "p": 1, "q": 1, "r": 2})
    a, b = root.children
    assert a.name == "乙・甲" and a.members == [("乙", 7), ("甲", 3)]
    assert b.name == "Foo系（C・A・B）"
    many = t.Node("c+pool", "ほか 5 頭", "pool", sires=list("abcde"))
    t.label_pools(many, {k: k.upper() for k in "abcde"}, {k: 1 for k in "abcde"})
    assert many.name == "A・B・C ほか2頭" and len(many.members) == 5


def test_win_quantile_interpolates_within_bands():
    n = len(t.WIN_BINS)
    one = np.zeros(n); one[0] = 1.0                             # 全部が最初の帯 → 帯の中を一様に
    assert math.isclose(t.win_quantile(one, 0.5), (t.WIN_EDGES[0] + t.WIN_EDGES[1]) / 2)
    flat = np.full(n, 1.0 / n)                                  # 全帯が同じ割合 → 幅は全帯の幅の 90%
    w = t.win_quantile(flat, 0.95) - t.win_quantile(flat, 0.05)
    assert math.isclose(w, 0.9 * (t.WIN_EDGES[-1] - t.WIN_EDGES[0]) , rel_tol=0.3) and w > one.sum() * 300


def test_shrink_moves_toward_parent_by_precision():
    post, var = t.shrink(1.0, v=1.0, prior=0.0, prior_var=0.0, tau2=1.0)
    assert math.isclose(post, 0.5) and math.isclose(var, 0.5)
    post, _ = t.shrink(1.0, v=0.01, prior=0.0, prior_var=0.0, tau2=1.0)
    assert post > 0.98                                         # 標本が多いほど自身の値
    post, var = t.shrink(math.nan, v=math.inf, prior=0.3, prior_var=0.1, tau2=0.2)
    assert post == 0.3 and math.isclose(var, 0.3)               # 値が無ければ親のまま
    # τ² の推定: ずれの 2 乗の平均 − 標本の分散
    assert math.isclose(t.estimate_tau2([(1, 0.5, 0, 0), (-1, 0.5, 0, 0), (1, 0.5, 0, 0)]), 0.5)
    assert t.estimate_tau2([(0.1, 1.0, 0, 0)] * 5) < 1e-9       # ばらつきが標本の分散より小さい → ほぼ 0


def test_eta2_and_permutation_detect_group_difference():
    rng = np.random.default_rng(1)
    g = np.repeat(["a", "b", "c"], 20)
    x = np.where(g == "a", 1.0, 0.0) + rng.normal(0, 0.3, 60)
    w = np.ones(60)
    e, e0, p = t.perm_test(x, w, g, n_perm=500)
    assert e > 0.5 and e0 < 0.1 and p < 0.01
    x0 = rng.normal(0, 1, 60)
    _, _, p0 = t.perm_test(x0, w, g, n_perm=500)
    assert p0 > 0.01
    assert t.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_pc1_orientation_and_child_order():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    mat = np.column_stack([z, -z + rng.normal(0, 0.1, 200), rng.normal(size=200)])
    load, mu, sd, share = t.pc1(mat, np.array([True, True, False]))
    assert load[0] > 0 and load[1] < 0 and load[2] == 0 and share > 0.5
    root = t.Node("ROOT", "r", "root", children=[t.Node("a", "a", "sire"), t.Node("b", "b", "sire"), t.Node("c", "c", "sire")])
    for nd, s in zip(root.children, [0.5, -1.0, 0.1]):
        nd.score = s
    t.order_children(root)
    assert [c.key for c in root.children] == ["b", "c", "a"]
