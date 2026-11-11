"""事前登録した判定（persona1_confirm.verdict）の段階が、登録したとおりに動くことの確認。"""
from src.scripts.data import persona1_confirm as C


def test_small_sample_is_not_judged():
    assert C.verdict({2025: 1.3}, 1.3, 1.1, 0.99, 299, None, False).startswith("判定不能")


def test_year_below_floor_fails_even_if_pooled_is_high():
    assert C.verdict({2025: 0.7, 2026: 2.0}, 1.4, 1.1, 0.99, 600, 1.2, True).startswith("不合格")


def test_strong_requires_lower_bound_and_ex_top3_for_c2():
    assert C.verdict({2025: 1.2, 2026: 1.3}, 1.25, 1.02, 0.99, 600, 1.05, True).startswith("強い合格")
    # 上位 3 的中を除くと 100% 未満なら、下限が 100% 超でも強い合格にならない（弱い合格に落ちる）
    assert C.verdict({2025: 1.2, 2026: 1.3}, 1.25, 1.02, 0.99, 600, 0.9, True).startswith("弱い合格")
    # C1 は上位 3 除く条件なし（初版の条件のとおり）
    assert C.verdict({2025: 1.2, 2026: 1.3}, 1.25, 1.02, 0.99, 600, None, False).startswith("強い合格")


def test_weak_needs_posterior_probability_90():
    assert C.verdict({2025: 1.1, 2026: 1.1}, 1.1, 0.9, 0.93, 600, 1.0, True).startswith("弱い合格")
    assert C.verdict({2025: 1.1, 2026: 1.1}, 1.1, 0.9, 0.85, 600, 1.0, True) == "不合格"
