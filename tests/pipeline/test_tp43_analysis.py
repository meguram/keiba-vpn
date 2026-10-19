import numpy as np
import pandas as pd

from src.scripts.data import tp43_analysis as an
from src.scripts.data import tp43_marks as tm


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.choice([0.0, 1.5, 4.0, 400.0], p=[0.7, 0.2, 0.09, 0.01], size=(n, len(tm.COMBOS))).astype(np.float32)
    rf = pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "year": np.repeat([2022, 2023, 2024, 2025], n // 4),
                       **{k: rng.normal(size=n) for k in tm.DEFS}})
    return rf, R


def test_cap_is_common_to_all_columns_and_keeps_small_returns():
    rf, R = _toy()
    Rc, caps = an.cap_returns(R)
    assert Rc.max() == an.CAP_X and (caps == an.CAP_X).all()
    assert np.array_equal(Rc[R <= an.CAP_X], R[R <= an.CAP_X])


def test_split_masks_and_global_best_use_only_design_years():
    rf, R = _toy()
    sm = an.split_masks(rf)
    assert sm["design"].sum() == 200 and sm["confirm"].sum() == 100 and not (sm["design"] & sm["test"]).any()
    Rc, _ = an.cap_returns(R)
    gb = an.global_best(Rc, sm)
    assert set(gb) == set(tm.SCHEME_NAMES) and all(m in tm.MARKS for m in gb.values())
    R2 = Rc.copy()
    R2[sm["test"]] = 0.0                                   # 検証年を変えても設計年の選択は変わらない
    assert an.global_best(R2, sm) == gb


def test_switch_value_rows_and_m6b_uses_design_selection_only():
    rf, R = _toy()
    sm = an.split_masks(rf)
    Rc, _ = an.cap_returns(R)
    bk = tm.buckets(rf, tm.cutoffs(rf))
    sel = an.select_m6b(Rc, bk, sm)
    gb = an.global_best(Rc, sm)
    rows = an.switch_value(R, Rc, bk, sm, sel, gb, n_boot=100)
    assert len(rows) == len(tm.DEFS) * len(tm.SCHEME_NAMES)
    R3 = Rc.copy()
    R3[sm["test"]] = 5.0
    assert an.select_m6b(R3, bk, sm) == sel                 # 選択は設計年だけで決まる


def test_strategy_matrix_counts_and_zero_for_no_bet():
    rf, R = _toy()
    sm = an.split_masks(rf)
    Rc, _ = an.cap_returns(R)
    bk = tm.buckets(rf, tm.cutoffs(rf))
    names, X = tm.strategy_matrix(Rc, bk, sm["test"])
    assert X.shape == (int(sm["test"].sum()), len(names)) and len(names) == 3080
    j = names.index(("cond_ent_荒", "M1_pwin", "tan1"))
    not_rough = bk["ent"].to_numpy()[sm["test"]] != 2
    assert (X[not_rough, j] == 0).all()
