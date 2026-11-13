"""重賞のルールベース予想家: 切り口の目録の検査（市場の情報・その走の結果を使わない、水準の作り方）。"""

import re

import numpy as np
import pandas as pd

from src.research.graded import catalog as C


def test_allowed_columns_have_no_market_or_result():
    bad = {c for c in C.ALLOWED_COLUMNS if c in C.FORBIDDEN_EXACT or re.search(C.FORBIDDEN_PATTERN, c)}
    assert not bad, f"使ってはいけない列が切り口に入っている: {sorted(bad)}"


def test_every_type_lists_its_columns_and_unique_ids():
    ids = [t.id for t in C.TYPES]
    assert len(ids) == len(set(ids))
    for t in C.TYPES:
        assert t.cols, t.id
        assert t.family in C.FAMILIES, t.id


def test_catalog_size_matches_decision():
    main = [t for t in C.TYPES if not t.optional]
    assert 110 <= len(main) <= 130          # A〜L ＋ 組み合わせ 15 で約 120 型（2026-11-13 確定）
    assert sum(t.family == "X" for t in main) == 15
    assert all(t.family == "M" for t in C.TYPES if t.optional)


def test_cut_is_right_closed_and_unknown():
    s = pd.Series([1, 2, 3, 4, 5, 6, 10, np.nan])
    out = C.fin_bucket(s).tolist()
    assert out == ["1着", "2-3着", "2-3着", "4-5着", "4-5着", "6-9着", "10着以下", C.UNKNOWN]


def _toy(n=6):
    rng = np.random.default_rng(0)
    cols = sorted(C.ALLOWED_COLUMNS | {"year", "grade", "prev_region"})
    d = pd.DataFrame({c: rng.integers(0, 5, n).astype(float) for c in cols})
    for c in ("sex",):
        d[c] = ["牡", "牝", "セ", "牡", "牝", "牡"][:n]
    for c in ("surf", "p_surf"):
        d[c] = "芝"
    for c in ("track", "p_track"):
        d[c] = "良"
    for c in ("io", "p_io"):
        d[c] = "外"
    for c in ("f_large", "f_small", "mf_large", "mf_small", "f_name", "jockey_name", "rkey", "p_rkey", "tozai", "venue", "vc", "p_vc",
              "direction", "p_direction", "ok_eval", "p_pace2", "prev_region", "jockey_id"):
        d[c] = "x"
    d["grade"] = "G2"
    d["year"] = [2016, 2017, 2018, 2019, 2020, 2025][:n]
    d["p_dt"] = pd.Timestamp("2020-01-01")
    d["dt"] = pd.Timestamp("2020-02-01")
    d["jw_max"] = False
    for c in ("p_mixed", "p_fillies", "j_foreign", "j_short"):
        d[c] = False
    return d


def test_levels_use_only_allowed_columns():
    """ALLOWED_COLUMNS（と年・前走の地域）だけの表から、全型の水準が作れる（ほかの列を読んでいない）。"""
    d = _toy()
    L = C.levels(d)
    assert list(L.columns) == [t.id for t in C.TYPES]
    assert len(L) == len(d)


def test_start_year_and_prev_foreign():
    d = _toy()
    d["prev_region"] = ["中央", "中央", "中央", "海外", "地方", "中央"]
    d["p_fin"] = 1.0
    L = C.levels(d)
    a01 = L["A01"].tolist()
    assert a01[0] == C.UNKNOWN                     # 2016 年は前走系を数えない
    assert a01[3] == C.PREV_FOREIGN and a01[4] == C.PREV_FOREIGN
    assert a01[5] == "1着"
    assert L["F01"].tolist()[1] == C.UNKNOWN       # 経歴系は 2018 年〜


def test_each_type_reads_only_its_declared_columns():
    """型ごとに、宣言した列（と率の帯の基準に使う年・グレード）だけの表から水準が作れる。"""
    d = _toy()
    for t in C.TYPES:
        sub = d[list(dict.fromkeys(list(t.cols) + ["year", "grade"]))]
        out = t.fn(sub)
        assert len(out) == len(d), t.id


def test_bool_columns_read_as_strings():
    """parquet の保存で bool が "True"/"False" の文字列になっても、真偽として読む。"""
    d = _toy()
    d["age"] = 3.0
    d["p_mixed"] = ["True", "False", "True", "False", "True", "False"]
    d["year"] = 2020
    out = C.BY_ID["B08"].fn(d).tolist()
    assert out == ["前走 古馬との混合", "前走 3歳限定"] * 3
