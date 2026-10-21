import numpy as np
import pandas as pd

from src.pipeline.inference import class_benchmark as cb
from src.scripts.data import class_benchmark_build as bld


def test_cell_parsing_handles_fullwidth_missing_class_and_steeplechase():
    assert cb.cell_of("サラ系３歳 未勝利", "未勝利", "3歳未勝利") == ("3歳", "未勝利")
    assert cb.cell_of("サラ系４歳以上 １勝クラス", "1勝", "4歳以上1勝クラス") == ("4歳以上", "1勝")
    assert cb.cell_of("サラ系２歳 新馬", "新馬", "2歳新馬") == ("2歳", "新馬")
    assert cb.cell_of("", "G1", "天皇賞(秋)") == (None, "G1")                       # クラス欄が空のグレード戦
    assert cb.cell_of("", "3勝", "是政ステークス(3勝)") == (None, "3勝")
    assert cb.cell_of("サラ系３歳以上 ３勝クラス", "3勝", "ストークステークス(3勝)") == ("3歳以上", "3勝")
    assert cb.cell_of("未勝利", "未勝利", "障害3歳以上未勝利") == (None, None)       # 障害は対象外
    assert cb.label_of("3歳以上", "1勝") == "3歳以上1勝クラス" and cb.label_of("2歳", "新馬") == "2歳新馬" and cb.label_of(None, "G1") == "G1"


def test_keys_and_distance_band():
    assert [cb.dist_band(d) for d in (1200, 1400, 1800, 2400, None)] == ["短", "中短", "中", "長", None]
    assert cb.keys_for("2歳", "新馬", "芝", "中短") == ["2歳|新馬|芝|中短", "2歳|新馬|芝", "2歳|新馬", "*|新馬"]
    assert cb.keys_for(None, "G1", None, None) == ["*|G1"]


def _table():
    return {"period": "2023-2025", "cells": {
        "3歳|未勝利": {"n": 100, "median": 40.0, "p25": 35.0, "p75": 45.0},
        "3歳|未勝利|ダ": {"n": 60, "median": 43.0, "p25": 38.0, "p75": 47.0, "level": "surface"},
        "*|G1": {"n": 40, "median": 65.0, "p25": 60.0, "p75": 70.0, "level": "tier"},
    }}


def test_lookup_prefers_the_most_specific_level_and_falls_back():
    t = _table()
    dirt = cb.lookup("サラ系３歳 未勝利", "未勝利", "3歳未勝利", "ダート", 1200, t)
    assert dirt["line"] == 38.0 and dirt["median"] == 43.0 and dirt["surface"] == "ダ" and dirt["label"] == "3歳未勝利" and dirt["level"] == "surface"
    turf = cb.lookup("サラ系３歳 未勝利", "未勝利", "3歳未勝利", "芝", 1600, t)           # 芝の表は無い → 年齢×クラス
    assert turf["line"] == 35.0 and turf["level"] == "cell"
    g1 = cb.lookup("", "G1", "天皇賞(秋)", "芝", 2000, t)                                  # 年齢が読めない → クラスのみ
    assert g1["line"] == 60.0 and g1["label"] == "G1"
    assert cb.lookup("サラ系２歳 新馬", "新馬", "2歳新馬", "芝", 1600, t) is None         # 表に無い
    assert cb.lookup("", "未勝利", "障害3歳以上未勝利", "芝", 3000, t) is None
    assert cb.lookup("サラ系３歳 未勝利", "未勝利", "3歳未勝利", "芝", 1600, {"cells": {}}) is None


def _w(n, mu, age="3歳", tier="未勝利", surf="芝", band="中短", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": age, "tier": tier, "surf": surf, "band": band, "idx": rng.normal(mu, 3, n)})


def test_build_cells_splits_only_when_the_difference_is_large_enough_and_n_is_enough():
    w = pd.concat([_w(80, 40, surf="芝", seed=1), _w(80, 46, surf="ダ", seed=2), _w(10, 60, surf="ダ", band="長", seed=3)])
    cells = bld.build_cells(w)
    assert cells["3歳|未勝利"]["n"] == 170 and cells["*|未勝利"]["n"] == 170
    assert "3歳|未勝利|芝" in cells and "3歳|未勝利|ダ" in cells                 # 中央値が約6点違う → 分ける
    near = pd.concat([_w(80, 40, surf="芝", seed=1), _w(80, 40.5, surf="ダ", seed=2)])
    assert not any(k.count("|") == 2 for k in bld.build_cells(near))             # 差が2点未満 → 分けない
    assert not any(k.count("|") == 3 for k in cells)                              # 距離帯は n<30 や差が小さければ作らない
    assert bld.build_cells(_w(20, 40)) == {}                                      # n<30 は表に載せない


def test_prior_max_uses_only_earlier_valid_runs():
    df = pd.DataFrame({"horse_id": ["A"] * 4 + ["B"], "race_id": list("1234") + ["5"], "date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-01-05"],
                       "idx": [40.0, 60.0, 50.0, 55.0, 70.0], "ok": [True, True, False, True, True]})
    pm = bld.prior_max(df)
    assert pd.isna(pm[0]) and pm[1] == 40.0 and pm[3] == 60.0 and pd.isna(pm[4])   # 3戦目(ok=False)は後の走の材料にならない


def test_cluster_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(200):
        for h in range(10):
            above = h < 5
            rows.append({"race_id": f"r{r}", "above": above, "win": int(rng.random() < (0.18 if above else 0.02)), "exp_win": 0.1})
    ci = bld.cluster_bootstrap_diff(pd.DataFrame(rows), "above", "win", "exp_win")
    assert ci[0] > 0.5 and ci[1] > ci[0]
