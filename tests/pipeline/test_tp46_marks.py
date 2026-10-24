import numpy as np
import pandas as pd

from src.scripts.data import tp46_marks as t


def _pay():
    return {"tansho": {1: 500.0}, "fukusho": {1: 150.0, 2: 200.0, 3: 300.0}, "umaren": {frozenset((1, 2)): 800.0},
            "wide": {frozenset((1, 2)): 300.0, frozenset((1, 3)): 400.0, frozenset((2, 3)): 600.0},
            "sanrenpuku": {frozenset((1, 2, 3)): 2000.0}}


def test_race_metrics_counts_hits_and_returns():
    mk = {1: "honmei", 2: "pair", 3: "anchor", 4: "anchor", 5: "none"}
    r = t.race_metrics(mk, _pay())
    assert r["honmei_win"] == 1.0 and r["honmei_top3"] == 1.0 and r["umaren_hit"] == 1.0 and r["top2cov"] == 1.0
    assert r["cover3"] == 1.0 and r["all3"] == 1.0 and r["n_marks"] == 4.0 and r["n_maybe"] == 2.0
    assert r["roi_tan"] == 5.0 and r["roi_umaren"] == 8.0 and r["roi_wide"] == 3.0
    # ◎+△2頭のボックス=3頭から3頭を選ぶ1点だけ: {1,3,4}は不的中、ただし箱は C(3,3)=1 点
    assert r["roi_trio_box"] == 0.0
    assert r["roi_trio_nagashi"] == (2000.0 / 100.0) / 2          # {1,2,3} が的中、{1,2,4} は外れ（2点で均等割り）


def test_race_metrics_without_honmei_is_nan_for_honmei_columns():
    r = t.race_metrics({1: "none", 2: "none"}, _pay())
    assert np.isnan(r["honmei_win"]) and r["n_marks"] == 0.0 and r["win_cover"] == 0.0


def test_p0_policy_equals_production_default(monkeypatch):
    monkeypatch.setenv("KEIBA_MARK_RULE", "v1")                 # P0 は v1（従来）。本番の既定が v2 でもこの比較は v1 で行う
    g = pd.DataFrame({"horse_number": range(1, 11), "p_win": [0.3, 0.2, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02],
                      "p_top2": [0.5, 0.4, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04],
                      "p_top3": [0.7, 0.55, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.06], "win_odds": [3.0] * 10})
    es = t.entries_of(g)
    from src.utils.race_probabilities import assign_mece_marks

    ref = [dict(e) for e in es]
    assign_mece_marks(ref)
    assert t.marks_of(es, "P0_current") == {e["horse_number"]: e["mark_type"] for e in ref}


def test_paired_boot_difference_sign():
    a, b = np.array([1.0, 1.0, 1.0, np.nan]), np.array([0.0, 0.5, 0.5, 1.0])
    m, lo, hi = t.paired_boot(a, b, n_boot=200)
    assert lo <= m <= hi and m > 0
