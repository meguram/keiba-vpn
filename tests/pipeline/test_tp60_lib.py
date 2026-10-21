"""tp-60: 血統別の回収率の集計・ブートストラップ・reality check・深い血量の回帰の部品を、小さな架空データで確かめる。"""

import numpy as np
import pandas as pd

from src.scripts.data import tp60_deep as D
from src.scripts.data import tp60_embfeat as EF
from src.scripts.data import tp60_lib as L
from src.scripts.data import tp60_rules as R


def _frame(n_race=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_race):
        n = 8
        win_idx = rng.integers(0, n)
        for h in range(n):
            odds = float(rng.uniform(2, 30))
            win = float(h == win_idx)
            rows.append({"race_id": f"R{r:04d}", "horse_id": f"H{r}_{h}", "year": 2022 + r % 3, "win": win, "top2": win, "top3": win,
                         "ret_win": win * odds * 100, "ret_place": win * 150.0, "q_win": 1.0 / n, "p_win": 1.0 / n, "key": "A" if h % 2 else "B", "cond": "all"})
    return pd.DataFrame(rows)


def test_collapse_and_agg_cells_roi_units():
    m = _frame()
    c = L.collapse(m.rename(columns={"key": "sire_id"}), "sire_id", "cond")
    r = L.agg_cells(c)
    assert set(r["key"]) == {"A", "B"}
    row = r[r["key"] == "A"].iloc[0]
    assert abs(row["roi_win"] - m[m["key"] == "A"]["ret_win"].mean() / 100.0) < 1e-9
    assert row["horses"] == (m["key"] == "A").sum()


def test_eb_shrink_pulls_toward_prior():
    out = L.eb_shrink(np.array([500.0, 5.0]), np.array([1.0, 1000.0]), mu=80.0, k=100.0)
    assert abs(out[0] - (500.0 + 8000.0) / 101.0) < 1e-9                  # n=1 の極端な値（500）は事前（80）にほぼ引き戻される
    assert abs(out[0] - 80.0) < 5.0
    assert abs(out[1] - (5.0 + 8000.0) / 1100.0) < 1e-9


def test_boot_diff_zero_when_selection_is_everything():
    m = _frame()
    sel = np.ones(len(m), dtype=bool)
    d = L.boot_diff(m, sel, sel, "ret_win", B=200)
    assert abs(d["diff"]) < 1e-12 and d["n_sel"] == len(m)


def test_reality_check_flags_a_true_winner_and_not_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 100, size=(3000, 50))
    assert R.reality_check(noise, B=200)["p"] > 0.05
    win = noise.copy()
    win[:, 7] += 40
    rc = R.reality_check(win, B=200)
    assert rc["best_idx"] == 7 and rc["p"] < 0.05


def test_ols_cluster_recovers_coefficient():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    b, se = D._ols_cluster(X, y, np.arange(n) // 4)
    assert abs(b[1] - 0.5) < 0.05 and se[1] > 0


def test_knn_table_excludes_self_and_finds_identical():
    E = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    idx, sim = EF.knn_table(E, k=2)
    assert 0 not in idx[0] and idx[0][0] == 1 and sim[0][0] > 0.999
    assert idx[2][0] == 3


def test_classify_line_picks_first_founder_in_tail_male_chain():
    from src.scripts.data import tp60_ped as PED

    assert PED.classify_line(["ディープインパクト", "サンデーサイレンスSunday Silence(米)", "Halo"]) == "SS系"
    assert PED.classify_line(["Kingmambo", "ミスタープロスペクターMr. Prospector(米)", "Raise a Native"]) == "MrP系"
    assert PED.classify_line(["Sadler's Wells", "ノーザンダンサーNorthern Dancer(加)"]) == "ND系"
    assert PED.classify_line(["unknown", ""]) == "その他"
