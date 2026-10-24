import numpy as np
import pandas as pd

from src.scripts.data import ptrain_cross as pc


def test_crosses_of_counts_within_8_gens_both_sides():
    anc = [{"path_fm": "FF", "horse_id": "A", "name": "a"}, {"path_fm": "MFM", "horse_id": "A"}, {"path_fm": "F" * 9, "horse_id": "B"},
           {"path_fm": "F" * 9, "horse_id": "B"}, {"path_fm": "FM", "horse_id": "C"}]
    c = pc.crosses_of(anc)
    assert [x[0] for x in c] == ["A"]                       # B は 9 世代なので除く、C は 1 回だけ
    assert c[0][2] == 2 and abs(c[0][3] - (0.25 + 0.125)) < 1e-9 and c[0][5] is True


def _L(n=40):
    d = pd.date_range("2022-01-01", periods=n, freq="7D")
    return pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "horse_id": [f"h{i % 5}" for i in range(n)], "date": d, "anc": "A", "blood": 0.375,
                         "res": np.where(np.arange(n) % 2 == 0, 0.3, -0.2), "c1": "芝|1", "c2": "芝|良", "c3": "東京|芝"})


def test_asof_cross_ignores_same_day_and_future():
    L = _L()
    a = pc.asof_cross(L).set_index(pc.KEY)
    L2 = L.copy()
    L2.loc[20:, "res"] = 0.5                                # 未来の結果を変えても、それ以前の行は変わらない
    b = pc.asof_cross(L2).set_index(pc.KEY)
    idx = a.index[:20]
    assert a.loc[idx].equals(b.loc[idx])
    assert abs(a.iloc[0]["P77_X_m_all_max"]) < 1e-9          # 最初の行は履歴なし


def test_horse_groups_balance_and_level():
    cl = pd.DataFrame({"horse_id": ["a", "a", "b", "c"], "anc": ["X", "Y", "X", "X"], "blood": [0.125, 0.0625, 0.125, 0.125], "gmin": [3, 4, 3, 3],
                       "both_sides": [False, False, True, False], "blood_f": [0.0, 0.0, 0.0625, 0.125], "blood_m": [0.125, 0.0625, 0.0625, 0.0]})
    h = pc.horse_groups(cl).set_index("horse_id")
    assert h.loc["a", "g_bal"] == "母方優勢" and h.loc["b", "g_bal"] == "同程度" and h.loc["c", "g_bal"] == "父方優勢"
    assert h.loc["a", "g_lvl"] == "濃" and h.loc["a", "g_near"] == "近い(~4代)"


def test_group_lift_asof_and_report_run():
    n = 60
    d = pd.date_range("2021-01-01", periods=n, freq="14D")
    R = pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "horse_id": [f"h{i % 4}" for i in range(n)], "date": d,
                      "res": np.where(np.arange(n) % 3 == 0, 0.3, -0.1), "c1": "芝|1", "c2": "芝|良",
                      "g_bal": ["母方優勢", "なし"] * (n // 2), "g_lvl": "濃", "g_near": "近い(~4代)"})
    a = pc.asof_group_lift(R)
    R2 = R.copy()
    R2.loc[40:, "res"] = 0.5
    b = pc.asof_group_lift(R2)
    assert a.iloc[:40].equals(b.iloc[:40])
    assert "g_bal|c1" in pc.group_report(R)
