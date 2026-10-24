import numpy as np
import pandas as pd

from src.scripts.data import jockey_switch_ev as J


def _frame():
    rows = []
    for day in pd.date_range("2024-01-06", periods=4, freq="7D"):
        for jid, win in (("A", 1), ("B", 0), ("C", 0)):
            rows.append({"date": day, "jockey_id": jid, "win": win, "surface": "芝"})
    return pd.DataFrame(rows)


def test_elite_flag_uses_only_wins_before_the_day(monkeypatch):
    monkeypatch.setattr(J, "TOPS", (1,))
    ef = J.elite_flags(_frame())[1]
    first, second = pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-13")
    # 初日は勝利数が全員 0（当日の勝ちは見ない）なので、順位は騎乗数も同じで ID 順 → A が先頭になるのは勝ちを見たからではない
    assert ef.loc[first].sum() == 1
    assert ef.loc[second, "A"] and not ef.loc[second, "B"]


def test_elite_flag_ignores_same_day_win(monkeypatch):
    monkeypatch.setattr(J, "TOPS", (1,))
    d = _frame()
    d.loc[(d["jockey_id"] == "C") & (d["date"] == pd.Timestamp("2024-01-13")), "win"] = 50     # 当日の C の大勝ちは当日の順位に影響しない
    ef = J.elite_flags(d)[1]
    assert ef.loc[pd.Timestamp("2024-01-13"), "A"]


def test_holm_is_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.5, 0.04])
    adj = J.holm(p)
    assert (adj >= p).all() and adj.max() <= 1.0
    assert abs(adj[0] - 0.004) < 1e-12
