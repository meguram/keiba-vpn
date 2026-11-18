"""好走距離レンジ（src/research/race/good_distance.py）の点数・レース区分・as-of・レンジ。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.race import good_distance as gd


def _rows(specs):
    """specs: (horse_id, date, surface, distance, points) の並び。"""
    return pd.DataFrame(specs, columns=["horse_id", "date", "surface", "distance", "points"])


def _pts(race_class, age, finish, race_id="r1", **extra):
    df = pd.DataFrame({"race_id": [race_id], "race_class": [race_class], "sex_age": [f"牡{age}"], "finish_position": [finish]})
    return gd.run_points(df).iloc[0]


def test_古馬戦の点数は勝ち5_2から3着3_4着以下1():
    assert _pts("サラ系３歳以上 １勝クラス", 4, 1)["points"] == 5
    assert _pts("サラ系３歳以上 １勝クラス", 4, 2)["points"] == 3
    assert _pts("サラ系４歳以上 オープン", 5, 3)["points"] == 3
    assert _pts("サラ系４歳以上 オープン", 5, 4)["points"] == 1


def test_世代戦は1から3着が3点で勝ちも3点():
    assert _pts("サラ系３歳 未勝利", 3, 1)["points"] == 3
    assert _pts("サラ系２歳 新馬", 2, 3)["points"] == 3
    assert _pts("サラ系３歳 未勝利", 3, 4)["points"] == 1


def test_着順0は点なし():
    assert np.isnan(_pts("サラ系３歳 未勝利", 3, 0)["points"])


def test_レース区分の判定():
    assert gd.race_kind_of_class("サラ系３歳以上 １勝クラス") == "古馬戦"
    assert gd.race_kind_of_class("サラ系４歳以上 オープン") == "古馬戦"
    assert gd.race_kind_of_class("サラ系３歳 未勝利") == "世代戦"
    assert gd.race_kind_of_class("サラ系２歳 新馬") == "世代戦"
    assert gd.race_kind_of_class("") is None


def test_空欄のレースは全馬が3歳以下なら世代戦_それ以外は古馬戦():
    df = pd.DataFrame({
        "race_id": ["a", "a", "b", "b"], "race_class": ["", "", "", ""],
        "sex_age": ["牡3", "牝2", "牡3", "牝5"], "finish_position": [1, 2, 1, 2],
    })
    out = gd.run_points(df)
    assert list(out["kind"]) == ["世代戦", "世代戦", "古馬戦", "古馬戦"]
    assert list(out["points"]) == [3, 3, 5, 3]


def test_距離は100m刻みに丸める():
    assert list(gd.round_distance([1150, 1149, 1200, 1250, 2399])) == [1200, 1100, 1200, 1300, 2400]


def test_as_ofは前走までだけを使う():
    runs = _rows([
        ("h", "2024-01-01", "芝", 1200, 1),
        ("h", "2024-02-01", "芝", 1200, 3),
        ("h", "2024-03-01", "芝", 2000, 5),
    ])
    r = gd.as_of_range(runs, 0.5, by_surface=False)
    assert r["n_prior"].tolist() == [0, 1, 2]
    assert r["pts_prior"].tolist() == [0, 1, 4]
    assert np.isnan(r["kd_short"].iloc[0])
    assert (r["kd_short"].iloc[1], r["kd_long"].iloc[1]) == (1200, 1200)
    # 3 走目の時点は 1200m が 4 点（相対値 1.0）、2000m はまだ数えない
    assert (r["kd_short"].iloc[2], r["kd_long"].iloc[2]) == (1200, 1200)


def test_同じ日の別レースは前の走に含めない():
    runs = _rows([
        ("h", "2024-01-01", "芝", 1200, 3),
        ("h", "2024-01-01", "芝", 2000, 5),
        ("h", "2024-02-01", "芝", 1600, 1),
    ])
    r = gd.as_of_range(runs, 0.1, by_surface=False)
    assert r["n_prior"].tolist() == [0, 0, 2]
    assert (r["kd_short"].iloc[2], r["kd_long"].iloc[2]) == (1200, 2000)


def test_相対値の合計は1():
    runs = _rows([
        ("h", "2024-01-01", "芝", 1200, 1), ("h", "2024-01-08", "芝", 1600, 3),
        ("h", "2024-01-15", "ダート", 1150, 5), ("h", "2024-01-22", "芝", 1600, 1),
    ])
    for by in (False, True):
        p = gd.profile_table(runs, by)
        assert p.groupby(["horse_id", "surface"])["rel"].sum().round(9).eq(1.0).all()


def test_単一距離なら短と長が同じ():
    runs = _rows([("h", f"2024-0{m}-01", "芝", 1600, 1) for m in range(1, 5)])
    r = gd.as_of_range(runs, 0.25, by_surface=False)
    last = r.iloc[-1]
    assert last["kd_short"] == last["kd_long"] == 1600
    lat = gd.latest_range(runs, 0.25, False).iloc[0]
    assert lat["kd_short"] == lat["kd_long"] == 1600 and lat["n_runs"] == 4


def test_τ以上の距離が無いときは最大の距離をレンジにする():
    runs = _rows([("h", f"2024-01-{d:02d}", "芝", 1000 + 200 * d, 1) for d in range(1, 8)] + [("h", "2024-02-01", "芝", 2400, 1)])
    r = gd.as_of_range(runs, 0.5, by_surface=False)
    last = r.iloc[-1]
    assert last["kd_short"] == 1200 and last["kd_long"] == 2400


def test_by_surfaceでは同じ馬場の走だけを使う():
    runs = _rows([
        ("h", "2024-01-01", "芝", 1200, 1),
        ("h", "2024-01-08", "ダート", 1800, 1),
        ("h", "2024-01-15", "芝", 1600, 1),
        ("h", "2024-01-22", "ダート", 1800, 1),
    ])
    merged = gd.as_of_range(runs, 0.1, by_surface=False)
    split = gd.as_of_range(runs, 0.1, by_surface=True)
    assert merged["n_prior"].tolist() == [0, 1, 2, 3]
    assert split["n_prior"].tolist() == [0, 0, 1, 1]
    # 4 走目（ダート）: 合算なら 1200〜1800、芝ダ別ならダートの 1800 だけ
    assert (merged["kd_short"].iloc[3], merged["kd_long"].iloc[3]) == (1200, 1800)
    assert (split["kd_short"].iloc[3], split["kd_long"].iloc[3]) == (1800, 1800)
    lat = gd.latest_range(runs, 0.1, True).set_index("surface")
    assert (lat.loc["芝", "kd_short"], lat.loc["芝", "kd_long"]) == (1200, 1600)


def test_走数が少ないときは不明():
    runs = _rows([("h", "2024-01-01", "芝", 1200, 1), ("h", "2024-02-01", "芝", 1400, 1), ("h", "2024-03-01", "芝", 1600, 1)])
    r = gd.as_of_range(runs, 0.1, by_surface=False, min_runs=3)
    assert r["kd_short"].isna().all()
    r1 = gd.as_of_range(runs, 0.1, by_surface=False, min_runs=2)
    assert r1["kd_short"].isna().tolist() == [True, True, False]


def test_点なしの走は走数にも数えず障害は除く():
    runs = _rows([
        ("h", "2024-01-01", "芝", 1200, np.nan),
        ("h", "2024-02-01", "障", 3000, 1),
        ("h", "2024-03-01", "芝", 1600, 1),
    ])
    r = gd.as_of_range(runs, 0.1, by_surface=False)
    assert r.loc[0, "n_prior"] == 0
    assert np.isnan(r.loc[1, "n_prior"])  # 障害の行は対象外（結果に出ない）
    assert r.loc[2, "n_prior"] == 0
    assert gd.latest_range(runs, 0.1, False)["n_runs"].tolist() == [1]


def test_馬ごとに分かれる():
    runs = _rows([
        ("a", "2024-01-01", "芝", 1200, 1), ("a", "2024-02-01", "芝", 1200, 1),
        ("b", "2024-01-15", "芝", 2400, 1), ("b", "2024-02-15", "芝", 2400, 1),
    ]).sample(frac=1, random_state=0)
    r = gd.as_of_range(runs, 0.1, by_surface=False)
    sec = {(runs.loc[i, "horse_id"], runs.loc[i, "date"]): (r.loc[i, "n_prior"], r.loc[i, "kd_short"]) for i in runs.index}
    assert sec[("a", "2024-02-01")] == (1, 1200)
    assert sec[("b", "2024-02-15")] == (1, 2400)


def test_age_ofは全角も読む():
    assert gd.age_of(pd.Series(["牡3", "牝１０", "せん5", "?"])).fillna(-1).tolist() == [3, 10, 5, -1]


def test_古馬戦で3歳馬が勝っても5点():
    assert _pts("サラ系３歳以上 ２勝クラス", 3, 1)["points"] == 5


def test_区分も年齢も読めないときは不明で1点():
    df = pd.DataFrame({"race_id": ["x", "x"], "race_class": ["", ""], "sex_age": ["?", "-"], "finish_position": [1, 2]})
    out = gd.run_points(df)
    assert list(out["kind"]) == ["不明", "不明"]
    assert list(out["points"]) == [1, 1]


def test_日付が欠けた走は履歴にも対象にも入れない():
    runs = _rows([
        ("h", None, "芝", 2400, 5),
        ("h", "2024-01-01", "芝", 1200, 1),
        ("h", "2024-02-01", "芝", 1200, 1),
    ])
    r = gd.as_of_range(runs, 0.1, by_surface=False)
    assert np.isnan(r.loc[0, "n_prior"])
    assert r.loc[1, "n_prior"] == 0 and r.loc[2, "n_prior"] == 1
    assert (r.loc[2, "kd_short"], r.loc[2, "kd_long"]) == (1200, 1200)
    assert gd.latest_range(runs, 0.1, False)["n_runs"].tolist() == [2]


def test_τ以上が無いとき1200が3点で他4距離が1点なら1200だけ():
    runs = _rows([
        ("h", "2024-01-01", "芝", 1200, 3), ("h", "2024-01-08", "芝", 1400, 1), ("h", "2024-01-15", "芝", 1600, 1),
        ("h", "2024-01-22", "芝", 1800, 1), ("h", "2024-01-29", "芝", 2000, 1), ("h", "2024-02-05", "芝", 1600, 1),
    ])
    last = gd.as_of_range(runs, 0.5, by_surface=False).iloc[-1]    # 相対値は 1200 が 3/7、他は 1/7 で、0.5 以上は無い
    assert (last["kd_short"], last["kd_long"]) == (1200, 1200)


def test_相対値がちょうどτのときは含まれる():
    specs = [("h", "2024-01-01", "芝", 1200, 3)] + [("h", f"2024-01-{d:02d}", "芝", 1400 + 200 * i, 5) for i, d in enumerate((8, 15, 22))]
    specs += [("h", "2024-02-01", "芝", 2000, 1), ("h", "2024-02-08", "芝", 2000, 1), ("h", "2024-03-01", "芝", 1600, 1)]
    last = gd.as_of_range(_rows(specs), 0.15, by_surface=False).iloc[-1]   # 合計 20 点: 1200 が 3/20=0.15、1400〜1800 が 0.25、2000 が 0.10
    assert (last["kd_short"], last["kd_long"]) == (1200, 1800)


def test_評価の群分けと複勝の定義():
    from src.scripts.data import good_distance_eval as ev
    dist = np.array([1000, 1200, 1600, 2000, 2400])
    grp = ev.group_of(dist, np.full(5, 1200.0), np.full(5, 2000.0))
    assert grp.tolist() == [1, 0, 0, 0, 2]
    fin = np.array([3, 3, 2, 3, 2, 1])
    nf = np.array([8, 7, 7, 5, 4, 18])
    assert ev.place_hits(fin, nf).tolist() == [1, 0, 1, 0, 1, 1]      # 8 頭以上は 3 着以内、5〜7 頭は 2 着以内（4 頭以下は評価に入れない）


def test_評価の中と外の差の計算():
    from src.scripts.data import good_distance_eval as ev
    # 1 レース: 中 2 頭（勝ち 1・複勝 2・期待 0.5/1.0）、短い側の外 1 頭（勝ち 0・複勝 0・期待 0.25/0.5）
    stat = np.array([[1, 1, 1, 0.5, 0.5], [1, 0, 1, 0.5, 0.5], [1, 0, 0, 0.25, 0.5]])
    rs = ev.group_stats(np.zeros(3, int), np.array([0, 0, 1]), stat, 1)
    c = ev.contrast(rs)
    assert (c["n_in"], c["n_out_short"], c["n_out_long"]) == (2, 1, 0)
    assert c["dwin_in"] == pytest.approx(0.0) and c["dwin_out"] == pytest.approx(-0.25)
    assert c["d_plc"] == pytest.approx(0.5 - (-0.5))


if __name__ == "__main__":
    pytest.main([__file__])
