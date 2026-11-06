import numpy as np

from src.research.race.running_map_best5 import best5

N = 10
EXT = {"n": N, "x": [0.0, 9.0], "y": [0.0, 9.0]}


def _horses(m):
    return [{"horse_number": i + 1, "horse_id": f"h{i+1}", "x": float(i % N), "y": float(i // N)} for i in range(m)]


def _grid(f):
    return [f(i, j) for i in range(N) for j in range(N)]


def test_none_without_grids():
    assert best5(_horses(8), EXT, None, None) is None
    assert best5([], EXT, [1.0] * 100, None) is None


def test_course_ranks_by_lift_and_returns_five():
    hs = _horses(8)
    lift = _grid(lambda i, j: 0.5 + 0.1 * i)          # x が大きいほど倍率が高い
    r = best5(hs, EXT, lift, None)
    nums = [p["horse_number"] for p in r["modes"]["course"]["picks"]]
    assert len(nums) == 5 and nums[0] == 8 and nums == sorted(nums, reverse=True)
    assert r["default"] == "both" and r["modes"]["both"]["fallback"] == "course"
    assert r["modes"]["day"]["fallback"] == "course"


def test_day_mode_falls_back_to_course_without_day_grid_not_to_list_order():
    hs = _horses(8)
    lift = _grid(lambda i, j: 1.5 - 0.1 * i)          # x が小さいほど高い
    r = best5(hs, EXT, lift, None)
    assert [p["horse_number"] for p in r["modes"]["day"]["picks"]] == [p["horse_number"] for p in r["modes"]["course"]["picks"]]
    assert r["modes"]["day"]["picks"][0]["horse_number"] == 1


def test_flat_day_grid_ties_break_by_course_lift_then_horse_number():
    hs = _horses(8)
    flat = [1.0] * (N * N)
    lift = _grid(lambda i, j: 0.5 + 0.1 * i)
    r = best5(hs, EXT, lift, flat)
    assert r["modes"]["day"]["fallback"] is None
    assert [p["horse_number"] for p in r["modes"]["day"]["picks"]][0] == 8       # 並び順（馬番 1 から）にならない
    r2 = best5(hs, EXT, flat, flat)
    assert [p["horse_number"] for p in r2["modes"]["both"]["picks"]] == [1, 2, 3, 4, 5]   # 全部同点なら馬番


def test_both_adds_day_and_deterministic():
    hs = _horses(8)
    course = _grid(lambda i, j: 1.0)
    day = _grid(lambda i, j: 1.0 + 0.1 * (i == 3))
    a = best5(hs, EXT, course, day)
    b = best5(hs, EXT, course, day)
    assert a == b
    assert a["modes"]["both"]["picks"][0]["horse_number"] == 4
    assert a["modes"]["both"]["fallback"] is None


def test_wrong_size_grid_ignored():
    assert best5(_horses(8), EXT, [1.0] * 5, None) is None
