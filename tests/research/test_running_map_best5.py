from src.research.race.running_map_best5 import W_DAY, best5

N = 10
EXT = {"n": N, "x": [0.0, 9.0], "y": [0.0, 9.0]}


def _horses(m):
    return [{"horse_number": i + 1, "horse_id": f"h{i+1}", "x": float(i % N), "y": float(i // N)} for i in range(m)]


def _grid(f):
    return [f(i, j) for i in range(N) for j in range(N)]


def _nums(r):
    return [p["horse_number"] for p in r["picks"]]


def test_none_without_grids():
    assert best5(_horses(8), EXT, None, None) is None
    assert best5([], EXT, [1.0] * 100, None) is None
    assert best5(_horses(8), EXT, [1.0] * 5, None) is None


def test_course_only_ranks_by_lift():
    r = best5(_horses(8), EXT, _grid(lambda i, j: 0.5 + 0.1 * i), None)
    assert len(r["picks"]) == 5 and r["source"] == "course" and _nums(r) == [8, 7, 6, 5, 4]


def test_flat_day_grid_ties_break_by_course_then_horse_number_not_list_order():
    flat = [1.0] * (N * N)
    r = best5(_horses(8), EXT, _grid(lambda i, j: 0.5 + 0.1 * i), flat)
    assert r["source"] == "both" and _nums(r)[0] == 8
    assert _nums(best5(_horses(8), EXT, flat, flat)) == [1, 2, 3, 4, 5]


def test_day_zone_adds_with_weight_and_is_deterministic():
    course = _grid(lambda i, j: 1.0)
    day = _grid(lambda i, j: 1.0 + 0.5 * (i == 3))
    a = best5(_horses(8), EXT, course, day)
    assert a == best5(_horses(8), EXT, course, day)
    assert _nums(a)[0] == 4 and a["w_day"] == W_DAY


def test_day_only():
    r = best5(_horses(8), EXT, None, _grid(lambda i, j: 1.0 + 0.1 * i))
    assert r["source"] == "day" and _nums(r)[0] == 8
