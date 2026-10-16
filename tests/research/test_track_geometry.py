"""track_geometry（OSM の形状から周回路・経路を作る）の単体テストと、保存済みの生成結果（routes.json）の整合検査。

重い依存（shapely・scikit-image・skan・networkx）は使わない。生成そのものは python -m src.research.race.track_geometry。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.research.race import track_geometry as tg

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ROOT / "data/meta/course_master/geo/routes.json"
MASTER = ROOT / "data/meta/course_master/jra_courses.json"
OSM = ROOT / "data/meta/course_master/geo/osm"


def stadium(straight=500.0, radius=100.0, step=4.0) -> np.ndarray:
    """反時計回りのスタジアム型（下の直線が x 方向へ進む）。周長 2*straight + 2*pi*radius。"""
    pts = []
    n = int(straight / step)
    for i in range(n):
        pts.append((-straight / 2 + i * step, -radius))
    for a in np.linspace(-math.pi / 2, math.pi / 2, int(math.pi * radius / step) + 1)[:-1]:
        pts.append((straight / 2 + radius * math.cos(a), radius * math.sin(a)))
    for i in range(n):
        pts.append((straight / 2 - i * step, radius))
    for a in np.linspace(math.pi / 2, 3 * math.pi / 2, int(math.pi * radius / step) + 1)[:-1]:
        pts.append((-straight / 2 + radius * math.cos(a), radius * math.sin(a)))
    return np.array(pts)


def test_circuit_length_and_curve_groups():
    c = tg.Circuit(stadium())
    assert abs(c.L_run - (1000 + 2 * math.pi * 100)) / c.L_run < 0.02
    O = c.oriented(False)
    groups = [g for g in tg.curve_groups(O["kappa"], O["ds"]) if g[2] > 0]
    assert len(groups) == 2
    assert all(abs(g[2] - 180) < 25 for g in groups)
    # 時計回りに向きを変えても、内側へ曲がる向きが正のまま
    Ocw = c.oriented(True)
    assert len([g for g in tg.curve_groups(Ocw["kappa"], Ocw["ds"]) if g[2] > 0]) == 2


def test_locate_goal_prefers_straight_near_stand():
    c = tg.Circuit(stadium())
    O = c.oriented(False)                          # 下の直線は x 方向、上の直線は −x 方向
    stand_bottom = np.array([[0.0, -160.0]])
    stand_top = np.array([[0.0, 160.0]])
    gb = tg.locate_goal(O, stand_bottom, 300.0)
    gt = tg.locate_goal(O, stand_top, 300.0)
    assert O["xy"][gb["idx"]][1] < 0 < O["xy"][gt["idx"]][1]
    # ゴールは直線の始まりから 300m 進んだ点
    straight_start = O["xy"][gb["i_exit"]]
    assert abs(math.hypot(*(O["xy"][gb["idx"]] - straight_start)) - 300.0) < 25.0


def test_first_curve_distance_on_synthetic_route():
    route = np.vstack([np.column_stack([np.linspace(0, 400, 101), np.zeros(101)]),
                       np.column_stack([400 + 150 * np.sin(np.linspace(0, math.pi, 60)), 150 - 150 * np.cos(np.linspace(0, math.pi, 60))])])
    d = tg.first_curve_distance(route)
    assert d is not None and 330 < d < 420


def test_route_points_slices_pts_q():
    circ = {"pts_q": [[float(i * 8), 0.0] for i in range(0, 400)], "L_run": 3200.0}
    r = tg.route_points(circ, 80.0)                # q = 0, 8, …, 80 の 11 点（スタート → ゴールの順）
    assert len(r) == 11 and r[0][0] == 80.0 and r[-1][0] == 0.0
    r2 = tg.route_points(circ, 80.0, stub=[[500.0, 5.0], [90.0, 0.0]])
    assert r2[0][0] == 500.0 and len(r2) == 13


def test_curve_threshold_is_documented_constant():
    assert tg.CURVE_THR == pytest.approx(1 / 260)


# ---------------------------------------------------------------- 保存済みの生成結果

@pytest.fixture(scope="module")
def routes():
    return json.loads(ROUTES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def master():
    return [r for r in json.loads(MASTER.read_text(encoding="utf-8")) if r.get("in_use") is not False]


def test_osm_data_saved_for_all_venues_with_attribution():
    for vc in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10"):
        d = json.loads((OSM / f"{vc}.json").read_text(encoding="utf-8"))
        assert "OpenStreetMap" in d["attribution"]
        assert any(f["kind"] == "track" for f in d["features"])
    assert "OpenStreetMap" in (ROOT / "data/meta/course_master/geo/README.md").read_text(encoding="utf-8")


def test_every_master_course_has_a_route_entry(routes, master):
    for r in master:
        key = f'{r["surface"]}|{r["variant"]}|{int(r["distance"])}'
        assert key in routes["venues"][r["venue_code"]]["courses"], key


def test_circuit_lengths_match_master(routes):
    errs = [c["err"] for v in routes["venues"].values() for c in v["circuits"].values()]
    assert len(errs) >= 24
    assert max(abs(e) for e in errs) <= 0.03          # 走行線の周長が、マスタの周長と 3% 以内


def test_goal_is_in_front_of_stand_and_circuits_are_complete(routes):
    for vc, v in routes["venues"].items():
        for key, c in v["circuits"].items():
            assert c["pts_q"] and len(c["pts_q"]) * tg.Q_STEP >= c["L_run"] - tg.Q_STEP
            assert c["goal"]["stand_offset_m"] < 220, (vc, key)         # ゴールとメインスタンドの距離


def test_course_routes_agree_with_master_first_curve(routes):
    errs, flagged, total = [], 0, 0
    for v in routes["venues"].values():
        for c in v["courses"].values():
            total += 1
            if c["flags"]:
                flagged += 1
                continue
            if c.get("first_curve_calc") is not None and c.get("first_curve_master") is not None:
                errs.append(abs(c["first_curve_calc"] - c["first_curve_master"]))
    assert total >= 105 and flagged <= 12              # 推定にとどまるコースは 1 割程度まで
    assert float(np.median(errs)) < 15 and float(np.mean(errs)) < 40      # 最初のコーナーまでの距離がマスタと合う


def test_pocket_starts_are_straight_extensions_inside_surface(routes):
    n = 0
    for v in routes["venues"].values():
        for c in v["courses"].values():
            if c.get("start") == "chute":
                n += 1
                assert 15 <= c["stub_len"] <= 500 and c["stub_inside"] >= 0.5
    assert n >= 35                                      # 引込み線から始まるコースが実形状で多数取れている


def test_known_special_starts(routes):
    t = routes["venues"]["05"]["courses"]
    assert t["芝||1600"]["start"] == "chute"           # 東京芝1600m: 向正面の引込み線
    assert t["ダート||1600"]["start"] == "chute"       # 東京ダート1600m: 向正面ポケット
    assert t["芝||2000"]["start"] == "main"            # 東京芝2000m: ゴール板を過ぎた先のホームストレッチ
    nk = routes["venues"]["06"]["courses"]
    assert nk["芝|外|1600"]["start"] == "chute" and nk["芝|内|2000"]["start"] == "chute"      # 中山の引込み線（外回り向正面・内回りのホーム側）
