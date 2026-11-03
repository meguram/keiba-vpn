"""出走馬の走法マップ（共通 2 次元化・グループ分け・入力の整形・コース適性）のテスト。データファイルは使わない。"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.research.race import running_map as rm
from src.research.race import running_map_aptitude as apt


def _runners(seed: int = 0, n: int = 14) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, len(rm.AXES)))
    Z[:, 1] = Z[:, 0] + rng.normal(scale=0.5, size=n)   # 末脚の速さと持続力は似ている
    return Z * rm.WEIGHTS


def _fake_ref(seed: int = 99) -> dict:
    """全馬共通の主成分（reference）の代わり。_global_projection と同じ手順で作る。"""
    Zw = _runners(seed, n=4000)
    mu = Zw.mean(0)
    Zc = Zw - mu
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    Y, A = rm._orient(Zc @ Vt[:2].T, Vt[:2].T)
    recon = Y @ A.T
    var = Zc.var(0)
    return {
        "mu": mu.tolist(), "A": A.tolist(),
        "r2": [None if v < 1e-12 else float(1 - (Zc[:, j] - recon[:, j]).var() / v) for j, v in enumerate(var)],
        "explained": float((S[:2] ** 2).sum() / (S ** 2).sum()),
    }


class ProjectTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(rm, "_data", return_value={"ref": _fake_ref()})
        p.start()
        self.addCleanup(p.stop)

    def test_orientation_tskick_up_front_left(self):
        for seed in range(5):
            pr = rm.project(_runners(seed))
            up = pr["loadings"][rm._UP_IDX].sum(0)
            self.assertAlmostEqual(float(up[0]), 0.0, places=9)    # 末脚の矢印は真上
            self.assertGreater(float(up[1]), 0.0)
            self.assertLessEqual(float(pr["loadings"][rm._FRONT_IDX][0]), 0.0)  # 先行力の矢印は左

    def test_same_runner_gets_same_position_in_any_race(self):
        Z = _runners(1, n=10)
        alone = rm.project(Z[:4])["coords"]
        with_others = rm.project(Z)["coords"][:4]
        np.testing.assert_allclose(alone, with_others, atol=1e-12)

    def test_r2_and_fidelity(self):
        pr = rm.project(_runners(2))
        self.assertTrue(0 < pr["explained"] <= 1)
        self.assertTrue(-1 <= pr["distance_spearman"] <= 1)
        self.assertTrue(all(r is None or -1e-9 <= r <= 1 + 1e-9 for r in pr["r2"]))
        self.assertEqual(pr["coords"].shape, (14, 2))
        self.assertEqual(pr["loadings"].shape, (len(rm.AXES), 2))


class GroupTest(unittest.TestCase):
    K = len(rm.AXES)

    def test_separated_clusters_are_found(self):
        rng = np.random.default_rng(0)
        a = rng.normal(loc=0, scale=0.1, size=(6, self.K))
        b = rng.normal(loc=3, scale=0.1, size=(5, self.K))
        lab = rm.group(np.vstack([a, b]))
        self.assertEqual(len(set(lab)), 2)
        self.assertEqual(len(set(lab[:6])), 1)
        self.assertEqual(len(set(lab[6:])), 1)

    def test_too_few_or_identical_is_one_group(self):
        self.assertEqual(set(rm.group(np.zeros((3, self.K)))), {0})
        self.assertEqual(set(rm.group(np.ones((8, self.K)))), {0})

    def test_describe_group_names_each_axis(self):
        Zall = np.zeros((10, self.K))
        Zall[:3, rm.AXIS_KEYS.index("ts")] = 2.0      # 3 頭だけ末脚が速い
        Zall[:3, rm.AXIS_KEYS.index("front")] = -2.0  # 同じ 3 頭は後ろから行く
        traits = rm.describe_group(Zall[:3], Zall)
        self.assertEqual({t["axis"] for t in traits}, {"末脚の速さ", "先行力"})
        self.assertEqual({t["text"] for t in traits}, {"末脚が速い", "後ろから行く"})


class SimilarTest(unittest.TestCase):
    def test_nearest_in_the_full_space_in_order(self):
        Z = np.array([[0.0] * 9, [0.1] * 9, [1.0] * 9, [3.0] * 9])
        near = rm.similar(Z, k=2)
        self.assertEqual([j for j, _ in near[0]], [1, 2])
        self.assertAlmostEqual(near[0][0][1], 0.1)
        self.assertEqual([j for j, _ in near[3]], [2, 1])

    def test_imputed_axes_are_left_out_of_the_distance(self):
        a = np.zeros(9)
        b = np.zeros(9); b[0] = np.nan          # 欠けた方面（地図では中央値で補う）は距離に使わない
        c = np.full(9, 0.5)
        near = rm.similar(np.array([a, b, c]), k=2, min_common=6)
        self.assertEqual(near[0][0], (1, 0.0))

    def test_pairs_with_too_few_common_axes_are_skipped(self):
        a = np.zeros(9)
        b = np.full(9, np.nan); b[:5] = 0.0     # 共通の方面が 5 つしかない
        near = rm.similar(np.array([a, b]), k=3, min_common=6)
        self.assertEqual(near, [[], []])


class TransformTest(unittest.TestCase):
    def test_percentile_ties_take_middle(self):
        q = np.quantile(np.r_[np.zeros(80), np.linspace(1, 2, 20)], np.linspace(0, 1, 1001))
        p = rm._to_pct(np.array([0.0, 2.0, np.nan]), q)
        self.assertAlmostEqual(float(p[0]), 0.4, delta=0.01)  # 0 が 80% を占めるので中央の 40%
        self.assertAlmostEqual(float(p[1]), 1.0)
        self.assertTrue(np.isnan(p[2]))

    def test_condition_and_surface_normalization(self):
        self.assertEqual([rm._norm_condition(s) for s in ("稍", "稍重", "不", "重", "良", "", None)],
                         ["稍重", "稍重", "不良", "重", "良", None, None])
        self.assertEqual([rm._norm_surface(s) for s in ("芝", "ダ", "ダート", "障", "")], ["芝", "ダート", "ダート", None, None])

    def test_race_conditions_falls_back_to_race_id_venue(self):
        c = rm.race_conditions({"date": "2026/05/04", "distance": "1600", "surface": "芝"}, "202605040301")
        self.assertEqual(c, {"date": "2026-05-04", "venue": "東京", "surface": "芝", "distance": 1600, "track_condition": None})
        self.assertIsNone(rm.race_conditions({"distance": 0}, "202605040301")["distance"])


class AptitudeLookupTest(unittest.TestCase):
    def _patch(self, courses):
        return mock.patch.object(rm, "_data", return_value={"ref": {"_aptitude": {"courses": courses}}})

    def test_returns_course_map_with_label(self):
        hit = {"n_runs": 400, "n_races": 30, "lift": [1.0], "sig": "2"}
        with self._patch({apt.course_key("東京", "芝", 1600): hit}):
            out = rm.aptitude_for({"venue": "東京", "surface": "芝", "distance": 1600})
        self.assertEqual(out["course"], "東京 芝 1600m")
        self.assertEqual(out["sig"], "2")

    def test_unknown_or_incomplete_course_is_none(self):
        with self._patch({}):
            self.assertIsNone(rm.aptitude_for({"venue": "東京", "surface": "芝", "distance": 1600}))
            self.assertIsNone(rm.aptitude_for({"venue": "東京", "surface": None, "distance": 1600}))
            self.assertIsNone(rm.aptitude_for({}))


class AptitudeBuildTest(unittest.TestCase):
    EXTENT = {"n": apt.GRID, "x": [-2.0, 2.0], "y": [-2.0, 2.0]}

    def _table(self, n_races: int = 120, field: int = 10, seed: int = 0, edge: bool = True) -> pd.DataFrame:
        """x>1 の馬だけ好走しやすい（edge=True）か、座標と着順が無関係なコース。"""
        rng = np.random.default_rng(seed)
        rows = []
        for r in range(n_races):
            xy = rng.normal(size=(field, 2))
            score = (xy[:, 0] > 1) * 3.0 * edge + rng.normal(size=field)
            order = (-score).argsort().argsort() + 1
            for i in range(field):
                rows.append({"race_id": f"r{r}", "venue": "東京", "surface": "芝", "distance": 1600,
                             "finish_position": int(order[i]), "x": xy[i, 0], "y2": xy[i, 1]})
        return pd.DataFrame(rows)

    def test_course_key(self):
        self.assertEqual(apt.course_key("中山", "ダート", 1800.0), "中山|ダート|1800")

    def test_extent_is_percentile_range(self):
        xy = np.random.default_rng(0).normal(size=(5000, 2))
        e = apt.extent_from(xy)
        self.assertEqual(e["n"], apt.GRID)
        self.assertLess(e["x"][0], -2.0)
        self.assertGreater(e["y"][1], 2.0)

    def test_prepare_drops_small_fields_and_missing_coords(self):
        t = self._table(n_races=3, field=10)
        small = pd.DataFrame({"race_id": ["s"] * 4, "venue": "東京", "surface": "芝", "distance": 1600,
                              "finish_position": [1, 2, 3, 4], "x": 0.0, "y2": 0.0})
        t.loc[0, "x"] = np.nan
        out = apt.prepare(pd.concat([t, small], ignore_index=True))
        self.assertNotIn("s", set(out["race_id"]))
        self.assertEqual(len(out), 29)
        self.assertTrue(np.allclose(out["e"], 0.3))
        self.assertEqual(float(out["y"].sum()), 9.0 - float(t.loc[0, "finish_position"] <= 3))

    def test_field_size_counts_horses_without_coordinates(self):
        t = self._table(n_races=1, field=5)
        t.loc[0, "x"] = np.nan
        out = apt.prepare(t)
        self.assertEqual(len(out), 4)
        self.assertTrue(np.allclose(out["e"], 3 / 5))

    def test_build_finds_the_good_zone_and_skips_small_courses(self):
        tab = apt.prepare(self._table(edge=True))
        other = tab.assign(distance=2000).head(100)  # 出走が少ないコース
        out = apt.build(pd.concat([tab, other], ignore_index=True), self.EXTENT)
        self.assertEqual(list(out["courses"]), [apt.course_key("東京", "芝", 1600)])
        c = out["courses"][apt.course_key("東京", "芝", 1600)]
        self.assertEqual(len(c["lift"]), apt.GRID ** 2)
        self.assertEqual(len(c["sig"]), apt.GRID ** 2)
        self.assertEqual(c["n_races"], 120)

        gx = np.linspace(*self.EXTENT["x"], apt.GRID)
        gy = np.linspace(*self.EXTENT["y"], apt.GRID)
        ix, iy = int(np.argmin(abs(gx - 1.6))), int(np.argmin(abs(gy - 0.0)))
        good = ix * apt.GRID + iy  # x が外側
        self.assertGreater(c["lift"][good], 1.3)
        bad = int(np.argmin(abs(gx + 1.0))) * apt.GRID + iy
        self.assertLess(c["lift"][bad], c["lift"][good])
        self.assertIn(c["sig"][good], "12")
        self.assertTrue(set(c["sig"]) <= {"0", "1", "2"})

    def test_no_effect_course_has_few_significant_cells(self):
        tab = apt.prepare(self._table(edge=False, seed=3))
        out = apt.build(tab, self.EXTENT)
        c = next(iter(out["courses"].values()))
        valid = sum(ch != "0" for ch in c["sig"])
        self.assertLess(c["sig"].count("2"), 0.15 * valid)


if __name__ == "__main__":
    unittest.main()
