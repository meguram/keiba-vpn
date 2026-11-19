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
    Z[:, 1] = Z[:, 0] + rng.normal(scale=0.5, size=n)   # TSとTS持続力は似ている
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
            self.assertLessEqual(float(pr["loadings"][rm._SIDE_IDX][0]), 0.0)  # Cr性能の矢印は左

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
        self.assertEqual({t["axis"] for t in traits}, {"TS", "先行力"})
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


class ExtrasScoreTest(unittest.TestCase):
    """補助の軸（急坂・道悪）の 10 点。基準が無ければ None、あれば 1〜10 の 0.5 点刻み。"""

    REF = {"k": 1.5, "ts": {"knots": [[50, 3.0], [60, 5.0]]},
           "axes": {"slope": {"mu": 53.0, "sd": 6.0, "n": 100, "sign": 1}}}

    def test_slope_score_with_and_without_reference(self):
        import pandas as pd

        row = pd.Series({"ax_slope": 59.0, "n_ax_slope": 4})
        self.assertEqual(rm._extras("h", row, ["slope"], None, self.REF)["slope"]["score"], 6.5)   # 5 + 1.5 × (59-53)/6
        self.assertIsNone(rm._extras("h", row, ["slope"], None, None)["slope"]["score"])
        unknown = pd.Series({"ax_slope": float("nan"), "n_ax_slope": 0})
        self.assertIsNone(rm._extras("h", unknown, ["slope"], None, self.REF)["slope"]["score"])

    def test_going_without_data_is_unknown(self):
        import pandas as pd

        out = rm._extras("h", pd.Series(dtype=float), ["going_turf"], None, self.REF)["going_turf"]
        self.assertIsNone(out["value"])
        self.assertIsNone(out["score"])


    def test_good_distance_range_extra(self):
        import pandas as pd

        gd = pd.DataFrame({"short_m": [1400.0, float("nan")], "long_m": [1800.0, float("nan")], "runs": [5, 2]}, index=["h1", "h2"])
        row = pd.Series(dtype=float)
        ok = rm._extras("h1", row, [], None, self.REF, gd)[rm.GD_KEY]
        self.assertEqual((ok["short_m"], ok["long_m"], ok["runs"]), (1400, 1800, 5))
        self.assertIsNone(ok["value"])
        few = rm._extras("h2", row, [], None, self.REF, gd)[rm.GD_KEY]   # 走数が少ない馬は不明（走数は出す）
        self.assertEqual((few["short_m"], few["long_m"], few["runs"]), (None, None, 2))
        none = rm._extras("zz", row, [], None, self.REF, gd)[rm.GD_KEY]   # 走が無い馬
        self.assertEqual((none["short_m"], none["long_m"], none["runs"]), (None, None, 0))
        self.assertNotIn(rm.GD_KEY, rm._extras("h1", row, [], None, self.REF, None))   # 元の表が無ければ出さない

    def test_good_distance_is_not_a_color_axis(self):
        self.assertNotIn(rm.GD_KEY, rm.EXTRA_LABELS)
        self.assertNotIn(rm.GD_KEY, rm._extra_keys({"venue": "中山", "surface": "芝"}))


def _flat_rows(year: int) -> pd.DataFrame:
    """race_result_flat の小さい版（馬場・馬 ID は区分型、日付は日時型）。平地 2 レース + 障害 1 レース + 重複 1 行。"""
    rows = [
        ("r1", "h1", f"{year}-05-01", "芝", 1600, "サラ系３歳以上 １勝クラス", "牡4", 1),
        ("r1", "h2", f"{year}-05-01", "芝", 1600, "サラ系３歳以上 １勝クラス", "牡5", 2),
        ("r1", "h2", f"{year}-05-01", "芝", 1600, "サラ系３歳以上 １勝クラス", "牡5", 2),   # 重複
        ("r2", "h1", f"{year}-06-01", "ダート", 1800, "サラ系３歳以上 １勝クラス", "牡4", 5),
        ("r3", "h1", f"{year}-07-01", "障害", 3000, "サラ系障害", "牡4", 1),
    ]
    d = pd.DataFrame(rows, columns=["race_id", "horse_id", "date", "surface", "distance", "race_class", "sex_age", "finish_position"])
    d["date"] = pd.to_datetime(d["date"])
    d["surface"] = d["surface"].astype("category")
    d["horse_id"] = d["horse_id"].astype("category")
    return d


class GoodDistanceLoadTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for y in (2023, 2024):
            (Path(self.tmp.name) / str(y)).mkdir()
            _flat_rows(y).to_parquet(Path(self.tmp.name) / str(y) / "race_result_flat.parquet")
        self.dir = Path(self.tmp.name)
        for target, kw in [(rm.rp, {"FLAT_DIR": self.dir}), (rm, {"GD_FIRST_YEAR": 2023}), (rm, {"_gd_last_year": lambda: 2024})]:
            p = mock.patch.multiple(target, **kw)
            p.start()
            self.addCleanup(p.stop)
        rm._gd_runs.cache_clear()
        rm._gd_failed_at = None
        self.addCleanup(rm._gd_runs.cache_clear)
        self.addCleanup(setattr, rm, "_gd_failed_at", None)

    def test_reads_flat_runs_only_with_points_and_small_types(self):
        d = rm._gd_runs()
        self.assertEqual(len(d), 6)   # 2 年 × (平地 3 走。障害・重複は除く)
        self.assertEqual(set(d["surface"].astype(str)), {"芝", "ダート"})
        self.assertEqual(list(d.columns), ["horse_id", "date", "surface", "distance", "points"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(d["date"]))
        self.assertEqual(str(d["distance"].dtype), "float32")
        self.assertEqual(str(d["points"].dtype), "float32")
        h1_win = d[(d["horse_id"] == "h1") & (d["surface"] == "芝")]
        self.assertEqual(h1_win["points"].tolist(), [5.0, 5.0])   # 古馬戦の勝ち 5 点
        self.assertEqual(sorted(d.loc[d["horse_id"] == "h2", "points"].tolist()), [3.0, 3.0])   # 古馬戦 2 着 3 点

    def test_range_for_horses_from_the_small_table(self):
        out = rm._gd_for(["h1", "h2"], "2025-01-01")
        self.assertIsNotNone(out)
        self.assertEqual(int(out.loc["h1", "runs"]), 4)   # 芝 2 走 + ダート 2 走
        self.assertEqual((out.loc["h1", "short_m"], out.loc["h1", "long_m"]), (1600.0, 1800.0))   # 1600m は 10/12、1800m は 2/12（>= 0.10）
        self.assertTrue(np.isnan(out.loc["h2", "short_m"]))   # 2 走だけは不明
        self.assertEqual(int(out.loc["h2", "runs"]), 2)
        before = rm._gd_for(["h1"], "2024-05-01")   # その日より前の走だけ（同日は含めない）
        self.assertEqual(int(before.loc["h1", "runs"]), 2)   # 2023 年の 2 走だけ（2024-05-01 の走は含めない）
        self.assertTrue(np.isnan(before.loc["h1", "short_m"]))

    def test_failure_returns_none_and_is_not_reread_per_race(self):
        with mock.patch.object(rm, "_gd_runs", side_effect=OSError("読めない")) as m:
            self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))
            self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))
            self.assertIsNone(rm._gd_for(["h2"], "2025-01-01"))
        self.assertEqual(m.call_count, 1)   # 失敗後は一定時間、全年の読み直しをしない
        rm.clear_cache()
        self.assertIsNotNone(rm._gd_for(["h1"], "2025-01-01"))   # キャッシュを消せば読み直す

    def test_failed_load_is_retried_after_the_wait(self):
        with mock.patch("time.monotonic", return_value=1000.0), mock.patch.object(rm, "_gd_runs", side_effect=OSError("読めない")) as m:
            self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))
            with mock.patch("time.monotonic", return_value=1000.0 + rm.GD_RETRY_SEC - 1):
                self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))
            self.assertEqual(m.call_count, 1)
            with mock.patch("time.monotonic", return_value=1000.0 + rm.GD_RETRY_SEC + 1):
                self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))   # 時間が過ぎたので読み直す（まだ失敗）
            self.assertEqual(m.call_count, 2)
        with mock.patch("time.monotonic", return_value=1000.0 + 2 * rm.GD_RETRY_SEC + 2):
            self.assertIsNotNone(rm._gd_for(["h1"], "2025-01-01"))   # 直っていれば付く

    def test_range_failure_of_one_race_does_not_block_the_next(self):
        from src.research.race import good_distance as gd

        real = gd.range_before
        with mock.patch.object(gd, "range_before", side_effect=ValueError("このレースだけ異常")):
            self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))
        self.assertIsNone(rm._gd_failed_at)   # 読み込みの失敗ではないのでフラグは立てない
        with mock.patch.object(gd, "range_before", side_effect=real) as m:
            self.assertIsNotNone(rm._gd_for(["h1"], "2025-01-01"))   # 次のレースにはすぐ付く
        self.assertEqual(m.call_count, 1)

    def test_missing_tables_give_none_and_are_read_again_once_they_appear(self):
        import shutil

        shutil.move(str(self.dir), str(self.dir) + "_moved")
        self.addCleanup(shutil.rmtree, str(self.dir) + "_moved", True)
        with mock.patch("time.monotonic", return_value=500.0):
            self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))
        self.assertIsNotNone(rm._gd_failed_at)
        shutil.move(str(self.dir) + "_moved", str(self.dir))   # 後からファイルができた
        with mock.patch("time.monotonic", return_value=500.0 + rm.GD_RETRY_SEC + 1):
            self.assertIsNotNone(rm._gd_for(["h1"], "2025-01-01"))   # 空の表は残らず、読み直して付く

    def test_a_missing_year_is_a_failure_not_a_cached_partial_table(self):
        (self.dir / "2023" / "race_result_flat.parquet").unlink()   # 2024 だけ残る（一部の年が欠ける）
        with mock.patch("time.monotonic", return_value=500.0):
            self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))
        self.assertEqual(rm._gd_runs.cache_info().currsize, 0)   # 不完全な表はキャッシュしない
        _flat_rows(2023).to_parquet(self.dir / "2023" / "race_result_flat.parquet")
        with mock.patch("time.monotonic", return_value=500.0 + rm.GD_RETRY_SEC + 1):
            out = rm._gd_for(["h1"], "2025-01-01")
        self.assertEqual(int(out.loc["h1", "runs"]), 4)

    def test_current_year_file_is_optional_but_added_when_present(self):
        (self.dir / "2025").mkdir()
        _flat_rows(2025).to_parquet(self.dir / "2025" / "race_result_flat.parquet")
        self.assertEqual(len(rm._gd_runs()), 9)

    def test_missing_tables_give_none(self):
        with mock.patch.object(rm.rp, "FLAT_DIR", self.dir / "none"):
            rm._gd_runs.cache_clear()
            self.assertIsNone(rm._gd_for(["h1"], "2025-01-01"))


class BuildRunningMapGdTest(unittest.TestCase):
    """build_running_map が好走距離レンジの表を良馬場・道悪の両方の地図へ渡す（失敗しても地図は出る）。"""

    def setUp(self):
        rm._gd_failed_at = None
        self.addCleanup(setattr, rm, "_gd_failed_at", None)
        self.calls: list[dict] = []

        def fake_assemble(entries, ids, prof, race_id, conditions, extra_keys, going, load_result, wet, peaks=None, gd=None):
            self.calls.append({"wet": wet, "gd": gd})
            return {"status": "ok", "horses": []}

        for target, kw in [(rm, {"_data": {"return_value": {"metrics": None, "sires": None, "ref": {}}}}),
                           (rm, {"_priors": {"return_value": None}}),
                           (rm, {"_assemble": {"side_effect": fake_assemble}}),
                           (rm.rp, {"profiles_as_of": {"return_value": pd.DataFrame({"horse_id": ["h1"]})}})]:
            for name, opts in kw.items():
                p = mock.patch.object(target, name, **opts)
                p.start()
                self.addCleanup(p.stop)

    def _build(self, surface="芝"):
        return rm.build_running_map([{"horse_id": "h1", "horse_number": 1}], "r", {"date": "2026-11-01", "surface": surface, "venue": "東京"})

    def test_attaches_table_to_good_and_wet_maps(self):
        tbl = pd.DataFrame({"short_m": [1400.0], "long_m": [1800.0], "runs": [5]}, index=["h1"])
        with mock.patch.object(rm, "_gd_for", return_value=tbl):
            out = self._build()
        self.assertEqual([c["wet"] for c in self.calls], [False, True])
        self.assertTrue(all(c["gd"] is tbl for c in self.calls))
        self.assertIn("wet", out)

    def test_map_is_still_built_when_gd_fails(self):
        with mock.patch.object(rm, "_gd_runs", side_effect=RuntimeError("壊れた")):
            out = self._build()
        self.assertEqual(out["status"], "ok")
        self.assertIn("wet", out)
        self.assertTrue(all(c["gd"] is None for c in self.calls))

    def test_gd_constants_follow_the_evaluation(self):
        from pathlib import Path

        from src.scripts.data import good_distance_eval as ev

        # 評価の選定結果（chosen）は実行ごとの出力 JSON にだけ入り、リポジトリには無い。確定した値は docs に書いてあるので、それと一致を見る
        doc = (Path(__file__).resolve().parents[2] / "docs" / "ideas" / "running_map_axes_fixed.md").read_text(encoding="utf-8")
        self.assertIn(f"相対値 **{rm.GD_TAU:.2f} 以上**", doc)
        self.assertIn(rm.GD_TAU, ev.TAUS)           # 評価の候補の中から採っている
        self.assertEqual(rm.GD_MIN_RUNS, ev.MIN_PRIOR)   # 表示の最低走数も評価と同じ


if __name__ == "__main__":
    unittest.main()
