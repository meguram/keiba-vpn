"""調査スクリプト（学習PC/VPS）と判断ツール（decide）のテスト。"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.pipeline.features.feature_store import FeatureStore
from src.scripts.diagnose import common, decide, diagnose_training_pc as tp, diagnose_vps as vp
from src.scripts.maintenance.make_pseudo_ensemble import train_pseudo_ensemble


def _report(role: str, **checks) -> dict:
    r = common.new_report(role)
    for cid, (status, values) in checks.items():
        common.add_check(r, cid, status, "", **values)
    return r


def _vps(avail=1000, peak=420, swap=2048, blocked=None, gcs_ms=40, db_ms=10, libs=None):
    checks = {
        "system": ("ok", {"mem_available_mb": avail, "swap_total_mb": swap, "disk_free_gb": 60, "libraries": libs or {"lightgbm": "4.6.0"}}),
        "inference": ("ok", {"peak_rss_mb": peak, "model_label": "x"}),
        "gcs_latency": ("ok", {"median_ms": gcs_ms}),
        "cloud_sql_latency": ("ok", {"median_ms": db_ms}),
        "netkeiba": ("ok", {"tried": blocked is not None, "blocked": bool(blocked), "status_code": 403 if blocked else 200}),
    }
    return _report("vps", **checks)


def _training(est=1.0, row_groups=5, model_mb=80, pseudo=False, libs=None):
    return _report(
        "training_pc",
        environment=("ok", {"libraries": libs or {"lightgbm": "4.6.1"}}),
        feature_row_read=("ok", {"est_sec_per_race_1000cols": est, "speedup": 8.0}),
        parquet_layout=("ok", {"row_groups_median": row_groups}),
        model=("ok", {"total_mb": model_mb, "load_sec": 1.2}),
        builder=("warn" if pseudo else "ok", {"is_pseudo": pseudo, "implemented": True, "builder": "pseudo" if pseudo else "real"}),
    )


class DecideRulesTest(unittest.TestCase):
    def test_unknown_without_reports(self):
        for d in decide.decide_all(None, None, None):
            self.assertEqual(d["verdict"], "unknown", d["id"])
            self.assertTrue(d["actions"])

    def test_inference_location_thresholds(self):
        self.assertEqual(decide.decide_inference_location(None, _vps(avail=1000, peak=420))["verdict"], "go")
        self.assertEqual(decide.decide_inference_location(None, _vps(avail=600, peak=420))["verdict"], "caution")
        sw = decide.decide_inference_location(None, _vps(avail=450, peak=420))
        self.assertEqual(sw["verdict"], "switch")
        self.assertTrue(any("GCP" in a for a in sw["actions"]))

    def test_low_swap_adds_action(self):
        d = decide.decide_inference_location(None, _vps(swap=0))
        self.assertTrue(any("スワップ" in a for a in d["actions"]))

    def test_feature_read(self):
        self.assertEqual(decide.decide_feature_read(_training(est=1.0))["verdict"], "go")
        slow = decide.decide_feature_read(_training(est=9.0))
        self.assertEqual(slow["verdict"], "caution")
        self.assertTrue(any("スナップショット" in a for a in slow["actions"]))
        self.assertTrue(any("行グループ" in a for a in decide.decide_feature_read(_training(row_groups=1))["actions"]))

    def test_library_alignment_compares_major_minor(self):
        ok = decide.decide_library_alignment(_training(libs={"lightgbm": "4.6.1"}), _vps(libs={"lightgbm": "4.6.0"}))
        self.assertEqual(ok["verdict"], "go")
        bad = decide.decide_library_alignment(_training(libs={"lightgbm": "4.6.1"}), _vps(libs={"lightgbm": "4.5.0"}))
        self.assertEqual(bad["verdict"], "switch")
        self.assertIn("lightgbm", bad["summary"])

    def test_scraping_location(self):
        self.assertEqual(decide.decide_scraping_location(_vps(blocked=None))["verdict"], "unknown")
        self.assertEqual(decide.decide_scraping_location(_vps(blocked=False))["verdict"], "go")
        self.assertEqual(decide.decide_scraping_location(_vps(blocked=True))["verdict"], "switch")

    def test_page_latency(self):
        self.assertEqual(decide.decide_page_latency(_vps(gcs_ms=40))["verdict"], "go")
        slow = decide.decide_page_latency(_vps(gcs_ms=200, db_ms=80))
        self.assertEqual(slow["verdict"], "caution")
        self.assertEqual(len(slow["actions"]), 2)

    def test_model_size(self):
        self.assertEqual(decide.decide_model_size(_training(model_mb=80), None)["verdict"], "go")
        self.assertEqual(decide.decide_model_size(_training(model_mb=900), None)["verdict"], "caution")

    def test_dynamic_features(self):
        rows = [{"feature": f"f{i}", "guess": "static", "label": "dynamic" if i < 5 else "static"} for i in range(100)]
        self.assertEqual(decide.decide_dynamic_features(rows)["verdict"], "go")
        rows = [{"feature": f"f{i}", "guess": "static", "label": "dynamic" if i < 30 else "static"} for i in range(100)]
        self.assertEqual(decide.decide_dynamic_features(rows)["verdict"], "caution")
        guess_only = [{"feature": f"f{i}", "guess": "dynamic" if i < 50 else "static", "label": ""} for i in range(100)]
        d = decide.decide_dynamic_features(guess_only)
        self.assertEqual(d["evidence"]["unlabeled"], 100)
        self.assertEqual(d["verdict"], "caution")

    def test_builder(self):
        self.assertEqual(decide.decide_builder(_training(pseudo=True))["verdict"], "caution")
        self.assertEqual(decide.decide_builder(_training(pseudo=False))["verdict"], "go")

    def test_markdown_and_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            t, v = Path(tmp, "t.json"), Path(tmp, "v.json")
            common.save_report(_training(), t)
            common.save_report(_vps(blocked=False), v)
            out = Path(tmp, "d.md")
            self.assertEqual(decide.main(["--training", str(t), "--vps", str(v), "--out", str(out)]), 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("T-45推論の実行場所", text)
            self.assertIn("確定", text)


class ReportTest(unittest.TestCase):
    def test_bad_status_rejected_and_errors_do_not_stop(self):
        r = common.new_report("vps")
        with self.assertRaises(ValueError):
            common.add_check(r, "x", "bogus")
        common.run_check(r, "boom", lambda _r: 1 / 0)
        common.run_check(r, "fine", lambda _r: ("ok", "d", {"a": 1}))
        self.assertEqual([c["status"] for c in r["checks"]], ["error", "ok"])

    def test_report_has_no_env_values(self):
        import os

        os.environ["GCS_PRIVATE_KEY"] = "SECRET-VALUE"
        try:
            presence = common.env_presence(["GCS_PRIVATE_KEY", "NOT_SET_XYZ"])
        finally:
            del os.environ["GCS_PRIVATE_KEY"]
        self.assertEqual(presence, {"GCS_PRIVATE_KEY": True, "NOT_SET_XYZ": False})
        self.assertNotIn("SECRET-VALUE", json.dumps(presence))

    def test_schema_version_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "r.json")
            p.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
            with self.assertRaises(ValueError):
                common.load_report(p)


class CollectorSmokeTest(unittest.TestCase):
    """実際の収集関数を、小さな合成データ（特徴量ストア・疑似モデル）で動かす。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.store_dir = root / "store"
        store = FeatureStore(base_dir=str(cls.store_dir))
        rng = np.random.default_rng(0)
        cls.names = ["odds_win", "f_speed", "weight_diff", "f_other"]
        for n in cls.names:
            rows = [(f"2026{r:08d}", f"H{r:04d}{h}", float(rng.random())) for r in range(10) for h in range(6)]
            store.save_feature_column(n, pd.DataFrame(rows, columns=["race_id", "horse_id", n]),
                                      table_block="race_horse_tbl", merge_keys=["race_id", "horse_id"])
        cls.model_dir = root / "model"
        train_pseudo_ensemble(cls.model_dir, 12, n_rows=200)
        cls.args = SimpleNamespace(base_dir=str(cls.store_dir), model_dir=str(cls.model_dir), sample_races=2,
                                   feature_sample=10, layout_sample=5, probe_rows=6, labels_csv=str(root / "labels.csv"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_training_collectors(self):
        status, _, v = tp.check_feature_store(None, self.args)
        self.assertEqual((status, v["n_columns"]), ("ok", 4))
        status, _, v = tp.check_row_read(None, self.args)
        self.assertEqual(status, "ok")
        self.assertEqual(v["result_rows"], 12)
        self.assertIn("est_sec_per_race_1000cols", v)
        status, _, v = tp.check_parquet_layout(None, self.args)
        self.assertIn(status, ("ok", "warn"))
        status, _, v = tp.check_model(None, self.args)
        self.assertEqual(status, "ok")
        self.assertEqual(v["n_features"], 12)
        self.assertGreater(v["peak_rss_mb"], 0)

    def test_label_template_guesses_dynamic_features(self):
        path = tp.write_label_template(self.args)
        rows = {r["feature"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
        self.assertEqual(rows["odds_win"]["guess"], "dynamic")
        self.assertEqual(rows["weight_diff"]["guess"], "dynamic")
        self.assertEqual(rows["f_speed"]["guess"], "static")
        self.assertEqual(rows["odds_win"]["label"], "")

    def test_vps_collectors_without_network(self):
        status, _, v = vp.check_system(None)
        self.assertIn(status, ("ok", "warn"))
        self.assertIn("mem_available_mb", v)
        args = SimpleNamespace(netkeiba_trial=False)
        self.assertEqual(vp.check_netkeiba(None, args)[0], "skip")
        args = SimpleNamespace(model_dir=str(self.model_dir), pseudo_features=12, probe_rows=6)
        status, _, v = vp.check_inference(None, args)
        self.assertEqual(status, "ok")
        self.assertEqual(v["n_features"], 12)

    def test_end_to_end_report_feeds_decide(self):
        training = common.new_report("training_pc")
        common.run_check(training, "environment", tp.check_environment)
        common.run_check(training, "model", lambda r: tp.check_model(r, self.args))
        common.run_check(training, "feature_row_read", lambda r: tp.check_row_read(r, self.args))
        vps = common.new_report("vps")
        common.run_check(vps, "system", vp.check_system)
        common.run_check(vps, "inference", lambda r: vp.check_inference(r, SimpleNamespace(model_dir=str(self.model_dir), pseudo_features=12, probe_rows=6)))
        decisions = {d["id"]: d for d in decide.decide_all(training, vps, None)}
        self.assertNotEqual(decisions["inference_location"]["verdict"], "unknown")
        self.assertNotEqual(decisions["feature_read"]["verdict"], "unknown")
        self.assertNotEqual(decisions["library_alignment"]["verdict"], "unknown")


if __name__ == "__main__":
    unittest.main()
