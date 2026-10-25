"""§77 の想定単勝・想定複勝の推論（odds_sec77）と、それを通す表示経路の単体テスト。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from src.pipeline.inference import odds_sec77 as os77
from src.pipeline.inference import odds_source


def _make_models(d: Path) -> dict:
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["f1", "f2", "f3"]).astype("float32")
    for name, y in (("share", -2.0 + 0.8 * X["f1"] + 0.1 * rng.normal(size=400)), ("place", 1.0 + 0.3 * X["f2"] + 0.1 * rng.normal(size=400))):
        m = lgb.LGBMRegressor(n_estimators=20, learning_rate=0.1, num_leaves=7, min_child_samples=5, verbose=-1)
        m.fit(X, y)
        m.booster_.save_model(str(d / f"{name}_model.txt"))
    meta = {"cols": ["f1", "f2", "f3"], "win_line": [0.1, 0.9], "place_line": [0.05, 1.1], "payback_win": 0.797, "place_floor": 1.05, "place_half_range": 0.1}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


class Sec77Test(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.meta = _make_models(self.dir)
        self.p = mock.patch.object(os77, "MODEL_DIR", self.dir)
        self.p.start()
        os77.reload()

    def tearDown(self):
        self.p.stop()
        os77.reload()
        self.tmp.cleanup()

    def _frame(self):
        rng = np.random.default_rng(1)
        n = 12
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
        X.insert(0, "horse_id", [f"h{i}" for i in range(n)])
        X.insert(0, "race_id", ["202608040801"] * 6 + ["202608040802"] * 6)
        return X

    def test_available_and_columns(self):
        self.assertTrue(os77.available())
        self.assertEqual(os77.feature_columns(), ["f1", "f2", "f3"])
        with mock.patch.dict("os.environ", {"KEIBA_ODDS_SEC77": "0"}):
            self.assertFalse(os77.available())

    def test_predict_matches_eval_transform(self):
        """odds_pred_eval の win_B / place_B と同じ変換（レース内正規化→払戻率→オッズ、複勝は指数→下限）。"""
        from src.scripts.data import odds_pred_eval as E

        X = self._frame()
        R = os77.predict_frame(X)
        pack = os77._load()
        sr = pack["share"].predict(X[["f1", "f2", "f3"]].astype("float32"))
        pr = pack["place"].predict(X[["f1", "f2", "f3"]].astype("float32"))
        rid = X["race_id"].to_numpy()
        a1, b1 = self.meta["win_line"]
        a2, b2 = self.meta["place_line"]
        want_win = E._odds_from_share(rid, E._norm_share(rid, a1 + b1 * sr))
        want_place = np.clip(np.exp(a2 + b2 * pr), 1.05, None)
        np.testing.assert_allclose(R["win_odds"].to_numpy(), want_win, rtol=1e-9)
        np.testing.assert_allclose(R["place_mid"].to_numpy(), want_place, rtol=1e-9)
        self.assertTrue((R["win_odds"] >= 1.0).all())
        self.assertTrue((R["place_min"] <= R["place_mid"]).all() and (R["place_mid"] <= R["place_max"]).all())

    def test_missing_columns_raise(self):
        X = self._frame().drop(columns=["f3"])
        with self.assertRaises(KeyError):
            os77.predict_frame(X)

    def test_payload_and_marker(self):
        X = self._frame()
        R = os77.predict_frame(X)
        g = R[R["race_id"] == "202608040801"]
        hn = {f"h{i}": i + 1 for i in range(6)}
        payload = os77.payload_for_race(g, hn, "テスト特別")
        self.assertEqual(len(payload["entries"]), 6)
        self.assertTrue(all(e["source"] == "sec77" for e in payload["entries"]))
        self.assertEqual([e["horse_number"] for e in payload["entries"]], [1, 2, 3, 4, 5, 6])
        self.assertIsNone(os77.payload_for_race(g, {"h0": 1}, "x"))     # 馬番が揃わなければ保存しない
        # 保存済みキャッシュを predicted_single_odds が sec77 と判定する
        with mock.patch("src.pipeline.inference.final_odds_service.load_cached_response", return_value=payload):
            single = odds_source.predicted_single_odds(object(), "202608040801")
        self.assertEqual(single["odds_model"], "sec77")
        self.assertTrue(os77.is_sec77(single))
        legacy = {"entries": [{**e, "source": "final_odds_ml"} for e in payload["entries"]]}
        with mock.patch("src.pipeline.inference.final_odds_service.load_cached_response", return_value=legacy):
            self.assertEqual(odds_source.predicted_single_odds(object(), "202608040801")["odds_model"], "legacy")

    def test_legacy_calibration_not_applied_to_sec77(self):
        from src.utils import paid_zone as PZ

        single = {"entries": [{"horse_number": 1, "win_odds": 3.0, "place_odds_min": 1.5, "place_odds_max": 2.0}], "odds_model": "sec77"}
        cfg = {"single_calibration": {"win": {"a": 1.0, "b": 1.0}, "place_min": {"a": 1.0, "b": 1.0}}}
        with mock.patch.object(PZ, "config", return_value=cfg):
            self.assertEqual(PZ.calibrate_single(single), single)
            legacy = {**single, "odds_model": "legacy"}
            self.assertNotEqual(PZ.calibrate_single(legacy)["entries"][0]["win_odds"], 3.0)

    def test_legacy_save_does_not_overwrite_sec77(self):
        from src.pipeline.inference import final_odds_service as fo

        sec = {"race_id": "202608040801", "race_name": "x", "elapsed_sec": 0.0, "entries": [{"horse_number": 1, "source": "sec77"}]}
        leg = {"race_id": "202608040801", "race_name": "x", "elapsed_sec": 0.0, "entries": [{"horse_number": 1, "source": "final_odds_ml"}]}
        with mock.patch.object(fo.FinalOddsCache, "load_cached", return_value=sec), mock.patch.object(fo.FinalOddsCache, "save_cached") as sv:
            fo.save_cached_response(object(), "202608040801", leg)
            sv.assert_not_called()
            fo.save_cached_response(object(), "202608040801", sec)       # sec77 の保存は通る
            sv.assert_called_once()
        with mock.patch.object(fo.FinalOddsCache, "load_cached", return_value=None), mock.patch.object(fo.FinalOddsCache, "save_cached") as sv:
            fo.save_cached_response(object(), "202608040801", leg)       # 既存が無ければ従来どおり保存
            sv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
