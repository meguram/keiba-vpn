"""推論用の父・母父キー（serving_parents）と、それを使う ped_* / LayerA の父・母父の面別勝率が、学習側と同じ値になること（合成データ）。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from src.pipeline.features import pedigree_context as pc
from src.pipeline.features import serving_layer_a as sla
from src.pipeline.features import serving_parents as spar
from src.pipeline.features import serving_ped_extras as spe
from tests.pipeline.test_serving_layer_a import _card, _synthetic
from tests.pipeline.test_serving_ped_extras import _runs


def _table(horses: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"horse_id": horses, "sire_id": [f"S{i % 5}" for i in range(len(horses))], "dam_sire_id": [f"D{i % 4}" for i in range(len(horses))]})


class ParentMapTest(unittest.TestCase):
    def test_base_wins_and_unknown_is_empty(self):
        with TemporaryDirectory() as d:
            pm = spar.ParentMap(pd.DataFrame({"horse_id": ["a", "b"], "sire_id": ["S1", ""], "dam_sire_id": ["D1", ""]}), dirs=[Path(d)])
            t = pm.lookup(["a", "b", "zzz"]).set_index("horse_id")
            self.assertEqual((t.loc["a", "sire_id"], t.loc["a", "dam_sire_id"]), ("S1", "D1"))
            self.assertEqual((t.loc["b", "sire_id"], t.loc["zzz", "sire_id"]), ("", ""))             # 表にある空は空のまま（学習側と同じ）。無い馬も空

    def test_new_horse_read_from_json_and_empty_not_cached(self):
        with TemporaryDirectory() as d:
            pm = spar.ParentMap(None, dirs=[Path(d)])
            self.assertEqual(pm.lookup(["2024100001"]).iloc[0]["sire_id"], "")
            p = Path(d) / "2024"
            p.mkdir()
            (p / "2024100001.json").write_text(json.dumps({"ancestors": [{"generation": 1, "position": 0, "horse_id": "SIRE1"}, {"generation": 2, "position": 2, "horse_id": "DAMSIRE1"}]}), encoding="utf-8")
            r = pm.lookup(["2024100001"]).iloc[0]
            self.assertEqual((r["sire_id"], r["dam_sire_id"]), ("SIRE1", "DAMSIRE1"))               # 後から JSON が出来た新馬は次の呼び出しで引ける


class PedStateWithKeysTest(unittest.TestCase):
    def test_matches_training_function_with_real_keys(self):
        h = _runs()
        pm = spar.ParentMap(_table(sorted(h["horse_id"].unique())))
        h = pm.attach(h)
        day = pd.Timestamp("2024-06-01")
        hist, tgt = h[h["date"] < day].reset_index(drop=True), h[h["date"] >= day].reset_index(drop=True)
        full = pd.concat([hist, tgt], ignore_index=True)
        df = full[["date", "race_id", "horse_id", "sire", "dam_sire", "surface", "distance", "track_condition", "venue_code", "finish_position"]].copy()
        ref = pc.build(df, full[["race_id", "horse_id", "pos_early", "l3f"]])
        st = spe.PedState.from_history(hist, blank_pedigree=False)
        t1 = tgt[tgt["date"] == tgt["date"].min()]
        got = st.features(t1)
        np.testing.assert_allclose(got.to_numpy(), ref.loc[len(hist) + t1.index].to_numpy(), rtol=0, atol=0, equal_nan=True)
        blank = spe.PedState.from_history(hist).features(t1)          # 血統名を空にした旧挙動とは、キーの粒度が違うので値が違う
        self.assertFalse(np.allclose(got.to_numpy(), blank.to_numpy(), equal_nan=True))


class LayerASireKeysTest(unittest.TestCase):
    def test_sire_prior_uses_keys(self):
        from src.pipeline.features.track_bias_pedigree import _norm_pedigree_name, _prior_win_rate_by_id_surface, _time_sort_seconds, surface_group

        full = _synthetic()
        target = "202406010604"
        pm = spar.ParentMap(_table(sorted(full["horse_id"].unique())))
        with mock.patch.object(sla, "get_map", return_value=pm), mock.patch.object(sla, "parent_keys_enabled", return_value=True):
            got = sla.build_layer_a_features(_card(full, [target]), full[full["race_id"] != target])
        # 学習側: 全行に ID を付けて同じ関数を当てる
        rr = pm.attach(full.assign(horse_id=full["horse_id"].astype(str))).copy()
        rr["surface_group"] = rr["surface"].map(surface_group)
        rr["_d"] = pd.to_datetime(rr["date"]).dt.normalize()
        rr["_st_sec"] = rr["start_time"].map({v: _time_sort_seconds(v) for v in rr["start_time"].unique()})
        ref = _prior_win_rate_by_id_surface(rr, "sire", "sire")
        g = got.drop(columns=["horse_number"], errors="ignore").merge(full[["race_id", "horse_id", "horse_number"]].astype({"horse_id": str}), on=["race_id", "horse_id"])
        r = g.merge(ref, on=["race_id", "horse_number"], suffixes=("", "_ref"))
        a, b = r["sire_prior_win_rate_surface"].to_numpy(float), r["sire_prior_win_rate_surface_ref"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        self.assertTrue(ok.sum() > 0)
        np.testing.assert_allclose(a[ok], b[ok], rtol=0.05, atol=0.05)
        # 空キー（旧挙動）とは違う
        with mock.patch.object(sla, "parent_keys_enabled", return_value=False):
            old = sla.build_layer_a_features(_card(full, [target]), full[full["race_id"] != target])
        self.assertFalse(np.allclose(old["sire_prior_win_rate_surface"].to_numpy(float), got["sire_prior_win_rate_surface"].to_numpy(float), equal_nan=True))


if __name__ == "__main__":
    unittest.main()
