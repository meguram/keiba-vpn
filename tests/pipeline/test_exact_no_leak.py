"""as-of 特徴量が『bit 単位で』未来・自分の結果に依存しないこと（端数リークの回帰テスト）。

「累積和 − 自分」は数学的には『自分より前の合計』だが、浮動小数点では (a + b) − b ≠ a になり得て、端数に自分の値 b の痕跡が残る。
木モデルはその端数（例: 厳密な 0 か 1e-16 か）を拾って今回の結果を当ててしまう（tp-34 で発見）。許容誤差つきの比較では見えないので、ここでは許容しない。
1 レースの結果をずらして、(a) そのレースの日より前の行、(b) そのレース自身の行 の特徴量が完全に同一であることを調べる。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import factory_conn as fc
from src.pipeline.features import factory_field as ff
from src.pipeline.features import factory_self as fs
from src.pipeline.features import historical_entity_stats as hes
from src.scripts.data.audit_exact_leak import compare, perturb
from tests.pipeline import test_factory_conn as tconn
from tests.pipeline import test_factory_field as tfield
from tests.pipeline import test_factory_self as tself

KEY = ["race_id", "horse_id"]


def _mid_race(df: pd.DataFrame) -> tuple[str, pd.Timestamp]:
    c = df.groupby("race_id").agg(n=("horse_id", "size"), d=("date", "first"))
    c = c[c["n"] >= 6].sort_values("d")
    rid = c.index[len(c) // 2]
    return str(rid), pd.Timestamp(c.loc[rid, "d"])


def _assert_exact(base: pd.DataFrame, pert: pd.DataFrame, dates: pd.Series, d0: pd.Timestamp, rid: str, label: str) -> None:
    r = compare(base, pert, KEY, dates, d0, rid)
    assert r["rows_before"] > 0 and r["rows_own"] > 0, (label, r["rows_before"], r["rows_own"])
    assert not r["violations_before"], (label, "未来の結果に依存", r["violations_before"])
    assert not r["violations_own"], (label, "自分の結果に依存", r["violations_own"], r["own_max_abs_diff"])


class PositiveControlTest(unittest.TestCase):
    def test_old_pattern_leaks_residue_and_new_pattern_is_exact(self):
        rng = np.random.default_rng(0)
        n = 4000
        g = np.repeat(np.arange(n), 2)
        a = rng.random(n) * 0.37
        b = rng.random(n) * 1e-3 + 0.1
        v = pd.Series(np.column_stack([a, b]).ravel())
        old = (v.groupby(g).cumsum() - v).to_numpy()[1::2]          # 2 行目の『自分より前の合計』= a のはず
        new = fs._prior_cumsum(v, [g]).to_numpy()[1::2]
        self.assertTrue(np.array_equal(new, a))                      # 新: a と完全に一致
        self.assertGreater(int((old != a).sum()), 0)                 # 旧: 端数に自分の値 b の痕跡が残る行がある（検出できること）


class ExactNoLeakTest(unittest.TestCase):
    def test_factory_self_groups(self):
        df = tself._toy()
        rid, d0 = _mid_race(df)
        names = [n for n in fs.GROUPS if n != "attrs"]
        a = fs.build_groups(df, only=names)
        b = fs.build_groups(perturb(df, rid), only=names)
        for g in a:
            _assert_exact(a[g], b[g], a[g].merge(df[KEY + ["date"]], on=KEY, how="left")["date"], d0, rid, f"self:{g}")

    def test_factory_conn_groups(self):
        df = tconn._toy()
        rid, d0 = _mid_race(df)

        def build(x):
            h = fc.prepare(x)
            groups = fc.build_all(h)
            F = pd.DataFrame({f"{g}:{c}": np.asarray(arr, dtype="float32") for g, d in groups.items() for c, arr in d.items()})
            F["race_id"], F["horse_id"], F["_d"] = h["race_id"].to_numpy(), h["horse_id"].to_numpy(), h["date"].to_numpy()
            return F

        a, b = build(df), build(perturb(df, rid))
        _assert_exact(a.drop(columns=["_d"]), b.drop(columns=["_d"]), a["_d"], d0, rid, "conn")

    def test_factory_field_groups(self):
        df = tfield._toy()
        df["horse_id"] = df["horse_number"].astype(str)                       # 特徴量は馬を特定しないので、馬番を ID 代わりにして行を対応づける
        pace = tfield._pace(df)
        rid, d0 = _mid_race(df)
        d2 = perturb(df, rid, ["finish_position"])
        for name in ff.GROUPS:
            kw = {"pace_table": pace} if name == "race_type" else {}
            x, y = ff.build_group(name, df, **kw), ff.build_group(name, d2, **kw)
            x = x.assign(race_id=df["race_id"].to_numpy(), horse_id=df["horse_id"].to_numpy()) if "horse_id" not in x.columns else x
            y = y.assign(race_id=d2["race_id"].to_numpy(), horse_id=d2["horse_id"].to_numpy()) if "horse_id" not in y.columns else y
            _assert_exact(x.reset_index(drop=True), y.reset_index(drop=True), df["date"], d0, rid, f"field:{name}")

    def test_layer_a_priors(self):
        df = tself._toy().assign(jockey_id=lambda d: d["jockey_id"], trainer_id=lambda d: d["trainer_id"])
        rr = df[["race_id", "horse_number", "horse_id", "jockey_id", "trainer_id", "date", "finish_position"]].copy()
        rid, d0 = _mid_race(rr)

        def build(r):
            t = hes.build_history_feature_table(r)
            return t.merge(r[["race_id", "horse_number", "horse_id"]].drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left").drop(columns=["horse_number"])

        a, b = build(rr), build(perturb(rr, rid, ["finish_position"]))
        _assert_exact(a, b, a.merge(rr[KEY + ["date"]].drop_duplicates(KEY), on=KEY, how="left")["date"], d0, rid, "layer_a_priors")


if __name__ == "__main__":
    unittest.main()
