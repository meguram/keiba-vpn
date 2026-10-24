"""tp-39 成長曲線の特徴量: as-of（未来・自分の結果に bit 単位で依存しない）、定数は FIT_YEARS だけから、傾きの手計算との一致、縮約の推定の回復。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp39_growth as tg
from src.scripts.data.audit_exact_leak import compare, perturb
from tests.pipeline import test_factory_self as tself

KEY = ["race_id", "horse_id"]


def _megu(df: pd.DataFrame) -> pd.DataFrame:
    """自前の指数の代用: 着順から決まる値 + 馬・レース固定の端数（結果が変わると値も変わる）。"""
    h = pd.util.hash_pandas_object(df[KEY].astype(str).apply("|".join, axis=1), index=False).to_numpy() % 1000 / 100.0
    v = 100.0 - 6.0 * df["finish_position"].to_numpy(float) + h
    return pd.DataFrame({"race_id": df["race_id"], "horse_id": df["horse_id"], "megu_index": v, "megu_abs": v + 3.0})


def _mid_race(df: pd.DataFrame) -> tuple[str, pd.Timestamp]:
    c = df.groupby("race_id").agg(n=("horse_id", "size"), d=("date", "first")).sort_values("d")
    rid = c.index[len(c) // 2]
    return str(rid), pd.Timestamp(c.loc[rid, "d"])


def _build(df: pd.DataFrame, consts: dict) -> dict[str, pd.DataFrame]:
    return tg.build_groups(tg.prepare(df, _megu(df)), consts)


class Tp39AsOfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = tself._toy(n_races=120, n_horses=14, seed=3)
        d = tg.prepare(cls.df, _megu(cls.df))
        cls.consts = tg.fit_constants(d)

    def test_exact_no_leak_all_groups(self):
        rid, d0 = _mid_race(self.df)
        a = _build(self.df, self.consts)
        b = _build(perturb(self.df, rid), self.consts)
        for g in tg.GROUPS:
            base, pert = a[g], b[g]
            dates = base.merge(self.df[KEY + ["date"]], on=KEY, how="left")["date"]
            r = compare(base, pert, KEY, dates, d0, rid)
            self.assertGreater(r["rows_before"], 0)
            self.assertGreater(r["rows_own"], 0)
            self.assertFalse(r["violations_before"], (g, "未来の結果に依存", r["violations_before"]))
            self.assertFalse(r["violations_own"], (g, "自分の結果に依存", r["violations_own"]))

    def test_truncation_gives_identical_features(self):
        """その日以降のデータを丸ごと捨てても、その日までの行の特徴量は同じ（浮動小数の丸め誤差 1e-9 以内）。
        未来のデータを使っていれば値が大きく変わるので、漏れの検出には十分。bit 単位の一致は、ライブラリの版（合計の順序）で崩れるため求めない。"""
        _, cut = _mid_race(self.df)
        full = _build(self.df, self.consts)
        part = _build(self.df[self.df["date"] <= cut].reset_index(drop=True), self.consts)
        for g in tg.GROUPS:
            m = full[g].merge(part[g], on=KEY, suffixes=("", "__p"))
            self.assertGreater(len(m), 50)
            for c in [c for c in full[g].columns if c not in KEY]:
                x, y = m[c].to_numpy(float), m[c + "__p"].to_numpy(float)
                same = np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
                self.assertTrue(same.all(), (g, c, int((~same).sum())))

    def test_constants_depend_only_on_fit_years(self):
        late = self.df.copy()
        late["date"] = late["date"] + pd.Timedelta(days=365 * 3)           # 2024 年: FIT_YEARS の外
        late["race_id"] = "9" + late["race_id"].str[1:]
        both = pd.concat([self.df, late], ignore_index=True).sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
        c1 = tg.fit_constants(tg.prepare(both, _megu(both)))
        mod = both.copy()
        m_ = mod["date"].dt.year >= 2024
        mod.loc[m_, "finish_position"] = (mod.loc[m_, "finish_position"] % 8) + 1
        mod.loc[m_, "weight"] = mod.loc[m_, "weight"] + 25
        c2 = tg.fit_constants(tg.prepare(mod, _megu(mod)))
        self.assertEqual(c1, c2)

    def test_slope_matches_manual_ols(self):
        d = tg.prepare(self.df, _megu(self.df))
        parts = tg.build_groups(d, self.consts)
        from src.pipeline.features import tp39_growth as T

        dd = T.add_size(T.add_residuals(T.apply_metrics(d, self.consts), self.consts), self.consts)
        S = parts["SL_comp"].reset_index(drop=True)
        dd = dd.reset_index(drop=True)
        checked = 0
        for h, g in dd.groupby("horse_id"):
            idx = g.index.to_list()
            for k in range(6, len(idx)):
                past = dd.loc[idx[:k]]
                ok = past["R_comp"].notna() & past["x_chg"].notna()
                if ok.sum() < 4:
                    continue
                x, y = past.loc[ok, "x_chg"].to_numpy(float), past.loc[ok, "R_comp"].to_numpy(float)
                sxx = ((x - x.mean()) ** 2).sum()
                if sxx <= 1e-3:
                    continue
                beta = ((x - x.mean()) * (y - y.mean())).sum() / sxx
                got = S.loc[idx[k], "gc_chg_comp_beta"]
                self.assertAlmostEqual(float(np.clip(beta, -4, 4)), float(got), places=8)
                self.assertAlmostEqual(float(S.loc[idx[k], "gc_chg_comp_n"]), float(ok.sum()))
                checked += 1
            if checked > 30:
                break
        self.assertGreater(checked, 10)

    def test_first_run_has_no_history_features(self):
        d = tg.prepare(self.df, _megu(self.df))
        parts = tg.build_groups(d, self.consts)
        first = d.groupby("horse_id").cumcount().to_numpy() == 0
        S = parts["SL_comp"]
        self.assertTrue(S.loc[first, "gc_chg_comp_beta"].isna().all())
        self.assertTrue((S.loc[first, "gc_chg_comp_n"] == 0).all())
        K = parts["KR_comp"]
        self.assertTrue(K.loc[first, "gc_kr_comp_adv_lev"].isna().all())
        G = parts["GR_comp"]
        self.assertTrue(G.loc[first, "gc_gr_comp_slope5"].isna().all())


class Tp39MeguTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = tself._toy(n_races=100, n_horses=14, seed=5)

    def test_megu_free_groups_do_not_depend_on_megu(self):
        """megu の値を全く別のものに替えても、megu 非依存の群は bit 単位で同じ（定数も）。"""
        m1 = _megu(self.df)
        m2 = m1.copy()
        m2["megu_index"] = np.random.default_rng(9).normal(50, 20, len(m2))
        m2["megu_abs"] = np.random.default_rng(10).normal(40, 15, len(m2))
        d1, d2 = tg.prepare(self.df, m1), tg.prepare(self.df, m2)
        c1, c2 = tg.fit_constants(d1), tg.fit_constants(d2)
        for m in tg.MEGU_FREE_METRICS:
            self.assertEqual(c1["detrend"][m], c2["detrend"][m])
            self.assertEqual(c1["shrink"][f"{m}_chg"], c2["shrink"][f"{m}_chg"])
            self.assertEqual(c1["pop"]["chg"][m], c2["pop"]["chg"][m])
        g1, g2 = tg.build_groups(d1, c1), tg.build_groups(d2, c2)
        for g in tg.MEGU_FREE_GROUPS:
            pd.testing.assert_frame_equal(g1[g], g2[g], check_exact=True)
        self.assertFalse(g1["SL_meg"].equals(g2["SL_meg"]))                 # megu 系は変わる（対照）

    def test_load_megu_filters_and_records_version(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "2024"
            y.mkdir()
            pd.DataFrame({"race_id": ["R1"] * 6, "horse_id": list("abcdef"), "megu_index_v3a_feat": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0], "megu_index_v3a_abs_feat": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
                          "megu_v3a_status": ["ok", "ok", "restored_clipped", "ok", "ok", None], "megu_v3a_low_confidence": [False, True, False, False, False, None],
                          "megu_v3a_version": ["vX"] * 5 + [None]}).to_parquet(y / "megu_index_flat.parquet")
            m = tg.load_megu(td, [2024])
            self.assertEqual(sorted(m["horse_id"]), ["a", "c", "d"])           # low_confidence・値が欠けた行・状態の無い行を除く
            self.assertEqual(float(m.set_index("horse_id").loc["a", "megu_index"]), 10.0)
            self.assertEqual(float(m.set_index("horse_id").loc["a", "megu_abs"]), 11.0)
            self.assertEqual(m.attrs["megu"]["version"], ["vX"])
            self.assertEqual(m.attrs["megu"]["dir"], Path(td).name)


class Tp39ShrinkageTest(unittest.TestCase):
    def test_fit_shrinkage_recovers_true_slope_variance(self):
        rng = np.random.default_rng(0)
        H, T = 700, 14
        beta = rng.normal(0.3, 0.2, H)
        h = np.repeat(np.arange(H), T)
        x = rng.normal(0, 0.8, H * T)
        y = beta[h] * x + rng.normal(0, 1.0, H * T)
        d = pd.DataFrame({"horse_id": h.astype(str), "year": 2020})
        for m in tg.FEATURE_METRICS + ("tim",):
            d[f"R_{m}"] = y
            for k in tg.KINDS:
                d[f"x_{k}"] = x
        sh = tg.fit_shrinkage(d)
        s = sh["comp_chg"]
        self.assertAlmostEqual(s["beta0"], 0.3, delta=0.07)                    # 標準誤差は約 0.023
        self.assertGreater(s["tau2_raw"], 0.02)
        self.assertLess(s["tau2_raw"], 0.07)                                 # 真値 0.04
        self.assertAlmostEqual(s["sigma2"], 1.0, delta=0.1)

    def test_no_heterogeneity_gives_floor(self):
        rng = np.random.default_rng(1)
        H, T = 700, 14
        h = np.repeat(np.arange(H), T)
        x = rng.normal(0, 0.8, H * T)
        y = 0.1 * x + rng.normal(0, 1.0, H * T)
        d = pd.DataFrame({"horse_id": h.astype(str), "year": 2020})
        for m in tg.FEATURE_METRICS + ("tim",):
            d[f"R_{m}"] = y
            for k in tg.KINDS:
                d[f"x_{k}"] = x
        s = tg.fit_shrinkage(d)["comp_chg"]
        self.assertLess(s["tau2_raw"], 0.01)                                 # 個体差なし → ほぼ 0（雑音の範囲）


if __name__ == "__main__":
    unittest.main()
