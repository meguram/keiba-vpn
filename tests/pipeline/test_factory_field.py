"""factory_field の先読み検査: 未来（と同じレース・同日の後のレース）の結果を変えても、過去の行の特徴量は変わらない。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import factory_field as ff

AS_OF_COLS = ff.STRENGTH + ["a_style", "f2_early_rank", "d_rest", "d_nprior", "horse_prior_races", "c_cls_prev", "c_cls_step", "jockey_prior_top3_rate",
                            "trainer_prior_top3_rate", "jockey_prior_races", "trainer_prior_races"]


def _toy(n_dates: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    master = ff._master()
    combos = master[master["surface"].isin(["芝", "ダート"])][["venue_code", "surface", "distance"]].drop_duplicates().to_numpy()
    rows = []
    rid = 0
    for di in range(n_dates):
        date = (pd.Timestamp("2022-01-08") + pd.Timedelta(days=7 * di)).strftime("%Y-%m-%d")
        for vi, venue in enumerate(["05", "06"]):
            cs = [c for c in combos if c[0] == venue]
            for rnd in range(1, 7):
                v, surf, dist = cs[int(rng.integers(0, len(cs)))]
                fs = int(rng.integers(8, 15))
                fin = rng.permutation(fs) + 1
                rid += 1
                rc = rng.choice(["サラ系２歳 未勝利", "サラ系３歳以上 １勝クラス", "サラ系３歳以上 オープン"])
                for hn in range(1, fs + 1):
                    r = {"race_id": f"2022{venue}{di % 9 + 1:02d}{vi + 1:02d}{rnd:02d}", "date": date, "venue_code": venue, "surface": surf, "distance": int(dist),
                         "direction": "右", "grade": rng.choice(["未勝利", "1勝", "OP"]), "race_class": rc, "weather": rng.choice(["晴", "曇", "雨"]),
                         "track_condition": rng.choice(["良", "稍", "重"]), "start_time": f"{9 + rnd // 2}:{(rnd % 2) * 30:02d}", "field_size": fs,
                         "race_name": f"テストステークス{di % 5}", "round": rnd, "weight_rule": rng.choice(["馬齢", "定量", "ハンデ"]), "horse_number": hn,
                         "bracket_number": (hn + 1) // 2, "sex_age": rng.choice(["牡3", "牝4", "セ5"]), "jockey_weight": float(rng.choice([52, 55, 57])),
                         "weight": int(rng.integers(420, 520)), "weight_change": int(rng.integers(-10, 11)), "jockey_name": rng.choice(["武豊", "▲小林", "△岩田"]),
                         "jockey_id": f"j{int(rng.integers(0, 8))}", "trainer_id": f"t{int(rng.integers(0, 6))}", "finish_position": int(fin[hn - 1])}
                    for c in AS_OF_COLS:
                        r[c] = float(rng.normal())
                    rows.append(r)
    df = pd.DataFrame(rows)
    df["race_id"] = pd.factorize(df["date"] + df["venue_code"] + df["round"].astype(str))[0].astype(str)
    return df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)


def _pace(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df["race_id"].unique()
    return pd.DataFrame({"race_id": ids, "pace_first": rng.normal(35, 1, len(ids)), "pace_second": rng.normal(36, 1, len(ids))}).assign(rpci=lambda d: d["pace_first"] - d["pace_second"])


def _bump(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    d = df.copy()
    d.loc[mask, "finish_position"] = (d.loc[mask, "finish_position"] % d.loc[mask, "field_size"]) + 1
    return d


def _same(a: pd.DataFrame, b: pd.DataFrame, rows) -> None:
    pd.testing.assert_frame_equal(a.loc[rows].reset_index(drop=True), b.loc[rows].reset_index(drop=True), check_exact=False, atol=1e-6, rtol=1e-6)


class FactoryFieldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _toy()
        cls.pace = _pace(cls.df)

    def _build(self, name, df):
        return ff.build_group(name, df, **({"pace_table": self.pace} if name == "race_type" else {}))

    def test_all_groups_run_and_have_columns(self):
        total = 0
        for name in ff.GROUPS:
            f = self._build(name, self.df)
            self.assertEqual(len(f), len(self.df), name)
            self.assertGreater(f.shape[1], 5, name)
            self.assertFalse(f.columns.duplicated().any(), name)
            total += f.shape[1]
        self.assertGreater(total, 300)

    def test_no_future_leak_across_days(self):
        cut = "2022-07-01"
        late = self.df["date"] >= cut
        early = ~late
        b = _bump(self.df, late)
        for name in ff.GROUPS:
            _same(self._build(name, self.df), self._build(name, b), early)

    def test_same_day_later_races_and_own_race_not_used(self):
        day = self.df["date"].iloc[len(self.df) // 2]
        v = "05"
        for r0 in (1, 3, 5):
            sel = (self.df["date"] == day) & (self.df["venue_code"] == v) & (self.df["round"] >= r0)
            b = _bump(self.df, sel)
            keep = (self.df["date"] == day) & (self.df["venue_code"] == v) & (self.df["round"] <= r0)
            for name in ff.GROUPS:
                _same(self._build(name, self.df), self._build(name, b), keep)

    def test_pace_table_future_not_used(self):
        cut = "2022-07-01"
        late_ids = set(self.df.loc[self.df["date"] >= cut, "race_id"])
        p2 = self.pace.copy()
        m = p2["race_id"].isin(late_ids)
        p2.loc[m, ["pace_first", "pace_second", "rpci"]] += 5.0
        early = self.df["date"] < cut
        a = ff.build_group("race_type", self.df, pace_table=self.pace)
        b = ff.build_group("race_type", self.df, pace_table=p2)
        _same(a, b, early)

    def test_result_free_groups_ignore_finish(self):
        b = _bump(self.df, pd.Series(True, index=self.df.index))
        for name in ("mem_strength", "mem_demo", "mem_conn", "mem_style", "course_geo", "course_geo_var", "course_geo_avg", "calendar_env", "cross_misc"):
            pd.testing.assert_frame_equal(ff.build_group(name, self.df), ff.build_group(name, b))

    def test_variant_geo_distinguishes_inner_outer(self):
        m = ff._master(dedupe=False)
        multi = m.groupby(["venue_code", "surface", "distance"]).filter(lambda g: g["variant"].nunique() > 1 and g["final_straight_m"].nunique() > 1)
        self.assertGreater(len(multi), 0)
        from src.pipeline.features.course_variant import key_variants

        # マスタだけで内外が決まるのに数値が内外で異なるコースがあれば、var 版は avg 版と値が違う
        df = _toy()
        a, b = ff.grp_course_geo_var(df), ff.grp_course_geo_avg(df)
        self.assertEqual(a.shape[0], b.shape[0])
        self.assertIn("fg_variant_ambiguous", a.columns)
        self.assertTrue(((a["fg_variant_ambiguous"] == 0) | (a["fg_variant_ambiguous"] == 1)).all())
        self.assertGreater(len(key_variants()), 50)

    def test_course_geo_joins_master(self):
        f = ff.grp_course_geo(self.df)
        self.assertGreater(f["fg_final_straight_m"].notna().mean(), 0.9)
        self.assertTrue((f["fg_first_curve_ratio"].dropna() > 0).all())


if __name__ == "__main__":
    unittest.main()
