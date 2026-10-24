"""特徴量ファクトリー G1（馬自身の履歴・状態）: 先読みが無いこと（将来の結果を変えても過去行の特徴量が変わらない）と、定義の確認。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs

MARGINS = ["", "ハナ", "クビ", "1/2", "3/4", "1", "1.1/4", "2", "大", "アタマ"]


def _toy(n_races: int = 90, n_horses: int = 14, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horses = [f"20190{i:05d}" for i in range(n_horses)]
    rows = []
    for r in range(n_races):
        d = pd.Timestamp("2021-01-09") + pd.Timedelta(days=7 * (r // 2) + (r % 2))
        field = list(rng.choice(horses, size=8, replace=False))
        fin = rng.permutation(8) + 1
        venue = ["05", "06", "08"][r % 3]
        dist = int(rng.choice([1200, 1600, 2000]))
        t0 = dist / 16.5
        for k, h in enumerate(field):
            f = int(fin[k])
            rows.append({
                "race_id": f"2021{venue}{(r // 12) + 1:02d}{(r % 12) + 1:02d}{(r % 12) + 1:02d}", "date": d, "start_time": f"{10 + r % 6}:{(r * 5) % 60:02d}", "venue_code": venue,
                "surface": "芝" if r % 3 else "ダート", "distance": float(dist), "direction": ["右", "左"][r % 2], "grade": ["未勝利", "1勝", "2勝"][r % 3], "race_class": "x",
                "weather": ["晴", "雨"][r % 2], "track_condition": ["良", "稍重", "重"][r % 3], "field_size": 8, "round": (r % 12) + 1, "bracket_number": (k // 2) + 1,
                "horse_number": k + 1, "horse_name": f"ホース{h[-2:]}ー", "horse_id": h, "sex_age": ["牡", "牝", "セ"][int(h[-1]) % 3] + str(3 + int(h[-2]) % 4),
                "jockey_id": f"j{int(rng.integers(0, 4))}", "trainer_id": f"t{int(h[-2:]) % 3}", "jockey_weight": 55.0 + (k % 3), "weight": int(440 + rng.integers(-20, 20)),
                "weight_change": int(rng.integers(-8, 8)), "finish_position": f, "margin": MARGINS[min(f - 1, 9)], "passing_order": "-".join(str(int(rng.integers(1, 9))) for _ in range(3)),
                "last_3f": float(34 + rng.random() * 3), "time_sec": float(t0 + f * 0.2 + rng.random() * 0.1),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)


def _late_modified(df: pd.DataFrame, cut: str) -> pd.DataFrame:
    """cut 以降のレースの『結果』（着順・着差・通過順・上がり・時計）だけを書き換える。出走前に分かる値は変えない。"""
    d2 = df.copy()
    late = d2["date"] >= pd.Timestamp(cut)
    rng = np.random.default_rng(99)
    d2.loc[late, "finish_position"] = (d2.loc[late, "finish_position"] % 8) + 1
    d2.loc[late, "margin"] = rng.choice(MARGINS, size=int(late.sum()))
    d2.loc[late, "passing_order"] = [f"{a}-{b}" for a, b in zip(rng.integers(1, 9, int(late.sum())), rng.integers(1, 9, int(late.sum())))]
    d2.loc[late, "last_3f"] = 33 + rng.random(int(late.sum())) * 4
    d2.loc[late, "time_sec"] = d2.loc[late, "time_sec"] + rng.random(int(late.sum())) * 3
    return d2


class NoLeakTest(unittest.TestCase):
    def test_every_group_is_as_of(self):
        df = _toy()
        bday = pd.Series(pd.to_datetime(["2019-03-23"] * len(df["horse_id"].unique())), index=sorted(df["horse_id"].unique()))
        cut = "2021-06-01"
        attrs = pd.DataFrame({"birthplace": ["新冠町", "浦河町", "安平町"] * 5}, index=sorted(df["horse_id"].unique()) + ["x"])
        a = fs.build_groups(df, birthday=bday, attrs=attrs)
        b = fs.build_groups(_late_modified(df, cut), birthday=bday, attrs=attrs)
        early = (df["date"] < pd.Timestamp(cut)).to_numpy()
        self.assertEqual(set(a), set(fs.GROUPS))
        for g in a:
            x, y = a[g].loc[early].drop(columns=fs.KEY).reset_index(drop=True), b[g].loc[early].drop(columns=fs.KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(x, y, check_exact=False, atol=1e-5, obj=g)

    def test_same_day_results_are_not_used(self):
        # 同じ日の他のレースの結果を変えても、その日の行の特徴量は変わらない（馬は 1 日 1 走）
        df = _toy()
        day = pd.Timestamp("2021-06-05")
        d2 = df.copy()
        m = d2["date"] == day
        d2.loc[m, "finish_position"] = (d2.loc[m, "finish_position"] % 8) + 1
        a, b = fs.build_groups(df), fs.build_groups(d2)
        mm = m.to_numpy()
        for g in ("rest", "career", "lags_fin", "seq_stats", "aff_calendar", "pair"):
            pd.testing.assert_frame_equal(a[g].loc[mm].drop(columns=fs.KEY).reset_index(drop=True), b[g].loc[mm].drop(columns=fs.KEY).reset_index(drop=True), check_exact=False, atol=1e-5, obj=g)


class DefinitionTest(unittest.TestCase):
    def test_parse_margin(self):
        self.assertEqual(fs.parse_margin("1.1/4"), 1.25)
        self.assertEqual(fs.parse_margin("3/4"), 0.75)
        self.assertEqual(fs.parse_margin("クビ"), 0.2)
        self.assertEqual(fs.parse_margin(""), 0.0)
        self.assertEqual(fs.parse_margin("2"), 2.0)
        self.assertTrue(np.isnan(fs.parse_margin("???")))

    def test_rest_and_career(self):
        df = _toy()
        g = fs.build_groups(df, only=["rest", "career"])
        h = df["horse_id"].iloc[0]
        idx = np.flatnonzero((df["horse_id"] == h).to_numpy())
        dates = df.loc[idx, "date"].to_numpy()
        rest = g["rest"].loc[idx, "rest_prev"].to_numpy()
        self.assertTrue(np.isnan(rest[0]))
        np.testing.assert_allclose(rest[1:], (dates[1:] - dates[:-1]) / np.timedelta64(1, "D"))
        n = g["career"].loc[idx, "car_n"].to_numpy()
        np.testing.assert_allclose(n, np.arange(len(idx)))
        fin = df.loc[idx, "finish_position"].to_numpy()
        wins = g["career"].loc[idx, "car_wins"].to_numpy()
        np.testing.assert_allclose(wins, np.concatenate([[0], np.cumsum(fin == 1)[:-1]]))

    def test_age_from_birthday(self):
        df = _toy(30)
        hs = sorted(df["horse_id"].unique())
        bday = pd.Series(pd.to_datetime(["2019-04-10"] * len(hs)), index=hs)
        g = fs.build_groups(df, birthday=bday, only=["age"])["age"]
        d0 = df["date"].iloc[0]
        self.assertAlmostEqual(float(g["age_days"].iloc[0]), float((d0 - pd.Timestamp("2019-04-10")).days), places=3)
        self.assertEqual(float(g["born_month"].iloc[0]), 4.0)

    def test_group_names_and_size(self):
        g = fs.build_groups(_toy(40))
        n = sum(x.shape[1] - 2 for x in g.values())
        self.assertGreater(n, 250)
        for k, v in g.items():
            self.assertFalse(any(("odds" in c or "popul" in c) for c in v.columns), k)


if __name__ == "__main__":
    unittest.main()
