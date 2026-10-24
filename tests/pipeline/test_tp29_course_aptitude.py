"""tp-29 コース適性（期待とずれた馬のグループ差分）の定義・リーク・基本動作のテスト。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp29_course_aptitude as ca


class GroupRuleTest(unittest.TestCase):
    def _race(self):
        # 3 着の時計 = 100.0
        rows = [
            # race, horse, fin, time, rank
            ("r", "a", 1, 99.5, 8), ("r", "b", 2, 99.8, 1), ("r", "c", 3, 100.0, 7),
            ("r", "d", 4, 100.2, 9),      # 4 着で 3 着との差 0.2 以内 → 期待以上（6 位以降）
            ("r", "e", 5, 100.3, 10),     # 差 0.3 > 0.2 → 期待以上ではない
            ("r", "f", 6, 100.3, 2),      # 6 着以降・差 0.3 以上・2 位 → 期待外れ
            ("r", "g", 7, 100.25, 3),     # 差 0.25 < 0.3 → 期待外れではない
            ("r", "h", 8, 101.0, 5),      # 5 位は対象外
            ("r", "i", -1, 0.0, 4),       # 中止
        ]
        return pd.DataFrame(rows, columns=["race_id", "horse_id", "finish_position", "time_sec", "rank"])

    def test_rules(self):
        d = self._race()
        g = ca.make_groups(d, "rank")
        got = d.assign(g1=g["g1"], g2=g["g2"]).set_index("horse_id")
        self.assertTrue(got.loc["a", "g2"] and got.loc["c", "g2"])
        self.assertFalse(got.loc["b", "g2"])                 # 2 位（人気馬）の 2 着は対象外
        self.assertTrue(got.loc["d", "g2"])
        self.assertFalse(got.loc["e", "g2"])
        self.assertTrue(got.loc["f", "g1"])
        self.assertFalse(got.loc["g", "g1"])
        self.assertFalse(got.loc["h", "g1"] or got.loc["h", "g2"])
        self.assertFalse(got.loc["i", "g1"] or got.loc["i", "g2"])

    def test_no_third_place_means_no_labels(self):
        d = self._race()
        d = d[d["horse_id"] != "c"]
        g = ca.make_groups(d, "rank")
        self.assertFalse(g["g1"].any() or g["g2"].any())


def _toy(n_years: int = 4, per_year: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2020, 2020 + n_years):
        for i in range(per_year):
            sire = rng.choice(["sire=A", "sire=B", "sire=C"])
            k2 = rng.choice(["01|芝|M", "02|芝|L"])
            g = rng.random()
            # k2=01|芝|M では sire=A が期待以上に出やすい
            p2 = 0.6 if (k2 == "01|芝|M" and sire == "sire=A") else 0.06
            g2 = g < p2
            g1 = (not g2) and rng.random() < 0.07
            r = {"race_id": f"{y}{i:04d}", "horse_id": f"h{rng.integers(0, 60)}", "year": y, "date": pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=int(i) % 360),
                 "g1": bool(g1), "g2": bool(g2), "surf": "芝", "k1": k2 + "|1600", "k2": k2, "k3": "芝|M" if k2.endswith("M") else "芝|L"}
            for c in ca.TOKEN_COLS:
                r[c] = np.nan
            r["sire"] = sire
            r["gate"] = rng.choice(["gate=in", "gate=mid", "gate=out"])
            rows.append(r)
    return pd.DataFrame(rows)


class FeatureTest(unittest.TestCase):
    def test_overrepresented_token_gets_positive_significant_score(self):
        F = _toy()
        feats, tables = ca.build_features(F)
        last = max(tables)
        t = tables[last]["k2"]
        row = t[(t["k"] == "01|芝|M") & (t["tok"] == "sire=A")].iloc[0]
        self.assertGreater(row["lor"], 0.3)
        self.assertTrue(bool(row["sig"]))
        r = F[(F["year"] == last) & (F["k2"] == "01|芝|M")]
        a = feats.loc[r[r["sire"] == "sire=A"].index, "apt2_sum"].mean()
        b = feats.loc[r[r["sire"] == "sire=B"].index, "apt2_sum"].mean()
        self.assertGreater(a, b)
        self.assertTrue(feats[F["year"] == 2020]["apt2_sum"].isna().all())      # 最初の年は過去が無い

    def test_no_leak_changing_future_labels_keeps_past_features(self):
        F = _toy()
        a, _ = ca.build_features(F)
        cutoff = pd.Timestamp("2023-01-01")
        G = F.copy()
        late = G["date"] >= cutoff
        rng = np.random.default_rng(1)
        G.loc[late, "g1"] = rng.random(late.sum()) < 0.3
        G.loc[late, "g2"] = ~G.loc[late, "g1"] & (rng.random(late.sum()) < 0.3)
        b, _ = ca.build_features(G)
        early = (F["date"] < cutoff).to_numpy()
        pd.testing.assert_frame_equal(a[early].reset_index(drop=True), b[early].reset_index(drop=True), check_exact=False, atol=1e-9)
        # 2023 年の行（統計は 2022 年以前だけ）の apt 列も、2023 年以降のラベルに依存しない
        y23 = (F["year"] == 2023).to_numpy()
        cols = [c for c in a.columns if c.startswith("apt")]
        pd.testing.assert_frame_equal(a.loc[y23, cols].reset_index(drop=True), b.loc[y23, cols].reset_index(drop=True), check_exact=False, atol=1e-9)

    def test_horse_history_is_strictly_before(self):
        F = pd.DataFrame({"race_id": ["a", "b", "c"], "horse_id": ["h", "h", "h"], "year": 2021, "date": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
                          "g1": [False, True, False], "g2": [True, False, True], "k2": "x", "surf": "芝"})
        hh = ca.horse_history(F)
        self.assertEqual(list(hh["hz_g2_all"]), [0, 1, 1])
        self.assertEqual(list(hh["hz_g1_all"]), [0, 0, 1])


class TokenTest(unittest.TestCase):
    def test_tokens_have_expected_names(self):
        df = pd.DataFrame({"sire_id": ["s1"], "dam_sire_id": [""], "jockey_id": ["j1"], "trainer_id": ["t1"], "a_style": [0.1], "horse_number": [1], "field_size": [16],
                           "bracket_number": [1], "sex_age": ["牡4"], "jockey_weight": [56.0], "weight": [480], "d_wt_chg_now": [0], "d_nprior": [3], "d_rest": [20],
                           "c_cls_step": [0.0], "b_dist_gap": [0.0], "b_n_course": [2]})
        t = ca.attribute_tokens(df)
        self.assertEqual(t.loc[0, "sire"], "sire=s1")
        self.assertTrue(pd.isna(t.loc[0, "dam_sire"]))
        self.assertEqual(t.loc[0, "gate"], "gate=in")
        self.assertEqual(t.loc[0, "age"], "age=a4")
        self.assertEqual(t.loc[0, "rest"], "rest=r_15_35")


if __name__ == "__main__":
    unittest.main()
