import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp64_pace_benefit as pb


def _runs(seed: int = 0, n_races: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    horses = [f"h{i}" for i in range(60)]
    for r in range(n_races):
        year = 2020 + r * 4 // n_races
        date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(r * 1460 / n_races) % 360)
        cell = int(rng.integers(0, 9))
        field = 10
        picks = rng.choice(horses, size=field, replace=False)
        early = rng.uniform(0, 1, size=field)
        fin = rng.permutation(field)
        for k, h in enumerate(picks):
            # スロー縦長(セル 2)では前の馬が良く走る、というシグナルを仕込む
            rows.append({"race_id": f"r{r:04d}", "horse_id": h, "date": date, "year": year, "field_size": field,
                         "early_norm": early[k], "fin_pct": fin[k] / (field - 1),
                         "cell_idx": cell, "style_idx": int(pb.style_id([early[k]])[0])})
    d = pd.DataFrame(rows)
    # セル 2 では前に行った馬の着順を良くする
    m = (d["cell_idx"] == 2) & (d["style_idx"] <= 1)
    d.loc[m, "fin_pct"] = d.loc[m, "fin_pct"] * 0.5
    return d


def _pcell(runs: pd.DataFrame) -> pd.DataFrame:
    ids = runs["race_id"].unique()
    p = np.full((len(ids), 9), 1 / 9)
    return pd.DataFrame(p, columns=[f"p{i}" for i in range(9)]).assign(race_id=ids)


class TestPaceBenefit(unittest.TestCase):
    def test_sign_of_planted_signal(self):
        d = _runs()
        tab = pb.benefit_table(d, 2024)
        self.assertGreater(tab[2, 0], 0.03)       # セル 2 の逃げは恵まれる
        self.assertLess(tab[2, 3], 0.0)           # 追込は相対的に不利

    def test_table_excludes_target_year_and_later(self):
        d = _runs()
        t0 = pb.benefit_table(d, 2022)
        d2 = d.copy()
        late = d2["year"] >= 2022
        d2.loc[late, "fin_pct"] = 1.0 - d2.loc[late, "fin_pct"]
        t1 = pb.benefit_table(d2, 2022)
        np.testing.assert_array_equal(t0, t1)

    def test_features_do_not_use_own_result(self):
        d = _runs()
        pc = _pcell(d)
        f0 = pb.build_features(d, pc, [2022, 2023])
        d2 = d.copy()
        # 2023 年の走の結果を書き換えても、その走自身の特徴は変わらない（次走以降の特徴だけが変わる）
        tgt = d2[d2["year"] == 2023].sort_values("date").iloc[:50]
        d2.loc[tgt.index, "fin_pct"] = 1.0 - d2.loc[tgt.index, "fin_pct"]
        f1 = pb.build_features(d2, pc, [2022, 2023])
        key = ["race_id", "horse_id"]
        first = tgt.sort_values(["date", "race_id"]).groupby("horse_id").head(1)[key]
        cols = [c for c in pb.FEATURE_COLS if c not in ("pb_exp_rel", "pb_exp_rank")]
        a = f0.merge(first, on=key).sort_values(key).reset_index(drop=True)[key + cols]
        b = f1.merge(first, on=key).sort_values(key).reset_index(drop=True)[key + cols]
        self.assertGreater(len(a), 5)
        pd.testing.assert_frame_equal(a, b)

    def test_prior_year_rows_unchanged_by_future(self):
        d = _runs()
        pc = _pcell(d)
        f0 = pb.build_features(d, pc, [2022])
        d2 = d[d["year"] <= 2022].copy()
        f1 = pb.build_features(d2, pc, [2022])
        pd.testing.assert_frame_equal(f0.sort_values(["race_id", "horse_id"]).reset_index(drop=True),
                                      f1.sort_values(["race_id", "horse_id"]).reset_index(drop=True))

    def test_columns_and_first_run_is_nan(self):
        d = _runs()
        f = pb.build_features(d, _pcell(d), [2022])
        self.assertEqual(list(f.columns), ["race_id", "horse_id"] + pb.FEATURE_COLS)
        self.assertTrue(f["pb_nruns"].between(0, 5).all())
        self.assertTrue(f.loc[f["pb_nruns"] == 0, "pb_ben_1"].isna().all())


if __name__ == "__main__":
    unittest.main()
