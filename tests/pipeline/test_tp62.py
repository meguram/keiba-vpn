"""tp-62（月次過半・年間 100% 超の馬券戦略探索）の部品のテスト。合成のチケット表で、集計・選択・帰無の再現を確かめる。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.scripts.data import tp62_lib as L


def synth(n_races_per_year: int = 40, years=(2022, 2023), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for y in years:
        for k in range(n_races_per_year):
            rid += 1
            month = 1 + (k % 12)
            date = f"{y}-{month:02d}-{1 + (k % 27):02d}"
            race_id = f"{y}01{rid:06d}"
            n = 8
            q = rng.dirichlet(np.ones(n) * 2.0)
            w = rng.dirichlet(np.ones(n) * 2.0)
            base = dict(race_id=race_id, date=date, year=y, n=n, r_p1=float(w.max()), r_kl=0.1, r_ent=1.8, r_q1=float(q.max()),
                        r_dom=0, r_even=0, r_upset=0)
            win = int(rng.choice(n, p=q)) + 1
            # 単勝（全頭）
            for i in range(n):
                odds = 0.8 / q[i]
                rows.append({**base, "bt": 0, "odds": odds, "p_model": w[i], "p_mix": 0.5 * w[i] + 0.5 * q[i], "q": q[i],
                             "ret": odds if (i + 1) == win else 0.0, "h1": i + 1, "h2": 0, "h3": 0})
            # 馬連（上位 3 組）
            order = np.argsort(-q)[:3]
            second = int(rng.choice([j for j in range(1, n + 1) if j != win]))
            for a, b in ((order[0], order[1]), (order[0], order[2]), (order[1], order[2])):
                od = 0.775 / max(q[a] * q[b], 1e-3)
                hit = {a + 1, b + 1} == {win, second}
                rows.append({**base, "bt": 2, "odds": od, "p_model": q[a] * q[b], "p_mix": q[a] * q[b], "q": q[a] * q[b],
                             "ret": od if hit else 0.0, "h1": a + 1, "h2": b + 1, "h3": 0})
    d = pd.DataFrame(rows)
    for c in ("odds", "p_model", "p_mix", "q", "ret", "r_p1", "r_q1"):
        d[c] = d[c].astype(np.float32)
    return d


class TestTp62(unittest.TestCase):
    def setUp(self):
        self.d = L.prepare(synth())
        self.races = L.build_races(self.d)
        self.meta, self.pr, self.pw = L.build_rules(self.d)
        self.grid = L.Grid(self.d, self.races, self.meta, self.pr, self.pw)

    def test_rule_count(self):
        # 券種 2 × スコア 2 × 閾値 6 × オッズ帯 5 × K 4
        self.assertEqual(len(self.meta), 2 * 2 * len(L.THETAS) * 5 * len(L.KS))

    def test_aggregation_matches_bruteforce(self):
        m = self.meta
        u = int(m[(m.bt == 0) & (m.score == "ev_mix") & (m.theta == 1.0) & (m.band == 0) & (m.k == 2)].index[0])
        d = self.d
        sel = d[(d.bt == 0) & (d.ev_mix >= 1.0)].copy()
        sel["rk"] = sel.groupby("ridx")["ev_mix"].rank(method="first", ascending=False)
        sel = sel[sel.rk <= 2]
        want_s = sel.groupby("ridx").size()
        want_r = sel.groupby("ridx")["ret"].sum()
        for r in want_s.index[:10]:
            self.assertAlmostEqual(self.grid.S[u, r], want_s[r])
        # 賭け金と払戻の総計（定額）が一致する（月 × 絞り込み 'all' の合計）
        f_all = self.races.mask_names.index("all")
        self.assertAlmostEqual(float(self.grid.Smon[u, :, f_all].sum()), float(want_s.sum()), places=3)
        self.assertAlmostEqual(float(self.grid.Rmon[u, :, f_all].sum()), float(want_r.sum()), places=2)
        # レース均等: 賭け金は購入のあるレース数、払戻は 1 点あたり平均の合計
        ue = u + self.grid.U
        self.assertAlmostEqual(float(self.grid.Smon[ue, :, f_all].sum()), float(len(want_s)), places=3)
        self.assertAlmostEqual(float(self.grid.Rmon[ue, :, f_all].sum()), float((want_r / want_s).sum()), places=2)

    def test_select_uses_only_training_years(self):
        sel = L.select(self.grid, [2022], "A", nmin=1)
        self.assertIsNotNone(sel)
        u, f, info = sel
        res = L.test_result(self.grid, u, f, 2023)
        self.assertEqual(res["year"], 2023)
        self.assertEqual(res["months"], 12)
        # 訓練年が空・件数が足りない条件では選べない
        self.assertIsNone(L.select(self.grid, [2022], "A", nmin=10_000))

    def test_null_sim_is_calibrated(self):
        sim = L.NullSim(self.d)
        sim.calibrate(n=40, seed=0)
        rng = np.random.default_rng(1)
        acc = np.zeros(6)
        cnt = np.zeros(6)
        for _ in range(60):
            ret = sim.draw_ret(rng)
            for b in (0, 2):
                m = sim.bt == b
                acc[b] += ret[m].sum()
                cnt[b] += m.sum()
        for b in (0, 2):
            self.assertAlmostEqual(acc[b] / cnt[b], L.RATE[b], delta=0.06)

    def test_power_curve_monotone(self):
        sim = L.NullSim(self.d)
        sim.calibrate(n=20, seed=0)
        m = self.meta
        u = int(m[(m.bt == 0) & (m.score == "ev_mix") & (m.theta == 1.0) & (m.band == 0) & (m.k == 3)].index[0])
        f_all = self.races.mask_names.index("all")
        pc = L.power_curve(self.grid, sim, u, f_all, 2023, targets=(0.8, 1.5), reps=60, seed=0)
        self.assertGreaterEqual(pc["1.5"]["A"], pc["0.8"]["A"])


if __name__ == "__main__":
    unittest.main()


class TestTp62R500(unittest.TestCase):
    def test_r500_constraints_hold(self):
        from src.scripts.data import tp62_r500 as R

        d = L.prepare(synth(60, years=(2022, 2023), seed=3))
        races = L.build_races(d)
        pcal = d["p_mix"].to_numpy(np.float32)
        meta, pr, pw, pu = R.build_rules_r500(d, pcal, 0.2, 5.0)
        self.assertEqual(len(meta), pu.shape[0])
        # 選ばれたレース: K 点すべてのオッズが 5K 以上、較正後の的中確率の併合が 20% 以上
        for rule in range(len(meta)):
            rows = pw[pr == rule]
            if len(rows) == 0:
                continue
            sub = d.iloc[rows]
            for r, g in sub.groupby("ridx"):
                k = len(g)
                self.assertTrue((g["odds"].to_numpy() >= 5.0 * k - 1e-6).all())
                self.assertGreaterEqual(pu[rule, r], 0.2 - 1e-6)
        grid = L.Grid(d, races, meta, pr, pw)
        grid.describe = lambda u, f: R.describe(grid, u, f)
        grid.pu = pu
        sel = L.select(grid, [2022], "A", nmin=1)
        if sel is not None:
            u, f, _ = sel
            self.assertIn("p_cal", grid.describe(u, f))
            det = R.detail(grid, u, f, 2023)
            self.assertIn("bet_races", det)
