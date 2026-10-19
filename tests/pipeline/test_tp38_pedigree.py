"""tp-38 の血統特徴量: 定義どおり（総当たりと一致）、未来・自分の結果に bit 単位で依存しない、位置の区分・近交・類似度の基本性質。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline.features import tp38_pedigree as P
from src.scripts.data.audit_exact_leak import compare, perturb


def _toy(n_horses: int = 24, n_races: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    pool = [f"anc{i:02d}" for i in range(14)]                      # 少ない祖先から選ぶので、クロス（同じ祖先が複数回）が自然に起きる
    horses = [f"h{i:03d}" for i in range(n_horses)]
    rows = []
    for h in horses:
        for g in range(1, 6):
            for p in range(2 ** g):
                rows.append((h, g, p, str(rng.choice(pool))))
    A = pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc_id"])
    runs = []
    for r in range(n_races):
        d = pd.Timestamp("2021-01-09") + pd.Timedelta(days=3 * (r // 2) + (r % 2))
        field = list(rng.choice(horses, size=8, replace=False))
        fin = rng.permutation(8) + 1
        surface = ["芝", "ダート"][r % 2]
        dist = [1200, 1600, 2000][r % 3]
        for k, h in enumerate(field):
            runs.append({"race_id": f"R{r:04d}", "horse_id": h, "date": d, "finish_position": int(fin[k]), "field_size": 8, "surface": surface, "distance": float(dist),
                         "venue_code": ["05", "06"][r % 2], "track_condition": ["良", "重"][r % 3 == 0]})
    B = pd.DataFrame(runs).sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    return A, B


def _setup(seed: int = 0):
    A, B = _toy(seed=seed)
    hi = pd.Index(sorted(set(A["horse_id"])))
    anc, uniq = P.prep_ancestors(A, hi)
    return A, B, hi, anc, uniq


def _brute_cell(B, anc, hi, i, cond_col, k1=30.0, k2=30.0, k_mu=50.0):
    """走 i の px_w を定義から直接計算する（遅い）。"""
    y = P.res_y(B)
    d0 = B["date"].iloc[i]
    ci = cond_col.iloc[i]
    h_i = hi.get_loc(B["horse_id"].iloc[i])
    prior = (B["date"] < d0).to_numpy()
    ok = prior & ~np.isnan(y)
    # 条件の平均 μ（全体へ縮約）
    mu0 = y[ok].mean() if ok.any() else 0.0
    inc = ok & (cond_col == ci).to_numpy()
    mu = (y[inc].sum() + k_mu * mu0) / (inc.sum() + k_mu)
    hcodes = hi.get_indexer(B["horse_id"]).astype(int)
    own = inc & (hcodes == h_i)
    So, No = y[own].sum(), own.sum()
    tab = anc.groupby(["h", "a", "pcls"])["w"].sum().reset_index()
    mine = tab[tab["h"] == h_i]
    num = den = 0.0
    for (a, pc), g in mine.groupby(["a", "pcls"]):
        w = g["w"].sum()
        # a 全体
        hs_a = set(tab[tab["a"] == a]["h"])
        mem_a = inc & np.isin(hcodes, list(hs_a))
        m_a = (y[mem_a].sum() - So + k1 * mu) / (mem_a.sum() - No + k1)
        hs_ap = set(tab[(tab["a"] == a) & (tab["pcls"] == pc)]["h"])
        mem_ap = inc & np.isin(hcodes, list(hs_ap))
        m_ap = (y[mem_ap].sum() - So + k2 * m_a) / (mem_ap.sum() - No + k2)
        num += w * (m_ap - mu)
        den += w
    return num / den


class CellFeaturesTest(unittest.TestCase):
    def test_matches_brute_force(self):
        A, B, hi, anc, uniq = _setup()
        R = P.Runs(B, hi)
        F = P.cell_features(R, anc, "sd", "t")
        cond = P.cond_keys(B)["sd"]
        rng = np.random.default_rng(1)
        for i in rng.choice(np.flatnonzero(B["date"] > B["date"].min() + pd.Timedelta(days=30)), size=6, replace=False):
            self.assertAlmostEqual(float(F["t_w"].iloc[i]), _brute_cell(B, anc, hi, int(i), cond), places=5)

    def test_exact_no_leak(self):
        A, B, hi, anc, uniq = _setup()
        rid = "R0040"
        d0 = B.loc[B["race_id"] == rid, "date"].iloc[0]
        B2 = perturb(B, rid, cols=["finish_position"])
        base = P.cell_features(P.Runs(B, hi), anc, "sd", "t")
        pert = P.cell_features(P.Runs(B2, hi), anc, "sd", "t")
        for nm, f0, f1 in (("cell", base, pert),):
            f0 = f0.assign(race_id=B["race_id"].to_numpy(), horse_id=B["horse_id"].to_numpy())
            f1 = f1.assign(race_id=B2["race_id"].to_numpy(), horse_id=B2["horse_id"].to_numpy())
            r = compare(f0, f1, ["race_id", "horse_id"], B["date"], d0, rid)
            self.assertGreater(r["rows_before"], 0)
            self.assertGreater(r["rows_own"], 0)
            self.assertFalse(r["violations_before"], (nm, "未来の結果に依存", r["violations_before"]))
            self.assertFalse(r["violations_own"], (nm, "自分の結果に依存", r["violations_own"]))

    def test_other_groups_exact_no_leak(self):
        A, B, hi, anc, uniq = _setup(seed=3)
        rid = "R0035"
        d0 = B.loc[B["race_id"] == rid, "date"].iloc[0]
        B2 = perturb(B, rid, cols=["finish_position"])
        H, kinds = P.cross_static(anc, len(hi))
        X, _ = P.horse_vectors(anc, len(hi), len(uniq))
        cl = (np.arange(len(hi)) % 5).astype(np.int32)
        sa, XS, depth = P.stallion_vectors(anc, len(uniq))
        ids = np.sort(anc.loc[anc["pcls"] == 0, "a"].unique())
        nbs = {role: P.neighbors(sa, XS, np.sort(anc.loc[anc["pcls"] == pc, "a"].unique()), np.sort(anc.loc[anc["pcls"] == pc, "a"].unique()), topk=4, min_sim=0.0) for role, pc in P.ROLE_PCLS.items()}
        self.assertGreater(len(ids), 1)

        def build(BB):
            R = P.Runs(BB, hi)
            return {"cross": P.cross_asof(R, anc, kinds), "clu": P.cluster_asof(R, cl, "sd", "k"), "sim": P.sim_features(R, anc, nbs, "sd", "s"),
                    "win": P.cell_features(R, anc, "all", "w", y=P.win_y(BB))}

        a, b = build(B), build(B2)
        for nm in a:
            f0 = a[nm].assign(race_id=B["race_id"].to_numpy(), horse_id=B["horse_id"].to_numpy())
            f1 = b[nm].assign(race_id=B2["race_id"].to_numpy(), horse_id=B2["horse_id"].to_numpy())
            r = compare(f0, f1, ["race_id", "horse_id"], B["date"], d0, rid)
            self.assertGreater(r["rows_before"], 0)
            self.assertFalse(r["violations_before"], (nm, "未来の結果に依存", r["violations_before"]))
            self.assertFalse(r["violations_own"], (nm, "自分の結果に依存", r["violations_own"]))


class StructureTest(unittest.TestCase):
    def test_pos_class_and_names(self):
        self.assertEqual(P.pos_name(1, 0), "S")
        self.assertEqual(P.pos_name(2, 2), "DS")           # 母父
        self.assertEqual(P.pos_name(3, 0), "SSS")
        self.assertEqual(P.pos_name(4, 15), "DDDD")
        pc = P.pos_class([1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3], [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(list(pc), [0, 1, 2, 3, 4, 5, 7, 6, 6, 6, 8, 8, 8, 9])
        self.assertTrue(set(range(P.N_PC)) >= set(P.pos_class(np.repeat(np.arange(1, 6), [2, 4, 8, 16, 32]), np.concatenate([np.arange(2 ** g) for g in range(1, 6)]))))

    def test_wright_inbreeding(self):
        # 同じ祖先が父側 3 世代・母側 4 世代にいる（3x4）→ 0.5^(3+4-1) = 1/64
        rows = [("x", g, p, f"u{g}_{p}") for g in range(1, 6) for p in range(2 ** g)]
        rows = [(h, g, p, "ND" if (g, p) in ((3, 0), (4, 8)) else a) for h, g, p, a in rows]
        A = pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc_id"])
        hi = pd.Index(["x"])
        anc, uniq = P.prep_ancestors(A, hi)
        H, kinds = P.cross_static(anc, 1)
        self.assertAlmostEqual(float(H["x_F"].iloc[0]), 1 / 64)
        self.assertEqual(int(H["x_n"].iloc[0]), 1)
        self.assertEqual(int(H["x_gmin"].iloc[0]), 4)
        self.assertEqual(int(kinds[kinds["kind"] == 2].shape[0]), 1)

    def test_similarity_finds_shared_pedigree(self):
        # 3 頭の種牡馬: s1 と s2 は血統表がほぼ同じ、s3 は別。s1 の最近傍は s2。
        rows = []
        def ped(h, base):
            for g in range(1, 6):
                for p in range(2 ** g):
                    rows.append((h, g, p, base.get((g, p), f"{h}_{g}_{p}")))
        shared = {(2, 0): "a", (2, 1): "b", (3, 0): "c", (3, 1): "d", (3, 2): "e", (3, 3): "f", (4, 0): "g", (4, 1): "h2"}
        # 種牡馬は各 horse の父（gen1 pos0）。その父の血統表（相対世代）が似ているように、gen2 以降を共有させる
        for i, h in enumerate(["k1", "k2", "k3"]):
            ped(h, {(1, 0): f"s{i + 1}", **(shared if i < 2 else {(2, 0): "zz", (2, 1): "yy", (3, 0): "xx"})})
        A = pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc_id"])
        hi = pd.Index(["k1", "k2", "k3"])
        anc, uniq = P.prep_ancestors(A, hi)
        sa, XS, depth = P.stallion_vectors(anc, len(uniq))
        code = {u: i for i, u in enumerate(uniq)}
        ids = np.array([code["s1"], code["s2"], code["s3"]])
        nb = P.neighbors(sa, XS, ids, ids, topk=2, min_sim=0.0)
        top = nb[nb["a"] == code["s1"]].sort_values("sim", ascending=False).iloc[0]
        self.assertEqual(int(top["j"]), code["s2"])
        self.assertGreater(float(top["sim"]), 0.5)


class RidgeTest(unittest.TestCase):
    def _toy_effects(self, seed=0):
        """真の φ（父 1, 母父 0.7, 他は 2^-(gen-1) の半分）と祖先の効果から成績を作り、リッジが位置の重みの大小関係を取り戻すか。"""
        from src.pipeline.features import tp38_ridge as RG

        rng = np.random.default_rng(seed)
        n_h, n_pool = 600, 40
        pool = [f"p{i:02d}" for i in range(n_pool)]
        eff = rng.normal(0, 0.15, size=n_pool)
        true_phi = RG.POS_PRIOR * 0.5
        true_phi[0], true_phi[4] = 1.0, 0.7
        rows = []
        zh = np.zeros(n_h)
        for h in range(n_h):
            for g in range(1, 6):
                for p in range(2 ** g):
                    a = int(rng.integers(0, n_pool))
                    rows.append((f"h{h:04d}", g, p, pool[a]))
                    zh[h] += true_phi[RG.pos_index(g, p)] * eff[a]
        A = pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc_id"])
        runs = []
        for r in range(1500):
            d = pd.Timestamp("2021-01-04") + pd.Timedelta(days=r // 3)
            field = rng.choice(n_h, size=8, replace=False)
            score = zh[field] * 2.5 + rng.normal(0, 0.05, size=8)
            fin = np.empty(8, dtype=int)
            fin[np.argsort(-score)] = np.arange(1, 9)
            for k, h in enumerate(field):
                runs.append({"race_id": f"R{r:05d}", "horse_id": f"h{h:04d}", "date": d, "finish_position": int(fin[k]), "field_size": 8, "surface": "芝", "distance": 1600.0, "venue_code": "05", "track_condition": "良"})
        B = pd.DataFrame(runs).sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
        return A, B, true_phi

    def test_recovers_position_weights_order_and_predicts(self):
        from src.pipeline.features import tp38_ridge as RG

        A, B, true_phi = self._toy_effects()
        hi = pd.Index(sorted(set(A["horse_id"])))
        anc, uniq = P.prep_ancestors(A, hi)
        R = P.Runs(B, hi)

        class S(RG.Spec):
            min_desc_g, min_desc_c, n_fold, n_als = 5, 5, 2, 3

        res, phis = RG.ridge_features(R, anc, len(hi), len(uniq), [("2021-07-01", "2022-01-01"), ("2022-01-01", "2022-07-01")], spec=S)
        phi = np.mean([p["phi"] for p in phis], axis=0)
        self.assertGreater(phi[0], phi[2])                  # 父 > 父父
        self.assertGreater(phi[4], phi[2])                  # 母父 > 父父（真の φ は母父を厚くしてある）
        self.assertGreater(phi[2], phi[30])                 # 近い世代 > 5 世代目
        z = res["als"]["z"]
        ok = z.notna().to_numpy() & ~np.isnan(R.y)
        self.assertGreater(ok.sum(), 3000)
        self.assertGreater(float(np.corrcoef(z[ok], R.y[ok])[0, 1]), 0.25)      # 馬の血統だけで成績の向きを当てられる

    def test_exact_no_leak(self):
        from src.pipeline.features import tp38_ridge as RG

        A, B, _ = self._toy_effects(seed=2)
        hi = pd.Index(sorted(set(A["horse_id"])))
        anc, uniq = P.prep_ancestors(A, hi)

        class S(RG.Spec):
            min_desc_g, min_desc_c, n_fold, n_als = 5, 5, 2, 1

        win = [("2021-07-01", "2022-01-01"), ("2022-01-01", "2022-07-01")]
        rid = "R00800"
        d0 = B.loc[B["race_id"] == rid, "date"].iloc[0]
        self.assertTrue(pd.Timestamp(win[0][0]) <= d0 < pd.Timestamp(win[1][0]))
        B2 = perturb(B, rid, cols=["finish_position"])
        a, _ = RG.ridge_features(P.Runs(B, hi), anc, len(hi), len(uniq), win, spec=S)
        b, _ = RG.ridge_features(P.Runs(B2, hi), anc, len(hi), len(uniq), win, spec=S)
        for nm in ("fix", "als"):
            f0 = a[nm].assign(race_id=B["race_id"].to_numpy(), horse_id=B["horse_id"].to_numpy())
            f1 = b[nm].assign(race_id=B2["race_id"].to_numpy(), horse_id=B2["horse_id"].to_numpy())
            r = compare(f0, f1, ["race_id", "horse_id"], B["date"], d0, rid)
            self.assertGreater(r["rows_own"], 0)
            self.assertFalse(r["violations_before"], (nm, r["violations_before"]))
            self.assertFalse(r["violations_own"], (nm, r["violations_own"]))


if __name__ == "__main__":
    unittest.main()
