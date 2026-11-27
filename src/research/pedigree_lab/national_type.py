"""国別の型そのもの（SmartRC の分類: 日・欧・米と clcode）を (A) に足した比べと記述（log.md「調べ H」。開発の行・時間の順）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.national_type --table     # 札の表と付く割合（着順を使わない）
    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.national_type
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.national_type --ae          # 父の札 × 芝ダ × 馬場の A/E（記述）
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.national_type --pair        # 父の札 × 母父の札 × 芝ダの A/E（記述・調べ I）
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.national_type --early       # 2016〜18 年の米・欧 × ダートの道悪の A/E（調べ K）
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.national_type --early2      # 同じ形の logit(q) の 2 次の回帰（調べ K2）

- 札の表: 2020〜24 年の smartrc_race_flat（2025・26 年は開けない） の各行の馬（hcode）について、手元の血統表の父・母父の id に f_・mf_ の札を、
  父の母父（3:2）・母の母父（3:6）の id に fmf_・mmf_ の札を対応させる。同じ種牡馬の id で札が割れたら、多い方を使う。
- 列: 父・母父・父の母父・母の母父 × 国の札（日・欧・米）の印と札の無い印の 16 列と、父・母父の clcode（日を除く 8 種）の印 16 列（nt_*・計 32 列）。
- 判定は版 − (A)（rulebased_wf と同じ窓・2019〜24 年・32 本の割り引き）。帰無は作らない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data

ROOT = Path("/home/jovyan/work/keiba-vpn/data/page_reference/tables")
CC = ["日", "欧", "米"]
CL = [f"300000{i}" for i in range(3, 10)] + ["3000010"]          # 日の型（3000001）は国の印と同じなので入れない（1 号）
SLOTS = {"s": (1, 0), "b": (2, 2), "sb": (3, 2), "db": (3, 6)}
SRC = {"s": "f", "b": "mf", "sb": "fmf", "db": "mmf"}


def stallion_table() -> pd.DataFrame:
    ped = data.load_ped()[["horse_id", "anc5"]]
    pos = {}
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        d = {}
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                d[(int(g), int(p))] = i
        pos[h] = d
    rows = []
    for y in range(2020, 2025):                                       # 2025・26 年は開けない（2 号）
        f = ROOT / str(y) / "smartrc_race_flat.parquet"
        if not f.exists():
            continue
        cols = ["hcode"] + [f"{SRC[k]}_{c}" for k in SRC for c in ("country", "clcode")]
        F = pd.read_parquet(f, columns=cols).drop_duplicates("hcode")
        for r in F.itertuples(index=False):
            d = pos.get(str(r.hcode))
            if not d:
                continue
            rr = r._asdict()
            for k, sl in SLOTS.items():
                sid = d.get(sl)
                if sid:
                    rows.append((sid, rr[f"{SRC[k]}_country"], rr[f"{SRC[k]}_clcode"]))
    T = pd.DataFrame(rows, columns=["sid", "country", "clcode"]).dropna()
    G = T.groupby(["sid", "country", "clcode"]).size().reset_index(name="n")
    split = G.groupby("sid").size()
    print("札が割れる種牡馬", int((split > 1).sum()), "/", int(len(split)), flush=True)      # 後から直した札の目印（1 号）
    T = G.sort_values("n", ascending=False).drop_duplicates("sid")
    return T


def add_cols(X: pd.DataFrame, T: pd.DataFrame) -> list[str]:
    ped = data.load_ped()[["horse_id", "anc5"]]
    ids = {}
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        d = {}
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                d[(int(g), int(p))] = i
        ids[h] = d
    cmap = dict(zip(T["sid"], T["country"]))
    lmap = dict(zip(T["sid"], T["clcode"]))
    new = []
    for k, sl in SLOTS.items():
        sid = X["horse_id"].map(lambda h: ids.get(h, {}).get(sl))
        cty = sid.map(cmap)
        for c in CC:
            X[f"nt_{k}_{c}"] = cty.eq(c).astype(float)
            new.append(f"nt_{k}_{c}")
        X[f"nt_{k}_na"] = cty.isna().astype(float)                    # 札の無い印を明示する（全部 0 と区別。1 号・2 号）
        new.append(f"nt_{k}_na")
        if k in ("s", "b"):
            cl = sid.map(lmap)
            for c in CL:
                X[f"nt_{k}_cl{c[-2:]}"] = cl.eq(c).astype(float)
                new.append(f"nt_{k}_cl{c[-2:]}")
        X[f"_cty_{k}"] = cty
    return new


def main() -> None:
    mode = "table" if "--table" in sys.argv else ("ae" if "--ae" in sys.argv else ("pair" if "--pair" in sys.argv else ("early" if "--early" in sys.argv else ("early2" if "--early2" in sys.argv else "fit"))))
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"national_type_{mode}", "国別の型（SmartRC の分類）の比べと記述"))
    T = stallion_table()
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    if mode == "early":
        # 調べ K: 2016〜18 年の行で、父の札が米・欧のダートの道悪の、同じ人気に対する A/E（全体で割る）
        X = X[(X["date"].dt.year.between(2016, 2018)) & X["split"].isin([0, 1])].reset_index(drop=True)
        add_cols(X, T)
        od = pd.to_numeric(X["odds"], errors="coerce").where(lambda v: v > 0)
        X["q"] = (1 / od) / (1 / od).groupby(X["race_id"]).transform("sum")
        X = X[X["_cty_s"].notna() & X["q"].notna()]
        tot = X["win"].sum() / X["q"].sum()
        rng = np.random.default_rng(3)
        res = {"全体の A/E（割る数）": round(float(tot), 4)}
        for c in ("米", "欧"):
            d = X[(X["_cty_s"] == c) & X["surf"].eq("D") & X["going"].ne("G")]
            sires = d["sire"].to_numpy(); su = np.unique(sires)
            idx = {s_: np.flatnonzero(sires == s_) for s_ in su}
            w, q = d["win"].to_numpy(float), d["q"].to_numpy()
            bs = []
            for _ in range(10000):
                ii = np.concatenate([idx[s_] for s_ in rng.choice(su, len(su))])
                bs.append(w[ii].sum() / q[ii].sum() / tot)
            res[f"{c}・ダ・道悪"] = {"走り": int(len(d)), "勝ち": int(w.sum()), "種牡馬": int(len(su)),
                                  "A/E（同じ人気に対して）": round(float(w.sum() / q.sum() / tot), 3), "95%": np.percentile(bs, [2.5, 97.5]).round(3).tolist()}
            print(c, res[f"{c}・ダ・道悪"], flush=True)
        path = data.OUT / "national_type_early.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        data.finish_all(str(path))
        return
    if mode == "early2":
        # 調べ K2: ダートの道悪の行で、勝ち ~ logit(q) + logit(q)^2 + 米の印 + 欧の印（日が基準）の米の印の係数
        from src.research.pedigree_lab.sire_market import irls
        X = X[X["split"].isin([0, 1]) & (X["date"] <= pd.Timestamp(data.DEV_END))].reset_index(drop=True)
        add_cols(X, T)
        od = pd.to_numeric(X["odds"], errors="coerce").where(lambda v: v > 0)
        X["q"] = (1 / od) / (1 / od).groupby(X["race_id"]).transform("sum")
        X = X[X["_cty_s"].notna() & X["q"].notna() & X["surf"].eq("D") & X["going"].ne("G")].reset_index(drop=True)
        qq = X["q"].clip(1e-4, 1 - 1e-4)
        X["lq"] = np.log(qq / (1 - qq))
        rng = np.random.default_rng(3)
        res = {}
        for nm, (a, b) in (("2016〜18 年（確かめ）", (2016, 2018)), ("2019〜24 年（見つけた期間）", (2019, 2024))):
            D = X[X["date"].dt.year.between(a, b)].reset_index(drop=True)
            lq = D["lq"].to_numpy()
            M = np.column_stack([np.ones(len(D)), lq, lq ** 2, D["_cty_s"].eq("米").to_numpy(float), D["_cty_s"].eq("欧").to_numpy(float)])
            y = D["win"].to_numpy(float)
            b0 = irls(M, y)
            sires = D["sire"].to_numpy(); su = np.unique(sires)
            idx = {s_: np.flatnonzero(sires == s_) for s_ in su}
            bs = []
            for _ in range(2000):
                ii = np.concatenate([idx[s_] for s_ in rng.choice(su, len(su))])
                bs.append(irls(M[ii], y[ii])[3:5])
            bs = np.array(bs)
            res[nm] = {"走り": int(len(D)), "米の印": [round(float(b0[3]), 3), np.percentile(bs[:, 0], [2.5, 97.5]).round(3).tolist()],
                       "欧の印": [round(float(b0[4]), 3), np.percentile(bs[:, 1], [2.5, 97.5]).round(3).tolist()]}
            print(nm, res[nm], flush=True)
        path = data.OUT / "national_type_early2.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        data.finish_all(str(path))
        return
    X = X[(X["date"].dt.year >= 2017) & (X["date"] <= pd.Timestamp(data.DEV_END))].reset_index(drop=True)
    new = add_cols(X, T)
    if mode == "table":
        T.to_parquet(data.OUT / "national_type_table.parquet", index=False)
        print("種牡馬の札", len(T), T["country"].value_counts().to_dict(), flush=True)
        print("付く割合（行）", {k: round(float(X[f"_cty_{k}"].notna().mean()), 3) for k in SLOTS}, flush=True)
        yr = X["date"].dt.year
        print("付く割合（年ごと）", {int(y): {k: round(float(X.loc[yr == y, f"_cty_{k}"].notna().mean()), 3) for k in SLOTS} for y in sorted(yr.unique())}, flush=True)
        data.finish_all(str(data.OUT / "national_type_table.parquet"))
        return
    if mode in ("ae", "pair"):
        P = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
        D = X[X["split"].isin([0, 1])].merge(P, on=["race_id", "horse_id"])
        od = pd.to_numeric(D["odds"], errors="coerce").where(lambda v: v > 0)
        D["q"] = (1 / od) / (1 / od).groupby(D["race_id"]).transform("sum")
        D = D[D["_cty_s"].notna() & D["q"].notna()]
        D["wet"] = np.where(D["going"].eq("G"), "良", "道悪")                 # 道悪は良以外のすべて（稍重・重・不良）
        rng = np.random.default_rng(3)
        res = {}
        keys = ["_cty_s", "_cty_b", "surf"] if mode == "pair" else ["_cty_s", "surf", "wet"]   # --pair: 父の札 × 母父の札 × 芝ダ（調べ I）
        if mode == "pair":
            D = D[D["_cty_b"].notna()]
        tot_a = D["win"].sum() / D["p_A"].sum()                           # 全体は、箱に入る行だけで作る（critic）
        tot_q = D["win"].sum() / D["q"].sum()
        nbox = D.groupby(keys).ngroups                                      # 多重性の箱の数は実際の箱の数（critic）
        for key, d in D.groupby(keys):
            sires = d["sire"].to_numpy(); su, cnt = np.unique(sires, return_counts=True)
            idx = {s_: np.flatnonzero(sires == s_) for s_ in su}
            w, e, q = d["win"].to_numpy(float), d["p_A"].to_numpy(), d["q"].to_numpy()
            ba, bq = [], []
            for _ in range(10000):                                          # 99.6〜99.7% の分位なので 1 万回（critic）
                ii = np.concatenate([idx[s_] for s_ in rng.choice(su, len(su))])
                ba.append(w[ii].sum() / e[ii].sum() / tot_a); bq.append(w[ii].sum() / q[ii].sum() / tot_q)
            lo, hi = 100 * 0.05 / (2 * nbox), 100 * (1 - 0.05 / (2 * nbox))   # 箱の数 × 両側（12 箱で 99.6%・18 箱で 99.7%）
            res[" ".join(map(str, key))] = {
                "走り": int(len(d)), "勝ち": int(w.sum()), "種牡馬": int(len(su)), "最大の種牡馬の割合": round(float(cnt.max() / cnt.sum()), 3),
                "A/E（(A) に対して・全体で割る）": [round(float(w.sum() / e.sum() / tot_a), 3), np.percentile(ba, [lo, hi]).round(3).tolist()],
                "A/E（同じ人気 q に対して・全体で割る）": [round(float(w.sum() / q.sum() / tot_q), 3), np.percentile(bq, [lo, hi]).round(3).tolist()]}
            print(key, res[" ".join(map(str, key))], flush=True)
        path = data.OUT / ("national_type_pair.json" if mode == "pair" else "national_type_ae.json")
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        data.finish_all(str(path))
        return
    from src.research.pedigree_lab.walkforward import fit_year
    use = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + new
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    o = X.loc[te, ["race_id", "horse_id"]].copy()
    o["p_TB"] = p[te]
    path = data.OUT / "national_type_wf.parquet"
    o.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
