"""(A) 種牡馬・個体ごとの集計 と (B) 規則で作れる値 の比べ（ユーザーの設計の方針・log.md「(A) 種牡馬ごとの集計 と (B) 規則で作れる値」）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.rulebased [--score]

- 注: 時間の順の学習と予測は、walkforward と同じく保留（split 2）を含む全行で行う。採点（--score）は開発の行だけ。
  系統の上位 30 と祖先 30 頭は、血統表の全馬（2025・26 年に走る馬を含む・着順は使わない）から数えた（critic）。

- (B) の列: 系統の区分（父系・母父系の 4・5 代上の祖先の名札）・多く出る祖先 30 頭の血量・近交・父と母自身の競走成績の形。
- 時間の順（walkforward.fit_year と同じ作り: 年 t を 2017〜t−2 年で学び t−1 年で止めどころと温度、t−1 年までで学び直す）で、
  t = 2019〜2024 の勝ちを (A)・(B)・(A)+(B) で当て、全体と「学習の期間（t より前）に父の産駒が 1 走も無い行」の改善を比べる。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, model
from src.research.pedigree_lab.walkforward import fit_year

N_LINE, N_FOUNDER = 30, 30


def anc_table(ped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                rows.append((h, int(g), int(p), i))
    return pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc"])


def own_record(runs: pd.DataFrame) -> pd.DataFrame:
    """馬自身の競走成績の形（2016 年以降の手元の走り）。"""
    g = runs.groupby("horse_id")
    return pd.DataFrame({"n": g.size(), "winr": g["win"].mean(), "gp": (-runs["fpct"]).groupby(runs["horse_id"]).mean(),
                         "dirt": (runs["surf"] == "D").groupby(runs["horse_id"]).mean(), "dist": g["distance"].mean(),
                         "grade": runs["grade"].isin(["G1", "G2", "G3"]).groupby(runs["horse_id"]).mean()})


def build_b(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ped = data.load_ped()
    T = anc_table(ped)
    T["bw"] = 0.5 ** T["gen"]
    B = pd.DataFrame({"horse_id": ped["horse_id"]})
    for nm, (g, p) in {"pl4": (4, 0), "pl5": (5, 0), "ml4": (4, 8), "ml5": (5, 16)}.items():
        s = T[(T["gen"] == g) & (T["pos"] == p)].set_index("horse_id")["anc"]
        top = s.value_counts().head(N_LINE).index
        B[f"b_{nm}"] = B["horse_id"].map(s).where(lambda v: v.isin(top), "other").fillna("other").astype("category")
    top_f = T.groupby("anc")["horse_id"].nunique().sort_values(ascending=False).head(N_FOUNDER).index
    W = T[T["anc"].isin(top_f)].groupby(["horse_id", "anc"])["bw"].sum().unstack(fill_value=0.0)
    W.columns = [f"b_blood_{i}" for i in range(W.shape[1])]
    B = B.merge(W.reset_index(), on="horse_id", how="left")
    cols = [c for c in B.columns if c != "horse_id"]
    B[[c for c in cols if c.startswith("b_blood_")]] = B[[c for c in cols if c.startswith("b_blood_")]].fillna(0.0)
    runs = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["horse_id", "date", "win", "fpct", "surf", "distance", "grade"])
    runs = runs[runs["date"] <= pd.Timestamp(data.DEV_END)]
    R = own_record(runs)
    ids = ped.set_index("horse_id")[["sire", "dam"]]
    for who in ("sire", "dam"):
        Rw = R.add_prefix(f"b_{who}own_")
        B = B.merge(ids[[who]].reset_index().merge(Rw, left_on=who, right_index=True, how="left").drop(columns=[who]), on="horse_id", how="left")
    cols = [c for c in B.columns if c != "horse_id"]
    X = X.merge(B, on="horse_id", how="left")
    for c in ("b_pl4", "b_pl5", "b_ml4", "b_ml5"):
        X[c] = X[c].astype("category")
    return X, cols + ["ib_F", "ib_SS"]


def score() -> dict:
    """保存した予測（rulebased_wf.parquet）を、行ごとの二値の対数損失（勝ち）で採点し直す（critic の指摘: 帯の比べには勝ち馬だけの物差しを使わない）。
    開発の行（split 0・1）・父が空でない行だけ。差（(A)+(B) − (A)・(B) − (A)）の区間は開催日のブートストラップ 1,000 回。"""
    P = pd.read_parquet(data.OUT / "rulebased_wf.parquet")
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire", "win", "n_run", "split"])
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)]
    fr = X.groupby("sire")["date"].min()
    S = P.merge(X, on=["race_id", "horse_id"])
    S = S[S["split"].isin([0, 1]) & S["sire"].fillna("").ne("")].reset_index(drop=True)
    unseen = (S["sire"].map(fr) >= pd.to_datetime(S["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    late = (S["date"].dt.year >= 2022).to_numpy()
    w, u = S["win"].to_numpy(float), 1 / S["n_run"].to_numpy(float)

    def bll(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    G = {k: bll(w, u) - bll(w, S[f"p_{k}"].to_numpy()) for k in ("A", "B", "AB")}
    day = S["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(3)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    out = {}
    for nm, mk in {"全体": np.ones(len(S), bool), "学習の期間に父の産駒が 1 走も無い": unseen, "それ以外": ~unseen,
                   "新しい父・2022〜24 年": unseen & late}.items():
        r = {"rows": int(mk.sum())}
        n_d = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        for k in ("A", "B", "AB"):
            r[k] = round(float(G[k][mk].mean()), 5)
        for a_, b_ in (("AB", "A"), ("B", "A")):
            d_d = np.bincount(dinv[mk], weights=(G[a_] - G[b_])[mk], minlength=len(du))
            bs = (Wb @ d_d) / (Wb @ n_d)
            r[f"{a_}_minus_{b_}"] = [round(float(d_d.sum() / n_d.sum()), 5), np.percentile(bs, [2.5, 97.5]).round(5).tolist()]
        out[nm] = r
    return out


def main() -> None:
    import sys
    if "--score" in sys.argv:
        data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "rulebased_score", "(A)(B) の予測の採点し直し（尤度だけ・記述）"))
        res = score()
        path = data.OUT / "rulebased_score.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False), flush=True)
        data.finish_all(str(path))
        return
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "rulebased",
                                   "(A) 種牡馬ごとの集計 と (B) 規則で作れる値 の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)
    sets = ablate.col_sets(X)
    X, bcols = build_b(X)
    acols = [c for c in sets["pure"] if not c.startswith("ib_")]
    S = {"A": acols, "B": bcols, "AB": acols + bcols}
    # 父の産駒の走りの重なり: 父の own record は、父自身が 2016 年以降に走った場合だけ値がある
    yr = X["date"].dt.year.to_numpy()
    first_run = X.groupby("sire")["date"].min()
    unseen = (X["sire"].map(first_run) >= pd.to_datetime(X["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    te = yr >= 2019
    res = {"cols": {k: len(v) for k, v in S.items()}, "bcols": bcols,
           "coverage": {"sire_own": round(float(X.loc[te, "b_sireown_n"].notna().mean()), 4),
                        "sire_own_unseen": round(float(X.loc[te & unseen, "b_sireown_n"].notna().mean()), 4),
                        "dam_own": round(float(X.loc[te, "b_damown_n"].notna().mean()), 4)}}
    P = {}
    for nm, cols in S.items():
        p = np.full(len(X), np.nan)
        for t in range(2019, 2025):
            p[yr == t] = fit_year(X, cols, "win", t, 2000)
        P[nm] = p
        print(nm, "done", flush=True)
    rid = X["race_id"].to_numpy()
    w = X["win"].to_numpy(float)
    u = 1 / X["n_run"].to_numpy(float)

    def gain(mk):
        m1 = mk & (w == 1)
        return {k: round(float(np.mean(np.log(np.clip(v[m1], 1e-9, 1)) - np.log(u[m1]))), 4) for k, v in P.items()} | {"wins": int(m1.sum()), "rows": int(mk.sum())}
    res["all"] = gain(te)
    res["unseen_sire"] = gain(te & unseen)
    res["seen_sire"] = gain(te & ~unseen)
    by = pd.to_numeric(X["horse_id"].str[:4], errors="coerce")
    min_by = X.drop_duplicates("horse_id").assign(by=by).groupby("sire")["by"].min()
    first = ((by == X["sire"].map(min_by)) & (X["sire"].map(min_by) >= 2014)).to_numpy()
    res["first_crop"] = gain(te & first)
    res["by_year"] = {int(t): gain(yr == t) for t in range(2019, 2025)}
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    for k, v in P.items():
        out[f"p_{k}"] = v[te]
    out.to_parquet(data.OUT / "rulebased_wf.parquet", index=False)
    path = data.OUT / "rulebased.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "bcols"}, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
