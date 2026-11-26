"""dropout 型の encoding の第 2 版（log.md「dropout 型の encoding の第 1 版」の次の計画。開発の行・記述・2025・26 は開けない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.dropout2 [--build]

- 置き換えを手で作らず、特徴量の作り（features4.build）で父を隠して作り直す（critic の指摘 6 つへの答え）。
  - 父の 10%（父の id の hash で固定）について、その産駒の血統表の父を「隠し_<父>_<生年>」に替える。
    同じ世代の産駒だけが同じ父の名札を持つので、どの世代も「初年度の産駒しかいない新しい父」として集計される（年の途中からの積み上がりも本物と同じ作り）。
  - 父の父・母父・母などほかの祖先は替えない（本物の新しい父も、父の父は分かっている）。
  - ニックス・全きょうだい・1 頭 1 票・rel_* も、作りの中で新しい父の値になる。
- 時間の順（walkforward と同じ: 2017〜t−2 年で学び t−1 年で止めどころと温度、t−1 年までで学び直す）で、
  学習の行のうち隠した父の産駒の行は作り直した列、それ以外と予測の行は本物の列を使う。採点は rulebased.score と同じ。
- 予想（log）: 新しい父の行で +3〜5%、全体はほぼ変わらない。
- 限界（critic 2 回目）: 父の父・父の父の父の集計には隠した父の他の世代の産駒が本物のまま残る。母・2 代母の集計にも、同じ母が隠した父との間に
  産んだ上の産駒が残る。k は隠した鍵で推定し直される。隠した父の産駒は、学習の行がすべて隠しの列になる（--age2 で 2 歳の年だけに絞れる）。
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, features4, model

FRAC = 0.10
AGE2 = "--age2" in sys.argv


def masked_sires(ped: pd.DataFrame) -> set:
    s = ped["sire"].dropna().astype(str)
    s = s[s != ""].drop_duplicates()
    h = pd.util.hash_pandas_object(s + "_mask", index=False).to_numpy()
    return set(s[(h % 1000) < FRAC * 1000])


def build() -> None:
    r = data.load_dev(reason="dropout 型の第 2 版（父を隠した特徴量の作り直し）", version="feat_v4_mask")
    ped = data.load_ped()
    ms = masked_sires(ped)
    by = ped["horse_id"].str[:4]
    m = ped["sire"].isin(ms)
    ped.loc[m, "sire"] = "隠し_" + ped.loc[m, "sire"].astype(str) + "_" + by[m]
    X, _ = features4.build(r, ped, sub=False)
    X["masked"] = X["horse_id"].isin(set(ped.loc[m, "horse_id"]))
    X.to_parquet(data.OUT / "feat_v4_mask.parquet", index=False)
    print("built", X.shape, int(X["masked"].sum()), flush=True)


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dropout2" + ("_age2" if AGE2 else ""),
                                   "dropout 型の第 2 版（父を隠した特徴量で学ぶ）の時間の順の比べ（尤度だけ・記述）"))
    if "--build" in sys.argv:
        build()
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)
    cols = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")]
    M = pd.read_parquet(data.OUT / "feat_v4_mask.parquet", columns=["race_id", "horse_id", "masked"] + cols)
    M = X[["race_id", "horse_id"]].merge(M, on=["race_id", "horse_id"], how="left")
    masked = M["masked"].fillna(False).to_numpy(bool)
    # --age2: 隠しの列を使うのは各世代の 2 歳の年の走りだけ（「初年度の産駒しかいない父」に当たる行。critic 2 回目の 4）
    age2 = (X["date"].dt.year - pd.to_numeric(X["horse_id"].str[:4], errors="coerce")).to_numpy() == 2
    yr = X["date"].dt.year.to_numpy()
    y = X["win"].to_numpy()
    rid = X["race_id"].to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        tr, va, te = (yr >= 2017) & (yr < t - 1), yr == t - 1, yr == t
        Xm = X[cols].copy()
        sw = (tr | va) & masked & (age2 if AGE2 else True)
        Xm.loc[sw, cols] = M.loc[sw, cols].to_numpy()
        mdl = lgb.train(model.PARAMS, lgb.Dataset(Xm.loc[tr], y[tr]), 2000,
                        valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        pv = mdl.predict(X.loc[va, cols], num_iteration=mdl.best_iteration)
        best = (1e9, 1.0)
        for temp in np.arange(0.6, 1.81, 0.05):
            best = min(best, (model.race_ll(y[va], model.race_norm(pv, rid[va], 1.0, temp), rid[va]), temp))
        m2 = lgb.train(model.PARAMS, lgb.Dataset(Xm.loc[tr | va], y[tr | va]), max(mdl.best_iteration, 50))
        p[te] = model.race_norm(m2.predict(X.loc[te, cols]), rid[te], 1.0, best[1])
        print(t, flush=True)
    te = yr >= 2019
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    out["p_DROP2"] = p[te]
    R = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    T = out.merge(R, on=["race_id", "horse_id"]).merge(X[["race_id", "horse_id", "date", "sire", "win", "n_run", "split"]], on=["race_id", "horse_id"])
    fr = X.groupby("sire")["date"].min()
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")].reset_index(drop=True)
    unseen = (T["sire"].map(fr) >= pd.to_datetime(T["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    w, u = T["win"].to_numpy(float), 1 / T["n_run"].to_numpy(float)

    def bll(yv, pv):
        pv = np.clip(pv, 1e-6, 1 - 1e-6)
        return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
    G = {k: bll(w, u) - bll(w, T[f"p_{k}"].to_numpy()) for k in ("A", "DROP2")}
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(6)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    res = {"masked_rows_share": round(float(masked.mean()), 4)}
    for nm, mk in {"全体": np.ones(len(T), bool), "学習の期間に父の産駒が 1 走も無い": unseen, "それ以外": ~unseen}.items():
        n_d = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        d_d = np.bincount(dinv[mk], weights=(G["DROP2"] - G["A"])[mk], minlength=len(du))
        bs = (Wb @ d_d) / (Wb @ n_d)
        res[nm] = {"rows": int(mk.sum()), "A": round(float(G["A"][mk].mean()), 5), "DROP2": round(float(G["DROP2"][mk].mean()), 5),
                   "DROP2_minus_A": [round(float(d_d.sum() / n_d.sum()), 5), np.percentile(bs, [2.5, 97.5]).round(5).tolist()]}
    path = data.OUT / ("dropout2_age2.json" if AGE2 else "dropout2.json")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
