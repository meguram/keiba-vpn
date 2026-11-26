"""系統 × その日の時計の帯の target encoding（log.md「系統 × その日の時計の帯の target encoding」。開発の行・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.timeband_te [--null 1]

- 日の帯は turf_going.day_speed（判定するレースの結果は入れない）。速い ≤ −0.5・遅い ≥ +0.5・ほか（欠けを含む）はふつう。
- 目的: −分位・3 着内。系統（ss・L4）× 芝ダ × 日の帯の as-of の平均を、系統 × 芝ダ → 全体へ m = 200 で縮め、基準（全体 × 芝ダ × 日の帯）を引く。
- 時間の順（walkforward.fit_year）で 2019〜24 年の勝ちを当て、rulebased_wf の p_A と比べる（開発の行・父が空でない行・行ごとの二値の改善）。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.lineage import line_ids
from src.research.pedigree_lab.turf_going import day_speed
from src.research.pedigree_lab.walkforward import fit_year

M = 200.0


def te_cols(X: pd.DataFrame, null_seed: int | None) -> tuple[pd.DataFrame, list[str]]:
    X = X.merge(line_ids(), on="horse_id", how="left")
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        H = X.drop_duplicates("horse_id")[["horse_id", "ss", "L4"]].reset_index(drop=True)
        H["by"] = H["horse_id"].str[:4]
        for c in ("ss", "L4"):
            v = H[c].to_numpy().copy()
            for ii in H.groupby("by").indices.values():
                v[ii] = v[ii][rng.permutation(len(ii))]
            X[c] = X["horse_id"].map(dict(zip(H["horse_id"], v)))
    db = day_speed(X)
    X["tband"] = np.where(db <= -0.5, "fast", np.where(db >= 0.5, "slow", "mid"))
    E = X.assign(_g=-X["fpct"].to_numpy(), _t3=X["top3"].astype(float), _k="1")
    cols = []
    for key in ("ss", "L4"):
        for j, v in enumerate(("fpct", "top3")):
            gb, gc = asof(E, ["_k", "surf", "tband"], ["_g", "_t3"])
            base = gb[:, j] / np.maximum(gc, 1)
            s1, c1 = asof(E, [key, "surf"], ["_g", "_t3"])
            g0, c0 = asof(E, ["_k", "surf"], ["_g", "_t3"])
            p1 = (s1[:, j] + M * (g0[:, j] / np.maximum(c0, 1))) / (c1 + M)
            s2, c2 = asof(E, [key, "surf", "tband"], ["_g", "_t3"])
            # 系統 × 芝ダの値に、全体の帯の差（base − 全体 × 芝ダ）を足したものを、系統 × 芝ダ × 帯の縮め先にする
            prior = p1 + (base - g0[:, j] / np.maximum(c0, 1))
            val = (s2[:, j] + M * prior) / (c2 + M)
            X[f"tb_{key}_{v}"] = val - base
            cols.append(f"tb_{key}_{v}")
        _, c2 = asof(E, [key, "surf", "tband"], ["_g"])
        X[f"tb_{key}_n"] = np.log1p(c2)
        cols.append(f"tb_{key}_n")
    return X, cols


def main() -> None:
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "timeband_te" + ("_null" if null_seed is not None else ""),
                                   "系統 × その日の時計の帯の target encoding の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    X, tcols = te_cols(X, null_seed)
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    cols = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + tcols
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, cols, "win", t, 2000)
    te = yr >= 2019
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    out["p_TB"] = p[te]
    R = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    T = out.merge(R, on=["race_id", "horse_id"]).merge(X[["race_id", "horse_id", "date", "sire", "win", "n_run", "split", "surf", "tband"]], on=["race_id", "horse_id"])
    fr = X.groupby("sire")["date"].min()
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")].reset_index(drop=True)
    unseen = (T["sire"].map(fr) >= pd.to_datetime(T["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    w, u = T["win"].to_numpy(float), 1 / T["n_run"].to_numpy(float)

    def bll(yv, pv):
        pv = np.clip(pv, 1e-6, 1 - 1e-6)
        return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
    G = {k: bll(w, u) - bll(w, T[f"p_{k}"].to_numpy()) for k in ("A", "TB")}
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(8)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    res = {"cols": tcols, "score": {}}
    masks = {"全体": np.ones(len(T), bool), "学習の期間に父の産駒が 1 走も無い": unseen,
             "芝・時計の遅い日": (T["surf"] == "T").to_numpy() & (T["tband"] == "slow").to_numpy(),
             "芝・時計の速い日": (T["surf"] == "T").to_numpy() & (T["tband"] == "fast").to_numpy()}
    for nm, mk in masks.items():
        n_d = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        d_d = np.bincount(dinv[mk], weights=(G["TB"] - G["A"])[mk], minlength=len(du))
        bs = (Wb @ d_d) / (Wb @ n_d)
        res["score"][nm] = {"rows": int(mk.sum()), "A": round(float(G["A"][mk].mean()), 5), "TB": round(float(G["TB"][mk].mean()), 5),
                            "TB_minus_A": [round(float(d_d.sum() / n_d.sum()), 5), np.percentile(bs, [2.5, 97.5]).round(5).tolist()]}
    path = data.OUT / ("timeband_te.json" if null_seed is None else f"timeband_te_null{null_seed}.json")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["score"], ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
