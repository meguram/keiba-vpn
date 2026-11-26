"""dropout 型の encoding（ユーザーの方針 (5)・11/26。開発の行・記述・2025・26 は開けない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.dropout_enc

- 学習の行（walkforward と同じ: 2017〜t−1 年）のうち、父の 10%（父の id の hash で選ぶ・年ごとに替える）の行で、父の段の列を
  「産駒がまだいない父」の値に置き換えて学ぶ。置き換えの規則（本物の新しい父の行の値に合わせた）:
  - 父の率（sire_<条件>_win/top3/fpct）→ 父の父の同じ条件の率（ss_<条件>_*）、無ければ父の父の全体の率
  - 件数（*_n）→ 0、父の全体との差（*_dfp）・ダート寄り（*_e）・残り（*_res_n）→ 0、距離の幅（sire_dz・sire_dsd）→ 欠け
  - 1 頭 1 票（sire_v_*）→ 欠け、ニックス（nick_*）→ 母父の同じ率・件数 0、rel_*（レースの中の相対の値）→ 0
- 予測の行（年 t）は置き換えない。時間の順で (A) と比べ、新しい父の行・それ以外を出す（rulebased.score と同じ採点）。
- 予想（log の次の計画 (1)）: 新しい父の行の改善が +3〜5%、全体はほぼ変わらない。
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, model

FRAC = 0.10


def dropped(X: pd.DataFrame, cols: list[str], mk: np.ndarray) -> pd.DataFrame:
    D = X.loc[mk, cols].copy()
    for c in cols:
        if c.startswith("rel_sire_") or c.startswith("rel_nick_"):
            D[c] = 0.0
        elif c.startswith("sire_v_"):
            D[c] = np.nan
        elif c in ("sire_dz", "sire_dsd"):
            D[c] = np.nan
        elif c.startswith("sire_") or c.startswith("nick_"):
            parts = c.split("_")
            stat = parts[-1]
            if stat == "n":
                D[c] = 0.0
            elif stat in ("dfp", "e") or c == "sire_res_n":
                D[c] = 0.0
            elif stat in ("win", "top3", "fpct"):
                cond = "_".join(parts[1:-1])
                if c.startswith("nick_"):
                    src = f"bms_{cond}_{stat}"
                    src = src if src in X.columns else f"bms_all_{stat}"
                else:
                    src = f"ss_{cond}_{stat}"
                    src = src if src in X.columns else f"ss_all_{stat}"
                D[c] = X.loc[mk, src].to_numpy()
    return D


def fit_year_drop(X: pd.DataFrame, cols: list[str], t: int, rounds: int = 2000) -> np.ndarray:
    yr = X["date"].dt.year.to_numpy()
    tr, va, te = (yr >= 2017) & (yr < t - 1), yr == t - 1, yr == t
    y = X["win"].to_numpy()
    rid = X["race_id"].to_numpy()
    h = pd.util.hash_pandas_object(X["sire"].fillna("") + f"_{t}", index=False).to_numpy()
    drop = (h % 1000) < FRAC * 1000
    Xm = X[cols].copy()
    m = (tr | va) & drop
    Xm.loc[m, cols] = dropped(X, cols, m).to_numpy()
    mdl = lgb.train(model.PARAMS, lgb.Dataset(Xm.loc[tr], y[tr]), rounds,
                    valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pv = mdl.predict(X.loc[va, cols], num_iteration=mdl.best_iteration)
    best = (1e9, 1.0)
    for temp in np.arange(0.6, 1.81, 0.05):
        q = model.race_norm(pv, rid[va], 1.0, temp)
        best = min(best, (model.race_ll(y[va], q, rid[va]), temp))
    m2 = lgb.train(model.PARAMS, lgb.Dataset(Xm.loc[tr | va], y[tr | va]), max(mdl.best_iteration, 50))
    return model.race_norm(m2.predict(X.loc[te, cols]), rid[te], 1.0, best[1])


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dropout_enc",
                                   "dropout 型の encoding の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)
    cols = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")]
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year_drop(X, cols, t)
        print(t, flush=True)
    te = yr >= 2019
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    out["p_DROP"] = p[te]
    R = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A"]]
    T = out.merge(R, on=["race_id", "horse_id"]).merge(X[["race_id", "horse_id", "date", "sire", "win", "n_run", "split"]], on=["race_id", "horse_id"])
    fr = X.groupby("sire")["date"].min()
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")].reset_index(drop=True)
    unseen = (T["sire"].map(fr) >= pd.to_datetime(T["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    w, u = T["win"].to_numpy(float), 1 / T["n_run"].to_numpy(float)

    def bll(yv, pv):
        pv = np.clip(pv, 1e-6, 1 - 1e-6)
        return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
    G = {k: bll(w, u) - bll(w, T[f"p_{k}"].to_numpy()) for k in ("A", "DROP")}
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(5)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    res = {}
    for nm, mk in {"全体": np.ones(len(T), bool), "学習の期間に父の産駒が 1 走も無い": unseen, "それ以外": ~unseen}.items():
        n_d = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        d_d = np.bincount(dinv[mk], weights=(G["DROP"] - G["A"])[mk], minlength=len(du))
        bs = (Wb @ d_d) / (Wb @ n_d)
        res[nm] = {"rows": int(mk.sum()), "A": round(float(G["A"][mk].mean()), 5), "DROP": round(float(G["DROP"][mk].mean()), 5),
                   "DROP_minus_A": [round(float(d_d.sum() / n_d.sum()), 5), np.percentile(bs, [2.5, 97.5]).round(5).tolist()]}
    path = data.OUT / "dropout_enc.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
