"""系統の階層の版（ユーザーの問い・11/26。開発の行・記述・2025・26 は開けない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.lineage [--course [--null 1]]

- --course: 系統 × コースの target encoding（ユーザーの具体案）を第 4 版に足した版 A_CTE（先に --course なしを流しておく）。--null は系統の札の帰無。

- 系統の段（父系の祖先の位置で決める規則。名前の一覧は使わない）: 中系統 = 父の父（ss）・その上 = 父の父の父（sss）・大系統 = 4 代上の父系の祖先（ssss）。
  第 4 版には ss（全体・芝ダ・距離帯・馬場・年齢 × 芝ダ）と sss（全体・芝ダ）の as-of の集計が既にある。ここでは大系統と、母父系の 4 代上（4:8）の集計を足す。
- 大系統の列: (ssss)・(ssss × 芝ダ)・(ssss × 芝ダ × 距離帯)・(ssss × 芝ダ × 馬場)・(ssss × 芝ダ × レースの年齢条件) の勝率と −分位の as-of の平均
  （自分を引く・擬似件数 200 で同じ条件の全体へ縮める）。母父系は (dsss × 芝ダ)・(dsss × 芝ダ × 距離帯)。
- 版: A = 第 4 版の pure（189+2 列）・LINE = A から父の個体の列（sire_*・rel_sire_*・nick_*・rel_nick_*・full_*・rel_full*）を外し、大系統の列を足す
  （父は系統の段だけで表す）・A_LINE = A ＋ 大系統の列。
- 時間の順（walkforward.fit_year）で 2019〜24 年の勝ちを当て、開発の行で行ごとの二値の改善を採点する（rulebased.score と同じ作り）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.walkforward import fit_year

K = 200.0
CONDS = {"all": [], "surf": ["surf"], "sd": ["surf", "dband"], "sg": ["surf", "going"], "ra": ["surf", "rage"]}
INDIV = ("sire_", "rel_sire_", "nick_", "rel_nick_", "full_", "rel_full")


def line_ids() -> pd.DataFrame:
    ped = data.load_ped()[["horse_id", "anc5"]]
    out = {"horse_id": [], "L4": [], "M4": []}
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        d = {}
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                d[(int(g), int(p))] = i
        out["horse_id"].append(h)
        out["L4"].append(d.get((4, 0), ""))
        out["M4"].append(d.get((4, 8), ""))
    return pd.DataFrame(out)


def line_cols(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = X.merge(line_ids(), on="horse_id", how="left")
    g = -X["fpct"].to_numpy()
    E = X.assign(_g=g, _w=X["win"].astype(float), _k="1")
    cols = []
    for key, conds in (("L4", CONDS), ("M4", {"surf": ["surf"], "sd": ["surf", "dband"]})):
        for cn, ck in conds.items():
            s, c = asof(E, [key] + ck, ["_g", "_w"])
            b, cb = asof(E, ["_k"] + ck, ["_g", "_w"])
            for j, v in enumerate(("fpct", "win")):
                base = b[:, j] / np.maximum(cb, 1)
                X[f"ln_{key}_{cn}_{v}"] = (s[:, j] + K * base) / (c + K) - base
                cols.append(f"ln_{key}_{cn}_{v}")
            X[f"ln_{key}_{cn}_n"] = np.log1p(c)
            cols.append(f"ln_{key}_{cn}_n")
    return X, cols


def _asof_stats(E: pd.DataFrame, keys: list[str], j: int) -> tuple[np.ndarray, np.ndarray]:
    """as-of の平均と件数（その日より前・自分を引く）。"""
    s_, c_ = asof(E, keys, ["_g", "_w"])
    return s_[:, j] / np.maximum(c_, 1), c_


KF_FOLDS = None


def _kfold_stats(E: pd.DataFrame, keys: list[str], j: int) -> tuple[np.ndarray, np.ndarray]:
    """普通の k 分割の TE（開催の単位の 5 分割。ほかの折の全期間の行で作る。未来の行が入る＝漏れの型の比べ用）。"""
    v = E[["_g", "_w"][j]].to_numpy(float)
    k = pd.MultiIndex.from_frame(E[keys].astype(str)).factorize()[0] if len(keys) > 1 else pd.factorize(E[keys[0]].astype(str))[0]
    f = KF_FOLDS
    tot_s = np.bincount(k, weights=v)
    tot_c = np.bincount(k).astype(float)
    m, c = np.zeros(len(E)), np.zeros(len(E))
    for q in np.unique(f):
        mk = f == q
        fs = np.bincount(k[mk], weights=v[mk], minlength=len(tot_s))
        fc = np.bincount(k[mk], minlength=len(tot_c)).astype(float)
        ss, cc = (tot_s - fs)[k[mk]], (tot_c - fc)[k[mk]]
        m[mk], c[mk] = ss / np.maximum(cc, 1), cc
    return m, c


def course_te(X: pd.DataFrame, null_seed: int | None = None, kfold_te: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """系統（中 = 父の父・大 = 父系の 4 代上）× コース（場 × 芝ダ × 距離）の target encoding（勝ち・−分位）。
    as-of（その日より前・自分を引く）で、系統 × コース → 系統 × 芝ダ × 距離帯 → 系統 × 芝ダ → 全体の順に、擬似件数 m = 200 で縮める。
    新しいコース・新しい系統の組は、上の段の値で埋まる。市場の残差の版は、オッズを特徴量の元に使わない決まりなので作らない。
    null_seed: 系統の札を、同じ大きさ（産駒の頭数の十分位）の系統どうしで、馬の単位で入れ替える帰無。"""
    X = X.copy()
    if "L4" not in X:
        X = X.merge(line_ids(), on="horse_id", how="left")
    X["course"] = X["venue"].astype(str) + "_" + X["surf"].astype(str) + "_" + X["distance"].astype(str)
    if null_seed is not None:
        # 帰無: 同じ生年の馬の中で、系統の札を馬の単位で入れ替える（札ごとの頭数はそのまま。大きな系統も入れ替わる。critic の指摘 1）
        rng = np.random.default_rng(null_seed)
        H = X.drop_duplicates("horse_id")[["horse_id", "ss", "L4"]].reset_index(drop=True)
        H["by"] = H["horse_id"].str[:4]
        for c in ("ss", "L4"):
            v = H[c].to_numpy().copy()
            for ii in H.groupby("by").indices.values():
                v[ii] = v[ii][rng.permutation(len(ii))]
            X[c] = X["horse_id"].map(dict(zip(H["horse_id"], v)))
    E = X.assign(_g=-X["fpct"].to_numpy(), _w=X["win"].astype(float), _k="1")
    levels = [["surf"], ["surf", "dband"], ["course"]]          # 上の段から
    cols = []
    stat = _kfold_stats if kfold_te else _asof_stats
    for key in ("ss", "L4"):
        for j, v in enumerate(("fpct", "win")):
            # 基準（系統によらない場の水準）: 全体 → 芝ダ → 距離帯 → コースへ同じ m で縮める
            gb, gc = stat(E, ["_k"], j)
            base = gb
            for lv in levels:
                s_, c_ = stat(E, ["_k"] + lv, j)
                base = (s_ * c_ + 200.0 * base) / (c_ + 200.0)
            # 系統の値: 系統 × 芝ダ → × 距離帯 → × コース。最初の縮め先は芝ダの基準
            sb, cb = stat(E, ["_k", "surf"], j)
            prior = (sb * cb + 200.0 * gb) / (cb + 200.0)
            for lv in levels:
                s_, c_ = stat(E, [key] + lv, j)
                prior = (s_ * c_ + 200.0 * prior) / (c_ + 200.0)
            X[f"cte_{key}_{v}"] = prior - base
            cols.append(f"cte_{key}_{v}")
        _, c_ = stat(E, [key, "course"], 0)
        X[f"cte_{key}_n"] = np.log1p(c_)
        cols.append(f"cte_{key}_n")
    return X, cols


def main() -> None:
    import sys as _s
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "lineage" + ("_course" if "--course" in _s.argv else "") + ("_kfold" if "--kfold-te" in _s.argv else "") + ("_null" if "--null" in _s.argv else ""),
                                   "系統の階層の版（大系統の as-of の集計・父の個体の列を外した版）の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)          # 2016 年の走りも as-of の元に入れる
    import sys
    course = "--course" in sys.argv
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    kfold_te = "--kfold-te" in sys.argv
    X, lcols = line_cols(X)
    if course:
        global KF_FOLDS
        mt = X["race_id"].str[:8]
        u = pd.Series(mt.unique())
        rngf = np.random.default_rng(20261126)
        KF_FOLDS = mt.map(dict(zip(u, rngf.integers(0, 5, len(u))))).to_numpy()
        X, ccols = course_te(X, null_seed, kfold_te)
    X = X[(X["date"].dt.year >= 2017)].reset_index(drop=True)
    pure = ablate.col_sets(X)["pure"]
    if course:
        # 比べる相手の p_A（rulebased）は ib_ を外した 189 列なので、ここも揃える（critic の指摘 2）
        S = {"A_CTE": [c for c in pure if not c.startswith("ib_")] + ccols}
    else:
        S = {"LINE": [c for c in pure if not c.startswith(INDIV)] + lcols, "A_LINE": pure + lcols}
    tag = "" if not course else ("_course" + ("_kfold" if kfold_te else "") + ("" if null_seed is None else f"_null{null_seed}"))
    yr = X["date"].dt.year.to_numpy()
    te = yr >= 2019
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    for nm, cols in S.items():
        p = np.full(len(X), np.nan)
        for t in range(2019, 2025):
            p[yr == t] = fit_year(X, cols, "win", t, 2000)
        out[f"p_{nm}"] = p[te]
        print(nm, len(cols), "done", flush=True)
    # A は rulebased の時間の順の予測（同じ作り）をそのまま使う
    R = pd.read_parquet(data.OUT / "rulebased_wf.parquet")[["race_id", "horse_id", "p_A", "p_B"]]
    out = out.merge(R, on=["race_id", "horse_id"], how="left")
    if course:
        prev = pd.read_parquet(data.OUT / "lineage_wf.parquet")[["race_id", "horse_id", "p_LINE", "p_A_LINE"]]
        out = out.merge(prev, on=["race_id", "horse_id"], how="left")
    out.to_parquet(data.OUT / f"lineage_wf{tag}.parquet", index=False)
    # 採点（開発の行・父が空でない行・行ごとの二値の改善・差の区間は開催日のブートストラップ）
    Y = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire", "win", "n_run", "split"])
    Y = Y[(Y["date"].dt.year >= 2017) & (Y["date"].dt.year <= 2024)]
    fr = Y.groupby("sire")["date"].min()
    T = out.merge(Y, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & T["sire"].fillna("").ne("")].reset_index(drop=True)
    unseen = (T["sire"].map(fr) >= pd.to_datetime(T["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    w, u = T["win"].to_numpy(float), 1 / T["n_run"].to_numpy(float)

    def bll(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    names = ["A", "B", "LINE", "A_LINE"] + (["A_CTE"] if course else [])
    G = {k: bll(w, u) - bll(w, T[f"p_{k}"].to_numpy()) for k in names}
    day = T["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(4)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    res = {"cols": {k: len(v) for k, v in S.items()}, "line_cols": lcols, "score": {}}
    for nm, mk in {"全体": np.ones(len(T), bool), "学習の期間に父の産駒が 1 走も無い": unseen, "それ以外": ~unseen}.items():
        r = {"rows": int(mk.sum())}
        n_d = np.bincount(dinv[mk], minlength=len(du)).astype(float)
        for k in names:
            r[k] = round(float(G[k][mk].mean()), 5)
        for a_ in [n_ for n_ in names if n_ != "A"]:
            d_d = np.bincount(dinv[mk], weights=(G[a_] - G["A"])[mk], minlength=len(du))
            bs = (Wb @ d_d) / (Wb @ n_d)
            r[f"{a_}_minus_A"] = [round(float(d_d.sum() / n_d.sum()), 5), np.percentile(bs, [2.5, 97.5]).round(5).tolist()]
        res["score"][nm] = r
    path = data.OUT / f"lineage{tag}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["score"], ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
