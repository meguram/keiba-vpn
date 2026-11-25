"""段 2 の深掘り: 第 5 版の案（log.md「第 5 版の計画」。試した数 8 以下）を、第 4 版との対で比べる。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.v5 <A|B|C|D|E|null> [--seed N]

- A: 主な率の列をレースの中の偏差に直し、絶対の水準と件数の列を外す。
- B: 第 4 版の pure の列で、強い正則化（min_data_in_leaf 2000・num_leaves 15・lambda_l2 50・extra_trees）。
- C: 複勝を勝ちのモデルから Henery 型で作る（2・3 着の確率を p^γ・p^δ に比例。γ・δ は開発の OOF で当てる）。
- D: 線形の条件付きロジット（rel_ の列・標準化・2 代母の単位の 5 分割）。
- E: 第 4 版 ＋ 5 代の祖先の子孫の適性（features5.ancestor_apt）。
- null: 同じ生年 × 性の中で ped の行を入れ替え、特徴量から作り直す（帰無。--seed で回を変える）。
判定の数: 開発の行の勝ちの改善（一様から）・第 4 版との対の差（2 代母の単位・開催のブートストラップ 1,000 回）・
Benter の全体の α と初ダートの α の上乗せ（2 代母の単位）。出力: OUT/v5_<案>.json を trials.jsonl にも追記。費用 0 円。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, gates, kfold, model
from src.research.pedigree_lab.hyp_tests import clogit


def base_frame(tag: str = "v4") -> pd.DataFrame:
    X = pd.read_parquet(data.OUT / f"feat_{tag}.parquet")
    return X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)


def oof_for(X: pd.DataFrame, cols: list[str], group: str, params: dict | None = None, targets=("win",)) -> dict:
    dev = X["split"].isin([0, 1]).to_numpy()
    hold = (X["split"] == 2).to_numpy()
    f = kfold.folds(X, dev, group)
    old = model.PARAMS
    if params:
        model.PARAMS = dict(old, **params)
    try:
        out = {}
        for tg in targets:
            p, info = kfold.run(X, cols, tg, f, dev, hold)
            out[tg] = p
            out[f"{tg}_info"] = info
    finally:
        model.PARAMS = old
    return out


def race_loss(X: pd.DataFrame, p: np.ndarray, dev: np.ndarray) -> pd.Series:
    """開発の行の、レースごとの勝ち馬の −log p。"""
    m = (X["win"].to_numpy() == 1) & dev
    return pd.Series(-np.log(np.clip(p[m], 1e-9, 1)), index=X.loc[m, "race_id"].to_numpy())


def paired_diff(X: pd.DataFrame, p_new: np.ndarray, p_old: np.ndarray, dev: np.ndarray, n_boot: int = 1000) -> dict:
    a, b = race_loss(X, p_old, dev), race_loss(X, p_new, dev)
    d = (a - b).groupby(level=0).mean()           # 正 = 新しい版が良い
    mt = X.drop_duplicates("race_id").set_index("race_id")["meeting"].reindex(d.index).to_numpy()
    u = pd.factorize(mt)[0]
    nu = u.max() + 1
    s, c = np.bincount(u, d.to_numpy(), minlength=nu), np.bincount(u, minlength=nu)
    rng = np.random.default_rng(9)
    bs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nu, nu), minlength=nu)
        bs.append((w * s).sum() / (w * c).sum())
    return {"diff": round(float(d.mean()), 5), "ci": np.percentile(bs, [2.5, 97.5]).round(5).tolist()}


def alphas(X: pd.DataFrame, p: np.ndarray, dev: np.ndarray) -> dict:
    """Benter の全体の α と、初ダート（gates の定義）の α の上乗せ（開催の束ね 100 回）。"""
    # 場面の印は 2016 年からの全行で作る（デビューの判定に 2016 年の走りが要る）
    full = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=gates.FCOLS)
    F = gates.scenes(full)[["race_id", "horse_id", "FD"]]
    Y = X.loc[dev, ["race_id", "horse_id", "meeting", "win", "odds", "horse_number", "sire"]].copy()
    Y = Y.merge(F, on=["race_id", "horse_id"], how="left")
    Y["FD"] = Y["FD"].fillna(False).astype(bool)
    Y["p_win"] = p[dev]          # merge は左の順を保つ
    Y = Y[Y["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    inv = 1.0 / Y["odds"].to_numpy()
    Y["q"] = inv / pd.Series(inv).groupby(Y["race_id"].to_numpy()).transform("sum").to_numpy()
    Y["lp"] = np.log(np.clip(Y["p_win"], 1e-5, 1))
    Y["lq"] = np.log(np.clip(Y["q"], 1e-5, 1))
    rc = pd.factorize(Y["race_id"])[0]
    b = clogit(Y[["lp", "lq"]].to_numpy(), Y["win"].to_numpy(float), rc)
    sc = gates.scene_alpha(Y, {"FD": Y["FD"].to_numpy()}, n_boot=100)
    return {"alpha_all": round(float(b[0]), 4), "beta": round(float(b[1]), 4), "FD": sc["FD"]}


def gain(X: pd.DataFrame, p: np.ndarray, dev: np.ndarray) -> float:
    m = (X["win"].to_numpy() == 1) & dev
    return round(float(np.mean(np.log(p[m])) - np.mean(np.log(1 / X["n_run"].to_numpy()[m]))), 5)


def top3_gain(X, p3, dev):
    y = X["top3"].to_numpy()[dev]
    n = X["n_run"].to_numpy()[dev]
    base = np.minimum(3, n) / n
    q = np.clip(p3[dev], 1e-6, 1 - 1e-6)
    ll = lambda z: -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z))
    return round(float(ll(base) - ll(q)), 5)


def v4_ref(X: pd.DataFrame) -> dict:
    out = {}
    for g in ("dd", "meeting"):
        K = pd.read_parquet(data.OUT / f"kf_v4_pure_{g}.parquet")
        K = X[["race_id", "horse_id"]].merge(K, on=["race_id", "horse_id"], how="left")
        out[g] = K
    return out


def cols_A(X: pd.DataFrame) -> list[str]:
    base = ablate.col_sets(X)["pure"]
    rate = [c for c in base if not c.startswith("rel_") and not c.endswith("_n") and not c.startswith("ib_")
            and not c.endswith("res_n")]
    rid = X["race_id"].to_numpy()
    new = []
    for c in rate:
        nm = f"rel_{c}"
        if nm not in X:
            v = X[c].astype(float)
            X[nm] = (v - v.groupby(rid).transform("mean")).astype("float32")
        new.append(nm)
    return sorted(set(new)) + ["ib_F", "ib_SS"]


def henery(p: np.ndarray, rid: np.ndarray, n_run: np.ndarray, g: float, d: float) -> np.ndarray:
    """2 着・3 着の確率を p^g・p^d に比例させた Harville（Henery 型）。頭数によらず 3 着内の確率（正解の top3 に合わせる）。
    3 頭以下の判定は表の行数で行うので、レースの全頭を渡すこと（n_run は使わない）。"""
    out = np.zeros_like(p)
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(p)]
    for s, e in zip(starts, ends):
        q1 = p[s:e] / p[s:e].sum()
        q2 = q1 ** g / (q1 ** g).sum()
        q3 = q1 ** d / (q1 ** d).sum()
        A = q1[:, None] * q2[None, :] / (1 - q2[:, None])
        np.fill_diagonal(A, 0)
        P2 = A.sum(0)
        if e - s <= 3:            # 3 頭以下は全頭が 3 着内
            out[s:e] = 1.0
            continue
        den = 1 - q3[:, None] - q3[None, :]          # 正解は 3 着内（top3）なので、頭数によらず 3 着内の確率（critic の指摘 4）
        np.fill_diagonal(den, 1)
        W = A / np.clip(den, 1e-6, None)
        out[s:e] = q1 + P2 + (W.sum() - W.sum(0) - W.sum(1)) * q3
    return np.clip(out, 1e-6, 1 - 1e-6)


def run_C(X: pd.DataFrame, ref: dict) -> dict:
    dev = X["split"].isin([0, 1]).to_numpy()
    K = ref["dd"]
    Y = X[["race_id", "horse_number", "top3", "n_run"]].copy()
    Y["p"] = K["p_win"].to_numpy()
    Y["p3_old"] = K["p_top3"].to_numpy()
    Y["dev"] = dev
    Y = Y.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid, n = Y["race_id"].to_numpy(), Y["n_run"].to_numpy()
    y = Y["top3"].to_numpy()
    dm = Y["dev"].to_numpy()
    # γ・δ を開発の行の半分（レースの偶奇）で当て、残りの半分で測る
    half = pd.factorize(Y["race_id"])[0] % 2 == 0
    ll = lambda z, m: float(-np.mean(y[m] * np.log(z[m]) + (1 - y[m]) * np.log(1 - z[m])))
    best = None
    for g in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):          # 1 回目は 0.7 の端に当たったので、下へ 1 回だけ広げた
        for d in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            z = henery(Y["p"].to_numpy(), rid, n, g, d)
            s = ll(z, dm & half)
            if best is None or s < best[0]:
                best = (s, g, d, z)
    _, g, d, z = best
    base = np.clip(np.minimum(3, n) / n, 1e-6, 1 - 1e-6)
    m2 = dm & ~half
    return {"gamma": g, "delta": d, "top3_gain_henery": round(ll(base, m2) - ll(z, m2), 5),
            "top3_gain_dedicated": round(ll(base, m2) - ll(np.clip(Y["p3_old"].to_numpy(), 1e-6, 1 - 1e-6), m2), 5)}


def run_D(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in X.columns if c.startswith("rel_") and "agesurf" not in c][:40]
    dev = X["split"].isin([0, 1]).to_numpy()
    f = kfold.folds(X, dev, "dd")
    M = X[cols].astype(float).fillna(0).to_numpy()
    M = (M - M[dev].mean(0)) / (M[dev].std(0) + 1e-9)
    y = X["win"].to_numpy(float)
    rid = X["race_id"].to_numpy()
    p = np.full(len(X), np.nan)
    for k in range(kfold.K):
        tr = dev & (f != k)
        te = f == k
        # 2 代母の単位の折はレースを割るので、学習も評価もレースの中の softmax を、その折の行だけで作る
        b = clogit(M[tr], y[tr], pd.factorize(rid[tr])[0], n_iter=12)
        p[te] = M[te] @ b            # 線形の得点（折ごとのモデル。尺度はほぼ同じ）
    # 全行の得点を合わせてから、レースの中の softmax で揃える
    eta = np.where(dev, p, -np.inf)
    mx = pd.Series(eta).groupby(rid).transform("max").to_numpy()
    e = np.where(dev, np.exp(eta - mx), 0.0)
    s = pd.Series(e).groupby(rid).transform("sum").to_numpy()
    p = np.where(dev, e / np.where(s > 0, s, 1), np.nan)
    return p, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["A", "B", "C", "D", "E", "F", "G", "N", "null", "ref"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-n", action="store_true", help="件数の列（*_n・*_nrun・res_n）を外す（帰無の世代・時代の説明の確かめ）")
    ap.add_argument("--keep-own", action="store_true", help="帰無の確かめ 3: as-of の集計で自分の過去走を引かない")
    a = ap.parse_args()
    t0 = time.time()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"v5_{a.case}_{a.seed}",
                                   f"段 2 の第 5 版の案 {a.case}（対数損失・α。回収率は見ない）"))
    res = {"case": a.case}
    if a.case == "null":
        from src.research.pedigree_lab import features4
        if a.keep_own:
            data.SUBTRACT_OWN = False
        r = data.load_dev(reason=f"帰無（血統の行の入れ替え・回 {a.seed}）", version=f"null_{a.seed}")
        ped = data.load_ped()
        by = ped["horse_id"].str[:4]
        sx = r.drop_duplicates("horse_id").set_index("horse_id")["sex"].reindex(ped["horse_id"]).fillna("?").to_numpy()
        grp = by.to_numpy() + "_" + sx
        rng = np.random.default_rng(1000 + a.seed)
        perm = np.arange(len(ped))
        for ii in pd.Series(np.arange(len(ped))).groupby(grp).indices.values():
            perm[ii] = rng.permutation(ii)
        P = ped.iloc[perm].reset_index(drop=True)
        P["horse_id"] = ped["horse_id"].to_numpy()        # 馬の id はそのまま、血統の行だけを入れ替える
        Xn, _ = features4.build(r, P, sub=False)
        Xn["split"] = data.split_meetings(Xn["meeting"])
        Xn = Xn[(Xn["date"].dt.year >= 2017)].reset_index(drop=True)
        cols = ablate.col_sets(Xn)["pure"]
        if a.no_n:
            cols = [c for c in cols if not (c.endswith("_n") or c.endswith("_nrun"))]
        o = oof_for(Xn, cols, "dd")
        dev = Xn["split"].isin([0, 1]).to_numpy()
        res |= {"seed": a.seed, "no_n": a.no_n, "dd_win_gain": gain(Xn, o["win"], dev), "n_cols": len(cols)}
        pd.DataFrame({"race_id": Xn["race_id"], "horse_id": Xn["horse_id"], "p_win": o["win"], "rage": Xn["rage"],
                      "win": Xn["win"], "n_run": Xn["n_run"], "split": Xn["split"]}).to_parquet(
            data.OUT / f"kf_null{a.seed}{'_noN' if a.no_n else ''}{'_keepown' if a.keep_own else ''}_dd.parquet", index=False)      # レースの年齢条件で分けるため
    else:
        X = base_frame({"E": "v5e", "F": "v6e", "G": "v7g"}.get(a.case, "v4"))
        dev = X["split"].isin([0, 1]).to_numpy()
        ref = v4_ref(X)
        res["v4_dd_win_gain"] = gain(X, ref["dd"]["p_win"].to_numpy(), dev)
        if a.case == "ref":          # 第 4 版を同じ道具（alphas）で測り直す（決め方の線の比べの元）
            res |= {"alpha_dd": alphas(X, ref["dd"]["p_win"].to_numpy(), dev)}
        elif a.case == "C":
            res |= run_C(X, ref)
        else:
            if a.case == "A":
                cols, params, tg = cols_A(X), None, ("win", "top3")
            elif a.case == "B":
                cols = ablate.col_sets(X)["pure"]
                params = dict(min_data_in_leaf=2000, num_leaves=15, lambda_l2=50.0, extra_trees=True)
                tg = ("win", "top3")
            elif a.case == "N":     # 第 4 版 pure から件数の列を外す（帰無の世代・時代の説明の確かめ・本物の側）
                cols = [c for c in ablate.col_sets(X)["pure"] if not (c.endswith("_n") or c.endswith("_nrun"))]
                params, tg = None, ("win",)
            elif a.case in ("E", "F", "G"):     # F: 第 4 版 ＋ 父・母父の埋め込み（features6）
                cols = ablate.col_sets(X)["pure"]
                params, tg = None, ("win",)
            if a.case == "D":
                pdd, cols = run_D(X)
                res |= {"n_cols": len(cols), "dd_win_gain": gain(X, pdd, dev),
                        "vs_v4_dd": paired_diff(X, pdd, ref["dd"]["p_win"].to_numpy(), dev), "alpha_dd": alphas(X, pdd, dev)}
            else:
                od = oof_for(X, cols, "dd", params, tg)
                om = oof_for(X, cols, "meeting", params, ("win",))
                res |= {"n_cols": len(cols), "dd_win_gain": gain(X, od["win"], dev), "meeting_win_gain": gain(X, om["win"], dev),
                        "memorize_gap": round(gain(X, om["win"], dev) - gain(X, od["win"], dev), 5),
                        "vs_v4_dd": paired_diff(X, od["win"], ref["dd"]["p_win"].to_numpy(), dev),
                        "alpha_dd": alphas(X, od["win"], dev), "info_dd_win": od["win_info"]}
                if "top3" in od:
                    res["dd_top3_gain"] = top3_gain(X, od["top3"], dev)
                    res["info_dd_top3"] = od["top3_info"]
                pd.DataFrame({"race_id": X["race_id"], "horse_id": X["horse_id"], "p_win": od["win"]}).to_parquet(
                    data.OUT / f"kf_v5{a.case}_dd.parquet", index=False)
    res["sec"] = round(time.time() - t0)
    print(json.dumps(res, ensure_ascii=False, default=str))
    name = f"v5_{a.case}" + (f"_{a.seed}" if a.case == "null" else "") + ("_noN" if a.no_n else "") + ("_keepown" if a.keep_own else "")
    res["keep_own"] = a.keep_own
    (data.OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    with open(data.OUT / "trials.jsonl", "a") as fh:
        fh.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
    data.finish_all(str(data.OUT / f"{name}.json"))


if __name__ == "__main__":
    main()
