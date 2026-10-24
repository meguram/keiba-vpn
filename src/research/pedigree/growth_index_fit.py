"""成長指数（E）の学習と評価。ホストで実行する。

  python -m src.research.pedigree.growth_index_fit --dir <growth_weight の出力ディレクトリ>

目的変数: 今走のめぐ指数（絶対値）− 直近水準（過去 5 走の加重平均）。1 点 = 0.1 秒。単走ノイズ σ_e が大きい（約 7 点）ので、検出できる相関の上限も併せて出す。
候補1: LightGBM（統制 + 全入力）の予測 − 統制だけの予測（精度の目安）
候補2: 4 つの要素の寄与の和（リッジ。簡易版）
候補4（採用）: 統制を馬単位の交差当てはめで引いた残差を、統制 + 要素の入力を入れた小さな木（300 本 × 葉 7）で予測する。
学習 2020〜2023 年 → 評価 2024〜2025 年（2026 年は使わない）。
出力: growth_index_eval.json、params_eval2023.json + model_eval2023.txt、params.json + model.txt（〜2025 年で再学習）、gi_group.parquet（年ごとに『その年より前』で学習。V4 への追加評価用）。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference import growth_index as G
from src.research.pedigree import growth_weight_stats as S

log = logging.getLogger("growth_index_fit")
LAMBDAS = [0.003, 0.03, 0.3, 1.0, 3.0]
CTRL_NAMES = G.CTRL_NAMES


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


def usable(df: pd.DataFrame) -> pd.Series:
    return df["y_now"].notna() & df["stage"].notna() & df["surface"].isin(["芝", "ダ"]) & df["sex"].isin(["牡", "牝", "セ"])


def inputs_of(df: pd.DataFrame) -> list[dict]:
    cols = ["stage", "weight_change_in", "weight_dev", "rest_days", "has_training", "level", "n_valid_prior", "stale_prev", "distance", "surface", "sex"] + G.OIKIRI_NUM
    recs = df[cols].to_dict("records")
    ped = df[[f"ped_{k}" for k in G.PED_KEYS]].to_dict("records")
    out = []
    for r, p in zip(recs, ped):
        r["weight_change"] = r.pop("weight_change_in")
        r["ped"] = {k: p[f"ped_{k}"] for k in G.PED_KEYS}
        out.append(r)
    return out


def build_X(df: pd.DataFrame, prm: dict) -> np.ndarray:
    return np.array([G.feature_vector(x, prm) for x in inputs_of(df)], dtype=float)


def ridge_fit(C: np.ndarray, Xg: dict[str, np.ndarray], names: dict[str, list[str]], y: np.ndarray, lam: float) -> dict:
    """候補2: 統制 C とグループ特徴量を標準化したリッジ。統制のペナルティは小さく、グループ側は lam×n。係数は元の尺度に戻す。"""
    n = len(y)
    cols, nm = [], []
    means = {}
    for g in G.GROUPS:
        X = Xg[g]
        m = np.nanmean(X, axis=0)
        m = np.where(np.isnan(m), 0.0, m)
        means[g] = m
        Xi = np.where(np.isnan(X), m, X)
        for j, k in enumerate(names[g]):
            nm.append((g, k))
            cols.append(Xi[:, j])
    A = np.column_stack([C] + cols)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (A - mu) / sd
    pen = np.r_[np.full(C.shape[1], 1e-6 * n), np.full(len(cols), lam * n)]
    beta = np.linalg.solve(Z.T @ Z + np.diag(pen), Z.T @ (y - y.mean()))
    coef = beta / sd
    return {"coef_ctrl": coef[: C.shape[1]], "coef": coef[C.shape[1]:], "names": nm, "means": means, "intercept": float(y.mean() - mu @ coef)}


def ridge_params(fit: dict) -> dict:
    groups = {}
    for g in G.GROUPS:
        sel = [i for i, (gg, _) in enumerate(fit["names"]) if gg == g]
        groups[g] = {"names": [fit["names"][i][1] for i in sel], "mean": [float(fit["means"][g][k]) for k in range(len(sel))], "coef": [float(fit["coef"][i]) for i in sel]}
    return {"groups": groups, "intercept": 0.0}


def lgb_fit(X: np.ndarray, y: np.ndarray, threads=4):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=threads, verbose=-1, random_state=1).fit(X, y)


def small_fit(X: np.ndarray, y: np.ndarray, threads=4):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=threads, verbose=-1, random_state=1).fit(X, y)


def oof_control(C: np.ndarray, y: np.ndarray, horses: np.ndarray, threads=4) -> np.ndarray:
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    for a_, b_ in GroupKFold(4).split(C, y, horses):
        oof[b_] = lgb_fit(C[a_], y[a_], threads).predict(C[b_])
    return oof


def group_sums(booster, X: np.ndarray) -> dict[str, np.ndarray]:
    sl = G.group_slices()
    contrib = booster.predict(X, pred_contrib=True)
    out = {g: contrib[:, sl[g]].sum(axis=1) for g in list(G.GROUPS) + ["ctrl"]}
    out["index"] = contrib.sum(axis=1)
    return out


def fit_adopted(df_fit: pd.DataFrame, threads=4, drop_groups=()) -> dict:
    """採用版の学習。drop_groups は感度分析用に、その要素の列を落とす。"""
    prm = {"level_mean": float(df_fit["level"].mean()), "rest_median": float(df_fit["rest_days"].median())}
    X = build_X(df_fit, prm)
    y = df_fit["y_now"].to_numpy(float)
    C = X[:, : len(CTRL_NAMES)]
    resid = y - oof_control(C, y, df_fit["horse_id"].to_numpy(), threads)
    keep = np.ones(X.shape[1], dtype=bool)
    sl = G.group_slices()
    for g in drop_groups:
        keep[sl[g]] = False
    model = small_fit(X[:, keep], resid, threads)
    return {"prm": prm, "model": model, "keep": keep, "ctrl_model": lgb_fit(C, y, threads), "resid_std": float(resid.std())}


def predict_adopted(fit: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = build_X(df, fit["prm"])
    if fit["keep"].all():
        return group_sums(fit["model"].booster_, X)
    return {"index": fit["model"].predict(X[:, fit["keep"]])}


def cluster_means(r: np.ndarray, q: np.ndarray, races: np.ndarray, nq: int):
    X = np.column_stack([(q == k).astype(float) for k in range(nq)])
    res = S.ols_cluster(X, r, races)
    means = [{"q": k + 1, "n": int((q == k).sum()), "mean": res["beta"][k], "lo": res["lo"][k], "hi": res["hi"][k]} for k in range(nq)]
    sub = (q == 0) | (q == nq - 1)
    X2 = np.column_stack([np.ones(sub.sum()), (q[sub] == nq - 1).astype(float)])
    d = S.ols_cluster(X2, r[sub], races[sub])
    return means, {"diff": d["beta"][1], "lo": d["lo"][1], "hi": d["hi"][1]}


def evaluate_index(name: str, idx: np.ndarray, r_now: np.ndarray, r_fut: np.ndarray, races: np.ndarray, level: np.ndarray) -> dict:
    out = {"name": name, "n": int(len(idx))}
    ok = ~np.isnan(idx) & ~np.isnan(r_now)
    out["pearson_now"] = float(np.corrcoef(idx[ok], r_now[ok])[0, 1])
    out["spearman_now"] = S.spearman(idx[ok], r_now[ok])
    b = S.ols_cluster(np.column_stack([np.ones(ok.sum()), idx[ok]]), r_now[ok], races[ok])
    out["slope_now"] = {"coef": b["beta"][1], "lo": b["lo"][1], "hi": b["hi"][1]}
    okf = ok & ~np.isnan(r_fut)
    out["pearson_fut"] = float(np.corrcoef(idx[okf], r_fut[okf])[0, 1])
    out["spearman_fut"] = S.spearman(idx[okf], r_fut[okf])
    q = pd.qcut(idx[ok], 5, labels=False, duplicates="drop").astype(int)
    out["quintiles_now"], out["top_minus_bottom_now"] = cluster_means(r_now[ok], q, races[ok], 5)
    qf = pd.qcut(idx[okf], 5, labels=False, duplicates="drop").astype(int)
    out["quintiles_fut"], out["top_minus_bottom_fut"] = cluster_means(r_fut[okf], qf, races[okf], 5)
    ter = pd.qcut(level[ok], 3, labels=False, duplicates="drop")
    out["by_level_tercile"] = [{"tercile": int(t) + 1, "n": int((ter == t).sum()), "pearson_now": float(np.corrcoef(idx[ok][ter == t], r_now[ok][ter == t])[0, 1])} for t in sorted(set(ter))]
    return out


def noise_ceiling(tr: pd.DataFrame, te: pd.DataFrame, r_now: np.ndarray, r_fut: np.ndarray) -> dict:
    """1 走の指数の測定ノイズ σ_e を、短い休養・同面・同距離帯の連続 2 走の差の分散 / 2 から見積もり、目的変数が持つノイズ分散と、検出できる相関の上限を出す。"""
    d = tr[(tr["rest_days"].between(7, 35)) & (tr["surface"] == tr["prev_surface"]) & ((tr["distance"] - tr["prev_distance"]).abs() <= 200) & tr["idx_v"].notna() & tr["prev_idx"].notna() & ~tr["prev_low_conf"].eq(True)]
    sigma2 = float(np.var((d["idx_v"] - d["prev_idx"]).to_numpy(float)) / 2.0)
    okn, okf = ~np.isnan(r_now), ~np.isnan(r_fut)
    noise_now = sigma2 * (1.0 + te["w2"].to_numpy(float)[okn])
    noise_fut = sigma2 * (1.0 / te["m_fut"].to_numpy(float)[okf] + te["w2"].to_numpy(float)[okf])
    vn, vf = float(np.var(r_now[okn])), float(np.var(r_fut[okf]))
    return {"sigma_e": float(np.sqrt(sigma2)), "pairs": int(len(d)), "var_target_now": vn, "noise_var_now": float(np.mean(noise_now)), "max_corr_now": float(np.sqrt(max(0.0, 1 - np.mean(noise_now) / vn))),
            "var_target_fut": vf, "noise_var_fut": float(np.mean(noise_fut)), "max_corr_fut": float(np.sqrt(max(0.0, 1 - np.mean(noise_fut) / vf))),
            "min_detectable_corr_now": float(3.0 / np.sqrt(okn.sum())), "n_now": int(okn.sum())}


def level_table(idx: np.ndarray, y_now: np.ndarray, r_now: np.ndarray, y_fut: np.ndarray, thr: list[float], races: np.ndarray) -> list[dict]:
    lv = np.array([G.stage_of_index(v, thr) for v in idx])
    out = []
    for k, lab in enumerate(G.LABELS):
        m = (lv == k) & ~np.isnan(y_now)
        if m.sum() < 30:
            continue
        one = np.ones((m.sum(), 1))
        ru = S.ols_cluster(one, (y_now[m] > 0).astype(float), races[m])
        rr = S.ols_cluster(one, r_now[m], races[m])
        rp = S.ols_cluster(one, (r_now[m] > 0).astype(float), races[m])
        ry = S.ols_cluster(one, y_now[m], races[m])
        mf = (lv == k) & ~np.isnan(y_fut)
        out.append({"level": k, "label": lab, "n": int(m.sum()), "share": float((lv == k).mean()),
                    "mean_resid": rr["beta"][0], "mean_resid_lo": rr["lo"][0], "mean_resid_hi": rr["hi"][0],
                    "share_resid_up": rp["beta"][0], "share_resid_up_lo": rp["lo"][0], "share_resid_up_hi": rp["hi"][0],
                    "mean_y_now": ry["beta"][0], "share_up": ru["beta"][0], "mean_y_fut": float(np.nanmean(y_fut[mf])) if mf.any() else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = Path(a.dir)
    df = pd.read_parquet(d / "gi_rows.parquet")
    df["date"] = pd.to_datetime(df["date"])
    ok = usable(df)
    tr = df[ok & (df["year"] <= 2023)].reset_index(drop=True)
    te = df[ok & (df["year"] >= 2024)].reset_index(drop=True)
    log.info("学習 %d 行（〜2023）/ 評価 %d 行（2024〜2025）", len(tr), len(te))
    res: dict = {"n_train": int(len(tr)), "n_test": int(len(te))}
    # ── 採用版（候補4）
    fit = fit_adopted(tr, a.threads)
    pr_te = predict_adopted(fit, te)
    idx4 = pr_te["index"]
    prm = fit["prm"]
    X_tr, X_te = build_X(tr, prm), build_X(te, prm)
    C_tr, C_te = X_tr[:, : len(CTRL_NAMES)], X_te[:, : len(CTRL_NAMES)]
    p0 = fit["ctrl_model"].predict(C_te)
    r_now = te["y_now"].to_numpy(float) - p0
    r_fut = te["y_fut"].to_numpy(float) - p0
    races, lev = te["race_id"].to_numpy(), te["level"].to_numpy(float)
    res["cand4_adopted"] = evaluate_index("候補4（採用）: 統制 + 要素を入れた小さな木が、統制の残差を予測", idx4, r_now, r_fut, races, lev)
    # ── 候補1: 大きなモデルの差
    m_full = lgb_fit(X_tr, tr["y_now"].to_numpy(float), a.threads)
    idx1 = m_full.predict(X_te) - p0
    res["cand1_lgb"] = evaluate_index("候補1: LightGBM（統制 + 全入力）の予測 − 統制のみの予測", idx1, r_now, r_fut, races, lev)
    # ── 候補2: 4 要素の寄与の和（リッジ）
    sl = G.group_slices()
    names = {g: list(G.group_features({"stage": "3歳前半"})[g]) for g in G.GROUPS}
    best = None
    sub_tr, sub_va = (tr["year"] <= 2022).to_numpy(), (tr["year"] == 2023).to_numpy()
    for lam in LAMBDAS:
        f = ridge_fit(C_tr[sub_tr], {g: X_tr[sub_tr][:, sl[g]] for g in G.GROUPS}, names, tr.loc[sub_tr, "y_now"].to_numpy(float), lam)
        p = ridge_params(f)
        ctr = C_tr[sub_va] @ f["coef_ctrl"] + f["intercept"]
        idx_va = np.array([sum(G.contributions(x, p).values()) for x in inputs_of(tr[sub_va])])
        corr = float(np.corrcoef(idx_va, tr.loc[sub_va, "y_now"].to_numpy(float) - ctr)[0, 1])
        best = (lam, corr) if best is None or corr > best[1] else best
    f2 = ridge_fit(C_tr, {g: X_tr[:, sl[g]] for g in G.GROUPS}, names, tr["y_now"].to_numpy(float), best[0])
    p2 = ridge_params(f2)
    idx2 = np.array([sum(G.contributions(x, p2).values()) for x in inputs_of(te)])
    res["cand2_additive"] = evaluate_index("候補2: 4要素の寄与の和（リッジ・簡易版）", idx2, r_now, r_fut, races, lev)
    res["cand2_lambda"] = best[0]
    res["noise"] = noise_ceiling(tr, te, r_now, r_fut)
    # ── 要素ごとの寄与・抜いたときの落ち
    res["components"] = {g: {"sd": float(pr_te[g].std()), "corr_resid_now": float(np.corrcoef(pr_te[g], r_now)[0, 1]), "mean_abs": float(np.abs(pr_te[g]).mean())} for g in list(G.GROUPS) + ["ctrl"]}
    res["leave_one_group_out"], res["only_group"] = {}, {}
    for g in G.GROUPS:
        fg = fit_adopted(tr, a.threads, drop_groups=(g,))
        res["leave_one_group_out"][g] = float(np.corrcoef(predict_adopted(fg, te)["index"], r_now)[0, 1])
        fo = fit_adopted(tr, a.threads, drop_groups=tuple(x for x in G.GROUPS if x != g))
        res["only_group"][g] = float(np.corrcoef(predict_adopted(fo, te)["index"], r_now)[0, 1])
    # ── 5 段階: 閾値は評価期間の分布の 10/30/70/90 パーセンタイル
    thr = [float(np.percentile(idx4, p)) for p in (10, 30, 70, 90)]
    res["thresholds"] = thr
    res["levels_test"] = level_table(idx4, te["y_now"].to_numpy(float), r_now, te["y_fut"].to_numpy(float), thr, races)
    res["by_year"] = {}
    for y in (2024, 2025):
        m = (te["year"] == y).to_numpy()
        res["by_year"][str(y)] = {"cand4_pearson_now": float(np.corrcoef(idx4[m], r_now[m])[0, 1]), "cand1_pearson_now": float(np.corrcoef(idx1[m], r_now[m])[0, 1]),
                                  "cand2_pearson_now": float(np.corrcoef(idx2[m], r_now[m])[0, 1]), "n": int(m.sum())}
    res["coverage_test"] = {"has_training": float((te["has_training"] == 1).mean()), "oikiri_eval": float(te["oikiri_eval"].notna().mean()),
                            "comment": float(te["comment_pos"].notna().mean()), "ped_sire": float(te["ped_sire_banse"].notna().mean()), "weight_dev": float(te["weight_dev"].notna().mean())}
    fit["model"].booster_.save_model(str(d / "model_eval2023.txt"))
    eval_params = {"version": "gi-2", "model": "model.txt", "fit_end": "2023-12-31", **fit["prm"], "thresholds": thr, "labels": G.LABELS, "reason_min_points": 0.3,
                   "feature_names": G.feature_names(), "level_table": res["levels_test"], "unit": "指数ポイント（1点=0.1秒）",
                   "target": "megu_v3a_abs_robust(今走) − 直近水準(過去5走を0.7^kで加重。前走が7日以内なら前走を除く)から、統制で説明できる分を引いた残差"}
    (d / "params_eval2023.json").write_text(json.dumps(_clean(eval_params), ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "growth_index_eval.json").write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("評価: 採用 %.4f / 候補1 %.4f / 候補2 %.4f / 上限 %.4f", res["cand4_adopted"]["pearson_now"], res["cand1_lgb"]["pearson_now"], res["cand2_additive"]["pearson_now"], res["noise"]["max_corr_now"])
    # ── 本番用: 〜2025 年で再学習
    fit_all = fit_adopted(df[ok].reset_index(drop=True), a.threads)
    fit_all["model"].booster_.save_model(str(d / "model.txt"))
    params = {**eval_params, **fit_all["prm"], "fit_end": "2025-12-31"}
    (d / "params.json").write_text(json.dumps(_clean(params), ensure_ascii=False, indent=1), encoding="utf-8")
    # ── V4 追加評価用の指数: 各年は『その年より前』で学習（2020〜2021 年は欠損）
    outs = []
    for y in (2022, 2023, 2024, 2025):
        t = df[ok & (df["date"] < pd.Timestamp(f"{y}-01-01"))].reset_index(drop=True)
        fy = fit_adopted(t, a.threads)
        cur = df[df["year"] == y].reset_index(drop=True)
        pr = predict_adopted(fy, cur)
        c = pd.DataFrame({"gi_index": pr["index"], **{f"gi_{g}": pr[g] for g in G.GROUPS}})
        c.loc[cur["level"].isna().to_numpy(), :] = np.nan
        c["race_id"], c["horse_id"] = cur["race_id"].to_numpy(), cur["horse_id"].to_numpy()
        c["gi_weight_dev"] = cur["weight_dev"].to_numpy()
        outs.append(c)
        log.info("指数 %d 年: 学習 %d 行", y, len(t))
    pd.concat(outs).to_parquet(d / "gi_group.parquet", index=False)
    log.info("完了")


if __name__ == "__main__":
    main()
