"""効くと分かった血統の列だけの小さなモデルと、訓練・評価の精度の表（ユーザーの依頼・log.md「効くと分かった列だけの小さなモデル」）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.compact_model [--fit]

- --fit: 小さなモデル（compact）と、年齢・性を足した版（compact_age）を、2 代母の単位の 5 分割で学ぶ（kfold.run と同じ作り）。
- 精度: 開発の行（2017〜24 年の split 0・1）の評価（OOF）と訓練（折のモデルを自分の学習の行に当てて、レースの中で揃えた予測の平均・in-sample）。
  v5bag の表は pr_top3 が全行で欠けなので、v5bag は勝ちだけ。市場の 3 着内は Harville。
  比べる相手は 一様・第 4 版（凍結の models_v4_pure_dd・kf_v4_pure_dd）・市場・v5bag（勝ちだけ）。
- 物差し: 一様からの対数損失の改善（勝ち = 勝ち馬の −log p、3 着内 = 二値の対数損失）・AUC・較正の傾き（十分位）・
  1 位の馬の勝率と 3 着内率・上位 3 頭の 3 着内率・Brier。
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.research.pedigree_lab import ablate, data, kfold, model
from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck

PREFIX = ("sire_surf_", "sire_all_", "sire_sd_", "bms_sd_", "bms_all_", "bms_surf_", "dam_all_", "dam_surf_", "sire_ra_", "sire_rasurf_",
          "dd_all_", "nick_all_", "rel_sire_surf", "rel_sire_sd", "rel_sire_rasurf", "rel_sire_all", "rel_bms_all", "rel_bms_sd",
          "rel_dam", "rel_dd", "rel_nick_all")
EXACT = ["rel_sire_dz", "dam_v_nrun", "dam_v_win", "dam_v_hi", "dd_v_nrun", "dd_v_win", "dd_v_hi", "dam_v_quick", "dd_v_quick",
         "sire_dz", "sire_dsd", "bms_dz", "bms_dsd", "ib_F", "ib_SS", "mat5_dirt", "lv5_dirt"]
AGE = ["st_age", "st_sex_f", "st_rel_age"]
V5BAG = "/home/jovyan/work/megu_cycles_out/rich_pred_v5bag.parquet"


def frame() -> tuple[pd.DataFrame, dict]:
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)
    sets = ablate.col_sets(X)                       # st_* を足す・第 4 版の pure の列
    F = ap_.base_frame()
    A, keys = ap_.group_a(F, "dirt")
    L = ap_.maternal_long(F, drop_common=False)
    F["mat5_dirt"], F["lv5_dirt"] = ck.mat_cols(F, A, keys, L)
    X = X.merge(F[["race_id", "horse_id", "mat5_dirt", "lv5_dirt"]], on=["race_id", "horse_id"], how="left")
    pure = sets["pure"]
    # "_e" で終わる列（features4 のダート寄り＝H13・「閉じる」）は外す（critic の指摘 2）
    compact = [c for c in pure if c.startswith(PREFIX) and "agesurf" not in c and not c.endswith("_e")] + [c for c in EXACT if c in X.columns]
    compact = list(dict.fromkeys(compact))
    return X, {"pure": pure, "compact": compact, "compact_age": compact + AGE}


def insample(X: pd.DataFrame, cols: list[str], target: str, f: np.ndarray, dev: np.ndarray, mdir) -> np.ndarray:
    """折のモデルを、その折の学習の行に当てた予測の平均（in-sample）。温度は保存の値で、レースの中で揃える。"""
    meta = json.loads((mdir / f"{target}_meta.json").read_text())
    # 折ごとにレースの中で揃えてから平均する（確率のまま平均すると鈍り、覚え込みが小さく出る。critic の指摘 5）
    rid = X["race_id"].to_numpy()
    total = np.ones(len(X)) if target == "win" else np.minimum(3, X["n_run"].to_numpy()).astype(float)
    acc, cnt = np.zeros(len(X)), np.zeros(len(X))
    for k in range(kfold.K):
        tr = dev & (f != k) & (f != (k + 1) % kfold.K)
        m = lgb.Booster(model_file=str(mdir / f"{target}_fold{k}.txt"))
        # 2 代母の単位の分けでは、同じレースの馬が別の折に分かれる。レースの全頭（開発の行）に当ててから揃え、学習の行の値だけを取る
        pk = np.full(len(X), np.nan)
        pk[dev] = model.race_norm(m.predict(X.loc[dev, meta["cols"]]), rid[dev], total[dev], meta["temp"])
        acc[tr] += pk[tr]
        cnt[tr] += 1
    p = np.full(len(X), np.nan)
    # 馬ごとに平均した折の組が違うので、平均の後にレースの中でもう一度揃える（critic）
    p[dev] = model.race_norm(acc[dev] / np.maximum(cnt[dev], 1), rid[dev], total[dev], 1.0)
    return p


def metrics(S: pd.DataFrame, p: np.ndarray, target: str) -> dict:
    y = S[target].to_numpy(float)
    rid = S["race_id"].to_numpy()
    n = S["n_run"].to_numpy(float)
    ok = ~np.isnan(p)
    if not ok.all():
        keep = pd.Series(ok).groupby(rid).transform("all").to_numpy()
        S, p, y, rid, n = S[keep], p[keep], y[keep], rid[keep], n[keep]
    u = 1 / n if target == "win" else np.minimum(3, n) / n
    if target == "win":
        gain = model.race_ll(y, u, rid) - model.race_ll(y, p, rid)
    else:
        gain = model.ll(y, u) - model.ll(y, p)
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"d": dec, "p": p, "y": y}).groupby("d").mean()
    lg = lambda v: np.log(v / (1 - v))
    slope = float(np.polyfit(lg(cal["p"].clip(1e-4, 0.999)), lg(cal["y"].clip(1e-4, 0.999)), 1)[0])
    rk = pd.Series(p).groupby(rid).rank(ascending=False, method="first").to_numpy()
    top3 = S["top3"].to_numpy(float)
    win = S["win"].to_numpy(float)
    return {"rows": int(len(S)), "races": int(pd.Series(rid).nunique()), "gain_vs_uniform": round(float(gain), 4),
            "auc": round(float(roc_auc_score(y, p)), 4), "calib_slope": round(slope, 3), "mean_ratio": round(float(y.mean() / p.mean()), 4),
            "deciles": cal.round(4).reset_index().to_dict("records"),
            "top1_win": round(float(win[rk == 1].mean()), 4), "top1_top3": round(float(top3[rk == 1].mean()), 4),
            "top3_top3": round(float(top3[rk <= 3].mean()), 4), "brier": round(float(np.mean((p - y) ** 2)), 5)}


def market(S: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rid = S["race_id"].to_numpy()
    inv = np.where(S["odds"].to_numpy(float) > 0, 1 / S["odds"].to_numpy(float), np.nan)
    q = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    # 3 着内は Harville（v5.henery の g = d = 1）。レースの行が続いている前提なので並べ替えてから戻す（critic の指摘 1）
    from src.research.pedigree_lab.v5 import henery
    o = np.lexsort((S["horse_number"].to_numpy(), rid))
    qq = np.nan_to_num(q, nan=1e-6)
    q3 = np.empty(len(S))
    q3[o] = henery(qq[o], rid[o], S["n_run"].to_numpy()[o], 1.0, 1.0)
    return q, np.where(np.isnan(q), np.nan, np.clip(q3, 1e-6, 0.999))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "compact_model",
                                   "効くと分かった列だけの小さなモデルと訓練・評価の精度（尤度・的中の記述だけ。回収率は見ない）"))
    X, sets = frame()
    dev = X["split"].isin([0, 1]).to_numpy()
    hold = (X["split"] == 2).to_numpy()
    f = kfold.folds(X, dev, "dd")
    print("cols", {k: len(v) for k, v in sets.items()}, flush=True)
    for nm in ("compact", "compact_age"):
        mdir = data.OUT / f"models_v4_{nm}_dd"
        path = data.OUT / f"kf_v4_{nm}_dd.parquet"
        if a.fit or not path.exists():
            mdir.mkdir(exist_ok=True)
            kfold.SAVE_DIR, kfold.OBJ = mdir, "binary"
            out = X[["race_id", "horse_id"]].copy()
            out["part"], out["fold"] = np.where(dev, "dev", "hold"), f
            for tg in ("win", "top3"):
                out[f"p_{tg}"], info = kfold.run(X, sets[nm], tg, f, dev, hold)
                print(nm, tg, info, flush=True)
            out.to_parquet(path, index=False)
    # 予測を集める
    S = X[dev].reset_index(drop=True)
    preds = {}
    K4 = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    k4 = X[["race_id", "horse_id"]].merge(K4[["race_id", "horse_id", "p_win", "p_top3", "fold"]], on=["race_id", "horse_id"], how="left")
    f4 = k4["fold"].fillna(-1).astype(int).to_numpy()
    assert (f4[dev] == f[dev]).all(), "第 4 版の折の並びが合わない"
    for tg in ("win", "top3"):
        preds[("v4", "eval", tg)] = k4[f"p_{tg}"].to_numpy()[dev]
        preds[("v4", "train", tg)] = insample(X, sets["pure"], tg, f4, dev, data.OUT / "models_v4_pure_dd")[dev]
        for nm in ("compact", "compact_age"):
            Kc = pd.read_parquet(data.OUT / f"kf_v4_{nm}_dd.parquet")
            kc = X[["race_id", "horse_id"]].merge(Kc[["race_id", "horse_id", f"p_{tg}"]], on=["race_id", "horse_id"], how="left")
            preds[(nm, "eval", tg)] = kc[f"p_{tg}"].to_numpy()[dev]
            preds[(nm, "train", tg)] = insample(X, sets[nm], tg, f, dev, data.OUT / f"models_v4_{nm}_dd")[dev]
    q, q3 = market(S)
    preds[("market", "eval", "win")], preds[("market", "eval", "top3")] = q, q3
    V = pd.read_parquet(V5BAG)[["race_id", "horse_id", "pr_win"]]
    v = S[["race_id", "horse_id"]].merge(V, on=["race_id", "horse_id"], how="left")["pr_win"].to_numpy()
    rid = S["race_id"].to_numpy()
    v = v / pd.Series(v).groupby(rid).transform("sum").to_numpy()
    preds[("v5bag", "eval", "win")] = v
    # 全モデルで共通のレースに揃える（どれかの予測が欠けたレースは全部から外す。critic の指摘 7）
    keep = np.ones(len(S), bool)
    for p in preds.values():
        keep &= pd.Series(~np.isnan(p)).groupby(rid).transform("all").to_numpy()
    S = S[keep].reset_index(drop=True)
    preds = {k: v[keep] for k, v in preds.items()}
    res = {"period": "2017-01-01〜2024-12-31（開発の行 split 0・1）", "cols": {k: len(v) for k, v in sets.items()},
           "compact_cols": sets["compact"], "table": {}}
    for (nm, part, tg), p in preds.items():
        res["table"][f"{nm}|{part}|{tg}"] = metrics(S, p, tg)
        r = res["table"][f"{nm}|{part}|{tg}"]
        print(nm, part, tg, {k: r[k] for k in ("rows", "gain_vs_uniform", "auc", "calib_slope", "top1_win", "top1_top3", "top3_top3", "brier")}, flush=True)
    u_w = S.assign(_u=1 / S["n_run"])["_u"].to_numpy()
    u_3 = (np.minimum(3, S["n_run"]) / S["n_run"]).to_numpy()
    res["table"]["uniform|eval|win"] = metrics(S, u_w + 1e-9 * np.random.default_rng(0).random(len(S)), "win")
    res["table"]["uniform|eval|top3"] = metrics(S, u_3 + 1e-9 * np.random.default_rng(0).random(len(S)), "top3")
    path = data.OUT / "compact_model.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
