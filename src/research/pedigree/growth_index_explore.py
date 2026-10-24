"""成長指数 E の補助検証: 要素ごとの LightGBM（統制 + その要素だけ）の寄与、全入力モデルの重要度、非線形な和（候補3）。

  python -m src.research.pedigree.growth_index_explore --dir <growth_weight の出力ディレクトリ>
学習 2020〜2023 年 → 評価 2024〜2025 年。目的変数は growth_index_fit と同じ（今走 − 直近水準）で、評価は統制のみのモデルの予測を引いた残差との相関。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference import growth_index as G
from src.research.pedigree import growth_index_fit as F

log = logging.getLogger("growth_index_explore")


def _mats(df: pd.DataFrame, prm: dict):
    """統制の行列と、要素ごとの (列名, 行列)。growth_index_fit.build_X の列を切り出す。"""
    X = F.build_X(df, prm)
    sl = G.group_slices()
    names = G.group_features({"stage": "3歳前半"})
    return X[:, sl["ctrl"]], {g: (list(names[g]), X[:, sl[g]]) for g in G.GROUPS}



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = Path(a.dir)
    df = pd.read_parquet(d / "gi_rows.parquet")
    df["date"] = pd.to_datetime(df["date"])
    ok = F.usable(df)
    tr, te = df[ok & (df["year"] <= 2023)].reset_index(drop=True), df[ok & (df["year"] >= 2024)].reset_index(drop=True)
    prm = {"level_mean": float(tr["level"].mean()), "rest_median": float(tr["rest_days"].median())}
    Ctr, Mtr = _mats(tr, prm)
    Cte, Mte = _mats(te, prm)
    ytr = tr["y_now"].to_numpy(float)
    m0 = F.lgb_fit(Ctr, ytr, a.threads)
    p0 = m0.predict(Cte)
    r = te["y_now"].to_numpy(float) - p0
    rf = te["y_fut"].to_numpy(float) - p0
    res: dict = {"single": {}, "importance": {}}
    parts = {}
    for g in G.GROUPS:
        m = F.lgb_fit(np.column_stack([Ctr, Mtr[g][1]]), ytr, a.threads)
        idx = m.predict(np.column_stack([Cte, Mte[g][1]])) - p0
        parts[g] = idx
        res["single"][g] = {"pearson_now": float(np.corrcoef(idx, r)[0, 1]), "sd": float(idx.std())}
        log.info("%s 単独: %.4f", g, res["single"][g]["pearson_now"])
    add = sum(parts.values())
    ok2 = ~np.isnan(rf)
    res["additive_of_single"] = {"pearson_now": float(np.corrcoef(add, r)[0, 1]), "pearson_fut": float(np.corrcoef(add[ok2], rf[ok2])[0, 1])}
    # 2 要素ずつ・全部
    for combo in (("oikiri", "stage"), ("oikiri", "weight"), ("oikiri", "ped"), ("stage", "weight", "ped"), tuple(G.GROUPS)):
        Xa = np.column_stack([Ctr] + [Mtr[g][1] for g in combo])
        Xb = np.column_stack([Cte] + [Mte[g][1] for g in combo])
        idx = F.lgb_fit(Xa, ytr, a.threads).predict(Xb) - p0
        res.setdefault("combos", {})["+".join(combo)] = {"pearson_now": float(np.corrcoef(idx, r)[0, 1]), "pearson_fut": float(np.corrcoef(idx[ok2], rf[ok2])[0, 1])}
        log.info("%s: %.4f", "+".join(combo), res["combos"]["+".join(combo)]["pearson_now"])
    # 全入力モデルの重要度（gain）
    names = G.CTRL_NAMES + [f"{g}:{n}" for g in G.GROUPS for n in Mtr[g][0]]
    Xa = np.column_stack([Ctr] + [Mtr[g][1] for g in G.GROUPS])
    mf = F.lgb_fit(Xa, ytr, a.threads)
    mf.booster_.params["x"] = 0
    gain = mf.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-gain)[:25]
    res["importance"] = [{"feature": names[i], "gain_share": float(gain[i] / gain.sum())} for i in order]
    (d / "growth_index_explore.json").write_text(json.dumps(F._clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("完了")


if __name__ == "__main__" and "--residual" not in __import__("sys").argv:
    main()


def residual_main() -> None:
    """候補3: 統制（直近水準など）で説明できる分を先に引いた残差を、要素の入力だけ（統制なし）で予測する小さな木のモデル。
    サーバでは統制変数が要らず、要素ごとの出力がそのまま寄与になる。統制の当てはめは馬単位の交差当てはめ（学習行への過学習を避ける）。"""
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--residual", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = Path(a.dir)
    df = pd.read_parquet(d / "gi_rows.parquet")
    df["date"] = pd.to_datetime(df["date"])
    ok = F.usable(df)
    tr, te = df[ok & (df["year"] <= 2023)].reset_index(drop=True), df[ok & (df["year"] >= 2024)].reset_index(drop=True)
    prm = {"level_mean": float(tr["level"].mean()), "rest_median": float(tr["rest_days"].median())}
    Ctr, Mtr = _mats(tr, prm)
    Cte, Mte = _mats(te, prm)
    ytr = tr["y_now"].to_numpy(float)
    oof = np.zeros(len(tr))
    for fi, (a_, b_) in enumerate(GroupKFold(4).split(Ctr, ytr, tr["horse_id"].to_numpy())):
        oof[b_] = F.lgb_fit(Ctr[a_], ytr[a_], a.threads).predict(Ctr[b_])
    rtr = ytr - oof
    p0 = F.lgb_fit(Ctr, ytr, a.threads).predict(Cte)
    r = te["y_now"].to_numpy(float) - p0
    rf = te["y_fut"].to_numpy(float) - p0
    okf = ~np.isnan(rf)
    import lightgbm as lgb

    def small(X, y):
        m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=a.threads, verbose=-1, random_state=1)
        return m.fit(X, y)

    res: dict = {"residual_std_train": float(rtr.std())}
    parts = {}
    for g in G.GROUPS:
        m = small(Mtr[g][1], rtr)
        parts[g] = m.predict(Mte[g][1])
        res.setdefault("single", {})[g] = {"pearson_now": float(np.corrcoef(parts[g], r)[0, 1]), "sd": float(parts[g].std())}
    add = sum(parts.values())
    res["additive"] = {"pearson_now": float(np.corrcoef(add, r)[0, 1]), "spearman_now": S_sp(add, r), "pearson_fut": float(np.corrcoef(add[okf], rf[okf])[0, 1]), "sd": float(add.std())}
    Xa = np.column_stack([Mtr[g][1] for g in G.GROUPS])
    Xb = np.column_stack([Mte[g][1] for g in G.GROUPS])
    joint = small(Xa, rtr).predict(Xb)
    res["joint"] = {"pearson_now": float(np.corrcoef(joint, r)[0, 1]), "pearson_fut": float(np.corrcoef(joint[okf], rf[okf])[0, 1])}
    # 2 要素の組（追い切り + 成長期）
    for combo in (("oikiri", "stage"), ("oikiri", "stage", "weight")):
        s = sum(parts[g] for g in combo)
        res.setdefault("additive_combos", {})["+".join(combo)] = float(np.corrcoef(s, r)[0, 1])
    # 候補4: 統制 + 要素の入力を 1 つの小さな木のモデルに入れ、統制の交差当てはめ残差を予測する
    Xc_tr = np.column_stack([Ctr] + [Mtr[g][1] for g in G.GROUPS])
    Xc_te = np.column_stack([Cte] + [Mte[g][1] for g in G.GROUPS])
    for ne, nl in ((150, 7), (300, 7), (300, 15)):
        m = lgb.LGBMRegressor(n_estimators=ne, learning_rate=0.05, num_leaves=nl, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=a.threads, verbose=-1, random_state=1).fit(Xc_tr, rtr)
        pr = m.predict(Xc_te)
        res.setdefault("cand4", {})[f"{ne}x{nl}"] = {"pearson_now": float(np.corrcoef(pr, r)[0, 1]), "pearson_fut": float(np.corrcoef(pr[okf], rf[okf])[0, 1])}
        log.info("候補4 %dx%d: %.4f", ne, nl, res["cand4"][f"{ne}x{nl}"]["pearson_now"])
    (d / "growth_index_explore_residual.json").write_text(json.dumps(F._clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("加法 %.4f / 一体 %.4f / 単独 %s", res["additive"]["pearson_now"], res["joint"]["pearson_now"], {k: round(v["pearson_now"], 3) for k, v in res["single"].items()})


def S_sp(a, b):
    from src.research.pedigree import growth_weight_stats as S

    return S.spearman(a, b)


if __name__ == "__main__" and "--residual" in __import__("sys").argv:
    residual_main()
