"""tp-71 想定単勝・複勝（V8。tp-12b と同じ手順）の 2026 年版。2025 年までで学習し、2026 年の特徴量（tp71_pred2026 が保存）に当てる。

python -m src.scripts.data.tp71_odds po2026 --feat-dir /home/jovyan/work/tp71_out/pred2026/feat --out /home/jovyan/work/tp71_out/po/po_2026.parquet

2022〜2025 年の po_<年>.parquet は betting_sim_v4/bsim_make_po.py が作ったもの（各年より前だけで学習）。2026 年だけこの版で作る。
特徴量に市場オッズ・人気は入れない（po.assert_no_odds_features）。2026 年の確定オッズは評価にも使わない（この段は予測だけ）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def po2026(a) -> None:
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from src.pipeline.betting import odds_model_eval as oe, pred_odds as po
    from src.pipeline.models import tp05_data

    threads = int(os.environ.get("TP71_THREADS", "4"))
    work = REPO / "data/local/modeling/training_pipeline"
    bet = REPO / "data/local/modeling/betting"
    wp_all = pd.read_parquet(bet / "odds_win_place.parquet")
    wp_all = wp_all[wp_all["win_odds"] > 0]
    D = tp05_data.load_dataset(work)
    df = D["df"]
    feats = D["FEATS"]["base+S+C1"]
    EP, _EX, _PED = tp05_data.load_extras_ped(df, work)
    po.assert_no_odds_features(list(feats) + list(EP.columns))
    T = df[["race_id", "horse_id", "date"]].reset_index().merge(wp_all[["race_id", "horse_id", "win_odds", "place_odds_min"]], on=["race_id", "horse_id"], how="inner")
    X = df.loc[T["index"], feats].reset_index(drop=True)
    X_ep = pd.concat([X, EP.loc[T["index"]].reset_index(drop=True)], axis=1)
    dates = pd.to_datetime(T["date"])
    yr = dates.dt.year.to_numpy()
    p_mkt = oe.market_prob(T["race_id"].to_numpy(), T["win_odds"].to_numpy())
    ylog = {"place": np.log(np.clip(T["place_odds_min"].fillna(1.1).to_numpy(), 1.0, None)), "pm": np.log(p_mkt)}
    cols = list(X_ep.columns)

    fs = sorted(Path(a.feat_dir).glob("*.parquet"))
    F = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    miss = [c for c in cols if c not in F.columns]
    assert not miss, miss[:10]
    Xt = F[cols].copy()
    cat_cols = [c for c in cols if str(X_ep[c].dtype) in ("category", "object")]
    bad = {}
    for c in cat_cols:   # カテゴリ列は文字列に揃える（学習側の水準に 2026 年を合わせる。未知の水準は欠損）
        tr_str = X_ep[c].astype(str)
        cats = sorted(tr_str.unique())
        X_ep[c] = pd.Categorical(tr_str, categories=cats)
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        Xt[c] = pd.Categorical(Xt[c].astype(str), categories=cats)
        bad[c] = float(Xt[c].isna().mean())
    print("カテゴリ列の未知の値の割合(最大5):", sorted(bad.items(), key=lambda x: -x[1])[:5], flush=True)
    for c in cols:
        if c not in cat_cols:
            Xt[c] = pd.to_numeric(Xt[c], errors="coerce").astype("float64")
    Y = 2026
    P = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
         "lambda_l2": 1.0, "verbosity": -1, "num_threads": threads, "seed": 0, "deterministic": True, "force_row_wise": True}
    P8 = dict(P, learning_rate=0.02, feature_fraction=0.7)
    HL = 1.0
    t0 = time.time()
    tr = np.flatnonzero(dates < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=10))
    tr_a, va = tr[yr[tr] < Y - 1], tr[yr[tr] == Y - 1]
    b0 = lgb.train(P8, lgb.Dataset(X_ep.iloc[tr_a], label=ylog["pm"][tr_a]), num_boost_round=3000, valid_sets=[lgb.Dataset(X_ep.iloc[va], label=ylog["pm"][va])],
                   callbacks=[lgb.early_stopping(80, verbose=False)])
    b = lgb.train(P8, lgb.Dataset(X_ep.iloc[tr], label=ylog["pm"][tr]), num_boost_round=int(max(50, b0.best_iteration * 1.1)))
    z = b.predict(Xt[cols])
    Xc = X_ep.copy()
    Xtc = Xt[cols].copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str)
        Xtc[c] = Xtc[c].astype(str)
    m = CatBoostRegressor(iterations=1500, learning_rate=0.06, depth=8, loss_function="RMSE", thread_count=threads, random_seed=0, verbose=0, cat_features=cat_cols)
    m.fit(Xc.iloc[tr], ylog["pm"][tr])
    z = (z + m.predict(Xtc)) / 2
    out = F[["race_id", "horse_id"]].copy()
    out["win_raw"] = 1.0 / np.exp(z)
    w = 0.5 ** (((Y - 1) - yr[tr]) / HL)
    Xp = X.iloc[tr]
    featsx = list(feats)
    bp = lgb.train(P, lgb.Dataset(Xp, label=ylog["place"][tr], weight=w), num_boost_round=500)
    out["place_raw"] = np.exp(bp.predict(Xt[featsx]))
    out["win_pred"] = po.normalize_win_odds(out["race_id"].to_numpy(), out["win_raw"].to_numpy())
    out["place_pred"] = np.clip(out["place_raw"], 1.0, None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"2026 {time.time() - t0:.0f}s 行数{len(out)} 勝ち想定の中央値 {out['win_pred'].median():.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("po2026")
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=po2026)
    q = sub.add_parser("v9")
    q.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    q.add_argument("--feat-dir", default="")
    q.add_argument("--out-dir", default="/home/jovyan/work/tp71_out/po9")
    q.set_defaults(fn=v9)
    a = ap.parse_args()
    a.fn(a)



def v9(a) -> None:
    """V9: 想定単勝・複勝を、v6e の学習表（1231 列。CY・BEST・RB・前段 e・補助ヘッド・血統・騎手調教師・コース環境などを含む）で学習し直す。
    手順は V8（tp-12b）と同じ walk-forward（年 Y は Y-01-01 の 10 日前までで学習、年 Y-1 で早期停止→全体で再学習）。LightGBM のみ（CatBoost は列が多く時間がかかるため省く）。
    市場オッズ・人気は入れない（列名を検査）。2026 年は学習・選択に使わず、--feat-dir に 2026 年の特徴量（tp71_pred2026 --v6 の出力）があるときだけ予測する。"""
    import lightgbm as lgb

    os.environ.setdefault("V6_STAGE", "e")
    os.environ.setdefault("V5_GROUPS", "CY,BEST,RB")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from src.pipeline.betting import odds_model_eval as oe, pred_odds as po
    from src.scripts.data import fit_final_v6 as f6

    threads = int(os.environ.get("TP71_THREADS", "8"))
    f6.patch()
    df, meta, mats, _A, _Af = f6.f4.build_matrices()
    names = [m[0] for m in f6.f4.MEMBERS]
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    cols = list(X_all.columns)
    po.assert_no_odds_features(cols)
    wp = pd.read_parquet(REPO / "data/local/modeling/betting/odds_win_place.parquet")
    wp = wp[wp["win_odds"] > 0][["race_id", "horse_id", "win_odds", "place_odds_min"]]
    T = meta[["race_id", "horse_id", "date"]].reset_index(drop=True)
    T["i"] = np.arange(len(T))
    T = T.merge(wp, on=["race_id", "horse_id"], how="inner")
    dates = pd.to_datetime(T["date"])
    yr = dates.dt.year.to_numpy()
    X = X_all.iloc[T["i"].to_numpy()].reset_index(drop=True)
    p_mkt = oe.market_prob(T["race_id"].to_numpy(), T["win_odds"].to_numpy())
    y_win = np.log(p_mkt)
    y_pl = np.log(np.clip(T["place_odds_min"].fillna(1.1).to_numpy(), 1.0, None))
    P = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
         "lambda_l2": 1.0, "verbosity": -1, "num_threads": threads, "seed": 0, "deterministic": True, "force_row_wise": True}
    P8 = dict(P, learning_rate=0.02, feature_fraction=0.5)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    F26 = None
    if a.feat_dir and Path(a.feat_dir).exists() and list(Path(a.feat_dir).glob("*.parquet")):
        F26 = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.feat_dir).glob("*.parquet"))], ignore_index=True)
        miss = [c for c in cols if c not in F26.columns]
        assert not miss, miss[:10]
    log = {}
    for Y in a.years:
        f = out_dir / f"po9_{Y}.parquet"
        if f.exists():
            print(Y, "既存", flush=True)
            continue
        t0 = time.time()
        tr = np.flatnonzero(dates < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=10))
        if Y == 2026:
            if F26 is None:
                print("2026 年の特徴量が無いので飛ばす", flush=True)
                continue
            Xte = F26[cols].apply(pd.to_numeric, errors="coerce").astype("float32")
            out = F26[["race_id", "horse_id"]].copy()
        else:
            te = np.flatnonzero(yr == Y)
            Xte = X.iloc[te]
            out = T.iloc[te][["race_id", "horse_id"]].copy()
        tr_a, va = tr[yr[tr] < Y - 1], tr[yr[tr] == Y - 1]
        b0 = lgb.train(P8, lgb.Dataset(X.iloc[tr_a], label=y_win[tr_a]), num_boost_round=3000, valid_sets=[lgb.Dataset(X.iloc[va], label=y_win[va])],
                       callbacks=[lgb.early_stopping(80, verbose=False)])
        b = lgb.train(P8, lgb.Dataset(X.iloc[tr], label=y_win[tr]), num_boost_round=int(max(50, b0.best_iteration * 1.1)))
        out["win_raw"] = 1.0 / np.exp(b.predict(Xte[cols]))
        w = 0.5 ** (((Y - 1) - yr[tr]) / 1.0)
        bp = lgb.train(P, lgb.Dataset(X.iloc[tr], label=y_pl[tr], weight=w), num_boost_round=500)
        out["place_raw"] = np.exp(bp.predict(Xte[cols]))
        out["win_pred"] = po.normalize_win_odds(out["race_id"].to_numpy(), out["win_raw"].to_numpy())
        out["place_pred"] = np.clip(out["place_raw"], 1.0, None)
        out.to_parquet(f, index=False)
        r = {}
        if Y != 2026:
            t = T.iloc[te]
            r = {"win_rmse_log": float(np.sqrt(((np.log(t["win_odds"].to_numpy()) - np.log(out["win_pred"].to_numpy())) ** 2).mean())),
                 "place_rmse_log": float(np.sqrt(((y_pl[te] - np.log(out["place_pred"].to_numpy())) ** 2).mean()))}
        log[Y] = {"train_rows": int(len(tr)), "best_iter": int(b0.best_iteration), "sec": round(time.time() - t0), **r}
        print(Y, log[Y], flush=True)
        (out_dir / "v9_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
