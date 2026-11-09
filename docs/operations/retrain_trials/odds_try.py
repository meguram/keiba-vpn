"""オッズ予測（単勝・複勝）の試行。odds_pred_eval の phase1 と同じ分け方（各年より前の年だけ）で OOF を作り、eval1 で測る。

    ODDS_TRY=2 PYTHONPATH=. python docs/operations/retrain_trials/odds_try.py
    → retrain_out/odds_try<N>/wp_raw_oof.parquet、続けて eval1 を同じ出力先で実行する

設定（環境変数）:
  ODDS_REFIT=1     早期終了で決めた本数で、学習＋検証の年をまとめて学習し直す
  ODDS_OBJ=softmax 単勝はレース内のソフトマックス（市場の占有率との交差エントロピー）で学習
  ODDS_SEEDS=n     乱数を変えて平均
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
N = os.environ.get("ODDS_TRY", "x")
OUT = Path(f"/home/jovyan/work/retrain_out/odds_try{N}")
os.environ["ODDS_PRED_OUT"] = str(OUT)
from src.scripts.data import odds_pred_eval as oe  # noqa: E402

oe.OUT = OUT
TRAIN_FROM = int(os.environ.get("ODDS_TRAIN_FROM", "2020"))
if os.environ.get("REBUILD_2016") == "1":     # 2016〜2025 年の土台と、2016〜2019 年を足したオッズの表
    from src.scripts.data import rebuild_groups_2016 as _rg

    _rg.redirect()
    oe.BET = _rg.BASE / "betting"
REFIT = os.environ.get("ODDS_REFIT") == "1"
OBJ = os.environ.get("ODDS_OBJ", "l2")
SEEDS = int(os.environ.get("ODDS_SEEDS", "1"))
THREADS = int(os.environ.get("ODDS_THREADS", "4"))
EXTRA = os.environ.get("ODDS_EXTRA", "")
REL = int(os.environ.get("ODDS_REL", "0"))  # 重要な上位 K 列のレース内の順位・平均との差を足す  # 追加の特徴量（parquet、race_id・horse_id で結合）
PARAMS = dict(learning_rate=0.06, num_leaves=63, min_data_in_leaf=60, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.4,
              lambda_l2=5.0, verbose=-1, num_threads=THREADS)
for kv in filter(None, os.environ.get("ODDS_PARAMS", "").split(",")):
    k, v = kv.split("=")
    PARAMS[k] = float(v) if "." in v else int(v)


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def group_index(race: np.ndarray) -> np.ndarray:
    return pd.factorize(race)[0]


def softmax_obj(gidx: np.ndarray, target: np.ndarray):
    def f(preds, _ds):
        m = pd.Series(preds).groupby(gidx).transform("max").to_numpy()
        e = np.exp(preds - m)
        p = e / np.bincount(gidx, e)[gidx]
        return p - target, np.maximum(p * (1 - p), 1e-6)
    return f


def softmax_ce(gidx, target):
    def f(preds, _ds):
        m = pd.Series(preds).groupby(gidx).transform("max").to_numpy()
        lp = preds - m - np.log(np.bincount(gidx, np.exp(preds - m)))[gidx]
        return "ce", float(-(target * lp).sum() / len(np.unique(gidx))), False
    return f


def fit_predict(Xtr, ytr, rtr, Xva, yva, rva, Xte, softmax: bool) -> tuple[np.ndarray, int]:
    import lightgbm as lgb

    preds, its = [], []
    for s in range(SEEDS):
        p = dict(PARAMS, seed=s)
        if softmax:
            gtr, gva = group_index(rtr), group_index(rva)
            ttr, tva = np.exp(ytr), np.exp(yva)  # y = log 占有率
            p["objective"] = softmax_obj(gtr, ttr)
            dtr = lgb.Dataset(Xtr, ttr, free_raw_data=False)
            dva = lgb.Dataset(Xva, tva, reference=dtr)
            m = lgb.train(dict(p, metric="None"), dtr, 3000, valid_sets=[dva], feval=softmax_ce(gva, tva),
                          callbacks=[lgb.early_stopping(60, verbose=False)])
        else:
            p["objective"] = OBJ if OBJ != "softmax" else "l2"
            dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
            m = lgb.train(p, dtr, 3000, valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)], callbacks=[lgb.early_stopping(60, verbose=False)])
        it = int(m.best_iteration or m.current_iteration())
        if REFIT:
            Xa, ya, ra = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), np.concatenate([rtr, rva])
            if softmax:
                p["objective"] = softmax_obj(group_index(ra), np.exp(ya))
                m = lgb.train(dict(p, metric="None"), lgb.Dataset(Xa, np.exp(ya)), int(it * 1.1))
            else:
                m = lgb.train(p, lgb.Dataset(Xa, ya), int(it * 1.1))
        preds.append(m.predict(Xte, raw_score=True) if softmax else m.predict(Xte))
        its.append(it)
    return np.mean(preds, axis=0), int(np.mean(its))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df, X = oe._frame_with_targets()
    keys = df[oe.KEY].astype(str).reset_index(drop=True)
    if EXTRA:
        ex = pd.read_parquet(EXTRA)
        ex["race_id"], ex["horse_id"] = ex["race_id"].astype(str), ex["horse_id"].astype(str)
        ex = keys.merge(ex.drop_duplicates(oe.KEY), on=oe.KEY, how="left").drop(columns=oe.KEY)
        ex = ex.select_dtypes(include=[np.number]).astype("float32")
        from src.pipeline.betting import pred_odds as po
        po.assert_no_odds_features(list(ex.columns))
        X = pd.concat([X.reset_index(drop=True), ex], axis=1)
        log(f"追加の列 {ex.shape[1]}")
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    wp = oe.load_wp()
    T = keys.assign(_i=np.arange(len(keys))).merge(wp[oe.KEY + ["win_odds", "place_mid", "p_mkt"]], on=oe.KEY, how="inner")
    T = T.sort_values(["race_id", "horse_id"]).reset_index(drop=True)  # ソフトマックスのためレースを連続に
    idx = T["_i"].to_numpy()
    Xs = X.iloc[idx].to_numpy(np.float32)
    yr = year[idx]
    rid = T["race_id"].to_numpy()
    y_share = np.log(T["p_mkt"].to_numpy())
    y_place = np.log(T["place_mid"].to_numpy())
    if REL:
        import lightgbm as lgb
        first = (yr < oe.YEARS[0] - 1) & (yr >= 2020)
        m0 = lgb.train(dict(PARAMS, objective="l2", seed=0), lgb.Dataset(Xs[first], y_share[first]), 400)
        imp = m0.feature_importance("gain")
        top = np.argsort(-imp)[:REL]
        F = pd.DataFrame(Xs[:, top], columns=[f"c{i}" for i in top])
        g = F.groupby(rid)
        rel = pd.concat([g.rank(pct=True).add_prefix("rk_"), (F - g.transform("mean")).add_prefix("dm_"), (F - g.transform("max")).add_prefix("dx_")], axis=1)
        Xs = np.hstack([Xs, rel.to_numpy(np.float32)])
        log(f"レース内の相対値 {rel.shape[1]} 列（上位 {REL} 列: {[X.columns[i] for i in top[:10]]} …）")
    log(f"行 {len(T)}、列 {Xs.shape[1]}、設定 REFIT={REFIT} OBJ={OBJ} SEEDS={SEEDS} PARAMS={ {k: v for k, v in PARAMS.items() if k != 'num_threads'} }")
    out = pd.DataFrame({"race_id": rid, "horse_id": T["horse_id"].to_numpy(), "year": yr, "share_raw": np.nan, "place_raw": np.nan})
    info = {}
    for Y in oe.YEARS:
        te = yr == Y
        tr_all = (yr < Y) & (yr >= TRAIN_FROM)
        va = tr_all & (yr == Y - 1)
        tr = tr_all & (yr < Y - 1)
        for name, y in (("share", y_share), ("place", y_place)):
            sm = name == "share" and OBJ == "softmax"
            p, it = fit_predict(Xs[tr], y[tr], rid[tr], Xs[va], y[va], rid[va], Xs[te], sm)
            if sm:  # 生のスコア → レース内の log 占有率（eval1 は share_raw を log 占有率として正規化する）
                g = pd.Series(p).groupby(rid[te])
                p = p - g.transform("max").to_numpy() - np.log(pd.Series(np.exp(p - g.transform("max").to_numpy())).groupby(rid[te]).transform("sum").to_numpy())
            out.loc[te, f"{name}_raw"] = p
            info[f"{Y}_{name}"] = {"iters": it, "n_train": int(tr.sum() + (va.sum() if REFIT else 0))}
            log(f"{Y} {name}: 木 {it} 本")
    out.to_parquet(OUT / "wp_raw_oof.parquet", index=False)
    (OUT / "wp_fit_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    oe.evaluate_phase1()
    r = json.loads((OUT / "phase1_results.json").read_text(encoding="utf-8"))["by_year"]
    s = {Y: {"win_B": round(v["win"]["win_B"]["rmse"], 4), "place_B": round(v["place"]["place_B"]["rmse"], 4)} for Y, v in r.items()}
    log("RMSE（B）: " + json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
