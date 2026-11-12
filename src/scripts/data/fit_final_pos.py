"""位置取り予想（pos_model 試行 6）の推論用の最終モデル（2016〜2025 の全行で 1 回学習）。ホストで実行（重い処理は /tmp/heavy_run.sh 経由）。

    POS_MODEL_OUT=data/local/modeling/pos_model python -m src.scripts.data.fit_final_pos --out retrain_out/base_2016/final_pos

``pos_model.make_oof`` と同じ 2 段構え:
- 1 段目: 全行（2016〜2025）で学習（乱数 3 本、SLOW 設定）。新しいレースの p1 はこれで出す。
- 2 段目: p1 に「その年の開始 10 日前より前だけで学習した 1 段目の予測」（2018〜2025 年の行）を使い、2018〜2025 年の全行で学習（乱数 5 本）。
  LambdaRank を 1 割混ぜる（``RANK_BLEND``）。
推論側は pos_model と同じく、1 段目・2 段目の乱数平均をそれぞれ 0〜1 に clip し、LambdaRank（RANK_BLEND）を混ぜる。
出力: ``stage1_s<i>.txt`` / ``stage2_s<i>.txt`` / ``rank.txt`` / ``meta.json``（特徴量の並び・カテゴリの水準・乱数・混合比）/ ``sanity.json``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.scripts.data import pos_model as pm

CATS = ("course", "venue_c", "cls", "going")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(pm.OUT / "features.parquet")
    levels = {}
    for c in CATS:
        df[c] = df[c].astype("category")
        levels[c] = [str(v) for v in df[c].cat.categories]
    y = df["pos_first"].to_numpy()
    lab = ~np.isnan(y)
    pm._log(f"走 {len(df):,}、ラベルあり {int(lab.sum()):,}（{df['year'].min()}〜{df['year'].max()}）")

    df["p1"] = np.nan
    for yv in range(pm.STACK_FROM, 2026):
        tr = (df["date"] < pd.Timestamp(f"{yv}-01-01") - pd.Timedelta(days=pm.GAP_DAYS)) & lab
        va = df["year"] == yv
        df.loc[va, "p1"] = pm._fit_predict(df.loc[tr, pm.FEATURES], y[tr], df.loc[va, pm.FEATURES], seeds=pm.STAGE1_SEEDS)
        pm._log(f"1 段目 OOF {yv}")
    S = pm._stage2_cols(df)
    cols2 = pm.FEATURES + list(S.columns)
    X2 = pd.concat([df[pm.FEATURES], S], axis=1)

    for sd in pm.STAGE1_SEEDS:
        b = lgb.train(dict(pm.SLOW, num_threads=pm.THREADS, seed=sd), lgb.Dataset(df.loc[lab, pm.FEATURES], label=y[lab]), num_boost_round=pm.SLOW_ROUNDS)
        b.save_model(str(out / f"stage1_s{sd}.txt"))
        pm._log(f"1 段目 最終 乱数 {sd}")

    tr2 = (df["year"] >= pm.STACK_FROM) & lab
    for sd in pm.STAGE2_SEEDS:
        b = lgb.train(dict(pm.SLOW, num_threads=pm.THREADS, seed=sd), lgb.Dataset(X2.loc[tr2, cols2], label=y[tr2]), num_boost_round=pm.SLOW_ROUNDS)
        b.save_model(str(out / f"stage2_s{sd}.txt"))
        pm._log(f"2 段目 最終 乱数 {sd}")

    d_tr, X_tr, y_tr = df.loc[tr2], X2.loc[tr2, cols2], y[tr2]
    order = np.argsort(d_tr["race_id"].to_numpy(), kind="stable")
    rid = d_tr["race_id"].to_numpy()[order]
    grp = pd.Series(rid).groupby(rid, sort=False).size().to_numpy()
    rel = np.clip((1 - y_tr[order]) * 30, 0, 30).astype(int)
    rk = lgb.train(dict(objective="lambdarank", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                        bagging_freq=1, verbose=-1, num_threads=pm.THREADS, seed=0, label_gain=list(range(31)), lambdarank_truncation_level=30),
                   lgb.Dataset(X_tr.iloc[order], label=rel, group=grp), num_boost_round=600)
    rk.save_model(str(out / "rank.txt"))

    p = np.mean([lgb.Booster(model_file=str(out / f"stage2_s{sd}.txt")).predict(X_tr) for sd in pm.STAGE2_SEEDS], axis=0)
    mae = float(np.mean(np.abs(np.clip(p, 0, 1) - y_tr)))
    (out / "meta.json").write_text(json.dumps({"features": pm.FEATURES, "stage2_cols": cols2, "cats": list(CATS), "levels": levels,
                                               "stage1_seeds": list(pm.STAGE1_SEEDS), "stage2_seeds": list(pm.STAGE2_SEEDS), "rank_blend": pm.RANK_BLEND,
                                               "stack_from": pm.STACK_FROM, "n_train_stage1": int(lab.sum()), "n_train_stage2": int(tr2.sum())},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "sanity.json").write_text(json.dumps({"in_sample_mae_stage2": mae, "pass_line": pm.PASS_LINE}, indent=1), encoding="utf-8")
    pm._log(f"保存 {out}（2 段目の学習内 MAE {mae:.4f}）")


if __name__ == "__main__":
    main()
