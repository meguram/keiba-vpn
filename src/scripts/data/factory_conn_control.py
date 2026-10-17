"""ファクトリー G2: 馬主系の特徴量の対照実験（効果が大きいので、先読み・取得時点のずれ・偶然でないかを確かめる）。

python -m src.scripts.data.factory_conn_control importance            # V0 + streak_o / win_o / ent_o の gain 上位
python -m src.scripts.data.factory_conn_control shuffle               # 馬主を馬のあいだでランダムに入れ替えて作り直す（効果が消えるはず）
python -m src.scripts.data.factory_conn_control big                   # 馬主を『JRA で 30 頭以上走らせた馬主』に限る（地方転籍・売却先などの小さな馬主を除く）
評価は valid 2024 + seed 2,3（第2段と同じ）。結果は factory/conn/control.json。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src.pipeline.features import factory_conn as fc
from src.scripts.data import factory_conn_eval as fe

KEY = fe.KEY
PICK = {"streak_o": "streak_o", "win_o": "win_o", "ent_o": "ent_o"}


def _frame(df: pd.DataFrame, h: pd.DataFrame, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    F = pd.DataFrame({c: np.asarray(a, dtype="float32") for c, a in cols.items()})
    F[KEY[0]], F[KEY[1]] = h["race_id"].to_numpy(), h["horse_id"].to_numpy()
    F = F.drop_duplicates(KEY)
    m = df[KEY].merge(F, on=KEY, how="left")
    m.index = df.index
    return m.drop(columns=KEY)


def _eval(tag: str, df, base, lp, meta, folds, threads, fin_map, o0, feats: pd.DataFrame, out: dict) -> None:
    from src.pipeline.models import feature_bakeoff as fb

    r, o = fe._run(tag, pd.concat([df[base], feats], axis=1), meta, folds[2:], [2, 3], threads, lp)
    c = fb.compare_oof(o0, o, fin_map, tag)
    out[tag] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()} | {"n_cols": int(feats.shape[1])}
    print(f"{tag}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}", flush=True)
    (fe.OUT / "control.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def importance() -> None:
    import lightgbm as lgb

    df, base, lp, meta, folds, threads, _ = fe._setup()
    tr = folds[2].train_index
    y = (meta["finish_position"].to_numpy() == 1).astype(int)
    res = {}
    for g in ("streak_o", "win_o", "ent_o"):
        X = pd.concat([df[base], fe._group_frame(df, g)], axis=1)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               reg_lambda=5, n_jobs=threads, importance_type="gain", verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        imp = pd.Series(m.feature_importances_, index=X.columns)
        new = imp[[c for c in X.columns if c not in base]].sort_values(ascending=False)
        res[g] = {"new_gain_share": float(new.sum() / imp.sum()), "top": {k: float(v / imp.sum()) for k, v in new.head(10).items()}}
        print(g, json.dumps(res[g], ensure_ascii=False)[:600], flush=True)
    (fe.OUT / "importance.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def variants(mode: str) -> None:
    df, base, lp, meta, folds, threads, fin_map = fe._setup()
    o0 = pd.read_parquet(fe.OUT / "oof_base_stage2.parquet")
    h = fc.prepare(fc.load_history())
    rng = np.random.default_rng(0)
    prof = h.drop_duplicates("horse_id")[["horse_id", "owner"]]
    if mode == "shuffle":
        perm = dict(zip(prof["horse_id"], rng.permutation(prof["owner"].to_numpy())))
        h["owner"] = h["horse_id"].map(perm)
    else:
        cnt = prof.groupby("owner")["horse_id"].nunique()
        keep = set(cnt[cnt >= 30].index)
        print("30 頭以上の馬主:", len(keep), "/", len(cnt), "馬カバー率", float(prof["owner"].isin(keep).mean()), flush=True)
        h["owner"] = h["owner"].where(h["owner"].isin(keep))
    groups = fc.build_all(h, entities=["o"], conds=fc.CONDS)
    out = json.loads((fe.OUT / "control.json").read_text(encoding="utf-8")) if (fe.OUT / "control.json").exists() else {}
    for g in ("streak_o", "win_o", "ent_o"):
        _eval(f"{mode}:{g}", df, base, lp, meta, folds, threads, fin_map, o0, _frame(df, h, groups[g]), out)


if __name__ == "__main__":
    {"importance": importance, "shuffle": lambda: variants("shuffle"), "big": lambda: variants("big")}[sys.argv[1]]()
