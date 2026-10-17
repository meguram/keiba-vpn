"""tp-25b: レース全体のペース予測と、ペース条件付きの能力の PoC。

python -m src.scripts.data.tp25b_eval build   # 学習フレーム・ペースの正解・コース形状・SmartRC 特徴量を作って保存
python -m src.scripts.data.tp25b_eval pace    # レース単位のペース予測（expanding OOF、特徴量セット別の精度）と、馬ごとの傾き・展開相性の特徴量
python -m src.scripts.data.tp25b_eval win     # LightGBM 固定・valid 2022〜2024 expanding・2 seed。V0 に各群を足した増分（test 2025 は使わない）

出力: data/local/modeling/training_pipeline/tp25b/
市場オッズ・人気は使わない。前段の予測は年ごとの expanding OOF（valid 年 Y の予測は Y の 10 日前より前だけで学習）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
OUT = OLD / "tp25b"
SMARTRC = OLD / "smartrc_ten1f"
TP25A = OLD / "tp25a" / "start_stage_oof.parquet"
THREADS = int(os.environ.get("TP25B_THREADS", "5"))
SEEDS = [0, 1]
SUF = "_a" if os.environ.get("TP25B_USE_TP25A") == "1" else ""
OOF_YEARS = (2021, 2022, 2023, 2024)


def _load_base():
    from src.pipeline.models import ablation as ab

    adopted = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    return df, base


def build() -> None:
    import polars as pl

    from src.pipeline.features import smartrc_ten1f as S
    from src.pipeline.features import tp25b_pace as P
    from src.pipeline.features.horse_history.common import load_runs

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df, base = _load_base()
    lab = pd.read_parquet(OLD / "oof/stage_labels.parquet")
    lab["race_id"], lab["horse_id"] = lab["race_id"].astype(str), lab["horse_id"].astype(str)
    key = df[["race_id", "horse_id"]].merge(lab.drop_duplicates(["race_id", "horse_id"])[["race_id", "horse_id", "pos_early", "pos_late", "l3f", "is_hurdle"]],
                                            on=["race_id", "horse_id"], how="left")
    l3f = key["l3f"].where(~key["is_hurdle"].fillna(False).astype(bool))
    g = key["race_id"]
    cnt = l3f.groupby(g).transform("count")
    df["l3f_dev"] = (l3f - l3f.groupby(g).transform("mean")).where(cnt >= 5).to_numpy()
    runs = load_runs().select("race_id", "date", "venue_code", "surface", "distance", "pace_first", "pace_second", "pace_bal", "rpci").unique("race_id").to_pandas()
    runs["race_id"] = runs["race_id"].astype(str)
    runs["venue_code"] = runs["venue_code"].astype(str).str.zfill(2)
    tg = P.race_targets(runs, lab)
    cov = runs.assign(year=pd.to_datetime(runs["date"]).dt.year).groupby("year").agg(
        races=("race_id", "size"), pace_first=("pace_first", lambda x: x.notna().mean()), pace_bal=("pace_bal", lambda x: x.notna().mean()), rpci=("rpci", lambda x: x.notna().mean()))
    tgt_cov = tg.merge(runs[["race_id", "date"]], on="race_id").assign(year=lambda d: pd.to_datetime(d["date"]).dt.year).groupby("year")[P.TARGETS].agg(lambda x: x.notna().mean())
    print("被覆率（ラップ・正解の有無）\n", cov.round(3).to_string(), "\n", tgt_cov.round(3).to_string(), flush=True)
    tg.to_parquet(OUT / "race_targets.parquet", index=False)
    cov.round(4).to_csv(OUT / "pace_label_coverage.csv")

    runners = pd.read_parquet(SMARTRC / "runners.parquet")
    career = pd.read_parquet(SMARTRC / "career.parquet")
    ent = df[["race_id", "horse_id", "date", "distance", "surface"]].copy()
    ent["surface"] = ent["surface"].astype(str)
    feats = S.build_features(ent, career, runners)
    assert (feats["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (feats["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
    sx_cols = S.CORE + S.RELATIVE + S.RUNNER_RANKS
    frame = df[["race_id", "horse_id", "date", "year", "finish_position", "field_size", "horse_number", "l3f_dev"]].copy()
    for c in sx_cols:
        frame[c] = feats[c].to_numpy()
    frame.to_parquet(OUT / "frame_extra.parquet", index=False)
    json.dump({"sx_cols": sx_cols}, open(OUT / "meta.json", "w"))
    print(f"[{time.time() - t0:.0f}s] build 完了 rows={len(df)} races={df['race_id'].nunique()}", flush=True)


def _race_frame(df, extra, geo, P, tp25a: bool):
    from src.pipeline.features import tp25b_pace as P_

    meta = json.loads((OUT / "meta.json").read_text())
    full = df.merge(extra.drop(columns=["date", "year", "finish_position", "field_size", "horse_number"]), on=["race_id", "horse_id"], how="left")
    full["sx_tenz"] = P_.add_tenz(full)
    start = None
    if tp25a and TP25A.exists():
        start = pd.read_parquet(TP25A)
        start["race_id"], start["horse_id"] = start["race_id"].astype(str), start["horse_id"].astype(str)
    R = P_.race_aggregates(full, geo, [c for c in meta["sx_cols"] if c.startswith("sx_race_")] + ["sx_ten_mean3", "sx_ten_mean5", "sx_ten_last"], start)
    return full, R


FEATURE_SETS = {
    "cond": lambda c: [x for x in c if x.startswith("r_") and x not in ("r_f2_pressure", "r_f2_nfront", "r_f2_nfront40", "r_f2_p_fast")],
    "cond+geo": lambda c: [x for x in c if (x.startswith("r_") and not x.startswith("r_f2")) or x.startswith("geo_")],
    "cond+geo+f2": lambda c: [x for x in c if x.startswith(("r_", "geo_"))],
    "cond+geo+style": lambda c: [x for x in c if x.startswith(("r_", "geo_", "s_")) or x == "s_late_mean"],
    "cond+geo+style+pacehist": lambda c: [x for x in c if x.startswith(("r_", "geo_", "s_", "h_"))],
    "cond+geo+f2+tenz": lambda c: [x for x in c if x.startswith(("r_", "geo_", "xz_"))],
    "cond+geo+f2+style3+tenz": lambda c: [x for x in c if x.startswith(("r_", "geo_", "xz_")) or x in ("s_min", "s_min3m", "s_n_lt25", "s_n_lt15")],
    "all(+SmartRC)": lambda c: [x for x in c if not x.startswith("e_")],
    "cond+geo+f2+tp25a": lambda c: [x for x in c if x.startswith(("r_", "geo_", "e_"))],
    "all+tp25a": lambda c: list(c),
}


def pace() -> None:
    import lightgbm as lgb

    from src.pipeline.features import tp25b_pace as P
    from src.pipeline.models import stage_models as sm

    t0 = time.time()
    df, base = _load_base()
    extra = pd.read_parquet(OUT / "frame_extra.parquet")
    tg = pd.read_parquet(OUT / "race_targets.parquet")
    geo = P.course_geo()
    use_a = os.environ.get("TP25B_USE_TP25A") == "1"
    full, R = _race_frame(df, extra, geo, P, tp25a=use_a)
    R = R.merge(tg, on="race_id", how="left")
    R["date"] = R["race_id"].map(df.groupby("race_id")["date"].first())
    R["year"] = pd.to_datetime(R["date"]).dt.year
    R = R.sort_values(["date", "race_id"]).reset_index(drop=True)
    R = R[~(df.drop_duplicates("race_id").set_index("race_id")["surface"].reindex(R["race_id"]).astype(str).eq("障").to_numpy())].reset_index(drop=True)
    feat_all = [c for c in R.columns if c not in ["race_id", "date", "year"] + P.TARGETS]
    params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40, "bagging_fraction": 0.8, "bagging_freq": 1,
              "feature_fraction": 0.7, "lambda_l2": 10.0, "verbosity": -1, "deterministic": True, "force_row_wise": True, "num_threads": THREADS, "seed": 0}
    sets = {k: f(feat_all) for k, f in FEATURE_SETS.items() if use_a or "tp25a" not in k}
    rows, preds = [], {}
    splits = sm.year_splits(R, OOF_YEARS, "expanding")
    for tcol in P.TARGETS:
        y = R[tcol].to_numpy(dtype=float)
        for name, cols in sets.items():
            pr = np.full(len(R), np.nan)
            for yr, tr, va in splits:
                tr = tr[np.isfinite(y[tr])]
                bst = lgb.train(params, lgb.Dataset(R.iloc[tr][cols], label=y[tr]), num_boost_round=400)
                pr[va] = bst.predict(R.iloc[va][cols])
            preds[(tcol, name)] = pr
            for yr in OOF_YEARS:
                m = (R["year"].to_numpy() == yr) & np.isfinite(y)
                if m.sum() == 0:
                    continue
                var = np.var(y[m])
                mse = np.mean((y[m] - pr[m]) ** 2)
                rows.append({"target": tcol, "set": name, "year": yr, "n": int(m.sum()), "r2": 1 - mse / var, "mae": float(np.mean(np.abs(y[m] - pr[m]))),
                             "corr": float(np.corrcoef(y[m], pr[m])[0, 1])})
            print(f"[{time.time() - t0:.0f}s] {tcol} {name}", flush=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / f"pace_accuracy_by_year{SUF}.csv", index=False)
    summ = rep[rep["year"].isin([2022, 2023, 2024])].groupby(["target", "set"])[["r2", "mae", "corr"]].mean().round(4)
    print(summ.to_string(), flush=True)
    summ.to_csv(OUT / f"pace_accuracy_summary{SUF}.csv")
    best = os.environ.get("TP25B_PACE_SET", "cond+geo+f2")
    pred_df = pd.DataFrame({"race_id": R["race_id"]})
    for tcol, pcol in zip(P.TARGETS, P.PRED_COLS):
        pred_df[pcol] = preds[(tcol, best)]
    # 障害レースは予測しない。2020 年は OOF が無いので欠損
    pred_df.to_parquet(OUT / f"race_pace_oof{SUF}.parquet", index=False)

    # 馬ごとの傾き（as-of）と展開相性
    ent = df[["race_id", "horse_id", "date", "year", "horse_number", "field_size", "finish_position", "a_style", "f2_nfront", "f2_lone_front"]].copy()
    if "a_style_late" in df.columns:
        ent["a_style_late"] = df["a_style_late"]
    ent["l3f_dev"] = extra["l3f_dev"].to_numpy()
    ent = ent.join(geo_for(df, geo))
    actual = tg[["race_id", "t_pf_z", "t_bal_z"]].rename(columns={"t_pf_z": "pf_z", "t_bal_z": "bal_z"})
    sl = P.pace_slopes(ent, actual)
    ix = P.interaction_features(ent, pred_df, sl)
    feats = pd.concat([ent[["race_id", "horse_id"]], sl, ix], axis=1)
    feats.to_parquet(OUT / f"features{SUF}.parquet", index=False)
    cols = {"slopes": list(sl.columns), "interaction": list(ix.columns)}
    json.dump(cols, open(OUT / f"feature_groups{SUF}.json", "w"), ensure_ascii=False, indent=1)
    print(f"[{time.time() - t0:.0f}s] features{SUF}.parquet {feats.shape}", flush=True)


def geo_for(df, geo):
    k = df[["venue_code", "surface", "distance"]].copy()
    k["venue_code"] = k["venue_code"].astype(str)
    k["surface"] = k["surface"].astype(str)
    k["distance"] = k["distance"].astype(int)
    return k.merge(geo, on=["venue_code", "surface", "distance"], how="left")[[c for c in geo.columns if c.startswith("geo_")]].set_axis(df.index)


def win() -> None:
    from src.pipeline.models import feature_bakeoff as fb
    from src.pipeline.models import zoo, zoo_runner
    from src.pipeline.models.time_split import yearly_expanding_folds

    t0 = time.time()
    df, base = _load_base()
    feats = pd.read_parquet(OUT / f"features{SUF}.parquet")
    assert (feats["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (feats["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
    groups = json.loads((OUT / f"feature_groups{SUF}.json").read_text())
    stage = pd.read_parquet(OLD / "oof/stage_features_expanding.parquet")
    stage["race_id"], stage["horse_id"] = stage["race_id"].astype(str), stage["horse_id"].astype(str)
    S3 = df[["race_id", "horse_id"]].merge(stage[["race_id", "horse_id", "s_time_pred", "s_time_gap", "s_rank_pred"]], on=["race_id", "horse_id"], how="left")
    S3.index = df.index
    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    fin_map = meta.set_index(["race_id", "horse_id"])["finish_position"]
    sl, ix = groups["slopes"], groups["interaction"]
    pp = [c for c in ix if c.startswith("pp_")]
    ex = [c for c in ix if c.startswith("ex_")]
    inter = [c for c in ix if c.startswith("ix_")]
    B = df[base]

    def X(*parts):
        return pd.concat([B] + [p for p in parts], axis=1)

    variants = {
        "V0_base": B,
        "V0S_base+S3": X(S3[["s_time_pred", "s_time_gap", "s_rank_pred"]]),
        "P1_pred_pace": X(feats[pp]),
        "P2_slopes": X(feats[sl]),
        "P3_pred+slopes+expected": X(feats[pp + sl + ex]),
        "P4_pred+interaction": X(feats[pp + inter]),
        "P5_all": X(feats[pp + sl + ex + inter]),
        "P6_all+S3": X(feats[pp + sl + ex + inter], S3[["s_time_pred", "s_time_gap", "s_rank_pred"]]),
    }
    only = os.environ.get("VARIANTS")
    if only:
        variants = {k: v for k, v in variants.items() if k in only.split(",") or k == "V0_base"}
    cache = OUT / f"oof_cache{SUF}"
    cache.mkdir(exist_ok=True)
    oofs, levels = {}, {}
    for name, Xv in variants.items():
        f = cache / f"{name}.pkl"
        if f.exists():
            res, oof = pd.read_pickle(f)
        else:
            res, oof = zoo_runner.run_oof(name, "lgb_binary", Xv, meta, folds, SEEDS, THREADS, params=dict(lp))
            pd.to_pickle((res, oof), f)
        oofs[name], levels[name] = oof, res
        print(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
    pairs = [(n, "V0_base", f"{n} − V0") for n in variants if n not in ("V0_base",)]
    if "V0S_base+S3" in variants and "P6_all+S3" in variants:
        pairs.append(("P6_all+S3", "V0S_base+S3", "P6 − V0S（現行前段に対する増分）"))
    cmp_ = {}
    for new, basev, label in pairs:
        c = fb.compare_oof(oofs[basev], oofs[new], fin_map, label)
        cmp_[label] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
        print(label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
    rep = {"levels": {k: {m: float(v[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")} for k, v in levels.items()},
           "by_fold": {k: v[["fold", "logloss_win", "logloss_top2", "logloss_top3"]].round(5).to_dict("records") for k, v in levels.items()},
           "compare": cmp_, "seeds": SEEDS, "rows": len(df)}
    (OUT / f"win_report{SUF}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    {"build": build, "pace": pace, "win": win}[sys.argv[1]]()
