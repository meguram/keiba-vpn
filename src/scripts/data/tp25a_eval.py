"""tp-25a: スタート(テン1F)・位置取りの前段モデルを作り込み、win への増分を測る。

python -m src.scripts.data.tp25a_eval build   # 学習表(修正後) + SmartRC の特徴量/目的変数 → tp25a/stage_frame.parquet
python -m src.scripts.data.tp25a_eval stage   # 年ごとの expanding OOF（2021〜2025）→ tp25a/start_stage_oof.parquet と前段の精度表
python -m src.scripts.data.tp25a_eval win     # V0 / +現行前段3列 / +今回の前段（raw・rel・agg）の勝率 LogLoss 差（valid 2022〜2024、LightGBM 固定、2 seed）
環境変数: TP25A_THREADS（既定 5）、TP25A_SEEDS（既定 0,1）、TP25A_VARIANTS（stage の入力セット A,B,C を絞る）
test 2025 は使わない（2025 年の OOF は他 PoC のために作るが win の評価には使わない）。市場オッズ・人気系の列は使わない。
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
SMART = OLD / "smartrc_ten1f"
OUT = OLD / "tp25a"
THREADS = int(os.environ.get("TP25A_THREADS", "5"))
SEEDS = [int(x) for x in os.environ.get("TP25A_SEEDS", "0,1").split(",")]
STAGE_YEARS = (2021, 2022, 2023, 2024, 2025)
PURGE = pd.Timedelta(days=10)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base():
    from src.pipeline.models import ablation as ab

    adopted = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    return df, base


def cmd_build() -> None:
    from src.pipeline.features import smartrc_ten1f as S
    from src.pipeline.features import tp25a_start as T

    OUT.mkdir(parents=True, exist_ok=True)
    df, base = load_base()
    runners = pd.read_parquet(SMART / "runners.parquet")
    career = pd.read_parquet(SMART / "career.parquet")
    ent = df[["race_id", "horse_id", "date", "distance", "surface"]].copy()
    ent["surface"] = ent["surface"].astype(str)
    sx = S.build_features(ent, career, runners)
    assert (sx["race_id"].to_numpy() == df["race_id"].to_numpy()).all()
    sx_cols = S.CORE + S.RELATIVE + S.RUNNER_RANKS
    tg = S.race_targets(df[["race_id", "horse_id", "date"]], career)
    assert (tg["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
    F = pd.DataFrame({"race_id": df["race_id"], "horse_id": df["horse_id"], "date": df["date"], "year": df["year"]})
    for c in sx_cols:
        F[c] = sx[c].to_numpy()
    for c in ("ten_actual", "ten_rel_actual", "c1_rel_actual"):
        F[c] = tg[c].to_numpy()
    F["y_class"] = T.three_class(F["c1_rel_actual"].to_numpy())
    F["y_rank"] = T.rank_label(F["race_id"].to_numpy(), F["ten_rel_actual"].to_numpy())
    hist = T.horse_history_features(df, career)
    for c in hist.columns:
        F[c] = hist[c].to_numpy()
    crs = T.add_course_start(df)
    for c in crs.columns:
        F[c] = crs[c].to_numpy()
    dd = df[["date", "jockey_id", "trainer_id", "venue_code", "surface", "distance", "horse_number", "field_size"]].copy()
    dd["ten_rel_actual"], dd["c1_rel_actual"] = F["ten_rel_actual"].to_numpy(), F["c1_rel_actual"].to_numpy()
    tend = T.tendency_features(dd)
    for c in tend.columns:
        F[c] = tend[c].to_numpy()
    new_cols = [c for c in F.columns if c.startswith(T.PREFIX)]
    F.to_parquet(OUT / "stage_frame.parquet", index=False)
    meta = {"rows": len(F), "sx_cols": sx_cols, "new_cols": new_cols, "base": base,
            "target_cover": {c: float(F[c].notna().mean()) for c in ("ten_rel_actual", "c1_rel_actual")},
            "class_share": {int(k): float(v) for k, v in F["y_class"][F["y_class"] >= 0].value_counts(normalize=True).items()}}
    (OUT / "stage_frame_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"stage_frame {F.shape} / 新規 {len(new_cols)} 列 / 目的変数の被覆 {meta['target_cover']} / 3区分 {meta['class_share']}")


def _inputs():
    from src.pipeline.features import tp25a_start as T

    df, base = load_base()
    F = pd.read_parquet(OUT / "stage_frame.parquet")
    assert (F["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (F["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
    meta = json.loads((OUT / "stage_frame_meta.json").read_text(encoding="utf-8"))
    X = pd.concat([df[base], F[[c for c in meta["sx_cols"] + meta["new_cols"]]]], axis=1)
    sets = {"A": base, "B": base + meta["sx_cols"], "C": base + meta["sx_cols"] + meta["new_cols"]}
    return df, F, X, sets


def _spearman(race: np.ndarray, a: np.ndarray, b: np.ndarray, min_n: int = 5) -> float:
    d = pd.DataFrame({"r": race, "a": a, "b": b}).dropna()
    d = d[d.groupby("r")["a"].transform("size") >= min_n]
    g = d.groupby("r")
    ra, rb = g["a"].rank(), g["b"].rank()
    ra, rb = ra - ra.groupby(d["r"]).transform("mean"), rb - rb.groupby(d["r"]).transform("mean")
    num = (ra * rb).groupby(d["r"]).sum()
    den = np.sqrt((ra ** 2).groupby(d["r"]).sum() * (rb ** 2).groupby(d["r"]).sum())
    return float((num / den.replace(0, np.nan)).mean())


def cmd_stage() -> None:
    import lightgbm as lgb

    df, F, X, sets = _inputs()
    year = F["year"].to_numpy()
    dates = pd.to_datetime(F["date"]).to_numpy()
    cats = {k: [c for c in v if str(X[c].dtype) == "category"] for k, v in sets.items()}
    p_reg = dict(objective="regression", metric="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbosity=-1, deterministic=True, force_row_wise=True, num_threads=THREADS)
    p_cls = dict(p_reg, objective="multiclass", num_class=3, metric="multi_logloss")
    p_rank = dict(p_reg, objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10, label_gain=[0, 1, 3, 7, 15])
    rounds = 400
    want = os.environ.get("TP25A_VARIANTS", "A,B,C").split(",")
    race = F["race_id"].to_numpy()
    ten, c1, ycls, yrank = F["ten_rel_actual"].to_numpy(), F["c1_rel_actual"].to_numpy(), F["y_class"].to_numpy(), F["y_rank"].to_numpy()
    res: dict[str, pd.DataFrame] = {}
    for var in want:
        cols = sets[var]
        o = pd.DataFrame({"race_id": F["race_id"], "horse_id": F["horse_id"], "year": F["year"]})
        o["ts_ten_pred"] = np.nan
        o["ts_c1_pred"] = np.nan
        o["ts_p_front"] = o["ts_p_mid"] = o["ts_p_back"] = np.nan
        o["ts_ten_rank"] = np.nan
        for Y in STAGE_YEARS:
            tr_all = (year < Y) & (dates < np.datetime64(f"{Y}-01-01") - np.timedelta64(10, "D"))
            va = np.flatnonzero(year == Y)
            Xtr_all = X.loc[tr_all, cols]
            for tname, y, prm, key in (("ten", ten, p_reg, "ts_ten_pred"), ("c1", c1, p_reg, "ts_c1_pred")):
                tr = np.flatnonzero(tr_all & ~np.isnan(y))
                bst = lgb.train(dict(prm, seed=0), lgb.Dataset(X.iloc[tr][cols], label=y[tr], categorical_feature=cats[var] or "auto"), num_boost_round=rounds)
                o.loc[va, key] = bst.predict(X.iloc[va][cols])
                _log(f"{var} {tname} {Y} 学習 {len(tr):,}")
            if var == "C" or os.environ.get("TP25A_ALL_TARGETS"):
                tr = np.flatnonzero(tr_all & (ycls >= 0))
                bst = lgb.train(dict(p_cls, seed=0), lgb.Dataset(X.iloc[tr][cols], label=ycls[tr], categorical_feature=cats[var] or "auto"), num_boost_round=rounds)
                P = bst.predict(X.iloc[va][cols])
                o.loc[va, ["ts_p_front", "ts_p_mid", "ts_p_back"]] = P
                tr = np.flatnonzero(tr_all & (yrank >= 0))
                # lambdarank はレースごとにまとまった並びが必要（df は日付・race_id 順なので連続）
                grp = pd.Series(race[tr]).groupby(race[tr], sort=False).size().to_numpy()
                bst = lgb.train(dict(p_rank, seed=0), lgb.Dataset(X.iloc[tr][cols], label=yrank[tr], group=grp, categorical_feature=cats[var] or "auto"), num_boost_round=300)
                o.loc[va, "ts_ten_rank"] = bst.predict(X.iloc[va][cols])
                _log(f"{var} cls+rank {Y}")
        res[var] = o
        o.to_parquet(OUT / f"stage_oof_{var}.parquet", index=False)
        if var == "C":
            o.to_parquet(OUT / "start_stage_oof.parquet", index=False)
            _log("保存: start_stage_oof.parquet")
    cmd_accuracy()


def cmd_accuracy() -> None:
    df, F, X, sets = _inputs()
    year = F["year"].to_numpy()
    race = F["race_id"].to_numpy()
    ten, c1, ycls = F["ten_rel_actual"].to_numpy(), F["c1_rel_actual"].to_numpy(), F["y_class"].to_numpy()
    res = {v: pd.read_parquet(OUT / f"stage_oof_{v}.parquet") for v in ("A", "B", "C") if (OUT / f"stage_oof_{v}.parquet").exists()}
    rows = []
    for var, o in res.items():
        for label, ys in (("2022-24", (2022, 2023, 2024)), ("2025", (2025,))):
            m = np.isin(year, ys)
            r = {"variant": var, "years": label}
            for tn, y, pc in (("ten", ten, "ts_ten_pred"), ("c1", c1, "ts_c1_pred")):
                ok = m & ~np.isnan(y) & o[pc].notna().to_numpy()
                r[f"{tn}_mae"] = float(np.abs(y[ok] - o[pc].to_numpy()[ok]).mean())
                r[f"{tn}_mae_base"] = float(np.abs(y[ok] - np.nanmean(y[ok])).mean())
                r[f"{tn}_spearman"] = _spearman(race[ok], y[ok], o[pc].to_numpy()[ok])
            if o["ts_p_front"].notna().any():
                ok = m & (ycls >= 0) & o["ts_p_front"].notna().to_numpy()
                P = o.loc[ok, ["ts_p_front", "ts_p_mid", "ts_p_back"]].to_numpy().clip(1e-9, 1)
                r["cls_logloss"] = float(-np.log(P[np.arange(ok.sum()), ycls[ok]]).mean())
                prior = np.bincount(ycls[ok]) / ok.sum()
                r["cls_logloss_prior"] = float(-np.log(prior[ycls[ok]]).mean())
                ok2 = m & ~np.isnan(ten) & o["ts_ten_rank"].notna().to_numpy()
                r["rank_spearman"] = _spearman(race[ok2], ten[ok2], -o["ts_ten_rank"].to_numpy()[ok2])
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "stage_accuracy.csv", index=False)
    print(tab.round(4).to_string())


def cmd_win() -> None:
    from src.pipeline.features import tp25a_start as T
    from src.pipeline.models import feature_bakeoff as fb, zoo, zoo_runner
    from src.pipeline.models.time_split import yearly_expanding_folds

    df, base = load_base()
    oof_path = Path(os.environ.get("TP25A_OOF") or OUT / "start_stage_oof.parquet")
    oof = pd.read_parquet(oof_path)
    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    fin_map = meta.set_index(["race_id", "horse_id"])["finish_position"]
    s3 = pd.read_parquet(OLD / "oof/stage_features_expanding.parquet")[["race_id", "horse_id", "s_time_pred", "s_time_gap", "s_rank_pred"]]
    s3 = df[["race_id", "horse_id"]].merge(s3.assign(race_id=s3["race_id"].astype(str), horse_id=s3["horse_id"].astype(str)), on=["race_id", "horse_id"], how="left")
    s3.index = df.index
    S3 = s3[["s_time_pred", "s_time_gap", "s_rank_pred"]]
    W = T.win_columns(df, oof)
    X0 = df[base]
    variants = {
        "V0": X0,
        "V0+S3": pd.concat([X0, S3], axis=1),
        "V0+raw": pd.concat([X0, W["raw"]], axis=1),
        "V0+raw+rel": pd.concat([X0, W["raw"], W["rel"]], axis=1),
        "V0+raw+agg": pd.concat([X0, W["raw"], W["agg"]], axis=1),
        "V0+raw+rel+agg": pd.concat([X0, W["raw"], W["rel"], W["agg"]], axis=1),
        "V0+S3+raw+rel+agg": pd.concat([X0, S3, W["raw"], W["rel"], W["agg"]], axis=1),
    }
    if os.environ.get("TP25A_DIRECT"):       # 対照: 前段を介さず、SmartRC/新規の入力特徴量を直接足す
        fm = json.loads((OUT / "stage_frame_meta.json").read_text(encoding="utf-8"))
        Fr = pd.read_parquet(OUT / "stage_frame.parquet")
        direct = Fr[fm["sx_cols"] + fm["new_cols"]]
        direct.index = df.index
        variants["V0+direct"] = pd.concat([X0, direct], axis=1)
        variants["V0+S3+direct"] = pd.concat([X0, S3, direct], axis=1)
    only = os.environ.get("TP25A_ONLY")
    if only:
        variants = {k: v for k, v in variants.items() if k in only.split(",") or k == "V0"}
    out = OUT / (os.environ.get("TP25A_TAG") or "win")
    out.mkdir(parents=True, exist_ok=True)
    oofs, levels = {}, {}
    t0 = time.time()
    for name, X in variants.items():
        res, o = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, SEEDS, THREADS, params=dict(lp))
        oofs[name] = o
        levels[name] = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
        _log(f"{name}: win {levels[name]['logloss_win']:.5f} top2 {levels[name]['logloss_top2']:.5f} top3 {levels[name]['logloss_top3']:.5f} ({time.time() - t0:.0f}s)")
    cmp_ = {}
    pairs = [(k, "V0") for k in variants if k != "V0"] + [("V0+S3+direct", "V0+S3")] + [(k, "V0+S3") for k in variants if k.startswith("V0+S3+")] + [(k, "V0+S3") for k in ("V0+raw+rel+agg",) if k in variants]
    for new, basev in pairs:
        if new == basev or new not in oofs or basev not in oofs:
            continue
        c = fb.compare_oof(oofs[basev], oofs[new], fin_map, f"{new} − {basev}")
        cmp_[f"{new} − {basev}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
        _log(f"{new} − {basev}: " + str({k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()}))
    (out / "report.json").write_text(json.dumps({"levels": levels, "compare": cmp_, "seeds": SEEDS, "oof": str(oof_path)}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    {"build": cmd_build, "stage": cmd_stage, "accuracy": cmd_accuracy, "win": cmd_win}[sys.argv[1]]()
