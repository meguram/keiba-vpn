"""tp-07（v4 の最終再学習）の土台: 先読み修正・血統キー復旧・内回り/外回りを入れた LayerA 母表と学習表、その V0 評価、前段 S3 の OOF。

python -m src.scripts.data.tp07_base build   # data/local/modeling/training_pipeline_v4/ に layer_a.parquet と features/train_table.parquet
python -m src.scripts.data.tp07_base eval    # V0（tp-03 採用 175 列）を旧 fixed 表 / 血統のみ復旧の表と比べる（valid 2022〜24, 2 seed）
python -m src.scripts.data.tp07_base stage   # 前段 S3（s_time_pred / s_time_gap / s_rank_pred）の年ごとの expanding OOF を新土台で作り直す

既存のディレクトリは上書きしない。履歴系の列（horse_history 127 列）は LayerA の騎手・調教師の通算成績に依存しないので、修正後の表（training_pipeline_fixed）から引き継ぐ。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
PED = ROOT / "data/local/modeling/training_pipeline_ped"
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
KEY = ["race_id", "horse_id"]


def build() -> None:
    import polars as pl

    from src.pipeline.features.course_variant import add_course_variant
    from src.pipeline.models.layer_a_dataset import build_layer_a_dataframe
    from src.pipeline.models.memory_utils import downcast_frame
    from src.pipeline.models.time_split import TRAINING_YEARS

    t0 = time.time()
    (V4 / "features").mkdir(parents=True, exist_ok=True)
    keys = pd.read_parquet(PED / "pedigree_keys.parquet")
    print(f"[{time.time() - t0:.0f}s] 血統キー {len(keys)} 頭", flush=True)
    full = build_layer_a_dataframe(years=list(TRAINING_YEARS), jra_only=True, pedigree_keys=keys)
    # 内回り/外回り: 出馬表の course_type は全行空だったので、(場, 面, 距離) からコースマスタで導出（決まらないレースは欠損 + フラグ）
    ct = add_course_variant(full, parsed_col="course_type", parsed_available=False)
    full["course_type_unknown"] = ct.isna().astype(np.int8)
    full["course_type"] = ct.fillna("不明")             # "" は内外の区別がないコース、"不明" はマスタで決まらないコース
    full = downcast_frame(full, keep=["race_id", "horse_id", "horse_number", "date"])
    full.to_parquet(V4 / "layer_a.parquet", index=False)
    print(f"[{time.time() - t0:.0f}s] 母表 {full.shape} / course_type 不明 {full['course_type_unknown'].mean():.4f} / 値 {full['course_type'].value_counts().to_dict()}", flush=True)

    fx = pl.read_parquet(FIXED / "features/train_table.parquet")
    la_cols_fixed = set(pl.read_parquet_schema(FIXED / "layer_a.parquet").keys())
    family = [c for c in fx.columns if c in KEY or c not in la_cols_fixed]          # 履歴系（horse_history 由来）の列
    la = pl.from_pandas(full.drop(columns=[c for c in ("past_races", "training") if c in full.columns]))     # 大きいリスト列は学習表に入れない（fixed と同じ）
    la = la.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    T = la.join(fx.select(family), on=KEY, how="left")
    assert T.height == fx.height == la.height, (T.height, fx.height, la.height)
    T.write_parquet(V4 / "features/train_table.parquet")
    (V4 / "features/feature_families.json").write_text((FIXED / "features/feature_families.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[{time.time() - t0:.0f}s] 学習表 {T.shape}", flush=True)

    # 旧 fixed 表との差（共通の列ごと: 変わった行の割合と平均絶対差）
    old = pl.read_parquet(FIXED / "features/train_table.parquet")
    j = T.select(KEY + [c for c in T.columns if c in old.columns and c not in KEY]).join(old.select([c for c in old.columns]), on=KEY, suffix="_old")
    rows = []
    for c in T.columns:
        if c in KEY or c + "_old" not in j.columns:
            continue
        a, b = j[c], j[c + "_old"]
        if a.dtype.is_numeric() and b.dtype.is_numeric():
            x, y = a.cast(pl.Float64).to_numpy(), b.cast(pl.Float64).to_numpy()
            both = np.isfinite(x) & np.isfinite(y)
            ch = float(np.mean(~(np.isclose(x, y, rtol=1e-6, atol=1e-9) | (np.isnan(x) & np.isnan(y)))))
            rows.append({"col": c, "changed_share": round(ch, 4), "mean_abs_diff": float(np.mean(np.abs(x[both] - y[both]))) if both.any() else None,
                         "null_new": float(np.mean(~np.isfinite(x))), "null_old": float(np.mean(~np.isfinite(y)))})
        else:
            ch = float((a.cast(pl.Utf8).fill_null("") != b.cast(pl.Utf8).fill_null("")).mean())
            rows.append({"col": c, "changed_share": round(ch, 4), "mean_abs_diff": None, "null_new": float(a.is_null().mean()), "null_old": float(b.is_null().mean())})
    diff = pd.DataFrame(rows).sort_values("changed_share", ascending=False)
    diff.to_csv(V4 / "diff_vs_fixed.csv", index=False)
    new_cols = [c for c in T.columns if c not in old.columns]
    (V4 / "build_report.json").write_text(json.dumps({"rows": T.height, "cols": T.width, "new_cols": new_cols, "changed_cols": int((diff["changed_share"] > 0).sum()),
                                                       "sire_unique": int(T["sire"].n_unique()), "dam_sire_unique": int(T["dam_sire"].n_unique())},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(diff[diff["changed_share"] > 0].head(40).to_string(), flush=True)
    print("新規列:", new_cols, flush=True)


def _base_cols() -> list[str]:
    adopted = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    return list(adopted["b0_features"]) + list(adopted["features"])


def _load(work: Path) -> pd.DataFrame:
    from src.pipeline.models import ablation as ab

    df = ab.load_table(work / "features/train_table.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    return df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)


def evaluate() -> None:
    import os

    from src.pipeline.models import feature_bakeoff as fb, zoo, zoo_runner
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    out = V4 / "impact"
    out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("TP07_THREADS", min(4, detect_compute(reserve_cores=6).threads)))
    seeds = [0, 1]
    base = _base_cols()
    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    d4 = _load(V4)
    missing = [c for c in base if c not in d4.columns]
    assert not missing, missing
    meta = d4[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(d4))
    fin_map = meta.set_index(KEY)["finish_position"]
    t0 = time.time()
    variants = {"V0_v4_base": d4[base], "V0_v4_base_no_course_type": d4[[c for c in base if c != "course_type"]]}
    res_all, oofs = {}, {}
    for name, X in variants.items():
        res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, seeds, threads, params=dict(lp))
        oofs[name], res_all[name] = oof, res
        print(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
    # 参照: 旧 fixed の V0 = 0.21731 / 0.33959 / 0.42705、血統のみ復旧 V2 = 0.21698 / 0.33934 / 0.42675（tp-22 / tp-24、同じ設定）
    ref = {"V0_fixed": [0.21731, 0.33959, 0.42705], "V2_ped_only": [0.21698, 0.33934, 0.42675]}
    cmp_ = {}
    base_name = "V0_v4_base_no_course_type"
    c = fb.compare_oof(oofs[base_name], oofs["V0_v4_base"], fin_map, "course_type（内外）を入れる効果")
    cmp_[c["variant"]] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
    print(c, flush=True)
    rep = {"levels": {k: {m: float(v[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")} for k, v in res_all.items()}, "reference": ref,
           "by_fold": {k: v[["fold", "logloss_win", "logloss_top2", "logloss_top3"]].round(5).to_dict("records") for k, v in res_all.items()},
           "compare": cmp_, "seeds": seeds, "rows": len(d4)}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("保存:", out / "report.json", flush=True)


def stage() -> None:
    """前段 S3 の年ごとの expanding OOF（tp-04 と同じ設定: 入力は base 175 列のみ、P5 走破タイム差・P6 正規化着順、seed 0、300 ラウンド）。"""
    import os

    from src.pipeline.models import stage_models as sm
    from src.pipeline.models.compute import detect_compute

    threads = int(os.environ.get("TP07_THREADS", min(4, detect_compute(reserve_cores=6).threads)))
    base = _base_cols()
    df = _load(V4)
    labels = pd.read_parquet(OLD / "oof/stage_labels.parquet")
    fit_params = json.loads((OLD / "features/fit_params.json").read_text(encoding="utf-8"))
    race_cls = sm.pace_class_labels(labels, fit_params["pace_q"], fit_params["rpci_q"])
    T = sm.build_targets(df, labels, race_cls)
    X = df[base]
    years = sorted(int(y) for y in df["year"].unique() if int(y) <= 2025)
    out = pd.DataFrame({"race_id": df["race_id"].to_numpy(), "horse_id": df["horse_id"].to_numpy()})
    t0 = time.time()
    for col, tname in (("s_time_pred", "t_time_dev"), ("s_rank_pred", "t_rank_norm")):
        pred = sm.oof_regression(df, X, T[tname].to_numpy(dtype=float), years, "expanding", threads)["base"]
        out[col] = pred.astype("float32")
        print(f"[{time.time() - t0:.0f}s] {col}: 非欠損 {np.isfinite(pred).mean():.3f}", flush=True)
    out["s_time_gap"] = (out["s_time_pred"] - out.groupby("race_id")["s_time_pred"].transform("min")).astype("float32")
    (V4 / "oof").mkdir(parents=True, exist_ok=True)
    out[["race_id", "horse_id", "s_time_pred", "s_time_gap", "s_rank_pred"]].to_parquet(V4 / "oof/stage_features_expanding.parquet", index=False)
    old = pd.read_parquet(OLD / "oof/stage_features_expanding.parquet")[["race_id", "horse_id", "s_time_pred", "s_time_gap", "s_rank_pred"]]
    m = out.merge(old, on=KEY, suffixes=("", "_old"))
    rep = {}
    for c in ("s_time_pred", "s_time_gap", "s_rank_pred"):
        ok = m[c].notna() & m[c + "_old"].notna()
        rep[c] = {"rows_both": int(ok.sum()), "corr": float(np.corrcoef(m.loc[ok, c], m.loc[ok, c + "_old"])[0, 1]),
                  "mean_diff": float((m.loc[ok, c] - m.loc[ok, c + "_old"]).mean()), "nonnull_new": float(m[c].notna().mean()), "nonnull_old": float(m[c + "_old"].notna().mean())}
    (V4 / "oof/stage_vs_old.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    {"build": build, "eval": evaluate, "stage": stage}[sys.argv[1]]()
