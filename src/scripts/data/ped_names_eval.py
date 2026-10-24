"""tp-24: 学習表の sire / dam_sire が空だった問題の復旧（父・母父を horse_id で付与）と、前後の比較。

python -m src.scripts.data.ped_names_eval build   # 血統キー表と、血統入りの母表・学習表を training_pipeline_ped/ に作る（既存は上書きしない）
python -m src.scripts.data.ped_names_eval eval    # LightGBM 固定・valid 2022〜2024 expanding・2 seed。test 2025 は使わない

比較:
  V0 修正後の表（血統が空）の BASE 175 列            V2 血統入りの表の BASE 175 列
  V1 V0 + ped_* 24 列（血統が空のまま = v3 の状態）   V3 V2 + ped_* 24 列（実際の血統キー）
差: V2−V0（父系の累計勝率・馬場傾向との交互作用 6 列）、V3−V1（ped_* も含めた血統の復旧）、V3−V2（ped_* の上乗せ）
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
SIRE_COLS = ["sire", "dam_sire", "sire_prior_win_rate_surface", "dam_sire_prior_win_rate_surface", "sire_x_track_bias_diff", "dam_sire_x_track_bias_diff",
             "sire_x_track_bias_weighted", "dam_sire_x_track_bias_weighted"]


def build() -> None:
    import polars as pl

    from src.pipeline.features.pedigree_keys import build_parent_table
    from src.pipeline.models.layer_a_dataset import build_layer_a_dataframe
    from src.pipeline.models.memory_utils import downcast_frame
    from src.pipeline.models.time_split import TRAINING_YEARS

    t0 = time.time()
    (PED / "features").mkdir(parents=True, exist_ok=True)
    keys_path = PED / "pedigree_keys.parquet"
    if keys_path.exists():
        keys = pd.read_parquet(keys_path)
    else:
        ids = set()
        for y in range(2020, 2027):
            ids |= set(pl.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_shutuba_flat.parquet", columns=["horse_id"])["horse_id"].to_list())
        keys = build_parent_table(ids)
        keys.to_parquet(keys_path, index=False)
    print(f"[{time.time() - t0:.0f}s] 血統キー {len(keys)} 頭 / 父あり {(keys['sire_id'] != '').mean():.4f} / 母父あり {(keys['dam_sire_id'] != '').mean():.4f}", flush=True)

    full = build_layer_a_dataframe(years=list(TRAINING_YEARS), jra_only=True, pedigree_keys=keys)
    full = downcast_frame(full, keep=["race_id", "horse_id", "horse_number", "date"])
    full.to_parquet(PED / "layer_a.parquet", index=False)
    print(f"[{time.time() - t0:.0f}s] 母表 {full.shape}", flush=True)

    # 履歴系の列（horse_history）は血統を使わないので、修正後の学習表から引き継ぎ、血統に関わる 8 列だけ差し替える
    fx = pl.read_parquet(FIXED / "features/train_table.parquet")
    la = pl.from_pandas(full[["race_id", "horse_id"] + SIRE_COLS])
    key = ["race_id", "horse_id"]
    assert fx.select(key).n_unique() == fx.height
    T = fx.drop(SIRE_COLS).join(la.with_columns(pl.col(pl.Float64).cast(pl.Float32)), on=key, how="left")
    assert T.height == fx.height, (T.height, fx.height)
    # 血統以外の LayerA 列が修正後の表と同じであることの確認
    other = [c for c in pl.read_parquet(FIXED / "layer_a.parquet", n_rows=1).columns if c not in SIRE_COLS and c in full.columns and c not in key]
    chk = pl.read_parquet(FIXED / "layer_a.parquet", columns=key + other).join(pl.from_pandas(full[key + other]), on=key, suffix="_p")
    diff = {}
    for c in other:
        a, b = chk[c], chk[c + "_p"]
        if a.dtype.is_numeric():
            d = int(((a.cast(pl.Float64) - b.cast(pl.Float64)).abs() > 1e-6).sum())
        else:
            d = int((a.cast(pl.Utf8) != b.cast(pl.Utf8)).sum())
        if d:
            diff[c] = d
    print("血統以外の LayerA 列で修正後の表と違う列:", diff or "なし", flush=True)
    T.write_parquet(PED / "features/train_table.parquet")
    for f in ("feature_families.json",):
        (PED / "features" / f).write_text((FIXED / "features" / f).read_text(encoding="utf-8"), encoding="utf-8")
    (PED / "build_report.json").write_text(json.dumps({"rows": T.height, "keys": len(keys), "non_pedigree_diff_cols": diff,
                                                       "sire_unique": int(T["sire"].n_unique()), "dam_sire_unique": int(T["dam_sire"].n_unique())}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{time.time() - t0:.0f}s] 学習表 {T.shape} / 父 {T['sire'].n_unique()} 種 / 母父 {T['dam_sire'].n_unique()} 種", flush=True)


def evaluate() -> None:
    import os

    from src.pipeline.features import pedigree_context
    from src.pipeline.models import ablation as ab, feature_bakeoff as fb, zoo, zoo_runner
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    out = PED / "impact"
    out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("TP24_THREADS", detect_compute(reserve_cores=6).threads))
    seeds = [0, 1]
    adopted = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])

    def load(work: Path) -> pd.DataFrame:
        df = ab.load_table(work / "features/train_table.parquet")
        df["race_id"] = df["race_id"].astype(str)
        df["horse_id"] = df["horse_id"].astype(str)
        return df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)

    dff, dfp = load(FIXED), load(PED)
    key = ["race_id", "horse_id"]
    assert (dff[key].to_numpy() == dfp[key].to_numpy()).all()
    labels = pd.read_parquet(OLD / "oof/stage_labels.parquet")
    pf = pedigree_context.build(dff, labels)
    pp = pedigree_context.build(dfp, labels)
    print("ped_* 24 列: 血統が空の版の一意値数（中央値）", int(pf.nunique().median()), "/ 血統あり", int(pp.nunique().median()), flush=True)
    meta = dff[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(dff))
    fin_map = meta.set_index(key)["finish_position"]

    variants = {
        "V0_fixed_base": dff[base],
        "V1_fixed_base+ped_blank": pd.concat([dff[base], pf], axis=1),
        "V2_ped_base": dfp[base],
        "V3_ped_base+ped": pd.concat([dfp[base], pp], axis=1),
    }
    oofs, res_all = {}, {}
    t0 = time.time()
    for name, X in variants.items():
        res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, seeds, threads, params=dict(lp))
        oofs[name], res_all[name] = oof, res
        print(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
    pairs = [("V2_ped_base", "V0_fixed_base", "父系6列の復旧（V2−V0）"), ("V3_ped_base+ped", "V1_fixed_base+ped_blank", "ped_* も含む血統の復旧（V3−V1）"),
             ("V3_ped_base+ped", "V2_ped_base", "ped_* 24列の上乗せ（V3−V2）"), ("V3_ped_base+ped", "V0_fixed_base", "全部（V3−V0）")]
    cmp_ = {}
    for new, basev, label in pairs:
        c = fb.compare_oof(oofs[basev], oofs[new], fin_map, label)
        cmp_[label] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
        print(label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
    rep = {"levels": {k: {m: float(v[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")} for k, v in res_all.items()},
           "by_fold": {k: v[["fold", "logloss_win", "logloss_top2", "logloss_top3"]].round(5).to_dict("records") for k, v in res_all.items()},
           "compare": cmp_, "seeds": seeds, "rows": len(dff)}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("保存:", out / "report.json", flush=True)


if __name__ == "__main__":
    {"build": build, "eval": evaluate}[sys.argv[1]]()
