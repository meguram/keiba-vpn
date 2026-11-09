"""学習の土台の表（LayerA 母表 + 馬の履歴の特徴量）を 2016〜2025 年で作り直す（再学習 2026-11、ユーザー指示「あるデータはすべて使う」）。

    python -m src.scripts.data.rebuild_base_2016 [--out /home/jovyan/work/retrain_out/base_2016] [--years 2016-2025]

- 本番の表（data/local/modeling/training_pipeline_v4 など）は書き換えない。出力は --out。
- 学習対象年の定数（TRAINING_YEARS = 2020〜2025）をこの実行の中だけ差し替える。
- 馬の履歴は 2016 年からの全走で作るので、2020〜2021 年の行も「2019 年以前の走」が入った値に変わる（今までは 2020 年より前の走が無かった）。
- 通過順・上がりの壊れたレースの補い（race_result_on_time）は、既存の取得済みの控え（training_pipeline/features/on_time_repair.parquet）だけを使い、
  GCS は読まない（ホストから GCS を繰り返し読まない決まり）。控えに無い 2016〜2019 年のレースは補わない。
- 2016〜2018 年は指数（race_index）が取得元に無く、馬柱（race_shutuba_past）は取得中（backfill_past）。無い列は欠損のまま。
出力: <out>/layer_a.parquet、<out>/features/{train_table.parquet, feature_families.json}、<out>/build_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
PED = ROOT / "data/local/modeling/training_pipeline_ped"
ON_TIME_CACHE = ROOT / "data/local/modeling/training_pipeline/features/on_time_repair.parquet"
KEY = ["race_id", "horse_id"]


def _years(s: str) -> list[int]:
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def _patch_years(years: list[int]) -> None:
    from src.pipeline.features.horse_history import common
    from src.pipeline.models import time_split, training_years

    ty = tuple(str(y) for y in years)
    training_years.TRAINING_YEARS = ty
    time_split.TRAINING_YEARS = ty
    common.YEARS = list(years)
    # キーワード専用の既定引数に定義時の TRAINING_YEARS が入っている（restrict_to_training_years）ので、それも差し替える
    kw = dict(time_split.restrict_to_training_years.__kwdefaults__ or {})
    kw["years"] = ty
    time_split.restrict_to_training_years.__kwdefaults__ = kw
    # 親モデルなどの年ごとの fold（yearly_expanding_folds）の学習の始まりも、既定の 2020 年から最初の年に
    kf = dict(time_split.yearly_expanding_folds.__kwdefaults__ or {})
    kf["first_train_year"] = ty[0]
    time_split.yearly_expanding_folds.__kwdefaults__ = kf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016")
    ap.add_argument("--years", default="2016-2025")
    ap.add_argument("--skip-layer-a", action="store_true", help="既にある <out>/layer_a.parquet を使う")
    a = ap.parse_args()
    years = _years(a.years)
    assert 2026 not in years, "2026 年は学習に使わない"
    out = Path(a.out)
    (out / "features").mkdir(parents=True, exist_ok=True)
    _patch_years(years)
    t0 = time.time()

    from src.pipeline.features.course_variant import add_course_variant
    from src.pipeline.features.horse_history.builder import build_from_runs, feature_columns
    from src.pipeline.features.horse_history.common import build_repair_map, load_runs
    from src.pipeline.models.layer_a_dataset import build_layer_a_dataframe
    from src.pipeline.models.memory_utils import downcast_frame

    la_path = out / "layer_a.parquet"
    if not a.skip_layer_a:
        keys = pd.read_parquet(PED / "pedigree_keys.parquet")
        full = build_layer_a_dataframe(years=[str(y) for y in years], jra_only=True, pedigree_keys=keys)
        ct = add_course_variant(full, parsed_col="course_type", parsed_available=False)
        full["course_type_unknown"] = ct.isna().astype(np.int8)
        full["course_type"] = ct.fillna("不明")
        full = downcast_frame(full, keep=["race_id", "horse_id", "horse_number", "date"])
        full.to_parquet(la_path, index=False)
        yc = pd.to_datetime(full["date"]).dt.year.value_counts().sort_index().to_dict()
        print(f"[{time.time() - t0:.0f}s] 母表 {full.shape} 年別 {yc}", flush=True)
        del full
        gc.collect()

    runs = load_runs(years=years)
    on_time = pl.read_parquet(ON_TIME_CACHE) if ON_TIME_CACHE.exists() else None
    F, _P = build_from_runs(runs, build_repair_map(la_path), on_time)
    fam = feature_columns(F.columns)
    fam_cols = sorted({c for v in fam.values() for c in v})
    print(f"[{time.time() - t0:.0f}s] 履歴 {F.shape} 群 {len(fam)} 列 {len(fam_cols)}", flush=True)
    drop = [c for c in ("past_races", "training") if c in pl.read_parquet_schema(la_path)]
    la = pl.read_parquet(la_path).drop(drop).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    T = la.join(F.select(KEY + fam_cols).with_columns(pl.col(pl.Float64).cast(pl.Float32)), on=KEY, how="left")
    assert T.height == la.height
    T.write_parquet(out / "features/train_table.parquet")
    (out / "features/feature_families.json").write_text(json.dumps(fam, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{time.time() - t0:.0f}s] 学習表 {T.shape}", flush=True)

    # 2020〜2025 年の行で、今の V4 の表と同じ列を比べる（行の一致・相関。履歴の列は 2019 年以前の走が入るので変わってよい）
    old = pl.read_parquet(V4 / "features/train_table.parquet")
    common_cols = [c for c in T.columns if c in old.columns and c not in KEY]
    j = T.select(KEY + common_cols).join(old.select(KEY + common_cols), on=KEY, how="inner", suffix="_old")
    rows = []
    for c in common_cols:
        x, y = j[c], j[c + "_old"]
        if x.dtype.is_numeric() and y.dtype.is_numeric():
            xa, ya = x.cast(pl.Float64).to_numpy(), y.cast(pl.Float64).to_numpy()
            both = np.isfinite(xa) & np.isfinite(ya)
            same = np.isclose(xa, ya, rtol=1e-5, atol=1e-7) | (np.isnan(xa) & np.isnan(ya))
            corr = float(np.corrcoef(xa[both], ya[both])[0, 1]) if both.sum() > 10 and xa[both].std() > 0 and ya[both].std() > 0 else None
            rows.append({"col": c, "changed_share": round(float(1 - same.mean()), 4), "corr": corr,
                         "null_new": round(float(np.mean(~np.isfinite(xa))), 4), "null_old": round(float(np.mean(~np.isfinite(ya))), 4)})
    diff = pd.DataFrame(rows).sort_values("changed_share", ascending=False)
    diff.to_csv(out / "diff_vs_v4_2020_2025.csv", index=False)
    yr = T["date"].cast(pl.Utf8).str.slice(0, 4)
    null_by_year = {}
    for c in [c for c in T.columns if c not in KEY][:]:
        if T[c].dtype.is_numeric():
            s = T.select(yr.alias("y"), pl.col(c).is_null().alias("n")).group_by("y").agg(pl.col("n").mean()).sort("y")
            null_by_year[c] = dict(zip(s["y"].to_list(), [round(v, 3) for v in s["n"].to_list()]))
    rep = {"rows": T.height, "cols": T.width, "rows_by_year": dict(zip(*np.unique(yr.to_numpy(), return_counts=True))),
           "rows_matched_v4": j.height, "rows_v4": old.height, "cols_common": len(common_cols),
           "cols_new_only": [c for c in T.columns if c not in old.columns], "cols_v4_only": [c for c in old.columns if c not in T.columns],
           "changed_cols_2020_2025": int((diff["changed_share"] > 0.001).sum())}
    rep["rows_by_year"] = {k: int(v) for k, v in rep["rows_by_year"].items()}
    (out / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "null_by_year.json").write_text(json.dumps(null_by_year, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)}, ensure_ascii=False), flush=True)
    print(diff.head(30).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
