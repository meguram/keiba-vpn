"""履歴特徴量（127 列）を、確定した結果から毎日作って蓄積する（cron: feature-accumulate）。

- 当年分の特徴量を ``data/local/modeling/feature_store/horse_history/features_<年>.parquet`` に置く。
  特徴量は as-of（行の日付より前の履歴だけで決まる）なので、過去の行は後から結果を足しても変わらない。
  ただし結果の取り込みが遅れた日があると、その日より後の行は取り込み後の再計算で変わる。そのため当年分は毎回作り直す。
- 2020〜2025 年（学習に使う年）は tp-02 が作る。ここでは作らない（2026 年は学習に使わない方針のため、年別のファイルに分けてある）。
- 推論モードで保存した特徴量（``data/local/inference/features/<日付>/``）が、結果の確定後に学習時と同じ計算で作った値と
  どれだけ食い違うかを、日付ごとに ``skew/<日付>.json`` に記録する（本番と学習のずれの監視）。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .builder import build_with_state, feature_columns
from .common import TABLES_DIR, build_repair_map, load_runs
from .inference import load_tiers

logger = logging.getLogger("pipeline.features.horse_history.accumulate")

JST = timezone(timedelta(hours=9))
REPO_ROOT = Path(__file__).resolve().parents[4]
KEY = ["race_id", "horse_id"]
DEFAULT_LAYER_A = "data/local/modeling/training_pipeline/layer_a.parquet"
MISMATCH_TOL = 1e-6


def store_dir() -> Path:
    return Path(os.environ.get("KEIBA_FEATURE_STORE_DIR") or REPO_ROOT / "data" / "local" / "modeling" / "feature_store" / "horse_history")


def inference_dir() -> Path:
    return Path(os.environ.get("KEIBA_INFERENCE_DIR") or REPO_ROOT / "data" / "local" / "inference")


def _empty_repair_map() -> pl.DataFrame:
    return pl.DataFrame(schema={"horse_id": pl.Utf8, "run_date": pl.Date, "l3f_fix": pl.Float64, "weight_fix": pl.Float64, "src_date": pl.Date})


def build_year(year: int, tables_dir: str | Path | None = None, layer_a: str | Path | None = None):
    """``year`` 年の行の特徴量。履歴は 2020 年〜``year`` 年の確定結果。推論用の状態も一緒に返す。"""
    t0 = time.perf_counter()
    runs = load_runs(tables_dir, years=list(range(2020, year + 1)))
    la = Path(layer_a) if layer_a else REPO_ROOT / DEFAULT_LAYER_A
    rm = build_repair_map(la) if la.exists() else _empty_repair_map()
    F, P, S, elo = build_with_state(runs, rm)
    cols = sorted({c for v in feature_columns(F.columns).values() for c in v})
    out = F.filter(pl.col("date").dt.year() == year).select(KEY + ["date"] + cols)
    info = {"history_rows": runs.height, "history_last_date": str(runs["date"].max()), "build_sec": round(time.perf_counter() - t0, 1)}
    return out, cols, info, (F, P, S, elo)


def _changed_rows(old: pl.DataFrame, new: pl.DataFrame, cols: list[str]) -> int:
    """前回の行のうち、今回の計算で特徴量が変わったものの数（結果の取り込み遅れの影響の目安）。"""
    j = old.join(new, on=KEY, suffix="_n")
    if j.height == 0:
        return 0
    bad = np.zeros(j.height, dtype=bool)
    for c in cols:
        if c not in old.columns or f"{c}_n" not in j.columns:
            continue
        a, b = j[c].to_numpy().astype(float), j[f"{c}_n"].to_numpy().astype(float)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        bad |= np.where(np.isnan(d), np.inf, d) > MISMATCH_TOL
    return int(bad.sum())


def skew_report(F: pl.DataFrame, cols: list[str], day_dir: Path, tiers: dict[str, int]) -> dict:
    """推論モードで保存した 1 日分の特徴量を、確定結果から作った値と比べる。"""
    files = sorted(day_dir.glob("*.parquet"))
    inf = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed") if files else pl.DataFrame()
    if inf.height == 0:
        return {"status": "no_inference_rows"}
    ref = F.join(inf.select(KEY).unique(), on=KEY)
    if ref.height == 0:
        return {"status": "results_not_yet", "inference_rows": inf.height}
    j = ref.join(inf, on=KEY, suffix="_inf")
    only_inf = inf.join(F.select(KEY), on=KEY, how="anti").height
    by_tier: dict[str, dict[str, int]] = {"0": {}, "1": {}, "2": {}}
    for c in cols:
        if f"{c}_inf" not in j.columns:
            continue
        a, b = j[c].to_numpy().astype(float), j[f"{c}_inf"].to_numpy().astype(float)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        n = int((np.where(np.isnan(d), np.inf, d) > MISMATCH_TOL).sum())
        if n:
            by_tier[str(tiers.get(c, -1))][c] = n
    return {"status": "compared", "compared_rows": j.height, "inference_rows": inf.height, "only_in_inference": only_inf,
            "mismatch_by_tier": by_tier}


def accumulate(year: int | None = None, tables_dir: str | Path | None = None, layer_a: str | Path | None = None) -> dict:
    year = year or datetime.now(JST).year
    out_dir = store_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    F, cols, info, state = build_year(year, tables_dir, layer_a)
    path = out_dir / f"features_{year}.parquet"
    changed = None
    if path.exists():
        try:
            changed = _changed_rows(pl.read_parquet(path), F, cols)
        except Exception as exc:  # 前回のファイルが壊れていても蓄積は続ける
            logger.warning("前回のファイルとの比較に失敗: %s", exc)
    tmp = path.with_suffix(".tmp")
    F.write_parquet(tmp)
    tmp.replace(path)

    tiers = load_tiers()
    skew: dict[str, dict] = {}
    inf_root = inference_dir() / "features"
    if inf_root.exists():
        (out_dir / "skew").mkdir(exist_ok=True)
        for d in sorted(p for p in inf_root.iterdir() if p.is_dir() and p.name.startswith(str(year))):
            rep = skew_report(F, cols, d, tiers)
            skew[d.name] = {k: v for k, v in rep.items() if k != "mismatch_by_tier"} | (
                {"mismatch_by_tier": {t: sorted(m) for t, m in rep["mismatch_by_tier"].items()}} if "mismatch_by_tier" in rep else {})
            if rep.get("status") == "compared":
                (out_dir / "skew" / f"{d.name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    bundle_meta = None
    if os.environ.get("KEIBA_WRITE_SERVING_BUNDLE", "1").strip().lower() not in ("0", "false", "no"):
        from .inference import graded_race_names
        from .serving import default_bundle_dir, write_bundle

        Ffull, P, S, elo = state
        try:
            graded = graded_race_names(tables_dir, list(range(2020, year + 1)))
        except Exception as exc:  # 重賞名の表が作れなくても、バンドル自体は作る
            logger.warning("重賞名の表が作れなかった: %s", exc)
            graded = {}
        t1 = time.perf_counter()
        bundle_meta = write_bundle(default_bundle_dir(), Ffull.columns, P, S, elo, {"graded_names": graded})
        bundle_meta = {k: v for k, v in bundle_meta.items() if k != "columns"} | {"write_sec": round(time.perf_counter() - t1, 1)}

    manifest = {
        "year": year, "built_at": datetime.now(JST).isoformat(timespec="seconds"), "rows": F.height, "races": F["race_id"].n_unique(),
        "first_date": str(F["date"].min()) if F.height else None, "last_date": str(F["date"].max()) if F.height else None,
        "features": len(cols), "changed_rows_vs_previous": changed, **info,
        "serving_bundle": bundle_meta,
        "note": "当年分は毎回作り直す（結果の取り込み遅れで過去の行が変わりうる）。2026 年は学習に使わない。",
    }
    (out_dir / f"manifest_{year}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest["skew"] = {d: s.get("status") for d, s in skew.items()}
    return manifest
