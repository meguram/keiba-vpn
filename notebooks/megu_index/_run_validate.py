#!/usr/bin/env python3
"""Post-run validation for megu_index notebook pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "output"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def validate_nb01() -> int:
    p = BASE / "nb01" / "megu_dataset.parquet"
    ps = BASE / "nb01" / "par_splits.parquet"
    if not p.exists():
        _fail(f"missing {p}")
    if not ps.exists():
        _fail(f"missing {ps}")

    df = pd.read_parquet(p)
    par = pd.read_parquet(ps)
    n = len(df)
    if n < 250_000:
        _fail(f"megu_dataset too small: {n}")
    if df["adjusted_time_sec"].notna().mean() < 0.99:
        _fail("adjusted_time_sec coverage < 99%")
    if df["finish_time_sec"].notna().mean() < 0.99:
        _fail("finish_time_sec coverage < 99%")
    if (df["finish_time_sec"] - df["adjusted_time_sec"]).abs().max() > 1e-6:
        _fail("finish_time_sec != adjusted_time_sec")
    if df["front_split_sec"].notna().mean() < 0.90:
        _fail("front_split_sec coverage < 90%")
    for col in ("race_t2nd_sec", "par_front_split_sec", "track_dev_sec", "tsi_raw", "class_group"):
        if col not in df.columns:
            _fail(f"megu_dataset missing column: {col}")
    if df["race_t2nd_sec"].notna().mean() < 0.99:
        _fail("race_t2nd_sec coverage < 99%")
    if df["par_front_split_sec"].notna().mean() < 0.90:
        _fail("par_front_split_sec coverage < 90%")
    tsi_chk = df.dropna(subset=["tsi_raw", "track_dev_sec"]).drop_duplicates(["date", "venue", "surface"])
    if (tsi_chk["tsi_raw"] + tsi_chk["track_dev_sec"]).abs().max() > 1e-6:
        _fail("tsi_raw != -track_dev_sec")
    has_legacy = "par_front_split_final" in par.columns
    has_continuous = {"par_intercept", "par_slope", "t2nd_ref"}.issubset(par.columns)
    if not has_legacy and not has_continuous:
        _fail("par_splits missing par_front_split_final or continuous coeffs")
    min_par_rows = 50 if has_legacy else 25
    if len(par) < min_par_rows:
        _fail(f"par_splits too few rows: {len(par)} (min={min_par_rows})")

    _ok(f"nb01 megu_dataset rows={n:,}, races={df['race_id'].nunique():,}, par_splits={len(par)}")
    return n


def validate_nb02(n01: int) -> None:
    cp = BASE / "nb02" / "coeff_pace.parquet"
    pt = BASE / "nb02" / "par_time_class.parquet"
    for path in (cp, pt):
        if not path.exists():
            _fail(f"missing {path}")

    coeff = pd.read_parquet(cp)
    par = pd.read_parquet(pt)
    if len(coeff) < 10:
        _fail(f"coeff_pace too few: {len(coeff)}")
    if not coeff["coeff_pace"].between(0.3, 1.5).all():
        bad = coeff.loc[~coeff["coeff_pace"].between(0.3, 1.5), "coeff_pace"]
        _fail(f"coeff_pace out of range: {bad.head().tolist()}")
    if len(par) < 50:
        _fail(f"par_time_class too few: {len(par)}")
    if "par_time_sec" not in par.columns:
        _fail("par_time_class missing par_time_sec")
    if par["par_time_sec"].notna().mean() < 0.95:
        _fail("par_time_sec coverage < 95%")
    cell_betas = par.drop_duplicates(["venue", "surface", "distance"])
    if (cell_betas["beta"] > 0).any():
        bad = cell_betas.loc[cell_betas["beta"] > 0, ["venue", "surface", "distance", "beta"]]
        _fail(f"par_time_class has beta>0 cells: {bad.head().to_dict()}")
    cell_betas = cell_betas.copy()
    cell_betas["diff_rank2_rank7"] = -5 * cell_betas["beta"]
    if cell_betas["diff_rank2_rank7"].min() < 1.5:
        _fail(f"par_time rank2-rank7 diff too small: min={cell_betas['diff_rank2_rank7'].min()}")

    _ok(f"nb02 coeff_pace={len(coeff)}, par_time_class={len(par)}")


def validate_nb03() -> None:
    p = BASE / "nb03" / "delta_track.parquet"
    if not p.exists():
        _fail(f"missing {p}")
    dt = pd.read_parquet(p)
    need = {"date", "venue", "surface", "delta_track_sec", "n_races", "is_fallback"}
    if not need.issubset(dt.columns):
        _fail(f"delta_track columns missing: {need - set(dt.columns)}")
    if len(dt) < 100:
        _fail(f"delta_track too few rows: {len(dt)}")
    fb = dt["is_fallback"].mean()
    if fb > 0.95:
        _fail(f"delta_track fallback rate too high: {fb:.1%}")

    _ok(f"nb03 delta_track rows={len(dt):,}, fallback={fb:.1%}")


def validate_nb04(n01: int) -> None:
    p = BASE / "nb04" / "megu_index.parquet"
    if not p.exists():
        _fail(f"missing {p}")
    mi = pd.read_parquet(p)
    if len(mi) != n01:
        _fail(f"megu_index rows {len(mi)} != nb01 {n01}")
    status = mi["computation_status"].value_counts(normalize=True)
    valid_rate = status.get("valid", 0)
    no_par_rate = status.get("no_par", 0)
    if valid_rate < 0.50:
        _fail(f"valid rate too low: {valid_rate:.1%} status={status.to_dict()}")
    if no_par_rate > 0.50:
        _fail(f"no_par rate too high: {no_par_rate:.1%}")
    valid = mi[mi["computation_status"] == "valid"]
    if valid["megu_index"].notna().mean() < 0.99:
        _fail("valid rows missing megu_index")
    med = valid["megu_index"].median()
    if not (30 <= med <= 70):
        _fail(f"megu_index median out of range: {med}")

    _ok(
        f"nb04 rows={len(mi):,}, valid={valid_rate:.1%}, no_par={no_par_rate:.1%}, median={med:.1f}"
    )


def validate_nb05(n01: int) -> None:
    p = BASE / "nb05" / "megu_final.parquet"
    if not p.exists():
        _fail(f"missing {p}")
    mf = pd.read_parquet(p)
    if len(mf) != n01:
        _fail(f"megu_final rows {len(mf)} != nb01 {n01}")
    for col in ("megu_a", "megu_b", "megu_c"):
        if col not in mf.columns:
            _fail(f"missing {col}")
    if not mf["n_valid_runs"].between(0, 5).all():
        _fail("n_valid_runs out of 0-5")
    cov = mf["megu_b"].notna().mean()
    if cov < 0.30:
        _fail(f"megu_b coverage too low: {cov:.1%}")

    _ok(f"nb05 rows={len(mf):,}, megu_b coverage={cov:.1%}")


def validate_nb06() -> None:
    report = BASE / "nb06" / "effectiveness_report.md"
    sp = BASE / "nb06" / "spearman_summary.parquet"
    for path in (report, sp):
        if not path.exists():
            _fail(f"missing {path}")
    ss = pd.read_parquet(sp)
    if "spearman_mean" not in ss.columns:
        _fail("spearman_summary missing spearman_mean")
    mi_rows = ss[ss["index_type"] == "megu_index"] if "index_type" in ss.columns else ss
    if len(mi_rows):
        sp_mean = float(mi_rows["spearman_mean"].iloc[0])
        if sp_mean >= 0:
            _fail(f"expected negative spearman for megu_index, got {sp_mean}")

    _ok(f"nb06 report exists, spearman rows={len(ss)}")


def main() -> None:
    nb = sys.argv[1] if len(sys.argv) > 1 else "all"
    n01 = 0
    if nb in ("nb01", "all"):
        n01 = validate_nb01()
    if nb in ("nb02", "all"):
        if not n01:
            n01 = len(pd.read_parquet(BASE / "nb01" / "megu_dataset.parquet"))
        validate_nb02(n01)
    if nb in ("nb03", "all"):
        validate_nb03()
    if nb in ("nb04", "all"):
        if not n01:
            n01 = len(pd.read_parquet(BASE / "nb01" / "megu_dataset.parquet"))
        validate_nb04(n01)
    if nb in ("nb05", "all"):
        if not n01:
            n01 = len(pd.read_parquet(BASE / "nb01" / "megu_dataset.parquet"))
        validate_nb05(n01)
    if nb in ("nb06", "all"):
        validate_nb06()
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
