"""ML warehouse の整合性チェック。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.scraper.export_tables import ALL_CATEGORIES, FLAT_CATEGORIES, ML_EXTRA_FLAT_CATEGORIES
from src.scripts.data.ml_warehouse.paths import (
    catalog_sqlite_path,
    horse_shards_dir,
    local_tables_year,
    manifest_path,
    warehouse_root,
)

REQUIRED_FLAT = set(FLAT_CATEGORIES) | set(ML_EXTRA_FLAT_CATEGORIES) | {"race_pair_odds"}


def _parquet_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    cols = pf.schema_arrow.names
    races = 0
    if "race_id" in cols:
        t = pq.read_table(path, columns=["race_id"])
        races = t.column("race_id").to_pandas().nunique()
    return {
        "exists": True,
        "rows": int(n),
        "unique_races": int(races),
        "columns": len(cols),
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }


def validate_year_parquets(base: Path, year: str) -> dict[str, Any]:
    td = local_tables_year(base, year)
    out: dict[str, Any] = {"year": year, "dir": str(td), "categories": {}, "missing": []}
    if not td.is_dir():
        out["error"] = "tables dir missing"
        return out
    for cat in sorted(REQUIRED_FLAT):
        p = td / f"{cat}_flat.parquet"
        st = _parquet_stats(p)
        out["categories"][cat] = st
        if not st.get("exists"):
            out["missing"].append(cat)
    rr = out["categories"].get("race_result", {})
    rs = out["categories"].get("race_shutuba", {})
    if rr.get("exists") and rs.get("exists"):
        out["race_id_gap"] = abs(rr.get("unique_races", 0) - rs.get("unique_races", 0))
    return out


def validate_symlinks(base: Path, years: list[str]) -> list[str]:
    wh = warehouse_root(base)
    issues: list[str] = []
    for y in years:
        flat_d = wh / "by_year" / y / "scraper_flat"
        if not flat_d.is_dir():
            issues.append(f"{y}: scraper_flat dir missing")
            continue
        for link in flat_d.iterdir():
            if link.is_symlink() and not link.resolve().exists():
                issues.append(f"{link}: broken symlink")
    return issues


def validate_sqlite_shards(base: Path) -> dict[str, Any]:
    cat_p = catalog_sqlite_path(base)
    out: dict[str, Any] = {
        "layout": "sharded_by_horse_id_prefix4",
        "catalog": str(cat_p),
        "catalog_exists": cat_p.exists(),
        "shards_dir": str(horse_shards_dir(base)),
        "shard_files": 0,
        "totals": {"profiles": 0, "history": 0, "training": 0, "bytes": 0},
        "registry": [],
    }
    sd = horse_shards_dir(base)
    if sd.is_dir():
        out["shard_files"] = len(list(sd.glob("*.sqlite3")))

    if not cat_p.exists():
        return out

    con = sqlite3.connect(str(cat_p))
    try:
        rows = con.execute(
            """SELECT shard, profile_count, history_count, training_count, file_bytes
               FROM shard_registry ORDER BY shard"""
        ).fetchall()
        for r in rows:
            out["registry"].append({
                "shard": r[0],
                "profiles": r[1],
                "history": r[2],
                "training": r[3],
                "mb": round((r[4] or 0) / 1e6, 2),
            })
            out["totals"]["profiles"] += r[1] or 0
            out["totals"]["history"] += r[2] or 0
            out["totals"]["training"] += r[3] or 0
            out["totals"]["bytes"] += r[4] or 0
        n_lookup = con.execute("SELECT COUNT(*) FROM horse_lookup").fetchone()[0]
        out["horse_lookup_count"] = n_lookup
    finally:
        con.close()
    return out


def run_full_report(base_dir: str | Path, years: list[str]) -> dict[str, Any]:
    base = Path(base_dir)
    return {
        "years": {y: validate_year_parquets(base, y) for y in years},
        "symlink_issues": validate_symlinks(base, years),
        "sqlite": validate_sqlite_shards(base),
        "export_categories_expected": list(ALL_CATEGORIES),
        "manifest": json.loads(manifest_path(base).read_text(encoding="utf-8"))
        if manifest_path(base).exists() else None,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 60)
    print("ML warehouse validation")
    print("=" * 60)
    for y, yr in report.get("years", {}).items():
        miss = yr.get("missing", [])
        rr = yr.get("categories", {}).get("race_result", {})
        print(
            f"\n[{y}] missing={len(miss)} "
            f"race_result races={rr.get('unique_races', '?')}"
        )
    sq = report.get("sqlite", {})
    print(f"\n[sqlite] {sq.get('layout')} shards={sq.get('shard_files')}")
    print(f"  totals: {sq.get('totals')}")
    for s in sq.get("registry", [])[:8]:
        print(f"    {s['shard']}: prof={s['profiles']} train={s['training']} {s['mb']}MB")
    if len(sq.get("registry", [])) > 8:
        print(f"    ... +{len(sq['registry']) - 8} shards")
    print("=" * 60)
