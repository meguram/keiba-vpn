from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.scripts.data.ml_warehouse.paths import (
    by_year_dir,
    local_tables_year,
    manifest_path,
    warehouse_root,
)

logger = logging.getLogger(__name__)


def _relpath_from_to(src_file: Path, dst_target: Path) -> str:
    return os.path.relpath(dst_target.resolve(), start=src_file.parent.resolve())


def ensure_symlink(link_path: Path, target_path: Path, *, force: bool = False) -> str:
    """link_path → target_path の相対シンボリックリンク。戻り値: ok | exists | skip."""
    if not target_path.exists():
        return "missing_target"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel = _relpath_from_to(link_path, target_path)
    if link_path.is_symlink() or link_path.exists():
        if force:
            link_path.unlink(missing_ok=True)
        else:
            return "exists"
    link_path.symlink_to(rel, target_is_directory=False)
    return "ok"


def materialize_layout(
    base_dir: str | Path,
    years: list[str],
    *,
    force_symlinks: bool = False,
) -> dict[str, Any]:
    """by_year 以下に local tables / 主要 parquet への symlink を張り manifest 用 dict を返す。"""
    root = Path(base_dir)
    wh = warehouse_root(root)
    wh.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "layout_version": 1,
        "warehouse_root": str(wh),
        "years": years,
        "by_year": {},
        "reference_parquets": [],
    }

    for y in years:
        src_dir = local_tables_year(root, y)
        dst_base = by_year_dir(root, y)
        flat_d = dst_base / "scraper_flat"
        race_d = dst_base / "scraper_race"
        year_entry: dict[str, Any] = {"scraper_flat": {}, "scraper_race": {}}

        if not src_dir.is_dir():
            manifest["by_year"][y] = year_entry
            continue

        for p in sorted(src_dir.glob("*.parquet")):
            if "_chunks_" in p.name or p.name.startswith("."):
                continue
            name = p.name
            if name.endswith("_flat.parquet"):
                sub = flat_d / name
                st = ensure_symlink(sub, p, force=force_symlinks)
                year_entry["scraper_flat"][name] = {
                    "link": str(sub.relative_to(wh)),
                    "target": str(p),
                    "status": st,
                }
            elif name.endswith("_race.parquet"):
                sub = race_d / name
                st = ensure_symlink(sub, p, force=force_symlinks)
                year_entry["scraper_race"][name] = {
                    "link": str(sub.relative_to(wh)),
                    "target": str(p),
                    "status": st,
                }

        manifest["by_year"][y] = year_entry

    ref_dir = wh / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    ref_candidates = [
        ("research_pedigree_race_index_race_result_slim",
         root / "data" / "research" / "pedigree_race_index" / "race_result_slim.parquet"),
        ("research_pedigree_race_index_horse_pedigree_cats",
         root / "data" / "research" / "pedigree_race_index" / "horse_pedigree_cats.parquet"),
        ("knowledge_track_speed_baselines",
         root / "data" / "knowledge" / "track_speed_baselines.parquet"),
        ("knowledge_track_speed_pace_baselines",
         root / "data" / "knowledge" / "track_speed_pace_baselines.parquet"),
    ]
    ts_dir = root / "data" / "analysis" / "track_speed"
    if ts_dir.is_dir():
        tsd = ref_dir / "track_speed_races"
        tsd.mkdir(parents=True, exist_ok=True)
        ts_links = []
        for p in sorted(ts_dir.glob("races_*.parquet")):
            link = tsd / p.name
            st = ensure_symlink(link, p, force=force_symlinks)
            ts_links.append({"name": p.name, "status": st, "target": str(p)})
        manifest["reference_parquets_track_speed"] = ts_links

    for label, target in ref_candidates:
        if not target.exists():
            manifest["reference_parquets"].append({
                "label": label, "status": "missing", "path": str(target)})
            continue
        link = ref_dir / f"{label}.parquet"
        st = ensure_symlink(link, target, force=force_symlinks)
        manifest["reference_parquets"].append({
            "label": label,
            "link": str(link.relative_to(wh)),
            "target": str(target),
            "status": st,
        })

    outp = manifest_path(root)
    outp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("manifest 保存: %s", outp)
    return manifest
