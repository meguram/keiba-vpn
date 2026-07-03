#!/usr/bin/env python3
"""Populate ``data/calculated_data/`` from page_reference / research (symlinks).

API は ``src.config.data_paths`` 経由で calculated_data を優先参照する。
大容量ディレクトリはコピーせずシンボリックリンクで共有する。

Usage:
  python3 -m src.scripts.data.sync_calculated_data_bundle
  python3 -m src.scripts.data.sync_calculated_data_bundle --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from src.config.data_paths import ROOT, calculated_data_root, page_reference_root

logger = logging.getLogger(__name__)

# relative_path_in_calculated -> relative_path_from_data/
LINKS: dict[str, str] = {
    "knowledge": "page_reference/knowledge",
    "note_aptitude_race": "page_reference/note_aptitude_race",
    "note_aptitude_race_l3": "page_reference/note_aptitude_race_l3",
    "pedigree_race_index": "page_reference/pedigree_race_index",
    "pedigree_map": "page_reference/pedigree_map",
    "bloodline_vector": "page_reference/bloodline_vector",
    "tables": "page_reference/tables",
    "race_lists": "page_reference/race_lists",
    "track_speed": "page_reference/track_speed",
    "race_performance": "page_reference/race_performance",
    "cushion": "page_reference/cushion",
    "meta": "page_reference/meta",
    "bloodline": "research/bloodline",
    "course_bloodline": "research/course_bloodline",
}


def _ensure_link(calc_root: Path, name: str, target_rel: str, *, dry_run: bool) -> str:
    data = ROOT / "data"
    target = data / target_rel
    link = calc_root / name
    if not target.exists():
        return f"skip (missing target): {name} -> {target_rel}"
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return f"ok (exists): {name}"
        if dry_run:
            return f"would replace symlink: {name}"
        link.unlink()
    elif link.exists():
        if dry_run:
            return f"would skip (not symlink, exists): {name}"
        return f"skip (not symlink): {name}"
    if dry_run:
        return f"would link: {name} -> ../{target_rel}"
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.join("..", target_rel), link)
    return f"linked: {name} -> ../{target_rel}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    calc = calculated_data_root()
    calc.mkdir(parents=True, exist_ok=True)
    logger.info("calculated_data=%s", calc)
    logger.info("page_reference=%s", page_reference_root())

    for name, rel in LINKS.items():
        logger.info("  %s", _ensure_link(calc, name, rel, dry_run=args.dry_run))

    pred_dir = calc / "predictions"
    pred_file = pred_dir / "predictions.json"
    legacy = ROOT / "data" / "processed" / "predictions.json"
    if legacy.is_file() and not pred_file.exists():
        if args.dry_run:
            logger.info("  would copy predictions.json from processed/")
        else:
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_file.write_bytes(legacy.read_bytes())
            logger.info("  copied predictions.json")
    elif not pred_file.exists():
        if args.dry_run:
            logger.info("  would create empty predictions stub")
        else:
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_file.write_text(
                '{"version":"1","races":[],"generated_at":null}\n',
                encoding="utf-8",
            )
            logger.info("  created empty predictions stub")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
