#!/usr/bin/env python3
"""
page_reference ポータブルバンドルへ、data/local に残っている UI 向けデータを移す。

移行対象:
  - race_lists/          → data/page_reference/race_lists/
  - meta/person/         → data/page_reference/meta/person/
  - meta/modeling/       → data/page_reference/meta/modeling/

移行後、旧パスにはシンボリックリンクを張り（既存スクリプト互換）、
新規書き込みは HybridStorage 経由で page_reference のみに保存される。

使い方:
  python3 -m src.scripts.data.migrate_page_reference_bundle --dry-run
  python3 -m src.scripts.data.migrate_page_reference_bundle
"""
from __future__ import annotations

import argparse
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.config.data_paths import (  # noqa: E402
    LEGACY_MODELING_META_DIR,
    LEGACY_PERSON_DIR,
    LEGACY_RACE_LISTS_DIR,
    MODELING_META_DIR,
    PAGE_REF,
    PERSON_DIR,
    RACE_LISTS_DIR,
)
logger = logging.getLogger(__name__)

MOVES: list[tuple[Path, Path, str]] = [
    (LEGACY_RACE_LISTS_DIR, RACE_LISTS_DIR, "race_lists"),
    (LEGACY_PERSON_DIR, PERSON_DIR, "meta/person"),
    (LEGACY_MODELING_META_DIR, MODELING_META_DIR, "meta/modeling"),
]


def _merge_tree(src: Path, dst: Path, *, dry_run: bool) -> tuple[int, int]:
    """src 配下のファイルを dst にコピー（既存は上書きしない）。"""
    copied = 0
    skipped = 0
    if not src.is_dir():
        return copied, skipped
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        target = dst / rel
        if target.exists():
            skipped += 1
            continue
        if dry_run:
            logger.info("[dry-run] copy %s -> %s", f, target)
            copied += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
    return copied, skipped


def _symlink_or_skip(link: Path, target: Path, *, dry_run: bool) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        logger.info("symlink 既存: %s", link)
        return
    if dry_run:
        logger.info("[dry-run] symlink %s -> %s", link, target)
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        backup = link.with_name(link.name + ".pre_page_ref_bak")
        if backup.exists():
            shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
        link.rename(backup)
        logger.info("旧ディレクトリを退避: %s", backup)
    link.symlink_to(target.resolve(), target_is_directory=True)
    logger.info("symlink 作成: %s -> %s", link, target)


def run(*, dry_run: bool) -> int:
    PAGE_REF.mkdir(parents=True, exist_ok=True)
    total_copied = 0
    total_skipped = 0

    for src, dst, label in MOVES:
        if not src.exists():
            logger.info("移行元なし (%s): %s", label, src)
            continue
        logger.info("=== %s: %s -> %s ===", label, src, dst)
        c, s = _merge_tree(src, dst, dry_run=dry_run)
        total_copied += c
        total_skipped += s
        logger.info("%s: copied=%d skipped(existing)=%d", label, c, s)
        if not dry_run and src.is_dir() and not src.is_symlink():
            _symlink_or_skip(src, dst, dry_run=False)

    logger.info(
        "完了 copied=%d skipped=%d dry_run=%s page_reference=%s",
        total_copied,
        total_skipped,
        dry_run,
        PAGE_REF,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
