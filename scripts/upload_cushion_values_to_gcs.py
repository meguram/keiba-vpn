#!/usr/bin/env python3
"""
data/jra_baba/cushion_values.json を年別に GCS へ保存する。
ロジックは scraper.jra_cushion_storage を参照。

使い方:
  cd keiba-vpn && python3 scripts/upload_cushion_values_to_gcs.py
  python3 scripts/upload_cushion_values_to_gcs.py --also-full
  python3 scripts/upload_cushion_values_to_gcs.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scraper.jra_cushion_storage import upload_cushion_values_to_gcs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--also-full", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = upload_cushion_values_to_gcs(
        json_path=args.json,
        also_full=args.also_full,
        dry_run=args.dry_run,
        base_dir=ROOT,
    )
    if not out.get("ok"):
        print("error:", out.get("error"), file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"dry-run: {out['total_rows']} 行 → {len(out['years'])} 年")
        print("prefix:", out.get("gcs_prefix"))
        for y in out["years"]:
            print(f"  {y}: {out['counts'][str(y)]} 行")
        if args.also_full:
            print("  + full.json")
        return 0
    for y in out["years"]:
        print(f"saved jra_cushion/{y}.json ({out['counts'][str(y)]} records)")
    if out.get("full_saved"):
        print(f"saved jra_cushion/full.json ({out['total_rows']} records)")
    print("prefix:", out.get("gcs_prefix"))
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
