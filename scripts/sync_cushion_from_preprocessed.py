#!/usr/bin/env python3
"""
GCS: preprocessed/others/cushion_data → jra_cushion 年別 JSON ＋ローカル cushion_values.json

使い方:
  cd keiba-vpn && python3 scripts/sync_cushion_from_preprocessed.py --dry-run
  python3 scripts/sync_cushion_from_preprocessed.py --years 2026
  python3 scripts/sync_cushion_from_preprocessed.py --no-local-json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scraper.jra_cushion_sync import sync_preprocessed_to_jra_cushion  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=str, default="", help="カンマ区切り（空欄＝自動検出）")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-local-json", action="store_true", help="cushion_values.json を触らない")
    args = p.parse_args()

    years_list = None
    if args.years.strip():
        years_list = [int(x.strip()) for x in args.years.split(",") if x.strip().isdigit()]

    out = sync_preprocessed_to_jra_cushion(
        years=years_list,
        dry_run=args.dry_run,
        update_local_json=not args.no_local_json,
        base_dir=ROOT,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
