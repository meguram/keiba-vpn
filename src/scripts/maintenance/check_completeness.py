#!/usr/bin/env python3
"""GCSデータ完全性チェック → 結果を data/meta/completeness_check.json に保存。

使い方::

    python scripts/maintenance/check_completeness.py
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.scraper.storage import HybridStorage  # noqa: E402

s = HybridStorage()
with s._blob_list_lock:
    s._blob_list_cache.clear()

JRA = frozenset(f'{i:02d}' for i in range(1, 11))
cats = [
    'race_result', 'race_shutuba', 'race_index', 'race_shutuba_past',
    'race_odds', 'race_paddock', 'race_oikiri', 'race_trainer_comment',
    'race_result_lap', 'smartrc_race', 'race_pair_odds', 'race_barometer',
]

results = {}
for year in ['2021', '2022', '2023', '2024', '2025']:
    base = set(k for k in s.batch_list_blobs('race_result', year)
               if len(k) >= 6 and k[4:6] in JRA)
    year_data = {"base_count": len(base), "categories": {}}
    for cat in cats:
        keys = set(k for k in s.batch_list_blobs(cat, year)
                   if len(k) >= 6 and k[4:6] in JRA)
        miss = sorted(base - keys)
        year_data["categories"][cat] = {
            "count": len(keys),
            "missing": len(miss),
            "samples": miss[:10],
        }
    results[year] = year_data

_OUT_DIR = os.path.join(_ROOT, "data", "meta")
os.makedirs(_OUT_DIR, exist_ok=True)
out_path = os.path.join(_OUT_DIR, "completeness_check.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print(f"Written to {out_path}")

for year, yd in sorted(results.items()):
    print(f"\n=== {year} (base: {yd['base_count']}) ===")
    for cat, cd in yd["categories"].items():
        if cd["missing"] > 0:
            print(f"  {cat:25s} {cd['count']:>5d}  MISS={cd['missing']:>4d}  ex: {cd['samples'][:3]}")
        else:
            print(f"  {cat:25s} {cd['count']:>5d}  OK")
