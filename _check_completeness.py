#!/usr/bin/env python3
"""GCSデータ完全性チェック → 結果をJSONに保存"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from scraper.storage import HybridStorage

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

out_path = os.path.join(os.path.dirname(__file__), "_completeness_check.json")
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
