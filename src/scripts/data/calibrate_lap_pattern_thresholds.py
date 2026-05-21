#!/usr/bin/env python3
"""ローカル race_result キャッシュからラップ型閾値を較正し JSON を出力する。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.utils.distance_band import distance_group_key, distance_m  # noqa: E402
from src.utils.lap_pattern import (  # noqa: E402
    compute_lap_metrics,
    parse_lap_times_sec,
    surface_group,
)


def load_races(cache_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(cache_dir.glob("race_result/**/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        laps = parse_lap_times_sec(data.get("lap_times"))
        if len(laps) < 4:
            continue
        m = compute_lap_metrics(laps)
        if not m:
            continue
        dist_m = distance_m(data.get("distance"), m["n_furlongs"])
        rows.append({
            "race_id": data.get("race_id"),
            "surface": data.get("surface"),
            "distance": dist_m,
            "surface_g": surface_group(data.get("surface")),
            "dist_g": distance_group_key(dist_m, m["n_furlongs"]),
            **m,
        })
    return rows


def main() -> None:
    cache = ROOT / "data" / "cache"
    rows = load_races(cache)
    print(f"races with laps: {len(rows)}")
    if not rows:
        return

    from collections import defaultdict
    import statistics as stats

    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[f"{r['surface_g']}|{r['dist_g']}"].append(r)

    out = {}
    for key, grp in sorted(by_key.items()):
        bursts = [r["burst_delta"] for r in grp if r.get("burst_delta") is not None]
        consumes = [r["consume_delta"] for r in grp if r.get("consume_delta") is not None]
        stds = [r["last5_std"] for r in grp if r.get("last5_std") is not None]
        ranges = [r["last5_range"] for r in grp if r.get("last5_range") is not None]
        baselines = [r["baseline_lap"] for r in grp]

        def pct(vals, p, default=0.0):
            if not vals:
                return default
            s = sorted(vals)
            i = min(len(s) - 1, max(0, int(len(s) * p)))
            return s[i]

        out[key] = {
            "n": len(grp),
            "baseline_lap_avg": round(stats.mean(baselines), 3) if baselines else None,
            "burst_p70": round(pct(bursts, 0.70), 3),
            "burst_p80": round(pct(bursts, 0.80), 3),
            "consume_p70": round(pct(consumes, 0.70), 3),
            "consume_p80": round(pct(consumes, 0.80), 3),
            "std_p30": round(pct(stds, 0.30), 3),
            "range_p30": round(pct(ranges, 0.30), 3),
        }

    out_path = ROOT / "data" / "local" / "meta" / "lap_pattern_thresholds_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
