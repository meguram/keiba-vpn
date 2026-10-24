"""ベースライン構築 CLI (改修後 2020-2025)"""

from __future__ import annotations

import argparse
import logging

from src.research.race.track_speed_engine import TrackSpeedEngine
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)


def main() -> None:
    script_basic_config()
    parser = argparse.ArgumentParser(description="馬場速度ベースライン構築")
    parser.add_argument("--years", nargs="*", default=["2020", "2021", "2022", "2023", "2024", "2025"])
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    eng = TrackSpeedEngine(args.base_dir)

    def log(msg: str) -> None:
        logger.info(msg)
        print(msg)

    bl = eng.build_baselines(years=args.years, progress_cb=log)
    log(f"完了: {len(bl)} グループ")

    # 全会場フォールバック行を追加
    import statistics
    from collections import defaultdict

    from src.research.race.track_speed_engine import (
        COND_POOLS,
        MIN_BASELINE_N,
        BASELINES_PATH,
        _min_std,
    )
    import pandas as pd

    races = eng.load_races_from_parquet(args.years)
    all_groups: dict[str, list[float]] = defaultdict(list)
    for _, r in races.iterrows():
        tc = str(r["track_condition"] or "").strip()
        for pool_name, cond_set in COND_POOLS.items():
            if tc not in cond_set:
                continue
            key = f"ALL|-|{r['surface']}|{int(r['distance'])}|{r['class_band']}|{pool_name}"
            all_groups[key].append(float(r.get("time_2nd_adj") or r["time_2nd"]))

    extra = []
    for key, times in all_groups.items():
        if len(times) < MIN_BASELINE_N:
            continue
        parts = key.split("|")
        extra.append({
            "key": key,
            "venue": "ALL",
            "layout": "-",
            "surface": parts[2],
            "distance": int(parts[3]),
            "class_band": parts[4],
            "cond_pool": parts[5],
            "mean": round(statistics.mean(times), 3),
            "std": round(max(statistics.stdev(times) if len(times) > 1 else 1.0, _min_std(int(parts[3]))), 3),
            "n": len(times),
        })

    if extra:
        df = pd.concat([bl, pd.DataFrame(extra)], ignore_index=True).drop_duplicates("key")
        df.to_parquet(BASELINES_PATH, index=False)
        log(f"ALL会場フォールバック追加: +{len(extra)} → 計 {len(df)}")


if __name__ == "__main__":
    main()
