"""取り直した調教の記録（手元 data/local/horse_training/）から、追い切り表（1 調教 1 行）を作る。ネットワーク・GCS は使わない。

python -m src.scripts.data.build_training_laps_past --horses-file data/local/meta/horse_training_past_targets.parquet --out OUT.parquet
対象は backfill_horse_training_past の進捗で『取得済み』の馬のうち、手元に記録があるもの。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.pipeline.features.training_laps import add_lap_features, record_to_rows
from src.scraper.horse_training_local import horse_training_json_path
from src.scripts.scraping.backfill_horse_training_past import PROGRESS, read_horses

ROOT = Path(__file__).resolve().parents[3]


def build(ids: list[str], base: Path) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    absent = 0
    for h in ids:
        f = horse_training_json_path(base, h)
        if not f.is_file():
            absent += 1
            continue
        rows += record_to_rows(json.loads(f.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame(), absent
    return add_lap_features(pd.DataFrame(rows)), absent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horses-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(ROOT))
    a = ap.parse_args()
    done = json.loads(PROGRESS.read_text(encoding="utf-8")).get("done", {})
    ids = [h for h in read_horses(Path(a.horses_file)) if done.get(h) in ("saved", "empty")]
    df, absent = build(ids, Path(a.base))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"馬 {len(ids)} / 手元に記録なし {absent} / 追い切り {len(df)} 本 -> {a.out}")


if __name__ == "__main__":
    main()
