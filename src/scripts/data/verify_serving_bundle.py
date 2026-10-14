#!/usr/bin/env python3
"""サービング用バンドルの検証（実データ）。3 段階に分け、小さいマシンの段階（serve）のメモリを単独で測る。

  python -m src.scripts.data.verify_serving_bundle build   --day 2024-07-06 --bundle /tmp/sb     # 当日より前の履歴でバンドルを作る（大きいメモリ）
  python -m src.scripts.data.verify_serving_bundle serve   --day 2024-07-06 --bundle /tmp/sb --out /tmp/sb_out.parquet   # 出走馬だけ読む（ピーク RSS を表示）
  python -m src.scripts.data.verify_serving_bundle compare --day 2024-07-06 --out /tmp/sb_out.parquet   # 全履歴の推論モード・学習時の値と突き合わせる
"""

from __future__ import annotations

import argparse
import resource
import time
from datetime import date


def _rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _card(day: str):
    import polars as pl

    from src.pipeline.features.horse_history.common import TABLES_DIR
    from src.pipeline.features.horse_history.inference import CARD_COLS

    c = pl.read_parquet(TABLES_DIR / day[:4] / "race_shutuba_flat.parquet").filter(pl.col("date") == day)
    return c.select([x for x in CARD_COLS + ["race_name"] if x in c.columns])


def cmd_build(a) -> int:
    import polars as pl

    from src.pipeline.features.horse_history.builder import build_with_state
    from src.pipeline.features.horse_history.common import build_repair_map, load_runs
    from src.pipeline.features.horse_history.inference import graded_race_names
    from src.pipeline.features.horse_history.serving import write_bundle

    cut = date.fromisoformat(a.day)
    runs = load_runs(years=list(range(2020, cut.year + 1))).filter(pl.col("date") < cut)
    rm = build_repair_map(a.layer_a).filter(pl.col("src_date") <= cut if a.repair_through_day else pl.col("src_date") < cut)
    t = time.perf_counter()
    F, P, S, elo = build_with_state(runs, rm)
    meta = write_bundle(a.bundle, F.columns, P, S, elo, {"graded_names": graded_race_names()})
    print(f"build: {time.perf_counter() - t:.0f}s ピークRSS {_rss():.0f} MB 馬 {meta['horses']} 行 {meta['rows']} 履歴の最終日 {meta['history_last_date']}")
    return 0


def cmd_serve(a) -> int:
    import polars as pl

    from src.pipeline.features.horse_history.serving import Bundle, compute_entrants

    r0 = _rss()
    t = time.perf_counter()
    b = Bundle(a.bundle)
    r1 = _rss()
    out, info = compute_entrants(b, _card(a.day), b.graded_names)
    out.write_parquet(a.out)
    print(f"serve: import後 {r0:.0f} MB → バンドル読込後 {r1:.0f} MB → 計算後ピーク {_rss():.0f} MB / 全体 {time.perf_counter() - t:.1f}s / "
          f"レース {info['races']} 行 {info['rows']} 履歴行 {info['history_rows']}")
    return 0


def cmd_compare(a) -> int:
    import numpy as np
    import polars as pl

    from src.pipeline.features.horse_history.builder import build_horse_history, feature_columns
    from src.pipeline.features.horse_history.common import build_repair_map, read_runs_raw
    from src.pipeline.features.horse_history.inference import build_inference_features, graded_race_names
    from src.pipeline.inference.weight_triggered_predict import REPO_ROOT

    cut = date.fromisoformat(a.day)
    got = pl.read_parquet(a.out)
    card = _card(a.day)
    raw = read_runs_raw(years=list(range(2020, cut.year + 1))).filter(pl.col("finish_position") > 0)
    rm = build_repair_map(a.layer_a).filter(pl.col("src_date") <= cut if a.repair_through_day else pl.col("src_date") < cut)
    full, _ = build_inference_features(raw.filter(pl.col("date") < a.day), card, rm, graded_names=graded_race_names())
    F, _ = build_horse_history(a.layer_a)
    ref = F.filter(pl.col("race_id").is_in(card["race_id"].unique().to_list()))
    cols = sorted({c for v in feature_columns(got.columns).values() for c in v})
    key = ["race_id", "horse_id"]

    def mism(x, y, label):
        j = x.join(y, on=key, suffix="_y")
        bad = {}
        for c in cols:
            if f"{c}_y" not in j.columns:
                continue
            p, q = j[c].to_numpy().astype(float), j[f"{c}_y"].to_numpy().astype(float)
            d = np.where(np.isnan(p) & np.isnan(q), 0.0, np.abs(p - q))
            n = int((np.where(np.isnan(d), np.inf, d) > 1e-9).sum())
            if n:
                bad[c] = n
        print(f"{label}: 比較 {j.height} 行 / 不一致の特徴量 {len(bad)} 個 {bad if bad else ''}")
        return bad

    mism(got, full, "バンドル vs 全履歴の推論モード")
    mism(got, ref, "バンドル vs 学習時の値")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "serve", "compare"])
    ap.add_argument("--day", default="2024-07-06")
    ap.add_argument("--bundle", default="/tmp/serving_bundle_test")
    ap.add_argument("--out", default="/tmp/serving_out.parquet")
    ap.add_argument("--repair-through-day", action="store_true", help="当日の出馬表の past_races による補修も含める（学習時のバッチと突き合わせるときの条件）")
    ap.add_argument("--layer-a", default="data/local/modeling/training_pipeline/layer_a.parquet")
    a = ap.parse_args()
    return {"build": cmd_build, "serve": cmd_serve, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
