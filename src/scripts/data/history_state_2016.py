"""馬の履歴の推論用の状態（当て直した係数 P・S・過去走・Elo の最終状態）を 2016〜2025 年の全走から作る（再学習 2026-11）。

    python -m src.scripts.data.history_state_2016 [--out /home/jovyan/work/retrain_out/base_2016]

学習の表（rebuild_base_2016）と同じ係数（斤量の効果 β など）を推論でも使うためのもの。本番の日次の蓄積（accumulate.py）は
履歴を 2020 年から作っているので、組み込むときはそちらも 2016 年からにする。出力: <out>/serving_bundle_hh/、<out>/features/history_params.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016")
    a = ap.parse_args()
    B = Path(a.out)
    from src.scripts.data import rebuild_base_2016 as rb

    years = list(range(2016, 2026))
    rb._patch_years(years)
    from src.pipeline.features.horse_history.builder import build_with_state
    from src.pipeline.features.horse_history.common import build_repair_map, load_runs
    from src.pipeline.features.horse_history.inference import graded_race_names
    from src.pipeline.features.horse_history.serving import write_bundle

    runs = load_runs(years=years)
    ot_path = ROOT / "data/local/modeling/training_pipeline/features/on_time_repair.parquet"
    ot = pl.read_parquet(ot_path) if ot_path.exists() else None
    F, P, S, elo = build_with_state(runs, build_repair_map(B / "layer_a.parquet"), ot)
    meta = write_bundle(B / "serving_bundle_hh", F.columns, P, S, elo, {"graded_names": graded_race_names(None, years)})
    tmp = B / "features/history_params.pkl.tmp"
    tmp.write_bytes(pickle.dumps({"P": P, "S": S}))
    tmp.replace(B / "features/history_params.pkl")
    print("保存", B / "serving_bundle_hh", json.dumps(meta, ensure_ascii=False, default=str)[:300], flush=True)
    print("sr_beta", P.get("sr_beta"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
