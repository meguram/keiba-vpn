"""1 レースの予測（馬体重の検知後。または発走 45 分前のフォールバック）。Cloud Tasks が Run Job の引数を上書きして起動する。

    python -m src.ops.jobs.predict_race --race-id 202605030811 --date 20261024

VPS は検知とイベントの書き込みだけを行い（250M に ML は載らない）、特徴量づくりと予測はここで行う。中身は従来の
``weight_triggered_predict.run_for_races``（出走馬の履歴だけで特徴量を作る bundle エンジン）と同じで、出力は
``data/local/inference/{features,snapshots,predictions_v3,predictions_v5,events}/<日>/`` → ディスパッチャが ``state/`` に上げる。
環境変数 KEIBA_V3_SERVING / KEIBA_V5_SHADOW でモードが決まる（並走期間は job_map の parallel_env で影モード固定）。
冪等（同じレースを二度実行しても同じ出力を上書きするだけ）。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def pick_race(rows: list[dict[str, Any]], race_id: str) -> dict[str, Any]:
    for r in rows:
        if str(r.get("race_id")) == race_id:
            return r
    return {"race_id": race_id}          # 発走時刻表に無くても、出馬表だけで特徴量は作れる


def run(race_id: str, date: str, *, weights_known: bool = True, storage: Any | None = None, ctx: Any | None = None, run_for_races: Any | None = None) -> list[dict[str, Any]]:
    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage(".")
    from src.pipeline.inference import race_day_workflow as wf

    row = pick_race(wf.load_day_schedule(storage, date), race_id)
    if ctx is None:
        from src.pipeline.inference.weight_triggered_predict import InferenceContext

        ctx = InferenceContext()
        ctx.warm()
    if run_for_races is None:
        from src.pipeline.inference.weight_triggered_predict import run_for_races
    return run_for_races([row], storage, ctx, weights_known={race_id: weights_known})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--weights-known", type=int, default=1)
    ns = ap.parse_args(argv)
    if len(ns.race_id) != 12 or not ns.race_id.isdigit():
        print("race_id は 12 桁の数字", file=sys.stderr)
        return 2
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    events = run(ns.race_id, ns.date, weights_known=bool(ns.weights_known))
    print(json.dumps(events, ensure_ascii=False, default=str))
    return 0 if events and all(e.get("status") == "ok" for e in events) else 1


if __name__ == "__main__":
    sys.exit(main())
