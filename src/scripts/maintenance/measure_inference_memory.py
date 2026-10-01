"""1レース分の予測のピークメモリと所要時間を測る。

VPS(2GB)で推論を動かせるかの判断材料にする。実データ（GCS/ローカルの出馬表・horse_result）が
必要なため、.env に GCS 認証が入った環境（VPS / 本番相当）で実行すること。

既定は開催日ワークフロー（特徴量ビルダー＋アンサンブル。``KEIBA_MODEL_STORE`` の現行版を使用）の
経路を測る。結果は保存しない。``--legacy`` で従来のライブ推論（keiba_model.pkl）の経路を測る。

使い方:
  KEIBA_ENV=prod python -m src.scripts.maintenance.measure_inference_memory --race-id 202605030811
  KEIBA_ENV=prod python -m src.scripts.maintenance.measure_inference_memory --race-id <ID> --repeat 3
  KEIBA_ENV=prod python -m src.scripts.maintenance.measure_inference_memory --race-id <ID> --legacy

出力: ステージ別の RSS（import後 / データ読み込み後 / 推論後）とピーク値、所要時間。
読み取りのみで、GCS やキャッシュへの書き込みは行わない（キャッシュ保存は呼ばない）。
"""

from __future__ import annotations

import argparse
import resource
import sys
import time


def _rss_mb() -> float:
    """現在のプロセスのピークRSS（MB）。Linuxでは ru_maxrss はKB単位。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-id", required=True, help="推論対象の race_id（出馬表が保存済みのもの）")
    parser.add_argument("--repeat", type=int, default=1, help="繰り返し回数（2回目以降はキャッシュ温まり後の値）")
    parser.add_argument("--legacy", action="store_true", help="従来のライブ推論経路（keiba_model.pkl）を測る")
    args = parser.parse_args(argv)

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()

    print(f"[基準] 素のプロセス: {_rss_mb():.0f}MB")

    t = time.perf_counter()
    from src.pipeline.inference.race_prediction_service import (
        build_race_data_from_storage,
        build_race_prediction_response,
    )
    from src.scraper.storage import HybridStorage

    print(f"[import後] ピーク {_rss_mb():.0f}MB（{time.perf_counter() - t:.1f}s）")

    storage = HybridStorage(".")

    t = time.perf_counter()
    race_data = build_race_data_from_storage(args.race_id, storage)
    entries = len((race_data.get("race_card") or {}).get("entries") or [])
    horses = len(race_data.get("horses") or {})
    print(
        f"[データ読み込み後] ピーク {_rss_mb():.0f}MB（{time.perf_counter() - t:.1f}s）"
        f" 出走{entries}頭 / horse_result取得{horses}頭"
    )
    if entries == 0:
        print("出馬表が見つかりません。race_id と GCS/ローカルデータを確認してください。", file=sys.stderr)
        return 1

    if not args.legacy:
        from src.pipeline.inference.race_day_workflow import load_predictor, predict_race

        t = time.perf_counter()
        predictor = load_predictor()
        print(
            f"[モデル取得・読込] ピーク {_rss_mb():.0f}MB（{time.perf_counter() - t:.1f}s）"
            f" version={predictor.version} 特徴量={len(predictor.feature_names)}列"
        )

    for i in range(1, args.repeat + 1):
        t = time.perf_counter()
        if args.legacy:
            result = build_race_prediction_response(args.race_id, storage, allow_scrape=False)
        else:
            result = predict_race(args.race_id, storage, predictor=predictor, persist=False)
        elapsed = time.perf_counter() - t
        print(
            f"[推論 {i}回目] ピーク {_rss_mb():.0f}MB / {elapsed:.2f}s"
            f" status={result.get('status')} model_type={result.get('model_type')}"
        )

    peak = _rss_mb()
    print(f"\n=== 結果: このプロセスのピークRSS = {peak:.0f}MB ===")
    print("判断の目安: VPS(2GB)でサービング・Redis・OS を除いた推論用の余裕は 400MB 前後。")
    print(f"  ピーク{peak:.0f}MB → ", end="")
    print("VPSで実行可能" if peak <= 400 else "余裕が少ない。メモリ上限付き実行かGCPを検討")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
