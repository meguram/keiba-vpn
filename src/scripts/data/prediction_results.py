"""予測結果の日次集計 CLI（data/local/prediction_results/<YYYYMMDD>.json）。

  python -m src.scripts.data.prediction_results --date 20261018            # 1 日（結果の状態で暫定/確定を自動判定）
  python -m src.scripts.data.prediction_results --from 20261001 --to 20261018   # 範囲（開催日だけ保存）
  python -m src.scripts.data.prediction_results --date 20261018 --dry-run  # 保存せず内容を表示
  python -m src.scripts.data.prediction_results --sync-gcs   # ローカルの日次記録をすべて GCS（ops/prediction_results/）へ置く（初回の移行。GCS_BUCKET が要る）
  python -m src.scripts.data.prediction_results --from 20260301 --to 20261024 --add-levels   # 保存済みの記録に推奨度だけを追記（印・着順・回収率は変えない）
  python -m src.scripts.data.prediction_results --from 20260101 --to 20261025 --add-target-bets   # 保存済みの記録に、画面の推奨馬券(target)の仮想購入と推奨度を追記（標準以上の集計に必要。nice で直列）

稼働中の自動実行は KEIBA_PREDICTION_RESULTS=1 のときだけ（raceday-result-runner の終了時=暫定、recent-retry の確定結果取得後=確定）。
この CLI は環境変数に関係なく動く。ホストでは nice をつけて直列で実行する。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

from src.pipeline.inference import prediction_results as pr


def _dates(a: str, b: str) -> list[str]:
    d0, d1 = datetime.strptime(a, "%Y%m%d"), datetime.strptime(b, "%Y%m%d")
    return [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--max-total-bet", type=int, default=pr.DEFAULT_MAX_TOTAL_BET)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync-gcs", action="store_true", help="ローカルの日次記録を GCS へ一括で置く")
    ap.add_argument("--add-levels", action="store_true", help="保存済みの日次記録に仮想購入の推奨度だけを追記する")
    ap.add_argument("--add-target-bets", action="store_true", help="保存済みの日次記録に、画面の推奨馬券（target）の仮想購入（推奨度つき）を追記する。推奨度の集計（見送りを除く）に必要")
    a = ap.parse_args(argv)
    if a.sync_gcs:
        from src.utils.project_env import load_project_dotenv

        load_project_dotenv()
        print(pr.sync_to_gcs())
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()           # GCS_BUCKET・KEIBA_ENV・KEIBA_PREDICTION_RESULTS_GCS などを .env から（--sync-gcs と同じ）
    if a.date:
        days = [a.date]
    elif a.d_from and a.d_to:
        days = _dates(a.d_from, a.d_to)
    else:
        ap.error("--date か --from/--to を指定してください")

    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    if a.add_target_bets:
        for d in days:
            if pr.read_day(d) is None:
                continue
            print(d, pr.add_target_bets(d, storage, dry_run=a.dry_run))
        return 0
    if a.add_levels:
        for d in days:
            if pr.read_day(d) is None:
                continue
            print(d, pr.add_levels(d, storage, dry_run=a.dry_run))
        return 0
    for d in days:
        if a.dry_run:
            doc = pr.build_day(d, storage, a.max_total_bet)
            doc.pop("races", None)
            print(json.dumps(doc, ensure_ascii=False, indent=1) if doc["summary"]["races_settled"] else f"{d}: 対象なし")
            continue
        doc = pr.aggregate_day(d, storage, max_total_bet=a.max_total_bet)
        s = doc["summary"]
        if s["races_settled"] == 0:
            print(f"{d}: 対象なし（予測または結果のあるレースが無い）")
            continue
        b = s["bets"]
        roi = f"{b['returned'] / b['invested'] * 100:.0f}%" if b["invested"] else "-"
        print(f"{d}: {doc['status']} 集計 {s['races_settled']}/{s['races']}R 本命1着 {s['marks']['honmei']['f1']}/{s['marks']['honmei']['n']} 回収率 {roi}"
              f"{'' if doc.get('saved') else '（確定済みの日を暫定で上書きしないため保存せず）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
