#!/usr/bin/env python3
"""
GCP利用コスト（指定日/既定: 前日 JST）をサービス別に集計し、Slackへ日次レポートする CLI。

## 採用方式（コメントとして明記）

GCPの日次利用コストを取得する方法には主に以下の選択肢がある。

1. **Cloud Billing の BigQuery 課金エクスポートをクエリする（本実装で採用）**
   GCPコンソールの「お支払い」→「課金データのエクスポート」→「BigQueryにエクスポート」で
   有効化した詳細エクスポートテーブルを `google-cloud-bigquery` で直接 SQL 集計する方式。
   GCP公式ドキュメントで日次コスト分析の標準的な方法として案内されており、サービス別・
   SKU別など柔軟な集計が可能、過去分の遡及集計もできる。**採用理由**: 既に
   `google-cloud-storage` 等の GCP クライアントライブラリ運用実績があり、本リポジトリの
   「GCS が source of truth」という既存方針と相性が良い。また Cloud Scheduler + Cloud Run
   Jobs（`docs/operations/gcp-cloud-run-jobs.md`）の既存ジョブ群と同じ薄い CLI ラッパー
   パターンに自然に収まる。
2. Cloud Billing Budgets API + Pub/Sub通知。予算アラートには向くが、サービス別の日次内訳の
   取得には不向き（採用せず）。
3. Cloud Billing Catalog API / Cost Management の REST API を直接叩く。BigQueryエクスポート
   より取得できる情報が限定的で、認証・ページングの実装コストが高い（採用せず）。

**前提条件（ユーザー側のGCPコンソール作業。コードでは自動化できない）**:
GCPコンソールの「お支払い」→「課金データのエクスポート」→「BigQueryにエクスポート」を
事前に有効化し、作成されたエクスポート先テーブル（例:
`project.dataset.gcp_billing_export_v1_XXXXXX`）を環境変数 `GCP_BILLING_BQ_TABLE` に
設定しておく必要がある。未設定・クエリ失敗時はその旨をSlackに通知し、CLI自体は正常終了する
（通知ジョブの障害がサイレントに消えないようにする一方、ジョブ自体を失敗させて後続の
Cloud Scheduler リトライ等を誘発しないようにする）。

元は存在しない新規ジョブ（既存に billing/cost 監視の仕組みは無かった）。
`src.utils.notify.notify_slack` を再利用し、他の `src/scripts/cloud_jobs/*` と同じ
「argparse で引数を受けて既存処理を呼ぶ薄いラッパー」パターンに合わせる。

Cloud Scheduler + Cloud Run Jobs から毎日 JST朝（例: 07:00、課金データの反映遅延を
踏まえた時刻）に実行する想定（docs/operations/gcp-cloud-run-jobs.md 参照）。

``--watch`` を付けると、同じ Slack 通知に『今月累計・月末見込み・移行フェーズの想定範囲との比較』を足し、
サービス別の急増（前日比/7 日平均比 3 倍など）と案 A 上限超えは別メッセージで警告する
（ロジックは ``src/utils/cost_watch.py``、しきい値は ``config/migration_phases.json``）。

Usage:
  python -m src.scripts.cloud_jobs.gcp_daily_cost_report
  python -m src.scripts.cloud_jobs.gcp_daily_cost_report --date 2026-09-30
  python -m src.scripts.cloud_jobs.gcp_daily_cost_report --watch
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

_JST = timezone(timedelta(hours=9))


def fetch_daily_cost(date: str, table: str) -> dict:
    """BigQuery課金エクスポートテーブルから指定日のサービス別コスト合計を取得する。

    Args:
        date: 対象日（``YYYY-MM-DD``）。
        table: 課金エクスポートテーブルの完全修飾名
            （``project.dataset.gcp_billing_export_v1_XXXXXX`` 形式）。

    Returns:
        ``{"date": date, "rows": [{"service": str, "cost": float}, ...], "total": float}``。
        ``rows`` はコスト降順。
    """
    from src.config.gcp_guard import assert_gcp_allowed

    assert_gcp_allowed("BigQuery")
    from google.cloud import bigquery

    query = f"""
        SELECT
          service.description AS service_description,
          SUM(cost) AS cost
        FROM `{table}`
        WHERE DATE(usage_start_time) = @target_date
        GROUP BY service_description
        ORDER BY cost DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_date", "DATE", date),
        ]
    )

    from src.config.gcp_credentials import build_gcp_credentials, gcp_project_id

    credentials = build_gcp_credentials()
    client = bigquery.Client(
        credentials=credentials,
        project=gcp_project_id() or None,
    )
    query_job = client.query(query, job_config=job_config)

    rows: list[dict] = []
    total = 0.0
    for row in query_job.result():
        cost = float(row["cost"] or 0.0)
        rows.append({"service": row["service_description"], "cost": cost})
        total += cost

    return {"date": date, "rows": rows, "total": total}


def fetch_cost_history(table: str, start: str, end: str) -> list[dict]:
    """課金エクスポートから start〜end（JST の日付、両端含む）の日別・サービス別の『請求額（クレジット差し引き後）』を取る。

    各行 ``{"date","service","cost","currency"}``。無料枠はクレジットとして差し引かれるので、請求に近い値になる。
    """
    from src.config.gcp_guard import assert_gcp_allowed

    assert_gcp_allowed("BigQuery")
    from google.cloud import bigquery

    query = f"""
        SELECT
          CAST(DATE(usage_start_time, "Asia/Tokyo") AS STRING) AS d,
          service.description AS service_description,
          currency AS currency,
          SUM(cost) + SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS cost
        FROM `{table}`
        WHERE usage_start_time >= TIMESTAMP_SUB(TIMESTAMP(@start_date), INTERVAL 1 DAY)
          AND DATE(usage_start_time, "Asia/Tokyo") BETWEEN @start_date AND @end_date
        GROUP BY d, service_description, currency
        ORDER BY d, cost DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start),
            bigquery.ScalarQueryParameter("end_date", "DATE", end),
        ]
    )
    from src.config.gcp_credentials import build_gcp_credentials, gcp_project_id

    client = bigquery.Client(credentials=build_gcp_credentials(), project=gcp_project_id() or None)
    return [
        {
            "date": row["d"],
            "service": row["service_description"],
            "cost": float(row["cost"] or 0.0),
            "currency": row["currency"],
        }
        for row in client.query(query, job_config=job_config).result()
    ]


def build_watch(target_date: str, rows: list[dict]) -> tuple[str, str | None, str]:
    """(日次通知に足す行, 警告メッセージまたは None, 円建ての先頭行)。rows は fetch_cost_history の結果。"""
    from src.utils import cost_watch as cw

    phases = cw.load_phases()
    fx = float(_fx_jpy_per_usd())
    table = cw.daily_table(rows, fx)
    summary = cw.month_summary(table, target_date, phases)
    spikes = cw.detect_spikes(table, target_date, phases["spike"])
    over = cw.over_ceiling(table, target_date, cw.service_ceilings(phases))
    alert = cw.format_spike_alert(target_date, spikes, over) if (spikes or over) else None
    return cw.format_summary_lines(summary), alert, format_cost_report_jpy(target_date, table.get(target_date, {}))


def _fx_jpy_per_usd() -> float:
    from src.scripts.ops import cost_estimate as ce

    return float(ce.load_config()["fx_jpy_per_usd"])


def _history_start(target_date: str) -> str:
    t = datetime.strptime(target_date, "%Y-%m-%d").date()
    return min(t.replace(day=1), t - timedelta(days=8)).isoformat()


def format_cost_report(date: str, rows: list[dict], total: float) -> str:
    """コスト集計結果を日本語のSlackメッセージに整形する。"""
    if not rows:
        return f"[GCP日次コストレポート] {date}: 合計 $0.00（対象データなし）"

    breakdown = ", ".join(f"{row['service']}: ${row['cost']:.2f}" for row in rows)
    return f"[GCP日次コストレポート] {date}: 合計 ${total:.2f}（{breakdown}）"


def format_cost_report_jpy(date: str, day: dict[str, float]) -> str:
    """JST の日付・クレジット差し引き後・円換算の先頭行（--watch の警告と同じ集計）。"""
    total = sum(day.values())
    if not day:
        return f"[GCP日次コストレポート] {date}: 合計 ¥0（対象データなし）"
    breakdown = ", ".join(f"{svc}: ¥{v:,.0f}" for svc, v in sorted(day.items(), key=lambda kv: -kv[1]))
    return f"[GCP日次コストレポート] {date}: 合計 ¥{total:,.0f}（{breakdown}）"


def _default_target_date() -> str:
    return (datetime.now(_JST) - timedelta(days=1)).strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default="",
        help="対象日 YYYY-MM-DD（省略時は前日 JST）",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="今月累計・月末見込み・フェーズ想定との比較を通知に足し、急増は別メッセージで警告する",
    )
    args = parser.parse_args(argv)

    target_date = args.date.strip() or _default_target_date()

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()

    from src.utils.notify import notify_slack

    table = os.environ.get("GCP_BILLING_BQ_TABLE", "").strip()
    if not table:
        message = (
            f"[GCP日次コストレポート] {target_date}: 取得失敗（GCP_BILLING_BQ_TABLE が"
            "未設定のため課金エクスポートテーブルを特定できません。GCPコンソールの"
            "「お支払い」→「課金データのエクスポート」→「BigQueryにエクスポート」を"
            "有効化し、対象テーブルを GCP_BILLING_BQ_TABLE に設定してください）"
        )
        print(message, file=sys.stderr)
        notify_slack(message)
        return 0

    try:
        result = fetch_daily_cost(target_date, table)
        message = format_cost_report(target_date, result["rows"], result["total"])
    except Exception as e:  # noqa: BLE001 - 通知ジョブ自体を落とさないため広く捕捉する
        message = f"[GCP日次コストレポート] {target_date}: 取得失敗（{e}）"
        print(message, file=sys.stderr)
        notify_slack(message)
        return 0

    alert = None
    if args.watch:
        try:
            rows = fetch_cost_history(table, _history_start(target_date), target_date)
            extra, alert, headline = build_watch(target_date, rows)
            message = f"{headline}\n{extra}"
        except Exception as e:  # noqa: BLE001 - 日次の基本通知は必ず出す
            message = f"{message}\n（想定との比較は取得失敗: {e}）"

    print(message)
    notify_slack(message)
    if alert:
        print(alert)
        notify_slack(alert)
    return 0


if __name__ == "__main__":
    sys.exit(main())
