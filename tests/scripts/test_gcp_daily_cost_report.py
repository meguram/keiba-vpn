"""
src.scripts.cloud_jobs.gcp_daily_cost_report のユニットテスト。

GCPコストモニタリング（docs/operations/deployment-vps-vs-gcp.md の「GCPコストモニタリング」/
docs/operations/gcp-cloud-run-jobs.md の#15）の検証:
  - fetch_daily_cost が対象テーブル・日付条件で google.cloud.bigquery.Client.query を
    正しく呼ぶこと（google.cloud.bigquery.Client はモック。実BigQuery接続はしない）
  - format_cost_report が妥当な日本語Slackメッセージ文字列を生成すること
  - GCP_BILLING_BQ_TABLE 未設定時・クエリ失敗時にエラー内容がSlackへ通知され、
    main() が例外を外に出さず正常終了（returncode 0）すること
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("GCS_BUCKET", "")

from src.scripts.cloud_jobs import gcp_daily_cost_report as job  # noqa: E402


def _make_row(service_description: str, cost: float):
    row = mock.MagicMock()
    row.__getitem__.side_effect = lambda key: {
        "service_description": service_description,
        "cost": cost,
    }[key]
    return row


class TestFetchDailyCost(unittest.TestCase):
    def test_queries_expected_table_and_date(self):
        with mock.patch("google.cloud.bigquery.Client") as mock_client_cls, mock.patch(
            "google.cloud.bigquery.QueryJobConfig"
        ) as mock_job_config_cls, mock.patch(
            "google.cloud.bigquery.ScalarQueryParameter"
        ) as mock_param_cls:
            mock_client = mock_client_cls.return_value
            mock_query_job = mock_client.query.return_value
            mock_query_job.result.return_value = [
                _make_row("Cloud Run", 1.23),
                _make_row("Cloud Tasks", 0.45),
            ]

            result = job.fetch_daily_cost(
                "2026-09-30", "proj.dataset.gcp_billing_export_v1_000000"
            )

            # Client.query に渡した SQL に対象テーブル・日付パラメータ条件が含まれること
            assert mock_client.query.call_count == 1
            args, kwargs = mock_client.query.call_args
            sql = args[0]
            assert "proj.dataset.gcp_billing_export_v1_000000" in sql
            assert "@target_date" in sql

            # ScalarQueryParameter に正しい日付が渡されていること
            mock_param_cls.assert_called_once_with("target_date", "DATE", "2026-09-30")

            # QueryJobConfig がクエリ実行時に job_config として使われていること
            assert kwargs["job_config"] is mock_job_config_cls.return_value

            assert result["date"] == "2026-09-30"
            assert result["rows"] == [
                {"service": "Cloud Run", "cost": 1.23},
                {"service": "Cloud Tasks", "cost": 0.45},
            ]
            assert round(result["total"], 2) == 1.68

    def test_no_rows_returns_zero_total(self):
        with mock.patch("google.cloud.bigquery.Client") as mock_client_cls, mock.patch(
            "google.cloud.bigquery.QueryJobConfig"
        ), mock.patch("google.cloud.bigquery.ScalarQueryParameter"):
            mock_client = mock_client_cls.return_value
            mock_client.query.return_value.result.return_value = []

            result = job.fetch_daily_cost("2026-09-30", "proj.dataset.table")

            assert result["rows"] == []
            assert result["total"] == 0.0


class TestFormatCostReport(unittest.TestCase):
    def test_formats_breakdown_with_total(self):
        message = job.format_cost_report(
            "2026-10-01",
            [
                {"service": "Cloud Run", "cost": 1.23},
                {"service": "Cloud Tasks", "cost": 0.45},
            ],
            1.68,
        )
        assert message.startswith("[GCP日次コストレポート] 2026-10-01: 合計 $1.68")
        assert "Cloud Run: $1.23" in message
        assert "Cloud Tasks: $0.45" in message

    def test_formats_empty_rows(self):
        message = job.format_cost_report("2026-10-01", [], 0.0)
        assert message == "[GCP日次コストレポート] 2026-10-01: 合計 $0.00（対象データなし）"


class TestMain(unittest.TestCase):
    def test_missing_table_notifies_and_returns_zero(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCP_BILLING_BQ_TABLE", None)
            with mock.patch(
                "src.utils.notify.notify_slack", return_value=True
            ) as mock_notify:
                rc = job.main(["--date", "2026-10-01"])

        assert rc == 0
        assert mock_notify.call_count == 1
        (message,), _ = mock_notify.call_args
        assert "GCP_BILLING_BQ_TABLE" in message
        assert "BigQueryにエクスポート" in message

    def test_query_failure_notifies_error_and_returns_zero(self):
        with mock.patch.dict(
            os.environ, {"GCP_BILLING_BQ_TABLE": "proj.dataset.table"}, clear=False
        ):
            with mock.patch(
                "src.utils.project_env.load_project_dotenv",
                return_value=None,
            ), mock.patch(
                "src.scripts.cloud_jobs.gcp_daily_cost_report.fetch_daily_cost",
                side_effect=RuntimeError("bigquery boom"),
            ), mock.patch(
                "src.utils.notify.notify_slack", return_value=True
            ) as mock_notify:
                rc = job.main(["--date", "2026-10-01"])

        assert rc == 0
        assert mock_notify.call_count == 1
        (message,), _ = mock_notify.call_args
        assert "取得失敗" in message
        assert "bigquery boom" in message

    def test_success_notifies_formatted_report(self):
        with mock.patch.dict(
            os.environ, {"GCP_BILLING_BQ_TABLE": "proj.dataset.table"}, clear=False
        ):
            with mock.patch(
                "src.utils.project_env.load_project_dotenv",
                return_value=None,
            ), mock.patch(
                "src.scripts.cloud_jobs.gcp_daily_cost_report.fetch_daily_cost",
                return_value={
                    "date": "2026-10-01",
                    "rows": [{"service": "Cloud Run", "cost": 2.0}],
                    "total": 2.0,
                },
            ), mock.patch(
                "src.utils.notify.notify_slack", return_value=True
            ) as mock_notify:
                rc = job.main(["--date", "2026-10-01"])

        assert rc == 0
        assert mock_notify.call_count == 1
        (message,), _ = mock_notify.call_args
        assert message == (
            "[GCP日次コストレポート] 2026-10-01: 合計 $2.00（Cloud Run: $2.00）"
        )

    def test_default_date_is_two_days_ago_jst(self):
        # 課金エクスポートの反映が約 1 日遅れるので、確定に近い一昨日を対象にする
        from datetime import datetime, timedelta

        with mock.patch.dict(
            os.environ, {"GCP_BILLING_BQ_TABLE": "proj.dataset.table"}, clear=False
        ):
            with mock.patch(
                "src.utils.project_env.load_project_dotenv",
                return_value=None,
            ), mock.patch(
                "src.scripts.cloud_jobs.gcp_daily_cost_report.fetch_daily_cost",
                return_value={"date": "x", "rows": [], "total": 0.0},
            ) as mock_fetch, mock.patch(
                "src.utils.notify.notify_slack", return_value=True
            ):
                job.main([])

        expected = (datetime.now(job._JST) - timedelta(days=2)).strftime("%Y-%m-%d")
        called_date = mock_fetch.call_args[0][0]
        assert called_date == expected



class ServiceDetailsTest(unittest.TestCase):
    def test_groups_skus_by_service_with_japanese_labels_and_drops_tiny(self):
        rows = [
            {"service": "Cloud Storage", "sku": "Download APAC", "cost": 853.5, "currency": "JPY"},
            {"service": "Cloud Storage", "sku": "Regional Standard Class A Operations", "cost": 59.1, "currency": "JPY"},
            {"service": "Cloud Storage", "sku": "Regional Standard Class B Operations", "cost": 49.7, "currency": "JPY"},
            {"service": "Cloud Storage", "sku": "Standard Storage Tokyo", "cost": 6.3, "currency": "JPY"},
            {"service": "Cloud Run", "sku": "Jobs CPU in asia-northeast1", "cost": 119.5, "currency": "JPY"},
            {"service": "Secret Manager", "sku": "Secret version replica storage", "cost": 0.2, "currency": "JPY"},
        ]
        lines = job.format_service_details(rows, 150.0)
        assert lines[0] == "・Cloud Storage ¥969（GCP の外への転送 ¥854・一覧・書き込み ¥59・読み出し ¥50）"
        assert lines[1] == "・Cloud Run ¥120（Jobs の CPU ¥120）"
        assert len(lines) == 2                     # 1 円未満のサービスは省く



class BlocksTest(unittest.TestCase):
    def test_outlook_uses_month_to_date_plus_recent_average(self):
        table = {"2026-11-01": {"a": 300.0}, "2026-11-02": {"a": 60.0}, "2026-11-03": {"a": 60.0}}
        o = job.month_outlook(table, "2026-11-03", 2000.0)
        assert o["mtd"] == 420.0 and o["avg"] == 140.0
        assert o["projected"] == 420.0 + 140.0 * 27 and o["level"] == "over"

    def test_blocks_have_header_outlook_and_service_table(self):
        rows = [{"service": "Cloud Storage", "sku": "Download APAC", "cost": 853.5, "currency": "JPY"},
                {"service": "Cloud Run", "sku": "Jobs CPU in asia-northeast1", "cost": 119.5, "currency": "JPY"}]
        o = {"mtd": 1000.0, "avg": 50.0, "projected": 1500.0, "budget": 2000.0, "pct": 75.0, "level": "ok"}
        text, blocks = job.build_report_blocks("2026-11-02", {"Cloud Storage": 853.5, "Cloud Run": 119.5}, rows, 150.0, o, ("2026-11-03", 12.0))
        assert blocks[0]["type"] == "header" and "11/2（月）" in blocks[0]["text"]["text"] and "¥973" in blocks[0]["text"]["text"]
        assert "🟢" in blocks[1]["fields"][1]["text"]
        table = blocks[2]["text"]["text"]
        assert "Cloud Storage" in table and "GCP の外への転送 ¥854" in table and table.index("Cloud Storage") < table.index("Cloud Run")
        assert "速報 11/3（火）: ¥12" in blocks[3]["elements"][0]["text"]
        assert "11/2（月）" in text


if __name__ == "__main__":
    unittest.main()
