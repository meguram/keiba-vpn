"""GCP Cloud Scheduler + Cloud Run Jobs 用の薄い CLI エントリポイント群。

このパッケージの各モジュールは、元は src/api/app.py 内の daemon thread や
scripts/cron/ の OS crontab から定期実行されていた既存の処理関数をそのまま呼ぶだけの
ラッパーである（処理ロジック自体の再実装は行わない）。VPS（サービング専用）と
GCP（スクレイピング・ML・スケジュール実行専用）の役割分担については
docs/operations/deployment-vps-vs-gcp.md、各ジョブの実行コマンド・頻度・リソース目安は
docs/operations/gcp-cloud-run-jobs.md を参照。
"""
