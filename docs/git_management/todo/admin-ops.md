# TODO: feature/admin-ops

**対象領域**: 管理者運用（cron・構造チェック・システム統計・ログ）
**関連ドキュメント**: [../feature-admin-ops.md](../feature-admin-ops.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- 構造チェック: `/api/structure-status`（最新結果）・`/api/structure-report`（Markdownレポート）・
  `POST /api/structure-check`（手動トリガー）・`/api/structure-check/schedule`（自動スケジュール状態）・
  `/api/structure-fingerprints`（全カテゴリのフィンガープリント）
- cronジョブ管理: `/api/admin/cron-jobs`（一覧・状態）＋個別即時実行トリガー4種
  （disk-cache-cleanup / queue-maintain / logs-retention / daily-shutuba）
- システム監視: `/api/admin/system-stats`（CPU/メモリ/ディスク/プロセス/ネットワーク）
- auto-scrape監視: `/api/admin/auto-scrape-status`（外部cron実行ジョブ一覧）・
  `POST /api/admin/auto-scrape/{task}/trigger`（即時起動）
- ログ確認: `/api/admin/server-logs`（開発者セッション必須）、`/server-logs`ページ
- `POST /api/admin/invalidate-race-list-caches`: race_lists関連インメモリキャッシュの即時クリア（dev-only、daily-race-lists cronから呼ばれる）
- ページ: `/ai-sla`・`/cron-jobs`・`/server-logs`
- `src/utils/notify.py`（2026-09-30追加）: `notify_slack()`。`/api/admin/cron-jobs`が対象とする
  4ジョブ（disk_cache_cleanup/queue_maintain/logs_retention/daily_shutuba）の失敗時にSlack通知
  （`SLACK_WEBHOOK_URL`未設定なら無害にスキップ）

## 目標（推測）

ここでの「ユーザ」は運用者（開発者・管理者）。サーバーにSSHせずブラウザ経由で異常検知・
復旧作業ができ、定期ジョブの死活を安心して任せられる状態にすることが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- 運用者が障害対応でSSH/直接ログ確認が必要になる場面がほぼ無くなっている
  （`/api/admin/*`だけで死活・原因調査・再実行が完結する）
- cronジョブの失敗が本画面上で検知でき、放置されたまま気づかない事象が無い
- 上記が実現できていれば、（運用者という）ユーザ向けの実装は完了したとみなせる

## 既知の課題

（無し）

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

- [x] cronジョブ失敗時の通知（Slack等）を追加する（現状は画面（`/api/admin/cron-jobs`）を
      見に行かないと失敗に気づけない） — 2026-09-30実装: `src/utils/notify.py`に`notify_slack()`を追加し、
      `/api/admin/cron-jobs`が対象とする4ジョブ（disk_cache_cleanup/queue_maintain/logs_retention/
      daily_shutuba）の失敗時（`except`ブロック）に通知呼び出しを追加。`SLACK_WEBHOOK_URL`未設定時は
      無害に何もしない（既存動作に影響なし）。`.env.example`に変数を追記。テスト:
      `tests/utils/test_notify.py`（3件）。**残作業**: 実際のSlack Incoming Webhook URLを
      `.env`に設定し、本番投入前に送信テストを行うこと（未設定のため今回は動作未確認）。
      なお `scripts/cron/*.sh`（git_pull_hourly等、現状crontab未登録）と
      `src/monitor/app.py`のCRON_JOBS（ログパターン一致で成否判定している自動スクレイプ系）は
      本ブランチのスコープ外のため対象外とした（下のTODOに切り出し）。
- [ ] `scripts/cron/*.sh`・`src/monitor/app.py`のCRON_JOBS（auto_scrape系）にもSlack通知を追加するか検討する
      （現状はログファイルのパターン一致で事後的に成否判定しているだけで、push型の通知は無い。
      `src/monitor/app.py`の`_parse_last_success()`が`status == "error"`を判定した箇所が候補）
- [ ] 実際にサーバー運用で発生する障害対応のうち、本画面だけで完結できていない作業を棚卸しする
      （SSH/直接ログ確認が必要な場面が残っていないか）
- [ ] `/api/structure-check`（構造チェック）の自動スケジュール実行状況を確認する

## メモ
