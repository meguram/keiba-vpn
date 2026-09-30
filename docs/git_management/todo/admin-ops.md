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

- [ ]

## メモ
