# TODO: feature/scraping-queue

**対象領域**: スクレイピング・キュー管理
**関連ドキュメント**: [../feature-scraping-queue.md](../feature-scraping-queue.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- 進捗確認: `/api/scrape-status`（日付別）・`/api/scrape-summary-all`（全日付一括）・`/api/scrape-dates`（済み日付一覧）
- 欠損検出・トリガー: `/api/scrape-missing`（欠損検出）・`/api/scrape-trigger`（個別トリガー）・`/api/check-scraped-status`（一括確認）
- キュー追加系: `add`（単発）・`add-job`（job_kind + tasks[]で任意投入）・`add-batch`（複数レース一括）・`enqueue-incomplete-dates`（データ保有率100%未満の開催日を自動投入）・`enqueue-scrape-period`（期間まとめて投入）
- キュー状態・制御: `status`・`progress`・`jobs`（一覧+統計）・`tasks`（投入可能タスク一覧）・`resume`・`recover`（孤児ジョブ復旧）・`kick`（即時起動）・`clear`・`stop-and-clear`・`hourly-maintenance-run`
- 失敗ジョブ処理: `failed/requeue`（再投入）・`failed/remove`（削除のみ）
- ワーカーログ: `worker-logs`（取得）・`worker-logs/clear`（クリア）
- ランタイム設定: `load-settings`（並列度等の取得/更新）・`local-mirror-config`（ローカルミラー保存設定の取得/更新）
- 未来レースカレンダー収集: `/api/fetch-future-calendar`（+`/status`）
- 自動スクレイプ（外部cron連携）: `/api/auto-scrape/status`・`/api/auto-scrape/run`（dev-only手動実行）・`/api/auto-scrape/run-status`
- 管理画面ページ4つ: `/scrape`・`/scrape-control`・`/queue-status`・`/scrape-upcoming`

## 目標（推測）

ユーザ（エンドユーザー）が見るレース・馬データが常に最新かつ欠損なく揃っていること。
ユーザ自身はこのキュー機能を直接使わないが、「見たいレースのデータが無い」という体験を
無くすための裏方機能。運用者（開発者）にとっては、キューが自律的に回り手動操作
（kick/recover等）が不要になることが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- 定常運用でユーザがデータ欠損（「取得中」「データがありません」表示）に遭遇しない状態が
  継続して維持できている
- 運用者が`kick`/`recover`/`stop-and-clear`等の緊急系エンドポイントを使う頻度が実質ゼロになっている
  （＝自動リカバリで十分）
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

（無し。以前 `/api/cushion/scrape`(+status) が誤ってここに分類されていたが `feature/cushion` へ移動済み）

## TODO（手動追記用）

- [ ]

## メモ
