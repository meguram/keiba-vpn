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
- `job_queue.py`のキューはローカルJSONファイル＋ファイルロック（`_LOCK_TIMEOUT`30分）を前提に、
  常駐ワーカースレッドが継続処理する設計（VPS運用前提）。また netkeiba スクレイピングは
  `NETKEIBA_MAX_CONCURRENT_REQUESTS=1`等の低負荷・低頻度設計で単一の安定した送信元IPからの
  アクセスを前提にしている。GCPへ移行する場合、**Compute Engineでのリフト&シフトなら
  現行方式のまま、予約静的IPでIP安定性も低コストに継続できる**。一方、Cloud Run等の
  サーバーレス構成を選ぶ場合はキュー基盤自体の再設計（Cloud Tasks等）と、outbound IP固定化
  （Serverless VPC Connector+Cloud NAT。netkeiba側のブロック・CAPTCHAリスク対策として重要、
  かつCompute Engineの予約IPより固定費が高くなりやすい）の両方が必要になる。
  本ワークロードの形状（常時稼働・安定IP必須）からはCompute Engineの方がコスト面で
  有利になりやすい。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [ ] `kick`/`recover`/`stop-and-clear`等の緊急系エンドポイントの実際の使用頻度を計測する
      （頻発しているなら自動リカバリ側の改善が必要）
- [ ] データ欠損検出（`/api/scrape-missing`）から再取得までの自動化率を確認する
      （現状は手動トリガー系エンドポイントが多く残っており、自動化しきれていない可能性がある）
- [ ] 欠損が一定時間解消されない場合のアラート通知（Slack等）の追加を検討する
      （通知を送る判断ロジック自体はホスト方式に関係ないが、実装先は下記を参照）

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] 上記アラート通知を追加する場合、現行の`job_queue.py`常駐ワーカースレッド内に
      そのまま実装できる

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] `job_queue.py`（ローカルJSONファイル+ファイルロック）のキュー基盤をCloud Tasks等へ
      置き換えてからでないと、上記アラート通知も含めた機能追加の前提が変わる。
      netkeibaスクレイピングのoutbound IP固定化（Cloud NAT等）も合わせて必要。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
