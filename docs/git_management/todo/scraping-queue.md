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
- **2026-10-01追加**: `job_queue.py` の `update_job_status` が `failed` 遷移時に `first_failed_at` を記録（`requeue_failed_jobs` で `pending` に戻っても保持）。既存の常駐メンテループ（`app.py` の `_queue_hourly_maintain_loop` → `run_hourly_queue_maintenance`）内で `check_stale_failed_jobs_and_notify`（既定6時間、`KEIBA_QUEUE_STALE_FAILED_ALERT_HOURS` で変更可）を毎回実行し、`first_failed_at` から閾値を超えて解消していない `failed` ジョブを `notify_slack`（`src/utils/notify.py`）で通知する。同一ジョブへの再通知は24時間（または閾値の大きい方）間隔を空ける（`data/queue/queue_stale_failed_alert_state.json`）。テスト: `tests/scraper/test_queue_stale_failed_alert.py`。

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
      （2026-10-01: スキップ。理由: 本番アクセスログ・呼び出し履歴がローカルに存在せず計測不可。
      `logs/`配下には`auto_scrape*.log`等のcronタスクログのみで、FastAPI側のアクセスログ
      （uvicorn access log 相当）や`job_queue.py`内のエンドポイント呼び出しカウンタは実装されておらず、
      ローカル環境から実測できる手段が無い。計測するには `/api/admin/server-logs` 相当のアクセスログ
      記録（エンドポイント別カウンタ等）を新規実装する必要がある。モックデータでの頻度計測は
      実態を反映しないため作成しなかった。）
- [x] データ欠損検出（`/api/scrape-missing`）から再取得までの自動化率を確認する
      （現状は手動トリガー系エンドポイントが多く残っており、自動化しきれていない可能性がある）
      — 2026-10-01対応: コードベース調査の結果、**定常運用は完全自動化済み**と判断。
      `scripts/cron/setup_all_cron.sh` が登録する全SLAタスク（`daily-race-lists`・`raceday-eve`・
      `raceday-runner`・`raceday-result-runner`・`raceday-evening`・`weekly-update`等）は
      `scripts/cron/run_auto_scrape_logged.sh` 経由で `python -m src.scraper.auto_scrape --task <T>`
      を直接実行し、`src/scraper/auto_scrape.py` は `_via_queue()`（既定True、env
      `KEIBA_AUTO_SCRAPE_USE_QUEUE`）が真の場合 `src/scraper/auto_scrape_queue.py` の
      各タスク関数が **HTTP API を経由せず** `ScrapeJobQueue.add_job`/`kick_process_queue_background`
      をin-processで直接呼び出してキュー投入・起動する。つまり「検出→キュー投入→起動」の
      一連の流れはcronだけで完結しており、`/api/scrape-missing`・`/api/scrape-queue/add`・
      `/api/scrape-queue/kick`等のHTTPエンドポイントを人間やcronが呼ぶ必要は無い。
      `/api/scrape-missing`・`/api/scrape-queue/enqueue-incomplete-dates`は
      `templates/admin/scrape.html`・`scrape_control.html`・`queue_status.html`からのみ
      参照されており（cronスクリプトからの呼び出しは`grep`で0件）、運用者が手動で補完確認・
      ピンポイント再取得する際のUI向け補助エンドポイントという位置づけ。したがって
      「手動トリガー系が多く残っている」こと自体は事実だが、それは自動化不足ではなく、
      定常フローとは別に人間向けの可視化・手動介入手段を提供する設計意図によるもの。
      自動化率としては定常SLAタスクの範囲でほぼ100%（cron以外の人手を要しない）と結論。
- [x] 欠損が一定時間解消されない場合のアラート通知（Slack等）の追加を検討する
      （通知を送る判断ロジック自体はホスト方式に関係ないが、実装先は下記を参照）
      — 2026-10-01対応: 実装した。既存の常駐メンテループ（`app.py` の
      `_queue_hourly_maintain_loop` → `job_queue.run_hourly_queue_maintenance`、既定1時間毎）に
      新規追加した `check_stale_failed_jobs_and_notify()` を組み込んだ（新規の常駐監視ループは
      作らず、既存スレッドに乗せたため小規模な変更で収まった）。`update_job_status` が
      `failed`遷移時に`first_failed_at`を記録し、hourly maintenanceの`requeue_failed_jobs`で
      `pending`に戻っても`first_failed_at`は保持されるため、「初めて失敗した時刻からの経過時間」
      で判定できる。閾値（既定6時間、`KEIBA_QUEUE_STALE_FAILED_ALERT_HOURS`で変更可）を超えて
      `failed`のまま（アクセス一時停止中で`pending`に戻せていないケースも含む）のジョブを
      `src/utils/notify.py`の`notify_slack()`で通知する。同一ジョブの再通知は24時間
      （または閾値の大きい方）間隔を空ける（`data/queue/queue_stale_failed_alert_state.json`で
      抑制状態を保持）。テスト追加: `tests/scraper/test_queue_stale_failed_alert.py`
      （`update_job_status`のfirst_failed_at付与・クリア、閾値超過時の通知、再通知抑制、
      閾値0での無効化を検証）。`python3 -m pytest tests/ --ignore=tests/scraper/manual
      --ignore=tests/research/manual` で既存460件+新規5件が全てpass。

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] 上記アラート通知を追加する場合、現行の`job_queue.py`常駐ワーカースレッド内に
      そのまま実装できる

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] `job_queue.py`（ローカルJSONファイル+ファイルロック）のキュー基盤をCloud Tasks等へ
      置き換えてからでないと、上記アラート通知も含めた機能追加の前提が変わる。
      netkeibaスクレイピングのoutbound IP固定化（Cloud NAT等）も合わせて必要。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
