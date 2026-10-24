# TODO: feature/admin-ops

**対象領域**: 管理者運用（cron・構造チェック・システム統計・ログ）
**関連ドキュメント**: [../feature-admin-ops.md](../feature-admin-ops.md)
**最終更新**: 2026-10-01

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
  （`SLACK_WEBHOOK_URL`未設定なら無害にスキップ）。2026-10-01: 上記4ジョブの**手動トリガー版**
  （`/api/admin/cron-jobs/{job}/trigger`）にも失敗時のSlack通知を追加（従来は自動ループ版のみ
  通知され非対称だった）。また`structure_check`の自動スケジューラ（`_scheduler_loop`、毎朝6:00 JST）
  の例外発生時にも通知を追加。
- `src/scraper/structure_monitor.run_daily_check()`の`notify`引数（2026-10-01修正）: 従来は未使用の
  deadパラメータで、CRITICALな構造変化を検知しても通知が一切発生しないバグだった。
  `notify=True`かつ`severity=="CRITICAL"`のときに`notify_slack()`を呼ぶよう修正。

## 目標（推測）

ここでの「ユーザ」は運用者（開発者・管理者）。サーバーにSSHせずブラウザ経由で異常検知・
復旧作業ができ、定期ジョブの死活を安心して任せられる状態にすることが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- 運用者が障害対応でSSH/直接ログ確認が必要になる場面がほぼ無くなっている
  （`/api/admin/*`だけで死活・原因調査・再実行が完結する）
- cronジョブの失敗が本画面上で検知でき、放置されたまま気づかない事象が無い
- 上記が実現できていれば、（運用者という）ユーザ向けの実装は完了したとみなせる

## 既知の課題

- `src/monitor/app.py`の`CRON_JOBS`一覧にある`git-pull`（表示上は「毎時」）は実際のcrontabには
  存在しない。`scripts/cron/git_pull_hourly.sh`は2026-09-30以前に手動実行専用（UI経由
  `POST /api/v1/admin/git-pull`）に移行済みで、`scripts/cron/setup_all_cron.sh`にも意図的に
  含まれていない。monitorの表示が実態とズレている（stale）が、本TODOのスコープ外のため未修正。
- `scripts/cron/*.sh`（OS crontab起動のバッチ）および`src/monitor/app.py`のCRON_JOBS表示
  （auto_scrape系・daily-race-lists等）には、失敗時のpush通知（Slack等）は無い。ログファイルの
  パターン一致による事後判定のみ（`_parse_last_success()`）。FastAPIプロセス内で動く
  disk_cache_cleanup等4ジョブ+structure_checkとはアーキテクチャが異なり（OS cronで独立起動する
  シェルスクリプト）、通知を追加するには各スクリプト自体の変更が必要になる。2026-10-01時点では
  対応を見送り（下記TODO参照）。
- 障害対応専用のrunbook（incident response手順書）は`docs/`配下に存在しない。
  `docs/operations/service-endpoints.md`・`server_architecture.html`にローカル実行コマンド
  （起動・ヘルスチェック・既知の問題）の記載はあるが、SSH前提の遠隔運用手順書ではない。
  `/server-logs`・`/api/admin/server-logs`はSSH不要で`logs/*.log`を閲覧できるが、対象はログ
  ファイルに限定され、プロセス確認・ポート確認・ディスク等のOSレベル診断は手動実行が前提。
- **本ファイルの内容は全面的にVPS運用（OS crontab・`src/api/app.py`内のdaemon thread・
  ローカル`logs/`ファイル）を前提にしている。** GCPへの移行を検討する場合、
  Compute Engineでのリフト&シフトなら現行のcronジョブ管理・daemon thread・ログ実装は
  概ねそのまま使える（ログはOps Agent導入でCloud Loggingへ転送可）。一方、Cloud Run等の
  サーバーレス構成を選ぶ場合はcronジョブ管理・構造チェックの自動スケジュール・ログ確認の
  実装がいずれも作り直しになる（daemon threadはCloud Scheduler+Cloud Run Jobsへ、
  ローカルログはCloud Loggingへ）。採用するGCPサービスにより対応が大きく変わる。
  詳細は [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

- [x] cronジョブ失敗時の通知（Slack等）を追加する（現状は画面（`/api/admin/cron-jobs`）を
      見に行かないと失敗に気づけない） — 2026-09-30実装: `src/utils/notify.py`に`notify_slack()`を追加し、
      `/api/admin/cron-jobs`が対象とする4ジョブ（disk_cache_cleanup/queue_maintain/logs_retention/
      daily_shutuba）の失敗時（`except`ブロック）に通知呼び出しを追加。`SLACK_WEBHOOK_URL`未設定時は
      無害に何もしない（既存動作に影響なし）。`.env.example`に変数を追記。テスト:
      `tests/utils/test_notify.py`（3件）。2026-09-30に実際のSlack Incoming Webhook URLを
      `.env`（gitignore対象・未コミット）に設定し、`notify_slack()`の実送信テストに成功済み
      （送信結果 True）。
      なお `scripts/cron/*.sh`（git_pull_hourly等、現状crontab未登録）と
      `src/monitor/app.py`のCRON_JOBS（ログパターン一致で成否判定している自動スクレイプ系）は
      本ブランチのスコープ外のため対象外とした（下のTODOに切り出し）。
- [x] `scripts/cron/*.sh`・`src/monitor/app.py`のCRON_JOBS（auto_scrape系）にもSlack通知を追加するか検討する
      — 2026-10-01検討結果: **今回は追加しない**。これらはFastAPIプロセス内のdaemon thread
      （disk_cache_cleanup等）とは異なり、OS crontabから独立起動するシェルスクリプトであり、
      通知を追加するには本番運用中のcronスクリプト自体を変更する必要があり対象範囲外と判断。
      代わりに、同じ「検討する」調査の過程で、スコープ的に地続きだった`structure_check`
      （`/api/admin/cron-jobs`の対象5件目で、FastAPIプロセス内スレッドとして動く）の
      通知がdeadパラメータのせいで実際には飛んでいないバグを発見し修正した。また元の4ジョブの
      手動トリガー版にも通知が無い非対称を発見し修正した（詳細は「現状の実装」参照）。
      OS crontab起動スクリプト側の課題は「既知の課題」に記録。
- [x] 実際にサーバー運用で発生する障害対応のうち、本画面だけで完結できていない作業を棚卸しする
      — 2026-10-01調査結果（「既知の課題」に記録）: `/server-logs`はSSH不要で`logs/*.log`を
      閲覧できるが対象はログファイルに限定。プロセス確認・ポート確認・ディスク等のOSレベル診断は
      `docs/operations/*`記載のコマンドを手動実行する前提で、専用runbookは存在しない。
      今回はrunbook新規作成は見送り（追加実装なしで棚卸しのみ完了とする）。
- [x] `/api/structure-check`（構造チェック）の自動スケジュール実行状況を確認する
      — 2026-10-01調査結果: 実体はcronではなくFastAPIプロセス内のdaemon thread（`_scheduler_loop`、
      毎朝6:00 JST、`src/api/app.py`）で、プロセスが6:00 JSTを跨いで継続起動していないと実行されない。
      調査時点では`data/local/meta/structure/`が空で過去の実行実績は確認できなかった。
      その過程で`run_daily_check(notify=True)`の`notify`引数が未使用のdeadパラメータという
      実バグを発見し修正済み（CRITICAL検知時に実際にSlack通知が飛ぶようにした）。

### 共通TODO（ホスト方式に関係ない）

（本ファイルでは該当なし。残りの開TODOはいずれもOS crontab・常駐プロセスの有無に依存する）

### VPS側（サービング）に残るTODO

- [x] `src/monitor/app.py`の`CRON_JOBS`一覧から実態と合っていない`git-pull`（毎時）表示を削除・修正する
      — 2026-10-02対応: `schedule`表示を「毎時」から「手動（UI経由 POST /api/v1/admin/git-pull。
      crontab未登録）」に修正（`src/monitor/app.py`の`git-pull`エントリ）。
- [x] 本番環境で`SLACK_WEBHOOK_URL`設定済みの状態で、`structure_check`のCRITICAL検知時に実際に
      Slack通知が届くことを確認する
      — 2026-10-02更新: `structure_check`はGCP側のCloud Run Jobs（`python -m src.scraper.run
      structure-check`、下記GCP側TODO参照）へ移行する方針のため、VPS上のdaemon thread
      （`_scheduler_loop`）前提での確認は意味を失った。通知ロジック自体（`notify_slack()`呼び出し）は
      どちらの実行環境でも同じPython関数内にあるため変更不要。実際の送信確認は、GCP側で
      `SLACK_WEBHOOK_URL`を設定した上でCloud Run Jobsとして実行された後に行う（本項目はVPS側の
      確認としては対応不要と判断してクローズ、GCP側TODOへ引き継ぎ）。
- [x] 「このラインまで実装できたらブランチを消してよい」の基準（SSH/直接ログ確認がほぼ不要になる）
      に対し、障害対応runbookの不在・OSレベル診断（プロセス/ポート/ディスク確認等）の手動実行が
      残っているギャップをどう解消するか方針を決める
      — 2026-10-02決定: VPS/GCP役割分担により、構造チェック・キュー処理・バッチ系の重い診断対象は
      GCP側（gcloud CLI/Cloud Console/Cloud Loggingで診断）へ移った。VPS側に残るのは
      disk-cache-cleanup・logs-retention・ダッシュボード・ログ閲覧という軽量な範囲のみになるため、
      基準を「VPS側はログ・/cron-jobs表示・ダッシュボードの可視化で完結する範囲に限定し、
      GCP側のジョブ診断はgcloud CLI/Cloud Loggingベースの別runbookに委ねる」に見直す。
      SSH前提の重厚なrunbookを新規作成する対応は不要と判断（VPS側の診断範囲が小さくなったため）。
      GCP側の診断手順書は未作成（別タスク）。

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [x] 上記3項目はいずれもサーバーレス移行で前提が変わる: git-pull表示問題はOS crontab自体が
      無くなるため実質解消（代わりにCloud Scheduler側の一覧表示を別途整備する）、
      structure_checkの通知確認はCloud Scheduler+Cloud Run Jobsへの移行後に同種の確認を
      改めて実施する、runbookはSSH前提ではなくgcloud CLI/Cloud Console/Cloud Loggingベースの
      診断手順に作り直す必要がある。全体方針は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)の
      GCP移行TODOを参照 — 2026-10-01対応: structure_check を Cloud Scheduler + Cloud Run Jobs
      から起動するための CLI エントリポイント（既存の `python -m src.scraper.run
      structure-check`。新規ファイル追加は不要だった）を確認し、実行コマンド・想定頻度
      （毎朝06:00 JST）・リソース目安・`gcloud scheduler jobs create http`登録コマンド例を
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)に
      まとめた。デプロイ設計図は
      [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../../scripts/gcp/deploy_cloud_run_jobs.sh)。
      実デプロイ・スケジューラ登録・移行後の通知再確認、git-pull表示問題・runbook刷新は
      本対応の範囲外のまま残る（ユーザー側作業）

## メモ
