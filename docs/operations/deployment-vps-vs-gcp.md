# デプロイ構成: ConoHa VPS（公開・サービング） + GCP（スクレイピング・ML・スケジュール実行）

**対象領域**: デプロイ・インフラ全体（cron・常駐プロセス・ストレージ・ネットワーク公開）
**関連ドキュメント**: [service-endpoints.md](./service-endpoints.md)・[server_architecture.html](./server_architecture.html)・
[../git_management/todo/admin-ops.md](../git_management/todo/admin-ops.md) ほか各TODOファイル
**最終更新**: 2026-10-02

**位置づけ（2026-10-02決定）**: VPS/GCPのどちらか一方を選ぶのではなく、**役割分担する**方針が
決定した。
- **ConoHa VPS**（2GB・軽量シェアードコア）: ページ公開・ユーザー向けAPIサービング専用。
  必要最低限のファイルデータのみ置く（大半はGCS参照）。重い処理（スクレイピング・ML学習・
  定期バッチ）は乗せない。
- **GCP**: スクレイピング・ML学習・スケジュール実行（定期バッチ）専用。
  VPSのCPU/メモリが軽量なため、重い処理はすべてここに移す。

共有バックボーンは既存の**GCS**（`HybridStorage`が既にsource of truthとして利用）。
GCPサービスアカウント認証情報は**ファイルではなく `.env` で管理**する
（2026-10-02決定）。`.env`（dev）・`.env.stg`（stg）・`.env.prod`（prod）の構成で、
`KEIBA_ENV=stg|prod` のとき `.env` の後に該当ファイルを上書きマージする
（`src/utils/project_env.load_project_dotenv()`）。サービスアカウントの各フィールドは
既存GCS接続と同じ `GCS_*` 環境変数（`GCS_PRIVATE_KEY`等）に書き、
`src/config/gcp_credentials.py` の `build_gcp_credentials()` がGCS/Cloud Tasks/Cloud SQL/
BigQuery等すべてのGCPクライアントへ共通で渡す（本ドキュメント・関連コードは
疎通済みを仮定して進める）。

<!--
  このファイルの構成: 現状／役割分担マッピング／ブリッジが必要な点／比較表／TODO。
  コードや運用方針が変わったら「役割分担マッピング」も合わせて更新する。
-->

## 現状（2026-10-01調査、分担前のVPS運用の実態）

- エントリ: `main.py` が uvicorn で `src.api.app:app`（FastAPI, :8000）を起動。
  Flask `/api/v1`(:5000) が仕様上の正、FastAPI(:8000)はJinja2 UI・管理画面・レガシーREST
  （`docs/operations/service-endpoints.md` の DEC-013）。
- 定期実行はOS crontab + `src/api/app.py`内のdaemon thread 7種以上
  （`_scheduler_loop`構造チェック・`_weekly_sire_agg_loop`・`_disk_cache_cleanup_loop`・
  `_queue_hourly_maintain_loop`・`_logs_retention_loop`・`_daily_shutuba_enqueue_loop`・
  キューワーカー `_queue_slot_worker`等）。すべて同一プロセス内で常時稼働。
- スクレイピングキューはローカルJSONファイル＋ファイルロック（`src/scraper/job_queue.py`）。
- ストレージはGCSが唯一のsource of truth（`HybridStorage`）。
- PostgreSQL・RedisはいずれもDocker Compose（`docker-compose.dev.yml`）のローカルコンテナ。
- MLflowも`mlflow/server/docker-compose.yml`で同一ホスト上にDocker Compose構成。
- 認証はHMAC署名付きクッキー（`src/api/auth.py`）。
- ライブ推論（`race_prediction_service`）はMLflow Registryを経由せず**ローカルpklファイル**
  （`models/keiba_model.pkl`）を直接読む実装（2026-10-01調査で判明）。

## 役割分担マッピング（機能領域 → 担当環境）

`docs/git_management/todo/*.md` の16機能領域を、今回決定した役割分担にマッピングする。
「表示/配信=VPS、実行=GCP」の形になる領域は両方に分かれる。

| 機能領域（TODOファイル） | 担当環境 | 理由・移行対象 |
|---|---|---|
| core-platform | **VPS** | 認証・ダッシュボード・ヘルスチェックはサービングの一部 |
| horse-profile | **VPS** | 馬名検索・馬詳細はリクエスト同期の読み取りAPI |
| race-detail | **VPS**（配信）/ **GCP**（予測実行） | 予測結果の配信はVPS、`POST .../predict`の実行自体はGCPのCloud Run Jobsに委譲しGCSへ書き込む方針に変更 |
| race-quality | **VPS**（配信）/ **GCP**（day一括推定の実行） | 日次一括推定はGCP側のCloud Scheduler+Jobsで事前計算、配信はVPSがGCSから読むだけ |
| tracking-difficulty | **VPS**（配信）/ **GCP**（precomputeバッチ） | 同上パターン |
| track-speed | **VPS**（配信）/ **GCP**（rebuild-baselines） | 同上パターン |
| odds-final-odds | **VPS**（配信）/ **GCP**（train・snapshot記録） | 学習・定期スナップショットはGCP、配信はVPS |
| betting | **VPS** | optimizeはリクエスト同期の軽量計算（オッズ無い場合のスクレイピング呼び出しのみ要注意） |
| growth-curve | **VPS** | 読み取り系API、計算もリクエスト同期で軽量 |
| myostatin | **VPS**（配信）/ **GCP**（recalculate定期実行） | 再計算バッチはGCP、knowledge参照・predictはVPS |
| bloodline-pedigree | **VPS**（配信）/ **GCP**（アーティファクトrebuild） | 65エンドポイントの大半は読み取り系でVPS、`POST rebuild`系の重い再構築処理はGCP |
| cushion | **GCP**（スクレイピング・ライブ取得・GCS同期） / **VPS**（data/statsの配信） | スクレイピングはGCP、配信のみVPS |
| scraping-queue | **GCP** | job_queue.py自体・netkeibaスクレイピング全体をGCPへ移動（VPSのCPU負荷を避ける主目的） |
| monitor-quality | **VPS** | 運用者向け監視画面・品質チェック結果表示（品質チェック自体がスクレイピング済みデータの検証なので軽量、VPSに残してよい） |
| model-training | **GCP** | 学習・アンサンブル・バックテスト・バックフィルは最も重い処理。VPSには絶対に乗せない |
| admin-ops | **VPS**（画面・ログ閲覧）/ **GCP**（cronジョブの実体） | `/cron-jobs`等の管理画面はVPSに残すが、disk-cache-cleanup以外のジョブ実体（スクレイピング系）はGCPへ。VPS側に残るのは軽量なdisk-cache-cleanup・logs-retention程度 |

## ブリッジが必要な点（分担により新たに生じる課題）

1. **PostgreSQL**: VPS（サービング）とGCP（バッチ）の両方からアクセスする必要が生じる。
   VPS上の自己ホストDocker ComposeのままではGCP側からの到達性確保（ファイアウォール開放等）
   が必要になり運用が複雑。**Cloud SQLへ移行するのが素直**（公開IP+SSL、または
   Cloud SQL Auth Proxy/Python Connectorで両側から接続）。これはコスト最適化の話ではなく、
   分担構成そのものに必要な変更。
2. **Redis**: サービング側（VPS）のリクエストキャッシュとして使われており、低レイテンシが
   重要。Memorystoreはデフォルトで同一VPC内からしかアクセスできずクロスクラウド接続が
   複雑なため、**VPS側に自己ホストのまま残す**のが現実的（GCP側のバッチ処理はRedisを
   直接必要としない設計のため問題にならない）。
3. **モデル配信**: ライブ推論は現状ローカルpklファイル（`models/keiba_model.pkl`）を直接
   読む実装。学習をGCPに移すと、学習済みモデルをGCS経由でVPSへ同期する仕組みが新規に必要
   （学習→GCS保存→VPS側が定期的に最新モデルをGCSから取得してローカルに配置、という経路）。
4. **スクレイピング・推論の手動トリガー**: 現状`/api/scrape-trigger`・`POST .../predict`等の
   dev-only手動トリガーはVPS上で即時実行する設計。分担後は、VPS側のトリガーはGCP側の
   Cloud Run Jobs/Cloud Tasksを呼び出すプロキシに変える必要がある。
5. **GCS経由の疎通**: サービスアカウント認証情報を両環境の `.env`（`GCS_*`）に設定すれば、
   GCS/Cloud Tasks/Cloud SQL/BigQuery等のGCPクライアントはすべて疎通する前提で進める。

## ユーザーリクエスト起点の計算処理をどこで実行するか（2026-10-02決定）

ページアクセス時にユーザーが起点となって何らかの計算を発生させるAPI（on-demand compute）を
GCP側に投げるべきか、という論点について、コスト・レイテンシの両面から以下を結論とする。

- **同期・軽量な計算（数ms〜数十ms、スクレイピングを伴わない）はVPSに残す**。
  例: growth-curveのオンデマンド計算（実測2〜4ms程度）、race-qualityのフォールバック計算、
  myostatin predict。VPS→GCPへの同期呼び出しは、ネットワーク往復（数十ms〜）に加えCloud Run
  のコールドスタート（スケールtoゼロ時は最大数秒）が乗るリスクがあり、ユーザーがページ表示を
  待っている最中に発生すると体感レイテンシが悪化する。常時ウォーム（min-instances≥1）に
  すればコールドスタートは避けられるが、スケールtoゼロのコストメリットが消え、かつ元の処理が
  数msで終わる以上GCPに移すコスト削減効果もほぼ無い。**レイテンシ・コストどちらの観点でも
  VPS側に残すのが正解**。
- **重い処理（学習・バックテスト・大量再計算・precompute系）は既存の非同期ジョブパターンを
  徹底する**: VPSはジョブ投入（Cloud Tasks経由）とジョブIDの返却のみを行い、即座にレスポンスする。
  実際の計算はGCP側（Cloud Run Jobs）で実行し、結果をGCSへ書き込む。ユーザーは`/status`ポーリング
  または次回アクセス時に結果を受け取る。この方式なら重い処理の実行時間がユーザーの初回リクエストの
  レイテンシに影響せず、かつ計算コストはGCPのpay-per-use（Cloud Run Jobsは実行時間のみ課金）に
  収まる。`POST /api/race/{race_id}/predict`・`POST /api/track-speed/rebuild-baselines`・
  `POST /api/bloodline/analyze`等、既に「投入→`/status`確認」の形になっているAPIはこの方針と
  整合している。
- **既知のギャップ（要修正）**: 以下のAPIは「GCSに結果が無ければその場で同期的にスクレイピングを
  実行する」フォールバックを持っており、VPSが直接スクレイピングしない方針と矛盾する。
  GCP側へのジョブ委譲（Cloud Tasks経由で投入し、取得完了まではキャッシュ無し/pending応答を返す）
  に変更する必要がある:
  - `/api/betting/pair-odds/{race_id}`（GCSに無い場合にスクレイピングを実行）
  - `/api/growth-curve/{horse_id}?fetch_speed_index=true`（race_index補完でGCS増時にスクレイピングが
    発生し得る経路）
  - `POST /api/bloodline/analyze`（内部でのスクレイピング発生有無を要確認）

## 比較表（移行前後の対比）

| 項目 | 分担前（VPSのみ） | 分担後 |
|---|---|---|
| 常駐daemon thread | VPS上のFastAPIプロセス内 | 軽量なもの（disk-cache-cleanup等）のみVPS継続、重いもの（構造チェック・daily-shutuba・週次集計等）はGCPのCloud Scheduler+Cloud Run Jobsへ |
| スクレイピングキュー | VPS上のローカルJSON+ファイルロック | GCPへ移動（Cloud Tasks + Cloud Run Jobs、または専用Compute Engine） |
| モデル学習・バックテスト・バックフィル | VPS上でバックグラウンド実行 | GCPのCloud Run Jobs（長時間ジョブ対応） |
| PostgreSQL | VPS上の自己ホスト | Cloud SQL（両環境から到達可能にするため） |
| Redis | VPS上の自己ホスト | 変更なし（VPS継続、サービング用途のため） |
| モデル配信 | ローカルpklファイル直接読み込み | 変更なし＋GCSからの同期経路を追加 |
| ログ | VPSローカル`logs/*.log` | VPS側はそのまま。GCP側のジョブはCloud Loggingへ出力 |

## CI/CD（2026-10-02決定）

「VPSは必要最低限のファイルだけ」を実際に実現するためのCI/CD設計を決定した。詳細設計は
[cicd-design.md](./cicd-design.md)。要点のみここに記す。

- **VPSにgitリポジトリ全体をpull/checkoutさせない**。代わりにCI側でサービング専用の
  軽量Dockerイメージ（[`Dockerfile.serving`](../../Dockerfile.serving)）をビルドし、
  VPSはそのイメージだけを受け取って動かす（[`docker-compose.serving.yml`](../../docker-compose.serving.yml)）。
  VPS上に実際に必要なファイルは`docker-compose.serving.yml`・`.env`・`.env.<環境>`
  程度まで減る（GCP認証情報は`.env`内の`GCS_*`）。
- イメージは`main.py`・`src/`・`templates/`・`static/`・`requirements.txt`のみを含む
  （`notebooks/`・`docs/`・`tests/`・`scripts/`・`data/`・`mlflow/`は含めない）。
  `src/api/app.py`の実際のimportをgrepした結果、`src/scraper`・`src/research`・
  `src/pipeline`の大半が開発者専用の管理画面・手動トリガー系エンドポイントから遅延import
  されており安全に分離できなかったため、現時点では`src/`全体を含めている
  （将来整理の余地ありと`Dockerfile.serving`内にコメントで明記）。
- サービング用イメージ: [`.github/workflows/deploy-vps.yml`](../../.github/workflows/deploy-vps.yml)が
  `stg`/`master`へのpushでビルドしGHCR（ghcr.io）へpushし、VPSへSSHして
  `docker compose pull && up -d`相当を実行する。
- GCP用バッチイメージ: [`.github/workflows/deploy-gcp.yml`](../../.github/workflows/deploy-gcp.yml)が
  `stg`/`master`へのpushでビルドしArtifact Registryへpushし、既存の
  [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../scripts/gcp/deploy_cloud_run_jobs.sh)を
  呼んでCloud Run Jobs/Cloud Schedulerを更新する。
- 既存の`scripts/cron/git_pull_hourly.sh`・`scripts/server/service_start.sh`等の
  「git pullしてVPS上でPythonを直接動かす」運用スクリプトは**変更・削除せず残す**
  （段階的移行の前提。Docker化は新たな選択肢として追加しただけ）。
- 必要なGitHub Secrets（VPS用・GCP用）・移行手順は`cicd-design.md`参照。本対応時点では
  いずれも未設定であり、ワークフロー自体の作成のみ行っている（実行時にSecrets未設定で
  失敗するのは許容・ユーザー側の今後の設定作業）。

## GCPコストモニタリング

2026-10-01時点ではGCPコストモニタリングの既存設計・実装は無かった（`src/`・`docs/`に
billing/cost関連の仕組みが存在しないことを確認済み）。GCP利用の本格化に備え、Cloud Billing
のBigQuery課金エクスポートをクエリして日次コストをSlackへ通知する
`python -m src.scripts.cloud_jobs.gcp_daily_cost_report`（詳細・前提条件は
[gcp-cloud-run-jobs.md の#15](./gcp-cloud-run-jobs.md)）を新規実装した。

## TODO（手動追記用）

### 整理・設計系の共通TODO
- [x] 役割分担（VPS=サービング、GCP=スクレイピング/ML/スケジュール実行）を決定し、
      機能領域ごとのマッピングを作成する — 2026-10-02対応: 上記「役割分担マッピング」表を作成
- [ ] GCPサービスアカウント認証情報を各環境の `.env` / `.env.stg` / `.env.prod` の `GCS_*` に設定する
      （本ドキュメント・関連実装は設定済みを前提に進めているが、実際の値の設定はユーザー側作業）
- [ ] Cloud SQLインスタンスを作成し、VPS・GCP双方からの接続情報（接続名・IP許可・認証情報）を
      `.env`に設定する — 2026-10-01対応: 接続実装（`src/db/cloud_sql.py`の
      `get_cloud_sql_engine()`、`src/db/session.py`の`KEIBA_DB_BACKEND=cloud_sql`分岐、
      `tests/db/test_cloud_sql.py`）は完了。実インスタンスの作成・`.env`への接続情報設定は
      ユーザー側作業として残る

### VPS側（サービング）のTODO
- [ ] `admin-ops.md`・`scraping-queue.md`等に残る「VPS上の軽量ジョブ」（disk-cache-cleanup・
      logs-retention等）以外のdaemon thread・cronをGCP側へ切り出した後、不要になった
      VPS側のコード・crontabエントリを削除する
- [x] モデル配信: GCSから最新モデル（`models/keiba_model.pkl`相当）を定期取得・反映する
      仕組みを実装する（新規） — 2026-10-01対応: `src/pipeline/models/model_sync.py`に
      `sync_latest_model_from_gcs()`（GCS側が新しい場合のみダウンロード。GCS無効・エラー時は
      例外を出さずFalseを返す）と、学習側（GCP側）用の`upload_model_to_gcs()`を実装。
      VPS側で手動または定期実行するための開発者専用エンドポイント
      `POST /api/admin/model/sync-from-gcs`（`src/api/app.py`）を追加し、
      `/api/train`の学習完了時（`_run_training()`）に`upload_model_to_gcs()`の呼び出しを追加
      （失敗してもログ警告のみで学習処理自体は失敗させない）。
      テスト: `tests/pipeline/test_model_sync.py`（`google.cloud.storage.Client`をモック）
- [ ] スクレイピング・予測実行の手動トリガー系エンドポイント（`/api/scrape-trigger`・
      `POST .../predict`等）を、GCP側Cloud Run Jobs/Cloud Tasksを呼び出すプロキシに変更する

### GCP側（スクレイピング・ML・スケジュール実行）のTODO
- [x] `job_queue.py`のキュー基盤をCloud Tasks + Cloud Run Jobsへ移行する
      — 2026-10-01対応: ジョブ投入経路を環境変数で分岐する最小実装（ローカルJSONキューと
      排他構成）。新規`src/scraper/cloud_tasks_queue.py`の`enqueue_via_cloud_tasks()`
      （`google.cloud.tasks_v2.CloudTasksClient`、`GCP_PROJECT_ID`/`CLOUD_TASKS_QUEUE`/
      `CLOUD_TASKS_LOCATION`/`CLOUD_RUN_JOBS_WORKER_URL`を使用）と`is_cloud_tasks_backend_enabled()`
      （`KEIBA_QUEUE_BACKEND=cloud_tasks`判定）を実装し、`ScrapeJobQueue.add_job`の冒頭で
      分岐（未設定時は従来通りローカルJSON+ファイルロック、VPS側は無変更）。Cloud Run Jobs/
      サービス側のPushワーカーは新規`POST /api/internal/cloud-tasks/process-job`
      （`src/api/app.py`）が受け、既存の`src.scraper.queue_tasks.execute_job`+`ScraperRunner`
      をそのまま呼ぶ（新規の実行ロジックは書いていない）。OIDC検証は
      `KEIBA_CLOUD_TASKS_VERIFY_OIDC=1`時のみ有効化する簡易スタブ（既定は無効）。
      `requirements.txt`に`google-cloud-tasks>=2.0.0`を追加。実GCP接続は本環境に
      認証ファイルが無いためモックで検証（`tests/scraper/test_cloud_tasks_queue.py`、
      `tests/api/test_cloud_tasks_internal_endpoint.py`）。
      `python3 -m pytest tests/ --ignore=tests/scraper/manual --ignore=tests/research/manual`
      で既存506件+新規17件が全てpass。netkeibaのoutbound IP固定化は下記の別TODOとして残る
- [x] `src/api/app.py`内の重いdaemon thread（`_scheduler_loop`・`_weekly_sire_agg_loop`・
      `_daily_shutuba_enqueue_loop`等）をCloud Scheduler + Cloud Run Jobsへ分離する
      — 2026-10-01対応: 上記3種に加え、`run_hourly_queue_maintenance`・レース質日次一括推定・
      追走難度precompute・track-speedベースライン再構築・ミオスタチン再計算・オッズ予測モデル
      学習を含む計10ジョブについて、Cloud Scheduler + Cloud Run Jobsから起動できる CLI
      エントリポイント（既存コマンドがあるものはそれを採用、無いものは新規
      `src/scripts/cloud_jobs/`パッケージを追加）を整備した。各ジョブの実行コマンド・頻度・
      リソース目安・スケジューラ登録コマンド例は
      [`docs/operations/gcp-cloud-run-jobs.md`](./gcp-cloud-run-jobs.md)、デプロイ設計図は
      [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../scripts/gcp/deploy_cloud_run_jobs.sh)。
      daemon thread 自体のコード削除・実際のGCPデプロイ・スケジューラ登録はこの対応では
      行っていない（CLI整備のみ。削除は上記VPS側TODO「不要になったVPS側のコード・crontab
      エントリを削除する」で別途対応）
- [ ] モデル学習・アンサンブル・バックテスト・バックフィルをCloud Run Jobsで実行する構成に移行する
- [ ] netkeibaスクレイピングのoutbound IP固定化（Serverless VPC Connector + Cloud NAT、
      またはCompute Engineの予約静的IP）を設定する
- [ ] 各バッチ処理のログをCloud Loggingへ出力する

## メモ
