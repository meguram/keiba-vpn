# CI/CD設計: VPS = 必要最低限のファイルだけ / GCP = バッチイメージ

**対象領域**: CI/CD・デプロイパイプライン
**関連ドキュメント**: [deployment-vps-vs-gcp.md](./deployment-vps-vs-gcp.md)（役割分担の決定）・
[gcp-cloud-run-jobs.md](./gcp-cloud-run-jobs.md)（Cloud Run Jobs 10ジョブの詳細）・
[service-endpoints.md](./service-endpoints.md)
**最終更新**: 2026-10-02

## 設計意図: なぜDockerイメージ化が「VPSに必要最低限のファイルだけ」を実現するか

[deployment-vps-vs-gcp.md](./deployment-vps-vs-gcp.md)で、ConoHa VPS（2GB・軽量シェアードコア）は
「ページ公開・ユーザー向けAPIサービング専用」に限定する方針が決まっている。しかし従来の
`scripts/cron/git_pull_hourly.sh`のような「VPS上でgit pullし、チェックアウトされたソースを
そのままPythonで直接実行する」運用では、VPS上に以下がすべて必要になってしまう。

- `notebooks/`（Jupyter専用、サービングには無関係）
- `docs/`（ドキュメント、実行時には不要）
- `tests/`（テストコード、実行時には不要）
- `scripts/`（cron・運用シェルスクリプト一式。サービングに不要なものも多数含む）
- `src/scraper/`配下のスクレイピング実行コード本体・`src/pipeline/`配下の学習コード
  （重い依存関係・重い処理そのもの。サービングでは呼ばれない）
- フルの`.git`履歴・`requirements.txt`の学習/スクレイピング系パッケージ一式

これらを置かなければ動かない状態は「必要最低限」とは言えず、しかもgitチェックオペレーション
自体がVPSの軽量リソースを消費する。

**解決策**: サービングに必要なものだけを含む軽量Dockerイメージ（[`Dockerfile.serving`](../../Dockerfile.serving)）
を**CI側（GitHub Actions）でビルド**し、GHCR（GitHub Container Registry）にpushする。
VPSは`docker compose pull`でこの**完成済みイメージ**を取得するだけでよく、ソースコードの
チェックアウトもビルドも一切発生しない。VPS上に実際に置く必要があるのは次の3種類のみ:

1. [`docker-compose.serving.yml`](../../docker-compose.serving.yml)
2. `.env`（本番用の環境変数。`.env.example`を元にVPS側で作成）
3. `config/gcp-service-account.json`（またはenv別の`config/gcp-service-account.<env>.json`。
   GCS/Cloud SQL/Cloud Tasks等への疎通に必要なサービスアカウント鍵。`.gitignore`済み）

イメージの中身は`main.py`・`src/`・`templates/`・`static/`・`requirements.txt`のみで、
`notebooks/`・`docs/`・`tests/`・`scripts/`・`data/`・`mlflow/`は含まれない
（`Dockerfile.serving.dockerignore`でビルドコンテキストからも除外）。
同じ理由でGCP側のバッチ実行用イメージも[`Dockerfile.gcp-jobs`](../../Dockerfile.gcp-jobs)として
分離している（サービング用イメージと混在させない）。

### `src/`を分割せず全体を含めている理由（既知の制約）

当初の想定では「`src/scraper/`のスクレイピング実行コード本体（`run.py`・`job_queue.py`等）は
サービングイメージから除外する」ことを目指した。実際に`src/api/app.py`を`grep`で調査したところ、
開発者専用の管理画面・手動トリガー系エンドポイント
（`/api/scrape-trigger`・backfill・キュー管理・cushion/baba live取得・構造チェック等、
`src/monitor/`相当の管理機能がFastAPI側に実装されている）が、関数内の遅延import
（`from src.scraper.run import ScraperRunner`等）で`src.scraper`パッケージの大半
（`run`・`job_queue`・`backfill`・`client`・`auto_scrape`・`queue_tasks`・`period_runners`・
`missing_races`・`monitor_backlog`・`monitor_future_eligible`・`netkeiba_top_race_list`・
`date_coverage`・`structure_monitor`・`jra_cushion*`・`jra_baba_live`・`html_archive`・
`scrape_access_pause`・`scrape_policy`・`verify_scrape_completeness`等）を直接参照していた。
同様に`src.research`（`pedigree`/`race`/`genes`配下）・`src.pipeline`
（`features`/`models`/`inference`/`mlflow`）・`src.db`・`src.scripts.maintenance`も
`app.py`から広く遅延importされている。

これらを安全に取り除くには、どのエンドポイントが実際にVPSから呼ばれ得るか
（dev-only管理画面を含む）を1つずつ洗い出し、不要なものをエンドポイント単位で削除・
あるいはプロキシ化（GCP Cloud Run Jobs呼び出しへの置き換え）する必要があり、本対応の
スコープを超える。安全重視のため、**現時点では`src/`全体をサービングイメージに含める**
選択をした（`Dockerfile.serving`内にコメントで明記）。

**将来整理の余地**: [deployment-vps-vs-gcp.md](./deployment-vps-vs-gcp.md)のVPS側TODO
「スクレイピング・予測実行の手動トリガー系エンドポイントを、GCP側Cloud Run Jobs/Cloud Tasksを
呼び出すプロキシに変更する」が完了すれば、`src.scraper`の実行コード本体への直接importが
app.py側から無くなり、サービングイメージから安全に除外できるようになる見込み。

### `requirements.txt`を分割していない理由

タスク方針上、全量インストールで構わないと判断した。`lightgbm`/`xgboost`/`catboost`/`optuna`
（学習系）・`pymupdf`/`playwright`（スクレイピング系）はサービングには本来不要だが、
`src/`全体を含める上記の判断と合わせて、まずは「確実に動く」ことを優先し全量をインストールする。
**将来的な最適化の余地**: `requirements-serving.txt`（サービングに必要な最小集合）を切り出し、
`Dockerfile.serving`側だけこれを使うようにすればイメージサイズ・ビルド時間を削減できる。

## VPSに実際に必要になるファイル一覧

VPS上の作業ディレクトリ（例: `~/keiba-vpn-serving/`）に置くのは次のみ:

```
~/keiba-vpn-serving/
├── docker-compose.serving.yml
├── .env
└── config/
    └── gcp-service-account.json   (または .<env>.json。.gitignore済み・別途配置)
```

`git clone`やフルのソースチェックアウトは不要。`docker-compose.serving.yml`の`api`サービスは
GHCR上の既成イメージ（`KEIBA_SERVING_IMAGE`環境変数で指定、既定は
`ghcr.io/<owner>/<repo>-serving:latest`）を`pull`するだけで、ソース・依存関係のビルドは
CI側（GitHub Actions）で完了済みのものを使う。

## GitHub Secrets 一覧（今後ユーザー側で設定する必要があるもの）

本対応ではワークフローファイル自体の作成のみを行っており、以下のSecretsはいずれも未設定の
前提である。設定しない限り、`stg`/`master`へのpush時にジョブが（想定どおり）失敗する。

### VPS用（[`deploy-vps.yml`](../../.github/workflows/deploy-vps.yml)）

| Secret | 内容 |
|---|---|
| `VPS_SSH_HOST` | VPSのホスト名/IP |
| `VPS_SSH_USER` | SSHログインユーザー |
| `VPS_SSH_KEY` | SSH秘密鍵（PEM形式。`appleboy/ssh-action`が使用） |
| `VPS_SSH_PORT` | （任意）SSHポート。未設定時は22 |
| `VPS_APP_DIR` | （任意）VPS上の`docker-compose.serving.yml`配置先。未設定時は`~/keiba-vpn-serving` |

GHCRへのpush自体は`${{ secrets.GITHUB_TOKEN }}`（GitHub Actionsが自動付与）で認証するため、
追加のレジストリ契約・Secrets登録は不要。

### GCP用（[`deploy-gcp.yml`](../../.github/workflows/deploy-gcp.yml)）

| Secret | 内容 |
|---|---|
| `GCP_SA_KEY` | サービスアカウントJSONキー（JSON文字列そのまま。Workload Identity利用時は不要） |
| `GCP_PROJECT_ID` | GCPプロジェクトID |
| `GCP_REGION` | （任意）既定`asia-northeast1` |
| `GCP_RUN_SA_EMAIL` | （任意）Cloud Run Jobs実行用サービスアカウント |
| `GCP_SCHEDULER_SA_EMAIL` | （任意）Cloud Scheduler呼び出し用サービスアカウント |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | （Workload Identity利用時のみ、`GCP_SA_KEY`の代わり） |
| `GCP_SERVICE_ACCOUNT` | （Workload Identity利用時のみ） |

`GCP_SA_KEY`（鍵方式）とWorkload Identity連携はどちらか一方を選べばよい。鍵方式のほうが
設定は単純だが鍵の漏洩リスクがあり、Workload Identityのほうが安全だが初回セットアップ
（GCP側でのプロバイダ作成）が必要。`deploy-gcp.yml`は鍵方式をデフォルトにしており、
コメントでWorkload Identityへの切り替え方法も記載している。

### 既知の制約: Cloud Scheduler登録の再実行

[`scripts/gcp/deploy_cloud_run_jobs.sh`](../../scripts/gcp/deploy_cloud_run_jobs.sh)の
`--apply`は`gcloud scheduler jobs create http`を呼ぶ実装であり、同名のスケジューラジョブが
既に存在する場合は2回目以降の実行でエラーになる（upsert/update相当の分岐は未実装）。
`deploy-gcp.yml`は`stg`/`master`へのpushごとに全10ジョブへ`--apply`するため、初回成功後の
2回目以降のpushでは（Cloud Run Jobs自体の`deploy`は冪等だが）Cloud Scheduler側の
`create`が失敗する可能性がある。これは既存スクリプトの制約であり、本対応ではスクリプト自体
の変更は行っていない。恒久対応が必要な場合は、`deploy_cloud_run_jobs.sh`側に
「スケジューラジョブが既に存在する場合は`update`に切り替える」分岐を追加することを推奨する
（別TODOとして`deployment-vps-vs-gcp.md`のGCP側TODOに追記するか、本ファイルの更新時に対応）。

## 既存のgit pull方式からDocker方式への移行手順

既存の`scripts/cron/git_pull_hourly.sh`・`scripts/server/service_start.sh`等は
**変更・削除しない**（段階的移行の前提。両方式が併存してよい）。

1. VPS上に新しい作業ディレクトリ（例: `~/keiba-vpn-serving/`）を作る
   （既存の`git pull`運用のチェックアウト先ディレクトリとは別にする。混在させない）。
2. `docker-compose.serving.yml`を手動でこのディレクトリへ配置する
   （または最初の1回だけ`scp`/`git clone --depth=1`等で取得。以降はこのファイルの更新も
   CI側のデプロイで配布する運用に寄せることもできるが、本対応ではCIはイメージのpushのみ
   行い、`docker-compose.serving.yml`自体の配布は含めていない点に注意。初回・ファイル更新時は
   手動配置が必要）。
3. VPS上に`.env`（本番用）と`config/gcp-service-account.json`を配置する。
4. GitHub Secrets（上記一覧）を設定する。
5. `stg`または`master`ブランチへpushすると、`deploy-vps.yml`が自動的にイメージをビルドして
   GHCRへpush、VPSへSSHして`docker compose -f docker-compose.serving.yml pull && up -d`を
   実行する。
6. 動作確認後、既存の`git pull`方式で起動していたプロセス（`service_start.sh`経由のuvicorn等）
   を停止し、ポート競合（`:8000`）を避ける。両方式を同時に同じポートで動かすことはできない。
   移行完了まではどちらか一方のみを起動する運用とする。
7. 問題が無ければ、既存の`scripts/cron/git_pull_hourly.sh`のcrontab登録を外す
   （スクリプト自体は削除しない。将来的に再度ソース直接実行に戻す選択肢を残す）。

### 移行時の注意点

- Redisは`docker-compose.serving.yml`側で新規にコンテナ化される
  （既存`docker-compose.dev.yml`のRedisとポート`6379`が重複する場合は、どちらか一方を
  停止するか、ポートマッピングを変更する）。
- PostgreSQLは`docker-compose.serving.yml`に含めていない（Cloud SQL移行方針のため）。
  移行時点でまだCloud SQLへの切り替えが完了していない場合、`.env`の`DATABASE_URL`は
  既存の自己ホストPostgreSQLを指すままでよいが、その場合はVPS上に
  `docker-compose.dev.yml`のPostgreSQLコンテナ（またはそれに相当するもの）を別途
  起動したままにしておく必要がある。
- `GOOGLE_APPLICATION_CREDENTIALS`は`src.config.gcp_credentials.ensure_google_application_credentials()`
  が未設定時に`config/gcp-service-account.json`（またはenv別ファイル）を自動解決するため、
  `docker-compose.serving.yml`のボリュームマウント先パス（`/app/config/gcp-service-account.json`）
  とコンテナ内の既定探索パスが一致していることを確認する。

## GCP側（Cloud Run Jobs）のデプロイ

[`deploy-gcp.yml`](../../.github/workflows/deploy-gcp.yml)は[`Dockerfile.gcp-jobs`](../../Dockerfile.gcp-jobs)
から`src/scripts/cloud_jobs/`ほか既存バッチ用コードを含むイメージをビルドしArtifact Registryへ
push後、既存の[`scripts/gcp/deploy_cloud_run_jobs.sh`](../../scripts/gcp/deploy_cloud_run_jobs.sh)
を`list`で得たジョブ名ごとに`--apply`する（このスクリプトは`--apply`に単一ジョブ名の明示を
要求し、全件一括適用を安全上サポートしていないため）。10ジョブの詳細（頻度・リソース目安等）は
[gcp-cloud-run-jobs.md](./gcp-cloud-run-jobs.md)を参照。
