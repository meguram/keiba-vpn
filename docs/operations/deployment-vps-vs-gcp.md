# デプロイ戦略: VPS運用 vs GCP運用（コスト最適なサービス選定）

**対象領域**: デプロイ・インフラ全体（cron・常駐プロセス・ストレージ・ネットワーク公開）
**関連ドキュメント**: [service-endpoints.md](./service-endpoints.md)・[server_architecture.html](./server_architecture.html)・
[../git_management/todo/admin-ops.md](../git_management/todo/admin-ops.md) ほか各TODOファイル
**最終更新**: 2026-10-01

**位置づけ**: 2026-10-01時点ではVPS運用が確定方針。GCPへの移行は「場合によっては」の
検討段階であり、まだ意思決定していない。本ファイルは移行するかどうかを決めるための
整理用メモであり、GCP側のTODOは**移行が決まってから着手する**。VPS側のTODOは現行方針の
まま改善を続けてよい。**GCP = Cloud Runに限定しない**。本ワークロードの特性
（常時起動の daemon thread・OS cron・安定した送信元IPが必要なスクレイピング）を踏まえ、
Compute Engineでのリフト&シフトも含めてコスト最適な組み合わせを検討する。

<!--
  このファイルの構成は docs/git_management/todo/*.md と揃えている
  （現状の実装／ベストプラクティス／比較表／TODO）。
  コードや運用方針・GCPの価格体系が変わったら「現状」「比較表」も合わせて更新する。
-->

## 現状（2026-10-01調査、VPS運用の実態）

- エントリ: `main.py` が uvicorn で `src.api.app:app`（FastAPI, :8000）を起動。
  `scripts/server/service_start.sh` が dev/stg/prod を一括起動する入口（プロファイルは
  `service_start.profiles.sh`）。prodプロファイルは「将来VPSへクローン想定」のコメントあり。
- **Flask `/api/v1`(:5000) が仕様上の正、FastAPI(:8000)はJinja2 UI・管理画面・レガシーREST
  で段階的廃止予定**（`docs/operations/service-endpoints.md` の DEC-013）。
- 定期実行はOS crontab（`scripts/cron/setup_all_cron.sh`が生成。watchdog・ログローテ・
  daily-race-lists・raceday系・weekly-update・GCS→Postgres同期・backfill等、十数エントリ）。
- 加えて `src/api/app.py` 内に **daemon thread が7種以上**（`_scheduler_loop`構造チェック・
  `_weekly_sire_agg_loop`・`_disk_cache_cleanup_loop`・`_queue_hourly_maintain_loop`・
  `_logs_retention_loop`・`_daily_shutuba_enqueue_loop`・キューワーカー `_queue_slot_worker`等）。
  これらは**FastAPIプロセスが継続起動していることが前提**で、プロセスが落ちると停止する
  （`*/3 * * * *` の watchdog cronがプロセスダウンを検知して再起動する役割）。
- スクレイピングキューはローカルJSONファイル＋ファイルロック（`src/scraper/job_queue.py`、
  `_LOCK_TIMEOUT=1800秒`）で、常駐ワーカースレッドが継続処理する設計。
- netkeibaスクレイピングは`NETKEIBA_MAX_CONCURRENT_REQUESTS=1`・低頻度リクエスト間隔
  （1.0〜4.0秒）という低負荷設計で、**単一の安定した送信元IPからの低頻度アクセス**を
  前提にしていると見られる。
- ストレージはGCSを唯一のsource of truthとする設計（`HybridStorage`）。
  `.env.example`に「本番サーバー側の`.env`にGCS認証情報を設定、ローカル開発では
  `KEIBA_GCS_ENABLED=false`推奨」と明記 — **GCS実利用はVPS本番のみ。つまりストレージ層は
  既にGCP（GCS）そのもので、「VPS vs GCP」は実質的にはコンピュート層とDB/キャッシュ層の
  話に限られる**。
- ログは`logs/`配下のローカルファイル出力のみ（GCSやクラウドログへの転送機構は無し）。
  `/server-logs`・`/api/admin/server-logs`はこのローカルファイルを読む前提。
- PostgreSQL・RedisはいずれもDocker Compose（`docker-compose.dev.yml`）のローカルコンテナ
  （マネージドサービスではない）。
- MLflowも`mlflow/server/docker-compose.yml`でTracking(SQLite backend)・モデル別Serving・
  nginxリバースプロキシを同一ホスト上にDocker Composeで構成。
- 認証はIP制限ではなくHMAC署名付きクッキー（`src/api/auth.py`、`DEV_SECRET_KEY`+`DEV_PASSWORD`）。
  **VPN接続前提の設計は見当たらない**。外部公開は開発用途では`tcpexposer`という逆トンネル
  サービス（SSH鍵ベース）を使っている（本番VPSでの公開方法は追加調査が必要）。
- PID管理（`.server.pid`・`.monitor.pid`等）もローカルファイルシステム前提。

## このワークロードの特性（GCPサービス選定の前提）

GCPのどのサービスがコスト最適かは、ワークロードの形状に強く依存する。本プロジェクトは
以下の特性を持つ:

- **常時バックグラウンド処理が主体**（cron・daemon thread・スクレイピングキューが24時間
  継続稼働）。ユーザーからの同時アクセスはスパイクせず、むしろ裏側の定期処理の方が負荷の
  大半を占める。
- **スクレイピング対象（netkeiba）がレート制限・ブロックに敏感**で、安定した単一の送信元IP
  からの低頻度アクセスを前提にした設計になっている。
- **単一運用者規模**（個人〜小規模チーム）で、マネージドサービスの運用負荷軽減より
  月額コストの最小化が優先度として高いと推測される。

→ この形状は「リクエスト駆動でスパイクし、アイドル時は$0にしたい」というCloud Runが
得意な領域とは異なり、**「ほぼ24時間稼働し続ける1プロセス」**というCompute Engine
（通常のVMインスタンス）が得意な領域に近い。

## VPSで運用する場合のベストプラクティス（現行方針を続ける場合の改善点）

- OS crontab・daemon thread・ローカルディスクを前提にした現在の設計はVPSでは自然であり、
  大きな再設計は不要。改善の余地があるのは可観測性・耐障害性の運用面:
  - Postgres/Redisのバックアップ・リストア手順を明文化する（現状Docker Composeの
    ローカルコンテナのみで、スナップショット運用の有無が未確認）
  - watchdog（3分毎のプロセス死活監視）に加えて、ディスク容量・メモリ（`KEIBA_PROFILE=vps`
    の省メモリ設定が機能しているか）のアラートを整備する
  - 開発用トンネル（tcpexposer）を本番公開でも使っていないか再確認し、本番は
    nginx等の恒久的なリバースプロキシ＋TLSに統一する
  - admin-ops.mdに記録済みの既知の課題（`git-pull`表示のstale化、OS crontab系への
    Slack通知未対応、SSH runbook不在）を順次解消する

## GCPで運用する場合のベストプラクティス（コスト最適な組み合わせを検討）

GCPには複数のコンピュート選択肢があり、コスト最適解はワークロードの形状で変わる。
Cloud Run前提で全面再設計するのが常に正解ではないため、以下を比較候補とする。

### 選択肢A: Compute Engine（VMでのリフト&シフト）

現在のVPSとほぼ同じ構成（1台の常時起動VM上でFastAPI/Flask・daemon thread・OS crontab・
Docker Compose上のPostgres/Redis/MLflowをそのまま稼働）をGCP上のVMで再現する案。

- **メリット**: アーキテクチャの再設計がほぼ不要（daemon thread・ファイルロック式キュー・
  OS crontab・Docker Composeすべてそのまま動く）。移行コスト・リスクが最小。
  静的外部IP（予約IP、時間課金数円/時間程度）を割り当てればスクレイピングの送信元IP安定性も
  VPSと同様に確保できる。
- **コスト最適化の余地**: e2-small/e2-medium等の小型シェアードコアVMで十分
  （既存の`KEIBA_PROFILE=vps`の省メモリ設定がそのまま活きる）。1年/3年の委託利用割引
  （CUD）や自動適用される継続利用割引（SUD）でさらに下げられる。日次バックフィル等の
  中断可能なバッチだけをSpot VM（通常の数割〜最大9割引、プリエンプション前提）に
  切り出すことも検討できる。
- PostgreSQL・Redis・MLflowは同一VM上のDocker Composeのまま運用を継続すれば、
  マネージドサービス（Cloud SQL・Memorystore）の最低利用料がかからず、現状のVPSと
  同程度のコスト感を維持できる。
- ログはCloud Logging用の**Ops Agent**をVMに入れるだけで、アプリ側のコード変更なしに
  ローカルログをCloud Loggingへ転送できる（少量ログなら無料枠内に収まりやすい）。
- 定期実行は、OS crontabのままでも良いが、実行状況の一覧性を上げたい場合のみ
  Cloud Scheduler（ジョブ単価が非常に安い）からVM上のHTTPエンドポイントを叩く形に
  一部切り替えることも低コストで可能（VM自体の構成変更は不要）。

### 選択肢B: Cloud Run（サービス/ジョブ）によるサーバーレス分解

- **メリット**: リクエストが無い間は$0（スケールtoゼロ）。負荷がスパイク的で、
  アイドル時間が長いワークロードには有利。
- **本プロジェクトでの制約**: 常時稼働が必要なdaemon thread・ファイルロック式キュー・
  ローカルログ・ローカルSQLite(MLflow)はCloud Runのステートレス・マルチインスタンス
  モデルと相性が悪く、以下の再設計コストが発生する:
  - daemon thread群 → Cloud Scheduler + Cloud Run Jobsへの分離（設計コスト）
  - `job_queue.py`のファイルロック式キュー → Cloud Tasks等への置き換え（設計コスト）
  - スクレイピングの送信元IP安定化 → Serverless VPC Connector + Cloud NAT
    （固定費が時間課金で発生。概算で月数千円〜。VMの予約IPより高くなりやすい）
  - ログ → Cloud Loggingへの統一自体は無料枠内で収まりやすいが、`/server-logs`実装の
    変更が必要
  - PostgreSQL/Redis/MLflowを自己ホストし続けられないため、Cloud SQL・Memorystore等の
    マネージドサービスへ移行する必要があり、低トラフィックでも**最低利用料金**が発生する
    （自己ホストなら実質0円増のところ、月数千円〜のベースコストが乗る）
- **結論**: 24時間動くdaemon thread・cron・安定IP必須のスクレイピングという現在の
  ワークロード形状では、Cloud Run化によるスケールtoゼロの恩恵よりも、Cloud NAT・
  Cloud Tasks・Cloud SQL・Memorystoreの最低利用料金の方が上回りやすく、**トータルコストでは
  Compute Engineより高くなる可能性が高い**。採用するなら、常時稼働部分はVM/選択肢Aに残し、
  バックフィル等の「スパイクして終わる」バッチ処理だけをCloud Run Jobsに切り出す
  **ハイブリッド構成**が現実的。

### 選択肢C: GKE（Kubernetes）

単一アプリ・単一運用者規模ではオーケストレーションの運用負荷・クラスタコストに対して
得られるメリットが小さく、コスト最適の観点では現時点では推奨しない
（将来、複数サービスへの分割・水平スケールが本格的に必要になった場合に再検討）。

### データ層（コスト視点の補足）

- **GCS**: 既に本番で利用中。追加の移行コストは無い。
- **Cloud SQL / Memorystore for Redis**: 低トラフィックな単一アプリでは、自己ホスト
  （VM上のDocker Compose）に比べて最低利用料金分だけ割高になりやすい。信頼性・自動バックアップ
  等の運用負荷軽減とのトレードオフとして、コスト最優先なら自己ホスト継続、運用負荷軽減を
  優先するなら移行、という判断軸になる。
- **Vertex AI**（モデル学習・推論エンドポイント）: 現在MLflowで自己ホストしている学習・
  配信をVertex AIに置き換えると、常時稼働エンドポイントや学習ジョブのノード時間課金が
  発生する。AutoMLや大規模分散学習等の明確な必要性が出るまでは、コスト最適の観点では
  現状のMLflow自己ホスト継続が有利。

## 比較表

| 項目 | VPS（現状） | GCP: Compute Engine（推奨候補） | GCP: Cloud Run（サーバーレス分解） |
|---|---|---|---|
| アーキテクチャ変更 | 不要 | ほぼ不要（リフト&シフト） | 大（daemon thread・キュー・ログ・DBの再設計） |
| 常駐daemon thread | そのまま動く | そのまま動く | 不可。Cloud Scheduler+Jobsへ分離必須 |
| 定期実行 | OS crontab | OS crontabのまま、または低コストでCloud Scheduler併用可 | Cloud Scheduler必須 |
| スクレイピングキュー | ローカルJSON+ファイルロック | そのまま動く | 要再設計（Cloud Tasks等） |
| スクレイピングIP安定性 | 固定IP前提、低コスト | 予約静的IPで同等に確保、低コスト | Cloud NAT必須、月数千円〜の固定費 |
| ログ | ローカル`logs/*.log` | Ops AgentでCloud Loggingへ転送可（コード変更不要） | Cloud Logging必須、ビューア実装も変更必要 |
| PostgreSQL/Redis/MLflow | 自己ホスト（Docker Compose） | 自己ホスト継続可（コスト最小） | マネージド移行必須（最低利用料金が乗る） |
| 月額コストの傾向 | VPSプラン料金 | 小型VM1台分（CUD/SUDで圧縮可）とほぼ同等 | 常時稼働ワークロードには割高になりやすい |
| 向いているワークロード形状 | 24時間稼働の単一プロセス | 24時間稼働の単一プロセス | リクエストがスパイクしアイドル時間が長い処理 |

## TODO（手動追記用）

### 方針確定前の共通TODO
- [ ] VPS継続かGCP移行かを、いつまでに・何を基準に決定するか決める
      （コスト・スクレイピングのブロックリスク・運用負荷・移行作業コストが主な論点になりそう）
- [ ] GCPへ移行する場合、Compute Engine（リフト&シフト）とCloud Run（サーバーレス分解）の
      実コスト見積り（想定リクエスト数・常時稼働時間・ログ量等を入れたGCP Pricing Calculator
      ベースの比較）を作成し、本ドキュメントの「比較表」の推測を実数で裏付ける
- [ ] 本番VPSの外部公開方法（tcpexposerが本番でも使われていないか、nginx等の恒久構成か）を確認する

### VPS継続する場合のTODO
- [ ] PostgreSQL/Redisのバックアップ・リストア手順を整備する
- [ ] `docs/git_management/todo/admin-ops.md`に記録済みの既知の課題
      （git-pull表示のstale化・OS crontab系へのSlack通知未対応・SSH runbook不在）を順次解消する

### GCPへ移行する場合のTODO（移行決定後に着手）
- [ ] まずCompute Engineでのリフト&シフト（アーキテクチャ変更最小）を優先候補として検証する
      （上記「選択肢A」）。Cloud Runへの全面移行は、常時稼働部分の再設計コストと
      Cloud NAT等の固定費がリフト&シフトのVMコストを上回る可能性が高いため、
      バックフィル等のスパイク的バッチ処理のみを対象にした部分採用（ハイブリッド構成）を
      優先検討する
- [ ] （Compute Engine採用時）予約静的IPでスクレイピングの送信元IP安定性を確保する
- [ ] （Compute Engine採用時）Ops AgentでCloud Loggingへのログ転送を設定する
      （アプリ側コード変更なしで導入できる想定）
- [ ] （Cloud Run系を一部採用する場合）`job_queue.py`のキュー基盤をCloud Tasks等へ
      置き換える設計を作る
- [ ] （Cloud Run系を一部採用する場合）`src/api/app.py`内の対象daemon threadを
      Cloud Scheduler + Cloud Run Jobsへ分離する設計を作る
- [ ] PostgreSQL/Redis/MLflowを自己ホスト継続するかCloud SQL/Memorystore/Vertex AIへ
      移行するかを、コスト（最低利用料金）と運用負荷軽減のトレードオフで判断する
- [ ] 認証をHMACクッキー+tcpexposerから、採用するコンピュート選択肢に応じた
      GCP標準機構（IAP等）へ移行するか検討する

## メモ
