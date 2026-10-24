# API の Cloud Run 化（フェーズ 2）手順書

設計は [`architecture_target.md`](./architecture_target.md)（5.5 認証とネットワーク / 7.2 フェーズ 2）。本書は、リポジトリ側で用意できたものと、**ユーザーが GCP / VPS で行う作業の順序**をまとめる。dev PC は GCP に接続できない（`gcp_guard`）ので、実行は Cloud Shell かホスト PC から。

## 1. 用意したもの（リポジトリ側）

| 区分 | ファイル | 役割 |
|---|---|---|
| イメージ | `Dockerfile.cloudrun` / `Dockerfile.cloudrun.dockerignore` / `Dockerfile.cloudrun.gcloudignore` / `requirements-cloudrun.txt` | 1 ワーカー・読み取り専用の API。API が使わないライブラリ（torch・mlflow・optuna・playwright・ノートブック等）を外した最小の依存。起動直後の RSS は約 90MB（遅延 import のため。テストが重いライブラリを読んでいないことを確認） |
| エントリ | `src/api/cloudrun_entry.py` | 1 コンテナで FastAPI（`src/api/app.py`）+ Flask（`/api/v1/*`）+ `/ops/*` を振り分ける。**FastAPI の lifespan（取得キューのワーカー・各種スケジューラ）は動かさない**。`app.py` / `flask_app.py` は無変更 |
| 認証 | `src/api/middleware/proxy_key.py` | 共有ヘッダ `X-Keiba-Proxy-Key`。既定オフ（`KEIBA_PROXY_KEY_REQUIRED=1` のときだけ）。新旧 2 本（`KEIBA_PROXY_KEY` / `KEIBA_PROXY_KEY_NEXT`）で無停止ローテーション。定数時間比較、鍵が 1 本も無いときは全拒否。鍵が通った要求だけ `X-Keiba-Client-IP` を接続元として採用（anti_scraping が実クライアント IP で働く） |
| ハートビート | `src/ops/heartbeat_schema.py` / `heartbeat_store.py` / `heartbeat_api.py` | VPS が 5 分おきに送る JSON のスキーマ・HMAC 署名・保存（GCS `ops/heartbeat/latest.json`）・評価（architecture_target.md 3.2 の閾値）・Slack 通知（同じ異常は 6 時間に 1 回、復旧も通知） |
| 局所データ | `src/scripts/ops/serving_state.py` / `config/serving_state.json` | GCS に無い「ローカル専用データ」（race_lists・推論の保存物・めぐ指数の表など）を束にして GCS へ置き（push）、Cloud Run の起動時に取り込む（pull、sha256 検証・パス抜け出し拒否） |
| デプロイ | `scripts/gcp/setup_cloud_run_api.sh` / `deploy_cloud_run_api.sh` / `rollback_cloud_run_api.sh` | 土台の作成 / ビルド・デプロイ・切替 / 1 つ前に戻す。**既定は表示のみ**、`--apply` で実行 |
| 確認 | `src/scripts/ops/cloudrun_smoke.py` | 鍵の有無での 200/401、ハートビート状態、ホスト API との確率の一致比較 |
| VPS | `scripts/ops/vps/templates/nginx-keiba-ui-cloudrun.conf` / `switch_api_upstream.sh` / `render_nginx_proxykey.sh` / `heartbeat_client.py` / `templates/keiba-heartbeat.{service,timer}` / `vps_03_cloudrun_heartbeat.sh` | nginx の上流を host ⇄ Cloud Run で切替（失敗時は自動で直前の設定へ）、鍵の nginx include 生成、ハートビート送信（標準ライブラリのみ、`MemoryMax=48M`） |

デプロイ時の環境変数は `KEIBA_ENV=prod` と `KEIBA_RUN_LOCATION=gcp` を必ず入れる（スクリプトが入れる。Slack 通知に【PROD】・実行場所 GCP が付く）。`KEIBA_ENV` が未設定ならエントリが起動を拒否する。

## 2. Cloud Run で動かないもの・注意

- **取得キュー・定期処理は動かない**（lifespan を動かさない設計）。取得は VPS の実行役 / Jobs、定期処理は Scheduler + Jobs の担当。管理者用の `/api/scrape-queue/*` などの書き込み系は Cloud Run 上では使わない（ローカルファイルがインスタンスごとに消える）。ホストの監視ポータルを使う。
- **その場の推論（live compute）はしない前提**。予測は Jobs が作って GCS（`race_predictions`）に置く。ホストの shadow/バックフィルの出力（`data/local/inference/predictions_v3` など）は `serving_state` の `core` 群で届く。モデル・状態（`inference/state`、約 90MB）が要る場合だけ `live_compute` 群を取り込む（`KEIBA_SERVING_STATE_GROUPS=core,live_compute`）。
- **DB（PostgreSQL）は使わない**。`/api/v1/races` など DB 経由の経路は 503 `database unavailable` を返す（もともと廃止予定）。
- **ログインのセッション**は `DEV_SECRET_KEY` / `DEV_PASSWORD`（Secret Manager）で署名される。**ホストと同じ値**にすると、VPS から見たログイン状態が切替の前後で続く。
- anti_scraping のレート制限状態は**インスタンスごとのメモリ**（最大 3）。VPS の nginx と Cloudflare 側の制限が一次防御。
- メモリ: コンテナの書き込み層もメモリ扱い。`serving_state` の core は展開後で数百 MB 未満の見込み（`python -m src.scripts.ops.serving_state push --dry-run` で実測。上限は `config/serving_state.json` の `limits`）。ディスクキャッシュは書かない（`DISK_L2_MIN_WEEKLY_ACCESSES`）。

## 3. 手順（上から順に。◎ = ユーザー作業、◇ = 私がやれる確認）

### A. GCP の土台（◎、Cloud Shell で）
1. `PROJECT_ID=<id> GCS_BUCKET=<bucket> bash scripts/gcp/setup_cloud_run_api.sh` で内容を確認 → 問題なければ `--apply`。API の有効化・Artifact Registry（古いイメージの自動掃除つき）・実行用サービスアカウント（バケットの `objectAdmin` と、使う Secret の `secretAccessor` だけ）・Secret の器ができる。
2. 秘密の値を入れる（スクリプトの最後に出る案内のとおり。値をチャット・リポジトリ・シェル履歴に残さない）。`keiba-proxy-key` と `keiba-heartbeat-secret` は `openssl rand -hex 32` で新しく作る。Slack とログイン用の鍵はホストの `.env` の値を、画面に出さずに渡す。
3. 予算アラートの設定（architecture_target.md 6.2）。

### B. ローカル専用データを GCS へ（◎ ホストで）
4. `python -m src.scripts.ops.serving_state push --dry-run`（入るファイルとサイズ。GCS に触れない）→ 想定内なら `python -m src.scripts.ops.serving_state push --group core`。以後、予測生成の最後（または cron）で定期的に push する（Phase 3 で Jobs に組み込む）。

### C. ビルドとデプロイ（◎）
5. `bash scripts/gcp/deploy_cloud_run_api.sh all`（表示確認）→ `all --apply`。イメージを Cloud Build で作り、新リビジョンを**トラフィック 0%** で作る（初回だけは仕様上 100%）。最小インスタンス 0・最大 3・同時実行 30・タイムアウト 60 秒・メモリ 2GiB。
6. 出力されたタグ URL（`https://r<TAG>---keiba-api-….a.run.app`）で確認:
   `KEIBA_PROXY_KEY=<値> python -m src.scripts.ops.cloudrun_smoke --base <タグURL> --race-id <直近のレースID>`（鍵なし 401・鍵あり 200・ハートビート状態）。
7. ホストとの出力比較: 上に `--compare-base https://<API_INTERNAL_HOST> --compare-header CF-Access-Client-Id=CF_ID --compare-header CF-Access-Client-Secret=CF_SECRET` を足す（`CF_ID` / `CF_SECRET` は環境変数名。勝率・連対率・複勝率が馬番ごとに一致すること）。数レース・数日分で確認する。
8. 問題なければ `bash scripts/gcp/deploy_cloud_run_api.sh promote --apply`（最新リビジョンに 100%）。

### D. VPS の準備（◎ VPS で root。SSH はホスト経由）
9. `vps_03_cloudrun_heartbeat.sh` を実行（スクリプト・テンプレート・タイマーを入れる。設定が揃うまでタイマーは有効にならない）。
10. `/etc/keiba-ui/vars.env` に `CLOUDRUN_HOST=<keiba-api-….a.run.app>`（https:// なし）を追記。
11. 鍵を入れて nginx に反映:
    `printf '%s' '<keiba-proxy-key の値>' | sudo bash /opt/keiba-ui/bin/set_secret.sh /etc/keiba-ui/proxykey.env KEIBA_PROXY_KEY` → `sudo /opt/keiba-ui/bin/render_nginx_proxykey.sh`
12. ハートビート設定: `HEARTBEAT_URL=https://<Cloud Run のホスト>/ops/heartbeat` と `KEIBA_HEARTBEAT_SECRET` を `/etc/keiba-ui/heartbeat.env` に（`set_secret.sh` で。鍵は `keiba-heartbeat-secret` と同じ値）→ `sudo systemctl enable --now keiba-heartbeat.timer`。`python3 /opt/keiba-ui/bin/heartbeat_client.py --print` で採取内容を確認できる。
13. Cloud Scheduler にハートビート評価を登録（5 分おき。Scheduler は 6〜14 ジョブの予算内）:
    `gcloud scheduler jobs create http keiba-heartbeat-check --location asia-northeast1 --schedule "*/5 * * * *" --uri https://<Cloud Run のホスト>/ops/heartbeat/check --http-method POST --headers X-Keiba-Proxy-Key=<値>`
    （Scheduler のヘッダはジョブ設定に平文で残る。プロジェクトの閲覧権限者が限られている前提。気になる場合は `/ops/heartbeat/check` 専用の鍵を別に作る）

### E. 切替（◎ VPS）
14. `sudo /opt/keiba-ui/bin/switch_api_upstream.sh cloudrun`。切替前に Cloud Run の `/api/health` を確認し、切替後に `http://127.0.0.1:8080/api/health` が 200 でなければ**自動で直前の設定に戻す**。状態は `switch_api_upstream.sh status`。
15. https://app.megu-keiba.com（prod）で予測・期待値・推奨が出ること、stg との差が無いことを確認。VPS の `healthcheck.sh`（上流 API に届くか）も通る。
16. 1 週間安定したら、ホストの FastAPI を prod 向けには使わなくてよい（stg は引き続きホスト）。

## 4. ロールバック

| 状況 | 操作 | 時間 |
|---|---|---|
| 新リビジョンが悪い | `bash scripts/gcp/rollback_cloud_run_api.sh --apply`（1 つ前のリビジョンに 100%。`--to <rev>` で指定） | 数秒 |
| Cloud Run 全体が悪い | VPS で `sudo /opt/keiba-ui/bin/switch_api_upstream.sh host`（nginx の上流をホストの api-internal に戻す） | 数秒 |
| 鍵の入れ替え | Cloud Run の `KEIBA_PROXY_KEY_NEXT` に新しい鍵（`WITH_NEXT_KEY=1` で再デプロイ）→ VPS の鍵を新しい値にして `render_nginx_proxykey.sh` → 古い鍵を外す。ハートビート鍵も `_NEXT` で同様 | 無停止 |

## 5. 費用（`python -m src.scripts.ops.cost_estimate`、1 ドル = 150 円）

案 A（本構成）の API 分は **Cloud Run サービス 0〜157 円/月**（最小インスタンス 0 のため、無料枠にほぼ収まる）。Artifact Registry は 8〜30 円（掃除ポリシーで 3 世代 + 7 日）。ハートビート受信は約 8,640 リクエスト/月 + 評価 8,640 回/月で無料枠内。Scheduler は 1 ジョブ追加（45 円/月前後の粒度）。案 A 全体は **約 300〜2,160 円/月**。**最小インスタンス 1 にすると約 2,900 円/月増えて予算超過**（コールドスタートが気になっても 0 のままにする）。単価はまだ公式ページで未確認の前提を含む（`architecture_target.md` 付録 A）。デプロイ後の実績は日次のコスト報告（`gcp_daily_cost_report`）で確認する。

## 6. 未解決・次の作業

- コールドスタート時間の実測（serving_state の取り込み + 初回の GCS 読み）。重ければ「core の絞り込み」か「起動後に非同期で取り込み」に変える。
- `serving_state` の定期 push（予測生成の最後に呼ぶ）は Phase 3（Jobs 化）で組み込む。それまでは手動かホストの cron。
- Cloud Run の ingress を「VPS（固定 IP）だけ」に絞る手段は無料枠では無い（Cloud Armor は有料）。共有ヘッダ鍵 + 最大 3 インスタンスで暴走を抑える。鍵が漏れた場合は §4 のローテーション。
- anti_scraping の本番化（`enforce`）はホストと同様に 1〜2 週間の検知ログを見てから。
- 監視ポータル・取得キュー管理の Cloud Run 移行は Phase 5。
