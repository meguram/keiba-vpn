# 開発者監視ポータル 仕様書

> **対象者**: 開発者のみ。エンドユーザーには非公開。  
> **ポート**: `23900`（FastAPI / Flask / Next.js とは完全独立。番号の正は [`config/ports.env`](../config/ports.env)）

---

## 0. サーバ一覧（クリックで各サーバの画面へ）

> リンクは **dev（開発 PC）で起動中**のときに開ける。ブラウザは WSL と同じ PC で開くこと。
> ポート番号は [`config/ports.env`](../config/ports.env) が唯一の正。`KEIBA_PORT_*` 環境変数で上書きできる（一覧を変えたらこの節も更新）。
> 方針: 3000 / 5000 / 5432 / 6379 / 8000 / 9090 など他アプリが既定で使うポートを避け、ephemeral 範囲（32768 以上）の外の `23xxx` に集約。dev=`231xx` / stg=`232xx` / prod=`233xx` / 共通=`239xx`。

### dev（モック UI 確認用・IDE を開くと自動起動）

| サーバ | ポート | 開く（クリック） | 備考 |
|--------|--------|------------------|------|
| Next.js モック UI | 23100 | [http://127.0.0.1:23100/](http://127.0.0.1:23100/) | `NEXT_PUBLIC_MOCK=true`。ユーザー向け UI |
| Flask `/api/v1` | 23110 | [ヘルス](http://127.0.0.1:23110/api/v1/health) | ログイン等。Next.js から `/api/v1/*` を中継 |
| FastAPI（レガシー UI / API） | 23120 | [UI](http://127.0.0.1:23120/) ・ [ヘルス](http://127.0.0.1:23120/api/health) ・ [API 仕様 (/docs)](http://127.0.0.1:23120/docs) | `data/dev_mock/` を読む |
| 監視ポータル（本書） | 23900 | [ダッシュボード](http://127.0.0.1:23900/) | `127.0.0.1` のみ bind。要ログイン |
| MLflow Tracking | 23130 | [http://127.0.0.1:23130/](http://127.0.0.1:23130/) | 既定ではオフ（`START_MLFLOW_LOCAL=true` で起動） |

監視ポータルの各ページ: [ダッシュボード `/`](http://127.0.0.1:23900/) ・ [サービス `/services`](http://127.0.0.1:23900/services) ・ [スクレイピング `/scraping`](http://127.0.0.1:23900/scraping) ・ [Cron `/cron`](http://127.0.0.1:23900/cron) ・ [データ `/data`](http://127.0.0.1:23900/data) ・ [システム `/system`](http://127.0.0.1:23900/system) ・ [ログ `/logs`](http://127.0.0.1:23900/logs) ・ [Git `/git`](http://127.0.0.1:23900/git) ・ [データストア `/data-stores`](http://127.0.0.1:23900/data-stores) ・ [仕様書 `/specs`](http://127.0.0.1:23900/specs)

### stg / prod（該当環境で起動したときのみ有効）

| 環境 | Next.js | Flask | FastAPI | MLflow |
|------|---------|-------|---------|--------|
| stg | [23200](http://127.0.0.1:23200/) | [23210](http://127.0.0.1:23210/api/v1/health) | [23220](http://127.0.0.1:23220/api/health) | [23230](http://127.0.0.1:23230/) |
| prod | [23300](http://127.0.0.1:23300/) | [23310](http://127.0.0.1:23310/api/v1/health) | [23320](http://127.0.0.1:23320/api/health) | [23330](http://127.0.0.1:23330/) |

### 自動起動と手動操作

- **自動起動**: Claude Code（`.claude/settings.json` の SessionStart フック）と Cursor（`.cursor/hooks.json` の sessionStart）でプロジェクトを開くと [`scripts/server/autostart_dev.sh`](../scripts/server/autostart_dev.sh) が走り、上の dev 4 サーバのうち落ちているものを起動する（冪等・バックグラウンド・ログは `logs/autostart_dev.log`）。`KEIBA_ENV=dev` 以外では何もしない。無効化: `KEIBA_AUTOSTART_DEV=0`。
- **状態確認**: `bash scripts/server/autostart_dev.sh --status`
- **手動起動**: `bash scripts/server/autostart_dev.sh`（または `bash scripts/server/service_start.sh --env dev --mock --with-fastapi`）
- **対象外**: PostgreSQL（5432）/ Redis（6379）は `DATABASE_URL` / `REDIS_URL` で明示設定する基盤サービスのため標準ポートのまま。`docker-compose.serving.yml` と `mlflow/server/`（VPS デプロイ用スタック）も従来のポート。

---

## 1. 目的

FastAPI・Flask・Next.js のサービス死活、スクレイピングキューの状態、データカバレッジ、システムメトリクス、Git 状態を **一画面で確認・管理**するための開発者専用ポータル。

通常のユーザー向けUIとは別ポートで動作し、独立したパスワード認証（`MONITOR_PASSWORD`）で保護される。

---

## 2. アーキテクチャ

```
ブラウザ（開発者）
      │
      ▼
Monitor App :23900         （スタンドアロン Flask）
      │
      ├─► FastAPI :23120   （内部 HTTP。ポートは現在の KEIBA_ENV）
      ├─► Flask   :23110   （内部 HTTP）
      ├─► Next.js :23100   （内部 HTTP ヘルスチェック）
      ├─► psutil           （CPU / メモリ / ディスク）
      └─► subprocess(git)  （Git 状態）
```

- `src/monitor/app.py` は既存の `src/api/flask_app.py` とは完全に独立した Flask インスタンスを起動する。
- ポート 23900 は外部から直接アクセスしないこと（tcpexposer トンネルには含めない）。

---

## 3. 認証

| 項目 | 内容 |
|------|------|
| 認証方式 | パスワード認証（Cookie セッション） |
| 環境変数 | `MONITOR_PASSWORD`（ログインパスワード）、`MONITOR_SECRET_KEY`（Flask セッション署名キー） |
| セッション有効期間 | 12 時間（`SESSION_COOKIE_SAMESITE=Strict`）|
| 保護対象 | `/` 以下のすべてのルート（`/login`・静的ファイルを除く）|

---

## 4. ページ仕様

### 4.1 ダッシュボード `/`

**目的**: 全サービスの死活とキー指標を一覧表示する。

| 表示項目 | データ源 | 更新間隔 |
|----------|---------|---------|
| FastAPI / Flask / Next.js 死活 | 各 `/health` エンドポイント | 30 秒 |
| キュー件数（pending / running / failed） | FastAPI `/api/scrape-jobs` | 30 秒 |
| 直近エラー数（過去 1 時間） | FastAPI `/api/scrape-jobs` の `failed_at` | 30 秒 |
| Git HEAD コミット | `subprocess(git log -1)` | 30 秒 |
| システム概要（CPU/Mem） | psutil | 30 秒 |

### 4.2 サービス詳細 `/services`

| 表示項目 | データ源 |
|----------|---------|
| 各サービスの HTTP ステータスコード | 各 `/health` |
| レスポンスタイム（ms） | curl タイミング |
| FastAPI uptime / keiba_env | FastAPI `/api/health` レスポンス本文 |
| Flask version / environment | Flask `/api/v1/health` レスポンス本文 |

### 4.2b Cron `/cron`

全 cron ジョブについて、**状態（待機中／実行中）・次回予定・最終実行・結果（成功／警告／失敗／見送り）・失敗の原因**を表示する（30 秒ごとに自動更新。行を押すと内容・保存先・直近の出力・直近の失敗が開く）。

| 表示項目 | データ源 |
|----------|---------|
| ジョブ一覧・スケジュール | 実際の crontab（`crontab -l`）。名前・説明は `src/scripts/docs/build_cron_schedule_html.py` の `JOBS` |
| 実行中／待機中 | `data/local/meta/cron_runs/<job_id>.running`（pid の生死）。記録の無いジョブは `ps` |
| 最終実行・結果 | `cron_runs/<job_id>.tsv`（`scripts/cron/cron_run.sh` が cron の各行を包んで記録）。無ければログの START/END 行、それも無ければログの更新時刻のみ（結果は「記録なし」） |
| 失敗・警告・見送りの原因 | 出力の末尾と終了コードを `src/monitor/cron_failure_patterns.py` の辞書と照合（原因・直し方のヒント付き）。辞書に無い失敗は「原因を自動判別できません」＋最後のエラー行 |

- 新しい失敗パターンを見つけたら、辞書（`cron_failure_patterns.py`）に足す（テストは `tests/monitor/test_cron_failure_patterns.py`）。
- 実行記録を残すには cron の行を `cron_run.sh` 経由にする（`scripts/cron/setup_all_cron.sh install` で登録される）。API は `GET /api/internal/cron-status`。

### 4.3 スクレイピング `/scraping`

| 表示項目 | データ源 |
|----------|---------|
| キュー状態（pending / running / done / failed 件数） | FastAPI `/api/scrape-jobs` |
| カバレッジカレンダー（年別）| FastAPI `/api/coverage-calendar` |
| スキーマ検証失敗数 | FastAPI `/api/scrape-jobs` → `schema_validation_failures` |
| 直近失敗ジョブ一覧（最大 20 件）| FastAPI `/api/scrape-jobs` |

### 4.4 データ `/data`

| 表示項目 | データ源 |
|----------|---------|
| GCS 統計（バケット名・ enabled フラグ）| FastAPI `/api/health` |
| カバレッジカレンダー（当年）| FastAPI `/api/coverage-calendar` |
| 行データカバレッジサマリー | FastAPI `/api/row-data-coverage` |

### 4.5 システムメトリクス `/system`

| 表示項目 | データ源 | 更新間隔 |
|----------|---------|---------|
| CPU 使用率（%）| psutil | 15 秒 |
| メモリ使用率（%・使用 GB / 合計 GB）| psutil | 15 秒 |
| ディスク使用率（%・使用 GB / 合計 GB）| psutil | 15 秒 |
| スワップ使用率 | psutil | 15 秒 |
| 実行中プロセス一覧（keiba 関連）| psutil | 15 秒 |

### 4.6 ログ `/logs`

| 表示項目 | データ源 |
|----------|---------|
| ログファイル選択（FastAPI / Flask / Next.js / watchdog）| ローカル `logs/` ディレクトリ |
| 末尾 200 行表示 | Flask `/api/v1/admin/server-logs` または直接ファイル読み込み |

### 4.7 Git 状態 `/git`

| 表示項目 | データ源 |
|----------|---------|
| 最新コミット（hash・author・date・message）| `git log -1` |
| dirty ファイル一覧 | `git status --porcelain` |
| 最終 git pull 結果 | `logs/git_pull_hourly.log` 末尾 |
| ブランチ名 | `git rev-parse --abbrev-ref HEAD` |

### 4.8 データストア `/data-stores`

**目的**: PostgreSQL / SQLite（`bloodline.db`・MLflow tracking DB）/ Parquet 各データストアの状態を一覧表示する。

| 表示項目 | データ源 |
|----------|---------|
| PostgreSQL 接続状態・テーブル一覧・最終ETL時刻 | `/api/internal/data-stores` |
| SQLite 各DB（存在・サイズ・テーブル一覧） | 同上 |
| Parquet ブロック（存在・件数・サイズ） | 同上 |
| jockey_trainer_stats manifest | 同上 |

### 4.9 仕様書ビューア `/specs`, `/specs/<filename>`

**目的**: `docs/decisions/` 配下の Markdown 仕様書（MASTER / AREA-xx / DEC-xxx）を、開発機を離れず
ブラウザ上で閲覧する。

| 表示項目 | データ源 |
|----------|---------|
| ファイル一覧（MASTER / AREA / DEC に分類） | `docs/decisions/*.md` のディレクトリ列挙 |
| 選択したファイルのレンダリング結果（表・コード・目次対応） | `markdown` ライブラリで変換（`/specs/<filename>`） |

`filename` は `docs/decisions/` 配下への path traversal を防ぐため、解決後のパスが
`docs/decisions/` 配下であることをチェックしている。

### 4.x `/models` — モデルの予測性能（`/services` に概要カードも出る）

各モデル（アンサンブル v2〜v5 と shadow、MLflow 登録モデル）の予測性能を、バージョン切り替え・重ね描きで見る。**重い計算はこの画面でしない**: `python -m src.scripts.data.model_perf_cache build`（読み取り専用）が `data/local/meta/model_perf/`（`KEIBA_MODEL_PERF_DIR` で変更）に JSON を作り、画面はそれを読むだけ。キャッシュが無いバージョンは『未計算』と出る。重い処理なのでホストでは `/tmp/heavy_run.sh model-perf python -m src.scripts.data.model_perf_cache build` で流す（LightGBM の重要度を読むため）。

| セクション | 内容 | データ源 |
|---|---|---|
| バージョン選択・指標・確率の種類 | チップで複数選択（重ね描き）。指標は 勝率/連対/複勝 LogLoss・Top1 的中率・勝率 ECE | — |
| 一覧 | test 2025 の LogLoss（勝率・連対・複勝）・AUC・Brier・ECE・Top1 的中率・3 着内率、2026 本番の勝率 LogLoss とレース数、学習の終わり・行数・特徴量数・メンバー・追加群・MLflow 名 | `final_eval_vN/test2025_predictions.parquet`（再計算）、`final_vN/bundle*/fit_info.json` |
| ① 本番での推移（週別） | 2026 年の予測×実着順。バージョンごとに線。LogLoss は ±1.96 標準誤差、Top1 は Wilson 区間の帯。**30 レース未満の週は白抜き**で参考扱い。無い版・期間は線を引かず『データなし』 | `inference/predictions_backfill/<bundle>/<日>/*.parquet`、`features/target/rank_tbl/2026` |
| ② 学習・評価時点の推移（年別） | 学習時の年別検証（出力が残る版のみ）＋ test 2025 の点 | `ensemble_v2/ensemble_v2_fold_metrics.csv` 等 |
| 校正 | 予測確率の等頻度 10 区間の平均と実際の割合（対角線が理想）。ECE | test 2025 の予測 |
| 特徴量の寄与率 | LightGBM 系メンバーのゲイン重要度。上位特徴量の横棒と、群別（馬主・調教・血統…）の版間比較 | バンドルの `lgb_{win,top3}.txt`。群分けは `config/model_feature_groups.json`（編集可・（推定）は根拠が弱い） |
| MLflow の個別モデル | カタログの各モデルの run 数・直近の run・直近の指標 | MLflow（`mlflow/runs/mlflow.db`） |

- **2026 年はテストセット②**（学習にも選択にも使っていない）。画面にも明示する。2025 は v1〜v5 の選択に使い終わったテストセット①。
- **含めない**: オッズ生データ・個人情報・秘密。XGBoost・NN・線形メンバーの重要度（LightGBM 系のみ）。permutation / SHAP は未実装（重いので事前計算ジョブに足す余地）。
- 不足の見え方: shadow は日付ごとにモデルが変わりうるので別系列。v3 の 2026 本番（バックフィル）は無い。v5 のバックフィルは一部の日のみ。

---

### 4.y `/prod` — prod（ConoHa VPS）の状況（`/services` の最上部に要約カードも出る）

**目的**: 公開（prod）の UI は VPS で動くが、監視ポータルはホストで動くため、これまで prod の状況がどこにも出なかった（`ENV_CONFIGS["prod"]` はローカルの 233xx を見るだけ）。VPS と周辺（GCP）の状況を 1 ページに集める。

| カード | 内容 | データの出どころ |
|---|---|---|
| 全体の信号 | 緑 / 黄 / 赤 と理由 | 下の各セクションから計算（`prod_status.evaluate_overall`）|
| VPS の資源と UI | UI の応答・版（release / env / commit / source_digest）、VPS→API(api-internal)の疎通、メモリ（使用 600MB 以下 / MemAvailable 1GB 以上の目標に対するメーター）、swap、ディスク、load、稼働時間 | VPS の採取（`vps_status_probe.py`）|
| ユニットとトンネル | keiba-ui / nginx / cloudflared-prod / keiba-fetch-agent / 各 timer の状態・再起動（前回収集からの増分）・メモリ | 同上 |
| 公開 URL の入口 | `app.megu-keiba.com` が Cloudflare Access のログインへ 302 するか（トンネルが落ちていても 302 になるため、生死の判定は VPS 内部の採取で行う）| ホストから HTTP |
| 昇格履歴 | promote.sh の直近の結果 | `data/local/meta/vps_promote_history.jsonl` |
| VPS ハートビート | 最終受信・現在の異常。クライアント導入前は『未導入』 | GCS `ops/heartbeat/latest.json`（読み取り専用の gcloud）|
| Cloud Scheduler | 予約（smoke-auto-* など）の状態・最終実行 | 読み取り専用の gcloud |
| netkeiba 到達試験 | netkeiba-smoke の直近の実行と各回の合否 | 読み取り専用の gcloud + `deploy/gcp/netkeiba_smoke/summarize.py` |
| 影モード | fetch-shadow の日次レポート。導入前は『未導入』 | GCS `ops/fetch_shadow/daily/` |
| cutover 判定 | ホスト cron を止めてよいか（合格/不合格/未着手の件数）| `cron_cutover_status --json` |
| stg との差分 | env parity の保存結果。未導入なら『未導入』 | `data/local/meta/env_parity/latest.json` |
| GCP の費用 | 読み取り専用の鍵に BigQuery の権限が無いため未取得（Slack の日次コスト報告を参照）| — |

- **収集方式**: リクエスト時に ssh / gcloud を叩かない。ホストの cron（3 分おき）の `python -m src.scripts.ops.collect_prod_status` が `data/local/meta/prod_status/latest.json` に原子的に書き、画面はそれを読むだけ（`src/monitor/prod_status.py`）。取得失敗・未導入は『未取得（理由）』で表示し、各カードに取得時刻を出す。キャッシュが 15 分以上古いと全体の信号を黄にする。
- **VPS に何も入れない**: 採取は `ssh keiba-vps python3 -` に読み取りだけのスクリプトを標準入力で流す（deploy ユーザーで読める範囲。新しい権限・ファイル・鍵は不要）。
- **差し替え層**: 各セクションは `{state: ok|missing|error, at, reason, data}` の同じ形で、ソース別アダプタ（`vps_adapter` / `heartbeat_adapter` …）が作る。最終形（ホストと無関係に公開）では、VPS のハートビート（GCS）→ GCP 上のモニターに、同じ形で差し替える。
- **通知**: `KEIBA_PROD_STATUS_NOTIFY=1` のときだけ、赤・黄を Slack（`alert` チャンネル、環境表示つき）へ。同じ内容は 1 時間に 1 回まで。既定は送らない。
- **導入**: `sudo -n bash scripts/ops/install_prod_status_cron.sh dir` → `bash scripts/ops/install_prod_status_cron.sh cron`（jovyan の crontab に 3 分おき、タグ `KEIBA-PROD-STATUS`）。今すぐ 1 回: `… once`。外す: `… remove`。
- 目標値と閾値: 使用 600MB 以下・MemAvailable 1GB 以上（600MB 未満は赤）・swap 200MB 超・ディスク 80%/85%・ハートビートの途絶 15 分は `docs/operations/architecture_target.md` 3.2 と `src/ops/heartbeat_schema.py` に合わせてある。

## 5. ファイル構成

```
src/monitor/
  __init__.py          # 空（パッケージ化）
  app.py               # Flask スタンドアロン（認証・ルーティング・API集約）
  templates/
    base.html          # 共通レイアウト（ダークテーマ、ナビゲーション）
    login.html         # ログインフォーム
    dashboard.html     # / ダッシュボード
    services.html      # /services
    scraping.html      # /scraping
    cron.html          # /cron 定期ジョブの状態
    data.html          # /data
    models.html        # /models モデルの予測性能（キャッシュを読むだけ）
  model_perf.py        # /models 用: 事前計算キャッシュの読み出し（欠けても落ちない）
    system.html        # /system
    logs.html          # /logs
    git.html           # /git
    data_stores.html   # /data-stores
    specs.html         # /specs, /specs/<filename>

scripts/server/
  start_monitor.sh     # 単体起動スクリプト（port は config/ports.env）

docs/
  monitoring-portal.md # 本ファイル
```

---

## 6. 起動方法

### 単体起動

```bash
# 既定ポート（config/ports.env の KEIBA_PORT_MONITOR = 23900）
bash scripts/server/start_monitor.sh

# ポート・bind アドレス指定
bash scripts/server/start_monitor.sh --port 23999 --host 127.0.0.1
```

### 自動起動（落ちていたら起動 / コンテナ再起動後も）

`scripts/server/ensure_monitor.sh` は、監視ポータルが応答しなければ起動する冪等スクリプト（既定 `127.0.0.1` 待ち受け。ログ `logs/ensure_monitor.log`）。
cron から次のように呼ぶ（stg ホスト PC はこの 2 行を root の crontab に登録済み）:

```cron
@reboot sleep 30 && /path/to/keiba-vpn/scripts/server/ensure_monitor.sh   # KEIBA-VPN-MONITOR
*/5 * * * * /path/to/keiba-vpn/scripts/server/ensure_monitor.sh            # KEIBA-VPN-MONITOR
```

### service_start.sh との連携

```bash
# dev モードで全サービス + 監視ポータルを起動
bash scripts/server/service_start.sh --env dev --monitor

# stg モードで全サービス + 監視ポータルを起動
bash scripts/server/service_start.sh --env stg --monitor
```

---

## 7. 環境変数

`.env` / `.env.example` に以下を追記：

| 変数名 | 必須 | 説明 |
|--------|------|------|
| `MONITOR_PASSWORD` | 必須 | 監視ポータルのログインパスワード |
| `MONITOR_SECRET_KEY` | 必須 | Flask セッション署名キー（ランダム 32 文字以上推奨）|
| `KEIBA_PROD_STATUS_NOTIFY` | 任意 | `1` で prod の状況が赤・黄のとき Slack に通知（既定は送らない）|
| `KEIBA_PROD_STATUS_DIR` | 任意 | prod 状況のキャッシュ置き場（既定 `data/local/meta/prod_status`）|

既存の `DEV_PASSWORD` / `DEV_SECRET_KEY` とは別管理（目的・権限が異なるため）。

> `MONITOR_PASSWORD` が未設定だと空パスワードでログインできてしまう。dev の自動起動は `127.0.0.1` のみに bind するが、必ず `.env` に設定すること。

---

## 8. 内部 API エンドポイント（JS ポーリング用）

監視ポータルは以下の内部 JSON API をブラウザ JS から呼び出す：

| パス | 説明 |
|------|------|
| `GET /api/internal/status` | 全サービス死活 + キュー概要 + Git HEAD |
| `GET /api/internal/system` | psutil メトリクス（CPU/Mem/Disk/Swap）|
| `GET /api/internal/git` | Git 詳細（log / status / pull ログ）|
| `POST /api/internal/git/pull` | `git pull origin <branch>` を実行して結果を返す |
| `GET /api/internal/logs?file=<name>` | 指定ログファイルの末尾 200 行 |
| `GET /api/internal/scraping` | スクレイピング詳細（キュー + カバレッジ）|
| `GET /api/internal/coverage/month?year=&month=` | 年月指定のカレンダーデータ（日別ステータス）|
| `GET /api/internal/coverage/date-matrix?date=YYYYMMDD` | 指定日の race×category カバレッジマトリクス（FastAPI `/api/date-race-matrix` を中継）|
| `GET /api/internal/cron-status` | 全 cron ジョブの状態（待機中/実行中）・次回予定・最終実行・結果（成功/警告/失敗/見送り）・失敗の原因（`src/monitor/cron_failure_patterns.py` の辞書で判別）。画面は `/cron` |
| `GET /api/internal/page-quality?date=YYYY-MM-DD` | ページ品質チェック結果（省略時は最新）|
| `POST /api/internal/page-quality/run` | ページ品質チェックをオンデマンド実行 |
| `GET /api/internal/model-perf` | モデルの予測性能（事前計算キャッシュ。`?summary=1` で /services 用の要約）|
| `GET /api/internal/prod-status` | prod（VPS）の状況（収集キャッシュ。`?summary=1` で /services 用の要約）|
| `GET /api/internal/data-stores` | PostgreSQL / SQLite / Parquet データストア状態（`/data-stores` ページ用）|

---

## 9. セキュリティ上の注意

- 監視ポータルはローカル（`127.0.0.1`）専用で起動し、外部向けには公開しない。
- `MONITOR_PASSWORD` は `DEV_PASSWORD` と異なる値を設定すること。
- ログビューアは `logs/` ディレクトリのみにアクセス制限し、パストラバーサルを防止する。
- Git 状態表示では `git log` / `git status` のみを実行し、書き込み操作は行わない。
