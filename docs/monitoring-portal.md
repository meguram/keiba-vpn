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

---

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
| `GET /api/internal/data-stores` | PostgreSQL / SQLite / Parquet データストア状態（`/data-stores` ページ用）|

---

## 9. セキュリティ上の注意

- 監視ポータルはローカル（`127.0.0.1`）専用で起動し、外部向けには公開しない。
- `MONITOR_PASSWORD` は `DEV_PASSWORD` と異なる値を設定すること。
- ログビューアは `logs/` ディレクトリのみにアクセス制限し、パストラバーサルを防止する。
- Git 状態表示では `git log` / `git status` のみを実行し、書き込み操作は行わない。
