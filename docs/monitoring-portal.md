# 開発者監視ポータル 仕様書

> **対象者**: 開発者のみ。エンドユーザーには非公開。  
> **ポート**: `9090`（FastAPI :8000 / Flask :5100 / Next.js :3000 とは完全独立）

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
Monitor App :9090          （スタンドアロン Flask）
      │
      ├─► FastAPI :8000    （内部 HTTP）
      ├─► Flask   :5100    （内部 HTTP）
      ├─► Next.js :3000    （内部 HTTP ヘルスチェック）
      ├─► psutil           （CPU / メモリ / ディスク）
      └─► subprocess(git)  （Git 状態）
```

- `src/monitor/app.py` は既存の `src/api/flask_app.py` とは完全に独立した Flask インスタンスを起動する。
- ポート 9090 は外部から直接アクセスしないこと（tcpexposer トンネルには含めない）。

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
    data.html          # /data
    system.html        # /system
    logs.html          # /logs
    git.html           # /git

scripts/server/
  start_monitor.sh     # 単体起動スクリプト（port 9090）

docs/
  monitoring-portal.md # 本ファイル
```

---

## 6. 起動方法

### 単体起動

```bash
# デフォルトポート 9090
bash scripts/server/start_monitor.sh

# ポート指定
bash scripts/server/start_monitor.sh --port 9191
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

---

## 8. 内部 API エンドポイント（JS ポーリング用）

監視ポータルは以下の内部 JSON API をブラウザ JS から呼び出す：

| パス | 説明 |
|------|------|
| `GET /api/internal/status` | 全サービス死活 + キュー概要 + Git HEAD |
| `GET /api/internal/system` | psutil メトリクス（CPU/Mem/Disk/Swap）|
| `GET /api/internal/git` | Git 詳細（log / status / pull ログ）|
| `GET /api/internal/logs?file=<name>` | 指定ログファイルの末尾 200 行 |
| `GET /api/internal/scraping` | スクレイピング詳細（キュー + カバレッジ）|

---

## 9. セキュリティ上の注意

- 監視ポータルはローカル（`127.0.0.1`）専用で起動し、外部向けには公開しない。
- `MONITOR_PASSWORD` は `DEV_PASSWORD` と異なる値を設定すること。
- ログビューアは `logs/` ディレクトリのみにアクセス制限し、パストラバーサルを防止する。
- Git 状態表示では `git log` / `git status` のみを実行し、書き込み操作は行わない。
