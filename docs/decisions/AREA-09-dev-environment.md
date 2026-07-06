# AREA-09 — 開発環境要件

**Status**: FINAL | **Last Updated**: 2026-07-06 | **Consolidates**: DEC-001（統合済み）、DEC-017（環境変数正規化）

---

## 概要

DEC-001 は競馬予測システム（keiba-vpn）のデータ要件・モデリング要件を定義した文書であり、開発環境固有の仕様（dev/stg/prod 環境分離、docker-compose 設計、デプロイフロー等）に関する記述は含まれていない。

以下に、DEC-001 から抽出できる **開発環境要件に直接関連する情報** と、DEC-017 で確定した環境変数正規名・Python バージョン・CI 設定を統合して記載する。

---

## 1. 実行環境に関する前提条件

DEC-001 から読み取れる環境要件は以下の通り。

| 項目 | 内容 |
|---|---|
| データベース | PostgreSQL（`BIGSERIAL`、`TIMESTAMPTZ`、`NUMERIC`、`ARRAY` 型を使用）|
| キャッシュ | Redis（TTL 管理付き、キャッシュキー: `prediction:{race_id}:{model_version}`、`lap:prediction:{race_id}:{model_version}`）|
| スキーママイグレーション | Alembic 等の DDL バージョン管理ツールを使用（N-11）|
| モデル管理 | MLflow 等のモデルバージョン管理基盤（F-16）|
| ML フレームワーク | LightGBM（初期実装）、LSTM（Phase 4 以降）|
| API | REST API（`GET /api/v1/races/{race_id}/predictions` 等）|

---

## 2. コンポーネント構成（推定）

DEC-001 の機能要件・非機能要件から導出されるサービスコンポーネント。

```
┌──────────────────────────────────────────────────────┐
│ keiba-vpn システム構成（DEC-001 から導出）              │
│                                                        │
│  scraper        ── netkeiba.com スクレイパー           │
│  db (PostgreSQL) ── Layer 1〜5 データ格納              │
│  redis          ── 予測結果キャッシュ                  │
│  api            ── REST API（/api/v1/...）             │
│  ml-worker      ── 特徴量生成・モデル学習・推論バッチ  │
│  mlflow         ── モデルバージョン管理                │
│  frontend       ── レース一覧・予測表示 UI             │
└──────────────────────────────────────────────────────┘
```

---

## 3. Redis キャッシュ設定

| 項目 | 値 |
|---|---|
| キャッシュキー（予測結果） | `prediction:{race_id}:{model_version}` |
| キャッシュキー（ラップ予測） | `lap:prediction:{race_id}:{model_version}` |
| TTL | 発走まで有効 / 発走後 60 秒で自動失効 |

---

## 4. スクレイパー実行設定

```python
SCRAPING_CONFIG = {
    "request_interval_sec": 2.0,
    "jitter_sec": (0.5, 1.5),
    "concurrent_workers": 1,          # シングルIP環境では並列1推奨
    "session_rotate_interval": 50,
    "retry_on_429": True,
    "retry_backoff_base_sec": 30,
    "user_agent_rotate": True,
}
```

---

## 5. 推論バッチ実行スケジュール

| ジョブ | 実行タイミング | 完了目標 |
|---|---|---|
| 出馬表取得 | レース3日前 06:00 JST / 毎日 06:00 更新 | — |
| オッズスナップショット | 発走当日 08:00 〜 発走時刻・5分毎（発走30分前は2分毎、5分前は1分毎） | — |
| 結果・ラップ取得 | 発走予定時刻 + 35分（5分間隔・最大6回リトライ） | — |
| 馬過去成績取得 | 結果収集完了後 | — |
| 推論バッチ | — | 発走3時間前までに完了（N-9） |

---

## 6. 未定義事項（本仕様書の対象外）

以下の事項は引き続き未定義であり、別途 DEC を作成して確定させる必要がある。

- dev / stg / prod の環境分離方針（ローカル PC、GPU サーバー、VPS 等の割り当て）
- docker-compose ファイルの具体的な設計（サービス定義、ネットワーク、ボリューム）
- CD（継続的デプロイ）パイプラインおよびデプロイフロー
- GPU 環境の要件（CUDA バージョン、GPU メモリ等）
- VPS スペック・OS 要件

> **備考**: 環境変数管理・Python バージョン・CI ワークフローは下記セクション 7〜9 で確定済み（DEC-017）。

---

## 7. 環境変数マスターリスト（DEC-017 — 正規名確定）

AREA-03/04 等に散在していた環境変数定義を本セクションに集約する。
**本リストが正規名の唯一の参照源**とし、他の AREA 文書は本セクションを参照すること。

### 7.1 必須環境変数

| 変数名 | 用途 | 例（example 値） |
|--------|------|-----------------|
| `DATABASE_URL` | PostgreSQL 接続 URL | `postgresql://keiba_user:keiba_pass@localhost:5432/keiba_db` |
| `APP_SECRET_KEY` | Flask セッション秘密鍵 | `change-me-in-production-use-random-32bytes` |
| `JWT_SECRET` | JWT 署名用秘密鍵 | `jwt-secret-change-me-in-production` |
| `GCS_BUCKET` | GCS バケット名 | `keiba-vpn-data` |
| `REDIS_URL` | Redis 接続 URL | `redis://localhost:6379/0` |
| `NETKEIBA_MAX_CONCURRENT_REQUESTS` | スクレイピング並列数（=1 固定） | `1` |
| `MLFLOW_TRACKING_URI` | MLflow トラッキングサーバー URL | `http://localhost:5000` |
| `FLASK_ENV` | Flask 実行環境 (development/production) | `development` |

### 7.2 `.env.example` テンプレート

プロジェクトルートに `.env.example` を配置し、開発者が `.env` を作成する際の雛形とする。
実際の秘密値は `.env` に記載し、`.gitignore` でコミット対象から除外すること。

```dotenv
# --- Database ---
# PostgreSQL 接続 URL（Alembic・SQLAlchemy 共通）
DATABASE_URL=postgresql://keiba_user:keiba_pass@localhost:5432/keiba_db

# --- Flask ---
# Flask セッション秘密鍵（本番では必ず強いランダム値に変更）
APP_SECRET_KEY=change-me-in-production-use-random-32bytes

# Flask 実行環境: development | production
FLASK_ENV=development

# --- Auth ---
# JWT 署名用秘密鍵（本番では必ず強いランダム値に変更）
JWT_SECRET=jwt-secret-change-me-in-production

# --- Storage ---
# Google Cloud Storage バケット名
GCS_BUCKET=keiba-vpn-data

# --- Cache ---
# Redis 接続 URL（DB 番号 /0 を使用）
REDIS_URL=redis://localhost:6379/0

# --- Scraping ---
# netkeiba スクレイピング並列数（シングルIP環境では 1 固定）
NETKEIBA_MAX_CONCURRENT_REQUESTS=1

# --- MLflow ---
# MLflow トラッキングサーバー URL
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 7.3 変数名の注意事項

- 他 AREA 文書で `SECRET_KEY` と記載されている箇所は `APP_SECRET_KEY` に読み替えること。
- `POSTGRES_*` 系の変数（`POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`）は CI/docker-compose 専用であり、アプリケーションコードからは `DATABASE_URL` を使用すること（セクション 9 参照）。

---

## 8. Python バージョン（P2-1 — 3.11 統一）

プロジェクト全体で Python **3.11** を使用する。3.10 以下の記述がある場合は 3.11 に更新すること。

### 8.1 バージョン固定ファイル

プロジェクトルートに `.python-version` を配置して pyenv によるバージョン固定を行う。

```
# .python-version
3.11
```

### 8.2 各ツールでのバージョン指定

| ツール | 設定箇所 | 記述例 |
|--------|----------|--------|
| pyenv | `.python-version` | `3.11` |
| Docker | `Dockerfile` | `FROM python:3.11-slim` |
| GitHub Actions | `ci.yml` | `python-version: "3.11"` |
| `pyproject.toml` | `[tool.poetry.dependencies]` | `python = "^3.11"` |

### 8.3 型ヒント・構文

Python 3.11 で利用可能な構文を積極的に使用する。

- `X | Y` 型ユニオン（`Optional[X]` の代わり）
- `tomllib`（標準ライブラリ）
- `ExceptionGroup` / `except*`（必要に応じて）

---

## 9. CI ワークフロー設定（P2-5 — `ci.yml` 統合）

### 9.1 ファイル統合方針

複数の GitHub Actions ワークフローファイルが存在する場合は **`ci.yml` に統合**する。

```
.github/workflows/
└── ci.yml          ← 単一ファイルに統合（lint / test / build を含む）
```

### 9.2 `ci.yml` 全体構成例

```yaml
name: CI

on:
  push:
    branches: [main, feature/**]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: keiba_user       # アプリ用 DB ユーザー
          POSTGRES_PASSWORD: keiba_pass   # アプリ用 DB パスワード
          POSTGRES_DB: keiba_db           # アプリ用 DB 名
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run lint
        run: |
          pip install ruff
          ruff check src/

      - name: Run tests
        env:
          DATABASE_URL: postgresql://keiba_user:keiba_pass@localhost:5432/keiba_db
          REDIS_URL: redis://localhost:6379/0
          APP_SECRET_KEY: ci-test-secret-key
          JWT_SECRET: ci-test-jwt-secret
          FLASK_ENV: development
          MLFLOW_TRACKING_URI: http://localhost:5000
          NETKEIBA_MAX_CONCURRENT_REQUESTS: 1
        run: |
          pytest src/ -v --tb=short
```

### 9.3 PostgreSQL 環境変数の説明

CI services ブロックで使用する `POSTGRES_*` 変数は Docker 公式イメージの初期化用であり、
アプリケーション接続には `DATABASE_URL`（セクション 7 参照）を使用する。

| 変数名 | 値 | 役割 |
|--------|-----|------|
| `POSTGRES_USER` | `keiba_user` | 作成される DB ユーザー名 |
| `POSTGRES_PASSWORD` | `keiba_pass` | 上記ユーザーのパスワード |
| `POSTGRES_DB` | `keiba_db` | 作成されるデータベース名 |

これらは `DATABASE_URL=postgresql://keiba_user:keiba_pass@localhost:5432/keiba_db` と対応する。

---

> **備考**: 本仕様書は DEC-001 および DEC-017 を参照している。dev/stg/prod 環境分離・docker-compose 構成・CD パイプラインは未決定であり、別途 DEC の作成を推奨する。
