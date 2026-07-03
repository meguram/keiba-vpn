# AREA-05 — 開発環境要件

**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001, DEC-007, DEC-008, DEC-009, DEC-010

---

## 概要

DEC-001 は競馬予測システム（keiba-vpn）のデータ要件・モデリング要件を定義した文書であり、開発環境固有の仕様（dev/stg/prod 環境分離、docker-compose 設計、デプロイフロー等）に関する記述は含まれていない。

以下に、DEC-001 から抽出できる **開発環境要件に直接関連する情報** のみを記載する。

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

## 6. 確定済み技術スタック（DEC-007/008/009/010 統合）

以下の事項は後続の DEC により確定済みである。

### 6-1. 技術スタック全体

| レイヤー | 技術 | 備考 |
|---|---|---|
| フロントエンド | **TypeScript / Next.js 14（App Router）** | DEC-001, DEC-007, DEC-008 |
| フロントエンドホスティング | **Vercel Hobby（無料）** | DEC-007, DEC-008 |
| バックエンド API | **Python 3.11 / Flask 3.x + Gunicorn 21.x** | DEC-007, DEC-008 |
| Web サーバー | **Nginx**（リバースプロキシ、静的ファイル直配信、gzip, keepalive） | DEC-007 |
| キャッシュ | **Redis 7**（VPS 内、maxmemory 256MB, allkeys-lru） | DEC-007, DEC-008 |
| データストレージ | **GCS**（Parquet 形式）+ **PostgreSQL**（OLTP） | DEC-002, DEC-008 |
| ML フレームワーク | **LightGBM**（初期）→ LSTM（Phase 4 以降） | DEC-001, DEC-008 |
| スクレイピング | **Python requests/BeautifulSoup4**（cron on VPS） | DEC-007, DEC-008 |
| バッチ実行 | **cron on VPS**（systemd unit 管理） | DEC-007, DEC-008 |
| VPN | **WireGuard on VPS**（スクレイピング出口 IP 管理） | DEC-007, DEC-008 |
| モデル週次再学習 | **Cloud Run Jobs**（2vCPU / 4GB、VPS 外） | DEC-010, DEC-012 |
| アラート | **Slack Webhook** | DEC-008, DEC-010 |
| 監視（外形） | **UptimeRobot 無料** | DEC-007 |
| スキーママイグレーション | **Alembic** | DEC-001 |

### 6-2. VPS スペック（ConoHa）

| 項目 | 値 |
|---|---|
| メモリ | 2 GB |
| SSD | 100 GB |
| 月額 | ¥1,320 |
| 目的 | Flask API・AI 推論バッチ・cron・Redis・WireGuard 全同居 |

### 6-3. GCS パス管理方針（DEC-010 確定）

- `keiba-vpn/src/scraper/gcs_paths.py` を GCS パス定数の **Single Source of Truth（SSoT）** として確立
- GCS バケット名: `keiba-vpn-data`
- ETL・Feature Store・Inference Worker・Cloud Run Jobs の全コンポーネントは `gcs_paths.py` からインポートして参照
- GCS パスのハードコードは静的解析（lint ルールまたはテスト）で **禁止**

### 6-4. プロセス管理（systemd）

```
unit1: keiba-api.service       → Flask API（Gunicorn、常駐）
unit2: keiba-inference.service → Inference Worker（バッチ、夜間のみ起動・完了後終了）
cron:  keiba-scraper           → スクレイパー（04:00 JST）
cron:  keiba-inference         → 推論バッチ（06:00 JST）
```

### 6-5. 残未定義事項

以下は現時点でも未定義であり、将来の DEC で確定させる必要がある:

- dev / stg / prod の環境分離方針（ローカル PC での開発環境 docker-compose 設計）
- CI/CD パイプラインの自動化方法（GitHub Actions 等）
- 環境変数・シークレット管理方法（`.env` ファイル、GCP Secret Manager 等）
- GPU 環境の要件（LSTM 移行時に必要になる可能性）