# AREA-09 — 開発環境要件

**Status**: FINAL | **Last Updated**: 2026-07-04 | **Consolidates**: DEC-001（統合済み）

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

## 6. 未定義事項（本仕様書の対象外）

DEC-001 には以下の事項が明示されていない。現時点では未定義であり、別途 DEC を作成して確定させる必要がある。

- dev / stg / prod の環境分離方針（ローカル PC、GPU サーバー、VPS 等の割り当て）
- docker-compose ファイルの具体的な設計（サービス定義、ネットワーク、ボリューム）
- CI/CD パイプラインおよびデプロイフロー
- 環境変数管理方法（`.env` ファイル、シークレット管理）
- GPU 環境の要件（CUDA バージョン、GPU メモリ等）
- VPS スペック・OS 要件

---

> **備考**: 本仕様書は DEC-001 のみを参照しており、開発環境要件の大部分が未決定の状態である。dev/stg/prod 環境設計、docker-compose 構成、デプロイフローを確定させる DEC の作成を推奨する。