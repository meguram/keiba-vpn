# keiba-vpn — マスター仕様書

> **改訂**: 2026-07-03 — archive/ ソースコードを元に実装実態に合わせて全面改訂
>
> 本文書は `docs/decisions/` の頂点ドキュメントです。各 AREA ドキュメントを参照する前に必ず読んでください。

---

## 0. システム概要

JRA 競馬レースの予測・情報提供 Web アプリケーション。netkeiba.com / SmartRC を定期スクレイピングし、LightGBM ベースの機械学習モデルで着順・複勝・オッズを予測する。

### 最重要設計原則

| # | 原則 | 要約 |
|---|------|------|
| P-01 | **テンポラルリーク禁止** | スナップショット特徴量は `as_of_race_id` で管理し、未来データの漏洩を構造的に防ぐ |
| P-02 | **GCS を唯一の正本** | スクレイピングデータの正本は GCS (`gs://chuou/data/preprocessed/netkeiba/pc/`) に置く。ローカルはキャッシュのみ |
| P-03 | **キュー駆動スクレイピング** | netkeiba 系の取得はすべて `ScrapeJobQueue` 経由。cron は直接スクレイプせずキュー投入のみ行う |
| P-04 | **スキーマ検証必須** | `HybridStorage.save` は保存前に `schemas.validate` を必ず実行する |
| P-05 | **大衆指標排除** | 学習特徴量から当日オッズ・人気を除外し、過学習を防ぐ |

---

## 1. 技術スタック

| レイヤ | 採用技術 | バージョン | 備考 |
|--------|----------|------------|------|
| **API サーバ** | FastAPI + Uvicorn | ≥0.104 / ≥0.24 | `python main.py` → `src.api.app:app` |
| **フロントエンド** | Jinja2 テンプレート + HTML/CSS/JS | — | `templates/` / `static/` |
| **スクレイピング** | requests / BeautifulSoup4 / Playwright | — | `src/scraper/` |
| **データストア** | GCS (JSON) + ローカル Parquet | — | HybridStorage 抽象化 |
| **ML** | LightGBM LambdaRank | ≥4.0 | `src/pipeline/models/` |
| **ML 管理** | MLflow | ≥2.12 | `mlflow/` に Docker Compose |
| **パイプライン** | pandas + pyarrow | ≥2.0 / ≥14.0 | 列指向 Parquet 特徴量ストア |
| **VPN** | WireGuard | — | ConoHa VPS 上 |
| **監視** | UptimeRobot / structlog | — | — |

---

## 2. システムアーキテクチャ

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ConoHa VPS (2GB RAM)                          │
│                                                                       │
│  ┌─────────────┐   ┌──────────────────────────────────────────────┐  │
│  │  cron jobs   │   │  FastAPI (main.py → src.api.app:app)         │  │
│  │  (JST)       │   │  ┌──────────────────────────────────────┐   │  │
│  │  raceday-eve │   │  │ /monitor  /scrape  /queue-status ...  │   │  │
│  │  raceday-run │   │  │ /api/scrape-queue/*                   │   │  │
│  │  backfill    │   │  │ /api/race-day-matrix                  │   │  │
│  │  daily-lists │→→→│  │ /api/auto-scrape/*                    │   │  │
│  │  weekly-upd  │   │  └──────────────────────────────────────┘   │  │
│  │  watchdog    │   │  ┌──────────────┐  ┌───────────────────┐    │  │
│  └─────────────┘   │  │ ScrapeWorker │  │  ML Inference      │    │  │
│                    │  │ (background) │  │  (RaceDayPipeline) │    │  │
│                    │  └──────┬───────┘  └───────────────────┘    │  │
│                    └─────────│────────────────────────────────────┘  │
│                              │ HybridStorage                          │
│  ┌──────────────────────┐    │  ┌─────────────────────────────────┐  │
│  │ data/queue/           │    │  │  data/local/                     │  │
│  │ scrape_queue.json     │←───┘  │  ├── features/  (Parquet)        │  │
│  └──────────────────────┘       │  ├── cache/     (L2 GCS mirror) │  │
│                                  │  ├── meta/      (coverage index)│  │
│  ┌───────────────────────────────┘  └── page_reference/ (local)   │  │
│  │  data/page_reference/            └─────────────────────────────┘  │
│  │  ├── race_lists/                                                    │
│  │  ├── race_day_schedule/                                             │
│  │  └── tables/                                                        │
│  └───────────────────────────────────────────────────────────────────┘
│                              │
│                              ↓ google-cloud-storage
│  gs://chuou/data/preprocessed/netkeiba/pc/
│    ├── race_shutuba/{YYYY}/{race_id}.json
│    ├── race_result/{YYYY}/{race_id}.json
│    ├── race_result_on_time/{YYYY}/{race_id}.json
│    ├── race_odds/{YYYY}/{race_id}.json
│    ├── race_barometer/{YYYY}/{race_id}.json  (2023〜; 2020-22 は N/A スタブ)
│    ├── horse_result/{horse_id[:4]}/{horse_id}.json
│    ├── horse_pedigree_5gen/{horse_id[:4]}/{horse_id}.json
│    └── ... (全カテゴリは AREA-06 参照)
└──────────────────────────────────────────────────────────────────────
```

---

## 3. データフロー

```
netkeiba.com / SmartRC / JRA
        │
        ↓ (ScrapeJobQueue 経由)
HybridStorage.save()
  ├── schemas.validate()  ← SchemaValidationError 時は GCS 非保存
  ├── GCS 正本保存
  ├── ローカル L2 キャッシュ更新
  └── date_coverage インデックス更新
        │
        ↓ (build_*.py CLI / cron)
data/local/features/ (Parquet 列指向ストア)
  ├── base_tbl/  (race_id, horse_id, jockey_id, trainer_id)
  ├── race_tbl/  (レース単位特徴量)
  ├── race_horse_tbl/  (出馬表・指数・斤量等)
  ├── horse_tbl/  (血統・プロファイル)
  ├── race_jockey_tbl/, race_trainer_tbl/
  └── target/rank_tbl/  (ラベル)
        │
        ↓ (build_layer_a_dataset → run_baseline_train)
MLflow Registry  →  models/*.lgb
        │
        ↓ (RaceDayPipeline / *_service.py)
推論結果 → FastAPI → UI
```

---

## 4. スクレイピング SLA 定義

| SLA | タイミング | 対象データ | cron タスク |
|-----|-----------|-----------|-------------|
| **SLA 0** | 毎日 JST 07:00 / 17:00 | race_lists | `daily-race-lists` |
| **SLA 1** | 前日 JST 18:00 | 出馬表・馬柱・追い切り | `raceday-eve` |
| **SLA 2** | 開催当日 JST 05:00-08:50 | JRA 馬場情報 | `jra-baba-morning` |
| **SLA 3** | 開催当日（発走 T-15 まで） | 出馬表・オッズ・SmartRC | `raceday-runner` |
| **SLA 4** | 各発走 T+15 分 | 速報結果 | `raceday-result-runner` |
| **SLA 5** | 開催当日 JST 17:30 | 速報まとめ・当日オッズ確定 | `raceday-evening` |
| **SLA 6** | 翌週金曜 JST 17:30 | race_result 確定・指数・barometer | `weekly-update` |
| **Backfill** | 深夜 JST 00:00-09:00 | 過去データ補完 | `backfill` |

---

## 5. ML モデルカタログ

| キー | 実験名 | モデル種別 | ステータス | サーブポート |
|------|--------|-----------|-----------|-------------|
| `keiba_lgbm` | keiba-prediction | LightGBM LambdaRank | ACTIVE | 5010 |
| `tracking_difficulty` | tracking-difficulty | LightGBM | ACTIVE | 5001 |
| `final_odds` | final-odds | LightGBM 3ヘッド | ACTIVE | 5003 |
| `pace_predictor` | pace-prediction | LightGBM | ACTIVE | 5004 |
| `finish_order` | finish-order | LightGBM | PLANNED | 5002 |

モデルの追加・変更手順: `docs/mlflow_platform.md` / `src/pipeline/mlflow/catalog.py`

---

## 6. API エンドポイント概要

### 公開・UI

| パス | 用途 |
|------|------|
| `GET /` | レース一覧 TOP |
| `GET /monitor` | スクレイピング監視ダッシュボード |
| `GET /scrape-control` | スクレイプ手動操作 |
| `GET /queue-status` | キュー状態ビューア |
| `GET /cron-jobs` | cron ジョブ一覧 |

### API（スクレイプ監視）

| パス | 用途 |
|------|------|
| `GET /api/health` | ヘルスチェック |
| `GET /api/scrape-jobs` | キュージョブ一覧 |
| `GET /api/date-race-matrix` | 日付×レース×カテゴリ 存在マトリクス |
| `GET /api/coverage-calendar` | カバレッジカレンダー |
| `POST /api/scrape-missing` | 欠損データ一括スクレイプ投入 |
| `POST /api/scrape-queue/add-batch` | バッチジョブ投入 |
| `GET /api/scrape-queue/status` | キュー状態取得 |

詳細は AREA-03 (backend) 参照。

---

## 7. ディレクトリ構造

```
keiba-vpn/
├── archive/        ← リファクタリング前のソースコード全体（参照用）
│   ├── src/
│   ├── templates/
│   ├── scripts/
│   ├── config/
│   └── ...
├── data/
│   ├── local/
│   │   ├── features/   ← Parquet 特徴量ストア
│   │   ├── cache/      ← GCS L2 ミラーキャッシュ
│   │   └── meta/       ← カバレッジインデックス等
│   ├── page_reference/ ← ローカル専用データ (race_lists, race_day_schedule)
│   ├── queue/          ← scrape_queue.json
│   └── predictions/    ← 推論結果 JSON
├── docs/
│   ├── decisions/      ← 本ディレクトリ（仕様書群）
│   ├── html/           ← HTML 形式システム文書
│   └── requirements/   ← 要件定義（scrape_process.md 等）
├── mlflow/
│   ├── server/         ← Docker Compose + Nginx
│   └── runs/           ← ローカルフォールバック
├── models/             ← 学習済みモデル *.lgb
├── logs/               ← 運用ログ
└── scripts/            ← シェルスクリプト (cron/, server/)
    （実コード: archive/scripts/ を参照）
```

---

## 8. 非機能要件

| 項目 | 目標値 | 備考 |
|------|--------|------|
| API レイテンシ | キャッシュヒット ≤200ms / ミス ≤2000ms | |
| スクレイプ成功率 | ≥99% /月 | |
| モデル評価 (Log Loss) | ベースライン比 −5% 以上 | CI ゲート |
| モデル評価 (Spearman ρ) | ≥0.55 | CI ゲート |
| ラップ MAE | ≤0.3 秒 | CI ゲート |
| テストカバレッジ | ≥80% | CI ゲート |
| メモリ上限 | 2GB (ConoHa VPS) | AREA-04 参照 |

---

## 9. ドキュメント一覧

| ファイル | 内容 |
|--------|------|
| **MASTER.md** | 本ファイル。全体概要・原則・フロー |
| **AREA-01-app-requirements.md** | ユーザー向け機能要件 |
| **AREA-02-frontend.md** | フロントエンド設計 (Jinja2/HTML) |
| **AREA-03-backend.md** | FastAPI バックエンド・エンドポイント詳細 |
| **AREA-04-ops.md** | 運用・cron・監視・デプロイ |
| **AREA-05-cost.md** | コスト計算・インフラ費用 |
| **AREA-06-data.md** | データ管理・GCS・HybridStorage・特徴量ストア |
| **AREA-07-modeling.md** | ML モデリング・学習・推論 |
| **AREA-08-testing.md** | テスト戦略・CI ゲート |
| **AREA-09-dev-environment.md** | 開発環境・インフラ構成 |

---

## 10. 開発フェーズ

| フェーズ | 内容 | 状態 |
|--------|------|------|
| **Phase 0** | データ基盤構築（スクレイプ・GCS・特徴量ストア） | ✅ 完了 |
| **Phase 1** | LightGBM ベースライン学習・評価 | ✅ 完了 |
| **Phase 2** | FastAPI API + 管理 UI（monitor, queue, scrape-control） | ✅ 完了 |
| **Phase 3** | 馬券最適化（BettingOptimizer / CompositeOptimizer） | 🔄 進行中 |
| **Phase 4** | LSTM / アンサンブル強化 | ⏳ 未着手 |
