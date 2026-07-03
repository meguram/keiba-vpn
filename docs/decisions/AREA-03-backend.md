# AREA-03: バックエンド設計

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂（Flask → FastAPI, PostgreSQL → GCS/Parquet）

---

## 1. 技術スタック

| 項目 | 採用 | バージョン |
|------|------|------------|
| **フレームワーク** | FastAPI | ≥0.104 |
| **ASGI サーバ** | Uvicorn | ≥0.24 |
| **テンプレート** | Jinja2 | ≥3.1 |
| **HTTP クライアント** | httpx / requests | ≥0.27 / ≥2.31 |
| **データストア** | GCS (JSON) + Parquet | — |
| **ジョブキュー** | ScrapeJobQueue (ファイルベース) | — |
| **認証** | セッション Cookie | — |

---

## 2. エントリポイント

```bash
# 起動
python main.py --port 8000

# アプリ定義
src/api/app.py → FastAPI(title="keiba-vpn API", version="3.0.0", lifespan=lifespan)
```

### `lifespan` での起動処理

```python
# 起動時
1. 構造チェック (StructureMonitor)
2. ScrapeJobQueue バックグラウンドワーカースレッド起動
3. 定期メンテスレッド起動（hourly_queue_maintenance）
# 終了時
4. ワーカースレッドの graceful shutdown
```

---

## 3. エンドポイント一覧

### 3-1. ヘルス・認証

| メソッド | パス | auth | 説明 |
|--------|------|------|------|
| GET | `/api/health` | No | 死活確認 |
| GET/POST | `/login` | No | セッションログイン |
| POST | `/logout` | Yes | ログアウト |
| GET | `/api/auth/status` | No | 認証状態確認 |

---

### 3-2. HTML ページ

| メソッド | パス | 説明 |
|--------|------|------|
| GET | `/` | TOP（レース一覧） |
| GET | `/monitor` | スクレイプ監視ダッシュボード |
| GET | `/scrape` | 手動スクレイプ UI |
| GET | `/scrape-control` | スクレイプキュー制御 UI |
| GET | `/queue-status` | キュー状態 UI |
| GET | `/cron-jobs` | cron ジョブ一覧 UI |
| GET | `/data-viewer` | データビューア UI |

---

### 3-3. スクレイプ監視 API

| メソッド | パス | 説明 |
|--------|------|------|
| GET | `/api/scrape-status` | 全体スクレイプ状態サマリ |
| GET | `/api/scrape-jobs` | キュージョブ一覧（`schema_validation_failures` 含む） |
| GET | `/api/scrape-summary-all` | カテゴリ別カバレッジサマリ |
| GET | `/api/date-race-matrix` | 日付×レース×カテゴリ 存在マトリクス |
| GET | `/api/coverage-calendar` | date_coverage インデックス参照 |
| GET | `/api/row-data-coverage` | 行固有派生カバレッジ |
| POST | `/api/scrape-trigger` | レガシー形式→キュー投入 |
| POST | `/api/scrape-missing` | 欠損データ一括投入 |
| GET | `/api/auto-scrape/status` | auto_scrape タスク状態 |
| POST | `/api/auto-scrape/run` | 手動タスク実行 |

#### `GET /api/date-race-matrix` パラメータ

```python
date: str  # YYYYMMDD
# レスポンス: { "date": str, "categories": list[str], "races": list[RaceMatrix] }
# RaceMatrix.coverage の値:
#   True  → データあり
#   False → 未取得
#   None  → N/A マーカー
```

#### `POST /api/scrape-missing` リクエスト

```python
{
    "date": "20260615",       # YYYYMMDD
    "category": "race_result", # カテゴリ名
    "race_ids": [...]          # 省略時は全レース
}
```

---

### 3-4. スクレイプキュー API

| メソッド | パス | 説明 |
|--------|------|------|
| POST | `/api/scrape-queue/add` | ジョブ追加 |
| POST | `/api/scrape-queue/add-job` | ジョブ追加（詳細版） |
| POST | `/api/scrape-queue/add-batch` | バッチジョブ追加 |
| GET | `/api/scrape-queue/status` | キュー状態取得 |
| GET | `/api/scrape-queue/progress` | キュー進捗 |
| GET | `/api/scrape-queue/tasks` | タスク一覧 |
| GET | `/api/scrape-queue/worker-logs` | ワーカーログ |
| POST | `/api/scrape-queue/resume` | キュー再開 |
| POST | `/api/scrape-queue/recover` | 失敗ジョブ回収 |
| POST | `/api/scrape-queue/kick` | ワーカーキック |
| POST | `/api/scrape-queue/clear` | キュークリア |
| POST | `/api/scrape-queue/stop-and-clear` | 停止＋クリア |
| POST | `/api/scrape-queue/failed/requeue` | 失敗ジョブ再キュー |
| POST | `/api/scrape-queue/failed/remove` | 失敗ジョブ削除 |
| POST | `/api/scrape-queue/enqueue-scrape-period` | 期間一括投入 |
| POST | `/api/scrape-queue/enqueue-incomplete-dates` | 未完了日投入 |
| GET/POST | `/api/scrape-queue/load-settings` | 設定読み込み |
| POST | `/api/scrape-queue/local-mirror-config` | ローカルミラー設定 |
| POST | `/api/scrape-queue/hourly-maintenance-run` | 時間メンテ実行 |

---

### 3-5. データ参照 API

| メソッド | パス | 説明 |
|--------|------|------|
| GET | `/api/data/{category}/{key}` | GCS データ直接取得 |
| GET | `/api/race/{race_id}` | レース詳細 |
| GET | `/api/race/{race_id}/bundle` | レースバンドル（全関連データ） |
| GET | `/api/race-list/{date}` | 日付別レース一覧 |
| GET | `/api/horse/{horse_id}/detail` | 馬詳細 |
| GET | `/api/horse/{horse_id}/recent_races` | 直近戦績 |
| GET | `/api/horse/{horse_id}/race_performance_history` | 成績履歴 |

---

### 3-6. 推論・ML API

| メソッド | パス | 説明 |
|--------|------|------|
| POST | `/api/race/{race_id}/predict` | レース予測実行 |
| GET | `/api/race/{race_id}/predictions` | 予測キャッシュ取得 |
| GET | `/api/odds/{race_id}` | オッズデータ |
| POST | `/api/train/*` | 学習トリガ（管理者） |
| GET | `/api/betting/{race_id}` | 馬券推奨（デフォルトパラメータで Kelly 最適化） |
| POST | `/api/betting/optimize` | 馬券ポートフォリオ最適化（パラメータ指定） |

---

### 3-7. 管理 API

| メソッド | パス | 説明 |
|--------|------|------|
| GET | `/api/admin/cron-jobs` | cron 一覧 |
| GET | `/api/admin/auto-scrape-status` | auto_scrape 状態 |
| GET | `/api/structure-*` | 構造モニタ |
| GET | `/api/backfill/*` | バックフィル状態 |

---

### 3-8. 血統・研究 API

| メソッド | パス | 説明 |
|--------|------|------|
| GET | `/api/pedigree-*` | 血統データ |
| GET | `/api/bloodline-*` | 血統分析 |
| GET | `/api/stallion-sire-tree/*` | 種牡馬系統ツリー |
| GET | `/api/track-speed/*` | トラック速度指数 |
| GET | `/api/cushion/*` | クッション値 |

---

## 4. `_SCRAPE_MISSING_CATEGORY_MAP` — カテゴリ → スクレイパータスクの対応

```python
_SCRAPE_MISSING_CATEGORY_MAP: dict[str, str] = {
    # 直接タスク
    "race_shutuba":               "race_shutuba",
    "race_result":                "race_result",
    "race_result_on_time":        "race_result_on_time",
    "race_result_lap":            "race_result_lap",
    "race_index":                 "race_index",
    "race_odds":                  "race_odds",
    "race_pair_odds":             "race_pair_odds",
    "race_paddock":               "race_paddock",
    "race_oikiri":                "race_oikiri",
    "race_trainer_comment":       "race_trainer_comment",
    "race_barometer":             "race_barometer",
    # 派生カテゴリ → 親タスクへマッピング
    "race_shutuba_meta":          "race_shutuba",
    "race_result_meta":           "race_result",
    "race_result_payoff":         "race_result",
    "race_result_track":          "race_result",
    "race_result_corner":         "race_result",
    "race_result_lap_times":      "race_result",
    "race_result_on_time_payoff": "race_result_on_time",
    "race_result_on_time_lap":    "race_result_on_time",
    "race_result_on_time_corner": "race_result_on_time",
}
```

---

## 5. スキーマ検証

- `HybridStorage.save` → `schemas.validate(category, data)` を必ず実行
- `KEIBA_SCHEMA_STRICT` 未設定 or `1`: 不合格時は GCS 非保存 + `SchemaValidationError`
- `KEIBA_SCHEMA_STRICT=0`: 診断のみ（GCS 保存は続行）
- 失敗カウント: `GET /api/scrape-jobs` の `schema_validation_failures` で確認

---

## 6. レート制限・安全制御

### スクレイピングレート

```python
SCRAPING_CONFIG = {
    "interval": 2.0,       # 秒間隔
    "workers": 1,          # 並列数（単一プロセス）
    "jitter": (0.5, 1.5),  # ランダム遅延倍率
    "429_retry": 30,       # 429 時のリトライ待機秒
}
```

### キュー並列数

```python
SCRAPE_QUEUE_PARALLEL = int(os.environ.get("SCRAPE_QUEUE_PARALLEL", "4"))  # 最大 32
```

---

## 7. エラーハンドリング

| 状況 | 処理 |
|------|------|
| netkeiba 429 Too Many Requests | 30 秒待機後リトライ |
| GCS 書き込み失敗 | ローカルキャッシュのみ保存、ログに記録 |
| スキーマ検証失敗 | GCS 非保存（Strict モード）、`schema_validation_failures` インクリメント |
| キューロックタイムアウト | stale running ジョブ回収（1 時間後） |
| アクセスエラー（403/401） | `pause_queue_for_access_error` で自動一時停止 |

---

## 8. 環境変数

| 変数 | デフォルト | 説明 |
|------|----------|------|
| `KEIBA_SCHEMA_STRICT` | `1` | スキーマ検証 strict モード |
| `KEIBA_SYNC_PED_TBL_ON_SHUTUBA` | 未設定 | 出馬表保存後に ped_tbl 自動生成 |
| `KEIBA_PED_TBL_MERGE_GEN5` | `1` | ped_tbl に Gen5 接木するか |
| `KEIBA_AUTO_SCRAPE_USE_QUEUE` | `1` | netkeiba 系はキュー経由 |
| `SCRAPE_QUEUE_PARALLEL` | `4` | キュー並列数 |
| `KEIBA_PRE_RACE_PREDICT_ENABLED` | 未設定 | T-15 スクレイプ完了後に自動推論 |
| `GCS_BUCKET` | `chuou` | GCS バケット名 |
