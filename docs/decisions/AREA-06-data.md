# AREA-06: データ管理設計

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂

---

## 1. ストレージアーキテクチャ

```
┌─────────────────────────────────────────────┐
│              HybridStorage                   │
│  GCS 正本 + ローカル L2 キャッシュ + メモリ LRU  │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    ▼                         ▼
GCS (gs://chuou/)         data/local/cache/
正本データ                L2 ディスクキャッシュ
（race_*, horse_*）       (GCS ミラー)
```

### データ種別

| 種別 | 保存先 | 管理クラス |
|------|--------|-----------|
| **スクレイピング JSON** | GCS 正本 + ローカル L2 キャッシュ | `HybridStorage` |
| **ローカル専用データ** | `data/page_reference/` | `HybridStorage` (local_only) |
| **Parquet 特徴量ストア** | `data/local/features/` | `FeatureStore` |
| **カバレッジインデックス** | `data/local/meta/date_coverage/` | `date_coverage.py` |
| **N/A インデックス** | `data/local/meta/not_available/` | `date_coverage.py` |
| **キューファイル** | `data/queue/scrape_queue.json` | `ScrapeJobQueue` |
| **推論キャッシュ** | GCS `chuou/data/others/{model}/` | `InferenceCacheMixin` |

---

## 2. GCS パス構造

### バケット: `gs://chuou/`

```
chuou/
├── data/preprocessed/netkeiba/pc/
│   ├── {race_category}/{YYYY}/{race_id}.json      ← レース系
│   └── {horse_category}/{horse_id[:4]}/{horse_id}.json  ← 馬系
└── data/others/
    ├── {other_category}/{key}.json                ← その他
    ├── jra_cushion/                               ← クッション値
    ├── race_predictions/{YYYY}/{race_id}.json     ← 推論キャッシュ
    └── ...
```

### GCS ベースパス定数（`HybridStorage`）

```python
GCS_BASE = "chuou/data/preprocessed/netkeiba/pc"
GCS_OTHERS = "chuou/data/others"
```

---

## 3. カテゴリ一覧と `id_type`

| カテゴリ | `id_type` | GCS パス形式 | ローカルキャッシュ |
|----------|-----------|-------------|-----------------|
| `race_lists` | `local_only` | — | `data/page_reference/race_lists/{date}.json` |
| `race_day_schedule` | `local_only` | — | `data/page_reference/race_day_schedule/{date}.json` |
| `race_shutuba` | `race` | `{BASE}/race_shutuba/{Y}/{race_id}.json` | あり |
| `race_shutuba_past` | `race` | `{BASE}/race_shutuba_past/{Y}/{race_id}.json` | あり |
| `race_result` | `race` | `{BASE}/race_result/{Y}/{race_id}.json` | あり |
| `race_result_on_time` | `race` | `{BASE}/race_result_on_time/{Y}/{race_id}.json` | あり |
| `race_result_lap` | `race` | `{BASE}/race_result_lap/{Y}/{race_id}.json` | あり |
| `race_index` | `race` | `{BASE}/race_index/{Y}/{race_id}.json` | あり |
| `race_barometer` | `race` | `{BASE}/race_barometer/{Y}/{race_id}.json` | あり |
| `race_odds` | `race` | `{BASE}/race_odds/{Y}/{race_id}.json` | あり |
| `race_pair_odds` | `race` | `{BASE}/race_pair_odds/{Y}/{race_id}.json` | あり |
| `race_paddock` | `race` | `{BASE}/race_paddock/{Y}/{race_id}.json` | あり |
| `race_oikiri` | `race` | `{BASE}/race_oikiri/{Y}/{race_id}.json` | あり |
| `race_trainer_comment` | `race` | `{BASE}/race_trainer_comment/{Y}/{race_id}.json` | あり |
| `smartrc_race` | `race` | `{BASE}/smartrc_race/{Y}/{race_id}.json` | あり |
| `horse_result` | `horse` | `{BASE}/horse_result/{id4}/{horse_id}.json` | あり |
| `horse_pedigree_5gen` | `horse` | `{BASE}/horse_pedigree_5gen/{id4}/{horse_id}.json` | あり |
| `horse_training` | `horse` | `{BASE}/horse_training/{id4}/{horse_id}.json` | あり |

---

## 4. `HybridStorage` クラス仕様

### 初期化

```python
storage = HybridStorage(
    gcs_bucket="chuou",
    local_cache_dir="data/local/cache",
    enable_memory_cache=True,
)
JsonStorage = HybridStorage  # エイリアス
```

### 主要 API

```python
# 保存（スキーマ検証付き）
def save(self, category: str, key: str, data: dict) -> bool

# 読み込み（メモリLRU → L2ディスク → GCS の順）
def load(self, category: str, key: str, bypass_cache: bool = False) -> dict | None

# 存在確認
def exists(self, category: str, key: str) -> bool
def exists_gcs(self, category: str, key: str) -> bool

# バッチ操作
def batch_list_blobs(self, category: str, year: str) -> dict[str, float]
def batch_check_keys(self, category: str, keys: list[str]) -> dict[str, float]

# 鮮度確認
def is_fresh(self, category: str, key: str, race_date: str = "") -> bool
def freshness_info(self, category: str, key: str, race_date: str = "") -> dict

# キャッシュ管理
def cleanup_disk_cache(self, *, max_age_seconds=None, min_weekly_accesses=None) -> dict
def list_keys(self, category: str, year: str | None = None) -> list[str]

# 複合取得
def race_status(self, race_id: str) -> dict[str, dict[str, bool]]
```

---

## 5. スキーマ検証

`src/scraper/schemas.py` が各カテゴリの JSON スキーマを管理。

```python
# 保存時に自動実行
result = schemas.validate(category, data)
# result.valid: bool
# result.errors: list[str]
```

| 環境変数 | 動作 |
|--------|------|
| `KEIBA_SCHEMA_STRICT=1`（デフォルト） | 不合格 → GCS 非保存、`SchemaValidationError` |
| `KEIBA_SCHEMA_STRICT=0` | 不合格でも GCS 保存（診断のみ） |

---

## 6. カバレッジインデックス

### `TRACK_CATEGORIES`（17 カテゴリ）

```python
# src/scraper/date_coverage.py
TRACK_CATEGORIES: list[str] = [
    # 出走前
    "race_shutuba",
    "race_shutuba_meta",
    "race_index",
    "race_paddock",
    "race_odds",
    # 速報
    "race_result_on_time",
    "race_result_on_time_payoff",
    "race_result_on_time_lap",
    "race_result_on_time_corner",
    # 確定
    "race_result",
    "race_result_meta",
    "race_result_payoff",
    "race_result_track",
    "race_result_corner",
    "race_result_lap_times",
    "race_result_lap",
    "race_barometer",
]
```

### インデックスファイル

```python
COVERAGE_DIR = Path(__file__).parent.parent.parent / "data/local/meta/date_coverage"
# → data/local/meta/date_coverage/{YYYY}/{YYYYMMDD}.json
# 内容: { "race_id": { "category": bool, ... }, ... }
```

### N/A インデックス

```python
NOT_AVAILABLE_DIR = Path(__file__).parent.parent.parent / "data/local/meta/not_available"
# → data/local/meta/not_available/{YYYY}/{YYYYMMDD}.json
# 内容: { "race_id": ["race_barometer", ...], ... }
```

**N/A スタブ JSON（GCS）**:
```json
{
  "_not_available": true,
  "_reason": "data not available for this period",
  "_created_at": "2026-07-03T00:00:00Z"
}
```

---

## 7. Parquet 特徴量ストア

### ルートディレクトリ

`data/local/features/`（`FeatureStore` の `FEATURES_DIR`）

### ブロック別 Parquet

| ブロック | マージキー | ファイル例 |
|---------|-----------|---------|
| `base_tbl/` | `race_id`, `horse_id`, `jockey_id`, `trainer_id` | `base_tbl/2024/horse_number.parquet` |
| `race_tbl/` | `race_id` | `race_tbl/2024/race_name.parquet` |
| `race_horse_tbl/` | `race_id`, `horse_id` | `race_horse_tbl/2024/weight.parquet` |
| `horse_tbl/` | `horse_id` | `horse_tbl/2024/sire_name.parquet` |
| `race_jockey_tbl/` | `race_id`, `jockey_id` | `race_jockey_tbl/2024/jk_win_rate.parquet` |
| `race_trainer_tbl/` | `race_id`, `trainer_id` | `race_trainer_tbl/2024/tr_win_rate.parquet` |

### ラベル

```
target/rank_tbl/{YYYY}/rank.parquet
  カラム: race_id, horse_id, rank
  生成: python -m src.pipeline.build_rank_target
```

### 馬単位エンティティ

```
horse/ped_tbl/{horse_id[:4]}/{horse_id}.parquet    ← 血統
horse/result_tbl/{horse_id[:4]}/{horse_id}.parquet ← 成績履歴
horse/training_tbl/{horse_id[:4]}/{horse_id}.parquet ← 調教
生成: python -m src.pipeline.build_horse_entity_store
```

### FeatureStore 主要 API

```python
class FeatureStore:
    def save_feature_column(self, name: str, df: pd.DataFrame,
                             table_block: str, merge_keys: list[str]) -> None
    def load_column(self, name: str, year: str = None) -> pd.DataFrame
    def load_columns(self, names: list[str], year: str = None) -> pd.DataFrame
    def build_training_matrix(self, feature_cols: list[str],
                               target_col: str) -> pd.DataFrame
    def save_snapshot(self, df: pd.DataFrame, name: str) -> None
    def load_snapshot(self, name: str) -> pd.DataFrame | None
```

---

## 8. データ整合性・バリデーション

### 要件行カタログ

`src/scraper/requirement_row_catalog.py` が `scrape_process.md` の行 ID (`row_id`) と GCS カテゴリを対応させる。
`row_id` ごとの取得状況は `requirement_row_trace`（GCS `others/`）に保管。

```bash
# バックフィル: 全 row_id のトレース再生成
python3 -m src.scripts.scraping.materialize_requirement_row_traces
```

### 出馬表 raw の特徴量登録

```bash
# raw テーブル特徴量を race_horse_tbl に登録
python -m src.pipeline.register_raw_table_features
# 設定: _raw_table_feature_selection.json
```

---

## 9. データ保持ポリシー

| データ種 | 保持期間 | 削除条件 |
|--------|---------|---------|
| GCS 正本 JSON | 無期限 | 手動削除のみ |
| L2 ディスクキャッシュ | 最大 30 日 | `cleanup_disk_cache`（低アクセス + 古いものを削除） |
| メモリ LRU | プロセス生存中 | LRU アルゴリズム |
| キューファイル | completed: 即時削除候補 | `hourly_queue_maintenance` |
| セッションログ | 7 日 | `rotate_logs.sh` |
| Parquet 特徴量 | 無期限 | 手動削除のみ |
