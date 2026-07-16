# AREA-06 — データ管理要件（GCS パス設計 SSoT, ETL パイプライン, Feature Store, Redis TTL 設計）
**Status**: FINAL | **Last Updated**: 2026-07-16 | **Consolidates**: DEC-001（統合済み）, TASK-055（GCSバケット名更新・モデリング仕様統合）, DEC-009（モデル保持ポリシー）, DEC-015（特徴量スキーマ正規定義）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトにおけるデータ管理要件を定義する。対象範囲は以下の6領域とする。

1. **GCS パス設計（`data_paths.py` SSoT）**
2. **ETL パイプライン**
3. **Feature Store（特徴量スナップショット管理）**
4. **Redis TTL 設計**
5. **ML モデリング仕様（アンサンブル・CV・2段階分解・ポストプロセス）**
6. **推論パイプライン RAM 管理**

全設計の大前提：**`as_of_race_id` によるスナップショット管理でテンポラルリーク（未来情報混入）を構造的に排除すること**。これはデータ基盤・モデリング・評価の全工程に適用される不変制約である。

### 1-1. 収集済みデータセット範囲

| 項目 | 内容 |
|---|---|
| **対象競馬** | 中央競馬（JRA 全場・全レース） |
| **収集開始日** | **2020年1月1日**（2020-01-01） |
| **収集終了** | 直近レース（週次・日次 SLA で継続収集中） |
| **収集元サイト** | netkeiba.com（DEC-024 確定） |
| **バックフィル完了状況** | 2020〜直近年度まで順次収集済み（夜間バッチ実施中） |

> **ML 学習データ制約**: Train 分割は **2020-01-01 以降のみ有効**。2019 年以前のデータは存在しない。
> モデル訓練・特徴量生成コードで2019年以前の日付を参照しないこと（AREA-07 §5-0 参照）。

---

## 2. データ層アーキテクチャ（5層構造）

ETL・Feature Store・GCS パス設計の全領域は以下の5層スキーマを基準とする。

| 層 | 名称 | 内容 | 更新ポリシー |
|---|---|---|---|
| Layer 1 | 静的マスター | races, entries, horses, jockeys, trainers, courses | 追記・参照更新 |
| Layer 2 | 確定結果 | race_results（着順・タイム・馬体重・コーナー通過順） | 追記のみ |
| Layer 3 | 集計特徴量スナップショット | horse_stats_snapshot, jockey_stats_snapshot, trainer_stats_snapshot | 追記のみ・`UNIQUE(entity_id, as_of_race_id)` |
| Layer 4 | ラップ・ペース・コーナー | race_lap_times, race_corner_positions, race_pace_summary | 追記のみ |
| Layer 5 | オッズ時系列 | race_odds_snapshot（snapshot_at 付き） | 追記のみ・削除不可 |

---

## 3. GCS パス設計（`data_paths.py` SSoT）

### 3-1. 設計方針

<!-- TASK-055: GCSバケット名を `keiba-vpn` から `magu-keiba-horse-racing-ai` に更新。バケットルートは環境変数 GCS_BUCKET で注入する従来方針を継続 -->
- `data_paths.py` をパス定義の **Single Source of Truth（SSoT）** とし、全モジュールからインポートして使用する
- バケット名は環境変数 `GCS_BUCKET` から注入する（ハードコード禁止）
- **GCS バケット**: `${GCS_BUCKET}`（本番値: `magu-keiba-horse-racing-ai`）/ **ルートプレフィックス**: `chuou/`
- スクレイピング済み JSON は **カテゴリ名ディレクトリ** 配下に配置する（layer 番号は使わない）
- 生データと変換後データの区別はしない — GCS は「単一の永続 JSON ストア」として機能する
- `race_id` / `venue_code` の命名規則は `gs://${GCS_BUCKET}/chuou/archive/jra/` 内の既存キーと必ず一致させること（FeatureReadyGate 参照ミス防止）

### 3-2. バケット構成（実装準拠）

<!-- TASK-055: バケット名は環境変数 GCS_BUCKET で注入（ハードコード禁止）。archive/features/models の3領域を統合管理 -->

```
gs://${GCS_BUCKET}/
└── chuou/
    ├── data/
    │   └── preprocessed/
    │       └── netkeiba/
    │           └── pc/
    │               ├── {category}/{year}/{race_id}.json      ← レース単位データ
    │               └── {category}/{prefix}/{horse_id}.json   ← 馬単位データ
    │                                                           (prefix = horse_id[:4])
    ├── data/
    │   └── others/
    │       └── {category}/{key}.json                         ← その他データ（jra_cushion 等）
    ├── archive/                                               ← 既存スクレイピング生データ（変更なし）
    │   └── jra/
    │       └── {YYYY}/{MM}/
    │           ├── race_card_{YYYYMMDD}_{venue_code}.json
    │           ├── race_result_{YYYYMMDD}_{venue_code}.json
    │           └── odds_{YYYYMMDD}_{venue_code}_{race_no}.json
    │           ⚠️ 上記 archive サブパス構造は推定値。
    │              gsutil ls gs://${GCS_BUCKET}/chuou/archive/ で
    │              実際の構造を確認し、features/models の命名を合わせること。
    ├── features/                                              ← Feature Store（新規）
    │   └── v{feature_version}/
    │       └── {YYYY}/{MM}/{DD}/
    │           ├── lag_{race_id}.parquet
    │           ├── rolling_{race_id}.parquet
    │           ├── group_agg_{YYYYMMDD}_{venue_code}.parquet
    │           └── _READY_{race_id}                          ← FeatureReadyGate フラグ（write-last）
    └── models/                                               ← Model Store（新規）
        └── v{model_version}/
            └── {YYYY}/{MM}/
                ├── lgbm_{YYYYMMDD}.pkl
                ├── xgb_{YYYYMMDD}.pkl
                ├── catboost_{YYYYMMDD}.cbm
                ├── feature_columns_{feature_version}.json
                └── _manifest_{YYYYMMDD}.json
```

**ローカルのみ（GCS 非使用）**:

```
data/page_reference/
├── race_lists/{YYYYMMDD}.json        ← 開催日別レース一覧
└── race_day_schedule/{YYYYMMDD}.json ← 発走時刻スナップショット

data/calculated_data/
├── horse_index/{prefix}/{horse_id}.json  ← 馬指数・偏差値
└── horse_names.json                      ← 馬名インデックス
```

### 3-3. パス定数定義（実装準拠）

<!-- TASK-055: GCS_BUCKET デフォルト参照先を magu-keiba-horse-racing-ai に更新。features/models パスを追加 -->

```python
# data_paths.py  ── GCS / ローカルパス定義 SSoT
import os

GCS_BUCKET  = os.environ["GCS_BUCKET"]   # 環境変数必須（本番値: magu-keiba-horse-racing-ai）
GCS_ROOT    = "chuou"
GCS_BASE    = f"{GCS_ROOT}/data/preprocessed/netkeiba/pc"
GCS_OTHERS  = f"{GCS_ROOT}/data/others"
GCS_ARCHIVE = f"{GCS_ROOT}/archive/jra"
GCS_FEATURES = f"{GCS_ROOT}/features"
GCS_MODELS   = f"{GCS_ROOT}/models"

# ── レース単位データ（カテゴリ / 年 / race_id） ──────────────────
def race_path(category: str, race_id: str) -> str:
    year = race_id[:4]
    return f"gs://{GCS_BUCKET}/{GCS_BASE}/{category}/{year}/{race_id}.json"

# ── 馬単位データ（カテゴリ / prefix / horse_id） ─────────────────
def horse_path(category: str, horse_id: str) -> str:
    prefix = horse_id[:4]
    return f"gs://{GCS_BUCKET}/{GCS_BASE}/{category}/{prefix}/{horse_id}.json"

# ── その他データ（jra_cushion 等） ────────────────────────────────
def others_path(category: str, key: str) -> str:
    return f"gs://{GCS_BUCKET}/{GCS_OTHERS}/{category}/{key}.json"

# ── Feature Store パス ────────────────────────────────────────────
def feature_lag_path(race_id: str, feature_version: str) -> str:
    yyyy, mm, dd = race_id[:4], race_id[4:6], race_id[6:8]
    return f"gs://{GCS_BUCKET}/{GCS_FEATURES}/v{feature_version}/{yyyy}/{mm}/{dd}/lag_{race_id}.parquet"

def feature_rolling_path(race_id: str, feature_version: str) -> str:
    yyyy, mm, dd = race_id[:4], race_id[4:6], race_id[6:8]
    return f"gs://{GCS_BUCKET}/{GCS_FEATURES}/v{feature_version}/{yyyy}/{mm}/{dd}/rolling_{race_id}.parquet"

def feature_group_agg_path(date_str: str, venue_code: str, feature_version: str) -> str:
    yyyy, mm, dd = date_str[:4], date_str[4:6], date_str[6:8]
    return f"gs://{GCS_BUCKET}/{GCS_FEATURES}/v{feature_version}/{yyyy}/{mm}/{dd}/group_agg_{date_str}_{venue_code}.parquet"

def feature_ready_path(race_id: str, feature_version: str) -> str:
    """FeatureReadyGate が確認するゼロバイトフラグファイル（write-last 原則）"""
    yyyy, mm, dd = race_id[:4], race_id[4:6], race_id[6:8]
    return f"gs://{GCS_BUCKET}/{GCS_FEATURES}/v{feature_version}/{yyyy}/{mm}/{dd}/_READY_{race_id}"

# ── Model Store パス ──────────────────────────────────────────────
def model_path(model_type: str, date_str: str, model_version: str) -> str:
    """model_type: lgbm / xgb / catboost"""
    yyyy, mm = date_str[:4], date_str[4:6]
    ext = {"lgbm": "pkl", "xgb": "pkl", "catboost": "cbm"}.get(model_type, "pkl")
    return f"gs://{GCS_BUCKET}/{GCS_MODELS}/v{model_version}/{yyyy}/{mm}/{model_type}_{date_str}.{ext}"

def model_manifest_path(date_str: str, model_version: str) -> str:
    yyyy, mm = date_str[:4], date_str[4:6]
    return f"gs://{GCS_BUCKET}/{GCS_MODELS}/v{model_version}/{yyyy}/{mm}/_manifest_{date_str}.json"

# ── ローカル: ページ参照系 ────────────────────────────────────────
LOCAL_PAGE_REF = "data/page_reference"

def race_lists_path(date_str: str) -> str:
    return f"{LOCAL_PAGE_REF}/race_lists/{date_str}.json"

def race_day_schedule_path(date_str: str) -> str:
    return f"{LOCAL_PAGE_REF}/race_day_schedule/{date_str}.json"

# ── ローカル: 計算済みデータ ──────────────────────────────────────
LOCAL_CALC = "data/calculated_data"

def horse_index_path(horse_id: str) -> str:
    prefix = horse_id[:4]
    return f"{LOCAL_CALC}/horse_index/{prefix}/{horse_id}.json"

# ── GCS パス参照定数（コード内参照用） ───────────────────────────
GCS_PATHS = {
    "archive":  f"gs://{GCS_BUCKET}/{GCS_ARCHIVE}",
    "features": f"gs://{GCS_BUCKET}/{GCS_FEATURES}/v{{feature_version}}",
    "models":   f"gs://{GCS_BUCKET}/{GCS_MODELS}/v{{model_version}}",
}
```

**主要カテゴリとパス例**:

| カテゴリ | パス例 |
|---|---|
| `race_detail` | `gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/race_detail/2025/202505010101.json` |
| `race_result` | `gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/race_result/2025/202505010101.json` |
| `race_odds` | `gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/race_odds/2025/202505010101.json` |
| `race_shutuba` | `gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/race_shutuba/2025/202505010101.json` |
| `horse_result` | `gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/horse_result/2000/2000110001.json` |
| `horse_pedigree_5gen` | `gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/horse_pedigree_5gen/2000/2000110001.json` |
| `jra_cushion` | `gs://${GCS_BUCKET}/chuou/data/others/jra_cushion/2025.json` |
| Feature Store lag | `gs://${GCS_BUCKET}/chuou/features/v3/2024/01/28/lag_20240128_nakayama_11.parquet` |
| Model Store LGBM | `gs://${GCS_BUCKET}/chuou/models/v1/2024/01/lgbm_20240128.pkl` |

**`race_id` 命名規則（archive と統一）**:

```
{YYYYMMDD}_{venue_code}_{race_no}
例: 20240128_nakayama_11
    20240128_kyoto_09

venue_code: chuou/archive/jra 内の既存 venue_code と同一キーを使用すること
```

**`_manifest_{YYYYMMDD}.json` スキーマ**:

```json
{
  "race_id":           "20240128_nakayama_11",
  "feature_version":   "v3",
  "layers_completed":  ["lag_features", "rolling_features", "group_agg_features"],
  "completed_at":      "2024-01-28T08:30:00Z",
  "row_count":         18,
  "checksum":          "sha256:abc123..."
}
```

> `checksum` は全 parquet ファイルの SHA-256 ハッシュ結合値。manifest 書き込みは全 parquet 書き込み後の最終ステップ（write-last 原則）。FeatureReadyGate は `_READY_` フラグファイルの存在と manifest の `checksum` を照合する。

### 3-4. ストレージ階層（HybridStorage）

```
L1: メモリ LRU キャッシュ
    TTL: 3,600 秒
    用途: 高頻度アクセスデータ（出馬表・直近レース一覧）

L2: ディスクキャッシュ（data/cache/）
    TTL: レース系データ 12 時間 / 馬系データ 2 日
    容量上限: 週次アクセス 2回未満のファイルを自動削除（disk_cache_cleanup）

L3: GCS（gs://${GCS_BUCKET}/）
    永続ストレージ。L1/L2 ミス時に自動フォールバック。
    読取後に L2 → L1 の順でウォームアップ。
```

**読取フロー**: L1 ヒット → 即返却 | L1 ミス → L2 確認 → L3 フォールバック → L2・L1 にキャッシュ

---

## 4. ETL パイプライン

### 4-1. パイプライン全体像

```
netkeiba.com / JRA
    │ HTTP スクレイピング（SLA 0〜6 + バックフィル）
    ▼
[Scrape]  スクレイパー → L1 メモリキャッシュ → L2 ディスクキャッシュ
                          → L3 GCS JSON（gs://${GCS_BUCKET}/chuou/）
                          (HybridStorage、data_paths.py 経由)
    │
    ▼
[Parse / Normalize]  JSON → 正規化レコード → PostgreSQL（Layer 1〜5 テーブル）
    │
    ▼
[Feature Store ETL]  Layer 3 ラグ特徴・ローリング統計・グループ集約生成
                     → GCS features/v{version}/ へ parquet 書き込み
                     → _READY_{race_id} フラグ作成（write-last）
    │
    ▼
[Snapshot Batch]  Layer 3 スナップショット生成（as_of_race_id 付与）
                  → PostgreSQL horse/jockey/trainer_stats_snapshot
    │
    ▼
[FeatureReadyGate]  _READY_ フラグ + manifest checksum 検証
    │
    ▼
[AI Trigger]  T-15バンドル完了 → 推論ジョブ起動（Stage 1 → Stage 2）
    │
    ▼
[Calculated Data]  馬指数・偏差値・馬名インデックス → data/calculated_data/
```

### 4-2. スクレイピング収集スケジュール

Cron SLA の詳細定義は **[AREA-04-ops.md](AREA-04-ops.md) セクション 2** を参照。本セクションでは ETL 観点での収集パターンのみ示す。

| フェーズ | SLA | 取得カテゴリ | 後続処理 |
|---|---|---|---|
| 前日準備 | SLA 1 (18:00) | race_shutuba / race_oikiri / horse_training | 追走難度・最終オッズ precompute |
| 当日朝 | SLA 2 (05:00-08:50) | jra_cushion | GCS others/ 格納 |
| T-15バンドル | SLA 3 (各レース T-15分) | race_detail + race_odds + race_paddock + race_barometer + race_trainer_comment + JRA馬場ライブ | AI 予測トリガ |
| 速報結果 | SLA 4 (T+15分) | race_result_on_time | 速報表示更新 |
| 確定結果 | SLA 5 (17:30) | race_result / race_result_lap / race_index / race_pair_odds | 馬場速度指数計算 |
| 週次更新 | SLA 6 (金曜) | horse_result（先週分） | 指数・偏差値・馬情報更新 |
| バックフィル | 夜間 (00:00-09:00) | 2020〜2026 年度別 fast / full フェーズ | 過去データ蓄積 |

### 4-3. スクレイピング設定（実装値）

```
netkeiba.com:
  リクエスト間隔: 2.2〜4.0 秒（ランダム + ガウスジッター）
  バースト制限: 14 req ごとに 6〜12 秒クールダウン
  セッションリフレッシュ: 150 req ごとに TLS/Cookie 再構築
  グローバル最大同時スロット: 4
  429/503: 初期 5s バックオフ・係数 2.5・最大 3 リトライ
```

### 4-4. Transform ルール

| 対象カテゴリ | 変換内容 | 格納先 |
|---|---|---|
| `race_shutuba` | 出走馬・騎手・枠順・オッズ抽出 | GCS + PostgreSQL Layer 1 |
| `race_result` | 着順・タイム・馬体重・コーナー通過順 | GCS + PostgreSQL Layer 2 |
| `race_result_lap` | 1F毎ラップ・ペース区分 | GCS + PostgreSQL Layer 4 |
| `race_odds` | 単複オッズ・snapshot_at 付きレコード | GCS + PostgreSQL Layer 5 |
| `horse_result` | 出走歴・着順・タイム・賞金 | GCS + PostgreSQL Layer 1/2 |
| `race_result_lap` | 1F毎ラップ時刻・ペース区分 | GCS + PostgreSQL Layer 4 |

---

## 5. 特徴量スキーマ正規定義（DEC-015）

<!-- DEC-015: 特徴量カラム名の正規定義。全モジュール（Feature Store ETL / 学習パイプライン / 推論パイプライン）はこのリストをインポートして参照すること。独自カラム名の定義禁止。 -->

### 5-1. 主要特徴量カラム一覧

| カラム名 | 型 | 説明 | 収集元 |
|---|---|---|---|
| `horse_past_results` | `JSONB / FLOAT[]` | 直近5走の着順・タイム・馬場状態（ラグ特徴） | `horse_result` |
| `jockey_stats` | `JSONB / FLOAT[]` | 騎手直近30日の勝率・連対率・騎乗数 | `jockey_stats_snapshot` |
| `course_affinity` | `FLOAT` | 馬×コース（距離・回り・芝/ダート）適性スコア | 集計特徴量 |
| `odds_win` | `FLOAT` | 単勝オッズ（暫定。`snapshot_at` 最新値） | `race_odds_snapshot` |
| `odds_place` | `FLOAT` | 複勝オッズ（暫定） | `race_odds_snapshot` |
| `odds_change_rate` | `FLOAT` | オッズ変動率（直前スナップショット比） | `race_odds_snapshot` |
| `track_condition` | `CATEGORY` | 馬場状態（良/稍重/重/不良）→ integer encoded | `race_detail` |

### 5-2. Temporal Leakage ポリシー

> **原則**: 対象レース出走前の情報のみ使用可。

| 特徴量 | 使用可否 | 備考 |
|---|---|---|
| 暫定オッズ（`snapshot_at` < 発走時刻） | ✅ 使用可 | `odds_win`、`odds_place`、`odds_change_rate` |
| 1走前の着順・タイム・賞金 | ✅ 使用可 | `as_of_race_id` スナップショット制約必須 |
| 最終オッズ（発走後確定値） | ❌ 使用不可 | リーク禁止。学習・推論ともに除外 |
| 当日走の着順・タイム（対象レース同日他レース含む） | ❌ 使用不可 | `window_end = as_of_race_id` で自動除外 |

Feature Store API 呼び出し時は必ず `get_snapshot(race_id, as_of=race_id)` の形式で `window_end=as_of_race_id` を明示すること（D-4対策）。

### 5-3. カラム名参照ルール

- カラム名の変更は本セクション（AREA-06 § 5-1）を更新してから、AREA-07 の特徴量定義（§ 4）へ伝播させる
- `feature_columns_{feature_version}.json`（Model Store 内）は本テーブルと同期させること
- 追加カラムは `DEC-015` 変更履歴として記録すること

---

## 6. モデル保持ポリシー（DEC-009）

<!-- DEC-009: AREA-06 と AREA-07 で共通のモデル保持ポリシーを統一定義。 -->

### 6-1. 複合保持ポリシー

モデルアーティファクト（`gs://${GCS_BUCKET}/chuou/models/v{model_version}/`）には以下の複合ポリシーを適用する：

| ルール | 内容 |
|---|---|
| **最新バージョン保持** | 最新3バージョンを常時保持（削除不可） |
| **経過日数による削除** | 作成日から 365 日経過したバージョンを削除 |
| **両条件の複合適用** | 「最新3バージョン保持」が「365日削除」より優先する（最新3件は365日を超えても保持） |

### 6-2. GCS ライフサイクル設定（参考）

```json
{
  "rule": [
    {
      "action": { "type": "Delete" },
      "condition": {
        "age": 365,
        "matchesPrefix": ["chuou/models/"],
        "numNewerVersions": 3
      }
    }
  ]
}
```

`numNewerVersions: 3` により「最新3バージョンより古いオブジェクト」かつ「作成から365日超過」の条件を両方満たした場合のみ削除される。

### 6-3. ModelRegistry との連携

- モデル登録時に `created_at` タイムスタンプと `version_tag` を `_manifest_{YYYYMMDD}.json` に記録する
- CI ゲートは本番投入前に「最新3バージョン以内か」を検証する
- 保持対象外のアーティファクトは `model_cleanup_job`（週次）が GCS ライフサイクルと連携して削除する

---

## 7. PostgreSQL データベース スキーマ定義（データディクショナリ）

**対象 DB**: `keiba_db_stg`（keiba_db_dev / keiba_db_prod も同一スキーマ。Alembic で管理）  
**テーブル総数**: 26（`alembic_version` 除く）  
**SSoT**: `src/db/models.py` + `alembic/versions/`

> 表記ルール: **PK** = 主キー / **FK→X.Y** = 外部キー（X テーブルの Y 列参照） / **NN** = NOT NULL / `*` = デフォルト値あり

---

### 7-1. Layer 1 — 静的マスター

#### `races` — レースマスター

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `race_id` | VARCHAR(20) | PK, NN | レース識別子（例: `202406010101`、12桁） |
| `race_name` | VARCHAR(200) | | レース名（例: 「春のステークス」） |
| `course` | VARCHAR(20) | | コース識別子 |
| `venue` | VARCHAR(20) | | 開催場（例: 東京、中山、阪神） |
| `surface` | VARCHAR(10) | | 馬場種別（`芝` / `ダート`） |
| `distance` | INTEGER | | 距離（m） |
| `direction` | VARCHAR(10) | | 回り（`右` / `左` / `直線`） |
| `weather` | VARCHAR(20) | | 天気 |
| `track_condition` | VARCHAR(10) | | 馬場状態（`良` / `稍重` / `重` / `不良`） |
| `start_time` | TIME | | 発走予定時刻 |
| `race_date` | DATE | | 開催日 |
| `field_size` | SMALLINT | | 出走頭数 |
| `grade` | VARCHAR(20) | | グレード（`G1` / `G2` / `G3` / `未勝利` 等） |
| `race_class` | VARCHAR(100) | | クラス詳細 |
| `weight_rule` | VARCHAR(50) | | 斤量ルール（`馬齢` / `別定` / `ハンデ` 等） |
| `is_excluded` | BOOLEAN | `*false` | 除外フラグ（障害・地方など除外対象） |
| `created_at` | TIMESTAMPTZ | `*now()` | レコード作成日時 |

#### `horses` — 馬マスター

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `horse_id` | VARCHAR(20) | PK, NN | 馬識別子（netkeiba 馬 ID） |
| `horse_name` | VARCHAR(100) | NN | 馬名（日本語） |
| `sex` | VARCHAR(5) | | 性別（`牡` / `牝` / `セ`） |
| `birth_year` | SMALLINT | | 生年 |
| `sire_id` | VARCHAR(20) | FK→sires.sire_id | 父馬 ID |
| `dam_sire` | VARCHAR(100) | | 母父名（テキスト） |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `jockeys` — 騎手マスター

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `jockey_id` | VARCHAR(20) | PK, NN | 騎手識別子 |
| `jockey_name` | VARCHAR(100) | NN | 騎手名 |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `trainers` — 調教師マスター

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `trainer_id` | VARCHAR(20) | PK, NN | 調教師識別子 |
| `trainer_name` | VARCHAR(100) | NN | 調教師名 |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `sires` — 種牡馬マスター

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `sire_id` | VARCHAR(20) | PK, NN | 種牡馬 ID |
| `sire_name` | VARCHAR(100) | NN | 種牡馬名 |
| `sire_line` | VARCHAR(50) | | 系統（例: サンデー系、ノーザンダンサー系） |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `courses` — コースマスター

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `course_id` | VARCHAR(20) | PK, NN | コース識別子 |
| `course_name` | VARCHAR(50) | NN | コース名 |
| `region` | VARCHAR(20) | | 地域 |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `entries` — 出走表（出馬表）

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `entry_id` | BIGINT | PK, NN | エントリ識別子（連番） |
| `race_id` | VARCHAR(20) | NN, FK→races.race_id | レース ID |
| `horse_id` | VARCHAR(20) | NN, FK→horses.horse_id | 馬 ID |
| `post_no` | SMALLINT | | 馬番 |
| `bracket_number` | SMALLINT | | 枠番（1〜8） |
| `jockey_id` | VARCHAR(20) | FK→jockeys.jockey_id | 騎手 ID |
| `trainer_id` | VARCHAR(20) | FK→trainers.trainer_id | 調教師 ID |
| `jockey_weight` | NUMERIC(4,1) | | 斤量（kg） |
| `weight` | SMALLINT | | 馬体重（kg） |
| `weight_change` | SMALLINT | | 馬体重増減（kg、前走比） |
| `sex_age` | VARCHAR(10) | | 性齢テキスト（例: `牡3`） |
| `created_at` | TIMESTAMPTZ | `*now()` | |

> **UNIQUE**: `(race_id, horse_id)` — 同一レース内での馬の重複登録を防止

---

### 7-2. Layer 2 — 確定結果

#### `race_results` — レース確定結果（着順・タイム）

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `result_id` | BIGINT | PK, NN | 結果識別子（連番） |
| `race_id` | VARCHAR(20) | NN, FK→races.race_id | レース ID |
| `horse_id` | VARCHAR(20) | NN, FK→horses.horse_id | 馬 ID |
| `finish_pos` | SMALLINT | | 着順（1〜。着外は NULL） |
| `finish_time_sec` | NUMERIC(7,2) | | 走破タイム（秒、例: `68.50`） |
| `margin` | VARCHAR(20) | | 着差テキスト（例: `1.1/2`） |
| `last_3f_sec` | NUMERIC(5,2) | | 上り3ハロン（秒） |
| `weight` | SMALLINT | | 馬体重（kg、当日計測値） |
| `jockey_id` | VARCHAR(20) | FK→jockeys.jockey_id | 騎手 ID |
| `created_at` | TIMESTAMPTZ | `*now()` | |

> **UNIQUE**: `(race_id, horse_id)` — 1レース1頭につき1レコード

---

### 7-3. Layer 3 — 集計特徴量スナップショット

> **重要**: `as_of_race_id` によるテンポラルスナップショット管理。  
> 対象レースの直前時点の統計のみを保持し、未来情報の混入を構造的に排除する（§ 5-2 参照）。

#### `horse_stats_snapshot` — 馬の統計スナップショット

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `snapshot_id` | BIGINT | PK, NN | スナップショット識別子 |
| `horse_id` | VARCHAR(20) | NN | 馬 ID |
| `as_of_race_id` | VARCHAR(20) | NN | 集計基準レース ID（このレース直前時点） |
| `as_of_date` | DATE | NN | 集計基準日 |
| `win_rate_all` | NUMERIC(5,4) | | 通算勝率（全条件） |
| `place_rate_all` | NUMERIC(5,4) | | 通算連対率 |
| `show_rate_all` | NUMERIC(5,4) | | 通算複勝率 |
| `sample_count` | SMALLINT | | 集計サンプル数 |
| `win_rate_turf` | NUMERIC(5,4) | | 芝勝率 |
| `win_rate_dirt` | NUMERIC(5,4) | | ダート勝率 |
| `win_rate_distance` | NUMERIC(5,4) | | 同距離帯勝率 |
| `win_rate_course` | NUMERIC(5,4) | | 同コース勝率 |
| `win_rate_going` | NUMERIC(5,4) | | 同馬場状態勝率 |
| `avg_last_3f` | NUMERIC(5,2) | | 平均上り3F（秒） |
| `speed_index_avg` | NUMERIC(6,2) | | スピード指数平均 |
| `speed_index_max` | NUMERIC(6,2) | | スピード指数最大値 |
| `running_style_score` | NUMERIC(5,2) | | 脚質スコア（逃=1 → 追=4） |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `jockey_stats_snapshot` — 騎手の統計スナップショット

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `snapshot_id` | BIGINT | PK, NN | |
| `jockey_id` | VARCHAR(20) | NN | 騎手 ID |
| `as_of_race_id` | VARCHAR(20) | NN | 集計基準レース ID |
| `as_of_date` | DATE | NN | |
| `win_rate_all` | NUMERIC(5,4) | | 通算勝率 |
| `place_rate_all` | NUMERIC(5,4) | | 通算連対率 |
| `show_rate_all` | NUMERIC(5,4) | | 通算複勝率 |
| `sample_count` | SMALLINT | | 集計サンプル数 |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `trainer_stats_snapshot` — 調教師の統計スナップショット

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `snapshot_id` | BIGINT | PK, NN | |
| `trainer_id` | VARCHAR(20) | NN | 調教師 ID |
| `as_of_race_id` | VARCHAR(20) | NN | 集計基準レース ID |
| `as_of_date` | DATE | NN | |
| `win_rate_all` | NUMERIC(5,4) | | 通算勝率 |
| `place_rate_all` | NUMERIC(5,4) | | 通算連対率 |
| `show_rate_all` | NUMERIC(5,4) | | 通算複勝率 |
| `sample_count` | SMALLINT | | 集計サンプル数 |
| `created_at` | TIMESTAMPTZ | `*now()` | |

---

### 7-4. Layer 4 — ラップ・ペース・コーナー

#### `race_lap_times` — 1Fごとラップタイム

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `race_id` | VARCHAR(20) | PK, NN | レース ID |
| `furlong_index` | SMALLINT | PK, NN | ハロン番号（1〜、後ろからカウント） |
| `lap_time_sec` | NUMERIC(4,2) | NN | そのハロンのラップタイム（秒） |
| `cumulative_sec` | NUMERIC(6,2) | | 累積タイム（秒） |

#### `race_corner_positions` — コーナー通過順

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `race_id` | VARCHAR(20) | PK, NN | レース ID |
| `horse_id` | VARCHAR(20) | PK, NN | 馬 ID |
| `corner_1` | SMALLINT | | 1コーナー通過順位 |
| `corner_2` | SMALLINT | | 2コーナー通過順位 |
| `corner_3` | SMALLINT | | 3コーナー通過順位 |
| `corner_4` | SMALLINT | | 4コーナー通過順位 |

#### `race_pace_summary` — レースペースサマリー

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `race_id` | VARCHAR(20) | PK, NN | レース ID |
| `first_3f_sec` | NUMERIC(5,2) | | 前半3F タイム（秒） |
| `last_3f_sec` | NUMERIC(5,2) | | 後半3F タイム（秒） |
| `pace_category` | VARCHAR(10) | | ペース区分（`S` / `M` / `H`） |
| `front_runner_count` | SMALLINT | | 逃げ・先行馬数 |
| `created_at` | TIMESTAMPTZ | `*now()` | |

---

### 7-5. Layer 5 — オッズ時系列

#### `race_odds_snapshot` — オッズスナップショット

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `snapshot_id` | BIGINT | PK, NN | スナップショット識別子 |
| `race_id` | VARCHAR(20) | NN | レース ID |
| `horse_id` | VARCHAR(20) | NN | 馬 ID |
| `snapshot_type` | VARCHAR(20) | NN | 種別（`win` / `place` 等） |
| `odds_value` | NUMERIC(7,1) | NN | 単勝オッズ |
| `odds_place_low` | NUMERIC(7,1) | | 複勝オッズ下限 |
| `odds_place_high` | NUMERIC(7,1) | | 複勝オッズ上限 |
| `snapshot_at` | TIMESTAMPTZ | NN | スナップショット取得日時 |

> `snapshot_at` が発走時刻より前のレコードのみ学習・推論に使用可（§ 5-2 参照）

---

### 7-6. めぐ指数（AREA-11）

#### `megu_index` — 馬×レース単位のめぐ指数

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | BIGINT | PK, NN | |
| `race_id` | VARCHAR(20) | NN | レース ID |
| `horse_id` | VARCHAR(20) | NN | 馬 ID |
| `finish_time_sec` | NUMERIC(6,2) | NN | 走破タイム（秒） |
| `par_time_sec` | NUMERIC(6,2) | | 基準タイム（セル別中央値） |
| `delta_pace_sec` | NUMERIC(5,3) | NN, `*0` | ペース補正量（秒） |
| `delta_track_sec` | NUMERIC(5,3) | NN, `*0` | 馬場補正量（秒） |
| `delta_weight_sec` | NUMERIC(5,3) | NN, `*0` | 斤量補正量（秒） |
| `delta_level_sec` | NUMERIC(5,3) | NN, `*0` | レースレベル補正量（秒） |
| `adjusted_time_sec` | NUMERIC(6,2) | NN | 補正後タイム（秒） |
| `megu_index` | NUMERIC(6,1) | NN | めぐ指数値（100 = par、1点 ≈ 0.1秒） |
| `field_quality` | NUMERIC(14,0) | | フィールドクオリティ（FQ、円単位） |
| `front_split_sec` | NUMERIC(5,2) | | 実測前半スプリット（秒）（migration 004） |
| `split_point_m` | INTEGER | | スプリット計測距離（m）（migration 004） |
| `tsi_raw` | NUMERIC(6,3) | | 馬場速度指数（TSI）の生値（migration 004） |
| `computation_status` | VARCHAR(20) | | 算出ステータス（`valid` / `out_of_range` / `no_par`）（migration 005） |
| `model_version` | VARCHAR(20) | NN, `*stg-v1` | モデルバージョン |
| `computed_at` | TIMESTAMPTZ | `*now()` | 算出日時 |

> **UNIQUE**: `(race_id, horse_id, model_version)`  
> **INDEX**: `(horse_id, computed_at DESC)` — 馬の直近指数取得用

#### `megu_par_time` — 基準タイムマスター

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | INTEGER | PK, NN | |
| `distance` | INTEGER | NN | 距離（m） |
| `course` | VARCHAR(20) | NN | 会場（例: 東京、中山） |
| `surface` | VARCHAR(10) | NN | 馬場種別（`芝` / `ダート`） |
| `track_condition` | VARCHAR(10) | NN | 馬場状態 |
| `par_time_sec` | NUMERIC(6,2) | NN | 基準タイム（秒） |
| `par_front_split_sec` | NUMERIC(5,2) | | 基準前半スプリット（秒） |
| `sample_count` | INTEGER | NN | 集計サンプル数 |
| `class_bucket` | VARCHAR(10) | NN, `*""` | クラスバケット（migration 006）。未勝利/1勝/... 等の区分 |
| `model_version` | VARCHAR(20) | NN, `*stg-v1` | モデルバージョン |
| `computed_at` | TIMESTAMPTZ | `*now()` | |

> **UNIQUE**: `(distance, course, surface, track_condition, class_bucket, model_version)`（migration 006 で `class_bucket` を追加）

#### `megu_regression_params` — OLS 回帰係数

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | INTEGER | PK, NN | |
| `param_name` | VARCHAR(50) | NN | 係数名（`beta_pace` / `beta_track` / `beta_weight` / `beta_level`） |
| `param_value` | NUMERIC(10,6) | NN | 係数値 |
| `std_error` | NUMERIC(10,6) | | 標準誤差 |
| `sample_count` | INTEGER | | 学習サンプル数 |
| `model_version` | VARCHAR(20) | NN, `*stg-v1` | モデルバージョン |
| `fitted_at` | TIMESTAMPTZ | `*now()` | 学習日時 |

> **UNIQUE**: `(param_name, model_version)`

#### `megu_delta_track` — 日 × 会場 × 馬場種別の馬場補正値（migration 007）

NB-03 が算出した `delta_track_sec` を `date × venue × surface` のキーで保持する独立テーブル。  
`megu_index.delta_track_sec`（per-horse 列）の導出元であり、馬場状態の時系列分析・開催間比較などの副次的分析に使用する。

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | INTEGER | PK, NN | |
| `date` | DATE | NN | 開催日 |
| `venue` | VARCHAR(10) | NN | 会場（例: 東京、阪神） |
| `surface` | VARCHAR(10) | NN | 馬場種別（`芝` / `ダート`） |
| `delta_track_sec` | NUMERIC(6,3) | NULL 可 | 馬場補正値（秒）。正=重馬場（タイム遅化）、負=軽馬場（タイム速化）。`is_fallback=true` のとき NULL |
| `n_races` | INTEGER | NN | 算出に使用したレース数 |
| `is_fallback` | BOOLEAN | NN, `*false` | `n_races < 3` のため補正値を 0 で代替したとき `true` |
| `model_version` | VARCHAR(20) | NN, `*stg-v1` | モデルバージョン |
| `computed_at` | TIMESTAMPTZ | `*now()` | 算出日時 |

> **UNIQUE**: `(date, venue, surface, model_version)`  
> **INDEX**: `(date, venue)` — 開催日 × 会場での時系列クエリ用

**DB への保存**: `src/pipeline/megu_index/save_delta_track.py`  
```bash
python -m src.pipeline.megu_index.save_delta_track \
    --parquet notebooks/megu_index/output/nb03/delta_track.parquet
```

---

### 7-7. AI 予測結果

#### `prediction_results` — AI 予測出力

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `prediction_id` | BIGINT | PK, NN | |
| `race_id` | VARCHAR(20) | NN | レース ID |
| `horse_id` | VARCHAR(20) | NN | 馬 ID |
| `model_version` | VARCHAR(50) | NN | モデルバージョン（例: `stg-mock-v1`） |
| `predicted_at` | TIMESTAMPTZ | `*now()` | 予測実行日時 |
| `win_prob` | NUMERIC(5,4) | | 勝率（0.0〜1.0） |
| `place_prob` | NUMERIC(5,4) | | 連対率 |
| `show_prob` | NUMERIC(5,4) | | 複勝率 |
| `predicted_win_odds` | NUMERIC(7,1) | | 予測単勝オッズ |
| `predicted_place_odds` | NUMERIC(7,1) | | 予測複勝オッズ |
| `expected_win_roi` | NUMERIC(7,2) | | 期待単勝回収率 |
| `expected_show_roi` | NUMERIC(7,2) | | 期待複勝回収率 |
| `predicted_position` | SMALLINT | | 予測着順 |
| `predicted_running_style` | VARCHAR(10) | | 予測脚質（`逃` / `先` / `差` / `追`） |

#### `prediction_lap_times` — ラップタイム予測

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `race_id` | VARCHAR(20) | PK, NN | レース ID |
| `model_version` | VARCHAR(50) | PK, NN | モデルバージョン |
| `furlong_index` | SMALLINT | PK, NN | ハロン番号 |
| `predicted_lap_sec` | NUMERIC(4,2) | | 予測ラップタイム（秒） |
| `predicted_pace_cat` | VARCHAR(10) | | 予測ペース区分 |

---

### 7-8. コース統計キャッシュ

#### `course_stats_cache` — コース別成績統計キャッシュ

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | INTEGER | PK, NN | |
| `track` | VARCHAR(20) | NN | コース名 |
| `distance` | INTEGER | NN | 距離（m） |
| `surface` | VARCHAR(10) | NN | 馬場種別 |
| `track_condition` | VARCHAR(10) | NN | 馬場状態 |
| `stat_type` | VARCHAR(30) | NN | 統計種別（例: `jockey` / `trainer` / `horse`） |
| `stat_key` | VARCHAR(50) | NN | 統計対象 ID |
| `n_runs` | INTEGER | | サンプル数 |
| `win_rate` | NUMERIC(5,4) | | 勝率 |
| `place_rate` | NUMERIC(5,4) | | 連対率 |
| `roi_win` | NUMERIC(7,4) | | 単勝回収率 |
| `computed_at` | TIMESTAMPTZ | `*now()` | 算出日時 |

---

### 7-9. 運用・ユーザー管理

#### `scrape_runs` — スクレイピング実行ログ

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `run_id` | BIGINT | PK, NN | 実行 ID |
| `target_type` | VARCHAR(30) | NN | 対象カテゴリ（例: `race_result`） |
| `target_id` | VARCHAR(20) | NN | 対象 ID（race_id 等） |
| `status` | VARCHAR(10) | NN | 状態（`ok` / `error` / `skip`） |
| `retry_count` | SMALLINT | `*0` | リトライ回数 |
| `started_at` | TIMESTAMPTZ | NN | 開始日時 |
| `finished_at` | TIMESTAMPTZ | | 終了日時 |
| `error_message` | TEXT | | エラーメッセージ |
| `gcs_path` | TEXT | | 保存先 GCS パス |

#### `users` — ユーザー

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | UUID | PK, NN, `*gen_random_uuid()` | ユーザー ID |
| `password_hash` | VARCHAR(255) | NN | パスワードハッシュ |
| `created_at` | TIMESTAMPTZ | `*now()` | |

#### `user_favorites` — お気に入り馬

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | INTEGER | PK, NN | |
| `user_id` | UUID | NN, FK→users.id | ユーザー ID |
| `horse_id` | VARCHAR(20) | NN | 馬 ID |
| `horse_name` | VARCHAR(100) | | 馬名（非正規化コピー） |
| `created_at` | TIMESTAMPTZ | `*now()` | |

> **UNIQUE**: `(user_id, horse_id)`

#### `saved_analyses` — 保存済み分析条件

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | UUID | PK, NN, `*gen_random_uuid()` | |
| `user_id` | UUID | FK→users.id | ユーザー ID |
| `name` | VARCHAR(100) | NN | 分析名 |
| `analysis_type` | VARCHAR(20) | NN | 分析種別 |
| `filter_conditions` | JSONB | NN | フィルタ条件（JSON） |
| `created_at` | TIMESTAMPTZ | `*now()` | |
| `last_run_at` | TIMESTAMPTZ | | 最終実行日時 |

#### `notification_settings` — 通知設定（F-09）

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | INTEGER | PK, NN | |
| `user_id` | UUID | NN, FK→users.id | ユーザー ID |
| `email` | VARCHAR(255) | | 通知先メールアドレス |
| `notify_favorite_race` | BOOLEAN | `*true` | お気に入り馬出走通知フラグ |
| `created_at` | TIMESTAMPTZ | `*now()` | |
| `updated_at` | TIMESTAMPTZ | `*now()` | |

#### `notification_logs` — 通知送信ログ（F-09）

| カラム | 型 | 制約 | 説明 |
|---|---|---|---|
| `id` | INTEGER | PK, NN | |
| `user_id` | UUID | NN, FK→users.id | ユーザー ID |
| `race_id` | VARCHAR(20) | NN | 通知対象レース ID |
| `horse_id` | VARCHAR(20) | NN | 通知対象馬 ID |
| `sent_at` | TIMESTAMPTZ | `*now()` | 送信日時 |
| `status` | VARCHAR(20) | `*sent` | 送信状態（`sent` / `failed`） |

---

### 7-10. テーブル関連図（主要 FK）

```
users ──────────────────────────────────────────────────┐
  │ id                                                   │
  ├── user_favorites.user_id                             │
  ├── saved_analyses.user_id                             │
  ├── notification_settings.user_id                      │
  └── notification_logs.user_id                          │
                                                         │
races ──────────────────────────────────────────────────┤
  │ race_id                                              │
  ├── entries.race_id                                    │
  ├── race_results.race_id                               │
  ├── race_lap_times.race_id                             │
  ├── race_corner_positions.race_id                      │
  ├── race_pace_summary.race_id                          │
  ├── race_odds_snapshot.race_id                         │
  ├── megu_index.race_id                                 │
  └── prediction_results.race_id                         │
                                                         │
horses ─────────────────────────────────────────────────┤
  │ horse_id                                             │
  ├── entries.horse_id                                   │
  ├── race_results.horse_id                              │
  ├── race_corner_positions.horse_id                     │
  ├── race_odds_snapshot.horse_id                        │
  ├── horse_stats_snapshot.horse_id                      │
  ├── megu_index.horse_id                                │
  └── prediction_results.horse_id                        │
                                                         │
sires ──────────────────────────────────────────────────┘
  │ sire_id
  └── horses.sire_id

jockeys
  │ jockey_id
  ├── entries.jockey_id
  ├── race_results.jockey_id
  └── jockey_stats_snapshot.jockey_id (非 FK・論理参照)

trainers
  │ trainer_id
  ├── entries.trainer_id
  └── trainer_stats_snapshot.trainer_id (非 FK・論理参照)
```

### 7-11. Alembic マイグレーション履歴

| バージョン | 内容 |
|---|---|
| `001_initial` | Layer 1〜5 + スナップショット + 予測結果 + ユーザー基本テーブル |
| `002_user_favorites_notifications` | user_favorites / notification_settings / notification_logs（F-12/F-09） |
| `003_megu_index` | megu_index / megu_par_time / megu_regression_params（AREA-11） |
| `004_megu_index_split_columns` | megu_index に `front_split_sec`, `split_point_m`, `tsi_raw` を追加 |
| `005_megu_index_computation_status` | megu_index に `computation_status` を追加、`delta_track_sec` を NULL 許容に変更 |
| `006_megu_par_class_bucket` | megu_par_time に `class_bucket` を追加、UNIQUE 制約を `class_bucket` 含む形に更新 |
| `007_megu_delta_track` | megu_delta_track テーブルを新規追加（date × venue × surface の馬場補正値） |
