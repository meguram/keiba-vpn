# AREA-06 — データ管理要件（GCS パス設計 SSoT, ETL パイプライン, Feature Store, Redis TTL 設計）
**Status**: FINAL | **Last Updated**: 2026-07-06 | **Consolidates**: DEC-001（統合済み）, TASK-055（GCSバケット名更新・モデリング仕様統合）, DEC-009（モデル保持ポリシー）, DEC-015（特徴量スキーマ正規定義）

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
