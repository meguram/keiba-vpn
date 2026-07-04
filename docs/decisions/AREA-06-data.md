# AREA-06 — データ管理要件（GCS パス設計 SSoT, ETL パイプライン, Feature Store, Redis TTL 設計）
**Status**: FINAL | **Last Updated**: 2026-07-04 | **Consolidates**: DEC-001（統合済み）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトにおけるデータ管理要件を定義する。対象範囲は以下の4領域とする。

1. **GCS パス設計（`gcs_paths.py` SSoT）**
2. **ETL パイプライン**
3. **Feature Store（特徴量スナップショット管理）**
4. **Redis TTL 設計**

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

- `data_paths.py` をパス定義の **Single Source of Truth（SSoT）** とし、全モジュールからインポートして使用する
- バケット名は環境変数 `GCS_BUCKET` から注入する（ハードコード禁止）
- スクレイピング済み JSON は **カテゴリ名ディレクトリ** 配下に配置する（layer 番号は使わない）
- 生データと変換後データの区別はしない — GCS は「単一の永続 JSON ストア」として機能する

### 3-2. バケット構成（実装準拠）

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
    └── data/
        └── others/
            └── {category}/{key}.json                         ← その他データ（jra_cushion 等）
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

```python
# data_paths.py  ── GCS / ローカルパス定義 SSoT
import os

GCS_BUCKET  = os.environ["GCS_BUCKET"]   # 環境変数必須
GCS_BASE    = "chuou/data/preprocessed/netkeiba/pc"
GCS_OTHERS  = "chuou/data/others"

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
```

**主要カテゴリとパス例**:

| カテゴリ | パス例 |
|---|---|
| `race_detail` | `chuou/data/preprocessed/netkeiba/pc/race_detail/2025/202505010101.json` |
| `race_result` | `chuou/data/preprocessed/netkeiba/pc/race_result/2025/202505010101.json` |
| `race_odds` | `chuou/data/preprocessed/netkeiba/pc/race_odds/2025/202505010101.json` |
| `race_shutuba` | `chuou/data/preprocessed/netkeiba/pc/race_shutuba/2025/202505010101.json` |
| `horse_result` | `chuou/data/preprocessed/netkeiba/pc/horse_result/2000/2000110001.json` |
| `horse_pedigree_5gen` | `chuou/data/preprocessed/netkeiba/pc/horse_pedigree_5gen/2000/2000110001.json` |
| `smartrc_race` | `chuou/data/preprocessed/netkeiba/pc/smartrc_race/2025/202505010101.json` |
| `jra_cushion` | `chuou/data/others/jra_cushion/2025.json` |

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
netkeiba.com / SmartRC / JRA
    │ HTTP スクレイピング（SLA 0〜6 + バックフィル）
    ▼
[Scrape]  スクレイパー → L1 メモリキャッシュ → L2 ディスクキャッシュ → L3 GCS JSON
                          (HybridStorage、data_paths.py 経由)
    │
    ▼
[Parse / Normalize]  JSON → 正規化レコード → PostgreSQL（Layer 1〜5 テーブル）
    │
    ▼
[Snapshot Batch]  Layer 3 スナップショット生成（as_of_race_id 付与）
                  → PostgreSQL horse/jockey/trainer_stats_snapshot
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
| 前日準備 | SLA 1 (18:00) | race_shutuba / race_oikiri / horse_training / smartrc_race | 追走難度・最終オッズ precompute |
| 当日朝 | SLA 2 (05:00-08:50) | jra_cushion | GCS others/ 格納 |
| T-15バンドル | SLA 3 (各レース T-15分) | race_detail + race_odds + race_paddock + race_barometer + race_trainer_comment + smartrc_race + JRA馬場ライブ | AI 予測トリガ |
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

SmartRC:
  リクエスト間隔: 2.0〜5.0 秒 / 日次上限: 1,000 req
  robots.txt 準拠、ブロック検知時に即停止
```

### 4-4. Transform ルール

| 対象カテゴリ | 変換内容 | 格納先 |
|---|---|---|
| `race_shutuba` | 出走馬・騎手・枠順・オッズ抽出 | GCS + PostgreSQL Layer 1 |
| `race_result` | 着順・タイム・馬体重・コーナー通過順 | GCS + PostgreSQL Layer 2 |
| `race_result_lap` | 1F毎ラップ・ペース区分 | GCS + PostgreSQL Layer 4 |
| `race_odds` | 単複オッズ・snapshot_at 付きレコード | GCS + PostgreSQL Layer 5 |
| `horse_result` | 勝率・連対率・複勝率・脚質スコア集計 | GCS + PostgreSQL Layer 3 スナップショット |
| `jra_cushion` | クッション値・含水率（PDF 解析） | GCS others/jra_cushion/{year}.json |
| `smartrc_race` | cr_value / first_furlong_time / estimated_popularity | GCS + 推論特徴量 |

### 4-5. 実行管理（`scrape_runs` テーブル）

```sql
CREATE TABLE scrape_runs (
    run_id         BIGSERIAL    PRIMARY KEY,
    target_type    VARCHAR(30)  NOT NULL,   -- 'race_card' / 'race_result' / 'odds' / 'horse_history'
    target_id      VARCHAR(20)  NOT NULL,   -- race_id または horse_id
    status         VARCHAR(10)  NOT NULL    -- 'SUCCESS' / 'FAILED' / 'RETRY'
                   CHECK (status IN ('SUCCESS','FAILED','RETRY')),
    retry_count    SMALLINT     DEFAULT 0,
    started_at     TIMESTAMPTZ  NOT NULL,
    finished_at    TIMESTAMPTZ,
    error_message  TEXT,
    gcs_path       TEXT                     -- 生データ保存先 GCS パス
);
```

- スクレイピング成功率の目標値: **≥ 99% / 月**
- DB 反映遅延の目標値: **≤ 10 分**（スクレイピング完了から）

---

## 5. Feature Store（特徴量スナップショット管理）

### 5-1. 設計原則

- Layer 3 の集計値は必ず **`as_of_race_id`（予測対象レース）に紐付けて保存** し、そのレース以後の情報を含めない（テンポラルリーク防止の根幹）
- スナップショットは **追記専用・不変（Immutable）** とし、更新・削除を禁止する
- DB には `UNIQUE(entity_id, as_of_race_id)` 制約を設け、二重登録を防止する

### 5-2. Feature Store テーブルスキーマ

#### `horse_stats_snapshot`

```sql
CREATE TABLE horse_stats_snapshot (
    snapshot_id          BIGSERIAL      PRIMARY KEY,
    horse_id             VARCHAR(20)    NOT NULL,
    as_of_race_id        VARCHAR(20)    NOT NULL,
    as_of_date           DATE           NOT NULL,
    win_rate_all         NUMERIC(5,4),
    win_rate_turf        NUMERIC(5,4),
    win_rate_dirt        NUMERIC(5,4),
    place_rate_all       NUMERIC(5,4),
    show_rate_all        NUMERIC(5,4),
    win_rate_distance    NUMERIC(5,4),
    win_rate_course      NUMERIC(5,4),
    avg_last_3f          NUMERIC(5,2),
    speed_index_avg      NUMERIC(6,2),
    speed_index_max      NUMERIC(6,2),
    running_style_score  NUMERIC(5,2),
    sample_count         SMALLINT,
    created_at           TIMESTAMPTZ    DEFAULT NOW(),
    UNIQUE (horse_id, as_of_race_id)
);
```

同様に `jockey_stats_snapshot`、`trainer_stats_snapshot` も `UNIQUE(entity_id, as_of_race_id)` 制約付きで定義する（スキーマ詳細は AREA-01-app-requirements.md を参照）。

### 5-3. スナップショット生成バッチ

- Layer 2 結果データ収集完了直後に自動トリガー
- 対象: 直前の `as_of_race_id` に紐付いたエンティティ（馬・騎手・調教師）の集計
- 集計期間: `race_date < as_of_race_id` の全過去成績（テンポラルリーク防止必須）
- 完了後に `horse_stats_snapshot` テーブルに INSERT（`ON CONFLICT DO NOTHING`）

### 5-4. テンポラルリーク防止チェックリスト

| チェック項目 | 担保方法 |
|---|---|
| 訓練データの時系列分割 | 常に過去レースで学習・未来レースで評価（ランダムシャッフル禁止） |
| スナップショット参照 | 推論時も同 race_id のスナップショットのみ使用 |
| オッズ特徴量 | 発走 N 分前の最終スナップショット固定使用 |
| CI テスト | テストデータ時系列分割によるリーク検知テスト自動実行（N-10） |

---

## 6. Redis TTL 設計

### 6-1. キャッシュキー体系

```
prediction:{race_id}:{model_version}       ← AI 予測結果
lap:prediction:{race_id}:{model_version}   ← ラップ予測結果
race:entries:{race_id}                     ← 出馬表
race:results:{race_id}                     ← 着順・ラップ
track:speed:{date}:{venue}                 ← TSI 指数
```

### 6-2. TTL ポリシー

| キャッシュキー | TTL | 無効化タイミング |
|---|---|---|
| `prediction:*` | 発走時刻まで有効 / 発走後 60 秒で自動失効 | 発走時刻 + 60s |
| `lap:prediction:*` | 同上 | 同上 |
| `race:entries:*` | 3,600 秒（出馬確定後は変化小） | 再スクレイピング完了時に明示的削除 |
| `race:results:*` | 300 秒（発走後 30 分で確定） | — |
| `track:speed:*` | 86,400 秒 | — |

### 6-3. キャッシュ整合性

- 推論バッチ完了時にキャッシュを `SET ... EX {ttl}` で上書き（`prediction:*` のみ）
- 出馬確定後の再スクレイピング完了時に `DEL race:entries:{race_id}` で強制削除
- 発走後は `prediction:*` を 60 秒 TTL で自動失効させ、発走後の古い予測表示を防ぐ

---

## 7. 未決定事項（後続 DEC で確定が必要な項目）

| # | 項目 | 理由 |
|---|---|---|
| DM-1 | GCS バケット命名規則（本番・ステージング分離） | `GCS_BUCKET` は env var — 命名規則は運用決定事項 |
| DM-2 | ディスクキャッシュ容量上限の明示 | 現状はアクセス頻度によるヒューリスティック削除のみ |
| DM-3 | GCS への書き込み失敗時のリトライ・アラート設計 | HybridStorage の障害挙動が未定義 |
| DM-4 | Feature Store の GCS バックアップ（DB スナップショット補完） | テーブルのみ永続化・GCS features/ は廃止した設計のため要確認 |