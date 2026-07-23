# DEC-025: GCS → PostgreSQL データ補完・対応表・定期実行・モニタリング

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-23 |
| **ステータス** | accepted（現状整理 + 推奨運用） |
| **関連 AREA** | AREA-04（cron / SLA）, AREA-06（GCS / ETL / PG スキーマ）, AREA-10（監視） |
| **実装の正** | `src/scripts/data/etl_stg_db.py`, `src/scripts/data/etl_ingest_race.py`, `src/db/etl/transform.py`, `src/api/monitor_coverage.py`, `src/api/quality_health.py` |

---

## 1. この文書の目的

競馬データ基盤において、次の 4 点を **誰でも追えるレベル** で整理する。

1. GCS から STG / PROD の PostgreSQL へデータが **どのロジックで補完されるか**
2. 各 GCS データ（パス）が PostgreSQL の **どの DB・どのテーブル** に紐づくか（理想は 1 対 1）
3. GCS → PostgreSQL の補完が **毎日の cron で走っているか**。未設定なら **GCS read コストを抑えた戦略**
4. GCS / PostgreSQL の **開催日・開催場・データポイント** が **どうモニタリングされているか**

> **用語**
> - **GCS**: Google Cloud Storage。スクレイピング結果 JSON の永続ストア（`HybridStorage` 経由）。
> - **PostgreSQL（PG）**: アプリ・めぐ指数・モニターが参照する RDBMS。環境変数 `DATABASE_URL` で接続先が決まる。
> - **STG / PROD**: スキーマは同一（Alembic 管理）。DB 名例: `keiba_db_stg`, `keiba_db_prod`（`.env.stg` / `.env.prod`）。
> - **補完**: GCS にある JSON を読み、PG テーブルに upsert すること。

---

## 2. 全体像（2 段パイプライン）

データは **必ずしも GCS 保存と同時に PG へ入らない**。現状は次の 2 段構成である。

```
[段階 A] スクレイピング cron（SLA 0〜6 + バックフィル）
    netkeiba → HybridStorage.save() → GCS JSON
    （同時にローカル索引 date_coverage / quality_health を更新）

[段階 B] ETL（GCS → PostgreSQL）  ← ★ cron 未設定。手動 / 品質修復時のみ
    race_lists から対象 race_id を列挙
    → GCS から JSON を load（race_shutuba + race_result をマージ）
    → PG へ upsert（races, entries, horses, …, race_results）
```

**重要**: 7/18・7/19 の事例でも、**段階 A（cron）は成功**していたが、**段階 B（確定結果の PG 反映）は weekly-update 待ち + ETL 未実行**のため PG が空のままだった。GCS と PG のズレは「障害」ではなく **設計上の非同期** が主因である。

---

## 3. GCS → PostgreSQL 補完ロジック（詳細）

### 3-1. バッチ ETL（STG 運用の主経路）— `etl_stg_db.py`

**コマンド例**

```bash
# STG DB へ投入（DATABASE_URL が keiba_db_stg を指すこと）
KEIBA_ENV=stg python3 -m src.scripts.data.etl_stg_db --recent-days 30
python3 -m src.scripts.data.etl_stg_db --year 2026 --batch-size 50
python3 -m src.scripts.data.etl_stg_db --dry-run   # GCS 読み取りのみ確認
```

**処理フロー（1 レース = `process_race()`）**

| 順 | 処理 | 入力 | 出力（PG テーブル） |
|----|------|------|---------------------|
| 1 | 対象日付の列挙 | `data/page_reference/race_lists/{YYYYMMDD}.json` | — |
| 2 | 対象 race_id 一覧 | 上記 JSON の `races[].race_id`（JRA のみは race_lists 側でフィルタ） | — |
| 3 | 出走表 + メタのマージ読込 | GCS `race_shutuba` + GCS `race_result` → `load_merged_race_card()` | — |
| 4 | レースマスタ upsert | マージ済み card | `races` |
| 5 | 出走馬・関係者 upsert | card.entries[] | `horses`, `jockeys`, `trainers`, `entries` |
| 6 | 確定結果 upsert | GCS `race_result` を別途 load | `race_results` |
| 7 | （STG のみ）モック予測 | 出走馬 ID 一覧 | `prediction_results`（`model_version=stg-mock-v1`） |

**マージの意味** (`src/utils/race_card_merge.py`)

- `race_shutuba` を正本とし、`race_result` で **レース名・馬名・性齢などの文字化け・欠損** を補完する。
- PG へ入る `races` / `entries` は **マージ後の card** がソースである。

**対象日付の決め方**

- `--recent-days N`: race_lists に存在し、今日から N 日以内の日付のみ。
- `--year YYYY`: その年の race_lists ファイルすべて。
- デフォルト（引数なし）: race_lists 上の **2026 年全日**（重いので本番 cron では `--recent-days` 推奨）。

**接続先 DB**

- `DATABASE_URL` 環境変数（未設定時 `keiba_db_stg`）。
- **STG も PROD も同じコード**。`.env.stg` / `.env.prod` の `DATABASE_URL` だけが違う。

**初回 STG 構築時**

- `scripts/server/setup_stg.sh` がバックグラウンドで `etl_stg_db --year 2026` を **1 回** 起動する（デプロイ時のブートストラップ。**毎日 cron ではない**）。

### 3-2. 単体 ETL（設計上の Layer 1〜5 経路）— `etl_ingest_race.py`

AREA-06 §4 で定義された **カテゴリ単位の正規 ETL**。1 race_id ずつ GCS を読み、`src/db/etl/transform.py` で PG に upsert する。

```bash
python3 -m src.scripts.data.etl_ingest_race race_shutuba 202602011101
python3 -m src.scripts.data.etl_ingest_race race_result 202602011101
python3 -m src.scripts.data.etl_ingest_race race_result_lap 202602011101
```

| category 引数 | transform 関数 | PG テーブル（Layer） |
|---------------|----------------|----------------------|
| `race_shutuba` | `transform_shutuba()` | `races`, `entries`, `horses`, `jockeys`, `trainers`（Layer 1） |
| `race_result`, `race_result_on_time` | `upsert_results()` | `race_results`（Layer 2） |
| `race_result_lap` | `upsert_lap_times()` 等 | `race_lap_times`, `race_corner_positions`, `race_pace_summary`（Layer 4） |
| `race_odds` | **未実装（TODO）** | `race_odds_snapshot`（Layer 5） |

各成功 run は `scrape_runs` テーブルに `SUCCESS` ログが残る（監査用）。

**現状**: バッチ運用の主経路は **3-1 の `etl_stg_db` のみ**。3-2 は手動・将来の完全 ETL 用。

### 3-3. 品質修復時の自動 PG 同期 — `quality_auto_remediation.py`

`/monitor` の品質チェック（STG）で fail/warn になった日について、修復プラン実行時に **限定的に PG 同期** する。

```
修復順: GCS メタ補完 → スクレイプキュー投入 → sync_db → megu 再計算
```

- `sync_db`: `etl_stg_db.process_race()` を **最大 50 race_id** まで逐次実行（1 件 commit）。
- **リアクティブ**（品質 NG 時のみ）。全開催日の定期同期ではない。

### 3-4. めぐ指数（Calculated）— GCS ではなく PG 直書き

`megu_index` テーブルは GCS JSON から ETL するのではなく、**パイプラインが PG に直接 upsert** する。

- 入力: `data/page_reference/tables/{年}/race_result_flat.parquet`（ローカル flat）+ PG の `race_results`
- 実行: `src/pipeline/megu_index/compute.py`（`raceday-evening` 後・品質修復後など）
- モニター Calculated タブ: `megu_index`（`model_version='v2'`）の頭数カバレッジを SQL 集計

---

## 4. GCS パス ↔ PostgreSQL 対応表

### 4-1. GCS 物理パス（SSoT: `src/config/data_paths.py`）

| 種別 | blob パス（バケット内相対） | 例 |
|------|---------------------------|-----|
| レース単位 | `chuou/data/preprocessed/netkeiba/pc/{category}/{year}/{race_id}.json` | `.../race_result/2026/202602011101.json` |
| 馬単位 | `chuou/data/preprocessed/netkeiba/pc/{category}/{prefix}/{horse_id}.json` | `.../horse_profile/2024/2024109107.json` |
| その他 | `chuou/data/others/{category}/{key}.json` | `jra_cushion` 等 |

フル URI: `gs://${GCS_BUCKET}/chuou/data/preprocessed/netkeiba/pc/...`  
（本番バケット例: `magu-keiba-horse-racing-ai`）

**GCS に載らない（ローカル正本）**

| パス | 用途 |
|------|------|
| `data/page_reference/race_lists/{YYYYMMDD}.json` | 開催日・race_id 一覧（ETL の日付軸） |
| `data/page_reference/race_day_schedule/{YYYYMMDD}.json` | 発走時刻 |
| `data/local/meta/date_coverage/{YYYY}/{YYYYMMDD}.json` | カテゴリ別充足率索引（GCS list 結果のキャッシュ） |
| `data/page_reference/tables/{YYYY}/race_result_flat.parquet` | flat 特徴量（めぐ指数・モニター Calculated） |

### 4-2. スクレイピングカテゴリ → PostgreSQL テーブル（1 対 1 理想 vs 現状）

凡例: **✅ ETL 実装済** / **△ 部分（etl_stg_db のみ等）** / **❌ GCS のみ（PG 未接続）** / **📁 ローカルのみ**

#### レース単位（Raw）

| GCS category | GCS パスキー | PostgreSQL（理想 1:1） | 現状 ETL | 備考 |
|--------------|-------------|------------------------|----------|------|
| `race_shutuba` | `{race_id}` | `races` + `entries` + マスタ | ✅ △ | `etl_stg_db` / `etl_ingest_race` |
| `race_result` | `{race_id}` | `race_results` | ✅ △ | 確定結果。weekly-update 後に GCS 充足 |
| `race_result_on_time` | `{race_id}` | （理想: `race_results` 速報用） | ❌ | PG には通常 `race_result` のみ反映 |
| `race_result_lap` | `{race_id}` | `race_lap_times`, `race_corner_positions`, `race_pace_summary` | △ | `etl_ingest_race` のみ。バッチ ETL 未接続 |
| `race_odds` | `{race_id}` | `race_odds_snapshot` | ❌ | transform に TODO |
| `race_pair_odds` | `{race_id}` | （専用テーブルなし） | ❌ | 現状 PG 未設計 |
| `race_index` | `{race_id}` | （専用テーブルなし） | ❌ | タイム指数 JSON。PG 未接続 |
| `race_paddock` | `{race_id}` | （専用テーブルなし） | ❌ | |
| `race_barometer` | `{race_id}` | （専用テーブルなし） | ❌ | netkeiba 未掲載レースあり |
| `race_oikiri` | `{race_id}` | （専用テーブルなし） | ❌ | |
| `race_trainer_comment` | `{race_id}` | （専用テーブルなし） | ❌ | |
| `race_shutuba_past` | `{race_id}` | （専用テーブルなし） | ❌ | |
| `smartrc` / `smartrc_race` | `{race_id}` | （専用テーブルなし） | ❌ | SmartRC 連携 JSON |

#### 馬単位

| GCS category | GCS パスキー | PostgreSQL（理想） | 現状 |
|--------------|-------------|-------------------|------|
| `horse_profile` | `{horse_id}` | `horses` 拡張 | ❌（週次は GCS のみ更新） |
| `horse_result` | `{horse_id}` | 出走歴（将来 `horse_past_races` 等） | ❌ |
| `horse_ped` / `horse_pedigree_5gen` | `{horse_id}` | 血統（将来） | ❌ |

#### 計算・派生（GCS 外）

| データ | 保存先 | PostgreSQL | 備考 |
|--------|--------|------------|------|
| めぐ指数 v2 | PG 直 | `megu_index` | `compute_for_date()` |
| めぐ PAR / 回帰係数 | PG | `megu_par_time`, `megu_regression_params` 等 | 別バッチ |
| flat レース結果 | ローカル parquet | 間接（megu 計算入力） | GCS read 不使用 |
| STG モック予測 | — | `prediction_results` | ETL 時に自動生成（本番モデルではない） |

#### 派生カテゴリ（物理 blob なし）

`race_result_meta`, `race_result_lap_times` 等は **親 JSON から抽出** された別パス blob。PG への独立 ETL は未整備。モニター上は **親があれば N/A** 扱い（`DERIVED_CATEGORY_PARENT`）。

### 4-3. 環境と DB の対応

| 環境 | `KEIBA_ENV` | PostgreSQL | GCS |
|------|-------------|------------|-----|
| dev | `dev` | 任意（ローカル `keiba_db`） | 通常オフ / ローカル JSON |
| stg | `stg` | `keiba_db_stg`（`.env.stg`） | オン |
| prod | `prod` | `keiba_db_prod`（`.env.prod`） | オン（同一バケット or 環境別は `.env` 次第） |

**スキーマ**: 3 環境とも Alembic 同一。テーブル定義の SSoT は `src/db/models.py`。

---

## 5. 定期実行（cron）の現状と推奨戦略

### 5-1. 現状サマリ（2026-07-23 実装済み）

| ジョブ種別 | cron 有無 | 内容 | GCS → PG |
|------------|-----------|------|----------|
| スクレイピング SLA 0〜6 | ✅ 毎日 | GCS へ JSON 保存 | 同期は別ジョブ |
| **`sync_pg_from_gcs.sh`（日次）** | **✅ 毎日 00:00 JST** | 直近 **7** 日を増分 ETL | **実施** |
| **スクレイプ後フック** | **✅ イベント駆動** | eve / evening / weekly 完了後 | **対象日のみ** |
| **weekly 後処理** | **✅ weekly 内蔵** | ① date_coverage ② めぐ→GCS ③ PG ETL | **直近10日開催日** |
| 品質修復 `sync_db` | △ イベント駆動 | fail/warn 日・最大 50R | 部分的 |
| めぐ指数 `compute_for_date` | △ | evening / 修復後 | PG 直（GCS read 最小） |

**実装ファイル**

- ETL: `src/scripts/data/etl_stg_db.py`（`run_etl()`, `--skip-if-pg-complete`）
- cron: `scripts/cron/sync_pg_from_gcs.sh` → `setup_all_cron.sh` の `KEIBA_PG_SYNC`
- スクレイプ連動: `src/scraper/auto_scrape.py` の `_trigger_pg_sync_for_dates()`
- prod モニターキャッシュ: ETL 後に `build_db_coverage_cache --recent-days`

**無効化**

- スクレイプ後同期: `KEIBA_PG_SYNC_AFTER_SCRAPE=0`
- cron 環境: `KEIBA_PG_SYNC_ENVS=stg`（デフォルト）。prod も同期する場合 `KEIBA_PG_SYNC_ENVS="stg prod"`

### 5-2. 増分 ETL の skip ロジック（GCS read 最小化）

#### 方針（実装済み）

1. **対象を「直近 N 日の開催日」に限定**（デフォルト `--recent-days 7` / `KEIBA_PG_SYNC_RECENT_DAYS=7`）。
2. **PG 側で充足している race_id はスキップ**（`query_pg_status_batch` + `filter_races_needing_sync`）。
3. **GCS JSON download は `process_race()` 内のみ**。存在確認は `batch_list_blobs`（list のみ、年単位 2 カテゴリ）。
4. **実行タイミング**: 毎日 **00:00 JST** cron 1 回 + スクレイプ完了直後（eve / evening / weekly）。夕方 cron（旧 18:15）はコスト削減のため廃止。

**コスト目安（GCS 金銭）**: 定常時は list のみで **月 $0.1 未満**。主コストは ETL CPU 時間。`recent-days=7` + cron 1 回/日で list 判定は **約 75% 削減**（旧: 14 日 × 2 回/日）。

#### skip 条件（`--skip-if-pg-complete` デフォルト ON）

| PG 状態 | GCS race_result | 動作 |
|---------|-----------------|------|
| races/entries なし | shutuba あり | **同期**（GCS download） |
| entries あり | result あり、finishers 0 | **同期**（結果投入） |
| finishers あり | result あり | **スキップ** |
| entries あり | result なし | **スキップ**（結果未公開） |
| shutuba なし | — | **スキップ** |

実行サマリ: `data/local/meta/etl_sync/{YYYY}/{YYYYMMDD}.json`

#### cron 登録

```bash
bash scripts/cron/setup_all_cron.sh install
# 手動: KEIBA_ENV=stg bash scripts/cron/sync_pg_from_gcs.sh
```

#### 将来改善（read コストさらに削減）

| 施策 | 効果 |
|------|------|
| `data/local/meta/etl_sync/{YYYYMMDD}.json` に最終同期 race_id を記録 | 同一 race の再 download 回避 |
| `scrape_runs` / GCS `_meta.scraped_at` と PG `race_results.updated_at` 比較 | 変更分のみ ETL |
| `etl_stg_db` に `--skip-if-pg-complete` フラグ追加 | PG に finishers ありなら GCS load スキップ |
| Layer 4・5 は `etl_ingest_race` を週次 1 回のみ | odds / lap の read 削減 |

#### weekly-update との役割分担（再確認）

| タイミング | 担当 | GCS | PG |
|------------|------|-----|-----|
| 開催日 17:30 `raceday-evening` | 速報 + オッズ | `race_result_on_time` 等 | **増分 ETL**（`KEIBA_PG_SYNC_AFTER_SCRAPE=1`） |
| 金曜 18:00 `weekly-update` | 確定 `race_result` 等（**直近10日**開催日） | GCS 更新 | **増分 ETL** + **めぐ指数 GCS 正本** |
| **（推奨）ETL cron** | GCS → PG | 読む | **upsert**（`recent-days=7`） |

**weekly 後処理**（`finalize_weekly_update`）:

1. `date_coverage` 一括更新（対象開催日）
2. `compute_megu_for_opening_dates(..., gcs_canonical=True)` → GCS 正本 parquet
3. `_trigger_pg_sync_for_dates` → 対象日のみ ETL

無効化: `KEIBA_WEEKLY_MEGU_BATCH=0` / `KEIBA_PG_SYNC_AFTER_SCRAPE=0`

---

## 6. モニタリング

### 6-1. モニターポータル `/monitor`

| UI | API | 見ているもの |
|----|-----|--------------|
| Raw タブ（STG） | `GET /api/date-raw-matrix?view=stg` | **GCS 存在** + **PG 行数** |
| Calculated タブ | `GET /api/date-calculated-matrix?view=stg` | flat parquet + **megu_index** |
| 品質ヘルス | `GET /api/quality-check/health?date=` | 3 段チェック結果 |
| カバレッジカレンダー | `GET /api/coverage-calendar` | 日別 GCS 充足率（ローカル索引） |
| 非開催日 | `GET /api/monitor/opening-date-info` | 「非開催（対象外）」ラベル |

**STG Raw マトリクスの列**

- **GCS 側**: `TRACK_CATEGORIES` 各カテゴリについて、レース行が ✅ / ❌ / N/A（派生・障害等）。
- **PG 側**: `pg_races`（races 行あり）, `pg_entries`（頭数）, `pg_race_results`（着順+タイムあり頭数）。

**課金安全設計**（`monitor_coverage.py` 先頭コメント）

- GCS: **`batch_list_blobs` のみ**（JSON download しない）。
- PG: STG/DEV は **リアルタイム SQL**。
- PROD: **`data/local/meta/db_coverage/{YYYY}/{YYYYMMDD}.json` キャッシュ**（日次更新想定）。

### 6-2. 品質ヘルス 3 段チェック

保存先: `data/local/meta/quality_health/{YYYY}/{YYYYMMDD}.json`

| チェック | 内容 | GCS | PG |
|----------|------|-----|-----|
| ① presence | カテゴリ存在 + PG 結果行 | list ベース充足率 | `pg_race_results > 0` |
| ② raw_content | race_result の必須フィールド・文字化け | **load**（内容検査） | 間接 |
| ③ calculated | 入着馬の megu カバレッジ | 不使用 | `megu_index` SQL |

非開催日（`opening_kind=no_meeting`）は **対象外（na）** として表示。修復ボタン無効。

### 6-3. ローカル索引（スクレイプ完了時更新）

| ファイル | 更新タイミング | 用途 |
|----------|----------------|------|
| `date_coverage/{YYYY}/{YYYYMMDD}.json` | スクレイプ / evening 後 | カレンダー色・カテゴリ別件数 |
| `quality_health/...` | 品質チェック実行後 | 品質ドット |
| `not_available/{category}/{year}.json` | 空レスポンス記録 | 未取得 vs 存在しないの区別 |
| `db_coverage/...` | （手動 / 将来 cron） | PROD PG キャッシュ |

### 6-4. 開催日・開催場・データポイントの見え方

1. **開催日**: `race_lists/{YYYYMMDD}.json` が正本。モニター日付一覧は **JRA 開催日のみ** フィルタ（非開催は「対象外」）。
2. **開催場**: race_id の venue コード（例: `202602011101` → 函館）+ `races.venue`。
3. **データポイント**: Raw マトリクスが **レース × カテゴリ** のセル。PG は **レース × (races/entries/results)**。

### 6-5. モニタリングのギャップ（改善余地）

| ギャップ | 影響 | 改善案 |
|----------|------|--------|
| PG ETL 未 cron | GCS–PG ズレが長期化 | §5-2 の日次 ETL |
| Layer 4/5 が PG マトリクスに未表示 | lap/odds の PG 充足が見えない | Calculated / 別タブ拡張 |
| PROD db_coverage 自動更新なし | prod モニターが stale | 日次 `build_db_coverage_cache_for_date` cron |
| アラート通知未決（AREA-10 OP-4） | サイレント劣化 | Slack 等 + 品質 fail 連携 |

---

## 7. 運用チェックリスト（開催週）

1. **土日開催後**: `/monitor` で対象日 Raw タブ → GCS 100% / PG `pg_race_results` 100% を確認。
2. **金曜 weekly-update 後**: 直近10日開催日の `race_result` が GCS 100% であること、`auto_scrape_status.json` の `megu_batch` / `pg_sync` を確認（`logs/weekly_update.log`）。
3. **PG が遅れている場合**:
   ```bash
   KEIBA_ENV=stg python3 -m src.scripts.data.etl_stg_db --recent-days 7
   ```
4. **品質 fail 時**: STG で「自動修復」→ `sync_db` + 再チェック（`quality_auto_remediation`）。
5. **ログ**: `logs/raceday_*.log`（GCS）、`logs/etl_stg_*.log`（PG、手動時）。

---

## 8. 関連ドキュメント

- [AREA-06-data.md](./AREA-06-data.md) — GCS パス SSoT・5 層スキーマ・ETL パイプライン全体像
- [AREA-04-ops.md](./AREA-04-ops.md) — cron SLA 0〜6 スケジュール
- [AREA-10-infra-monitoring.md](./AREA-10-infra-monitoring.md) — Watchdog・リソース監視
- [DEC-014-db-schema-alignment.md](./DEC-014-db-schema-alignment.md) — PG スキーマ正本
- [docs/operations/service-endpoints.md](../operations/service-endpoints.md) — API・CLI 一覧

---

## 9. 決定事項（要約）

1. **GCS → PG の主経路は `etl_stg_db.run_etl()`**（shutuba+result マージ → Layer 1/2）。完全 Layer 1〜5 は `etl_ingest_race` が将来経路。
2. **日次 cron + スクレイプ後フックで PG を自動追従**（`sync_pg_from_gcs.sh` / `_trigger_pg_sync_for_dates`）。
3. **`--skip-if-pg-complete` で GCS download を最小化**（list + PG SQL のみで skip 判定）。
4. **モニタリング**は `/monitor`（GCS list + PG SQL + 品質 3 段）で開催日単位。非開催は「対象外」表示。
5. **1 対 1 対応**は Raw カテゴリのうち **`race_shutuba`→entries系・`race_result`→race_results のみ実装済**。他カテゴリは GCS のみが正。
