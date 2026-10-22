# データをどこから取得するか

## config

- 日付サンプル: race_date = `20230625`
- レースサンプル: race_id = `202309030811`
- 馬サンプル: horse_id = `2019105219`
- **サンプル取得テスト**: `python3 tests/scraper/manual/requirements_sample_scrape_test.py`（`--quick` 省略時は `skip_existing=False` で再取得。馬調教のみ `max_pages=2` に上限）。**`--export-samples`** で取得戻り値を `docs/requirements/data/scrape_process_samples/*.json` に保存する（**このディレクトリは git 管理しない**。実データのサンプルは公開リポジトリに置かない）。下表「テスト結果」列は **2026-06-12** 実施分（netkeiba ログイン成功後に全項目を実行）。**各項目の `detail` には `schemas.validate` の結果（`schema=PASS` / `schema=FAIL …`）が付与される**（スクリプト改修後の再実行時）。
- **「PASS」の意味（重要）**: 表のテスト結果は **取得の可否と最低限の件数チェック** が主で、**ビジネス正しさ・将来の HTML 変更への耐性・欠損の許容範囲までは保証しません。** 保存 JSON の形・型の最低限は `src/scraper/schemas.py` の `validate(category, data)` で検査できます（`SCHEMA_VERSION` 付き）。**本番のキュー経路では `HybridStorage.save` が保存前に同検証を必ず実行**し、`_meta.schema_validation` / `_meta.scrape_validation_status` を付与します。既定（`KEIBA_SCHEMA_STRICT=1` または未設定）では不合格時は GCS へ書かず `SchemaValidationError` となり、ジョブは `failed`・`failure_reason=schema_validation`・エラーメッセージ先頭 `[schema_validation]` です。緩和のみ保存する場合は `KEIBA_SCHEMA_STRICT=0`。管理者向け一覧: `/monitor` のパネル、`GET /api/scrape-jobs` の `schema_validation_failures` と `stats.schema_validation_failed`、`GET /api/scrape-queue/status` の `queue.schema_validation_failed`。固定例によるユニットテスト: `python3 -m unittest tests.scraper.test_schema_examples_validate -v`。保存経路とスキーマの結合: `tests/scraper/test_hybrid_storage_schema.py`。**実スクレイプ JSON 正本（サンプル ID）**: `docs/requirements/data/scrape_process_samples/`（`requirements_sample_scrape_test.py --export-samples` で取得結果を保存し MD 更新。キャッシュのみ取り込みは `python3 -m src.scripts.docs.gen_scrape_process_samples --from-cache`。MD 再整形のみは `python3 -m src.scripts.docs.gen_scrape_process_samples`）。正本の説明と JSON Schema 例: `docs/requirements/data/schemas/README.md`。
- SLA（以下すべて日本時間）
    0. 毎日17:00
    1. レース前日17:00
    2. レース当日8:00/8:30/9:00
    3. レース当日出走10分前
    4. レース当日出走30分後
    5. レース当日18:00
    6. レース終了後、1回目に訪れる金曜日17:00
    7. その他（何らかのデータ取得語など、依存があるもの）

## netkeiba

実装の主たるエントリは `src/scraper/run.py` の `ScraperRunner`（パーサーは `src/scraper/parsers.py`）。レース一覧のトップ取得は `src/scraper/netkeiba_top_race_list.py` の関数群を併用。

**GCS 保存先の読み方**: バケットは環境変数 `GCS_BUCKET`。Blob 名は次のとおり（先頭に `gs://<バケット名>/` を付けたものがフル URI）。構造化 JSON は `HybridStorage`（`src/scraper/storage.py` の `GCS_BASE`）、gzip 生 HTML は `HtmlArchive`（`src/scraper/html_archive.py` の `GCS_RAW_BASE`）。HTML はカテゴリごとに件数上限があり全レース分が無い場合がある。

**行単位の設計（要件表 1 行 = 1 GCS パス）**: 下表の各行は **物理的にユニークな GCS パスを持つ**。複数行が同一のスクレイピング結果 JSON（`race_shutuba`・`race_result` 等）を共有していた箇所については、必要フィールドだけを抽出した **行固有派生カテゴリ**（`race_shutuba_meta`・`race_result_meta`・`horse_profile` 等）を新設し、そこへ格納する。派生カテゴリの生成・バックフィル: `python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths`（`--year 2026 --dry-run` でドライラン確認後、`--year-start 2020 --year-end 2026 --include-horses` で本番）。抽出ロジックは `src/scraper/row_data_extractor.py`。`row_id` ↔ 正本カテゴリの対応の単一ソース: `src/scraper/requirement_row_catalog.py`。行固有トレース（`requirement_row_trace`）のバックフィル: `python3 -m src.scripts.scraping.materialize_requirement_row_traces`。**発走時刻**の行固有データ（`nk_race_day_schedule`）は `race_day_schedule`（`data/page_reference/race_day_schedule/{YYYYMMDD}.json`・local_only）に保存し、`_fetch_race_schedule_storage` が**合成より優先**して読む。

| タイトル | 説明 | SLA(データ取得タイミング) | 該当リンク | ソースコードパス | 関数名 | テスト結果 (2026-06-12) | 正本 JSON（フィールド・role） / 生 HTML / 行固有トレース |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-- |
| 出馬表HTML | 任意のレースの出走馬テーブル | 1, 3 | https://race.netkeiba.com/race/shutuba.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_race_card` | PASS — entries=17 | 正本: `chuou/data/preprocessed/netkeiba/pc/race_shutuba/2023/202309030811.json`（`entries[]` / role=`entries_table`）<br>HTML: `chuou/data/raw/html/race_shutuba/2023/202309030811.html.gz`<br>**行トレース** (`nk_shutuba_entries`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_shutuba_entries.json` |
| レース情報HTML | 任意のレースのレース名やグレード、コース情報、馬場等の情報 | 1, 3 | https://race.netkeiba.com/race/shutuba.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_race_card` | PASS — race_name 取得可 | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_shutuba_meta/2023/202309030811.json`（`race_name`/`surface`/`distance` 等メタ。`race_shutuba` から `entries` を除いて抽出）<br>HTML: `chuou/data/raw/html/race_shutuba/2023/202309030811.html.gz`（出馬表と共通 URL・共通 HTML）<br>**行トレース** (`nk_shutuba_race_meta`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_shutuba_race_meta.json` |
| レースタイム指数HTML | 任意のレースの出走馬タイム指数 | 1 | https://race.netkeiba.com/race/speed.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_speed_index` | PASS — entries=17 | 正本: `chuou/data/preprocessed/netkeiba/pc/race_index/2023/202309030811.json`<br>HTML: `chuou/data/raw/html/race_index/2023/202309030811.html.gz`<br>**行トレース** (`nk_speed_index`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_speed_index.json` |
| レース調子偏差値HTML | 任意のレースの出走馬調子偏差値 | 1 | https://race.sp.netkeiba.com/barometer/score.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_barometer` | PASS — entries=17 | 正本: `chuou/data/preprocessed/netkeiba/pc/race_barometer/2023/202309030811.json`（API のみ・生 HTML アーカイブなし）<br>**行トレース** (`nk_barometer`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_barometer.json` |
| レースパドックHTML | 任意のレースのパドック評価 | 3 | https://race.netkeiba.com/race/paddock.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_paddock` | PASS — entries=5（掲出頭数に依存） | 正本: `chuou/data/preprocessed/netkeiba/pc/race_paddock/2023/202309030811.json`<br>HTML: `chuou/data/raw/html/race_paddock/2023/202309030811.html.gz`<br>**行トレース** (`nk_paddock`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_paddock.json` |
| レースオッズHTML | 任意のレースの暫定オッズ | 3 | https://race.netkeiba.com/odds/index.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_odds` | PASS — entries=17 | 正本: `chuou/data/preprocessed/netkeiba/pc/race_odds/2023/202309030811.json`（API のみ・生 HTML アーカイブなし）<br>**行トレース** (`nk_odds`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_odds.json` |
| レース結果HTML | 任意のレースの結果テーブル（着順・タイム等） | 4, 5 | https://race.netkeiba.com/race/result.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_race_result_on_time` | PASS — entries=17 | 正本: `chuou/data/preprocessed/netkeiba/pc/race_result_on_time/2023/202309030811.json`（`entries[]` / role=`entries_table`）<br>HTML: `chuou/data/raw/html/race_result_on_time/2023/202309030811.html.gz`<br>**行トレース** (`nk_result_on_time`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_result_on_time.json` |
| レース払戻HTML | 任意のレースの払戻テーブル（速報） | 4, 5 | https://race.netkeiba.com/race/result.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_race_result_on_time` | WARN — payoff_keys=0（速報 HTML では払戻ブロック欠落の可能性。確定値は DB 行参照） | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_on_time_payoff/2023/202309030811.json`（`payoff` フィールドのみ。`race_result_on_time` から抽出）<br>HTML: `chuou/data/raw/html/race_result_on_time/2023/202309030811.html.gz`（レース結果 HTML と共通）<br>**行トレース** (`nk_payoff_html`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_payoff_html.json` |
| レースラップHTML | 任意のレースのレースラップテーブル（速報） | 4, 5 | https://race.netkeiba.com/race/result.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_race_result_on_time` | WARN — lap_times=0（速報 HTML では欠落の可能性。確定値は DB 行参照） | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_on_time_lap/2023/202309030811.json`（`lap_times[]` + `pace`。`race_result_on_time` から抽出）<br>HTML: `chuou/data/raw/html/race_result_on_time/2023/202309030811.html.gz`（レース結果 HTML と共通）<br>**行トレース** (`nk_lap_html`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_lap_html.json` |
| レース通過順位HTML | 任意のレースのコーナー通過順位推移（速報） | 4, 5 | https://race.netkeiba.com/race/result.html?race_id=202309030811 | `src/scraper/run.py` | `ScraperRunner.scrape_race_result_on_time` | WARN — corner_passing=0（速報 HTML では欠落の可能性。確定値は DB 行参照） | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_on_time_corner/2023/202309030811.json`（`corner_passing[]`。`race_result_on_time` から抽出）<br>HTML: `chuou/data/raw/html/race_result_on_time/2023/202309030811.html.gz`（レース結果 HTML と共通）<br>**行トレース** (`nk_corner_html`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_corner_html.json` |
| レース個別ラップHTML | 任意のレースの出走馬個別ラップテーブル | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result_lap` | PASS — entries_lap=17, lap_times=11 | 正本①: `chuou/data/preprocessed/netkeiba/pc/race_result_on_time/2023/202309030811.json`（速報参照 / role=`result_page`）<br>正本②: `chuou/data/preprocessed/netkeiba/pc/race_result_lap/2023/202309030811.json`（`entries_lap[]` 確定値 / role=`per_horse_lap`）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`（db 結果ページ）<br>**行トレース** (`nk_per_horse_lap_html`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_per_horse_lap_html.json` |
| 馬プロフィール | 馬のプロフィールデータ（名前・性齢・毛色・馬主等） | 7（出馬表HTMLのSLA=1におけるデータ取得後、出走馬を対象に実施） | https://db.netkeiba.com/horse/2019105219/ | `src/scraper/run.py` | `ScraperRunner.scrape_horse` | PASS — イクイノックス | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/horse_profile/2019/2019105219.json`（`horse_name`/`sex`/`color`/`owner` 等プロフィール系フィールド。`horse_result` から `race_history` を除いて抽出）<br>HTML: `chuou/data/raw/html/horse_profile/2019/2019105219.html.gz`<br>**行トレース** (`nk_horse_profile`): `chuou/data/others/requirement_row_trace/horse_2019105219_nk_horse_profile.json` |
| 馬過去成績 | 任意の馬の過去全成績 | 7（出馬表HTMLのSLA=1におけるデータ取得後、出走馬を対象に実施） | https://db.netkeiba.com/horse/result/2019105219/ | `src/scraper/run.py` | `ScraperRunner.scrape_horse` | PASS — race_history=10（表示上限による件数） | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/horse_race_history/2019/2019105219.json`（`race_history[]` フィールド。`horse_result` から抽出）<br>HTML: `chuou/data/raw/html/horse_result_html/2019/2019105219.html.gz`（プロフィールとは別 URL の戦績ページ）<br>**行トレース** (`nk_horse_history`): `chuou/data/others/requirement_row_trace/horse_2019105219_nk_horse_history.json` |
| 馬血統データ | 任意の馬の5世代血統表 | 7（出馬表HTMLのSLA=1におけるデータ取得後、出走馬を対象に実施） | https://db.netkeiba.com/horse/ped/2019105219/ | `src/scraper/run.py` | `ScraperRunner.scrape_horse_pedigree_5gen` / `ScraperRunner.scrape_horse` | PASS — ancestors=62 | 正本: `chuou/data/preprocessed/netkeiba/pc/horse_pedigree_5gen/2019/2019105219.json`（`ancestors[]`）<br>HTML: `chuou/data/raw/html/horse_ped/2019/2019105219.html.gz`<br>**行トレース** (`nk_horse_pedigree`): `chuou/data/others/requirement_row_trace/horse_2019105219_nk_horse_pedigree.json` |
| 馬調教 | 任意の馬の追い切り履歴 | 7（出馬表HTMLのSLA=1におけるデータ取得後、出走馬を対象に実施） | https://db.netkeiba.com/horse/training.html?id=2019105219 | `src/scraper/run.py` | `ScraperRunner.scrape_horse_training` | PASS — entries=55（max_pages=2 制限下） | 正本: `chuou/data/preprocessed/netkeiba/pc/horse_training/2019/2019105219.json`（`training[]`）<br>HTML（ページ単位）: `chuou/data/raw/html/horse_training/2019/2019105219_p1.html.gz`（`_p2`…も同シャード）<br>**行トレース** (`nk_horse_training`): `chuou/data/others/requirement_row_trace/horse_2019105219_nk_horse_training.json` |
| レースID一覧 | 任意の日付に開催するレースのrace_id一覧 | 7（毎日17:00に、向こう14日分の日付を対象に実施） | https://race.netkeiba.com/top/race_list.html?kaisai_date=20230625 | `src/scraper/run.py` | `ScraperRunner.scrape_race_list` | PASS — n_races=36 | 正本: `data/page_reference/race_lists/20230625.json`（**GCS 非対象・local_only**）<br>HTML（断片）: `chuou/data/raw/html/race_lists/2023/20230625.html.gz`<br>**行トレース** (`nk_race_list`): `chuou/data/others/requirement_row_trace/date_20230625_nk_race_list.json` |
| レース発走時間 | 任意の日付に開催するレースの発走時間 | 7（毎日17:00に、向こう14日分の日付を対象に実施） | https://race.netkeiba.com/top/race_list.html?kaisai_date=20230625 | `src/scraper/auto_scrape.py` / `src/scraper/race_day_schedule.py` | `_fetch_race_schedule` / `synthesize_race_day_schedule_payload` | PASS — n_slots=36 | 正本: `data/page_reference/race_day_schedule/20230625.json`（**local_only**・`race_lists`＋`race_shutuba` から合成スナップショット）。スナップショット不在時は `_fetch_race_schedule_storage` が実行時合成にフォールバック<br>**行トレース** (`nk_race_day_schedule`): `chuou/data/others/requirement_row_trace/date_20230625_nk_race_day_schedule.json` |
| レース結果DB | 任意のレースの結果テーブル（確定） | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result` | PASS — entries=17 | 正本: `chuou/data/preprocessed/netkeiba/pc/race_result/2023/202309030811.json`（`entries[]` / role=`entries_table`）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`<br>**行トレース** (`nk_db_race_result`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_db_race_result.json` |
| レース情報DB | 任意のレースのレース名やグレード、コース情報、馬場等の情報（確定） | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result` | PASS | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_meta/2023/202309030811.json`（`race_name`/`surface`/`distance` 等メタ。`race_result` から `entries`/`payoff`/`lap_times`/`corner_passing` を除いて抽出）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`（レース結果 DB と共通）<br>**行トレース** (`nk_db_race_info`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_db_race_info.json` |
| レース払戻DB | 任意のレースの払戻テーブル（確定） | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result` | PASS — payoff_keys=8 | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_payoff/2023/202309030811.json`（`payoff` フィールドのみ。`race_result` から抽出）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`（レース結果 DB と共通）<br>**行トレース** (`nk_db_payoff`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_db_payoff.json` |
| レース馬場情報DB | 任意のレースの馬場情報（確定） | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result` | PASS — track_condition=良 | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_track/2023/202309030811.json`（`track_condition`/`weather`/`track_condition_turf`/`track_condition_dirt`。`race_result` から抽出）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`（レース結果 DB と共通）<br>**行トレース** (`nk_db_track`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_db_track.json` |
| レース通過順位DB | 任意のレースのコーナー通過順位推移（確定） | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result` / `ScraperRunner.scrape_race_result_lap` | PASS — corner_passing=5 | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_corner/2023/202309030811.json`（`corner_passing[]`。`race_result_lap` から抽出）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`<br>**行トレース** (`nk_db_corner`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_db_corner.json` |
| レースラップDB | 任意のレースのレースラップテーブル（確定） | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result` / `ScraperRunner.scrape_race_result_lap` | PASS — lap_times=11 | **行固有正本**: `chuou/data/preprocessed/netkeiba/pc/race_result_lap_times/2023/202309030811.json`（`lap_times[]` + `pace`。`race_result_lap` から抽出）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`（レース結果 DB と共通）<br>**行トレース** (`nk_db_lap`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_db_lap.json` |
| レース個別ラップDB | 任意のレースの出走馬個別ラップテーブル（確定） | 6 | https://db.netkeiba.com/race/202309030811/ | `src/scraper/run.py` | `ScraperRunner.scrape_race_result_lap` | PASS — entries_lap=17 | 正本: `chuou/data/preprocessed/netkeiba/pc/race_result_lap/2023/202309030811.json`（`entries_lap[]`）<br>HTML: `chuou/data/raw/html/race_result/2023/202309030811.html.gz`（レース結果 DB と共通・`scrape_race_result_lap` が取得）<br>**行トレース** (`nk_db_per_horse_lap`): `chuou/data/others/requirement_row_trace/race_202309030811_nk_db_per_horse_lap.json` |

### 実装メモ（URL・補助モジュール）

- **レース一覧**: `ScraperRunner.scrape_race_list` は `src/scraper/netkeiba_top_race_list.py` の `fetch_races_for_kaisai_date` 等を内部で利用し得る（ブラウザと同じ XHR 断片）。静的 `race_list.html` の単体取得専用関数はない。
- **発走時刻**: `race_day_schedule/{YYYYMMDD}.json` があれば `_fetch_race_schedule_storage`（`auto_scrape_queue.py`）が**先にそれ**を読む。無い場合は保存済み `race_lists` と `race_shutuba` から組み立てる。一覧保存後に自動生成するには `KEIBA_WRITE_RACE_DAY_SCHEDULE_ON_RACE_LIST`（`run.py`）。
- **ラップ**: `scrape_race_result_lap` は `scrape_race_result` と**同一 URL**（`https://db.netkeiba.com/race/{race_id}/`）を取得する。`_fetch_parse_save` は `use_cache=skip_existing` を渡すため、`scrape_race_result` が先に `use_cache=True` で実行済みであればディスクキャッシュヒットで実通信は発生しない。ただし `skip_existing=False`（週次更新等）では同一 URL に二重リクエストが起きうる。将来的には `scrape_race_result` の `parsed` から `build_race_result_lap_payload` を同時呼び出しする統合も可能。`src/scraper/parsers.py` の `DbHorseLaptimeTableParser`（各馬細ラップ AJAX）はクラス定義があるが `ScraperRunner` からは未接続。
- **GCS パス規則**: `HybridStorage._gcs_blob_path`（レース key は `/{年4桁}/{race_id}.json`、馬 key は `/{馬ID先頭4桁}/{horse_id}.json`）、`HtmlArchive._gcs_blob_path`（レース・馬とも key 先頭4桁シャードの `/{shard}/{key}.html.gz`）。`race_lists` の HTML は `race` 扱いでシャードが日付先頭4桁。

### 発走時刻表の設計メモ（開催日ランナー・T-15 等との関係）

**現状の設計（実装済み）**: `race_day_schedule/{YYYYMMDD}.json` を `page_reference` に保存し（local_only）、`_fetch_race_schedule_storage` がスナップショット存在時はそれを優先して読む。行トレース key: `date_{YYYYMMDD}_nk_race_day_schedule`。

**スナップショットが無い場合のフォールバック合成ロジック**は `_fetch_race_schedule`（`auto_scrape.py`）および `_fetch_race_schedule_storage`（`auto_scrape_queue.py`）に集約。

1. `race_lists/{開催日YYYYMMDD}` の `races[]` を走査する。
2. 各 `race_id` について `race_shutuba` を読み、`start_time`（例: `15:40`）を当日の `datetime`（`post_time`）に変換する。
3. `start_time` が空のときは **R 番のみから機械的に推定**するフォールバックがある（実発走と一致しない可能性がある）。

**時刻精度の注意点**:

- **開催日ランナー**（`task_raceday_runner`）や速報結果ランナーは、`post_time` に対して「発走 15 分前」「発走 15 分後」へスリープする。`post_time` が推定値だと T-15 バンドル（`raceday_pre_race_pipeline`）の前提とズレる。
- 当日朝までに全 `race_shutuba` が揃っていない場合は合成表が不安定になる。`KEIBA_WRITE_RACE_DAY_SCHEDULE_ON_RACE_LIST=1` で一覧取得直後に自動保存するか、`materialize_requirement_row_traces --schedule-date` で手動生成すること。

## JRA公式

| タイトル | 説明 | SLA(データ取得タイミング) | 該当リンク | ソースコードパス | 関数名 | テスト結果 (2026-06-12) | GCS / ローカル保存先 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 馬場情報 | クッション値当の当日馬場発表情報 | 2 | https://www.jra.go.jp/keiba/baba/ | `src/scraper/jra_baba_live.py` | `JRABabaLiveScraper.scrape` | PASS — records=18（クッション値マージ保存）。含水率は Playwright の Chromium 未インストールのためログに ERROR、静的 HTML 系は成功） | **ライブ取得の主保存はローカル**: `data/page_reference/cushion/cushion_values.json` ・ `data/page_reference/cushion/cushion_{YYYY}.json`。年別で GCS に載せる場合の例: `chuou/data/others/jra_cushion/{YYYY}.json`（`HybridStorage` カテゴリ `jra_cushion`、`src/scraper/jra_cushion_storage.py`） |


## smartrc

IPブロックのため中止（運用方針としてデータ取得を行わない場合のラベル例）

| タイトル | 説明 | SLA(データ取得タイミング) | 該当リンク | ソースコードパス | 関数名 | テスト結果 (2026-06-12) | GCS 保存先（サンプル race_id 適用時） |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| （運用中止） | 本要件書では取得方針を中止と記載。コードには参照実装が残る場合あり | - | - | NaN | NaN | NA — 要件上テスト対象外（`ScraperRunner.scrape_smartrc` は実行していない） | 有効化時の例: `chuou/data/preprocessed/netkeiba/pc/smartrc_race/2023/202309030811.json` |

補足: 参照用に `src/scraper/run.py` の `ScraperRunner.scrape_smartrc` および `src/scraper/smartrc_client.py` の `SmartRCClient` が存在するが、上表は「現在の要件上の取得対象」に対し中止ラベルを付与している。

## 保存 JSON のサンプル（折りたたみ）

本節は、各行の保存 JSON の**置き場の対応**を示します。**実データのサンプルは公開リポジトリに置きません**（netkeiba 利用規約 第14条）。手元で確認したいときは `requirements_sample_scrape_test.py --export-samples` で各自のローカルに書き出せます（`scrape_process_samples/` は git 管理外）。この節に自動で埋め込むスクリプト（`gen_scrape_process_samples`）は、公開するコミットには使わないでください。

**各 `<details>` ブロックは上表の 1 行（`row_id`）に対応**しており、その行の行固有 GCS パスに格納されるデータを示します。派生カテゴリ（`race_shutuba_meta` 等）はソース JSON から必要フィールドを抽出したものです。

ローカルの書き出し先は `docs/requirements/data/scrape_process_samples/{row_id}.json`（git 管理外）です。

- 既存 L2 のみ取り込み: `python3 -m src.scripts.docs.gen_scrape_process_samples --from-cache`
- フル取得して書き出し: `python3 tests/scraper/manual/requirements_sample_scrape_test.py --export-samples`（`--quick` でキャッシュ優先可）
- JSON のみ手直し後に MD 再整形: `python3 -m src.scripts.docs.gen_scrape_process_samples`

<!-- SCRAPE_PROCESS_SAMPLES_AUTO_BEGIN -->

以下は **実際のスクレイプ保存 JSON**（ローカル L2 `data/cache/` または `requirements_sample_scrape_test.py --export-samples` の戻り値を `scrape_process_samples/` に書き出したもの）を整形したものです。
- レース ID: `202309030811`
- 一覧日付: `20230625`
- 馬 ID: `2019105219`

<details>
<summary><code>nk_shutuba_entries</code> 出馬表HTML（`race_shutuba`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_shutuba_race_meta</code> レース情報HTML（`race_shutuba_meta`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_speed_index</code> タイム指数HTML（`race_index`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_barometer</code> 調子偏差値HTML（`race_barometer`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_paddock</code> パドックHTML（`race_paddock`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_odds</code> オッズHTML（`race_odds`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_result_on_time</code> 結果HTML（`race_result_on_time`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_payoff_html</code> 払戻HTML 速報（`race_result_on_time_payoff`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_lap_html</code> ラップHTML 速報（`race_result_on_time_lap`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_corner_html</code> 通過順位HTML 速報（`race_result_on_time_corner`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_per_horse_lap_html</code> 個別ラップHTML（`race_result_lap`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_horse_profile</code> 馬プロフィール（`horse_profile`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_horse_history</code> 馬過去成績（`horse_race_history`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_horse_pedigree</code> 馬血統データ（`horse_pedigree_5gen`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_horse_training</code> 馬調教（`horse_training`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_race_list</code> レースID一覧（`race_lists`・`20230625`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_race_day_schedule</code> 発走時刻表（`race_day_schedule`・`20230625`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_db_race_result</code> 結果DB（`race_result`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_db_race_info</code> レース情報DB（`race_result_meta`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_db_payoff</code> 払戻DB（`race_result_payoff`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_db_track</code> 馬場情報DB（`race_result_track`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_db_corner</code> 通過順位DB（`race_result_corner`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_db_lap</code> ラップDB（`race_result_lap_times`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<details>
<summary><code>nk_db_per_horse_lap</code> 個別ラップDB（`race_result_lap`）</summary>

```json
{"_note": "実データのサンプルは公開リポジトリに置かない（netkeiba 利用規約 第14条）。保存 JSON の構造は src/scraper/schema_defs.json、観測した型・件数は docs/requirements/data/schemas/observed/ を参照"}
```

</details>


<!-- SCRAPE_PROCESS_SAMPLES_AUTO_END -->
