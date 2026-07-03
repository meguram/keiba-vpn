# データをどこから取得するか

## config

- 日付サンプル: race_date = `20230625`
- レースサンプル: race_id = `202309030811`
- 馬サンプル: horse_id = `2019105219`
- **サンプル取得テスト**: `python3 tests/scraper/manual/requirements_sample_scrape_test.py`（`--quick` 省略時は `skip_existing=False` で再取得。馬調教のみ `max_pages=2` に上限）。**`--export-samples`** で取得戻り値を `docs/requirements/data/scrape_process_samples/*.json` に保存し、本書「保存 JSON のサンプル」折りたたみを実データで更新する。下表「テスト結果」列は **2026-06-12** 実施分（netkeiba ログイン成功後に全項目を実行）。**各項目の `detail` には `schemas.validate` の結果（`schema=PASS` / `schema=FAIL …`）が付与される**（スクリプト改修後の再実行時）。
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

本節は **config 節のサンプル**（`race_date=20230625` / `race_id=202309030811` / `horse_id=2019105219`）で実際に取得した保存 JSON です（ダミー値ではありません）。

**各 `<details>` ブロックは上表の 1 行（`row_id`）に対応**しており、その行の行固有 GCS パスに格納されるデータを示します。派生カテゴリ（`race_shutuba_meta` 等）はソース JSON から必要フィールドを抽出したものです。

正本は `docs/requirements/data/scrape_process_samples/{row_id}.json` と同一です。**マーカー内は自動生成**のため手編集しないでください。

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
{
  "race_id": "202309030811",
  "race_name": "宝塚記念",
  "surface": "芝",
  "distance": 2200,
  "direction": "右",
  "weather": "曇",
  "track_condition": "良",
  "start_time": "15:40",
  "venue": "阪神",
  "field_size": 17,
  "date": "2023-06-25",
  "round": 11,
  "grade": "OP",
  "race_class": "サラ系３歳以上 オープン",
  "weight_rule": "定量",
  "course_type": "",
  "entries": [
    {
      "horse_number": 1,
      "bracket_number": 1,
      "horse_name": "ライラック",
      "horse_id": "2019103588",
      "sex_age": "牝4",
      "jockey_weight": 56.0,
      "jockey_name": "Ｍデムーロ",
      "jockey_id": "05212",
      "trainer_name": "相沢",
      "trainer_id": "01020",
      "weight": 430,
      "weight_change": -8,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 2,
      "bracket_number": 1,
      "horse_name": "カラテ",
      "horse_id": "2016106606",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "菅原明",
      "jockey_id": "01179",
      "trainer_name": "辻野",
      "trainer_id": "01183",
      "weight": 538,
      "weight_change": 0,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 3,
      "bracket_number": 2,
      "horse_name": "ダノンザキッド",
      "horse_id": "2018104963",
      "sex_age": "牡5",
      "jockey_weight": 58.0,
      "jockey_name": "北村友",
      "jockey_id": "01102",
      "trainer_name": "安田隆",
      "trainer_id": "00438",
      "weight": 532,
      "weight_change": -2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 4,
      "bracket_number": 2,
      "horse_name": "ボッケリーニ",
      "horse_id": "2016104618",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "浜中",
      "jockey_id": "01115",
      "trainer_name": "池江",
      "trainer_id": "01071",
      "weight": 466,
      "weight_change": -2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 5,
      "bracket_number": 3,
      "horse_name": "イクイノックス",
      "horse_id": "2019105219",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "ルメール",
      "jockey_id": "05339",
      "trainer_name": "木村",
      "trainer_id": "01126",
      "weight": 492,
      "weight_change": 0,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 6,
      "bracket_number": 3,
      "horse_name": "スルーセブンシーズ",
      "horse_id": "2018105269",
      "sex_age": "牝5",
      "jockey_weight": 56.0,
      "jockey_name": "池添",
      "jockey_id": "01032",
      "trainer_name": "尾関",
      "trainer_id": "01103",
      "weight": 446,
      "weight_change": -2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 7,
      "bracket_number": 4,
      "horse_name": "プラダリア",
      "horse_id": "2019100109",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "菱田",
      "jockey_id": "01144",
      "trainer_name": "池添",
      "trainer_id": "01144",
      "weight": 466,
      "weight_change": 2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 8,
      "bracket_number": 4,
      "horse_name": "ヴェラアズール",
      "horse_id": "2017105082",
      "sex_age": "牡6",
      "jockey_weight": 58.0,
      "jockey_name": "松山",
      "jockey_id": "01126",
      "trainer_name": "渡辺",
      "trainer_id": "01155",
      "weight": 520,
      "weight_change": 0,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 9,
      "bracket_number": 5,
      "horse_name": "ジャスティンパレス",
      "horse_id": "2019105346",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "鮫島駿",
      "jockey_id": "01157",
      "trainer_name": "杉山晴",
      "trainer_id": "01157",
      "weight": 470,
      "weight_change": -2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 10,
      "bracket_number": 5,
      "horse_name": "ディープボンド",
      "horse_id": "2017102170",
      "sex_age": "牡6",
      "jockey_weight": 58.0,
      "jockey_name": "和田竜",
      "jockey_id": "01018",
      "trainer_name": "大久保",
      "trainer_id": "01058",
      "weight": 502,
      "weight_change": -2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 11,
      "bracket_number": 6,
      "horse_name": "ジェラルディーナ",
      "horse_id": "2018105081",
      "sex_age": "牝5",
      "jockey_weight": 56.0,
      "jockey_name": "武豊",
      "jockey_id": "00666",
      "trainer_name": "斉藤崇",
      "trainer_id": "01151",
      "weight": 466,
      "weight_change": 3,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 12,
      "bracket_number": 6,
      "horse_name": "アスクビクターモア",
      "horse_id": "2019104706",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "横山武",
      "jockey_id": "01170",
      "trainer_name": "田村",
      "trainer_id": "01027",
      "weight": 480,
      "weight_change": -2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 13,
      "bracket_number": 7,
      "horse_name": "ジオグリフ",
      "horse_id": "2019105056",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "岩田望",
      "jockey_id": "01174",
      "trainer_name": "木村",
      "trainer_id": "01126",
      "weight": 510,
      "weight_change": 0,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 14,
      "bracket_number": 7,
      "horse_name": "ブレークアップ",
      "horse_id": "2018106273",
      "sex_age": "牡5",
      "jockey_weight": 58.0,
      "jockey_name": "川田",
      "jockey_id": "01088",
      "trainer_name": "吉岡",
      "trainer_id": "01176",
      "weight": 494,
      "weight_change": -2,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 15,
      "bracket_number": 8,
      "horse_name": "ユニコーンライオン",
      "horse_id": "2016110103",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "坂井",
      "jockey_id": "01163",
      "trainer_name": "矢作",
      "trainer_id": "01075",
      "weight": 520,
      "weight_change": 0,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 16,
      "bracket_number": 8,
      "horse_name": "モズベッロ",
      "horse_id": "2016100915",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "角田河",
      "jockey_id": "01199",
      "trainer_name": "森田",
      "trainer_id": "01142",
      "weight": 500,
      "weight_change": 4,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    },
    {
      "horse_number": 17,
      "bracket_number": 8,
      "horse_name": "ドゥラエレーデ",
      "horse_id": "2020103626",
      "sex_age": "牡3",
      "jockey_weight": 53.0,
      "jockey_name": "幸",
      "jockey_id": "00732",
      "trainer_name": "池添",
      "trainer_id": "01144",
      "weight": 506,
      "weight_change": -6,
      "odds": 0.0,
      "popularity": 0,
      "sire": "",
      "dam_sire": ""
    }
  ],
  "_meta": {
    "scraped_at": 1781190679.1546166,
    "scraped_at_jst": "2026-06-12 00:11:19"
  }
}
```

</details>


<details>
<summary><code>nk_shutuba_race_meta</code> レース情報HTML（`race_shutuba_meta`）</summary>

```json
{
  "race_id": "202309030811",
  "race_name": "宝塚記念",
  "surface": "芝",
  "distance": 2200,
  "direction": "右",
  "weather": "曇",
  "track_condition": "良",
  "start_time": "15:40",
  "venue": "阪神",
  "field_size": 17,
  "date": "2023-06-25",
  "round": 11,
  "grade": "OP",
  "race_class": "サラ系３歳以上 オープン",
  "weight_rule": "定量",
  "course_type": ""
}
```

</details>


<details>
<summary><code>nk_speed_index</code> タイム指数HTML（`race_index`）</summary>

```json
{
  "race_id": "202309030811",
  "entries": [
    {
      "horse_number": 1,
      "horse_name": "ライラック",
      "horse_id": "2019103588",
      "time_index_m": 109,
      "speed_max": 110,
      "speed_avg": 102,
      "speed_distance": 110,
      "speed_course": 110,
      "speed_recent": [
        94,
        109,
        110
      ],
      "all_txt_c": [
        4
      ],
      "odds": 158.9,
      "popularity": 13
    },
    {
      "horse_number": 2,
      "horse_name": "カラテ",
      "horse_id": "2016106606",
      "time_index_m": 110,
      "speed_max": 110,
      "speed_avg": 103,
      "speed_distance": 85,
      "speed_course": 97,
      "speed_recent": [
        87,
        110,
        105
      ],
      "all_txt_c": [
        7
      ],
      "odds": 180.2,
      "popularity": 15
    },
    {
      "horse_number": 3,
      "horse_name": "ダノンザキッド",
      "horse_id": "2018104963",
      "time_index_m": 110,
      "speed_max": 112,
      "speed_avg": 104,
      "speed_distance": 0,
      "speed_course": 112,
      "speed_recent": [
        0,
        110,
        90
      ],
      "all_txt_c": [
        5
      ],
      "odds": 35.1,
      "popularity": 8
    },
    {
      "horse_number": 4,
      "horse_name": "ボッケリーニ",
      "horse_id": "2016104618",
      "time_index_m": 109,
      "speed_max": 109,
      "speed_avg": 100,
      "speed_distance": 107,
      "speed_course": 109,
      "speed_recent": [
        91,
        109,
        102
      ],
      "all_txt_c": [
        7
      ],
      "odds": 28.3,
      "popularity": 6
    },
    {
      "horse_number": 5,
      "horse_name": "イクイノックス",
      "horse_id": "2019105219",
      "time_index_m": 116,
      "speed_max": 116,
      "speed_avg": 111,
      "speed_distance": 0,
      "speed_course": 0,
      "speed_recent": [
        0,
        116,
        111
      ],
      "all_txt_c": [
        4
      ],
      "odds": 1.3,
      "popularity": 1
    },
    {
      "horse_number": 6,
      "horse_name": "スルーセブンシーズ",
      "horse_id": "2018105269",
      "time_index_m": 103,
      "speed_max": 106,
      "speed_avg": 100,
      "speed_distance": 105,
      "speed_course": 98,
      "speed_recent": [
        106,
        103,
        96
      ],
      "all_txt_c": [
        5
      ],
      "odds": 55.7,
      "popularity": 10
    },
    {
      "horse_number": 7,
      "horse_name": "プラダリア",
      "horse_id": "2019100109",
      "time_index_m": 108,
      "speed_max": 108,
      "speed_avg": 101,
      "speed_distance": 108,
      "speed_course": 108,
      "speed_recent": [
        96,
        108,
        104
      ],
      "all_txt_c": [
        4
      ],
      "odds": 262.5,
      "popularity": 16
    },
    {
      "horse_number": 8,
      "horse_name": "ヴェラアズール",
      "horse_id": "2017105082",
      "time_index_m": 105,
      "speed_max": 110,
      "speed_avg": 104,
      "speed_distance": 0,
      "speed_course": 105,
      "speed_recent": [
        0,
        105,
        110
      ],
      "all_txt_c": [
        6
      ],
      "odds": 42.1,
      "popularity": 9
    },
    {
      "horse_number": 9,
      "horse_name": "ジャスティンパレス",
      "horse_id": "2019105346",
      "time_index_m": 102,
      "speed_max": 109,
      "speed_avg": 106,
      "speed_distance": 102,
      "speed_course": 108,
      "speed_recent": [
        108,
        102,
        109
      ],
      "all_txt_c": [
        4
      ],
      "odds": 8.5,
      "popularity": 2
    },
    {
      "horse_number": 10,
      "horse_name": "ディープボンド",
      "horse_id": "2017102170",
      "time_index_m": 99,
      "speed_max": 114,
      "speed_avg": 107,
      "speed_distance": 114,
      "speed_course": 115,
      "speed_recent": [
        106,
        99,
        108
      ],
      "all_txt_c": [
        6
      ],
      "odds": 16.6,
      "popularity": 5
    },
    {
      "horse_number": 11,
      "horse_name": "ジェラルディーナ",
      "horse_id": "2018105081",
      "time_index_m": 105,
      "speed_max": 113,
      "speed_avg": 108,
      "speed_distance": 113,
      "speed_course": 113,
      "speed_recent": [
        0,
        105,
        112
      ],
      "all_txt_c": [
        5
      ],
      "odds": 13.8,
      "popularity": 3
    },
    {
      "horse_number": 12,
      "horse_name": "アスクビクターモア",
      "horse_id": "2019104706",
      "time_index_m": 101,
      "speed_max": 109,
      "speed_avg": 105,
      "speed_distance": 103,
      "speed_course": 109,
      "speed_recent": [
        99,
        101,
        109
      ],
      "all_txt_c": [
        4
      ],
      "odds": 14.3,
      "popularity": 4
    },
    {
      "horse_number": 13,
      "horse_name": "ジオグリフ",
      "horse_id": "2019105056",
      "time_index_m": 0,
      "speed_max": 104,
      "speed_avg": 105,
      "speed_distance": 0,
      "speed_course": 91,
      "speed_recent": [
        0,
        0,
        0
      ],
      "all_txt_c": [
        4
      ],
      "odds": 83.1,
      "popularity": 11
    },
    {
      "horse_number": 14,
      "horse_name": "ブレークアップ",
      "horse_id": "2018106273",
      "time_index_m": 100,
      "speed_max": 104,
      "speed_avg": 97,
      "speed_distance": 103,
      "speed_course": 100,
      "speed_recent": [
        104,
        100,
        91
      ],
      "all_txt_c": [
        5
      ],
      "odds": 113.4,
      "popularity": 12
    },
    {
      "horse_number": 15,
      "horse_name": "ユニコーンライオン",
      "horse_id": "2016110103",
      "time_index_m": 103,
      "speed_max": 107,
      "speed_avg": 100,
      "speed_distance": 107,
      "speed_course": 107,
      "speed_recent": [
        0,
        103,
        99
      ],
      "all_txt_c": [
        7
      ],
      "odds": 176.0,
      "popularity": 14
    },
    {
      "horse_number": 16,
      "horse_name": "モズベッロ",
      "horse_id": "2016100915",
      "time_index_m": 99,
      "speed_max": 103,
      "speed_avg": 98,
      "speed_distance": 109,
      "speed_course": 109,
      "speed_recent": [
        89,
        99,
        103
      ],
      "all_txt_c": [
        7
      ],
      "odds": 480.4,
      "popularity": 17
    },
    {
      "horse_number": 17,
      "horse_name": "ドゥラエレーデ",
      "horse_id": "2020103626",
      "time_index_m": 0,
      "speed_max": 95,
      "speed_avg": 94,
      "speed_distance": 0,
      "speed_course": 56,
      "speed_recent": [
        0,
        0,
        95
      ],
      "all_txt_c": [
        3
      ],
      "odds": 30.2,
      "popularity": 7
    }
  ],
  "_meta": {
    "scraped_at": 1781190682.851525,
    "scraped_at_jst": "2026-06-12 00:11:22"
  }
}
```

</details>


<details>
<summary><code>nk_barometer</code> 調子偏差値HTML（`race_barometer`）</summary>

```json
{
  "race_id": "202309030811",
  "entries": [
    {
      "horse_number": 5,
      "horse_name": "イクイノックス",
      "horse_id": "2019105219",
      "finish_order": 1,
      "index_total": 113,
      "index_start": 102,
      "index_chase": 104,
      "index_closing": 106
    },
    {
      "horse_number": 6,
      "horse_name": "スルーセブンシーズ",
      "horse_id": "2018105269",
      "finish_order": 2,
      "index_total": 114,
      "index_start": 100,
      "index_chase": 104,
      "index_closing": 108
    },
    {
      "horse_number": 9,
      "horse_name": "ジャスティンパレス",
      "horse_id": "2019105346",
      "finish_order": 3,
      "index_total": 112,
      "index_start": 102,
      "index_chase": 106,
      "index_closing": 103
    },
    {
      "horse_number": 11,
      "horse_name": "ジェラルディーナ",
      "horse_id": "2018105081",
      "finish_order": 4,
      "index_total": 113,
      "index_start": 100,
      "index_chase": 105,
      "index_closing": 101
    },
    {
      "horse_number": 10,
      "horse_name": "ディープボンド",
      "horse_id": "2017102170",
      "finish_order": 5,
      "index_total": 110,
      "index_start": 105,
      "index_chase": 109,
      "index_closing": 101
    },
    {
      "horse_number": 7,
      "horse_name": "プラダリア",
      "horse_id": "2019100109",
      "finish_order": 6,
      "index_total": 110,
      "index_start": 105,
      "index_chase": 107,
      "index_closing": 103
    },
    {
      "horse_number": 4,
      "horse_name": "ボッケリーニ",
      "horse_id": "2016104618",
      "finish_order": 7,
      "index_total": 109,
      "index_start": 101,
      "index_chase": 107,
      "index_closing": 104
    },
    {
      "horse_number": 8,
      "horse_name": "ヴェラアズール",
      "horse_id": "2017105082",
      "finish_order": 8,
      "index_total": 108,
      "index_start": 102,
      "index_chase": 105,
      "index_closing": 103
    },
    {
      "horse_number": 13,
      "horse_name": "ジオグリフ",
      "horse_id": "2019105056",
      "finish_order": 9,
      "index_total": 109,
      "index_start": 105,
      "index_chase": 109,
      "index_closing": 99
    },
    {
      "horse_number": 17,
      "horse_name": "ドゥラエレーデ",
      "horse_id": "2020103626",
      "finish_order": 10,
      "index_total": 103,
      "index_start": 108,
      "index_chase": 111,
      "index_closing": 93
    },
    {
      "horse_number": 12,
      "horse_name": "アスクビクターモア",
      "horse_id": "2019104706",
      "finish_order": 11,
      "index_total": 105,
      "index_start": 107,
      "index_chase": 110,
      "index_closing": 94
    },
    {
      "horse_number": 14,
      "horse_name": "ブレークアップ",
      "horse_id": "2018106273",
      "finish_order": 12,
      "index_total": 105,
      "index_start": 107,
      "index_chase": 111,
      "index_closing": 93
    },
    {
      "horse_number": 3,
      "horse_name": "ダノンザキッド",
      "horse_id": "2018104963",
      "finish_order": 13,
      "index_total": 103,
      "index_start": 107,
      "index_chase": 110,
      "index_closing": 93
    },
    {
      "horse_number": 16,
      "horse_name": "モズベッロ",
      "horse_id": "2016100915",
      "finish_order": 14,
      "index_total": 104,
      "index_start": 104,
      "index_chase": 107,
      "index_closing": 96
    },
    {
      "horse_number": 15,
      "horse_name": "ユニコーンライオン",
      "horse_id": "2016110103",
      "finish_order": 15,
      "index_total": 103,
      "index_start": 109,
      "index_chase": 112,
      "index_closing": 91
    },
    {
      "horse_number": 2,
      "horse_name": "カラテ",
      "horse_id": "2016106606",
      "finish_order": 16,
      "index_total": 103,
      "index_start": 107,
      "index_chase": 111,
      "index_closing": 93
    },
    {
      "horse_number": 1,
      "horse_name": "ライラック",
      "horse_id": "2019103588",
      "finish_order": 17,
      "index_total": 102,
      "index_start": 105,
      "index_chase": 109,
      "index_closing": 94
    }
  ],
  "_meta": {
    "scraped_at": 1781190683.5042465,
    "scraped_at_jst": "2026-06-12 00:11:23"
  }
}
```

</details>


<details>
<summary><code>nk_paddock</code> パドックHTML（`race_paddock`）</summary>

```json
{
  "race_id": "202309030811",
  "entries": [
    {
      "horse_number": 3,
      "horse_name": "イクイノックス",
      "horse_id": "2019105219",
      "paddock_rank": "B",
      "paddock_comment": "有馬記念時と比べると見劣るが、体は仕上がっている"
    },
    {
      "horse_number": 5,
      "horse_name": "ジャスティンパレス",
      "horse_id": "2019105346",
      "paddock_rank": "A",
      "paddock_comment": "キョロキョロして周回。踏み込み上々で好状態"
    },
    {
      "horse_number": 5,
      "horse_name": "ディープボンド",
      "horse_id": "2017102170",
      "paddock_rank": "A",
      "paddock_comment": "張りがあって迫力十分。脚どりも力強く、状態いい"
    },
    {
      "horse_number": 6,
      "horse_name": "ジェラルディーナ",
      "horse_id": "2018105081",
      "paddock_rank": "B",
      "paddock_comment": "テンションは高くても走れるタイプ。仕上がる"
    },
    {
      "horse_number": 6,
      "horse_name": "アスクビクターモア",
      "horse_id": "2019104706",
      "paddock_rank": "B",
      "paddock_comment": "馬体が引き締まっている。春２戦以上のデキだろう"
    }
  ],
  "_meta": {
    "scraped_at": 1781190687.483203,
    "scraped_at_jst": "2026-06-12 00:11:27"
  }
}
```

</details>


<details>
<summary><code>nk_odds</code> オッズHTML（`race_odds`）</summary>

```json
{
  "race_id": "202309030811",
  "entries": [
    {
      "horse_number": 1,
      "win_odds": 158.9,
      "place_odds_min": 12.3,
      "place_odds_max": 30.5,
      "popularity": 13
    },
    {
      "horse_number": 2,
      "win_odds": 180.2,
      "place_odds_min": 14.5,
      "place_odds_max": 36.1,
      "popularity": 15
    },
    {
      "horse_number": 3,
      "win_odds": 35.1,
      "place_odds_min": 3.1,
      "place_odds_max": 7.2,
      "popularity": 8
    },
    {
      "horse_number": 4,
      "win_odds": 28.3,
      "place_odds_min": 2.7,
      "place_odds_max": 6.2,
      "popularity": 6
    },
    {
      "horse_number": 5,
      "win_odds": 1.3,
      "place_odds_min": 1.1,
      "place_odds_max": 1.1,
      "popularity": 1
    },
    {
      "horse_number": 6,
      "win_odds": 55.7,
      "place_odds_min": 5.6,
      "place_odds_max": 13.5,
      "popularity": 10
    },
    {
      "horse_number": 7,
      "win_odds": 262.5,
      "place_odds_min": 17.5,
      "place_odds_max": 43.5,
      "popularity": 16
    },
    {
      "horse_number": 8,
      "win_odds": 42.1,
      "place_odds_min": 4.7,
      "place_odds_max": 11.1,
      "popularity": 9
    },
    {
      "horse_number": 9,
      "win_odds": 8.5,
      "place_odds_min": 1.6,
      "place_odds_max": 3.0,
      "popularity": 2
    },
    {
      "horse_number": 10,
      "win_odds": 16.6,
      "place_odds_min": 2.2,
      "place_odds_max": 4.7,
      "popularity": 5
    },
    {
      "horse_number": 11,
      "win_odds": 13.8,
      "place_odds_min": 2.4,
      "place_odds_max": 5.3,
      "popularity": 3
    },
    {
      "horse_number": 12,
      "win_odds": 14.3,
      "place_odds_min": 2.3,
      "place_odds_max": 5.0,
      "popularity": 4
    },
    {
      "horse_number": 13,
      "win_odds": 83.1,
      "place_odds_min": 8.8,
      "place_odds_max": 21.6,
      "popularity": 11
    },
    {
      "horse_number": 14,
      "win_odds": 113.4,
      "place_odds_min": 7.4,
      "place_odds_max": 18.1,
      "popularity": 12
    },
    {
      "horse_number": 15,
      "win_odds": 176.0,
      "place_odds_min": 16.2,
      "place_odds_max": 40.3,
      "popularity": 14
    },
    {
      "horse_number": 16,
      "win_odds": 480.4,
      "place_odds_min": 53.0,
      "place_odds_max": 132.9,
      "popularity": 17
    },
    {
      "horse_number": 17,
      "win_odds": 30.2,
      "place_odds_min": 4.5,
      "place_odds_max": 10.6,
      "popularity": 7
    }
  ],
  "_meta": {
    "scraped_at": 1781190687.6626334,
    "scraped_at_jst": "2026-06-12 00:11:27"
  }
}
```

</details>


<details>
<summary><code>nk_result_on_time</code> 結果HTML（`race_result_on_time`）</summary>

```json
{
  "race_id": "202309030811",
  "race_name": "宝塚記念",
  "surface": "芝",
  "distance": 2200,
  "direction": "右",
  "weather": "曇",
  "track_condition": "良",
  "start_time": "15:40",
  "venue": "阪神",
  "date": "2023-06-25",
  "round": 11,
  "grade": "OP",
  "entries": [
    {
      "finish_position": 1,
      "bracket_number": 3,
      "horse_number": 5,
      "horse_name": "イクイノックス",
      "horse_id": "2019105219",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "ルメー",
      "jockey_id": "05339",
      "finish_time": "2:11.2",
      "time_sec": 131.2,
      "margin": "",
      "passing_order": "16-16-13-9",
      "last_3f": 34.8,
      "odds": 1.3,
      "popularity": 1,
      "weight": 492,
      "weight_change": 0,
      "trainer_name": "木村",
      "trainer_id": "01126"
    },
    {
      "finish_position": 2,
      "bracket_number": 3,
      "horse_number": 6,
      "horse_name": "スルーセブンシーズ",
      "horse_id": "2018105269",
      "sex_age": "牝5",
      "jockey_weight": 56.0,
      "jockey_name": "池添",
      "jockey_id": "01032",
      "finish_time": "2:11.2",
      "time_sec": 131.2,
      "margin": "クビ",
      "passing_order": "17-17-16-12",
      "last_3f": 34.6,
      "odds": 55.7,
      "popularity": 10,
      "weight": 446,
      "weight_change": -2,
      "trainer_name": "尾関",
      "trainer_id": "01103"
    },
    {
      "finish_position": 3,
      "bracket_number": 5,
      "horse_number": 9,
      "horse_name": "ジャスティンパレス",
      "horse_id": "2019105346",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "鮫島駿",
      "jockey_id": "01157",
      "finish_time": "2:11.4",
      "time_sec": 131.4,
      "margin": "1",
      "passing_order": "12-13-11-9",
      "last_3f": 35.1,
      "odds": 8.5,
      "popularity": 2,
      "weight": 470,
      "weight_change": -2,
      "trainer_name": "杉山晴",
      "trainer_id": "01157"
    },
    {
      "finish_position": 4,
      "bracket_number": 6,
      "horse_number": 11,
      "horse_name": "ジェラルディーナ",
      "horse_id": "2018105081",
      "sex_age": "牝5",
      "jockey_weight": 56.0,
      "jockey_name": "武豊",
      "jockey_id": "00666",
      "finish_time": "2:11.4",
      "time_sec": 131.4,
      "margin": "アタマ",
      "passing_order": "14-14-6-3",
      "last_3f": 35.5,
      "odds": 13.8,
      "popularity": 3,
      "weight": 466,
      "weight_change": 3,
      "trainer_name": "斉藤崇",
      "trainer_id": "01151"
    },
    {
      "finish_position": 5,
      "bracket_number": 5,
      "horse_number": 10,
      "horse_name": "ディープボンド",
      "horse_id": "2017102170",
      "sex_age": "牡6",
      "jockey_weight": 58.0,
      "jockey_name": "和田竜",
      "jockey_id": "01018",
      "finish_time": "2:11.6",
      "time_sec": 131.6,
      "margin": "1",
      "passing_order": "7-7-8-6",
      "last_3f": 35.5,
      "odds": 16.6,
      "popularity": 5,
      "weight": 502,
      "weight_change": -2,
      "trainer_name": "大久保",
      "trainer_id": "01058"
    },
    {
      "finish_position": 6,
      "bracket_number": 4,
      "horse_number": 7,
      "horse_name": "プラダリア",
      "horse_id": "2019100109",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "菱田",
      "jockey_id": "01144",
      "finish_time": "2:11.6",
      "time_sec": 131.6,
      "margin": "クビ",
      "passing_order": "11-10-11-12",
      "last_3f": 35.3,
      "odds": 262.5,
      "popularity": 16,
      "weight": 466,
      "weight_change": 2,
      "trainer_name": "池添",
      "trainer_id": "01144"
    },
    {
      "finish_position": 7,
      "bracket_number": 2,
      "horse_number": 4,
      "horse_name": "ボッケリーニ",
      "horse_id": "2016104618",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "浜中",
      "jockey_id": "01115",
      "finish_time": "2:11.7",
      "time_sec": 131.7,
      "margin": "クビ",
      "passing_order": "12-10-13-16",
      "last_3f": 35.1,
      "odds": 28.3,
      "popularity": 6,
      "weight": 466,
      "weight_change": -2,
      "trainer_name": "池江",
      "trainer_id": "01071"
    },
    {
      "finish_position": 8,
      "bracket_number": 4,
      "horse_number": 8,
      "horse_name": "ヴェラアズール",
      "horse_id": "2017105082",
      "sex_age": "牡6",
      "jockey_weight": 58.0,
      "jockey_name": "松山",
      "jockey_id": "01126",
      "finish_time": "2:11.9",
      "time_sec": 131.9,
      "margin": "1.1/4",
      "passing_order": "14-14-16-16",
      "last_3f": 35.2,
      "odds": 42.1,
      "popularity": 9,
      "weight": 520,
      "weight_change": 0,
      "trainer_name": "渡辺",
      "trainer_id": "01155"
    },
    {
      "finish_position": 9,
      "bracket_number": 7,
      "horse_number": 13,
      "horse_name": "ジオグリフ",
      "horse_id": "2019105056",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "岩田望",
      "jockey_id": "01174",
      "finish_time": "2:11.9",
      "time_sec": 131.9,
      "margin": "ハナ",
      "passing_order": "7-7-8-6",
      "last_3f": 35.8,
      "odds": 83.1,
      "popularity": 11,
      "weight": 510,
      "weight_change": 0,
      "trainer_name": "木村",
      "trainer_id": "01126"
    },
    {
      "finish_position": 10,
      "bracket_number": 8,
      "horse_number": 17,
      "horse_name": "ドゥラエレーデ",
      "horse_id": "2020103626",
      "sex_age": "牡3",
      "jockey_weight": 53.0,
      "jockey_name": "幸",
      "jockey_id": "00732",
      "finish_time": "2:12.3",
      "time_sec": 132.3,
      "margin": "2",
      "passing_order": "2-2-2-2",
      "last_3f": 36.5,
      "odds": 30.2,
      "popularity": 7,
      "weight": 506,
      "weight_change": -6,
      "trainer_name": "池添",
      "trainer_id": "01144"
    },
    {
      "finish_position": 11,
      "bracket_number": 6,
      "horse_number": 12,
      "horse_name": "アスクビクターモア",
      "horse_id": "2019104706",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "横山武",
      "jockey_id": "01170",
      "finish_time": "2:12.3",
      "time_sec": 132.3,
      "margin": "クビ",
      "passing_order": "4-5-4-3",
      "last_3f": 36.5,
      "odds": 14.3,
      "popularity": 4,
      "weight": 480,
      "weight_change": -2,
      "trainer_name": "田村",
      "trainer_id": "01027"
    },
    {
      "finish_position": 12,
      "bracket_number": 7,
      "horse_number": 14,
      "horse_name": "ブレークアップ",
      "horse_id": "2018106273",
      "sex_age": "牡5",
      "jockey_weight": 58.0,
      "jockey_name": "川田",
      "jockey_id": "01088",
      "finish_time": "2:12.3",
      "time_sec": 132.3,
      "margin": "クビ",
      "passing_order": "3-3-3-3",
      "last_3f": 36.5,
      "odds": 113.4,
      "popularity": 12,
      "weight": 494,
      "weight_change": -2,
      "trainer_name": "吉岡",
      "trainer_id": "01176"
    },
    {
      "finish_position": 13,
      "bracket_number": 2,
      "horse_number": 3,
      "horse_name": "ダノンザキッド",
      "horse_id": "2018104963",
      "sex_age": "牡5",
      "jockey_weight": 58.0,
      "jockey_name": "北村友",
      "jockey_id": "01102",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "1.1/2",
      "passing_order": "4-5-6-9",
      "last_3f": 36.5,
      "odds": 35.1,
      "popularity": 8,
      "weight": 532,
      "weight_change": -2,
      "trainer_name": "安田隆",
      "trainer_id": "00438"
    },
    {
      "finish_position": 14,
      "bracket_number": 8,
      "horse_number": 16,
      "horse_name": "モズベッロ",
      "horse_id": "2016100915",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "角田河",
      "jockey_id": "01199",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "アタマ",
      "passing_order": "10-10-13-12",
      "last_3f": 36.2,
      "odds": 480.4,
      "popularity": 17,
      "weight": 500,
      "weight_change": 4,
      "trainer_name": "森田",
      "trainer_id": "01142"
    },
    {
      "finish_position": 15,
      "bracket_number": 8,
      "horse_number": 15,
      "horse_name": "ユニコーンライオン",
      "horse_id": "2016110103",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "坂井",
      "jockey_id": "01163",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "ハナ",
      "passing_order": "1-1-1-1",
      "last_3f": 36.9,
      "odds": 176.0,
      "popularity": 14,
      "weight": 520,
      "weight_change": 0,
      "trainer_name": "矢作",
      "trainer_id": "01075"
    },
    {
      "finish_position": 16,
      "bracket_number": 1,
      "horse_number": 2,
      "horse_name": "カラテ",
      "horse_id": "2016106606",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "菅原明",
      "jockey_id": "01179",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "ハナ",
      "passing_order": "4-3-4-6",
      "last_3f": 36.5,
      "odds": 180.2,
      "popularity": 15,
      "weight": 538,
      "weight_change": 0,
      "trainer_name": "辻野",
      "trainer_id": "01183"
    },
    {
      "finish_position": 17,
      "bracket_number": 1,
      "horse_number": 1,
      "horse_name": "ライラック",
      "horse_id": "2019103588",
      "sex_age": "牝4",
      "jockey_weight": 56.0,
      "jockey_name": "Ｍデム",
      "jockey_id": "05212",
      "finish_time": "2:12.7",
      "time_sec": 132.7,
      "margin": "1/2",
      "passing_order": "7-7-8-12",
      "last_3f": 36.4,
      "odds": 158.9,
      "popularity": 13,
      "weight": 430,
      "weight_change": -8,
      "trainer_name": "相沢",
      "trainer_id": "01020"
    }
  ],
  "field_size": 17,
  "payoff": {},
  "lap_times": [],
  "pace": {},
  "corner_passing": [],
  "_meta": {
    "result_schema_profile": "race_live_archive",
    "result_schema_kind": "race_live_result",
    "scraped_at": 1781190706.0083513,
    "scraped_at_jst": "2026-06-12 00:11:46"
  }
}
```

</details>


<details>
<summary><code>nk_payoff_html</code> 払戻HTML 速報（`race_result_on_time_payoff`）</summary>

```json
{
  "race_id": "202309030811",
  "payoff": {}
}
```

</details>


<details>
<summary><code>nk_lap_html</code> ラップHTML 速報（`race_result_on_time_lap`）</summary>

```json
{
  "race_id": "202309030811",
  "lap_times": [],
  "pace": {}
}
```

</details>


<details>
<summary><code>nk_corner_html</code> 通過順位HTML 速報（`race_result_on_time_corner`）</summary>

```json
{
  "race_id": "202309030811",
  "corner_passing": []
}
```

</details>


<details>
<summary><code>nk_per_horse_lap_html</code> 個別ラップHTML（`race_result_lap`）</summary>

```json
{
  "race_id": "202309030811",
  "lap_times": [
    12.4,
    10.5,
    11.1,
    12.6,
    12.3,
    12.4,
    12.5,
    11.9,
    11.7,
    12.0,
    11.8
  ],
  "pace": {
    "first_half_3f": 34.0,
    "second_half_3f": 35.5,
    "t1f": 12.4,
    "t3f": 34.0,
    "l1f": 11.8,
    "l3f": 35.5
  },
  "corner_passing": [
    {
      "corner": 0,
      "label": "馬場指数",
      "order_text": "-7 (?)"
    },
    {
      "corner": 1,
      "label": "1コーナー",
      "order_text": "(*15,17)14(2,3,12)(1,10,13)16,7(4,9)(8,11)5,6"
    },
    {
      "corner": 2,
      "label": "2コーナー",
      "order_text": "15,17(2,14)(3,12)(1,10,13)(4,7,16)9(8,11)5,6"
    },
    {
      "corner": 3,
      "label": "3コーナー",
      "order_text": "(*15,17)14(2,12)(3,11)(1,10,13)(7,9)(4,16,5)(8,6)"
    },
    {
      "corner": 4,
      "label": "4コーナー",
      "order_text": "(*15,17)(14,12,11)(2,10,13)(3,9,5)(1,7,16,6)(4,8)"
    }
  ],
  "entries_lap": [
    {
      "horse_number": 5,
      "horse_id": "2019105219",
      "horse_name": "イクイノックス",
      "passing_order": "16-16-13-9",
      "last_3f": 34.8
    },
    {
      "horse_number": 6,
      "horse_id": "2018105269",
      "horse_name": "スルーセブンシーズ",
      "passing_order": "17-17-16-12",
      "last_3f": 34.6
    },
    {
      "horse_number": 9,
      "horse_id": "2019105346",
      "horse_name": "ジャスティンパレス",
      "passing_order": "12-13-11-9",
      "last_3f": 35.1
    },
    {
      "horse_number": 11,
      "horse_id": "2018105081",
      "horse_name": "ジェラルディーナ",
      "passing_order": "14-14-6-3",
      "last_3f": 35.5
    },
    {
      "horse_number": 10,
      "horse_id": "2017102170",
      "horse_name": "ディープボンド",
      "passing_order": "7-7-8-6",
      "last_3f": 35.5
    },
    {
      "horse_number": 7,
      "horse_id": "2019100109",
      "horse_name": "プラダリア",
      "passing_order": "11-10-11-12",
      "last_3f": 35.3
    },
    {
      "horse_number": 4,
      "horse_id": "2016104618",
      "horse_name": "ボッケリーニ",
      "passing_order": "12-10-13-16",
      "last_3f": 35.1
    },
    {
      "horse_number": 8,
      "horse_id": "2017105082",
      "horse_name": "ヴェラアズール",
      "passing_order": "14-14-16-16",
      "last_3f": 35.2
    },
    {
      "horse_number": 13,
      "horse_id": "2019105056",
      "horse_name": "ジオグリフ",
      "passing_order": "7-7-8-6",
      "last_3f": 35.8
    },
    {
      "horse_number": 17,
      "horse_id": "2020103626",
      "horse_name": "ドゥラエレーデ",
      "passing_order": "2-2-2-2",
      "last_3f": 36.5
    },
    {
      "horse_number": 12,
      "horse_id": "2019104706",
      "horse_name": "アスクビクターモア",
      "passing_order": "4-5-4-3",
      "last_3f": 36.5
    },
    {
      "horse_number": 14,
      "horse_id": "2018106273",
      "horse_name": "ブレークアップ",
      "passing_order": "3-3-3-3",
      "last_3f": 36.5
    },
    {
      "horse_number": 3,
      "horse_id": "2018104963",
      "horse_name": "ダノンザキッド",
      "passing_order": "4-5-6-9",
      "last_3f": 36.5
    },
    {
      "horse_number": 16,
      "horse_id": "2016100915",
      "horse_name": "モズベッロ",
      "passing_order": "10-10-13-12",
      "last_3f": 36.2
    },
    {
      "horse_number": 15,
      "horse_id": "2016110103",
      "horse_name": "ユニコーンライオン",
      "passing_order": "1-1-1-1",
      "last_3f": 36.9
    },
    {
      "horse_number": 2,
      "horse_id": "2016106606",
      "horse_name": "カラテ",
      "passing_order": "4-3-4-6",
      "last_3f": 36.5
    },
    {
      "horse_number": 1,
      "horse_id": "2019103588",
      "horse_name": "ライラック",
      "passing_order": "7-7-8-12",
      "last_3f": 36.4
    }
  ],
  "_meta": {
    "scraped_at": 1781190781.7651346,
    "scraped_at_jst": "2026-06-12 00:13:01"
  }
}
```

</details>


<details>
<summary><code>nk_horse_profile</code> 馬プロフィール（`horse_profile`）</summary>

```json
{
  "horse_id": "2019105219",
  "horse_name": "イクイノックス",
  "name_en": "Equinox",
  "birthday": "2019年3月23日",
  "trainer": "木村哲也",
  "owner": "シルクレーシング",
  "breeder": "ノーザンファーム",
  "birthplace": "安平町",
  "total_earnings": 175655,
  "career": "10戦8勝 [8-2-0-0]",
  "career_record": [
    8,
    2,
    0,
    0
  ],
  "major_wins": [
    "23'ジャパンC(G1)"
  ],
  "sire": "",
  "dam": "",
  "dam_sire": ""
}
```

</details>


<details>
<summary><code>nk_horse_history</code> 馬過去成績（`horse_race_history`）</summary>

```json
{
  "horse_id": "2019105219",
  "horse_name": "イクイノックス",
  "race_history": [
    {
      "date": "2023/11/26",
      "venue": "5東京8",
      "weather": "曇",
      "race_round": 12,
      "race_name": "ジャパンC(GI)",
      "race_id": "202305050812",
      "field_size": 18,
      "bracket_number": 1,
      "horse_number": 2,
      "odds": 1.3,
      "popularity": 1,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 58.0,
      "surface": "芝",
      "distance": 2400,
      "time_index": 117,
      "track_condition": "良",
      "finish_time": "2:21.8",
      "time_sec": 141.8,
      "margin": "-0.7",
      "passing_order": "117",
      "last_3f": 103.0,
      "weight": 498,
      "weight_change": 4,
      "winner": "33.5"
    },
    {
      "date": "2023/10/29",
      "venue": "4東京9",
      "weather": "晴",
      "race_round": 11,
      "race_name": "天皇賞(秋)(GI)",
      "race_id": "202305040911",
      "field_size": 11,
      "bracket_number": 6,
      "horse_number": 7,
      "odds": 1.3,
      "popularity": 1,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 58.0,
      "surface": "芝",
      "distance": 2000,
      "time_index": 123,
      "track_condition": "良",
      "finish_time": "1:55.2",
      "time_sec": 115.2,
      "margin": "-0.4",
      "passing_order": "123",
      "last_3f": 111.0,
      "weight": 494,
      "weight_change": 2,
      "winner": "34.2"
    },
    {
      "date": "2023/06/25",
      "venue": "3阪神8",
      "weather": "曇",
      "race_round": 11,
      "race_name": "宝塚記念(GI)",
      "race_id": "202309030811",
      "field_size": 17,
      "bracket_number": 3,
      "horse_number": 5,
      "odds": 1.3,
      "popularity": 1,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 58.0,
      "surface": "芝",
      "distance": 2200,
      "time_index": 113,
      "track_condition": "良",
      "finish_time": "2:11.2",
      "time_sec": 131.2,
      "margin": "0.0",
      "passing_order": "113",
      "last_3f": 104.0,
      "weight": 492,
      "weight_change": 0,
      "winner": "34.8"
    },
    {
      "date": "2023/03/25",
      "venue": "メイダン",
      "weather": "晴",
      "race_round": 0,
      "race_name": "ドバイシーマC(GI)",
      "race_id": "2023J0032508",
      "field_size": 10,
      "bracket_number": 0,
      "horse_number": 7,
      "odds": 1.4,
      "popularity": 1,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 56.5,
      "surface": "芝",
      "distance": 2410,
      "time_index": 0,
      "track_condition": "良",
      "finish_time": "2:25.65",
      "time_sec": 145.65,
      "margin": "",
      "passing_order": "",
      "last_3f": 0.0,
      "weight": 0,
      "weight_change": 0,
      "winner": ""
    },
    {
      "date": "2022/12/25",
      "venue": "5中山8",
      "weather": "晴",
      "race_round": 11,
      "race_name": "有馬記念(GI)",
      "race_id": "202206050811",
      "field_size": 16,
      "bracket_number": 5,
      "horse_number": 9,
      "odds": 2.3,
      "popularity": 1,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 55.0,
      "surface": "芝",
      "distance": 2500,
      "time_index": 0,
      "track_condition": "良",
      "finish_time": "2:32.4",
      "time_sec": 152.4,
      "margin": "-0.4",
      "passing_order": "",
      "last_3f": 0.0,
      "weight": 492,
      "weight_change": 4,
      "winner": "35.4"
    },
    {
      "date": "2022/10/30",
      "venue": "4東京9",
      "weather": "晴",
      "race_round": 11,
      "race_name": "天皇賞(秋)(GI)",
      "race_id": "202205040911",
      "field_size": 15,
      "bracket_number": 4,
      "horse_number": 7,
      "odds": 2.6,
      "popularity": 1,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 56.0,
      "surface": "芝",
      "distance": 2000,
      "time_index": 0,
      "track_condition": "良",
      "finish_time": "1:57.5",
      "time_sec": 117.5,
      "margin": "-0.1",
      "passing_order": "",
      "last_3f": 0.0,
      "weight": 488,
      "weight_change": 4,
      "winner": "32.7"
    },
    {
      "date": "2022/05/29",
      "venue": "2東京12",
      "weather": "晴",
      "race_round": 11,
      "race_name": "東京優駿(GI)",
      "race_id": "202205021211",
      "field_size": 18,
      "bracket_number": 8,
      "horse_number": 18,
      "odds": 3.8,
      "popularity": 2,
      "finish_position": 2,
      "jockey_name": "ルメール",
      "jockey_weight": 57.0,
      "surface": "芝",
      "distance": 2400,
      "time_index": 0,
      "track_condition": "良",
      "finish_time": "2:21.9",
      "time_sec": 141.9,
      "margin": "0.0",
      "passing_order": "",
      "last_3f": 0.0,
      "weight": 484,
      "weight_change": -8,
      "winner": "33.6"
    },
    {
      "date": "2022/04/17",
      "venue": "3中山8",
      "weather": "曇",
      "race_round": 11,
      "race_name": "皐月賞(GI)",
      "race_id": "202206030811",
      "field_size": 18,
      "bracket_number": 8,
      "horse_number": 18,
      "odds": 5.7,
      "popularity": 3,
      "finish_position": 2,
      "jockey_name": "ルメール",
      "jockey_weight": 57.0,
      "surface": "芝",
      "distance": 2000,
      "time_index": 0,
      "track_condition": "良",
      "finish_time": "1:59.8",
      "time_sec": 119.8,
      "margin": "0.1",
      "passing_order": "",
      "last_3f": 0.0,
      "weight": 492,
      "weight_change": 10,
      "winner": "34.6"
    },
    {
      "date": "2021/11/20",
      "venue": "5東京5",
      "weather": "晴",
      "race_round": 11,
      "race_name": "東京スポーツ杯2歳S(GII)",
      "race_id": "202105050511",
      "field_size": 12,
      "bracket_number": 1,
      "horse_number": 1,
      "odds": 2.6,
      "popularity": 1,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 55.0,
      "surface": "芝",
      "distance": 1800,
      "time_index": 0,
      "track_condition": "良",
      "finish_time": "1:46.2",
      "time_sec": 106.2,
      "margin": "-0.4",
      "passing_order": "",
      "last_3f": 0.0,
      "weight": 482,
      "weight_change": 8,
      "winner": "32.9"
    },
    {
      "date": "2021/08/28",
      "venue": "4新潟5",
      "weather": "曇",
      "race_round": 5,
      "race_name": "2歳新馬",
      "race_id": "202104040505",
      "field_size": 15,
      "bracket_number": 2,
      "horse_number": 2,
      "odds": 4.6,
      "popularity": 2,
      "finish_position": 1,
      "jockey_name": "ルメール",
      "jockey_weight": 54.0,
      "surface": "芝",
      "distance": 1800,
      "time_index": 0,
      "track_condition": "良",
      "finish_time": "1:47.4",
      "time_sec": 107.4,
      "margin": "-1.0",
      "passing_order": "",
      "last_3f": 0.0,
      "weight": 474,
      "weight_change": 0,
      "winner": "34.5"
    }
  ]
}
```

</details>


<details>
<summary><code>nk_horse_pedigree</code> 馬血統データ（`horse_pedigree_5gen`）</summary>

```json
{
  "horse_id": "2019105219",
  "sex": "",
  "sire": "キタサンブラック",
  "dam": "シャトーブランシュ",
  "dam_sire": "キングヘイロー",
  "ancestors": [
    {
      "generation": 1,
      "position": 0,
      "name": "キタサンブラック",
      "horse_id": "2012102013",
      "sex": "牡"
    },
    {
      "generation": 2,
      "position": 0,
      "name": "ブラックタイド",
      "horse_id": "2001103312",
      "sex": "牡"
    },
    {
      "generation": 3,
      "position": 0,
      "name": "サンデーサイレンスSunday Silence(米)",
      "horse_id": "000a00033a",
      "sex": "牡"
    },
    {
      "generation": 4,
      "position": 0,
      "name": "Halo",
      "horse_id": "000a0012bf",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 0,
      "name": "Hail to Reason",
      "horse_id": "000a000f2b",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 1,
      "name": "Cosmah",
      "horse_id": "000a007459",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 1,
      "name": "Wishing Well",
      "horse_id": "000a008c1e",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 2,
      "name": "Understanding",
      "horse_id": "000a0019b6",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 3,
      "name": "Mountain Flower",
      "horse_id": "000a008c1d",
      "sex": ""
    },
    {
      "generation": 3,
      "position": 1,
      "name": "ウインドインハーヘアWind in Her Hair(愛)",
      "horse_id": "000a0003a2",
      "sex": "牝"
    },
    {
      "generation": 4,
      "position": 2,
      "name": "Alzao",
      "horse_id": "000a001cb4",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 4,
      "name": "Lyphard",
      "horse_id": "000a001205",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 5,
      "name": "Lady Rebecca",
      "horse_id": "000a009170",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 3,
      "name": "Burghclere",
      "horse_id": "000a00922c",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 6,
      "name": "Busted",
      "horse_id": "000a000e0d",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 7,
      "name": "Highclere",
      "horse_id": "000a008341",
      "sex": ""
    },
    {
      "generation": 2,
      "position": 1,
      "name": "シュガーハート",
      "horse_id": "2005106935",
      "sex": "牝"
    },
    {
      "generation": 3,
      "position": 2,
      "name": "サクラバクシンオー",
      "horse_id": "1989108341",
      "sex": "牡"
    },
    {
      "generation": 4,
      "position": 4,
      "name": "サクラユタカオー",
      "horse_id": "1982101222",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 8,
      "name": "テスコボーイ",
      "horse_id": "000a000355",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 9,
      "name": "アンジェリカ",
      "horse_id": "1955103119",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 5,
      "name": "サクラハゴロモ",
      "horse_id": "1984104366",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 10,
      "name": "ノーザンテースト",
      "horse_id": "000a000258",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 11,
      "name": "クリアアンバー",
      "horse_id": "000a00058e",
      "sex": ""
    },
    {
      "generation": 3,
      "position": 3,
      "name": "オトメゴコロ",
      "horse_id": "1990104600",
      "sex": "牝"
    },
    {
      "generation": 4,
      "position": 6,
      "name": "ジャッジアンジェルーチ",
      "horse_id": "000a000d30",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 12,
      "name": "Honest Pleasure",
      "horse_id": "000a00062c",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 13,
      "name": "Victorian Queen",
      "horse_id": "000a008824",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 7,
      "name": "テイズリー",
      "horse_id": "000a000364",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 14,
      "name": "Lyphard",
      "horse_id": "000a001205",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 15,
      "name": "Tizna",
      "horse_id": "000a008aad",
      "sex": ""
    },
    {
      "generation": 1,
      "position": 1,
      "name": "シャトーブランシュ",
      "horse_id": "2010104274",
      "sex": "牝"
    },
    {
      "generation": 2,
      "position": 2,
      "name": "キングヘイロー",
      "horse_id": "1995104427",
      "sex": "牡"
    },
    {
      "generation": 3,
      "position": 4,
      "name": "ダンシングブレーヴDancing Brave(米)",
      "horse_id": "000a0000cc",
      "sex": "牡"
    },
    {
      "generation": 4,
      "position": 8,
      "name": "Lyphard",
      "horse_id": "000a001205",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 16,
      "name": "Northern Dancer",
      "horse_id": "000a000e04",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 17,
      "name": "Goofed",
      "horse_id": "000a007c8d",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 9,
      "name": "Navajo Princess",
      "horse_id": "000a0081bb",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 18,
      "name": "Drone",
      "horse_id": "000a001668",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 19,
      "name": "Olmec",
      "horse_id": "000a0081ba",
      "sex": ""
    },
    {
      "generation": 3,
      "position": 5,
      "name": "グッバイヘイローGoodbye Halo(米)",
      "horse_id": "000a0062e2",
      "sex": "牝"
    },
    {
      "generation": 4,
      "position": 10,
      "name": "Halo",
      "horse_id": "000a0012bf",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 20,
      "name": "Hail to Reason",
      "horse_id": "000a000f2b",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 21,
      "name": "Cosmah",
      "horse_id": "000a007459",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 11,
      "name": "Pound Foolish",
      "horse_id": "000a00877d",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 22,
      "name": "Sir Ivor",
      "horse_id": "000a000dd9",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 23,
      "name": "Squander",
      "horse_id": "000a007e4f",
      "sex": ""
    },
    {
      "generation": 2,
      "position": 3,
      "name": "ブランシェリー",
      "horse_id": "1998101846",
      "sex": "牝"
    },
    {
      "generation": 3,
      "position": 6,
      "name": "トニービンTony Bin(愛)",
      "horse_id": "1983109006",
      "sex": "牡"
    },
    {
      "generation": 4,
      "position": 12,
      "name": "カンパラ",
      "horse_id": "000a000827",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 24,
      "name": "Kalamoun",
      "horse_id": "000a0016c2",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 25,
      "name": "State Pension",
      "horse_id": "000a00859a",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 13,
      "name": "Severn Bridge",
      "horse_id": "000a00859c",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 26,
      "name": "Hornbeam",
      "horse_id": "000a000e25",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 27,
      "name": "Priddy Fair",
      "horse_id": "000a00859b",
      "sex": ""
    },
    {
      "generation": 3,
      "position": 7,
      "name": "メゾンブランシュ",
      "horse_id": "1989107947",
      "sex": "牝"
    },
    {
      "generation": 4,
      "position": 14,
      "name": "Alleged",
      "horse_id": "000a0012f7",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 28,
      "name": "Hoist the Flag",
      "horse_id": "000a0012be",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 29,
      "name": "Princess Pout",
      "horse_id": "000a007f75",
      "sex": ""
    },
    {
      "generation": 4,
      "position": 15,
      "name": "ブランシユレイン",
      "horse_id": "000a00006f",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 30,
      "name": "Nureyev",
      "horse_id": "000a001676",
      "sex": ""
    },
    {
      "generation": 5,
      "position": 31,
      "name": "Belga",
      "horse_id": "000a0082dc",
      "sex": ""
    }
  ],
  "ancestor_count": 62,
  "source": "queue_horse_pedigree_5gen",
  "_meta": {
    "scraped_at": 1781192590.1600595,
    "scraped_at_jst": "2026-06-12 00:43:10",
    "schema_validation": {
      "schema_version": 2,
      "passed": true,
      "top_missing": [],
      "top_type_errors": [],
      "top_constraint_errors": [],
      "entry_count": 0,
      "entry_issues": {}
    },
    "scrape_validation_status": "ok"
  }
}
```

</details>


<details>
<summary><code>nk_horse_training</code> 馬調教（`horse_training`）</summary>

```json
{
  "horse_id": "2019105219",
  "total_items": 9,
  "pages_fetched": 1,
  "entries": [
    {
      "race_info": "2023/11/26  東京12R ジャパンカップ 結果 ： 1着",
      "date": "2023-11-22",
      "day_of_week": "水",
      "course": "美Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "84.6 67.5 52.4 37.8 11.3",
      "lap_times": [
        84.6,
        67.5,
        52.4,
        37.8,
        11.3
      ],
      "position": "3",
      "leg_color": "馬也",
      "evaluation": "気配抜群",
      "rank": "A",
      "comment": "この時季でも毛ヅヤはピカピカで、走りのリズムも非常にいい。反動は皆無。"
    },
    {
      "race_info": "2023/11/26  東京12R ジャパンカップ 結果 ： 1着",
      "date": "2023-11-19",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 52.7 38.4 25.2 12.3",
      "lap_times": [
        52.7,
        38.4,
        25.2,
        12.3
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/11/26  東京12R ジャパンカップ 結果 ： 1着",
      "date": "2023-11-15",
      "day_of_week": "水",
      "course": "美Ｗ",
      "track_condition": "稍",
      "rider": "助手",
      "time_raw": "(7)97.7 66.2 51.5 37.3 11.4",
      "lap_times": [
        66.2,
        51.5,
        37.3,
        11.4
      ],
      "position": "3",
      "leg_color": "馬也",
      "evaluation": "気配抜群",
      "rank": "A",
      "comment": ""
    },
    {
      "race_info": "2023/11/26  東京12R ジャパンカップ 結果 ： 1着",
      "date": "2023-11-12",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 54.9 40.1 26.0 12.8",
      "lap_times": [
        54.9,
        40.1,
        26.0,
        12.8
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-25",
      "day_of_week": "水",
      "course": "美Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "84.7 67.6 52.5 37.3 11.3",
      "lap_times": [
        84.7,
        67.6,
        52.5,
        37.3,
        11.3
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "気配抜群",
      "rank": "A",
      "comment": "直線では２頭の間をこじ開けるように伸びてきた。長めを乗り込み、態勢万全。"
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-22",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 53.6 38.8 25.2 12.0",
      "lap_times": [
        53.6,
        38.8,
        25.2,
        12.0
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-18",
      "day_of_week": "水",
      "course": "美Ｗ",
      "track_condition": "良",
      "rider": "ルメー",
      "time_raw": "(7)93.7 65.4 51.4 37.4 11.8",
      "lap_times": [
        65.4,
        51.4,
        37.4,
        11.8
      ],
      "position": "5",
      "leg_color": "馬也",
      "evaluation": "好気配",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-15",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "重",
      "rider": "助手",
      "time_raw": "- 53.7 39.1 25.4 12.4",
      "lap_times": [
        53.7,
        39.1,
        25.4,
        12.4
      ],
      "position": "",
      "leg_color": "Ｇ強",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-11",
      "day_of_week": "水",
      "course": "美Ｗ",
      "track_condition": "稍",
      "rider": "助手",
      "time_raw": "(8)113.5 67.4 52.5 38.2 12.2",
      "lap_times": [
        67.4,
        52.5,
        38.2,
        12.2
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "好気配",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-08",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 55.8 41.2 26.9 13.1",
      "lap_times": [
        55.8,
        41.2,
        26.9,
        13.1
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-04",
      "day_of_week": "水",
      "course": "美Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "86.1 69.2 54.2 39.4 12.4",
      "lap_times": [
        86.1,
        69.2,
        54.2,
        39.4,
        12.4
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "好気配",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2023-10-01",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 60.0 43.1 28.5 14.0",
      "lap_times": [
        60.0,
        43.1,
        28.5,
        14.0
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-06-21",
      "day_of_week": "水",
      "course": "ＣＷ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "82.4 67.9 53.1 37.4 11.3",
      "lap_times": [
        82.4,
        67.9,
        53.1,
        37.4,
        11.3
      ],
      "position": "5",
      "leg_color": "馬也",
      "evaluation": "本調子",
      "rank": "A",
      "comment": "追うごとに気持ちが入り、今週は機敏な走りで鋭い伸び脚。盤石の仕上がり。"
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-06-18",
      "day_of_week": "日",
      "course": "栗坂",
      "track_condition": "良",
      "rider": "",
      "time_raw": "- 53.1 38.0 24.2 11.8",
      "lap_times": [
        53.1,
        38.0,
        24.2,
        11.8
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-06-14",
      "day_of_week": "水",
      "course": "ＣＷ",
      "track_condition": "良",
      "rider": "ルメー",
      "time_raw": "79.3 65.0 50.8 36.8 11.6",
      "lap_times": [
        79.3,
        65.0,
        50.8,
        36.8,
        11.6
      ],
      "position": "5",
      "leg_color": "馬也",
      "evaluation": "追毎良化",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-06-11",
      "day_of_week": "日",
      "course": "栗坂",
      "track_condition": "重",
      "rider": "",
      "time_raw": "- 54.8 38.5 24.5 12.0",
      "lap_times": [
        54.8,
        38.5,
        24.5,
        12.0
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-06-08",
      "day_of_week": "木",
      "course": "ＣＷ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "84.2 69.3 54.4 39.4 12.2",
      "lap_times": [
        84.2,
        69.3,
        54.4,
        39.4,
        12.2
      ],
      "position": "5",
      "leg_color": "馬也",
      "evaluation": "動き軽快",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-06-03",
      "day_of_week": "土",
      "course": "南Ｗ",
      "track_condition": "重",
      "rider": "助手",
      "time_raw": "85.5 69.9 55.1 39.9 12.2",
      "lap_times": [
        85.5,
        69.9,
        55.1,
        39.9,
        12.2
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-06-01",
      "day_of_week": "木",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "(7)100.8 69.9 54.8 40.1 12.7",
      "lap_times": [
        69.9,
        54.8,
        40.1,
        12.7
      ],
      "position": "3",
      "leg_color": "馬也",
      "evaluation": "好気配",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2023/06/25  阪神11R 宝塚記念 結果 ： 1着",
      "date": "2023-05-28",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 58.8 43.1 28.2 13.8",
      "lap_times": [
        58.8,
        43.1,
        28.2,
        13.8
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/12/25  中山11R 有馬記念 結果 ： 1着",
      "date": "2022-12-21",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "84.7 67.2 52.2 37.6 11.4",
      "lap_times": [
        84.7,
        67.2,
        52.2,
        37.6,
        11.4
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "気配抜群",
      "rank": "A",
      "comment": "自らグイグイ進んで馬なりのまま突き抜けた。この時季でも毛ヅヤはピカピカ。"
    },
    {
      "race_info": "2022/12/25  中山11R 有馬記念 結果 ： 1着",
      "date": "2022-12-18",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "稍",
      "rider": "助手",
      "time_raw": "- 53.5 39.1 25.1 12.1",
      "lap_times": [
        53.5,
        39.1,
        25.1,
        12.1
      ],
      "position": "",
      "leg_color": "一杯",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/12/25  中山11R 有馬記念 結果 ： 1着",
      "date": "2022-12-14",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "稍",
      "rider": "ルメー",
      "time_raw": "(7)96.0 67.3 52.6 38.4 12.0",
      "lap_times": [
        67.3,
        52.6,
        38.4,
        12.0
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "気配抜群",
      "rank": "A",
      "comment": ""
    },
    {
      "race_info": "2022/12/25  中山11R 有馬記念 結果 ： 1着",
      "date": "2022-12-11",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 55.0 40.2 25.9 12.6",
      "lap_times": [
        55.0,
        40.2,
        25.9,
        12.6
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/12/25  中山11R 有馬記念 結果 ： 1着",
      "date": "2022-12-07",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "稍",
      "rider": "助手",
      "time_raw": "84.1 68.8 54.1 40.1 12.8",
      "lap_times": [
        84.1,
        68.8,
        54.1,
        40.1,
        12.8
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "好気配",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2022/12/25  中山11R 有馬記念 結果 ： 1着",
      "date": "2022-12-04",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 57.8 42.5 27.8 13.7",
      "lap_times": [
        57.8,
        42.5,
        27.8,
        13.7
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/10/30  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2022-10-26",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "86.8 69.6 54.1 38.6 11.5",
      "lap_times": [
        86.8,
        69.6,
        54.1,
        38.6,
        11.5
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "態勢万全",
      "rank": "A",
      "comment": "ひと追いごとに上向き、今週は促した程度で圧巻の加速力。体の張りも抜群。"
    },
    {
      "race_info": "2022/10/30  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2022-10-23",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 54.1 39.8 25.7 12.3",
      "lap_times": [
        54.1,
        39.8,
        25.7,
        12.3
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/10/30  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2022-10-19",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "重",
      "rider": "ルメー",
      "time_raw": "82.3 67.1 51.7 37.6 11.7",
      "lap_times": [
        82.3,
        67.1,
        51.7,
        37.6,
        11.7
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "気配抜群",
      "rank": "A",
      "comment": ""
    },
    {
      "race_info": "2022/10/30  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2022-10-16",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 54.5 39.9 25.7 12.6",
      "lap_times": [
        54.5,
        39.9,
        25.7,
        12.6
      ],
      "position": "",
      "leg_color": "Ｇ強",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/10/30  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2022-10-12",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "(7)102.5 71.7 56.7 41.4 13.6",
      "lap_times": [
        71.7,
        56.7,
        41.4,
        13.6
      ],
      "position": "5",
      "leg_color": "馬也",
      "evaluation": "好気配",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2022/10/30  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2022-10-09",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 58.1 42.8 27.9 14.0",
      "lap_times": [
        58.1,
        42.8,
        27.9,
        14.0
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/10/30  東京11R 天皇賞(秋) 結果 ： 1着",
      "date": "2022-10-05",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "87.7 70.3 55.0 39.9 12.6",
      "lap_times": [
        87.7,
        70.3,
        55.0,
        39.9,
        12.6
      ],
      "position": "2",
      "leg_color": "馬也",
      "evaluation": "迫力十分",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2022/05/29  東京11R 東京優駿 結果 ： 2着",
      "date": "2022-05-25",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "ルメー",
      "time_raw": "86.0 68.4 52.9 38.2 11.6",
      "lap_times": [
        86.0,
        68.4,
        52.9,
        38.2,
        11.6
      ],
      "position": "2",
      "leg_color": "馬也",
      "evaluation": "態勢整う",
      "rank": "B",
      "comment": "派手な時計は出していないが、動きは滑らかで、体も締まっている。好気配。"
    },
    {
      "race_info": "2022/05/29  東京11R 東京優駿 結果 ： 2着",
      "date": "2022-05-22",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "稍",
      "rider": "助手",
      "time_raw": "- 54.2 39.3 24.8 11.9",
      "lap_times": [
        54.2,
        39.3,
        24.8,
        11.9
      ],
      "position": "2",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/05/29  東京11R 東京優駿 結果 ： 2着",
      "date": "2022-05-18",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "稍",
      "rider": "ルメー",
      "time_raw": "83.0 67.7 52.6 38.2 12.0",
      "lap_times": [
        83.0,
        67.7,
        52.6,
        38.2,
        12.0
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "好調子",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2022/04/17  中山11R 皐月賞 結果 ： 2着",
      "date": "2022-04-13",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "ルメー",
      "time_raw": "84.8 68.3 53.3 38.6 11.6",
      "lap_times": [
        84.8,
        68.3,
        53.3,
        38.6,
        11.6
      ],
      "position": "5",
      "leg_color": "馬也",
      "evaluation": "態勢万全",
      "rank": "A",
      "comment": "休養を経て体つきが良くなった。馬なりのまま加速し、動きは今までで一番。"
    },
    {
      "race_info": "2022/04/17  中山11R 皐月賞 結果 ： 2着",
      "date": "2022-04-10",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 55.9 40.5 25.7 12.7",
      "lap_times": [
        55.9,
        40.5,
        25.7,
        12.7
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/04/17  中山11R 皐月賞 結果 ： 2着",
      "date": "2022-04-06",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "稍",
      "rider": "ルメー",
      "time_raw": "(7)98.2 67.0 51.5 37.2 11.4",
      "lap_times": [
        67.0,
        51.5,
        37.2,
        11.4
      ],
      "position": "5",
      "leg_color": "馬也",
      "evaluation": "迫力満点",
      "rank": "A",
      "comment": ""
    },
    {
      "race_info": "2022/04/17  中山11R 皐月賞 結果 ： 2着",
      "date": "2022-04-03",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 55.7 40.6 26.4 13.0",
      "lap_times": [
        55.7,
        40.6,
        26.4,
        13.0
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2022/04/17  中山11R 皐月賞 結果 ： 2着",
      "date": "2022-03-30",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "(7)98.8 68.0 53.4 39.5 12.2",
      "lap_times": [
        68.0,
        53.4,
        39.5,
        12.2
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "好気配",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2022/04/17  中山11R 皐月賞 結果 ： 2着",
      "date": "2022-03-27",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "重",
      "rider": "助手",
      "time_raw": "- 56.3 41.6 27.3 13.6",
      "lap_times": [
        56.3,
        41.6,
        27.3,
        13.6
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/11/20  東京11R 東京スポーツ杯2歳S 結果 ： 1着",
      "date": "2021-11-17",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 68.0 53.0 38.2 11.7",
      "lap_times": [
        68.0,
        53.0,
        38.2,
        11.7
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "動き良化",
      "rank": "B",
      "comment": "まだスッと動けなかった初戦から行きっぷり、反応ともに良化。成長している。"
    },
    {
      "race_info": "2021/11/20  東京11R 東京スポーツ杯2歳S 結果 ： 1着",
      "date": "2021-11-14",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 54.7 39.3 25.3 12.4",
      "lap_times": [
        54.7,
        39.3,
        25.3,
        12.4
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/11/20  東京11R 東京スポーツ杯2歳S 結果 ： 1着",
      "date": "2021-11-10",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "重",
      "rider": "助手",
      "time_raw": "86.0 69.4 54.5 39.8 11.6",
      "lap_times": [
        86.0,
        69.4,
        54.5,
        39.8,
        11.6
      ],
      "position": "4",
      "leg_color": "馬也",
      "evaluation": "気配上々",
      "rank": "B",
      "comment": ""
    },
    {
      "race_info": "2021/11/20  東京11R 東京スポーツ杯2歳S 結果 ： 1着",
      "date": "2021-11-07",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 55.3 40.3 26.5 13.0",
      "lap_times": [
        55.3,
        40.3,
        26.5,
        13.0
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/11/20  東京11R 東京スポーツ杯2歳S 結果 ： 1着",
      "date": "2021-11-03",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 68.8 53.2 38.5 11.8",
      "lap_times": [
        68.8,
        53.2,
        38.5,
        11.8
      ],
      "position": "2",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/11/20  東京11R 東京スポーツ杯2歳S 結果 ： 1着",
      "date": "2021-10-31",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 58.2 42.4 27.7 13.6",
      "lap_times": [
        58.2,
        42.4,
        27.7,
        13.6
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/08/28  新潟5R 2歳新馬 結果 ： 1着",
      "date": "2021-08-25",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 65.9 51.7 37.6 11.6",
      "lap_times": [
        65.9,
        51.7,
        37.6,
        11.6
      ],
      "position": "4",
      "leg_color": "Ｇ一",
      "evaluation": "態勢整う",
      "rank": "B",
      "comment": "２頭で挟んで実戦を意識させたが、伸びやかなフォームで鋭伸。態勢ＯＫ。"
    },
    {
      "race_info": "2021/08/28  新潟5R 2歳新馬 結果 ： 1着",
      "date": "2021-08-22",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 54.9 39.9 26.0 12.7",
      "lap_times": [
        54.9,
        39.9,
        26.0,
        12.7
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/08/28  新潟5R 2歳新馬 結果 ： 1着",
      "date": "2021-08-18",
      "day_of_week": "水",
      "course": "南Ｗ",
      "track_condition": "稍",
      "rider": "助手",
      "time_raw": "83.1 68.2 53.1 38.2 11.7",
      "lap_times": [
        83.1,
        68.2,
        53.1,
        38.2,
        11.7
      ],
      "position": "3",
      "leg_color": "直強",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/08/28  新潟5R 2歳新馬 結果 ： 1着",
      "date": "2021-08-15",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "不",
      "rider": "助手",
      "time_raw": "- 58.8 43.6 28.2 14.1",
      "lap_times": [
        58.8,
        43.6,
        28.2,
        14.1
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/08/28  新潟5R 2歳新馬 結果 ： 1着",
      "date": "2021-08-12",
      "day_of_week": "木",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 71.3 55.0 39.3 12.0",
      "lap_times": [
        71.3,
        55.0,
        39.3,
        12.0
      ],
      "position": "3",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/08/28  新潟5R 2歳新馬 結果 ： 1着",
      "date": "2021-08-08",
      "day_of_week": "日",
      "course": "美坂",
      "track_condition": "重",
      "rider": "助手",
      "time_raw": "- 56.9 42.0 27.8 13.7",
      "lap_times": [
        56.9,
        42.0,
        27.8,
        13.7
      ],
      "position": "",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    },
    {
      "race_info": "2021/08/28  新潟5R 2歳新馬 結果 ： 1着",
      "date": "2021-08-05",
      "day_of_week": "木",
      "course": "南Ｗ",
      "track_condition": "良",
      "rider": "助手",
      "time_raw": "- 73.0 56.7 41.5 13.2",
      "lap_times": [
        73.0,
        56.7,
        41.5,
        13.2
      ],
      "position": "2",
      "leg_color": "馬也",
      "evaluation": "順調",
      "rank": "C",
      "comment": ""
    }
  ],
  "_meta": {
    "scraped_at": 1781192597.0731978,
    "scraped_at_jst": "2026-06-12 00:43:17",
    "schema_validation": {
      "schema_version": 2,
      "passed": true,
      "top_missing": [],
      "top_type_errors": [],
      "top_constraint_errors": [],
      "entry_count": 55,
      "entry_issues": {}
    },
    "scrape_validation_status": "ok"
  }
}
```

</details>


<details>
<summary><code>nk_race_list</code> レースID一覧（`race_lists`・`20230625`）</summary>

```json
{
  "date": "20230625",
  "races": [
    {
      "race_id": "202302010601",
      "round": 1,
      "venue": "函館",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202302010602",
      "round": 2,
      "venue": "函館",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202302010603",
      "round": 3,
      "venue": "函館",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202302010604",
      "round": 4,
      "venue": "函館",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202302010605",
      "round": 5,
      "venue": "函館",
      "race_name": "2歳新馬"
    },
    {
      "race_id": "202302010606",
      "round": 6,
      "venue": "函館",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202302010607",
      "round": 7,
      "venue": "函館",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202302010608",
      "round": 8,
      "venue": "函館",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202302010609",
      "round": 9,
      "venue": "函館",
      "race_name": "北海ハンデキャップ(2勝クラス)"
    },
    {
      "race_id": "202302010610",
      "round": 10,
      "venue": "函館",
      "race_name": "HTB杯(2勝クラス)"
    },
    {
      "race_id": "202302010611",
      "round": 11,
      "venue": "函館",
      "race_name": "大沼S(L)"
    },
    {
      "race_id": "202302010612",
      "round": 12,
      "venue": "函館",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202305030801",
      "round": 1,
      "venue": "東京",
      "race_name": "2歳未勝利"
    },
    {
      "race_id": "202305030802",
      "round": 2,
      "venue": "東京",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202305030803",
      "round": 3,
      "venue": "東京",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202305030804",
      "round": 4,
      "venue": "東京",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202305030805",
      "round": 5,
      "venue": "東京",
      "race_name": "2歳新馬"
    },
    {
      "race_id": "202305030806",
      "round": 6,
      "venue": "東京",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202305030807",
      "round": 7,
      "venue": "東京",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202305030808",
      "round": 8,
      "venue": "東京",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202305030809",
      "round": 9,
      "venue": "東京",
      "race_name": "八ヶ岳特別(2勝クラス)"
    },
    {
      "race_id": "202305030810",
      "round": 10,
      "venue": "東京",
      "race_name": "甲州街道S(3勝クラス)"
    },
    {
      "race_id": "202305030811",
      "round": 11,
      "venue": "東京",
      "race_name": "パラダイスS(L)"
    },
    {
      "race_id": "202305030812",
      "round": 12,
      "venue": "東京",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202309030801",
      "round": 1,
      "venue": "阪神",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202309030802",
      "round": 2,
      "venue": "阪神",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202309030803",
      "round": 3,
      "venue": "阪神",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202309030804",
      "round": 4,
      "venue": "阪神",
      "race_name": "3歳未勝利"
    },
    {
      "race_id": "202309030805",
      "round": 5,
      "venue": "阪神",
      "race_name": "2歳新馬"
    },
    {
      "race_id": "202309030806",
      "round": 6,
      "venue": "阪神",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202309030807",
      "round": 7,
      "venue": "阪神",
      "race_name": "3歳以上1勝クラス"
    },
    {
      "race_id": "202309030808",
      "round": 8,
      "venue": "阪神",
      "race_name": "城崎特別(1勝クラス)"
    },
    {
      "race_id": "202309030809",
      "round": 9,
      "venue": "阪神",
      "race_name": "舞子特別(2勝クラス)"
    },
    {
      "race_id": "202309030810",
      "round": 10,
      "venue": "阪神",
      "race_name": "花のみちS(3勝クラス)"
    },
    {
      "race_id": "202309030811",
      "round": 11,
      "venue": "阪神",
      "race_name": "宝塚記念(G1)"
    },
    {
      "race_id": "202309030812",
      "round": 12,
      "venue": "阪神",
      "race_name": "リボン賞(2勝クラス)"
    }
  ],
  "_meta": {
    "race_list_source": "db.netkeiba.com",
    "scraped_at": 1781192604.3473468,
    "scraped_at_jst": "2026-06-12 00:43:24",
    "schema_validation": {
      "schema_version": 2,
      "passed": true,
      "top_missing": [],
      "top_type_errors": [],
      "top_constraint_errors": [],
      "entry_count": 36,
      "entry_issues": {}
    },
    "scrape_validation_status": "ok"
  }
}
```

</details>


<details>
<summary><code>nk_race_day_schedule</code> 発走時刻表（`race_day_schedule`・`20230625`）</summary>

```json
{
  "date_fmt": "20230625",
  "iso_date": "2023-06-25",
  "slots": [
    {
      "race_id": "202302010601",
      "venue": "函館",
      "round": 1,
      "race_name": "3歳未勝利",
      "start_time_str": "09:50",
      "post_time_iso": "2023-06-25T09:50:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030801",
      "venue": "東京",
      "round": 1,
      "race_name": "2歳未勝利",
      "start_time_str": "09:55",
      "post_time_iso": "2023-06-25T09:55:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030801",
      "venue": "阪神",
      "round": 1,
      "race_name": "3歳未勝利",
      "start_time_str": "10:05",
      "post_time_iso": "2023-06-25T10:05:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010602",
      "venue": "函館",
      "round": 2,
      "race_name": "3歳未勝利",
      "start_time_str": "10:15",
      "post_time_iso": "2023-06-25T10:15:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030802",
      "venue": "東京",
      "round": 2,
      "race_name": "3歳未勝利",
      "start_time_str": "10:25",
      "post_time_iso": "2023-06-25T10:25:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030802",
      "venue": "阪神",
      "round": 2,
      "race_name": "3歳未勝利",
      "start_time_str": "10:35",
      "post_time_iso": "2023-06-25T10:35:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010603",
      "venue": "函館",
      "round": 3,
      "race_name": "3歳未勝利",
      "start_time_str": "10:45",
      "post_time_iso": "2023-06-25T10:45:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030803",
      "venue": "東京",
      "round": 3,
      "race_name": "3歳未勝利",
      "start_time_str": "10:55",
      "post_time_iso": "2023-06-25T10:55:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030803",
      "venue": "阪神",
      "round": 3,
      "race_name": "3歳未勝利",
      "start_time_str": "11:05",
      "post_time_iso": "2023-06-25T11:05:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010604",
      "venue": "函館",
      "round": 4,
      "race_name": "3歳未勝利",
      "start_time_str": "11:15",
      "post_time_iso": "2023-06-25T11:15:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030804",
      "venue": "東京",
      "round": 4,
      "race_name": "3歳未勝利",
      "start_time_str": "11:25",
      "post_time_iso": "2023-06-25T11:25:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030804",
      "venue": "阪神",
      "round": 4,
      "race_name": "3歳未勝利",
      "start_time_str": "11:35",
      "post_time_iso": "2023-06-25T11:35:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010605",
      "venue": "函館",
      "round": 5,
      "race_name": "2歳新馬",
      "start_time_str": "12:05",
      "post_time_iso": "2023-06-25T12:05:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030805",
      "venue": "東京",
      "round": 5,
      "race_name": "2歳新馬",
      "start_time_str": "12:15",
      "post_time_iso": "2023-06-25T12:15:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030805",
      "venue": "阪神",
      "round": 5,
      "race_name": "2歳新馬",
      "start_time_str": "12:25",
      "post_time_iso": "2023-06-25T12:25:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010606",
      "venue": "函館",
      "round": 6,
      "race_name": "3歳未勝利",
      "start_time_str": "12:35",
      "post_time_iso": "2023-06-25T12:35:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030806",
      "venue": "東京",
      "round": 6,
      "race_name": "3歳未勝利",
      "start_time_str": "12:45",
      "post_time_iso": "2023-06-25T12:45:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030806",
      "venue": "阪神",
      "round": 6,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "12:55",
      "post_time_iso": "2023-06-25T12:55:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010607",
      "venue": "函館",
      "round": 7,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "13:05",
      "post_time_iso": "2023-06-25T13:05:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030807",
      "venue": "東京",
      "round": 7,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "13:15",
      "post_time_iso": "2023-06-25T13:15:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030807",
      "venue": "阪神",
      "round": 7,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "13:25",
      "post_time_iso": "2023-06-25T13:25:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010608",
      "venue": "函館",
      "round": 8,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "13:35",
      "post_time_iso": "2023-06-25T13:35:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030808",
      "venue": "東京",
      "round": 8,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "13:45",
      "post_time_iso": "2023-06-25T13:45:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030808",
      "venue": "阪神",
      "round": 8,
      "race_name": "城崎特別(1勝クラス)",
      "start_time_str": "13:55",
      "post_time_iso": "2023-06-25T13:55:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010609",
      "venue": "函館",
      "round": 9,
      "race_name": "北海ハンデキャップ(2勝クラス)",
      "start_time_str": "14:05",
      "post_time_iso": "2023-06-25T14:05:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030809",
      "venue": "東京",
      "round": 9,
      "race_name": "八ヶ岳特別(2勝クラス)",
      "start_time_str": "14:15",
      "post_time_iso": "2023-06-25T14:15:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030809",
      "venue": "阪神",
      "round": 9,
      "race_name": "舞子特別(2勝クラス)",
      "start_time_str": "14:25",
      "post_time_iso": "2023-06-25T14:25:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010610",
      "venue": "函館",
      "round": 10,
      "race_name": "HTB杯(2勝クラス)",
      "start_time_str": "14:40",
      "post_time_iso": "2023-06-25T14:40:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030810",
      "venue": "東京",
      "round": 10,
      "race_name": "甲州街道S(3勝クラス)",
      "start_time_str": "14:50",
      "post_time_iso": "2023-06-25T14:50:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030810",
      "venue": "阪神",
      "round": 10,
      "race_name": "花のみちS(3勝クラス)",
      "start_time_str": "15:01",
      "post_time_iso": "2023-06-25T15:01:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010611",
      "venue": "函館",
      "round": 11,
      "race_name": "大沼S(L)",
      "start_time_str": "15:20",
      "post_time_iso": "2023-06-25T15:20:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030811",
      "venue": "東京",
      "round": 11,
      "race_name": "パラダイスS(L)",
      "start_time_str": "15:30",
      "post_time_iso": "2023-06-25T15:30:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030811",
      "venue": "阪神",
      "round": 11,
      "race_name": "宝塚記念(G1)",
      "start_time_str": "15:40",
      "post_time_iso": "2023-06-25T15:40:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202302010612",
      "venue": "函館",
      "round": 12,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "16:05",
      "post_time_iso": "2023-06-25T16:05:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202305030812",
      "venue": "東京",
      "round": 12,
      "race_name": "3歳以上1勝クラス",
      "start_time_str": "16:15",
      "post_time_iso": "2023-06-25T16:15:00+09:00",
      "time_source": "shutuba"
    },
    {
      "race_id": "202309030812",
      "venue": "阪神",
      "round": 12,
      "race_name": "リボン賞(2勝クラス)",
      "start_time_str": "16:30",
      "post_time_iso": "2023-06-25T16:30:00+09:00",
      "time_source": "shutuba"
    }
  ],
  "_meta": {
    "source": "synthesized",
    "built_from": "race_lists+race_shutuba",
    "generated_at": 1781202148.920007
  }
}
```

</details>


<details>
<summary><code>nk_db_race_result</code> 結果DB（`race_result`）</summary>

```json
{
  "race_id": "202309030811",
  "race_name": "第64回宝塚記念",
  "grade": "G1",
  "surface": "芝",
  "direction": "右",
  "distance": 2200,
  "weather": "曇",
  "track_condition": "良",
  "start_time": "15:40",
  "date": "2023-06-25",
  "venue": "阪神",
  "round": 0,
  "entries": [
    {
      "finish_position": 1,
      "bracket_number": 3,
      "horse_number": 5,
      "horse_name": "イクイノックス",
      "horse_id": "2019105219",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "ルメール",
      "jockey_id": "05339",
      "finish_time": "2:11.2",
      "time_sec": 131.2,
      "margin": "",
      "passing_order": "16-16-13-9",
      "last_3f": 34.8,
      "odds": 1.3,
      "popularity": 1,
      "weight": 492,
      "weight_change": 0,
      "trainer_name": "木村哲也",
      "trainer_id": "01126"
    },
    {
      "finish_position": 2,
      "bracket_number": 3,
      "horse_number": 6,
      "horse_name": "スルーセブンシーズ",
      "horse_id": "2018105269",
      "sex_age": "牝5",
      "jockey_weight": 56.0,
      "jockey_name": "池添謙一",
      "jockey_id": "01032",
      "finish_time": "2:11.2",
      "time_sec": 131.2,
      "margin": "クビ",
      "passing_order": "17-17-16-12",
      "last_3f": 34.6,
      "odds": 55.7,
      "popularity": 10,
      "weight": 446,
      "weight_change": -2,
      "trainer_name": "尾関知人",
      "trainer_id": "01103"
    },
    {
      "finish_position": 3,
      "bracket_number": 5,
      "horse_number": 9,
      "horse_name": "ジャスティンパレス",
      "horse_id": "2019105346",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "鮫島克駿",
      "jockey_id": "01157",
      "finish_time": "2:11.4",
      "time_sec": 131.4,
      "margin": "1",
      "passing_order": "12-13-11-9",
      "last_3f": 35.1,
      "odds": 8.5,
      "popularity": 2,
      "weight": 470,
      "weight_change": -2,
      "trainer_name": "杉山晴紀",
      "trainer_id": "01157"
    },
    {
      "finish_position": 4,
      "bracket_number": 6,
      "horse_number": 11,
      "horse_name": "ジェラルディーナ",
      "horse_id": "2018105081",
      "sex_age": "牝5",
      "jockey_weight": 56.0,
      "jockey_name": "武豊",
      "jockey_id": "00666",
      "finish_time": "2:11.4",
      "time_sec": 131.4,
      "margin": "アタマ",
      "passing_order": "14-14-6-3",
      "last_3f": 35.5,
      "odds": 13.8,
      "popularity": 3,
      "weight": 466,
      "weight_change": 3,
      "trainer_name": "斉藤崇史",
      "trainer_id": "01151"
    },
    {
      "finish_position": 5,
      "bracket_number": 5,
      "horse_number": 10,
      "horse_name": "ディープボンド",
      "horse_id": "2017102170",
      "sex_age": "牡6",
      "jockey_weight": 58.0,
      "jockey_name": "和田竜二",
      "jockey_id": "01018",
      "finish_time": "2:11.6",
      "time_sec": 131.6,
      "margin": "1",
      "passing_order": "7-7-8-6",
      "last_3f": 35.5,
      "odds": 16.6,
      "popularity": 5,
      "weight": 502,
      "weight_change": -2,
      "trainer_name": "大久保龍",
      "trainer_id": "01058"
    },
    {
      "finish_position": 6,
      "bracket_number": 4,
      "horse_number": 7,
      "horse_name": "プラダリア",
      "horse_id": "2019100109",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "菱田裕二",
      "jockey_id": "01144",
      "finish_time": "2:11.6",
      "time_sec": 131.6,
      "margin": "クビ",
      "passing_order": "11-10-11-12",
      "last_3f": 35.3,
      "odds": 262.5,
      "popularity": 16,
      "weight": 466,
      "weight_change": 2,
      "trainer_name": "池添学",
      "trainer_id": "01144"
    },
    {
      "finish_position": 7,
      "bracket_number": 2,
      "horse_number": 4,
      "horse_name": "ボッケリーニ",
      "horse_id": "2016104618",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "浜中俊",
      "jockey_id": "01115",
      "finish_time": "2:11.7",
      "time_sec": 131.7,
      "margin": "クビ",
      "passing_order": "12-10-13-16",
      "last_3f": 35.1,
      "odds": 28.3,
      "popularity": 6,
      "weight": 466,
      "weight_change": -2,
      "trainer_name": "池江泰寿",
      "trainer_id": "01071"
    },
    {
      "finish_position": 8,
      "bracket_number": 4,
      "horse_number": 8,
      "horse_name": "ヴェラアズール",
      "horse_id": "2017105082",
      "sex_age": "牡6",
      "jockey_weight": 58.0,
      "jockey_name": "松山弘平",
      "jockey_id": "01126",
      "finish_time": "2:11.9",
      "time_sec": 131.9,
      "margin": "1.1/4",
      "passing_order": "14-14-16-16",
      "last_3f": 35.2,
      "odds": 42.1,
      "popularity": 9,
      "weight": 520,
      "weight_change": 0,
      "trainer_name": "渡辺薫彦",
      "trainer_id": "01155"
    },
    {
      "finish_position": 9,
      "bracket_number": 7,
      "horse_number": 13,
      "horse_name": "ジオグリフ",
      "horse_id": "2019105056",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "岩田望来",
      "jockey_id": "01174",
      "finish_time": "2:11.9",
      "time_sec": 131.9,
      "margin": "ハナ",
      "passing_order": "7-7-8-6",
      "last_3f": 35.8,
      "odds": 83.1,
      "popularity": 11,
      "weight": 510,
      "weight_change": 0,
      "trainer_name": "木村哲也",
      "trainer_id": "01126"
    },
    {
      "finish_position": 10,
      "bracket_number": 8,
      "horse_number": 17,
      "horse_name": "ドゥラエレーデ",
      "horse_id": "2020103626",
      "sex_age": "牡3",
      "jockey_weight": 53.0,
      "jockey_name": "幸英明",
      "jockey_id": "00732",
      "finish_time": "2:12.3",
      "time_sec": 132.3,
      "margin": "2",
      "passing_order": "2-2-2-2",
      "last_3f": 36.5,
      "odds": 30.2,
      "popularity": 7,
      "weight": 506,
      "weight_change": -6,
      "trainer_name": "池添学",
      "trainer_id": "01144"
    },
    {
      "finish_position": 11,
      "bracket_number": 6,
      "horse_number": 12,
      "horse_name": "アスクビクターモア",
      "horse_id": "2019104706",
      "sex_age": "牡4",
      "jockey_weight": 58.0,
      "jockey_name": "横山武史",
      "jockey_id": "01170",
      "finish_time": "2:12.3",
      "time_sec": 132.3,
      "margin": "クビ",
      "passing_order": "4-5-4-3",
      "last_3f": 36.5,
      "odds": 14.3,
      "popularity": 4,
      "weight": 480,
      "weight_change": -2,
      "trainer_name": "田村康仁",
      "trainer_id": "01027"
    },
    {
      "finish_position": 12,
      "bracket_number": 7,
      "horse_number": 14,
      "horse_name": "ブレークアップ",
      "horse_id": "2018106273",
      "sex_age": "牡5",
      "jockey_weight": 58.0,
      "jockey_name": "川田将雅",
      "jockey_id": "01088",
      "finish_time": "2:12.3",
      "time_sec": 132.3,
      "margin": "クビ",
      "passing_order": "3-3-3-3",
      "last_3f": 36.5,
      "odds": 113.4,
      "popularity": 12,
      "weight": 494,
      "weight_change": -2,
      "trainer_name": "吉岡辰弥",
      "trainer_id": "01176"
    },
    {
      "finish_position": 13,
      "bracket_number": 2,
      "horse_number": 3,
      "horse_name": "ダノンザキッド",
      "horse_id": "2018104963",
      "sex_age": "牡5",
      "jockey_weight": 58.0,
      "jockey_name": "北村友一",
      "jockey_id": "01102",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "1.1/2",
      "passing_order": "4-5-6-9",
      "last_3f": 36.5,
      "odds": 35.1,
      "popularity": 8,
      "weight": 532,
      "weight_change": -2,
      "trainer_name": "安田隆行",
      "trainer_id": "00438"
    },
    {
      "finish_position": 14,
      "bracket_number": 8,
      "horse_number": 16,
      "horse_name": "モズベッロ",
      "horse_id": "2016100915",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "角田大河",
      "jockey_id": "01199",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "アタマ",
      "passing_order": "10-10-13-12",
      "last_3f": 36.2,
      "odds": 480.4,
      "popularity": 17,
      "weight": 500,
      "weight_change": 4,
      "trainer_name": "森田直行",
      "trainer_id": "01142"
    },
    {
      "finish_position": 15,
      "bracket_number": 8,
      "horse_number": 15,
      "horse_name": "ユニコーンライオン",
      "horse_id": "2016110103",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "坂井瑠星",
      "jockey_id": "01163",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "ハナ",
      "passing_order": "1-1-1-1",
      "last_3f": 36.9,
      "odds": 176.0,
      "popularity": 14,
      "weight": 520,
      "weight_change": 0,
      "trainer_name": "矢作芳人",
      "trainer_id": "01075"
    },
    {
      "finish_position": 16,
      "bracket_number": 1,
      "horse_number": 2,
      "horse_name": "カラテ",
      "horse_id": "2016106606",
      "sex_age": "牡7",
      "jockey_weight": 58.0,
      "jockey_name": "菅原明良",
      "jockey_id": "01179",
      "finish_time": "2:12.6",
      "time_sec": 132.6,
      "margin": "ハナ",
      "passing_order": "4-3-4-6",
      "last_3f": 36.5,
      "odds": 180.2,
      "popularity": 15,
      "weight": 538,
      "weight_change": 0,
      "trainer_name": "辻野泰之",
      "trainer_id": "01183"
    },
    {
      "finish_position": 17,
      "bracket_number": 1,
      "horse_number": 1,
      "horse_name": "ライラック",
      "horse_id": "2019103588",
      "sex_age": "牝4",
      "jockey_weight": 56.0,
      "jockey_name": "Ｍ．デム",
      "jockey_id": "05212",
      "finish_time": "2:12.7",
      "time_sec": 132.7,
      "margin": "1/2",
      "passing_order": "7-7-8-12",
      "last_3f": 36.4,
      "odds": 158.9,
      "popularity": 13,
      "weight": 430,
      "weight_change": -8,
      "trainer_name": "相沢郁",
      "trainer_id": "01020"
    }
  ],
  "field_size": 17,
  "payoff": {
    "単勝": {
      "numbers": "5",
      "payout": "130",
      "popularity": "1"
    },
    "複勝": [
      {
        "numbers": "5",
        "payout": "110",
        "popularity": "1"
      },
      {
        "numbers": "6",
        "payout": "560",
        "popularity": "10"
      },
      {
        "numbers": "9",
        "payout": "170",
        "popularity": "2"
      }
    ],
    "枠連": {
      "numbers": "3 - 3",
      "payout": "2,280",
      "popularity": "8"
    },
    "馬連": {
      "numbers": "5 - 6",
      "payout": "2,340",
      "popularity": "8"
    },
    "ワイド": [
      {
        "numbers": "5 - 6",
        "payout": "970",
        "popularity": "11"
      },
      {
        "numbers": "5 - 9",
        "payout": "240",
        "popularity": "1"
      },
      {
        "numbers": "6 - 9",
        "payout": "2,930",
        "popularity": "28"
      }
    ],
    "馬単": {
      "numbers": "5 → 6",
      "payout": "2,660",
      "popularity": "9"
    },
    "三連複": {
      "numbers": "5 - 6 - 9",
      "payout": "4,030",
      "popularity": "14"
    },
    "三連単": {
      "numbers": "5 → 6 → 9",
      "payout": "13,630",
      "popularity": "36"
    }
  },
  "lap_times": [
    12.4,
    10.5,
    11.1,
    12.6,
    12.3,
    12.4,
    12.5,
    11.9,
    11.7,
    12.0,
    11.8
  ],
  "pace": {
    "first_half_3f": 34.0,
    "second_half_3f": 35.5,
    "t1f": 12.4,
    "t3f": 34.0,
    "l1f": 11.8,
    "l3f": 35.5
  },
  "corner_passing": [
    {
      "corner": 0,
      "label": "馬場指数",
      "order_text": "-7 (?)"
    },
    {
      "corner": 1,
      "label": "1コーナー",
      "order_text": "(*15,17)14(2,3,12)(1,10,13)16,7(4,9)(8,11)5,6"
    },
    {
      "corner": 2,
      "label": "2コーナー",
      "order_text": "15,17(2,14)(3,12)(1,10,13)(4,7,16)9(8,11)5,6"
    },
    {
      "corner": 3,
      "label": "3コーナー",
      "order_text": "(*15,17)14(2,12)(3,11)(1,10,13)(7,9)(4,16,5)(8,6)"
    },
    {
      "corner": 4,
      "label": "4コーナー",
      "order_text": "(*15,17)(14,12,11)(2,10,13)(3,9,5)(1,7,16,6)(4,8)"
    }
  ],
  "_meta": {
    "result_schema_kind": "db_race_result",
    "scraped_at": 1781192608.785349,
    "scraped_at_jst": "2026-06-12 00:43:28",
    "schema_validation": {
      "schema_version": 2,
      "passed": true,
      "top_missing": [],
      "top_type_errors": [],
      "top_constraint_errors": [],
      "entry_count": 17,
      "entry_issues": {}
    },
    "scrape_validation_status": "ok"
  }
}
```

</details>


<details>
<summary><code>nk_db_race_info</code> レース情報DB（`race_result_meta`）</summary>

```json
{
  "race_id": "202309030811",
  "race_name": "第64回宝塚記念",
  "grade": "G1",
  "surface": "芝",
  "direction": "右",
  "distance": 2200,
  "start_time": "15:40",
  "date": "2023-06-25",
  "venue": "阪神",
  "round": 0,
  "field_size": 17
}
```

</details>


<details>
<summary><code>nk_db_payoff</code> 払戻DB（`race_result_payoff`）</summary>

```json
{
  "race_id": "202309030811",
  "payoff": {
    "単勝": {
      "numbers": "5",
      "payout": "130",
      "popularity": "1"
    },
    "複勝": [
      {
        "numbers": "5",
        "payout": "110",
        "popularity": "1"
      },
      {
        "numbers": "6",
        "payout": "560",
        "popularity": "10"
      },
      {
        "numbers": "9",
        "payout": "170",
        "popularity": "2"
      }
    ],
    "枠連": {
      "numbers": "3 - 3",
      "payout": "2,280",
      "popularity": "8"
    },
    "馬連": {
      "numbers": "5 - 6",
      "payout": "2,340",
      "popularity": "8"
    },
    "ワイド": [
      {
        "numbers": "5 - 6",
        "payout": "970",
        "popularity": "11"
      },
      {
        "numbers": "5 - 9",
        "payout": "240",
        "popularity": "1"
      },
      {
        "numbers": "6 - 9",
        "payout": "2,930",
        "popularity": "28"
      }
    ],
    "馬単": {
      "numbers": "5 → 6",
      "payout": "2,660",
      "popularity": "9"
    },
    "三連複": {
      "numbers": "5 - 6 - 9",
      "payout": "4,030",
      "popularity": "14"
    },
    "三連単": {
      "numbers": "5 → 6 → 9",
      "payout": "13,630",
      "popularity": "36"
    }
  }
}
```

</details>


<details>
<summary><code>nk_db_track</code> 馬場情報DB（`race_result_track`）</summary>

```json
{
  "race_id": "202309030811",
  "weather": "曇",
  "track_condition": "良"
}
```

</details>


<details>
<summary><code>nk_db_corner</code> 通過順位DB（`race_result_corner`）</summary>

```json
{
  "race_id": "202309030811",
  "corner_passing": [
    {
      "corner": 0,
      "label": "馬場指数",
      "order_text": "-7 (?)"
    },
    {
      "corner": 1,
      "label": "1コーナー",
      "order_text": "(*15,17)14(2,3,12)(1,10,13)16,7(4,9)(8,11)5,6"
    },
    {
      "corner": 2,
      "label": "2コーナー",
      "order_text": "15,17(2,14)(3,12)(1,10,13)(4,7,16)9(8,11)5,6"
    },
    {
      "corner": 3,
      "label": "3コーナー",
      "order_text": "(*15,17)14(2,12)(3,11)(1,10,13)(7,9)(4,16,5)(8,6)"
    },
    {
      "corner": 4,
      "label": "4コーナー",
      "order_text": "(*15,17)(14,12,11)(2,10,13)(3,9,5)(1,7,16,6)(4,8)"
    }
  ]
}
```

</details>


<details>
<summary><code>nk_db_lap</code> ラップDB（`race_result_lap_times`）</summary>

```json
{
  "race_id": "202309030811",
  "lap_times": [
    12.4,
    10.5,
    11.1,
    12.6,
    12.3,
    12.4,
    12.5,
    11.9,
    11.7,
    12.0,
    11.8
  ],
  "pace": {
    "first_half_3f": 34.0,
    "second_half_3f": 35.5,
    "t1f": 12.4,
    "t3f": 34.0,
    "l1f": 11.8,
    "l3f": 35.5
  }
}
```

</details>


<details>
<summary><code>nk_db_per_horse_lap</code> 個別ラップDB（`race_result_lap`）</summary>

```json
{
  "race_id": "202309030811",
  "lap_times": [
    12.4,
    10.5,
    11.1,
    12.6,
    12.3,
    12.4,
    12.5,
    11.9,
    11.7,
    12.0,
    11.8
  ],
  "pace": {
    "first_half_3f": 34.0,
    "second_half_3f": 35.5,
    "t1f": 12.4,
    "t3f": 34.0,
    "l1f": 11.8,
    "l3f": 35.5
  },
  "corner_passing": [
    {
      "corner": 0,
      "label": "馬場指数",
      "order_text": "-7 (?)"
    },
    {
      "corner": 1,
      "label": "1コーナー",
      "order_text": "(*15,17)14(2,3,12)(1,10,13)16,7(4,9)(8,11)5,6"
    },
    {
      "corner": 2,
      "label": "2コーナー",
      "order_text": "15,17(2,14)(3,12)(1,10,13)(4,7,16)9(8,11)5,6"
    },
    {
      "corner": 3,
      "label": "3コーナー",
      "order_text": "(*15,17)14(2,12)(3,11)(1,10,13)(7,9)(4,16,5)(8,6)"
    },
    {
      "corner": 4,
      "label": "4コーナー",
      "order_text": "(*15,17)(14,12,11)(2,10,13)(3,9,5)(1,7,16,6)(4,8)"
    }
  ],
  "entries_lap": [
    {
      "horse_number": 5,
      "horse_id": "2019105219",
      "horse_name": "イクイノックス",
      "passing_order": "16-16-13-9",
      "last_3f": 34.8
    },
    {
      "horse_number": 6,
      "horse_id": "2018105269",
      "horse_name": "スルーセブンシーズ",
      "passing_order": "17-17-16-12",
      "last_3f": 34.6
    },
    {
      "horse_number": 9,
      "horse_id": "2019105346",
      "horse_name": "ジャスティンパレス",
      "passing_order": "12-13-11-9",
      "last_3f": 35.1
    },
    {
      "horse_number": 11,
      "horse_id": "2018105081",
      "horse_name": "ジェラルディーナ",
      "passing_order": "14-14-6-3",
      "last_3f": 35.5
    },
    {
      "horse_number": 10,
      "horse_id": "2017102170",
      "horse_name": "ディープボンド",
      "passing_order": "7-7-8-6",
      "last_3f": 35.5
    },
    {
      "horse_number": 7,
      "horse_id": "2019100109",
      "horse_name": "プラダリア",
      "passing_order": "11-10-11-12",
      "last_3f": 35.3
    },
    {
      "horse_number": 4,
      "horse_id": "2016104618",
      "horse_name": "ボッケリーニ",
      "passing_order": "12-10-13-16",
      "last_3f": 35.1
    },
    {
      "horse_number": 8,
      "horse_id": "2017105082",
      "horse_name": "ヴェラアズール",
      "passing_order": "14-14-16-16",
      "last_3f": 35.2
    },
    {
      "horse_number": 13,
      "horse_id": "2019105056",
      "horse_name": "ジオグリフ",
      "passing_order": "7-7-8-6",
      "last_3f": 35.8
    },
    {
      "horse_number": 17,
      "horse_id": "2020103626",
      "horse_name": "ドゥラエレーデ",
      "passing_order": "2-2-2-2",
      "last_3f": 36.5
    },
    {
      "horse_number": 12,
      "horse_id": "2019104706",
      "horse_name": "アスクビクターモア",
      "passing_order": "4-5-4-3",
      "last_3f": 36.5
    },
    {
      "horse_number": 14,
      "horse_id": "2018106273",
      "horse_name": "ブレークアップ",
      "passing_order": "3-3-3-3",
      "last_3f": 36.5
    },
    {
      "horse_number": 3,
      "horse_id": "2018104963",
      "horse_name": "ダノンザキッド",
      "passing_order": "4-5-6-9",
      "last_3f": 36.5
    },
    {
      "horse_number": 16,
      "horse_id": "2016100915",
      "horse_name": "モズベッロ",
      "passing_order": "10-10-13-12",
      "last_3f": 36.2
    },
    {
      "horse_number": 15,
      "horse_id": "2016110103",
      "horse_name": "ユニコーンライオン",
      "passing_order": "1-1-1-1",
      "last_3f": 36.9
    },
    {
      "horse_number": 2,
      "horse_id": "2016106606",
      "horse_name": "カラテ",
      "passing_order": "4-3-4-6",
      "last_3f": 36.5
    },
    {
      "horse_number": 1,
      "horse_id": "2019103588",
      "horse_name": "ライラック",
      "passing_order": "7-7-8-12",
      "last_3f": 36.4
    }
  ],
  "_meta": {
    "scraped_at": 1781190781.7651346,
    "scraped_at_jst": "2026-06-12 00:13:01"
  }
}
```

</details>


<!-- SCRAPE_PROCESS_SAMPLES_AUTO_END -->
