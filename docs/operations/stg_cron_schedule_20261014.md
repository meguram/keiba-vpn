# stg（ホスト PC）のスクレイピング cron: 時間割と取得データの網羅状況（2026-10-14）

ホスト PC の root の crontab（UTC。以下は JST に換算）。定義の正は `scripts/cron/setup_all_cron.sh`（`install` で再登録）。グループ別の一覧は `docs/html/operations/cron_schedule.html`（`python -m src.scripts.docs.build_cron_schedule_html` で再生成）。バックアップは `data/local/meta/cron_backups/`。
cron デーモンは稼働中（2026-10-13 23:44 UTC 起動）。ホストの再起動で止まった時間帯のジョブは実行されない点に注意（10/14 07:30 JST の runner は欠落、非開催日のため影響なし）。

## 1. 時間割（JST）

| 時刻 | ジョブ | 内容 |
|---|---|---|
| 毎日 00:00 | PG 同期、backfill 2026（fast） | GCS→PostgreSQL 増分、直近 7 開催日の結果・出馬表 |
| 毎日 01:00 / 02:00 / 03:00 / 04:00 | backfill 2025 / 2024 / 2023 / 2022（fast）、火金 02:00 に 2021、水土 03:00 に 2020 | 過去年の取りこぼし補填（日付ごとに 1 回で「完了」扱い） |
| 毎日 04:30 | ログローテーション | |
| 毎日 05:00〜08:50（10 分ごと） | jra-baba-morning | JRA 馬場（クッション値・含水率）。開催日のみ |
| 毎日 05:30 | 騎手・調教師統計 | |
| 毎日 06:00 | backfill horse | 馬情報一括 |
| 毎日 06:10 **（新）** | recent-retry | 直近 21 日の未取得データの取り直し |
| 毎日 07:00 | daily-race-lists（朝） | 今日以降の全開催日のレース一覧 |
| 毎日 07:00 **（新）** | export_tables_daily | 当年の flat テーブルを増分更新（推論の履歴。`--refresh` 付き。実測 約 4〜5 分） |
| 毎日 07:40 **（新）** | raceday-comment（朝） | 開催日なら当日の厩舎コメントを取り直す（20:00 で取れなかった／増えた分） |
| 毎日 07:40 **（新）** | feature-accumulate | 当年分の履歴特徴量（127 列）を確定結果から作り直して蓄積し、推論モードとのずれを記録。最後にサービング用バンドルを GCS へ公開（実測 約 30 秒 + 公開） |
| 毎日 07:30 | raceday-runner、raceday-result-runner | 開催日のみ常駐。各R T-60 オッズ、T-15 バンドル、T+15 速報結果 |
| 毎日 07:50 / 08:00 / 09:00 | backfill full 2026 / 2025 / 2024 | 補助データ（速報結果など） |
| 毎日 08:00 **（新）** | raceday-weight-runner | 各R 発走 65〜30 分前に出馬表を取り直し、馬体重が揃ったら推論モードで特徴量を作成（開催日のみ） |
| 毎日 08:00 | ページ品質チェック | |
| 毎日 08:30 | barometer-retry | 公開済み・未取得のバロメーター |
| 毎日 17:00 | daily-race-lists（夕） | |
| 毎日 17:30 | raceday-evening | 速報結果・確定オッズ（開催日のみ） |
| 毎日 18:00 | raceday-eve | 翌日が開催日なら出馬表・馬柱・追い切り・SmartRC |
| 毎日 19:40 **（新）** | recent-retry | 同上（2 回目） |
| 毎日 20:00 **（新）** | raceday-comment（夜） | 翌日が開催日なら厩舎コメントを取得（前日 16 時頃に公開される想定。コメントが出ているレースだけ保存） |
| 毎日 21:00 **（新）** | new-horse-pedigree | 今後 1 週間の出走予定馬のうち血統テーブル（ped_tbl）が無い馬だけ 5 世代血統を取得 |
| 常時（起動時 + 5 分ごと） **（新）** | ensure_postgres | ローカル PostgreSQL が止まっていれば起動 |
| 金 18:00 / 18:30 / 19:30 | weekly-update / horse-name-index / barometer-retry | 直近 10 日の確定結果・指数・バロメーター、馬名インデックス |
| 常時 | watchdog（3 分ごと）、リモートアクセス（2 分ごと） | |

**ガード（新）**: daily-race-lists・weekly-update・barometer-retry・recent-retry・backfill の全行は `scripts/cron/yield_to_rescrape.sh` 経由。優先処理（`run_priority_rescrape.sh`）が動いている間は見送り（`logs/yield_skips.log` に記録）、終われば自動で通常どおり動く。netkeiba は同一アカウントで別セッションがログインすると取り直し側のセッションが無効になるため。止めたいときは `data/local/meta/rescrape/YIELD_DISABLED` を置く。raceday 系・JRA 馬場・export は見送らない。

## 2. 取得データの扱い（加工の有無と保存先）

netkeiba を取得する全ジョブは共通の流れで保存する。

1. **生の HTML は原則保存しない。** ページ構造の変更検出用に、カテゴリごと最大 10 件だけ GCS に gzip で置く。ローカルには残さない。
2. **構造化**: パーサーが HTML から項目を取り出して JSON にする（値は画面のまま。集計・派生量は作らない）。取得時刻とスキーマ検証の結果（`_meta`）を付ける。
3. **検証**: スキーマ不適合は GCS に保存せず、`data/local/quarantine/` に隔離し、違反を `data/local/meta/schema_violations/` に記録する（既定）。
4. **保存先（正本）**: GCS `chuou/data/preprocessed/netkeiba/pc/<カテゴリ>/<年 または馬IDの先頭4桁>/<ID>.json`。ローカルは補助のキャッシュ（メモリ、週 2 回以上読まれたキーだけの `data/cache/`）。GCS 同形のミラー（`data/local/mirror/`）は設定で無効（ホストは未設定）。
5. **ローカルのみのデータ**: レース一覧（`data/page_reference/race_lists`）、発走時刻表（`data/page_reference/race_day_schedule`）、JRA 馬場（`data/page_reference/cushion`。GCS へは手動アップロード）。
6. **加工して別の場所に置くもの**: 履歴テーブル（`data/page_reference/tables`）、騎手・調教師統計（`data/features`）、馬場速度指数（`data/analysis/track_speed`）、推論用の特徴量（`data/local/inference`）、DB への取り込み（PostgreSQL）。いずれも GCS の構造化 JSON を元に作る。

ジョブ別:

| ジョブ | データの加工 | 保存先 |
|---|---|---|
| 前日夕方の出馬表・馬柱・追い切り | 構造化 + 推定（追走難度・想定オッズ。モデルによる計算結果を保存） | GCS: 出馬表・馬柱・追い切り・SmartRC は構造化JSON、追走難度・想定オッズは推定結果のキャッシュ。ローカル: 取得状況 data/local/meta/date_coverage、発走時刻表 data/page_reference/race_day_schedule（ローカルのみ）。DB 同期を起動（ホストでは失敗中） |
| 開催日ランナー（オッズ・出馬表） | 構造化のみ（T-15 の予測トリガはモックで保存なし） | GCS: 出馬表・オッズ（T-60/T-15 の履歴を同じJSONに保持）・馬柱・追い切り・SmartRC。JRA 馬場はローカル data/page_reference/cushion |
| 速報結果ランナー | 構造化のみ。全R後に結果テーブルの増分追記と馬場速度指数の計算 | GCS: 速報結果。ローカル: data/page_reference/tables（race_result_flat の増分）、data/analysis/track_speed（馬場速度指数） |
| 馬体重の検知ランナー | 構造化（出馬表）+ 計算（127特徴量） | GCS: 出馬表。ローカルのみ: data/local/inference（特徴量 parquet・予測時点の出馬表スナップショット・イベント記録。GCS には上げない） |
| 開催日夕方の結果・確定オッズ | 構造化のみ | GCS: 速報結果・確定オッズ・SmartRC。ローカル: 取得状況 date_coverage。DB 同期を起動（ホストでは失敗中） |
| JRA 馬場（クッション値・含水率） | 構造化（JRA 公式ページを整形したJSON） | ローカルのみ: data/page_reference/cushion。GCS へは自動では上げない（手動アップロード） |
| レース一覧 | 構造化のみ | ローカルのみ: data/page_reference/race_lists。キューにレース単位のジョブが溜まっていれば、その分は GCS へ |
| 週次更新 | 構造化 + 計算（めぐ指数）+ DB 取り込み | GCS: 確定結果・指数・バロメーター・馬情報。PostgreSQL: ETL（ホストでは DB 未起動・psycopg2 未導入のため失敗中） |
| 馬名インデックス | 計算（馬名リスト・成長曲線の集計） | ローカルのみ: data/calculated_data |
| バロメーターの取り直し | 構造化のみ | GCS: race_barometer |
| 直近の未取得データの取り直し | 構造化のみ | GCS: race_result・race_result_lap・race_index・race_trainer_comment。ローカル: 試行状態 data/local/meta/recent_retry_state.json |
| backfill（年度別・馬情報） | 構造化のみ | GCS: 同じカテゴリ（馬情報は horse_result）。ローカル: 進捗台帳 data/local/meta/backfill_progress.json、取得状況 date_coverage |
| 履歴テーブルの増分更新 | 加工（JSON を表に平坦化。1行=1馬の _flat と 1行=1レースの _race の Parquet） | ローカルのみ: data/page_reference/tables/<年>/ |
| 履歴特徴量の蓄積 | 計算（as-of の履歴特徴量。学習と同じ計算） | ローカルのみ: data/local/modeling/feature_store/horse_history（features_<年>.parquet、manifest、skew/<日付>.json）と、小さなメモリのマシン向けのサービング用バンドル serving_bundle（約140MB）。2026年は学習に使わない |
| 騎手・調教師統計 | 計算（成績の集計） | ローカルのみ: data/features/jockey_trainer_stats ほか（Parquet） |
| GCS → PostgreSQL 同期 | 加工（ETL で正規化テーブルへ） | PostgreSQL（ホストでは毎晩失敗: DB 未起動・psycopg2 未導入） |
| ページ品質チェック | 検査のみ | ローカル: data/local/page_quality/latest.json（過去30件は history/） |

注: DB（PostgreSQL）への取り込みは、ホストでは DB が起動しておらず psycopg2 も未導入のため、現状は毎晩失敗する（取得と GCS への保存には影響しない）。グループ別の一覧は `docs/html/operations/cron_schedule.html`。

## 3. 取得したい生データの網羅状況

「直近の開催日の取得済み / 全レース」は 2026-10-14 の実測（GCS）。

| 生データ | 用途 | 担当ジョブ | 10/3・4 | 10/10 | 10/11 | 10/12 | 評価 |
|---|---|---|---|---|---|---|---|
| 出馬表 `race_shutuba` | 前日の tier 0、馬体重（tier 2） | eve、T-15、馬体重検知（新） | 24/24 | **18/24** | 24/24 | 24/24 | 10/10 の 6R 欠け（未調査） |
| 馬柱・追い切り・オッズ・馬連系 | 特徴量・期待値 | eve、T-60/T-15、evening | 揃う | 揃う | 揃う | 揃う | ○ |
| 速報結果 `race_result_on_time` | 履歴の補修・同日の馬場差 | result-runner、evening | 揃う | 揃う | 揃う | 揃う | ○ |
| **確定結果 `race_result`** | 履歴の主材料（着順・時計・上り・通過順） | weekly（金）、backfill、**recent-retry（新）** | 24/24 | **0/24** | **0/24** | **0/24** | **穴だった。recent-retry で解消** |
| 結果ラップ `race_result_lap` | 馬別ラップ（2024-01 以降のみ存在） | **recent-retry（新）** | **0** | 0 | 0 | 0 | 穴だった（定期経路なし）。recent-retry で解消 |
| タイム指数 `race_index` | 母表の has_speed_index | weekly、recent-retry、取り直し（2020〜2025） | 24/24 | 24/24 | 24/24 | **0/24** | recent-retry で解消 |
| 厩舎コメント | 追加特徴量候補 | recent-retry（新） | 24/24 | 0/24 | 0/24 | 0/24 | 穴だった。recent-retry で解消 |
| パドック | 追加特徴量候補（上位 5 頭のみ） | backfill full、予測パイプライン | 24/24 | 24/24 | 24/24 | 0/24 | △ 定期経路が backfill のみ |
| バロメーター | 追加特徴量候補 | weekly、barometer-retry | 23/24 | 公開前 | 公開前 | 公開前 | ○（翌週金曜に公開） |
| 馬ページ・血統 | 馬の属性 | weekly、backfill horse、手動（5 世代） | – | – | – | – | △ 5 世代血統は手動 |
| JRA 馬場 | tier 1 | jra-baba-morning | – | – | – | – | ○ |

### 見つかった問題と対応

1. **最新週末の確定結果が取れていなかった（重大）**: backfill は 10/12 0:00 JST に 10/11 分を取りに行ったが、db.netkeiba にまだ結果が無く（`0 entries`）、それでも台帳に「完了」と記録したため、以降一度も再試行されなかった。金曜の weekly-update まで確定結果が入らず、土曜の推論の履歴が最大 5 日古くなる。→ `recent-retry`（新）が、直近 21 日の未取得のレース × カテゴリだけを 1 日 2 回、間隔 3 時間・最大 30 回まで試す。初回の対象は 336 件（確定結果 72、結果ラップ 168、タイム指数 24、厩舎コメント 72）。
2. **取り直しとの衝突**: daily-race-lists が、キューに溜まった 2026 年分の日付ジョブをすべて処理しており（`_kick_and_wait` が最長 48 時間キュー全体の完了を待つ）、4 時間以上走ってログインを繰り返していた。→ 上記のガードで、優先処理の実行中は見送る。**既に走っていた 1 回（17:00 JST 起動）は止めていない。**
3. 推論用の履歴テーブル（`race_result_flat` の 2026 年分）が更新されていなかった（最終日 2026-10-04）。→ `export_tables_daily`（新）を 07:00 JST に追加。
4. 馬体重の検知ランナー（新）を 08:00 JST に追加。mode は `features`（特徴量の作成まで。予測は起動しない）。

5. **DB（PostgreSQL）への取り込みがホストで毎晩失敗していた（2026-10-15 に解消）**: 原因は「psycopg2 未導入」ではなく、**コンテナの再起動後に PostgreSQL 14 のクラスタ（`14/main`、DB `keiba_db_stg` 431MB）が停止したままだった**こと（接続拒否）。psycopg（v3）は導入済み。クラスタを起動して手動で同期を実行し、成功した（直近 10 日・120 レース、エラー 0、`exit=0`）。再発防止に `scripts/server/ensure_postgres.sh`（止まっていれば起動。冪等）を cron に追加した（起動時 + 5 分おき、タグ `KEIBA_PG_ENSURE`、ログ `logs/ensure_postgres.log`）。
6. **10/09〜10/10 に cron が `KEIBA_ENV=dev` で動き、取得データの一部が GCS ではなく `data/dev_mock` に入っていた（2026-10-15 に復旧済み）**: 原因は環境設定の解決（dev になると保存先が dev ストアになる）。→ 実行ラッパー `run_auto_scrape_logged.sh` は dev に解決された場合、取得せず終了する（`KEIBA_ALLOW_DEV_CRON=1` で解除）。dev ストアに残っていた実データのうち GCS に無い 61 件は、`python -m src.scripts.maintenance.restore_dev_store_to_gcs --since 2026-10-09 --apply` で**作成のみ**（既存は上書きしない）で追加し、再実行で残り 0 件を確認した。以前の「速報結果 180 件が無い」は、キー照合の誤りによる誤検知だった。
7. **履歴テーブルが 10/04 で止まっていた（2026-10-15 に修正）**: `export_tables_daily` は「自分の `race_result_flat` に載っているレース」を基準レース集合にしていたため、**`race_result` に新しいレースが永久に入らなかった**（flat の最終日 2026-10-04、10/10〜12 の 72 レースが未反映）。→ `--refresh`（各カテゴリが自分の flat に無いレースを GCS から追記する）を追加して日次ジョブで使う。手動実行で 72 レース・980 行が追記され、最終日は 2026-10-12 になった。テスト: `tests/scraper/test_export_refresh.py`。
8. **`feature-accumulate` が今朝 07:40 に失敗していた**: 距離の型（Float64/Int64）の不一致（`read_runs_raw` に型をそろえる修正が入る前のコードで動いた）。修正済みのコードで手動再実行する。
9. **厩舎コメント・新馬の血統・サービングバンドル配布の定期処理を追加（2026-10-15、ユーザー承認）**: `raceday-comment`（20:00・07:40）、`new-horse-pedigree`（21:00）、`feature-accumulate` の最後のバンドル公開（`chuou/models/serving_bundle/<版>/` と `LATEST.json`。3 版を残す。小さなマシンは `python -m src.pipeline.features.horse_history.bundle_sync fetch` で取得）。cushion の GCS 2026 年分は統合して更新済み（旧版は `data/local/backup/jra_cushion/`）。
10. **結果ページの表記変更でパースが黙って失敗していた（2026-10-15 に修正、ユーザー承認で取り直し）**: 2026 年に db.netkeiba の結果ページのヘッダが「芝右 外1200m」と内外回りを含む形になり、旧パーサの正規表現（方向の直後に距離）が合わず、**距離 0・面が空の結果が 260 レース（2026 年の約 10%）保存されていた**。履歴特徴量の入力検証（`check_runs`）が止めたため、`feature-accumulate` が失敗し、推論バンドルの更新も止まっていた。→ (1) コース表記の読み取りを `parsers.parse_course` に共通化（「芝右1600m」「芝直1000m」「芝直線」「芝右 外1200m」「内2周」「外・内」「障芝 右 …」など。結果・速報結果・出馬表の 3 パーサで共用、`tests/scraper/test_parse_course.py` と `test_result_header_parse.py`）。(2) コース表記を読めなかった結果ページは**保存しない**ガード（`run._race_result_header_unreadable`）。(3) 保存済みの 260 件は `repair_race_result_header` で取り直し、面・方向・距離だけ書き換え（旧版は `data/local/backup/race_result_header/`）。(4) 日次エクスポートの最後に `parse_audit`（必須項目の欠損検査）を実行し、問題があれば `logs/export_tables_daily.log` に `PARSE_AUDIT_FAIL` を出す。(5) `round`（レース番号）が古い結果データで 0 のままだったため、エクスポートで race_id の末尾 2 桁から補う。新しい表記が出たら `parse_audit` が翌日までに検知する。
11. **2026 年の出馬表 1,674 件でヘッダが文字化け・空（修正の影響を回避済み、取り直しはユーザー判断待ち）**: クライアントの文字コード判定の修正（2026-10-13 20:45 JST）より前に取得した出馬表は、レース名・クラスが文字化けまたは空で保存されている（`race_class` が空の平地レース 1,520 件 + 日付・面・頭数も空の 116 件。馬の行は読めている）。履歴特徴量は `race_class` の空欄を障害レースの印にしていたため、これらの平地レースが「障」に誤分類されていた（バンドルの 2026 年分）。→ 障害の判定を「`race_class` が空 **かつ** 方向が空」に変更（2020〜2023 年は従来と完全一致、2024・2025 年は各 1 レースが平地に直る。2026 年は 1,540 レースが直る）。`parse_audit` は「`race_class` だけが空で方向がある」レースを兆候として検出する。バンドルは修正後に作り直して公開した（履歴は 2026-10-12 まで）。出馬表そのものの取り直し（1,674 件、netkeiba へ約 1.5 時間）は GCS の上書きになるため、承認後に実施する。
12. **2024 年の 1 レース（202404030612）の結果の面・距離が空**（同じ表記の読み取り不良。修復は同じスクリプトで 1 件、承認待ち）。

### 未解決・要確認

- 10/10 の出馬表 6R 欠け、10/12 のパドック未取得の原因。
- db.netkeiba の確定結果が何時ごろから取れるか（recent-retry の 06:10 / 19:40 で取れた時刻が `data/local/meta/recent_retry_state.json` と `logs/recent_retry.log` に残る）。取れる時刻が分かれば、1 日 2 回の時刻を詰める。
- キューに残る 2026 年分の日付ジョブ（race_list 24・date_all 5）が優先処理の後に一気に流れる。10/16 18:00 JST の raceday-eve（翌日の出馬表・馬柱・追い切り）がキューの後ろに並ばないよう、10/16 昼までにキューが空になっているかを確認する。
- 優先処理が終わった後の 2026 年分の取得（`scrape_from_health_plan`）が、ログインを使うジョブとまた重ならないか。
