# keiba-vpn — エージェント向けメモ

競馬データ基盤（スクレイピング・特徴量・ML・Web）。README の詳細は `README.html`。

## レイアウト

すべての Python コードは **`src/`** に集約され、役割ごとにサブパッケージで分割されている。シェルスクリプトとデータ・テスト・ノートブックはトップレベルに残す。

| 領域 | 役割 |
|------|------|
| `src/api/` | FastAPI（`app.py`）。エントリは `python main.py` → `src.api.app:app` |
| `src/monitor/` | **開発者専用監視ポータル**（スタンドアロン Flask。ポートは `config/ports.env`）。`app.py` + `templates/`。起動: `bash scripts/server/start_monitor.sh` / `service_start.sh --env dev --monitor`。仕様書: `docs/monitoring-portal.md` |
| `src/scraper/` | netkeiba / SmartRC / JRA 取得・パース・キュー（CLI は `python -m src.scraper.X`） |
| `src/pipeline/` | 学習・特徴量。`features/`（store/layout/builder/stats）・`models/`（学習・予測子）・`inference/`（race_day/betting/composite_optimizer）に分割。`build_*.py` などの CLI はルート（`python -m src.pipeline.build_X`） |
| `src/research/` | リサーチ層。`pedigree/`（血統・種牡馬）、`genes/`（遺伝子マーカー）、`race/`（コース・トラック・レース品質）、`scripts/`（キュー投入等の CLI）に分類 |
| `src/scripts/` | 運用 Python CLI。`scraping/`・`data/`・`docs/`（HTML 埋め込み等）・`maintenance/` にサブ分類（`python -m src.scripts.<role>.<file>`） |
| `src/utils/` | 横断ユーティリティ（ロギング・パス解決） |
| `templates/` `static/` | Jinja2 / CSS（`src/api/app.py` から `BASE_DIR` 経由でマウント） |
| `notebooks/` | Jupyter ノートブック専用。`pedigree/`・`feature_engineering/`（伴走スクリプト・出力含む）・`modeling/` |
| `data/` | 生データ・メタ・特徴量（大容量。不要な一括編集は避ける） |
| `data/page_reference/` | **UI ポータブルバンドル**（`race_lists`・血統アーティファクト・`meta/person` 等）。別 PC へはこのディレクトリをコピー（`data/page_reference/BUNDLE.md`） |
| `data/local/` | 運用・パイプライン（血統シャード・キュー meta・features 等。page 用は `page_reference` へ移行済み） |
| `data/meta/structure/` | 各 JSON の構造メタ |
| `data/local/features/` | 特徴量ストア（旧表記 `data/features/` は存在しない）。ブロック例: `base_tbl`（4キーのみの ``shutuba``）・`race_tbl`・`race_horse_tbl`・`horse_tbl`（血統・馬プロファイル等）・`race_jockey_tbl` / `race_trainer_tbl`・`jockey_tbl` / `trainer_tbl`・`jockey_trainer_stats/`（メタ）。**ラベル**: `target/rank_tbl/<年>/rank.parquet`（`race_id`,`horse_id`,`rank`）は `python -m src.pipeline.build_rank_target`。**馬単位エンティティ**（`horse/ped_tbl|result_tbl|training_tbl/{馬ID先頭4桁}/{horse_id}.parquet`、年ではなく馬シャード）は `python -m src.pipeline.build_horse_entity_store`。`docs/html/data/data_features_reference.html` の `#join-architecture`。出馬表 raw は `python -m src.pipeline.register_raw_table_features` → `_raw_table_feature_selection.json`。騎手・調教師は `python -m src.pipeline.build_jockey_trainer_stats`（定期: `scripts/cron/update_jockey_trainer_stats.sh` 等）。 |
| `docs/html/` | システム文書。**全 HTML（`docs/html/` の各ページ・`docs/decisions/`・`docs/operations/`・`docs/other_predictor/`）を 1 冊にした `docs/html/book.html`（部・章・節の目次、章間リンク、専門用語の解説つき）から読む**。各ページの HTML（子ファイル）は `design/`・`data/`・`operations/`・`pedigree/`・`modeling/chapters/` などに置き（`modeling/` 直下に HTML は置かない）、元の HTML を直したら `python -m src.scripts.docs.build_docs_book` で本を作り直す。`docs/` に HTML を足したら `build_docs_book.py` の `PARTS` にも載せる（`tests/scripts/test_build_docs_book.py` が載せ忘れ・リンク切れ・id 重複を検出）。古くなった設計の章には `PARTS` の `status`/`note`/`now` で『旧設計：異なる点と現行の章』を付ける。本文の専門用語は自動で押せる形（クリックで解説が開く）になり、解説と図は `src/scripts/docs/modeling_glossary.py` の `TERMS` / `DIAGRAMS` で定義して巻末「用語集」にも載る（用語を足すときはここだけ編集）。**同じ事実（ポート・カテゴリ数・cron・精度）を複数ページに手書きしない**: ポートは `config/ports.env`、カテゴリは `schema_defs.json`、確定事項は `training_pipeline_decisions` が正で、他はそれを参照する。要件定義書 `docs/html/index.html` は `/update-requirements` が再生成する |
| `scripts/` | シェルスクリプトのみ。`server/`（API 起動）・`cron/`（定期実行）。Python の CLI は `src/scripts/` を参照 |
| `docs/operations/` | 運用メモ。**ディレクトリ構成**: `PROJECT_LAYOUT.md`。**起動 URL・ポート**: `service-endpoints.md` |
| `mlflow/` | MLflow 関連。`data/`（Docker サーバの DB + artifacts、`.gitignore`）・`runs/`（ローカルフォールバックのファイルストア、`.gitignore`）・`server/`（Docker Compose / Nginx / `setup.sh`）。**モデル一覧・追加手順**: `docs/html/design/mlflow_platform.html` / `src/pipeline/mlflow/catalog.py` |
| `config/` | `settings.yaml`（MLflow モデルキー等。`src/config/` はコード側プロファイル） |
| `tests/` | unittest（`api/` `pipeline/` `research/` `scraper/`） |
| `main.py` | サーバエントリ（`python main.py`。ポート省略時は `config/ports.env`） |
| `frontend/lib/features.ts` | **UI 機能フラグの唯一の正**。現在の UI のゴールは「AI予測表示 → 期待値 → 推奨馬券」の最小構成で、それ以外の画面は `enabled:false`（コード残置・ホーム/ナビ/直リンクで非表示、直リンクは `/` へリダイレクト）。一時有効化は `NEXT_PUBLIC_ENABLE_FEATURES=all` または `megu-index,growth-curve`。新画面を足したら必ずここに登録 |
| オッズの時点別運用 | `race_odds` / `race_pair_odds`（馬連・ワイド・馬単・**3連複**）はトップが最新で `phase`(t60/t15)・`snapshot_at`・履歴 `snapshots`（`src/scraper/odds_snapshot.py`）。開催日ランナー（キュー版）が各R **T-60 でオッズのみ、T-15 でバンドル**を取得（`auto_scrape_queue.build_odds_events`）。`KEIBA_ODDS_LATE_SNAPSHOTS=1` のときだけ、発走直前 **T-10 / T-5 / T-2 に単複 `race_odds` のみ**を追加で取って `snapshots` に積む（phase=t10/t05/t02。仕様・確認手順: `docs/operations/late_odds_snapshots.md`）。推奨は `odds_mode=auto/actual/predicted`（`src/pipeline/inference/odds_source.py`、予測オッズは単勝・複勝のみ）。**実データ取得シナリオ + 所要時間計測（T-X 判断用）**: `python3 tests/scraper/manual/live_odds_scenarios.py --date YYYYMMDD --timing --yes`（`--timing-only` / `--offline` / `--capture-dev`、結果は `data/local/meta/scrape_timing/`）。3連複 API(type=7)・3連単 API(type=8)の応答形式は実データで確認済み（キー "AABBCC"・値 [odds, "0.0", 人気]）。**3連単** は別カテゴリ `race_trifecta_odds`（`[1着,2着,3着,odds,人気]` の全組合せ。`KEIBA_ODDS_TRIFECTA=1` のときだけ cron の T-60/T-15/確定で取得、過去分は `python -m src.scripts.data.backfill_trifecta_odds`、想定オッズの較正は `src.scripts.data.trifecta_est_calibration`） |
| dev の表示データ | dev の UI は既定で **dev ストア（`data/dev_mock` + `data/page_reference`）をバックエンド経由で表示**（フロント内蔵モックに戻すなら `KEIBA_DEV_UI_MOCK=1`）。ストアの一部は実データ: `python3 -m src.scripts.scraping.capture_dev_day --date YYYYMMDD --yes`（実アクセスあり・36R で約30分・再開可。現在は 2026-09-06 の36R取り込み済み。AI予測/予測オッズはその実馬・実オッズに対するモック `_meta.mock_prediction`）。`make dev-mock` は `_meta.dev_mock` のモックだけを作り直し、実データは消さない。**別 PC でも同じ表示にする最小データは git 管理**: `dev_fixture/real_day.tar.gz`（実データ 1 開催日の UI 最小カテゴリ。約 0.3MB・再現可能な形式）+ `MANIFEST.json`。`git pull` 後は `make dev-data`（= `python3 -m src.scripts.data.dev_fixture ensure`。モック生成 + 実データ日の展開、冪等）、または dev 自動起動（`autostart_dev.sh`）が自動で行う。取り込み直したら `make dev-fixture-pack DATE=YYYYMMDD`。中身は netkeiba 取得データの抜粋。dev はこの PC からの直アクセスを**自動ログイン済み**として扱う（`src/api/auth.py` `dev_auto_login_applies`。KEIBA_ENV=dev・ループバック・Host が localhost のときだけ。公開トンネル経由は対象外。止めるなら `KEIBA_DEV_AUTO_LOGIN=0`）。週間予測の既定日は `NEXT_PUBLIC_DEFAULT_RACE_DATE`（`service_start.profiles.sh`）。推奨馬券の購入上限は UI で指定（既定 1R 合計 2,000 円、空欄/0 で上限なし。API は `max_total_bet`、`src/api/v1/delegates.py` の `DEFAULT_MAX_TOTAL_BET`） |
| 馬連・ワイド・3連複の確率 | `src/pipeline/inference/exotic_probabilities.py`（勝率・連対率・複勝率に整合する割引 Harville＝Benter 型。α はレースごとに連対率・複勝率へ当てはめ）。仕様・文献・検証: `docs/html/modeling/chapters/exotic_bet_probabilities.html`。学習済みモデルの出力での検証・調整: `python3 -m src.scripts.data.validate_exotic_probabilities predictions --from YYYYMMDD --to YYYYMMDD [--write-config]`（モック予測は除外、200R 未満は書かない） |
| ハロンごとのラップ予想 | `src/pipeline/models/lap_forecast.py`（ラップ傾向＝スロー/ミドル/ハイの発生確率 + 傾向ごとのハロン別タイムの混合）。特徴量 `src/pipeline/features/tp55_lapfeat.py`、学習表 `src/scripts/data/tp55_build.py`、評価 `tp55_eval.py`。予測 API は環境変数 `KEIBA_LAP_FORECAST=1` のときだけ `lap_forecast`（保存済みの予想を付与。既定は無効、`src/pipeline/inference/lap_forecast_service.py`）。保存は `python -m src.scripts.data.tp55_backfill --dates YYYY-MM-DD ...`（出馬表が書き出された日。保存先 `data/local/inference/lap_forecast/`、`KEIBA_LAP_FORECAST_DIR` で変更）。仕様・精度・先読みの検査: `docs/html/modeling/chapters/lap_forecast.html` |
| 追走難度 v2 の前夜の保存 | `KEIBA_TRACKING_V2=1` のとき、raceday-eve が翌日分の追走難度 v2 を `data/local/inference/tracking_v2/<日>/<race_id>.json` に保存し（`src/pipeline/inference/tracking_v2_service.py`）、予測 API は保存を優先（取消で馬番の集合が変わったレースだけ計算し直し）。保存と実測の週次の突き合わせは `src/pipeline/inference/tracking_v2_eval.py`（`KEIBA_TRACKING_V2_EVAL=1`）。仕様・cron・バックフィル: `docs/operations/tracking_v2_precompute.md` |
| レース選択の価値表示・買い目の流し表示 | レース詳細の R チップ/週間予測カードは、推奨ポートフォリオの**期待回収率（較正後）**で色分け（段の閾値は `config/ev_calibration.json` の `levels`、★オレンジ=閾値以上かつその日の上位20%、灰=見送り）。表示する期待値は過去の確定オッズ・確定払戻に合わせて較正した値で（`src/utils/ev_calibration.py`、検証: `docs/html/modeling/chapters/ev_calibration_adopt.html`）、ポートフォリオの期待回収率は年別の範囲を併記する目安。API: `GET /api/v1/betting/race-values?date=YYYYMMDD`（`delegates.race_value_summary`、30秒キャッシュ・ログイン必須）、フロント: `frontend/lib/raceValue.ts`。推奨買い目の表は券種ごと＋同じ軸から流せるものは流し表示（`frontend/lib/betGrouping.ts`。馬連/ワイド=軸1頭、馬単=1着/2着軸、3連複=軸1頭の全組合せ or 軸2頭。金額・確率は変えず表示だけまとめる）。確率が未検証なので価値表示も目安。**EV の水準**（モデルの EV は実現回収率より高く出る）の監査と縮約は `docs/html/modeling/chapters/ev_calibration_audit.html`、縮約の設定は `config/ev_calibration.json`（既定は無効、`src/utils/ev_calibration.py`） |
| 確率の市場補正 | `KEIBA_PROB_CALIBRATION=1`（既定オフ）で、実オッズが全頭に揃う 8 頭以上のレースの勝率・連対率・複勝率を、単勝オッズの市場確率との対数線形ブレンドに補正し（`src/utils/prob_calibration.py`、係数 `config/prob_calibration.json`）、EV・印・推奨を補正後の確率で付け直す。保存済みの予測は書き換えず読み出し時に計算。応答は `*_model`（元の確率）と `prob_calibration{applied,reason,...}`。補正したレースは `config/ev_calibration_pc.json`（二重補正を避けた EV 較正）。運用・再当てはめ: `docs/operations/prob_calibration.md`、検証: `docs/html/modeling/chapters/prob_calibration_gap.html` |
| `config/ports.env` | **全ローカルサーバのポートの唯一の正**（dev=231xx / stg=232xx / prod=233xx / monitor=23900）。bash は `scripts/server/ports.sh`、Python は `src/config/ports.py`。クリックで開ける一覧は `docs/monitoring-portal.md` §0。dev は IDE を開くと `scripts/server/autostart_dev.sh` が自動起動（Claude Code `.claude/settings.json` / Cursor `.cursor/hooks.json`） |

**ped_tbl 増分**: `python -m src.pipeline.sync_ped_tbl_for_horses --horse-ids …`（ローカル `horse_pedigree_5gen` 参照）。出馬表 `race_shutuba` 保存直後の自動生成は `.env` で `KEIBA_SYNC_PED_TBL_ON_SHUTUBA=1` のときのみ。接木は `KEIBA_PED_TBL_MERGE_GEN5`（未設定時 1）。

**保存前スキーマ検証**: `HybridStorage.save` が `schemas.validate` を必ず実行。**不合格のときは「どの項目がどの値で」を記録する**（`data/local/meta/schema_violations/<category>.jsonl`、拒否したデータ本体は `data/local/quarantine/`、保存した場合は `_meta.schema_validation.violations`）。確認は `python -m src.scraper.schema_violations summary|show`。`KEIBA_SCHEMA_STRICT` 未設定または `1` で不合格時は GCS 非保存・`SchemaValidationError`（キューは `failure_reason=schema_validation`）。診断のみ許容する場合は `KEIBA_SCHEMA_STRICT=0`。モニター `/monitor` と `GET /api/scrape-jobs` の `schema_validation_failures` を参照。

**要件表↔ストレージの行単位整合**: `docs/requirements/data/scrape_process.md` の netkeiba 表は `src/scraper/requirement_row_catalog.py` の `row_id` と対応。参照 JSON は `requirement_row_trace`（GCS `others/`）。発走時刻スナップショットは `race_day_schedule`（`data/page_reference/race_day_schedule/`）。バックフィル: `python3 -m src.scripts.scraping.materialize_requirement_row_traces`。

**行固有派生カテゴリ（物理分割済み）**: 複数行が共有していた canonical JSON（`race_shutuba` / `race_result_on_time` / `race_result` / `race_result_lap` / `horse_result`）から必要フィールドを抽出した派生カテゴリが GCS に存在する（2020-2026 年・全馬）。抽出ロジック: `src/scraper/row_data_extractor.py`。カテゴリ一覧: `race_shutuba_meta` / `race_result_on_time_payoff` / `_lap` / `_corner` / `horse_profile` / `horse_race_history` / `race_result_meta` / `_payoff` / `_track` / `race_result_corner` / `race_result_lap_times`（合計 11 カテゴリ、261,173 件保存）。追加・再実行: `python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths --year-start 2020 --year-end 2026 --include-horses`。

**スクレイピング SLA・スケジュール**: 各エンドポイントの公開時期と定期取得の対応は `src/scraper/sla_spec.py`（テスト `tests/scraper/test_sla_spec.py` が `TASK_CATALOG` とタスクリストの食い違いを検出）、監査結果・穴の一覧は `docs/operations/scraping_sla_audit_20261013.md`。cron は **`scripts/auto_scrape.sh <mode>` 経由**（cron から python3 を直接呼ぶと bs4 無しで失敗）。JST マシンの雛形は `scripts/cron/crontab.jst.example`（`setup_all_cron.sh` は UTC マシン前提）。公開時刻の実測は `python3 -m src.scraper.availability_watch --auto --yes`（dev 専用）。バロメーターは `barometer-retry`（公開済み・未取得のみ）。

## 作業の指針（短く）

- 既存の命名・モジュール分割に合わせ、**依頼範囲だけ**変更する。
- ファイル探索は `Read` / リポジトリ `Grep` / `Glob` を優先（巨大 `data/` の丸読みは避ける）。
- モデリングや指標の意味を変える変更は、**`docs/html/modeling/` の該当仕様**と整合を取る。
- スクレイパや保存形式を触る場合は、検証系スクリプト（`src/scraper/validate_storage.py` 等）の有無を確認してから進める。

## 環境

- Python: `requirements.txt`、`.env.example` → `.env`（認証はユーザー環境）。
- **dev（開発PC）は GCP へ一切接続しない**: `.env` に `KEIBA_ENV=dev` が必須（未設定は prod 扱い）。dev では `HybridStorage` が `GCS_BUCKET` を無視して `data/dev_mock/` を読み書きし、GCS・Cloud SQL・Cloud Tasks・BigQuery のクライアント生成は `GcpAccessForbidden`（`src/config/gcp_guard.py`）。モックは `make dev-mock`（`src.scripts.data.make_dev_mock`）で生成し、スキーマ定義済みの**全カテゴリ**にスキーマ適合のサンプルを持つ（派生 11 カテゴリは本番と同じ `row_data_extractor` で生成。新スキーマを足したらモック生成も足す。`tests/scraper/test_dev_mode.py` が検出）。テストは `tests/conftest.py` が `KEIBA_ENV` を空にして CI と同条件で走らせる（dev の挙動テストは `tests/scraper/test_dev_mode.py`）。
- テスト一式: リポジトリルートで `make test`（内部で `.github/workflows/ci.yml` と同じ pytest 実行順序・除外設定を再現。DB/Redis 未起動でも大半は動く）。Makefile を使わない場合は `python3 -m pytest tests/ --ignore=tests/scraper/manual --ignore=tests/research/manual`
- **スキーマ**: 正本は `src/scraper/schema_defs.json`（git 管理・全環境同一）。実データからの再構成は `python -m src.scraper.schema_infer collect|report|apply`（観測プロファイル `docs/requirements/data/schemas/observed/` も git 管理。dev のモックは推論に使わない）。新カテゴリを `CATEGORY_MAP` に足したら、スキーマか `no_schema_reason` を必ず追加（`tests/scraper/test_schema_infer.py` が検出）。
- **stg の受け入れ判定（2020 年以降が全件揃い、スキーマに適合）**: `bash docs/todos/verify/stg_data_complete.sh`（= `KEIBA_ENV=stg python -m src.data_health --require-complete`。未達は終了コード 3）。
- **データ存在チェック／ヘルスチェック**: `make data-health`（= `python -m src.data_health`。stg は `KEIBA_ENV=stg make data-health`）。環境別の要件は `src/data_health/spec.py`、出力は `{latest.html,latest.json,scrape_plan.json,history.jsonl}`（不足期間のヒートマップ・インフラ疎通・スクレイピング計画）。2020 年以降の全データについて**スキーマ適合（健全性）**まで検証し（`DATA_HEALTH_VALIDATE=full` 既定。台帳で差分のみ再検証、1 回の download 件数は `DATA_HEALTH_VALIDATE_BUDGET`）、race_id ↔ 開催日・場・R・レース名のキーテーブル（`race_keys.csv`）と、不足・不適合・未検証の race_id 配列（`race_ids/*.txt`）を出す。設定は `DATA_HEALTH_*`（`src/data_health/config.py`、`.env.example` 末尾）・`--env`、結果は環境別に `data/local/meta/data_health/<環境キー>/` へ保存し全環境の一覧は同 `index.html`（他 PC の結果は `--import <結果ディレクトリ>`）。**ダッシュボード（サーバ不要）**: 同ディレクトリの `dashboard.html` をブラウザで開くだけで dev / stg(=prod) をタブ切替・自動更新で確認（`make data-health-dashboard`、実行中の進捗も表示。更新はチェック実行時点）。テスト: `tests/data_health/`。
- 騎手・調教師統計のマージキー検証: `python3 -m unittest tests.pipeline.test_jockey_trainer_stats -v`
- netkeiba 実 HTML を叩く手動スモーク（unittest 対象外）: `tests/scraper/manual/netkeiba_horse_page_smoke.py`, `tests/scraper/manual/netkeiba_speed_index_smoke.py`
- 血統メタクラスタの手動検証（unittest 対象外）: `tests/research/manual/verify_*.py`。バックテスト・仮説検証バッチ: `src/research/pedigree/backtest_*.py`, `*_evidence.py`（中間 Parquet は `data/analysis/pedigree/`）

### WSL2 と Cursor（`installServerScript` / `Wsl/Service/0x80072746`）

**症状**: `Connection to Cursor server failed: [wsl exec: installServerScript]`、標準出力に「既存の接続はリモート ホストに強制的に切断されました」、`Wsl/Service/0x80072746`。終了コード `4294967295`（符号なしの -1）は、Windows 側が WSL との通信・子プロセスを異常扱いしたときに出やすい。

**想定される原因**（併発しやすい）:

1. **WSL2 の VM が落ちた／応答不能** — メモリ不足（OOM）、スワップ不足、短時間の CPU・I/O 飽和で vmmem が不安定になる。
2. **ホスト–ゲスト間の接続リセット** — スリープ復帰、Hyper-V / 仮想スイッチの再構成、VPN やセキュリティ製品が WSL のソケットやパイプを切断する。
3. **Cursor のリモートサーバ展開が途中で切れた** — 上記 1 または 2 のあいだに `installServerScript` が走ると、同じメッセージで失敗する。

**本リポジトリで起きやすい文脈**: `build_horse_pedigree_10gen` や `build_pedigree_10gen_3view_index` など、大量の JSON / Parquet を扱う処理はメモリを消費する。Cursor エージェントや IDE と同時に WSL 上で走らせると、WSL の既定メモリ上限に達して 1 が起き、続けて Cursor 接続だけが切れる、というパターンが現実的にある。

**改善手順（上から試す）**:

1. Windows の PowerShell で `wsl --shutdown` を実行し、数十秒待ってから Cursor を起動し直す。
2. `C:\Users\<ユーザー名>\.wslconfig` に `[wsl2]` を置き、例として `memory=8GB` 以上・`swap=8GB` 程度・`processors=4` などを明示する（マシンに合わせて調整）。保存後に必ず `wsl --shutdown` で反映。
3. `wsl --update` と、可能なら Windows Update を最新にする。
4. WSL 内で `rm -rf ~/.cursor-server` してから、Cursor から WSL ワークスペースを開き直す（展開済みサーバが壊れている場合の切り分け）。
5. **重いバッチは Cursor の外で回す** — 例: `nohup` / `tmux`、または Windows ターミナルから WSL のみ起動して `scripts/run_after_scrape_missing_5gen_10gen_chain.sh` を実行し、IDE とのメモリ競合を避ける。

完全に IDE 側のバグだけで説明できない場合もあるが、実務では **WSL のメモリ・安定性** と **接続を切るソフトウェア** の切り分けが効く。

## ユーザー向け応答

- ユーザーとの説明・コミットメッセージは **日本語**（プロジェクトルールに従う）。
