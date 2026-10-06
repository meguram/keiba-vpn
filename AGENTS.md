# keiba-vpn — エージェント向けメモ

競馬データ基盤（スクレイピング・特徴量・ML・Web）。README の詳細は `README.html`。

## レイアウト

すべての Python コードは **`src/`** に集約され、役割ごとにサブパッケージで分割されている。シェルスクリプトとデータ・テスト・ノートブックはトップレベルに残す。

| 領域 | 役割 |
|------|------|
| `src/api/` | FastAPI（`app.py`）。エントリは `python main.py` → `src.api.app:app` |
| `src/monitor/` | **開発者専用監視ポータル**（スタンドアロン Flask :9090）。`app.py` + `templates/`。起動: `bash scripts/server/start_monitor.sh` / `service_start.sh --env dev --monitor`。仕様書: `docs/monitoring-portal.md` |
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
| `data/features/` | 特徴量ストア。ブロック例: `base_tbl`（4キーのみの ``shutuba``）・`race_tbl`・`race_horse_tbl`・`horse_tbl`（血統・馬プロファイル等）・`race_jockey_tbl` / `race_trainer_tbl`・`jockey_tbl` / `trainer_tbl`・`jockey_trainer_stats/`（メタ）。**ラベル**: `target/rank_tbl/<年>/rank.parquet`（`race_id`,`horse_id`,`rank`）は `python -m src.pipeline.build_rank_target`。**馬単位エンティティ**（`horse/ped_tbl|result_tbl|training_tbl/{馬ID先頭4桁}/{horse_id}.parquet`、年ではなく馬シャード）は `python -m src.pipeline.build_horse_entity_store`。`docs/html/data/data_features_reference.html` の `#join-architecture`。出馬表 raw は `python -m src.pipeline.register_raw_table_features` → `_raw_table_feature_selection.json`。騎手・調教師は `python -m src.pipeline.build_jockey_trainer_stats`（定期: `scripts/cron/update_jockey_trainer_stats.sh` 等）。 |
| `docs/html/` | システム文書（`ARCHITECTURE.html` / `modeling/*.html` など。設計の正） |
| `scripts/` | シェルスクリプトのみ。`server/`（API 起動）・`cron/`（定期実行）。Python の CLI は `src/scripts/` を参照 |
| `docs/operations/` | 運用メモ。**ディレクトリ構成**: `PROJECT_LAYOUT.md`。**起動 URL・ポート**: `service-endpoints.md` |
| `mlflow/` | MLflow 関連。`data/`（Docker サーバの DB + artifacts、`.gitignore`）・`runs/`（ローカルフォールバックのファイルストア、`.gitignore`）・`server/`（Docker Compose / Nginx / `setup.sh`）。**モデル一覧・追加手順**: `docs/html/design/mlflow_platform.html` / `src/pipeline/mlflow/catalog.py` |
| `config/` | `settings.yaml`（MLflow モデルキー等。`src/config/` はコード側プロファイル） |
| `tests/` | unittest（`api/` `pipeline/` `research/` `scraper/`） |
| `main.py` | サーバエントリ（`python main.py --port 8000`） |

**ped_tbl 増分**: `python -m src.pipeline.sync_ped_tbl_for_horses --horse-ids …`（ローカル `horse_pedigree_5gen` 参照）。出馬表 `race_shutuba` 保存直後の自動生成は `.env` で `KEIBA_SYNC_PED_TBL_ON_SHUTUBA=1` のときのみ。接木は `KEIBA_PED_TBL_MERGE_GEN5`（未設定時 1）。

**保存前スキーマ検証**: `HybridStorage.save` が `schemas.validate` を必ず実行。**不合格のときは「どの項目がどの値で」を記録する**（`data/local/meta/schema_violations/<category>.jsonl`、拒否したデータ本体は `data/local/quarantine/`、保存した場合は `_meta.schema_validation.violations`）。確認は `python -m src.scraper.schema_violations summary|show`。`KEIBA_SCHEMA_STRICT` 未設定または `1` で不合格時は GCS 非保存・`SchemaValidationError`（キューは `failure_reason=schema_validation`）。診断のみ許容する場合は `KEIBA_SCHEMA_STRICT=0`。モニター `/monitor` と `GET /api/scrape-jobs` の `schema_validation_failures` を参照。

**要件表↔ストレージの行単位整合**: `docs/requirements/data/scrape_process.md` の netkeiba 表は `src/scraper/requirement_row_catalog.py` の `row_id` と対応。参照 JSON は `requirement_row_trace`（GCS `others/`）。発走時刻スナップショットは `race_day_schedule`（`data/page_reference/race_day_schedule/`）。バックフィル: `python3 -m src.scripts.scraping.materialize_requirement_row_traces`。

**行固有派生カテゴリ（物理分割済み）**: 複数行が共有していた canonical JSON（`race_shutuba` / `race_result_on_time` / `race_result` / `race_result_lap` / `horse_result`）から必要フィールドを抽出した派生カテゴリが GCS に存在する（2020-2026 年・全馬）。抽出ロジック: `src/scraper/row_data_extractor.py`。カテゴリ一覧: `race_shutuba_meta` / `race_result_on_time_payoff` / `_lap` / `_corner` / `horse_profile` / `horse_race_history` / `race_result_meta` / `_payoff` / `_track` / `race_result_corner` / `race_result_lap_times`（合計 11 カテゴリ、261,173 件保存）。追加・再実行: `python3 -m src.scripts.scraping.migrate_row_data_to_unique_paths --year-start 2020 --year-end 2026 --include-horses`。

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
