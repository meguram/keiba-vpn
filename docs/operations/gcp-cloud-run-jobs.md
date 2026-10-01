# GCP Cloud Scheduler + Cloud Run Jobs 移行ガイド（定期ジョブ一覧）

**対象領域**: GCP側（スクレイピング・ML・スケジュール実行専用）に移行する定期ジョブの実行コマンド・頻度・リソース目安
**関連ドキュメント**: [deployment-vps-vs-gcp.md](./deployment-vps-vs-gcp.md)（役割分担の全体方針）・
[service-endpoints.md](./service-endpoints.md)・[AGENTS.md](../../AGENTS.md)
**実行コマンドの設計図**: [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../scripts/gcp/deploy_cloud_run_jobs.sh)
**最終更新**: 2026-10-01

## 位置づけ

現在 `src/api/app.py` 内の FastAPI プロセス常駐 daemon thread、および `scripts/cron/` の
OS crontab で動いている定期ジョブ群を、GCP の **Cloud Scheduler + Cloud Run Jobs** へ移行する
ための CLI エントリポイントをまとめる。本ドキュメントと `scripts/gcp/deploy_cloud_run_jobs.sh`
は設計図であり、**実際の GCP デプロイ・スケジューラ登録はこのドキュメント作成時点では行っていない**
（別途ユーザー側作業）。

各ジョブの CLI は `argparse` で引数を受けて既存の処理関数をそのまま呼ぶだけの薄いラッパーである
（処理ロジック自体の再実装はしていない）。既に `src/pipeline/` 等に移行済みのバッチ CLI がある
ジョブは、新規ファイルを作らずそのコマンドをそのまま載せている。

## ジョブ一覧

| # | ジョブ | 実行コマンド | 新規/既存 |
|---|---|---|---|
| 1 | 構造チェック | `python -m src.scraper.run structure-check` | 既存 (`src/scraper/run.py`) |
| 2 | キュー定期メンテ | `python -m src.scripts.cloud_jobs.queue_maintenance` | 新規 |
| 3 | 出馬表自動取得 | `python -m src.scripts.cloud_jobs.daily_shutuba_enqueue` | 新規 |
| 4 | 週次種牡馬集計 | `python -m src.scripts.maintenance.aggregate_sire_aptitude` | 既存 |
| 5 | 騎手・調教師統計 | `python -m src.pipeline.build_jockey_trainer_stats` | 既存 |
| 6 | レース質日次一括推定 | `python -m src.scripts.cloud_jobs.race_quality_day` | 新規 |
| 7 | 追走難度precompute | `python -m src.scripts.maintenance.precompute_tracking_difficulty_all --skip-existing` | 既存 |
| 8 | track-speedベースライン再構築 | `python -m src.research.race.build_track_speed_baselines` | 既存 |
| 9 | ミオスタチン再計算 | `python -m src.scripts.cloud_jobs.myostatin_recalculate` | 新規（処理ロジックは `src/research/genes/myostatin.py` の `recalculate_myostatin_genotypes()` に切り出し） |
| 10 | オッズ予測モデル学習 | `python -m src.scripts.data.train_final_odds_model` | 既存 |
| 11 | 血統アーティファクト再構築（pair_lift/role_lift等） | `python -m src.research.pedigree.build_pair_lift_profiles` 等、`src/api/app.py`の`_BLOODLINE_ARTIFACTS`の`rebuilder`に列挙された各モジュール | 既存（モジュールごとに`if __name__ == "__main__"`対応済み） |
| 12 | 種牡馬ツリー再構築 | `python -m src.research.pedigree.build_full_sire_tree` | 既存（`relevant_stallion_ids`再生成は`src/api/app.py`の`_regenerate_relevant_stallion_ids`に private実装のまま残存。CLI化する場合は切り出しが別途必要、未対応） |
| 13 | 5代血統整備（開催日範囲一括） | `python -c "from src.research.pedigree.race_pedigree_5gen_prefetch import batch_race_pedigree_5gen_date_range; ..."`（`/api/pedigree/batch-race-ensure-5gen`と同じ関数） | 既存（直接呼べるCLIラッパーは無し、簡易`python -c`呼び出しか`src/scripts/cloud_jobs/`への薄いラッパー追加が必要。未作成） |
| 14 | クッション値ライブ取得 | `python -m src.scraper.jra_baba_live`（`JRABabaLiveScraper.scrape()`、構造変更検知・Slack通知込み） | 既存 |

---

### 1. 構造チェック

- **実行コマンド**: `python -m src.scraper.run structure-check`
  （内部で `src.scraper.structure_monitor.run_daily_check(auto_reparse=True, notify=True)` を呼ぶ。
  引数省略時はサンプル race_id/horse_id を自動検出する。元の daemon thread
  `_scheduler_loop`（`src/api/app.py`）と同一の呼び方）
- **想定実行頻度**: 毎日 06:00 JST（元 daemon thread `STRUCTURE_CHECK_HOUR_JST=6` /
  `STRUCTURE_CHECK_MINUTE_JST=0` と同一）
- **リソース目安**: メモリ 1GiB、タイムアウト 900秒（15分）。
  全カテゴリのページ構造を実際にフェッチ＆比較し、CRITICAL 時は再パースも行うため
  netkeiba への実アクセスを伴う。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http structure-check-daily \
    --schedule="0 6 * * *" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/structure-check:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 2. キュー定期メンテ

- **実行コマンド**: `python -m src.scripts.cloud_jobs.queue_maintenance`
  （`src.scraper.job_queue.run_hourly_queue_maintenance()` をそのまま呼ぶ）
- **想定実行頻度**: 1時間ごと（元 daemon thread
  `SCRAPE_QUEUE_HOURLY_MAINTENANCE_SEC`既定 3600秒と同一）
- **リソース目安**: メモリ 256MiB、タイムアウト 300秒（5分）。
  ストールjob回収・completedレコード削除・Slack通知のみでスクレイピング自体は行わない軽量処理。
- **注意**: `run_hourly_queue_maintenance()` は内部で `kick_process_queue_background()` /
  `kick_urgent_process_queue_background()` を呼び、新規 daemon thread で
  `ScrapeJobQueue.process_queue()` を起動する。Cloud Run Jobs の1回限りの実行では、
  CLI プロセスが先に終了すると起動直後の daemon thread が完了を待たれずに終了する可能性がある
  （元の FastAPI 常駐プロセスでは問題にならない挙動の差）。実際のキュー処理（スクレイピング実体）
  は別途常時稼働するワーカー（GCP側のCompute Engine/Cloud Run サービス等、`scraping-queue`
  TODOの移行先）に委譲する前提とし、本ジョブは「キューの掃除」役に限定するのが安全。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http queue-maintenance-hourly \
    --schedule="0 * * * *" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/queue-maintenance:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 3. 出馬表自動取得（daily-shutuba）

- **実行コマンド**: `python -m src.scripts.cloud_jobs.daily_shutuba_enqueue`
  （`src.scraper.period_runners.enqueue_race_tasks_for_race_period()` を、元の
  `_daily_shutuba_enqueue_loop`（`src/api/app.py`）と同じデフォルト引数
  `tasks=["race_shutuba"], smart_skip=True, jra_only=True, limit=500, priority=10`、
  期間は今日〜+14日で呼ぶ）
- **想定実行頻度**: 毎日 07:00 JST（元 daemon thread既定
  `DAILY_SHUTUBA_HOUR_JST=7` / `DAILY_SHUTUBA_MINUTE_JST=0` /
  `DAILY_SHUTUBA_DAYS_AHEAD=14` と同一）
- **リソース目安**: メモリ 512MiB、タイムアウト 600秒（10分）。
  race_lists の走査とキュー投入のみ（実スクレイピングは別ワーカー）。
- **注意**: ジョブ#2と同様、投入後に `kick_process_queue_background()` を呼ぶが
  実処理は別ワーカーに委譲する前提。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http daily-shutuba-enqueue \
    --schedule="0 7 * * *" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/daily-shutuba-enqueue:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 4. 週次種牡馬集計

- **実行コマンド**: `python -m src.scripts.maintenance.aggregate_sire_aptitude`
  （既存 CLI。`run_aggregation(storage, _current_week_label())` を呼ぶ。
  元の `_weekly_sire_agg_loop`（`src/api/app.py`）と同一処理）
- **想定実行頻度**: 毎週月曜 06:00 JST（元 daemon thread と同一）
- **リソース目安**: メモリ 2GiB、タイムアウト 1800秒（30分）。
  過去レース全体を集計するため中〜大規模なメモリを要する。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http weekly-sire-agg \
    --schedule="0 6 * * 1" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/weekly-sire-agg:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 5. 騎手・調教師統計

- **実行コマンド**: `python -m src.pipeline.build_jockey_trainer_stats`
  （既存 CLI。AGENTS.md 記載の正式エントリ。現行は
  `scripts/cron/update_jockey_trainer_stats.sh` 経由で OS crontab から呼ばれている）
- **想定実行頻度**: 毎日 05:30 JST（現行 crontab `30 20 * * *` UTC と同一）
- **リソース目安**: メモリ 2GiB、タイムアウト 1800秒（30分）。
  過去レース全体の騎手・調教師成績を集計するため中規模なメモリ・時間を要する。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http jockey-trainer-stats-daily \
    --schedule="30 5 * * *" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/jockey-trainer-stats:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 6. レース質日次一括推定

- **実行コマンド**: `python -m src.scripts.cloud_jobs.race_quality_day`
  （`src.research.race.race_quality_model.analyze_date()` を、元の
  `GET /api/race-quality/day`（`src/api/app.py`）と同じ呼び方で実行。`--date` 省略時は
  本日 JST）
- **想定実行頻度**: **元はオンデマンドAPI呼び出しのみで常駐処理・cronは存在しない**。
  レース結果が確定した後に実行する必要があるため、提案値として毎日 19:00 JST
  （JRA開催日の結果確定後）を想定。実際の運用頻度はユーザー側で確認・調整すること。
- **リソース目安**: メモリ 1GiB、タイムアウト 600秒（10分）。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http race-quality-day \
    --schedule="0 19 * * *" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/race-quality-day:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 7. 追走難度precompute

- **実行コマンド**: `python -m src.scripts.maintenance.precompute_tracking_difficulty_all --skip-existing`
  （既存のバッチ CLI。`POST /api/race/{race_id}/tracking-difficulty/precompute`
  （`src/api/app.py`）と同じ `build_tracking_difficulty_response()` /
  `save_cached_response()` を使い、全レースを `--skip-existing`（既定True）で
  増分計算する。1レースのみ処理する単発版は `precompute_tracking_difficulty.py`）
- **想定実行頻度**: **元はオンデマンドAPI呼び出し + 手動バッチ実行のみで固定cronは存在しない**。
  提案値として毎日 08:00 JST（出馬表取得後、レース前の事前計算用）。
- **リソース目安**: メモリ 2GiB、タイムアウト 3600秒（1時間）。
  `--skip-existing` 既定ONの日次差分実行なら数分〜十数分で終わる想定だが、
  初回フルバックフィル（`--no-skip-existing`）は数時間かかる可能性があるため
  別途ジョブ実行時間を上限いっぱいまで確保しておく。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http tracking-difficulty-precompute \
    --schedule="0 8 * * *" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/tracking-difficulty-precompute:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 8. track-speedベースライン再構築

- **実行コマンド**: `python -m src.research.race.build_track_speed_baselines`
  （既存 CLI。`TrackSpeedEngine.build_baselines()` を呼ぶ。
  `POST /api/track-speed/rebuild-baselines`（`src/api/app.py`）の
  `_run_track_speed_baselines()` と同じ処理だが、同関数が追加で呼ぶ
  `invalidate_sigma_cache()` はサービング側プロセスのインメモリキャッシュ無効化であり
  本バッチには対象プロセスが無いため含まれない。VPS側サービングプロセスには、
  本ジョブ完了後にベースラインファイル更新を反映させるための再読込・再起動の仕組みが
  別途必要になる点に注意）
- **想定実行頻度**: **元はUI手動トリガーのみで固定cronは存在しない**。
  提案値として毎週日曜 04:00 JST（週内のレース結果が揃った後）。
- **リソース目安**: メモリ 2GiB、タイムアウト 1800秒（30分）。
  2020-2025年の全レースparquetを読み込んで集計するため中規模なメモリを要する。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http track-speed-rebuild-baselines \
    --schedule="0 4 * * 0" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/track-speed-rebuild-baselines:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 9. ミオスタチン再計算

- **実行コマンド**: `python -m src.scripts.cloud_jobs.myostatin_recalculate`
  （処理ロジックを `POST /api/myostatin/recalculate`（`src/api/app.py`）のハンドラ内から
  `src.research.genes.myostatin.recalculate_myostatin_genotypes()` へ切り出し、
  API・本CLIの両方から共通で呼ぶように変更した。**実装上の注記**: 元のAPIハンドラは
  JSONファイルパスを `data/local/knowledge/myostatin_genes.json` にハードコードしていたが、
  実際のナレッジベースファイルは `src.config.data_paths.MYOSTATIN_GENES_JSON`
  （本リポジトリでは `data/calculated_data/knowledge/myostatin_genes.json`）にのみ存在し、
  同じモジュール内の `GET /api/myostatin` 等は元々 `MYOSTATIN_GENES_JSON` を使っていた。
  切り出しに合わせてこの不整合も修正し、`recalculate_myostatin_genotypes()` の既定パスを
  `MYOSTATIN_GENES_JSON` に統一した）
- **想定実行頻度**: **元はUI手動トリガーのみで固定cronは存在しない**。
  ナレッジベースの更新頻度は低い（種牡馬の新規登録時のみ変化）ため、提案値として
  毎月1日 05:00 JST。
- **リソース目安**: メモリ 256MiB、タイムアウト 120秒（2分）。
  JSONファイル（数百件規模）の読み書きのみの軽量処理。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http myostatin-recalculate-monthly \
    --schedule="0 5 1 * *" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/myostatin-recalculate:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

### 10. オッズ予測モデル学習

- **実行コマンド**: `python -m src.scripts.data.train_final_odds_model`
  （既存 CLI。`FinalOddsTrainer.train()` を呼ぶ。
  `POST /api/odds/train`（`src/api/app.py`）の `_run_odds_training()` と同じ処理。
  学習年・評価年は既定 `--train-years 2020,2021,2022,2023,2024 --eval-years 2025`
  のため、年が変わる場合は呼び出し側で更新する運用が必要）
- **想定実行頻度**: **元はUI手動トリガーのみで固定cronは存在しない**。
  提案値として毎週日曜 03:00 JST（週次の最終オッズデータ蓄積を反映した再学習）。
- **リソース目安**: メモリ 4GiB、タイムアウト 7200秒（2時間）。
  複数年のレース全件を使ったML学習かつMLflowへの登録を伴うため最も重い処理。
- **Cloud Scheduler 設定コマンド例**:
  ```bash
  gcloud scheduler jobs create http odds-train-weekly \
    --schedule="0 3 * * 0" \
    --time-zone="Asia/Tokyo" \
    --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/odds-train:run" \
    --http-method=POST \
    --oauth-service-account-email="${SCHEDULER_SA_EMAIL}" \
    --location="${REGION}"
  ```

## 共通の注意事項

- 上記の `gcloud scheduler jobs create http` コマンドはいずれも例であり、
  `${PROJECT_ID}` / `${REGION}` / `${SCHEDULER_SA_EMAIL}` はユーザー側で実値に置き換えること。
  実行・登録はこのドキュメント作成時点では行っていない。
- Cloud Run Jobs の実行 API を呼ぶ Cloud Scheduler ジョブには、
  `roles/run.invoker` を付与したサービスアカウントによる OIDC/OAuth 認証が必要
  （`--oauth-service-account-email`）。
- 各ジョブのコンテナイメージ・デプロイ設計図は
  [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../scripts/gcp/deploy_cloud_run_jobs.sh) を参照。
- ジョブ#6〜#10は現状オンデマンド実行のみで固定の自動実行頻度が存在しないため、
  上記の「提案値」は本ドキュメント作成時点での推測であり、実際の運用で調整が必要。
- 認証・GCS疎通については [deployment-vps-vs-gcp.md](./deployment-vps-vs-gcp.md) の
  サービスアカウント設定を前提とする。
