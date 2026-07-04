# AREA-10 — インフラ・サーバーモニタリング要件
**Status**: FINAL | **Last Updated**: 2026-07-04 | **Consolidates**: AREA-04 §6（監視・アラート）拡張

---

## 0. 前提条件

**対象環境**: ConoHa VPS — RAM 2GB 制約（AREA-01 §1）
**監視対象**: keiba-vpn の全プロセス（API サーバー / MLflow / スクレイパー / Watchdog / PostgreSQL / Redis）

---

## 1. リソース監視しきい値

### 1-1. CPU

| メトリクス | 警告（WARN） | 危険（CRIT） | 対処 |
|---|---|---|---|
| 平均 CPU 使用率（1分） | ≥ 70% | ≥ 90% | スクレイパー同時スロット数を削減 |
| 推論バッチ実行中の CPU | ≥ 90%（一時的可） | ≥ 90% が 10分継続 | バッチ実行時間帯を見直し |

### 1-2. メモリ（RAM 2GB 上限）

| プロセス | 通常使用量目安 | 上限（kill 対象） | 備考 |
|---|---|---|---|
| Flask API（`flask_app.py`） | 80-120 MB | 300 MB | 推論結果 Redis キャッシュ効果込み |
| PostgreSQL | 100-200 MB | 400 MB | `shared_buffers` 64MB 推奨 |
| Redis | 30-80 MB | 150 MB | `maxmemory 128mb` 設定 |
| MLflow サーバー×4 | 各 80-150 MB | 各 250 MB | keiba_lgbm / tracking_difficulty / final_odds / pace_predictor |
| スクレイパー（SLA 3 T-15バンドル） | 60-100 MB | 200 MB | バンドル実行時はスパイクに注意 |
| **合計上限目安** | 550-850 MB | **1,800 MB** | 残 200MB はカーネル・OS 予備 |

**全体アラート**:
- 使用率 ≥ 80%（1,640 MB）: WARN — スクレイパー1スロット削減
- 使用率 ≥ 90%（1,840 MB）: CRIT — 非必須プロセス（MLflow PLANNED）を停止

### 1-3. ディスク・ストレージ

| パス | 用途 | WARN | CRIT | 対処 |
|---|---|---|---|---|
| `/` (root) | OS + アプリ | 70% 使用 | 85% 使用 | ログローテーション実施 |
| `data/cache/` | L2 ディスクキャッシュ | 2 GB | 4 GB | `disk_cache_cleanup` 強制実行 |
| `data/queue/` | ジョブキュー | 100 MB | 500 MB | 完了済みジョブを削除 |
| `data/calculated_data/` | 計算済みデータ | 1 GB | 3 GB | 古い指数ファイルをアーカイブ |
| PostgreSQL データディレクトリ | DB データ | 5 GB | 10 GB | 古い scrape_runs レコードを VACUUM |
| ログ (`logs/`) | アプリログ | 500 MB | 1 GB | `rotate_logs` cron が 04:30 JST に実行 |

**ディスク成長速度の目安**:
- `data/cache/`: SLA 3 T-15バンドル実行で 1日あたり最大 50 MB 増加
- PostgreSQL: race_odds_snapshot（Layer 5 時系列）が最大成長源。月 100-200 MB 増加見込み

### 1-4. ネットワーク

| メトリクス | WARN | CRIT | 備考 |
|---|---|---|---|
| 受信帯域（netkeiba スクレイピング） | 10 Mbps 継続 | 50 Mbps 継続 | リクエスト間隔 2.2-4.0 秒で自然に制限 |
| 送受信エラー率 | ≥ 1% | ≥ 5% | IP ブロック検知の指標 |
| 外向き GCS 転送量 | — | — | GCS 転送費は AREA-05 に計上 |

---

## 2. プロセス死活監視

### 2-1. Watchdog（3分ごと自動確認）

Watchdog（`server_watchdog`、cron `*/3 * * * *`）が以下を監視し、停止時は自動再起動する。

| プロセス | 監視方法 | 自動再起動 | 再起動失敗時のアラート |
|---|---|---|---|
| Flask API | `/api/v1/health` HTTP GET | ○ | CRIT アラート |
| MLflow keiba_lgbm（port 5010） | HTTP GET | ○ | WARN アラート |
| MLflow tracking_difficulty（5001） | HTTP GET | ○ | WARN アラート |
| MLflow final_odds（5003） | HTTP GET | ○ | WARN アラート |
| MLflow pace_predictor（5004） | HTTP GET | ○ | WARN アラート |
| PostgreSQL | `pg_isready` | △（`systemctl start`） | CRIT アラート |
| Redis | `redis-cli ping` | △（`systemctl start`） | WARN アラート |

### 2-2. Watchdog が検出しない障害

以下は Watchdog の監視外のため別途 cron またはログアラートで対応する。

| 障害パターン | 検知方法 | 対処 |
|---|---|---|
| スクレイパーのサイレント失敗（HTTPエラーなしに空データ取得） | `scrape_runs` テーブルの SUCCESS 率監視 | 週次レポートで確認 |
| PostgreSQL スロークエリ（≥ 5秒） | `pg_stat_statements` のログ | インデックス追加・クエリ見直し |
| GCS 書き込み失敗 | `scrape_runs.status = 'FAILED'` 連続検知 | 手動 GCS 認証確認 |
| Redis OOM（`maxmemory` 超過） | Redis の `WARN OUT OF MEMORY` ログ | `maxmemory-policy allkeys-lru` が蒸発対象を削除 |

---

## 3. アラート設計

### 3-1. 通知チャネル（未決定事項 OP-4、AREA-04 §9 参照）

現時点ではアラート通知チャネルは未決定（Slack / メール / PagerDuty のいずれか）。
以下のトリガーを通知対象と定める。

| 重大度 | トリガー条件 | 期待対応時間 |
|---|---|---|
| CRIT | メモリ使用率 ≥ 90%、Flask API 再起動失敗、PostgreSQL 停止 | 即時（< 15分） |
| WARN | メモリ使用率 ≥ 80%、任意 MLflow プロセス停止・再起動、scrape_runs FAILED 率 ≥ 5% | 当日中（< 4時間） |
| INFO | Watchdog が自動再起動を実施、ディスク使用率 ≥ 70% | 翌日までに確認 |

### 3-2. ログ設計

| ログファイル | 対象プロセス | ローテーション | 保持期間 |
|---|---|---|---|
| `logs/api.log` | Flask API | 日次 | 30日 |
| `logs/scraper.log` | スクレイパー全般 | 日次 | 14日 |
| `logs/watchdog.log` | Watchdog | 日次 | 7日 |
| `logs/cron.log` | Cron ジョブ全般 | 日次 | 14日 |
| `logs/mlflow/` | MLflow サーバー×4 | 週次 | 30日 |

ローテーション cron: `30 19 * * *`（JST 04:30）、削除: `0 3 * * *`（JST 12:00）

---

## 4. メトリクス収集実装（ツール未決定）

> **未決定事項 OP-3**: 監視基盤ツール（Prometheus / Grafana / Sentry 等）は AREA-04 §9 参照。
> 以下は収集すべきメトリクスの定義。ツール選定後に実装する。

### 4-1. 収集対象メトリクス

```
# システム系（10秒〜1分間隔）
node_cpu_usage_percent
node_memory_used_bytes
node_memory_available_bytes
node_disk_read_bytes_total
node_disk_write_bytes_total
node_filesystem_used_bytes{path="/data/cache"}
node_filesystem_used_bytes{path="/data/queue"}
node_network_receive_errors_total
node_network_transmit_errors_total

# アプリ系（1分間隔）
keiba_api_response_time_p50_ms
keiba_api_response_time_p99_ms
keiba_api_error_rate_percent
keiba_scrape_success_rate_daily
keiba_scrape_runs_failed_total
keiba_prediction_cache_hit_rate
keiba_mlflow_process_up{model="keiba_lgbm"}
keiba_mlflow_process_up{model="tracking_difficulty"}
keiba_mlflow_process_up{model="final_odds"}
keiba_mlflow_process_up{model="pace_predictor"}

# DB 系（5分間隔）
pg_active_connections
pg_slow_queries_count{threshold="5s"}
pg_table_size_bytes{table="race_odds_snapshot"}
pg_table_size_bytes{table="scrape_runs"}
redis_used_memory_bytes
redis_hit_rate_percent
```

### 4-2. SLO 定義（非機能要件）

AREA-01 §5（N- 番号）との対応:

| SLO | 目標値 | メトリクス | 関連要件 |
|---|---|---|---|
| API キャッシュヒット時レスポンス | ≤ 200 ms | `keiba_api_response_time_p99_ms{cache="hit"}` | N-1 |
| API キャッシュミス時レスポンス | ≤ 2,000 ms | `keiba_api_response_time_p99_ms{cache="miss"}` | N-2 |
| スクレイピング成功率 | ≥ 99% / 月 | `keiba_scrape_success_rate_monthly` | N-6 |
| DB 反映遅延 | ≤ 10 分 | `keiba_etl_lag_minutes` | N-7 |
| オッズスナップショット欠損率 | ≤ 1% | `keiba_odds_snapshot_missing_rate` | N-8 |
| 推論バッチ完了（発走3時間前） | 100% | `keiba_inference_batch_on_time_rate` | N-9 |

---

## 5. VPS メモリ制約対応（ConoHa VPS 2GB）

### 5-1. プロセス優先順位（OOM Kill 対象の順序）

メモリ逼迫時の停止優先順位（低い番号 = 先に停止対象）:

1. MLflow lap_lstm（PLANNED 状態・未使用）
2. MLflow lap_predictor（PLANNED 状態）
3. MLflow pace_predictor（非レース日は不要）
4. MLflow final_odds（SLA 1 完了後は不要）
5. Redis（一時停止可・再起動後にキャッシュ再構築）
6. MLflow tracking_difficulty
7. MLflow keiba_lgbm（最重要推論モデル・最後まで維持）
8. PostgreSQL（停止=サービス停止）
9. Flask API（停止=サービス停止）

### 5-2. メモリ節約設定

```
# PostgreSQL: /etc/postgresql/*/main/postgresql.conf
shared_buffers = 64MB          # デフォルト 128MB から削減
work_mem = 4MB
maintenance_work_mem = 32MB
max_connections = 20           # デフォルト 100 から削減

# Redis: /etc/redis/redis.conf
maxmemory 128mb
maxmemory-policy allkeys-lru   # LRU でキャッシュ蒸発

# Flask API: gunicorn workers
workers = 2                    # VPS 2GB では 2 workers 推奨
worker_class = sync
timeout = 120
```

---

## 6. 未決定事項

| # | 項目 | 関連 |
|---|---|---|
| MON-1 | 監視基盤ツール選定（Prometheus + Grafana / Sentry / Datadog / 自作スクリプト） | AREA-04 OP-3 |
| MON-2 | アラート通知チャネル（Slack / メール / LINE）と宛先 | AREA-04 OP-4 |
| MON-3 | PostgreSQL `shared_buffers` 最適値（実負荷計測後に調整） | AREA-04 OP-1 |
| MON-4 | `scrape_runs` テーブルの成長監視・VACUUM スケジュール設計 | AREA-06 DM-4 |
