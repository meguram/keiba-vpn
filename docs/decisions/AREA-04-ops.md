# AREA-04 — 運用最適化要件（Cron SLA / プロセス分離 / Circuit Breaker / 監視・アラート / デプロイ / ロールバック）
**Status**: FINAL | **Last Updated**: 2026-07-06 | **Consolidates**: DEC-001（統合済み）, TASK-049（統合済み）

---

## 1. 概要

本仕様書は keiba-vpn プロジェクトの運用最適化要件を定義する。Cron ジョブ SLA・プロセス分離・Circuit Breaker・監視・アラート・デプロイ・ロールバックを対象とする。

ConoHa VPS 2GB 構成を前提に、UI の見た目を一切変えずに**パフォーマンス・可用性・運用コスト**の3軸を最大化する。フロントエンドのキャッシュ最適化・バンドル削減、バックエンドの N+1 解消・BFF 集約、インフラの OOM 防止設計の3方向で同時進行する。

---

## 2. Cron スケジュール・SLA（実装準拠）

システムタイムゾーン: UTC。cron 記述は UTC。実行スクリプトは `TZ=Asia/Tokyo` を付与して JST で動作。
実装: `scripts/cron/setup_all_cron.sh`（マスター） + `src/scraper/auto_scrape.py`（タスク定義）

### 2-0. スクレイピング対象カテゴリ × タスク対応表

| ストレージカテゴリ | 書込タスク | タイミング（JST） | 備考 |
|---|---|---|---|
| `race_lists` | `daily-race-lists` | 07:00・17:00 毎日 | 今日〜14日先の全開催日 |
| `jra_cushion` | `jra-baba-morning` | 05:00-08:50 毎10分（開催日のみ） | クッション値・含水率 |
| `race_shutuba` | `raceday-eve`（主）・`raceday-runner`（FB） | 18:00 前日 / T-15分 | 出馬表（枠順・騎手確定版） |
| `race_shutuba_past` | `raceday-eve` | 18:00 前日 | 馬柱（過去5走） |
| `race_oikiri` | `raceday-eve`・`backfill-full` | 18:00 前日 / 深夜 | 追い切りタイム |
| `horse_training` | `raceday-eve` | 18:00 前日 | 調教師コメント |
| `race_detail` | `raceday-runner`（T-15） | T-15分 各R | 出走確定情報 |
| `race_odds` | `raceday-runner`（T-15）・`backfill-full` | T-15分 / 深夜 | 単勝・馬連オッズ |
| `race_paddock` | `raceday-runner`（T-15） | T-15分 各R | パドックコメント |
| `race_barometer` | `raceday-runner`（T-15）・`backfill-full` | T-15分 / 深夜 | レースバロメーター |
| `race_trainer_comment` | `raceday-runner`（T-15） | T-15分 各R | 調教師コメント（当日） |
| `race_result_on_time` | `raceday-result-runner`（T+15） | T+15分 各R | 速報結果 |
| `race_result` | `raceday-evening`・`weekly-update`・`backfill-fast` | 17:30 / 17:30金 / 深夜 | 確定着順・タイム |
| `race_pair_odds` | `raceday-evening`・`weekly-update` | 17:30 | 馬連・3連複確定オッズ |
| `race_index` | `raceday-evening`・`weekly-update`・`backfill-full` | 17:30 / 17:30金 / 深夜 | 速度指数・偏差値 |
| `horse_result` | `weekly-update`・`backfill-horse` | 17:30 金曜 / 06:00 毎日 | 馬個別戦績 |
| `horse_name_index` | `horse-name-index` | 18:00 金曜 | 馬名→horse_id マッピング |
| `growth_curve` | `horse-name-index` | 18:00 金曜 | 成長曲線（calculated_data） |
| 騎手・調教師統計 | `update_jt_stats` | 05:30 毎日 | `data/features/jockey_trainer_stats/` |

---

### 2-1. 常時監視

| SLA | cron（UTC） | JST | タスク | 内容 |
|---|---|---|---|---|
| Watchdog | `*/3 * * * *` | 常時 3分ごと | `server_watchdog` | API + MLflow プロセス死活監視・自動再起動。`@reboot` にも登録（起動後 15秒待機） |
| 構造監視 | 毎日 JST 06:00 | 06:00 | `structure-scheduler` | ページ構造変更検知、`versions.json` 更新 |
| ログ保全 | `30 19 * * *` | 04:30 | `rotate_logs` | ログローテーション。JST 12:00 に保全期間超ファイル削除 |
| キャッシュ保守 | 86,400 秒ごと（起動直後+以降毎日） | — | `disk_cache_cleanup` | `data/cache/` の古ファイル削除（週次アクセス < 2回は対象） |
| キュー保守 | 3,600 秒ごと | — | `queue_hourly_maintain` | 失敗ジョブを待機状態に戻し・完了レコード削除 |

### 2-2. 定期取得（毎日）

| SLA | cron（UTC） | JST | タスク名 | 実行内容 | 非開催日動作 |
|---|---|---|---|---|---|
| SLA 0a | `0 22 * * *` | 07:00 | `daily-race-lists` | 今日〜末尾の全開催日 race_lists 取得・更新 | 実行（全曜日） |
| SLA 0b | `0 8 * * *` | 17:00 | `daily-race-lists` | 同上（夕方更新） | 実行（全曜日） |
| JT統計 | `30 20 * * *` | 05:30 | `update_jt_stats` | 騎手・調教師統計再生成 | 実行（全曜日） |
| SLA 1 | `0 9 * * *` | 18:00 | `raceday-eve` | **翌日が開催日のみ**: race_shutuba + race_shutuba_past + race_oikiri + horse_training → 追走難度・最終オッズ precompute | 即終了 |
| SLA 2 | `*/10 20-23 * * *` | 05:00-08:50 毎10分 | `jra-baba-morning` | jra_cushion（クッション値・含水率）ポーリング | 開催日のみ実取得 |

### 2-3. 開催日リアルタイム

| SLA | cron（UTC） | JST | タスク名 | 実行内容 |
|---|---|---|---|---|
| SLA 3 | `30 22 * * *` | 07:30 | `raceday-runner` | 開催日常駐。各R 発走 **T-15分** に T-15バンドル（race_detail + race_odds + race_paddock + race_barometer + race_trainer_comment + JRA馬場ライブ）→ AI 予測トリガ |
| SLA 4 | `30 22 * * *` | 07:30 | `raceday-result-runner` | 開催日常駐。各R 発走 **T+15分** に race_result_on_time（速報結果）取得 |
| SLA 5 | `30 8 * * *` | 17:30 | `raceday-evening` | 確定結果（race_result）+ race_pair_odds + race_index → 馬場速度指数計算トリガ |

### 2-4. 週次

| SLA | cron（UTC） | JST | タスク名 | 実行内容 |
|---|---|---|---|---|
| SLA 6 | `0 9 * * 5` | 18:00 金曜 | `weekly-update` | **直近10日**の開催日: `race_result`（確定）・`race_pair_odds`・`race_index`・`horse_result` 一括更新 → 完了後 `date_coverage` / めぐ指数→GCS / PG 増分同期 |
| — | `30 9 * * 5` | 18:30 金曜 | `horse-name-index` | 馬名リスト（`horse_name_index`）+ 成長曲線（`growth_curve`）→ `calculated_data` 一括更新 |

### 2-5. バックフィル（夜間 / 年度別）

| cron（UTC） | JST | 対象年 | フェーズ | 最大件数 |
|---|---|---|---|---|
| `0 15 * * *` | 00:00 | 2026 | fast（race_result + race_shutuba） | 7日分 |
| `0 16 * * *` | 01:00 | 2025 | fast | 5日分 |
| `0 17 * * *` | 02:00 | 2024 | fast | 5日分 |
| `0 18 * * *` | 03:00 | 2023 | fast | 5日分 |
| `0 19 * * *` | 04:00 | 2022 | fast | 5日分 |
| `0 21 * * *` | 06:00 | 全年 | horse（`horse_result` 一括） | 一括 |
| `30 22 * * *` | 07:30 | 2026 | full（`race_index`・`race_odds`・`race_barometer`・`race_oikiri` 等補助データ） | 5日分 |
| `0 23 * * *` | 08:00 | 2025 | full（同上） | 3日分 |
| `0 0 * * *` | 09:00 | 2024 | full（同上） | 3日分 |
| `0 17 * * 1,4` | 02:00 月木 | 2021 | fast（週2回） | 5日分 |
| `0 18 * * 2,5` | 03:00 火金 | 2020 | fast（週2回） | 5日分 |

---

## 3. プロセス分離

| プロセス種別 | 役割 | 備考 |
|---|---|---|
| スクレイパープロセス | netkeiba.com データ収集 | グローバル最大同時 4スロット、バースト制限付き |
| スナップショット集計バッチ | `*_stats_snapshot` 生成 | `as_of_race_id` 紐付け、results 収集完了後に起動 |
| 追走難度 precompute | `tracking_difficulty` キャッシュ事前計算 | raceday-eve 完了後に起動（`KEIBA_EVE_PRECOMPUTE_TRACKING=1`） |
| 最終オッズ precompute | `final_odds_prediction` キャッシュ事前計算 | raceday-eve 完了後に起動（`KEIBA_EVE_PRECOMPUTE_FINAL_ODDS=1`） |
| AI 予測トリガ | T-15バンドル完了後に Stage 1→2 推論実行 | `KEIBA_PRE_RACE_PREDICT_ENABLED=1` 時に起動 |
| 推論バッチプロセス | Stage 1 → Stage 2 順次推論・結果書込 | 発走 3 時間前までに完了（N-9）; systemd MemoryMax=512M, CPUQuota=60% |
| API サーバー（Flask / Gunicorn） | REST API 提供・Redis キャッシュ参照 | キャッシュヒット ≤ 200 ms（N-1）、ミス ≤ 2,000 ms（N-2）; systemd MemoryMax=384M |
| DDL マイグレーションプロセス | Alembic スキーマバージョン管理 | デプロイ時に独立実行（N-11） |
| ETL 集約バッチ | `aggregate_predictions.py` による馬単位ファイル → manifest.json → full.json 生成 | Phase 2 以降。`scrape_runs` テーブルに完了記録 |

**シングル IP 環境制約**: netkeiba.com スクレイパーはグローバルスロット 4 以内を厳守し、IP ブロックを防止する。

---

## 4. VPS メモリバジェット

<!-- TASK-049 決定により具体的な割り当て値を追記。旧「要対応」記述を置き換え -->

ConoHa VPS 2GB 制約下での systemd MemoryMax 設定を以下の通り確定する。

| プロセス | systemd サービス名 | MemoryMax | CPUQuota |
|---|---|---|---|
| 推論バッチ | `keiba-infer.service` | 512MB | 60% |
| Web サーバー（Gunicorn） | `keiba-web.service` | 384MB | — |
| Redis | `redis.conf` maxmemory | 128MB | — |
| 全プロセス合計 | — | ≤ 1.5GB | — |

> **注**: 残余 512MB は PostgreSQL・OS・その他プロセス向け。LightGBM / LSTM モデルロード時のメモリ見積もりは推論バッチ 512MB 上限内に収めること。
>
> **要対応 (OP-1)**: 各モデルロード時の実測メモリ使用量を計測し、512MB 制約との適合を確認すること（後続 DEC で報告）。

---

## 5. Circuit Breaker

DEC-001 には Circuit Breaker パターンの明示的な記述は存在しないが、以下のリトライ・バックオフ設定がその代替機能を一部担っている。

### 5-1. スクレイパーのリトライ制御

```
netkeiba.com:
  リクエスト間隔: 2.2〜4.0 秒（ランダム + ガウスジッター）
  バースト制限: 14 req ごとに 6〜12 秒クールダウン
  セッションクールダウン: 60 req ごとに 22〜40 秒
  セッションリフレッシュ: 150 req ごとに TLS/Cookie 再構築
  グローバル最大同時スロット: 4
  UA ローテーション: Chrome/Firefox/Edge × Windows/Mac/Linux 8種
  429/503 バックオフ: 初期 5s・係数 2.5・最大 3 リトライ
  403: UA 即時ローテーション後リトライ
  日次上限: 5,000 req / セッション上限: 500 req
```

### 5-2. 結果スクレイパーのリトライ

```yaml
results:
  trigger: "発走予定時刻 + 35分"
  retry: "5分間隔 × 最大6回"  # 合計最大 30 分間リトライ
```

### 5-3. 要対応事項

> 後続 DEC で以下を確定する必要がある。
> - Circuit Breaker ライブラリの選定（例: `pybreaker`・`tenacity`）
> - 閾値定義: 連続失敗 N 回でオープン状態遷移、クールダウン時間
> - Circuit Breaker 適用対象: netkeiba.com HTTP クライアント・Redis・PostgreSQL 接続

---

## 6. 監視・アラート

### 6-1. スクレイピング成功率監視（N-6、R-2）

| 項目 | 目標値 | アラート条件 |
|---|---|---|
| スクレイピング成功率 | ≥ 99% / 月 | 週次で閾値以下になった場合に通知（R-2 対策） |
| DB 反映遅延 | ≤ 10 分 | 超過時アラート（N-7） |
| オッズスナップショット欠損率（発走前 5 分以内） | ≤ 1% | 超過時アラート（N-8） |

### 6-2. 実行ログ管理（F-5）

```sql
-- scrape_runs テーブル（スクレイプ実行ログ）
-- カラム: target_type, status, retry_count
-- 監視基盤はこのテーブルを参照して成功率・失敗率を集計すること
-- Phase 2 以降: ETL 集約ステップ（aggregate_predictions.py）の完了もこのテーブルに記録する
```

### 6-3. API パフォーマンス監視

<!-- TASK-049 により BFF エンドポイント追加に伴い項目を追記 -->

| 項目 | SLO |
|---|---|
| キャッシュヒット時レスポンスタイム | ≤ 200 ms |
| キャッシュミス時レスポンスタイム | ≤ 2,000 ms |
| BFF エンドポイント `GET /api/v1/races/{race_id}/full` レスポンスタイム (p95) | ≤ 300 ms |
| Redis キャッシュヒット率（`predictions:{race_id}:full`） | ≥ 80% |
| Gunicorn OOM Kill 発生率 | 0件/月 |

### 6-4. 推論バッチ完了監視

- 発走 3 時間前までに推論バッチが完了していない場合はアラートを発報する。

### 6-5. モデル品質モニタリング（Phase 4 以降）

- 特徴量重要度のモニタリング
- データドリフト検知
- 障害・エラー通知アラートの整備

### 6-6. テンポラルリーク検知

- CI パイプラインにおいてテストデータ時系列分割によるリーク検知テストを自動実行する。
- `as_of_race_id` 紐付けの単体テストを必須化する（F-3 実装時）。

### 6-7. フロントエンド性能監視（TASK-049 追加）

| 項目 | 目標値 |
|---|---|
| 静的データ系 Flask への無駄リクエスト削減率 | ≥ 60% 削減 |
| 初回 JS バンドルサイズ（チャートライブラリ分離後） | ≥ 30% 削減 |
| モバイル LCP（dynamic import 適用後） | ≤ 2.5s |
| GCS リクエスト数（full.json 一本化後） | N回 → 1回/レース |

---

## 7. デプロイ

### 7-1. スキーママイグレーション

- DDL 変更は **Alembic** でバージョン管理し、全スキーマ変更をマイグレーションファイルとして記録する。
- デプロイ時はマイグレーションプロセスを API サーバー起動前に独立実行する。

### 7-2. デプロイ順序（依存関係）

```
1. DDL マイグレーション実行（Alembic）
2. スクレイパープロセス起動
3. API サーバー（Flask / Gunicorn）起動
4. 推論バッチプロセス起動
5. Cron ジョブ有効化
```

---

## 8. Docker 構成

ConoHa VPS 環境での Docker Compose 構成を以下の通り定義する。

### 8-1. 使用イメージバージョン

| サービス | ベースイメージ | 備考 |
|---|---|---|
| backend | `python:3.11-slim` | Python バージョンは **3.11** に統一 |
| redis | `redis:7-alpine` | Redis 7 系（セキュリティ・パフォーマンス向上） |
| nginx | `nginx:stable-alpine` | リバースプロキシ |

> **注**: Python バージョンは `3.11` に統一する。`3.10` 以前の記述が他ファイルに残っている場合は `3.11` に修正すること。

### 8-2. Python ライブラリバージョン

| ライブラリ | バージョン | 備考 |
|---|---|---|
| lightgbm | `4.x`（4.0 以上） | LightGBM 4 系を使用。3.x 系の記述は `>=4.0,<5.0` に更新 |
| Flask | `2.x` / `3.x` | 既存 flask_app.py に準拠 |

### 8-3. nginx リバースプロキシ設定

Flask（Gunicorn）は **5000 番ポート**で起動する。nginx upstream の `proxy_pass` は必ず `http://backend:5000` を指定すること。

```nginx
upstream backend_app {
    server backend:5000;
}

server {
    listen 80;

    location /api/ {
        proxy_pass http://backend:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location / {
        proxy_pass http://backend:5000;
        proxy_set_header Host $host;
    }
}
```

> **注意（P1-4）**: `proxy_pass http://backend:8000` は誤り。Flask は 8000 番ではなく **5000 番**で起動するため、8000 のままにすると 502 Bad Gateway が発生する。必ず `backend:5000` を使用すること。

### 8-4. Docker Compose サービス定義（抜粋）

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    # Flask / Gunicorn は 5000 番ポートで起動
    expose:
      - "5000"
    environment:
      - FLASK_ENV=production

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru

  nginx:
    image: nginx:stable-alpine
    ports:
      - "80:80"
    depends_on:
      - backend
```

### 8-5. Dockerfile（backend）

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "src.api.flask_app:app"]
```

> `requirements.txt` に `lightgbm>=4.0,<5.0` を記述すること（3.x 系は非対応）。