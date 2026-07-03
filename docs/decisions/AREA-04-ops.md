# AREA-04: 運用・cron・監視設計

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂

---

## 1. プロセス構成

| プロセス | 起動方法 | 役割 |
|--------|--------|------|
| **FastAPI サーバ** | `python main.py --port 8000` | API + UI サーブ |
| **ScrapeWorker** | FastAPI lifespan 内 background thread | キュー消化 |
| **MLflow サーバ** | `mlflow/server/` Docker Compose | モデル管理・推論サーブ |
| **server_watchdog.sh** | cron `*/3 * * * *` | API + MLflow 死活確認 → 自動再起動 |

### watchdog の動作

```bash
# scripts/server/server_watchdog.sh
# 1. FastAPI プロセスを確認
# 2. 停止していれば再起動 (nohup python3 main.py ...)
# 3. MLflow サーバも同様に確認・再起動
# 4. ログ: logs/watchdog.log
```

---

## 2. cron スケジュール

> **管理方法**: `scripts/cron/setup_all_cron.sh` が唯一の cron 管理ソース。
> 手動編集は禁止。変更は `setup_all_cron.sh` を修正して `install` し直すこと。

```bash
bash scripts/cron/setup_all_cron.sh install   # crontab に登録
bash scripts/cron/setup_all_cron.sh show       # 現在の cron を表示
bash scripts/cron/setup_all_cron.sh remove     # keiba-vpn cron を削除
bash scripts/cron/setup_all_cron.sh status     # 実行状態確認
```

> **注意**: 時刻はすべて UTC で crontab に記述する（`CRON_TZ` は Debian vixie-cron で無効）。

### 全 cron エントリ（JST 換算）

| JST 時刻 | UTC | タスク | ログ |
|--------|-----|--------|------|
| 常時 `*/3分` | 常時 | watchdog（API+MLflow 自動再起動） | — |
| 再起動時 | @reboot | watchdog 初回起動 | — |
| 04:30 | 19:30 UTC | ログローテーション | `logs/rotate_logs.log` |
| 05:00-08:50 毎10分 | 20:00-23:50 UTC | `jra-baba-morning` | `logs/jra_baba_morning.log` |
| 05:30 | 20:30 UTC | 騎手・調教師統計更新 | 内蔵 |
| 07:00 | 22:00 UTC | `daily-race-lists`（朝） | `logs/daily_race_lists_am.log` |
| 07:30 | 22:30 UTC | `raceday-runner` | `logs/raceday_runner.log` |
| 07:30 | 22:30 UTC | `raceday-result-runner` | `logs/raceday_result_runner.log` |
| 07:30 | 22:30 UTC | `backfill --year 2026 --phase full --max-dates 5` | `logs/backfill_full_2026.log` |
| 08:00 | 23:00 UTC | `backfill --year 2025 --phase full --max-dates 3` | `logs/backfill_full_2025.log` |
| 09:00 | 00:00 UTC | `backfill --year 2024 --phase full --max-dates 3` | `logs/backfill_full_2024.log` |
| 17:00 | 08:00 UTC | `daily-race-lists`（夕方） | `logs/daily_race_lists_pm.log` |
| 17:30 | 08:30 UTC | `raceday-evening` | `logs/raceday_evening.log` |
| 17:30 金曜 | 08:30 UTC 金曜 | `weekly-update` | `logs/weekly_update.log` |
| 18:00 | 09:00 UTC | `raceday-eve` | `logs/raceday_eve.log` |
| 18:00 金曜 | 09:00 UTC 金曜 | `horse-name-index` | `logs/horse_name_index.log` |
| 00:00 | 15:00 UTC | `backfill --year 2026 --phase fast --max-dates 7` | `logs/backfill_2026.log` |
| 01:00 | 16:00 UTC | `backfill --year 2025 --phase fast --max-dates 5` | `logs/backfill_2025.log` |
| 02:00 | 17:00 UTC | `backfill --year 2024 --phase fast --max-dates 5` | `logs/backfill_2024.log` |
| 03:00 | 18:00 UTC | `backfill --year 2023 --phase fast --max-dates 5` | `logs/backfill_2023.log` |
| 04:00 | 19:00 UTC | `backfill --year 2022 --phase fast --max-dates 5` | `logs/backfill_2022.log` |
| 06:00 | 21:00 UTC | `backfill --phase horse` | `logs/backfill_horse.log` |
| 02:00 月木 | 17:00 UTC 月木 | `backfill --year 2021 --phase fast --max-dates 5` | `logs/backfill_2021.log` |
| 03:00 火金 | 18:00 UTC 火金 | `backfill --year 2020 --phase fast --max-dates 5` | `logs/backfill_2020.log` |

---

## 3. cron タスク詳細

### `raceday-eve`（JST 18:00）

前日夕方。翌日の出馬表・馬柱・追い切りを取得する。

```python
# auto_scrape_queue.py: run_raceday_eve_for_date()
投入タスク: ["race_shutuba", "race_shutuba_past", "race_oikiri", "smartrc"]
smart_skip: False  # 強制再取得
完了後: _eve_precomputes() → 追走難度・想定オッズ precompute
```

**重要な実装注意**:
`_ensure_race_list_date()` は race_list 未存在時:
1. ローカルファイル（`data/page_reference/race_lists/{ymd}.json`）を確認
2. GCS を確認
3. なければ `race_list` ジョブを投入し、最大 5 分ポーリング（キュー全体が空くのを待たない）

これにより、キューが混雑していても raceday-eve がタイムアウトしない。

---

### `raceday-runner`（JST 07:30）

開催当日。発走 T-15 まで常駐し、タスクを投入する。

```python
# auto_scrape_queue.py: task_raceday_runner()
T15_RACE_TASKS = ["race_shutuba", "race_odds", "race_pair_odds",
                  "race_shutuba_past", "race_oikiri", "smartrc"]
処理フロー:
  1. race_list 取得 → 発走時刻を取得
  2. 各レース T-15 になるまで sleep
  3. T-15 到達 → 上記タスクを一括投入
  4. JRA 馬場情報更新
  5. AI 予測トリガ（KEIBA_PRE_RACE_PREDICT_ENABLED=1 の場合）
```

---

### `raceday-result-runner`（JST 07:30）

開催当日。各レース終了後（発走 T+15）に速報結果を取得する。

```python
# 各レース発走+15分後に race_result_on_time を投入
smart_skip: False
```

---

### `raceday-evening`（JST 17:30 毎日 / 週 `weekly-update` と同時）

当日夕方。速報まとめ・オッズ確定版・SmartRC を取得する。

```python
投入タスク: ["race_result_on_time", "race_odds", "race_pair_odds", "smartrc"]
完了後: _trigger_track_speed_for_date() → トラック速度指数計算
```

---

### `weekly-update`（JST 17:30 金曜）

週次更新。確定結果・指数・barometer・馬プロフィールを取得する。

```python
処理フロー:
  ① レース: ["race_result", "race_index", "race_barometer"]
  ② 馬: ["horse_profile"] → horse_result, horse_pedigree_5gen
  ③ 血統: ["horse_pedigree_5gen"] (smart_skip=True)
  ④ 馬名インデックス再構築
```

---

### `backfill`（深夜）

過去データ補完。年度別・フェーズ別で実行する。

```bash
python -m src.scraper.backfill --year YYYY --phase fast|horse|full --max-dates N
```

| フェーズ | 対象 | 主要タスク |
|--------|------|--------|
| `fast` | レース結果・出馬表 | race_result, race_shutuba |
| `horse` | 馬プロフィール | horse_result, horse_pedigree_5gen |
| `full` | 補助データ全体 | race_result_on_time, race_odds, race_index 等 |

---

## 4. ログ管理

### ログファイル

```
logs/
├── server.log              ← FastAPI メインログ
├── watchdog.log            ← watchdog 動作ログ
├── raceday_eve.log         ← raceday-eve 実行ログ
├── raceday_runner.log      ← raceday-runner 実行ログ
├── raceday_result_runner.log
├── raceday_evening.log
├── weekly_update.log
├── daily_race_lists_am.log
├── daily_race_lists_pm.log
├── jra_baba_morning.log
├── horse_name_index.log
├── backfill_*.log          ← 年度別バックフィルログ
├── rotate_logs.log
└── session_*.log           ← セッションログ（7日で削除）
```

### ログローテーション（JST 04:30 毎日）

`scripts/cron/rotate_logs.sh`:
- セッションログ: 7 日以上前を削除
- 追記型ログ: サイズが一定以上で gzip アーカイブ

---

## 5. 監視

### Watchdog（3 分間隔）

```bash
# scripts/server/server_watchdog.sh
- FastAPI プロセス確認 → 停止していれば `nohup python3 main.py ...` で再起動
- MLflow コンテナ確認 → 停止していれば `docker compose up -d`
- 起動待機: 15 秒
```

### 外形監視

- **UptimeRobot**: `GET /api/health` に 5 分間隔 HTTP チェック
- アラート: メール通知（設定は UptimeRobot ダッシュボード）

---

## 6. デプロイ手順

```bash
# 1. リポジトリ更新
cd /home/jovyan/work/keiba-vpn
git pull origin main

# 2. API サーバ再起動
pkill -f "python.*main.py" || true
sleep 2
nohup python3 main.py --port 8000 > logs/server.log 2>&1 &

# 3. cron 更新（変更がある場合）
bash scripts/cron/setup_all_cron.sh install

# 4. MLflow 更新（変更がある場合）
cd mlflow/server
docker compose pull && docker compose up -d
```

### ロールバック

```bash
git checkout HEAD~1
# API サーバ再起動（上記手順）
```

---

## 7. メモリバジェット（ConoHa VPS 2GB）

| コンポーネント | 割当目安 |
|-------------|---------|
| OS + カーネル | 300 MB |
| FastAPI + Uvicorn | 200〜400 MB |
| ScrapeWorker (background) | 200〜300 MB |
| LightGBM Booster (推論時) | 200 MB |
| MLflow (Docker) | 300 MB |
| バックフィル（深夜のみ） | 300〜500 MB |
| バッファ | 残り |

**同時実行の制約**:
- バックフィルとスクレイプワーカーは同時に重い処理が走らないように cron スケジュールを調整済み
- バックフィル深夜（JST 00:00〜09:00）、スクレイプ主活動（JST 07:00〜18:00）

---

## 8. 障害対応 FAQ

### Q1: API が応答しない

```bash
# プロセス確認
ps aux | grep main.py
# 強制再起動
kill -9 <PID>
nohup python3 main.py --port 8000 > logs/server.log 2>&1 &
```

### Q2: キューが詰まっている

```bash
# キュー状態確認
cat data/queue/scrape_queue.json | python3 -c "import json,sys; q=json.load(sys.stdin); print({s: len([j for j in q['jobs'] if j['status']==s]) for s in ['pending','running','failed']})"
# 失敗ジョブの再キュー → POST /api/scrape-queue/failed/requeue
# 強制クリア → POST /api/scrape-queue/stop-and-clear
```

### Q3: crontab が消えた

```bash
bash scripts/cron/setup_all_cron.sh install
crontab -l  # 確認
```

### Q4: WSL2 接続失敗（Cursor IDE）

AGENTS.md の「WSL2 と Cursor」セクションを参照。
主に: `wsl --shutdown` → 再起動 で解消。
