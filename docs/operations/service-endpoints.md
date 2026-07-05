# サービス・エンドポイント一覧

> **最終更新**: 2026-07-04  
> **用途**: ローカル開発・stg/prod 運用時に「何をどの URL / ポートで起動するか」を一覧する。  
> **関連**: [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md) · [environment-stg.md](../environment-stg.md) · [DEC-013 Flask 正](../decisions/DEC-013-api-framework-flask-primary.md) · `scripts/server/` · `.env.example`

## 一括起動

リポジトリルートで `./service_start` を実行すると、ローカル開発向けの HTTP サービスをまとめて起動します（実体: `scripts/server/service_start.sh`）。**既定は `--env dev`**（FastAPI ホットリロード・Flask debug・`next dev`）。

```bash
# パターン A（既定）: dev — Next.js モックのみ :3000（API 不要）
./service_start
./service_start --env dev          # 上記と同じ

# dev 実 API 開発（ホットリロード）
./service_start --env dev --full   # FastAPI :8000 + Flask :5100 + next dev

# 環境プロファイル
./service_start --env stg          # 本番相当 + KEIBA_ENV=stg + meguai-stg トンネル
./service_start --env prod         # 本番相当（FastAPI --prod / next build+start）

# 状態確認
./service_start --status

# その他
./service_start --minimal          # FastAPI + MLflow のみ
./service_start --mock             # Next.js モックのみ
./service_start --with-model-serve # Docker MLflow model serve（:5001 等）
```

| プロファイル | 用途 | FastAPI | Flask | Next.js | MLflow（ローカル） | tcpexposer |
|---|---|---|---|---|---|---|
| `dev`（既定） | モック UI のみ | — | — | mock `:3000` | オフ | `meguai-dev` → :3000 |
| `dev --full` | ホットリロード開発 | reload | `:5100` debug | `npm run dev` | オフ | `meguai-dev` → :3000 |
| `stg` | 本 PC・本番相当 | `--prod` | `:5000` | `dev` `:3001` 実 API | 既定オフ | `meguai-stg` → :3001 |
| `prod` | VPS 想定（stg クローン） | `--prod` | `:5000` | `build+start` `:3001` | 既定オフ | （将来） |

ポート・URL の上書き: 環境変数、または `scripts/server/service_start.local.env`（`.example` をコピー）。

| 起動後 URL（dev モック） | サービス |
|---|---|
| `http://127.0.0.1:3000` | Next.js モック（`NEXT_PUBLIC_MOCK=true`） |
| `https://meguai-dev.tcpexposer.com/` | 上記の外部公開 |

| 起動後 URL（stg / prod） | サービス |
|---|---|
| `http://127.0.0.1:8000` | FastAPI（レガシー UI / API） |
| `http://127.0.0.1:5000` | Flask `/api/v1` |
| `http://127.0.0.1:3001` | Next.js（`KEIBA_API_URL=http://127.0.0.1:5000`） |
| `https://meguai-stg.tcpexposer.com/` | stg 外部公開（Next.js :3001） |

PostgreSQL（`:5432`）と Redis（`:6379`）はスクリプト対象外です。未起動の場合は警告を出します。ログは `logs/` に出力されます。

**tcpexposer トンネル**: ワークスペース起動時に `tunnel_tcpexposer.sh autostart-all` で **dev**（`meguai-dev` → `:3000`）と **stg**（`meguai-stg` → `:3001`）を自動接続。無効化: `KEIBA_AUTO_TCPEXPOSER=0`。手動: `./scripts/server/tunnel_tcpexposer.sh stg check` / `dev stop` / `stg background`。

**環境の役割**: **dev** = モック UI のみ。**stg** = 本 PC 上の本番相当（実 API・GCS・DB）。**prod**（将来）= stg 構成を VPS へクローン。

---

## 状態確認（クイック）

```bash
# Watchdog 経由（FastAPI :8000 + MLflow Tracking :5000）
./scripts/server/server_watchdog.sh --status

# 主要 HTTP ヘルス
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool   # FastAPI（レガシー UI/API）
curl -s http://127.0.0.1:5000/api/v1/health | python3 -m json.tool # Flask（仕様準拠 /api/v1）

# リッスン中ポート
ss -tlnp | grep -E ':(3000|3001|5000|5001|5010|5432|6379|8000)\s'
```

| 確認項目 | 期待 |
|---|---|
| FastAPI 稼働 | `GET http://127.0.0.1:8000/api/health` → HTTP 200 |
| Flask 稼働 | `GET http://127.0.0.1:5000/api/v1/health` → `{"status":"ok",...}` |
| Next.js 稼働 | ブラウザで `http://127.0.0.1:3001/`（stg）または `:3000`（dev モック）が表示される |
| MLflow Tracking | `GET http://127.0.0.1:5000/health` または experiments API が 200 |
| PostgreSQL | `DATABASE_URL` で接続可能 |
| Redis | `REDIS_URL` で PING 応答 |

---

## アーキテクチャ概要

DEC-013 により **Flask `/api/v1`（:5000）が仕様上の正**です。  
**FastAPI（:8000）** は Jinja2 UI・管理画面・レガシー `/api/*` を担い、段階廃止予定です。

```
ブラウザ
  ├─ http://127.0.0.1:3001     Next.js stg（frontend）
  ├─ http://127.0.0.1:3000     Next.js dev モック
  │     └─ /api/v1/* ─rewrite─► Flask :5000（KEIBA_API_URL）
  │
  ├─ http://127.0.0.1:8000     FastAPI（UI + レガシー REST）
  └─ http://127.0.0.1:5000     Flask /api/v1（仕様準拠 REST）

推論・学習
  ├─ MLflow Tracking     :5000（※ Flask とポート競合に注意）
  └─ MLflow Model Serve  :5001〜:5006, :5010

データ基盤
  ├─ PostgreSQL          :5432
  └─ Redis               :6379
```

> **ポート競合**: MLflow Tracking の既定ポート（`5000`）と Flask API（`FLASK_PORT=5000`）は同じです。  
> 同時起動する場合は **Flask を `FLASK_PORT=5100` 等に変更**するか、MLflow を Docker 内のみに閉じ、ホスト側 Flask を :5000 にします。  
> Watchdog（`server_watchdog.sh`）は **FastAPI :8000 + MLflow :5000** の組み合わせを監視します（Flask は対象外）。

---

## 1. HTTP サービス

### 1-1. FastAPI — レガシー Web + 管理 API（既定メイン）

| 項目 | 値 |
|---|---|
| **URL（ローカル）** | `http://127.0.0.1:8000` |
| **バインド** | `0.0.0.0:8000`（`PORT` 環境変数で変更可） |
| **エントリ** | `python main.py` / `python main.py --port 9000` |
| **本番相当** | `python main.py --prod --workers 4` |
| **開発（reload 限定）** | `scripts/server/run_uvicorn_dev.sh` |
| **再起動** | `scripts/server/restart_server.sh` / `restart_server.sh dev` |
| **ヘルス** | `GET /api/health` |
| **Watchdog 監視** | ✅（`server_watchdog.sh`） |

**主な画面・API（抜粋）**

| パス | 説明 |
|---|---|
| `/` | ダッシュボード（Jinja2） |
| `/login` | 開発者ログイン |
| `/monitor` | スクレイプ・カバレッジ監視 |
| `/cron-jobs` | 定期実行状態（開発者） |
| `/data-viewer` | GCS / ローカルデータ閲覧 |
| `/api/health` | ヘルス・環境情報（`keiba_env`, `gcs_bucket` 等） |
| `/api/auth/status` | 認証状態 |
| `/api/scrape-queue/*` | スクレイプキュー操作 |
| `/api/race/{race_id}/bundle` | レース JSON バンドル |
| `/api/pedigree-race-stats/query` | 種牡馬成績（レガシー） |
| `/docs` | OpenAPI（FastAPI 自動生成） |

stg 表示: `.env` に `KEIBA_ENV=stg` → レスポンスヘッダ `X-Keiba-Env: stg`。詳細は [environment-stg.md](../environment-stg.md)。

---

### 1-2. Flask — 仕様準拠 REST API（`/api/v1`）

| 項目 | 値 |
|---|---|
| **URL（ローカル）** | `http://127.0.0.1:5000` |
| **バインド** | `0.0.0.0:${FLASK_PORT:-5000}` |
| **起動** | `python main.py --flask-api` |
| **シェル** | `scripts/server/start_flask_api.sh` |
| **ヘルス** | `GET /api/v1/health` |
| **Watchdog 監視** | ❌（手動起動） |

**MASTER §4-3 エンドポイント（抜粋）**

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/api/v1/races` | レース一覧（`?date=YYYYMMDD`） |
| GET | `/api/v1/races/{race_id}` | レース詳細・出馬表 |
| GET | `/api/v1/races/{race_id}/entries` | 出走馬一覧 |
| GET | `/api/v1/races/{race_id}/results` | 着順・ラップ |
| GET | `/api/v1/races/{race_id}/predictions` | AI 予測 T-1〜T-9（+ pace 統合 JSON） |
| GET | `/api/v1/races/{race_id}/predictions/laps` | ラップ予測 T-10/T-11 |
| GET | `/api/v1/races/{race_id}/tracking-difficulty` | 追走難易度 |
| GET | `/api/v1/races/{race_id}/final-odds` | 最終オッズ予測 |
| GET | `/api/v1/horse/{horse_id}/growth-curve` | 成長曲線 |
| GET | `/api/v1/track-speed/day` | 馬場速度指数（`?date=&venue=`） |
| GET | `/api/v1/pedigree-race-stats/query` | 種牡馬成績 |
| GET | `/api/v1/bloodline-cluster/horse-aptitude` | 血統クラスタ適性 |
| GET | `/api/v1/pedigree/race-note-3d-v2` | 血統適性マップ |
| POST | `/api/v1/betting/optimize` | Kelly 馬券最適化（**ログイン必須**） |
| POST | `/api/v1/auth/login` | パスワードログイン |
| GET | `/api/v1/admin/health` | 内部向け（127.0.0.1 / VPN のみ） |

**レガシーブリッジ**: `KEIBA_LEGACY_API=http://127.0.0.1:8000` を設定すると、一部分析 API が FastAPI にフォールバックします（`src/api/v1/delegates.py`）。

---

### 1-3. Next.js — フロントエンド（App Router）

| 項目 | 値 |
|---|---|
| **URL（ローカル）** | `http://127.0.0.1:3001`（stg） / `http://127.0.0.1:3000`（dev モック） |
| **起動** | `cd frontend && npm run dev` |
| **本番ビルド** | `cd frontend && npm run build && npm start` |
| **API プロキシ** | `next.config.js` — `/api/v1/*` → `KEIBA_API_URL`（**既定 `http://127.0.0.1:5000`**） |
| **モックモード** | `NEXT_PUBLIC_MOCK=true`（API なしで UI 確認） |

**主要ルート（`frontend/app/`）**

| パス | 画面 |
|---|---|
| `/` | ダッシュボード |
| `/login` | ログイン |
| `/races` | レース一覧 |
| `/race/[id]` | レース詳細（4 タブ） |
| `/tracking-difficulty` | 追走難易度 |
| `/growth-curve` | 成長曲線 |
| `/track-speed` | トラックスピード指数 |
| `/race-quality` | レース品質 |
| `/bloodline-cluster` | 血統クラスタ |
| `/bloodline-vector` | 血統ベクトル |
| `/pedigree-map` | 血統マップ |
| `/pedigree-race-stats` | 種牡馬成績 |
| `/note-aptitude-race` | 血統適性マップ |
| `/myostatin` | Myostatin |
| `/betting` | 馬券最適化 |

---

## 2. MLflow（Tracking + Model Serving）

### 2-1. ローカル直接起動（Watchdog / CLI）

| サービス | URL | 起動 |
|---|---|---|
| MLflow Tracking | `http://127.0.0.1:5000` | `server_watchdog.sh` が自動起動 / `mlflow server --port 5000` |
| バックエンド DB | `sqlite:///mlflow/runs/mlflow.db` | — |
| アーティファクト | `mlflow/runs/artifacts/` | — |

### 2-2. Model Serving ポート（`src/pipeline/mlflow/catalog.py`）

| モデルキー | ポート | 環境変数（例） | 状態 |
|---|---|---|---|
| `tracking_difficulty` | **5001** | `KEIBA_MLFLOW_SERVE_TRACKING_DIFFICULTY_URI` | active |
| `finish_order` | **5002** | `KEIBA_MLFLOW_SERVE_FINISH_ORDER_URI` | planned |
| `final_odds` | **5003** | `KEIBA_MLFLOW_SERVE_FINAL_ODDS_URI` | active |
| `pace_predictor` | **5004** | `KEIBA_MLFLOW_SERVE_PACE_URI` | active |
| `lap_predictor` | **5005** | `KEIBA_MLFLOW_SERVE_LAP_PREDICTOR_URI` | planned |
| `lap_lstm` | **5006** | `KEIBA_MLFLOW_SERVE_LAP_LSTM_URI` | planned（Phase 4） |
| `keiba_lgbm` | **5010** | `KEIBA_MLFLOW_SERVE_KEIBA_LGBM_URI` | active |

未起動時はローカル Booster / ヒューリスティックにフォールバック（`.env.example` 参照）。

### 2-3. Docker Compose（`mlflow/server/docker-compose.yml`）

```bash
cd mlflow/server
docker compose up -d mlflow mlflow-serve-tracking   # 基本
docker compose --profile all-models up -d             # 全 serve
```

| サービス | ホストポート | 説明 |
|---|---|---|
| `mlflow` | **5000** | Tracking UI + API |
| `mlflow-serve-tracking` | **5001** | 追走難度モデル |
| `mlflow-serve-finish-order` | **5002** | profile: `finish-order` |
| `mlflow-serve-final-odds` | **5003** | profile: `final-odds` |
| `mlflow-serve-keiba-lgbm` | **5010** | profile: `keiba-lgbm` |
| `keiba-api` | **8000** | FastAPI コンテナ |
| `nginx` | **80**, **443** | MLflow UI（`/`）+ Keiba API（`/keiba/` → 8000） |

Nginx 経由の Keiba API 例: `http://127.0.0.1/keiba/api/health`

---

## 3. データ基盤（TCP・非 HTTP）

| サービス | 接続先（既定） | 環境変数 | 用途 |
|---|---|---|---|
| **PostgreSQL** | `localhost:5432` | `DATABASE_URL=postgresql+psycopg://keiba:keiba@localhost:5432/keiba` | Layer 1〜5・predictions |
| **Redis** | `localhost:6379/0` | `REDIS_URL=redis://localhost:6379/0` | 予測キャッシュ L2/L3/L4 |
| **GCS** | `gs://${GCS_BUCKET}/chuou/...` | `GCS_BUCKET` 他 GCS_* | 生データ SSoT（HTTP ではない） |

マイグレーション: `alembic upgrade head`

---

## 4. バッチ・ワーカー（HTTP サーバではない）

| 処理 | 起動例 | 備考 |
|---|---|---|
| 推論バッチ | `scripts/server/run_inference.sh` | T-15 トリガ / 手動 |
| スクレイパー | `python -m src.scraper.run ...` | Cron SLA 0〜6 |
| キューワーカー | FastAPI 内 / `data/queue/scrape_queue.json` | `/api/scrape-queue/*` で制御 |
| Layer 3 スナップショット | `python -m src.db.batch.stats_snapshot` | DB 集計 |
| ETL | `python -m src.scripts.data.etl_ingest_race race_shutuba {race_id}` | GCS → PostgreSQL |
| git pull（手動） | `POST /api/v1/admin/git-pull`（dev/stg UI ナビのボタン） | スクリプト本体: `scripts/cron/git_pull_hourly.sh`。ログ: `logs/git_pull.log`。開発者ログイン必須 |

---

## 5. 推奨ローカル起動セット

> パターン A〜C は `./service_start`（[一括起動](#一括起動)）でも起動できます。

### パターン A — フル開発（UI + 仕様 API + DB）

```bash
# 1. インフラ（別途 Docker 等）
# PostgreSQL :5432, Redis :6379

# 2. Flask 仕様 API（MLflow と競合する場合は FLASK_PORT=5100）
export FLASK_PORT=5100
./scripts/server/start_flask_api.sh &

# 3. FastAPI レガシー UI
./scripts/server/restart_server.sh dev &

# 4. MLflow Tracking
./scripts/server/server_watchdog.sh --mlflow-only   # または docker compose

# 5. フロント
cd frontend
KEIBA_API_URL=http://127.0.0.1:5100 npm run dev
```

### パターン B — VPS / stg 最小（Watchdog 準拠）

```bash
# FastAPI :8000 + MLflow :5000 を watchdog が監視・自動再起動
./scripts/server/server_watchdog.sh
# cron: */3 * * * * .../server_watchdog.sh
```

### パターン C — フロントのみ（モック）

```bash
cd frontend
NEXT_PUBLIC_MOCK=true npm run dev
# → http://127.0.0.1:3000 （API 不要）
```

---

## 6. 環境変数クイックリファレンス

| 変数 | 既定 | 説明 |
|---|---|---|
| `PORT` | `8000` | FastAPI（`main.py`） |
| `FLASK_PORT` | `5000` | Flask `/api/v1` |
| `KEIBA_API_URL` | `http://127.0.0.1:5000` | Next.js → Flask プロキシ先 |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow クライアント |
| `KEIBA_LEGACY_API` | （未設定） | Flask → FastAPI ブリッジ |
| `DATABASE_URL` | — | PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis |
| `NEXT_PUBLIC_MOCK` | `false` | フロント mock データ |

---

## 7. 変更履歴

| 日付 | 内容 |
|---|---|
| 2026-07-04 | 初版作成（FastAPI / Flask / Next.js / MLflow / インフラ / ポート競合注意） |
| 2026-07-04 | `./service_start` 一括起動スクリプト追加 |
| 2026-07-04 | dev/stg 分離: dev モック :3001 / stg 本番相当 :3000 + `meguai-stg` トンネル |
