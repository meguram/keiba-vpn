# AREA-09: 開発環境・インフラ設計

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂

---

## 1. 技術スタック全体

| レイヤ | 技術 | バージョン |
|--------|------|------------|
| **API サーバ** | FastAPI + Uvicorn | ≥0.104 / ≥0.24 |
| **テンプレート** | Jinja2 | ≥3.1 |
| **HTTP クライアント** | requests / BeautifulSoup4 | ≥2.31 / ≥4.12 |
| **データ処理** | pandas / pyarrow / numpy | ≥2.0 / ≥14.0 / ≥1.24 |
| **ML** | LightGBM / scikit-learn | ≥4.0 / ≥1.3 |
| **ML 管理** | MLflow | ≥2.12 |
| **スクレイピング** | requests / Playwright | ≥2.31 / ≥1.40 |
| **GCS** | google-cloud-storage | ≥2.0 |
| **スキーマ検証** | jsonschema / Pandera | ≥4.20 |
| **日本語処理** | pykakasi | ≥2.3 |
| **VPN** | WireGuard | — |
| **Web サーバ** | Nginx（リバースプロキシ） | — |
| **コンテナ** | Docker（MLflow 用のみ） | — |
| **OS** | Ubuntu / Debian (WSL2 上) | — |
| **Python** | 3.11 | — |

---

## 2. インフラ構成

```
[インターネット]
      │
      ↓ WireGuard VPN
[ConoHa VPS 2GB / 100GB SSD]
      │
      ├── Nginx (80/443)
      │     └→ localhost:8000 (FastAPI)
      │
      ├── python main.py --port 8000 (FastAPI)
      │     ├── ScrapeWorker (background thread)
      │     └── hourly_maintenance (background thread)
      │
      ├── MLflow server (Docker Compose)
      │     ├── mlflow-server (port 5000)
      │     └── nginx (port 5001-5010 → MLflow serve)
      │
      └── cron (vixie-cron)
            └── scripts/cron/ の各タスク
```

---

## 3. Python 環境

```bash
# conda 環境（本番・VPS）
conda activate base  # または専用 env
python --version  # Python 3.11.x

# 依存ライブラリのインストール
pip install -r archive/requirements.txt

# または開発環境
pip install -r requirements-dev.txt  # 将来作成予定
```

### `requirements.txt` 主要パッケージ

```text
fastapi>=0.104.0
uvicorn>=0.24.0
jinja2>=3.1.0
httpx>=0.27.0
requests>=2.31.0
beautifulsoup4>=4.12.0
playwright>=1.40.0
pandas>=2.0.0
pyarrow>=14.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0
optuna>=3.0.0
mlflow>=2.12.0
google-cloud-storage>=2.0.0
jsonschema>=4.20.0
pykakasi>=2.3.0
structlog
```

---

## 4. 環境変数（`.env`）

`.env.example` をコピーして `.env` を作成する。

```bash
cp .env.example .env
```

### 主要変数

```bash
# GCS
GCS_BUCKET=chuou
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# スクレイピング制御
KEIBA_SCHEMA_STRICT=1           # スキーマ検証（0 で緩和）
KEIBA_AUTO_SCRAPE_USE_QUEUE=1   # netkeiba はキュー経由
SCRAPE_QUEUE_PARALLEL=4         # キュー並列数

# 機能フラグ
KEIBA_SYNC_PED_TBL_ON_SHUTUBA=0 # 出馬表保存時に ped_tbl 自動生成
KEIBA_PED_TBL_MERGE_GEN5=1      # Gen5 血統を ped_tbl に接木
KEIBA_PRE_RACE_PREDICT_ENABLED=0 # T-15 自動推論

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# 認証（API ログイン）
KEIBA_ADMIN_PASSWORD=your_password_here
```

---

## 5. ローカル開発環境のセットアップ

```bash
# 1. リポジトリクローン
git clone https://github.com/meguram/keiba-vpn.git
cd keiba-vpn

# 2. Python 環境
pip install -r archive/requirements.txt  # アーカイブの requirements を参照

# 3. 環境変数
cp .env.example .env
# .env を編集して GCS 認証・各種設定を入力

# 4. データディレクトリ作成
mkdir -p data/{local/{features,cache,meta},page_reference/{race_lists,race_day_schedule},queue,predictions}

# 5. API サーバ起動
python main.py --port 8000

# 6. ブラウザで確認
open http://localhost:8000
open http://localhost:8000/monitor
```

---

## 6. MLflow サーバ（Docker）

```bash
# 起動
cd mlflow/server
docker compose up -d

# 確認
open http://localhost:5000
```

`mlflow/server/` には以下が含まれる:
- `docker-compose.yml`
- `nginx.conf`（MLflow serve ポートのプロキシ）
- `setup.sh`

詳細: `docs/mlflow_platform.md`

---

## 7. コードレイアウト方針

| ディレクトリ | 役割 | 備考 |
|------------|------|------|
| `archive/src/` | 参照用ソースコード | リファクタリング前の実装 |
| `archive/templates/` | Jinja2 テンプレート（参照用） | |
| `archive/scripts/` | cron シェルスクリプト（参照用） | **cron は archive/ を参照中** |
| `archive/config/` | 設定ファイル（参照用） | |
| `data/` | データファイル | `.gitignore` で GCS キャッシュ除外 |
| `docs/` | ドキュメント | 本仕様書群 |
| `mlflow/` | MLflow 関連 | `server/` に Docker Compose |
| `models/` | 学習済みモデル `*.lgb` | `.gitignore` で除外 |
| `logs/` | 運用ログ | `.gitignore` で除外 |

### cron での archive/ 参照

現在 cron は `archive/scripts/cron/` にあるシェルスクリプトと、
`archive/src/scraper/` のコードを参照している（`python -m src.scraper.*` の形式）。

リファクタリング後は `src/` に新コードを配置し、参照先を切り替える。

---

## 8. Git 運用

### ブランチ戦略

- `main`: 本番コード。直接 push は避ける
- `feature/*`: 機能追加
- `fix/*`: バグ修正

### コミットメッセージ規約

```
<type>: <日本語で変更の概要>

type:
  feat:    新機能
  fix:     バグ修正
  refactor: リファクタリング
  docs:    ドキュメント
  test:    テスト
  chore:   その他（依存更新・設定等）
```

### Push（GitHub PAT）

Classic PAT（`repo` スコープ）を使用。Fine-grained PAT は "Contents: Read and write" が必要で、設定が複雑なため非推奨。

```bash
git remote set-url origin https://<TOKEN>@github.com/meguram/keiba-vpn.git
```

---

## 9. WSL2 固有の注意事項

（詳細は `AGENTS.md` の「WSL2 と Cursor」セクションを参照）

### `.wslconfig` 推奨設定

```ini
# C:\Users\<ユーザー名>\.wslconfig
[wsl2]
memory=8GB
swap=8GB
processors=4
```

### 重い処理の実行

バックフィルや大量 Parquet 処理は Cursor IDE の外（tmux/Windows ターミナル）で実行し、VPS メモリ競合を避ける。

```bash
# tmux で実行
tmux new-session -d -s backfill
tmux send-keys -t backfill "cd /home/jovyan/work/keiba-vpn && python -m src.scraper.backfill --year 2024 --phase full" Enter
```

---

## 10. 未定義事項（将来 DEC で決定）

| 項目 | 現状 | 将来対応 |
|------|------|---------|
| dev/stg/prod 環境分離 | prod のみ | `.env` の環境変数で切り替え |
| CI/CD | 手動 git pull | GitHub Actions 検討 |
| シークレット管理 | `.env` ファイル | GCP Secret Manager 検討 |
| GPU 要件（LSTM）| 未着手 | Phase 4 で VPS GPU オプション or Cloud Run |
| 型チェック | 未設定 | mypy 導入検討 |
| コードフォーマット | 未設定 | black + isort 導入検討 |
