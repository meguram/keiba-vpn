# プロジェクトディレクトリ構成

競馬データ基盤（keiba-vpn）のトップレベルレイアウト。詳細な設計は `docs/html/design/ARCHITECTURE.html`、エージェント向け短縮版は `AGENTS.md`。

## 概要図

```
keiba-vpn/
├── main.py                 # FastAPI エントリ（python main.py）
├── service_start           # 全 HTTP サービス一括起動 → scripts/server/service_start.sh
├── frontend/               # Next.js（一般ユーザー UI、既定 :3000）
├── src/                    # すべての Python アプリケーションコード
├── scripts/                # シェルのみ（server/ cron/ 運用ラッパ）
├── templates/ static/        # FastAPI / Flask 用 Jinja2・CSS
├── notebooks/              # Jupyter（pedigree / feature_engineering / modeling）
├── config/                 # settings.yaml（MLflow モデル登録等）
├── data/                   # 生データ・特徴量・キュー（大容量、多くは .gitignore）
├── docs/                   # 要件・決定・運用・HTML 設計書
├── tests/                  # unittest / pytest
├── mlflow/                 # MLflow サーバ Compose・ローカル runs
├── models/                 # 学習済みモデル配置（pace_predictor 等）
└── alembic/                # DB マイグレーション
```

## `src/` 配下

| パス | 役割 |
|------|------|
| `src/api/` | FastAPI（`app.py`）・Flask（`flask_app.py`） |
| `src/scraper/` | netkeiba / SmartRC / JRA 取得・キュー・保存 |
| `src/pipeline/` | 特徴量・学習・推論・MLflow 連携 |
| `src/research/` | 血統・遺伝子・コース等のリサーチ |
| `src/scripts/` | 運用 Python CLI（下表） |
| `src/utils/` | ロギング・パス・横断ユーティリティ |
| `src/config/` | デプロイプロファイル・データパス解決 |
| `src/db/` | SQLAlchemy / DB 接続 |

### `src/scripts/` のサブ分類

| サブディレクトリ | 用途 | 実行例 |
|------------------|------|--------|
| `scraping/` | キュー投入・バッチスクレイプ・カバレッジ監査 | `python3 -m src.scripts.scraping.batch_scrape` |
| `data/` | GCS バックフィル・ML ウェアハウス・指数同期 | `python3 -m src.scripts.data.backfill_all` |
| `maintenance/` | 健全性チェック・stg スモーク・事前計算 | `python3 -m src.scripts.maintenance.verify_stg_smoke` |
| `docs/` | HTML 仕様書へのサンプル埋め込み | `python3 -m src.scripts.docs.gen_scrape_process_samples` |
| `ci/` | GitHub Actions 用静的チェック | `python3 -m src.scripts.ci.check_no_shuffle` |

## `scripts/`（シェルのみ）

| パス | 用途 |
|------|------|
| `scripts/server/` | API 起動・`service_start.sh`・tcpexposer トンネル |
| `scripts/cron/` | crontab 投入・定期 git pull・ログローテーション |
| `scripts/*.sh` | 運用ラッパ（`auto_scrape.sh` 等） |

Python CLI は **`src/scripts/`** に集約。旧 `scripts/*.py` は移行済み。

## データ・生成物

| パス | 備考 |
|------|------|
| `data/features/` | 特徴量 Parquet ストア |
| `data/page_reference/` | UI ポータブルバンドル（別 PC へコピー可、`BUNDLE.md`） |
| `data/local/` `data/queue/` | 運用メタ・スクレイプキュー（gitignore） |
| `tmp/` | 一時 JSON 等（gitignore、`tmp/.gitkeep` のみ追跡） |
| `logs/` | 実行ログ（gitignore） |

## フロントエンド・公開 URL

- 開発: `./service_start --env dev` → Next.js `http://127.0.0.1:3000/`
- ポート一覧: [service-endpoints.md](service-endpoints.md)
- 外部トンネル: `scripts/server/tunnel_tcpexposer.sh`（`.cursor/hooks` で自動起動可）

## 削除・整理（2026-07）

- **`archive/`** … リファクタ前スナップショット（DEC-012）を削除。旧コードは Git 履歴を参照。
- **`tmp/scrape_queue_*.json`** … 18MB の一時キュー JSON を削除し `tmp/` を gitignore。
- **`scripts/*.py`** … `src/scripts/maintenance/` または `scraping/` へ移動。
- **`templates/` `static/` `notebooks/` `config/settings.yaml`** … 空ディレクトリだったため、`archive/` から実体を復元。
