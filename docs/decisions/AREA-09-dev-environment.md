# AREA-09 — 開発環境要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-004(プロセス分離), DEC-008(VPS構成), 新規追加

---

## 1. 環境概要

| 環境 | ホスト | 目的 | prod 機能の包含 |
|---|---|---|---|
| **dev / stg** | ローカル PC（ハイパフォーマンス + GPU） | 開発・モデリング・テスト・stg 動作確認 | ✅ prod の全機能を包含 |
| **prod** | ConoHa VPS 2GB/100GB SSD（常時稼働） | ユーザー向けサービス提供 | — |

> **方針**: dev/stg は同一ローカル PC 上で管理し、prod と同等の構成を再現できること。  
> prod で動くすべての機能は dev/stg でテスト済みの状態で deploy する。

---

## 2. dev / stg 環境（ローカル PC）

### ハードウェア要件（最小）

| 項目 | 要件 | 用途 |
|---|---|---|
| CPU | 8 コア以上推奨 | LightGBM 学習 / SHAP 計算 |
| RAM | 16GB 以上 | 全サービス同時起動 + 学習ジョブ |
| GPU | NVIDIA GPU（CUDA 対応）推奨 | 将来の Neural ODE モデル学習 |
| ストレージ | 100GB 以上（SSD 推奨） | データセット / GCS ローカルキャッシュ |

### 実行するワークロード

| カテゴリ | 内容 |
|---|---|
| **モデリング** | LightGBM 学習（全データ使用可）、ハイパーパラメータ探索、Feature Engineering |
| **データセット確認** | EDA（探索的データ分析）、特徴量の可視化・統計確認 |
| **SHAP 分析** | 全頭 SHAP 計算（prod では上位 10 頭のみ）、特徴量重要度の深掘り |
| **E2E テスト** | Playwright E2E、統合テスト、バックエンド単体テスト |
| **CI ローカル実行** | lint / pytest / vitest をローカルで確認 |
| **stg 動作確認** | prod 相当の構成で動作確認（ユーザーには非公開） |

### サービス構成（Docker Compose 推奨）

```yaml
# docker-compose.dev.yml（概要）
services:
  flask-api:        # Flask + Gunicorn（prod 同等設定）
  inference-worker: # LightGBM バッチワーカー
  scraper:          # ETL スクレイパー（JRA または テストフィクスチャ）
  redis:            # Redis 7（maxmemory=256MB prod 設定に合わせる）
  postgres:         # PostgreSQL（shared_buffers=128MB prod 設定に合わせる）
  nextjs:           # Next.js dev server（または prod build）
  prometheus:       # メトリクス（prod 同等監視の確認用）
  grafana:          # ダッシュボード確認用
```

### 環境変数（`.env.dev`）

```bash
FLASK_ENV=development
DATABASE_URL=postgresql://localhost:5432/keiba_dev
REDIS_URL=redis://localhost:6379/0
GCS_BUCKET=keiba-vpn-data-dev        # dev 専用バケット
GCS_USE_LOCAL_CACHE=true              # GCS アクセスをローカルキャッシュで代替可
MODEL_CURRENT_PATH=models/current/model.lgb
LOG_LEVEL=DEBUG
SLACK_WEBHOOK_URL=                    # dev では空白可（通知不要）
```

### dev 専用機能（prod には不要）

| 機能 | 内容 |
|---|---|
| GCS ローカルキャッシュ | `GCS_USE_LOCAL_CACHE=true` で GCS アクセスをローカルディレクトリで代替 |
| テストフィクスチャ用スクレイパー | JRA に接続せず静的フィクスチャからデータを読み込む |
| Jupyter / JupyterLab | EDA・モデル評価・SHAP 可視化 |
| MLflow（任意） | 実験管理・モデルバージョントラッキング |
| full SHAP 計算 | 全頭対象（prod は上位 10 頭のみ）|
| デバッグモード | Flask debug=True, ホットリロード |

### dev でのモデル学習ルール

- ローカル PC での学習は **dev/stg のみ許可**
- 学習済みモデルは GCS `models/v{N}/model.lgb` にアップロードして Cloud Run Jobs と同じパスに配置
- prod VPS 上での学習は **引き続き禁止**（Cloud Run Jobs または dev ローカルのみ）

---

## 3. prod 環境（ConoHa VPS）

詳細は **[AREA-04-ops.md](AREA-04-ops.md)** 参照。

### prod のみに適用される制約

| 制約 | 内容 |
|---|---|
| メモリ制限 | 2GB 以内で全サービスを稼働 |
| VPS での学習禁止 | LightGBM 学習は Cloud Run Jobs または dev ローカルのみ |
| SHAP 実行制限 | 上位 10 頭のみ、推論と時間分離 |
| ログレベル | INFO 以上のみ出力 |
| デバッグモード無効 | Flask debug=False 必須 |

---

## 4. 環境間の差異一覧

| 機能 | dev / stg | prod |
|---|---|---|
| ホスト | ローカル PC（GPU）| ConoHa VPS 2GB |
| Flask モード | debug=True / False（stg）| debug=False |
| モデル学習 | ✅ 許可 | ❌ 禁止 |
| SHAP 対象 | 全頭 | 上位 10 頭 |
| GCS | dev 専用バケット or ローカルキャッシュ | `keiba-vpn-data` 本番バケット |
| DB データ | シードデータ or 匿名化データ | 本番データ |
| E2E テスト | ✅ 実行 | ❌ 実行しない |
| Slack アラート | 任意（省略可）| ✅ 必須 |
| Prometheus + Grafana | ✅ 起動 | Phase 2 以降 |
| ログレベル | DEBUG | INFO |
| Circuit Breaker | 動作確認用テストあり | 本番稼働 |
| Jupyter | ✅ 起動 | ❌ 不要 |

---

## 5. デプロイフロー

```
[ローカル dev/stg]
  ├─ 機能開発 → unit test → integration test → E2E test
  ├─ モデル学習 → GCS にアップロード → stg で推論動作確認
  └─ docker-compose.dev.yml で prod 相当の動作を確認

   │ GitHub PR → CI 通過 → main マージ
   ▼

[prod: ConoHa VPS]
  └─ git pull → pip install → systemctl restart
```

### ブランチ戦略

| ブランチ | 対応環境 | ルール |
|---|---|---|
| `feature/*` | dev | 自由開発 |
| `develop` | stg（ローカル）| PR 必須、CI 通過必須 |
| `main` | prod | PR + CI + レビュー必須 |

---

## 6. 非機能要件（環境系）

| ID | 要件 |
|---|---|
| NFR-ENV-01 | dev/stg と prod の Docker イメージは同一ベースイメージを使用 |
| NFR-ENV-02 | prod デプロイ前に stg で E2E テストを通過すること |
| NFR-ENV-03 | 本番データを dev/stg 環境で使用しない |
| NFR-ENV-04 | dev でモデル学習を行った場合、GCS にアップロードして stg で動作確認後に prod へ deploy |
| NFR-ENV-05 | 環境変数は `.env.dev` / `.env.prod` で管理し、`.gitignore` で除外 |
| NFR-ENV-06 | dev/stg では GCS アクセスをローカルキャッシュで代替可能にすること（オフライン開発対応）|
