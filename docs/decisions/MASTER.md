# keiba-vpn — マスター仕様書

> **最終更新**: 2026-07-03  
> **参照ドキュメント**: AREA-01〜09（旧 DEC-001〜011 は `archive/` に移動済み）

---

## 概要

JRA（日本中央競馬会）データをスクレイピングし、LightGBM による予測スコアと統計分析を Next.js フロントエンドで提供する競馬予測 Web アプリ。

**最重要制約**: ConoHa VPS 2GB/100GB SSD は契約済み固定費。2GB 以内での安定稼働を最優先とする。

---

## 1. アプリケーション要件

詳細 → **[AREA-01-app-requirements.md](AREA-01-app-requirements.md)**

**提供機能**:
- AI 予測スコア付き出馬表表示（全頭: ログイン済み / TOP3: ゲスト）
- 条件別単勝率・ROI 分析、馬券種別収支分析
- コース別種牡馬ランキング（horses.sire_id）
- データ鮮度バナー（FRESH / STALE / STALE_CIRCUIT_OPEN）

**データフロー（当日）**: 06:00 ETL → 08:10 推論バッチ → 12:00 差分オッズ更新 → 毎週月曜 Cloud Run 再学習

**未解決（Human Review 必須）**:
- H-01: JRA スクレイピング ToS / robots.txt 法的確認（ETL 開始前に必須）
- H-02: ETL 遅延フォールバックポリシー（503 vs 前日データ配信）

---

## 2. フロントエンド

詳細 → **[AREA-02-frontend.md](AREA-02-frontend.md)**

| 技術 | 決定 |
|---|---|
| フレームワーク | TypeScript + Next.js 14 App Router |
| ホスティング | Vercel Hobby（¥0; 商用化で Pro ¥2,500/月） |
| レンダリング | ISR 300s（レース一覧）/ ISR 3600s（予測）/ CSR（当日出馬表） |
| 主要要件 | LCP ≤2.5s, Lighthouse ≥85, 375/768/1280px, PWA（Phase 2） |

---

## 3. バックエンド

詳細 → **[AREA-03-backend.md](AREA-03-backend.md)**

| 技術 | 決定 |
|---|---|
| フレームワーク | Python 3.11 + Flask 3.x（500+ 同時接続で FastAPI 移行 / ADR-001） |
| WSGI | Gunicorn 21.x（gthread, w=2, t=4, preload_app=True） |
| キャッシュ | Redis 7（VPS, 256MB allkeys-lru）4 層キャッシュ |
| DB | PostgreSQL（VPS, shared_buffers=128MB） |
| 認証 | Flask-Login + WireGuard VPN |

**4 層キャッシュ**: L1 lru_cache 60s → L2 Redis → L3 PostgreSQL race_cache → L4 GCS フォールバック

**レイテンシ目標**: P50 ≤50ms / P99 ≤200ms（Redis HIT）/ E2E P95 ≤1,200ms

---

## 4. 運用最適化

詳細 → **[AREA-04-ops.md](AREA-04-ops.md)**

**プロセス分離**:
- `keiba-api.service`: Flask API（常時）
- `keiba-inference.service`: LightGBM バッチ（cron 08:10）
- `keiba-scraper.service`: ETL（cron 06:00, 12:00）
- スクレイパーと推論は **90 分以上の間隔を強制**

**VPS メモリ（ピーク ~1,762MB / 2,048MB = 86%)**:
OS 256 + Nginx 50 + Flask(CoW) 300 + LightGBM(CoW) 200 + PostgreSQL 200 + Redis 256 + Scraper 300 + バッファ 238

**Circuit Breaker**: tenacity 3 回リトライ（4s→8s→60s）→ 5 失敗で pybreaker OPEN → GCS フォールバック + Slack アラート

**可用性目標**: ≥99.5%

---

## 5. コスト

詳細 → **[AREA-05-cost.md](AREA-05-cost.md)**

| 項目 | 月額 |
|---|---|
| ConoHa VPS 2GB（固定） | ¥1,320 |
| GCS `keiba-vpn-data` | ¥200〜500 |
| Vercel / WireGuard / Redis / Cloudflare / Slack | ¥0 |
| **合計** | **¥1,520〜1,820** |

**スケールアップトリガー**: ETL >30分 → Cloud Run Jobs / 50〜500 ユーザー → Cloud Run 推論 / 500+ → Vertex AI

---

## 6. データ管理

詳細 → **[AREA-06-data.md](AREA-06-data.md)**

**ストレージ**: VPS `/data`（プライマリ）+ GCS `keiba-vpn-data`（バックアップ/モデル）

**GCS パス SSoT**: `keiba-vpn/src/scraper/gcs_paths.py`（ハードコード禁止 / CI lint 強制）

**主要パス**:
```
raw/race_card/{date}/       features/static/dt={date}/
models/current/model.lgb    models/v{version}/model.lgb
```

**Feature Store**: 静的（週次）+ 動的（レース 30 分前）/ スキーマ: data-engineer 管理

---

## 7. モデリング管理

詳細 → **[AREA-07-modeling.md](AREA-07-modeling.md)**

**原則**: Gunicorn ワーカー内での `predict()` 禁止 / バッチ + Redis キャッシュ配信 / VPS 学習禁止

**推論バッチ**: cron 08:10 → ETL フラグ確認 → Pandera バリデーション → LightGBM → SHAP（上位 10 頭, 時間分離） → GCS + Redis 保存

**ModelRegistry**: RWLock による hot-reload、CoW で worker 間共有（preload_app=True）

**学習**: Cloud Run Jobs（2vCPU/4GB, 週次月曜 02:00 JST）、テンポラルリーク防止

**精度目標**: Logloss ≤2.2 / Calibration ≤0.05 / ROI ≥-15% / 単勝的中 ≥20% 初期 / ≥30% 中期

---

## 8. テスト

詳細 → **[AREA-08-testing.md](AREA-08-testing.md)**

| レイヤー | ツール | カバレッジ目標 |
|---|---|---|
| バックエンド Unit | pytest | ≥80%（推論バッチは ≥90%） |
| フロントエンド Unit | Vitest + React Testing Library | ≥70% |
| 統合テスト | pytest（fakeredis, 実 PostgreSQL） | ETL→Redis, 4 層キャッシュ, 認証フロー |
| E2E | Playwright（TypeScript） | dev/stg 環境で実行 |

**CI 必須ゲート**: lint（ruff / ESLint / GCS ハードコード検出）+ unit test + モデルサイズ <50MB + 推論 P99 ≤200ms

---

## 9. 開発環境

詳細 → **[AREA-09-dev-environment.md](AREA-09-dev-environment.md)**

| 環境 | ホスト | 主な役割 |
|---|---|---|
| **dev / stg** | ローカル PC（GPU 搭載, RAM ≥16GB） | モデリング・データ分析・E2E テスト・stg 動作確認 |
| **prod** | ConoHa VPS 2GB（常時稼働） | ユーザー向けサービス提供 |

**dev/stg の追加機能**（prod には不要）:
- LightGBM 学習（ローカル可）、全頭 SHAP 計算、Jupyter/EDA
- GCS ローカルキャッシュ（`GCS_USE_LOCAL_CACHE=true`）
- docker-compose.dev.yml で prod 相当構成を再現
- E2E テスト（Playwright）実行

**デプロイフロー**: `feature/*` → dev → `develop`（stg 動作確認 + E2E pass）→ CI → `main` → prod VPS

---

## 未解決事項

| ID | 内容 | 優先度 |
|---|---|---|
| H-01 | **JRA スクレイピング ToS / robots.txt 法的確認** — ETL 開始前に必須 | 緊急 |
| H-02 | ETL 遅延フォールバックポリシー（503 全件 vs 前日データ配信） | 高 |
| H-03 | PostgreSQL shared_buffers 実際値の確認（メモリバジェット最終確定） | 高 |
| H-04 | SHAP 時間分離のロードテスト（Phase 2 移行前） | 中 |
| H-05 | `/admin/circuit-status` 認証強化（現状 IP 制限 / Basic 認証、暫定） | 中 |
| H-06 | Google OAuth 実装タイミング（Phase 2 予定） | 低 |

---

## フェーズロードマップ

| Phase | 主要タスク |
|---|---|
| Phase 1 | gcs_paths.py 移行, Circuit Breaker, プロセス分離, モデルバージョニング, Cloud Run 学習移行, race_cache |
| Phase 2 | 4 層キャッシュ完全実装, ISR/Skeleton/カード UI, structlog/otel, Feature Store, PWA, Google OAuth |
| Phase 3 | Freshness チェック, Grafana ダッシュボード, Lighthouse CI ≥85, スキーマオーナーシップ文書化 |
