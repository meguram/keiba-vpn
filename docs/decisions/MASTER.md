# keiba-vpn — マスター仕様書
> 最終更新: 2026-07-03 | 参照DEC: DEC-001〜DEC-010

---

## 1. プロジェクト概要

keiba-vpn は、日本競馬（JRA）のレースデータをスクレイピングして LightGBM モデルで予測スコアを算出し、データ分析機能とともに Next.js フロントエンドで提供するマルチユーザー競馬予測・データ分析 Web アプリケーションである。

**主要機能**:
- AI予測スコア（LightGBM バッチ推論）の出馬表並列表示
- 条件別勝率・単勝回収率の統計分析（距離・馬場・クラス・騎手等のフィルタ）
- コース別好成績種牡馬ランキング
- 定期スクレイピングパイプライン（GCS へ Parquet 形式で保存）

**制約条件**:
- ConoHa VPS（メモリ 2GB / SSD 100GB）は契約済み固定費
- VPS 2GB メモリ制約内での安定稼働を最優先

---

## 2. 技術スタック

| レイヤー | 採用技術 | 採用理由 |
|---------|---------|---------|
| フロントエンド | TypeScript + Next.js 14 (App Router) | React エコシステム・SSG/ISR/CSR 使い分け・OpenAPI 自動型生成 |
| フロントエンドホスティング | Vercel（Hobby プラン、無料） | CDN 配信・デプロイ自動化・VPS メモリ消費ゼロ。商用化時は Pro（¥2,500/月）へ移行 |
| バックエンド API | Python 3.11 + Flask 3.x | 既存コード資産・AI/ML ライブラリとの親和性。同時接続 500 超過時に FastAPI 移行を検討（ADR-001） |
| WSGI サーバー | Gunicorn 21.x（worker\_class=gthread, workers=2, threads=4, preload\_app=True） | CoW によるモデルメモリ共有・I/O バウンド処理の並行化 |
| リバースプロキシ | Nginx | — |
| キャッシュ | Redis 7（VPS 内同居、maxmemory 256MB、allkeys-lru） | 追加コスト¥0・超低レイテンシ。障害時は GCS 直接読み取りにフォールバック |
| データストア（主系） | VPS ローカル /data（SSD 100GB）+ GCS（Parquet、長期保存・バックアップ） | DEC-002 確定。GCS バケット: `keiba-vpn-data` |
| DB | PostgreSQL（VPS 内同居、shared\_buffers=128MB） | レース結果・予測・ユーザー認証・race\_cache テーブル管理 |
| ETL / スクレイピング | Python 3.11（requests + BeautifulSoup4）+ cron on VPS | 追加コスト¥0。ETL 処理時間 30 分超過時に Cloud Run Jobs へ移行（ADR-004） |
| AI 推論 | LightGBM + scikit-learn（バッチワーカー専用プロセス） | GPU 不要・推論 ~12ms/レース・VPS 2GB 制約内。DEC-004・DEC-009 準拠 |
| SHAP | SHAP TreeExplainer（上位 10 頭のみ、時間差実行） | メモリ制約対応 |
| モデル学習 | Cloud Run Jobs（2vCPU / 4GB、週次） | VPS 上での学習実行は禁止 |
| 認証 | Flask-Login（セッション認証）+ WireGuard VPN（アクセス制御） | VPS 上で完結・追加コスト¥0 |
| GCS パス管理 | `keiba-vpn/src/scraper/gcs_paths.py`（Single Source of Truth） | 全コンポーネントがこのファイルの定数をインポート。ハードコード禁止 |
| 監視 | UptimeRobot（無料、外形監視）+ Slack Webhook アラート | 初期フェーズは追加コスト¥0 |

> **矛盾解消（DEC-001 vs DEC-004/DEC-008）**: DEC-001 はバックエンドを FastAPI と定義しているが、DEC-004・DEC-008（新しい決定）では既存コード資産・移行コスト最小化の観点から Flask を採用と確定。FastAPI への移行は同時接続 500 超過時のトリガーで再検討する。

---

## 3. アーキテクチャ設計

### 3-1. システム構成図

```
┌────────────────────────────────────────────────────────────────────┐
│  ユーザー (モバイル / PC)                                           │
│  Next.js フロントエンド（Vercel CDN、PWA 対応）                     │
└──────────────────────────┬─────────────────────────────────────────┘
                            │ HTTPS
┌──────────────────────────▼─────────────────────────────────────────┐
│  ConoHa VPS 2GB RAM / SSD 100GB                                    │
│  ┌─────────────────┐  ┌────────────────────┐  ┌─────────────────┐ │
│  │ Flask API        │  │ Inference Worker    │  │ Redis 256MB     │ │
│  │ systemd unit1    │  │ systemd unit2       │  │ allkeys-lru     │ │
│  │ Gunicorn 2w×4t  │  │ cron 08:00 daily    │  └─────────────────┘ │
│  └────────┬────────┘  └──────────┬──────────┘                     │
│           │                      │        ┌────────────────────────┤
│           └────────────┬─────────┘        │ PostgreSQL             │
│                        │                  │ shared_buffers=128MB   │
│                        └──────────────────└────────────────────────┤
│  WireGuard VPN / Nginx                                             │
└────────────────────────────────────────────────────────────────────┘
        ↑ スクレイピング (cron)              ↑ モデル・特徴量
┌───────┴─────────┐          ┌──────────────┴──────────────────────┐
│ JRA 外部サイト   │          │ Google Cloud Storage (GCS)          │
│ Circuit Breaker  │          │ keiba-vpn-data バケット              │
│ + Retry (tenacity)│         │ パス定義: src/scraper/gcs_paths.py  │
└─────────────────┘          │ raw/{YYYY-MM-DD}/                   │
                              │ features/static/dt={date}/          │
                              │ features/dynamic/dt={date}/...      │
                              │ models/v{N}/model.lgb               │
                              │ models/current/model.lgb            │
                              └─────────────────────────────────────┘
                                          ↑ 週次学習
                              ┌───────────┴─────────────────────────┐
                              │ Cloud Run Jobs (2vCPU / 4GB)        │
                              │ 学習専用・VPS 上での学習禁止          │
                              └─────────────────────────────────────┘
```

### 3-2. VPS メモリ予算（確定値）

| コンポーネント | 予算 |
|--------------|------|
| OS + systemd | 256 MB |
| Flask / Gunicorn（workers=2、preload_app CoW 共有） | 200 MB |
| LightGBM Booster（CoW = 物理 1 コピー） | 200 MB |
| PostgreSQL（shared_buffers=128MB + work_mem 等） | 200 MB |
| Redis（maxmemory 256MB） | 256 MB |
| スクレイパー（レース日ピーク・時間分離済み） | 300 MB |
| OOM Killer 回避バッファ | 388 MB |
| **ピーク合計（時間分離後）** | **約 1,600 MB ✅（2GB の 78%）** |

> SHAP 計算（~200MB）は LightGBM バッチ推論と時間差で別 cron ジョブとして実行し、同時起動を禁止する。

### 3-3. GCS ファイルパス設計（`src/scraper/gcs_paths.py` で一元管理）

```python
GCS_BUCKET             = "keiba-vpn-data"
RAW_RACE_CARD_PREFIX   = "raw/race_card/{date}/"
RAW_ODDS_PREFIX        = "raw/odds/{date}/{race_id}/"
FEATURE_STATIC_PREFIX  = "features/static/dt={date}/"
FEATURE_DYNAMIC_PREFIX = "features/dynamic/dt={date}/race_id={race_id}/"
MODEL_PREFIX           = "models/v{version}/"
MODEL_CURRENT_PATH     = "models/current/model.lgb"
MODEL_ROLLBACK_PATH    = "models/v{version}/model.lgb"
```

全コンポーネント（ETL・Inference Worker・Cloud Run Jobs・Flask API）はこのファイルの定数をインポートして使用する。パス文字列のハードコードは CI（lint ルール）で検出・禁止する。

### 3-4. 4段キャッシュ設計（`GET /api/v1/predictions/{race_id}`）

| 段 | 種別 | TTL | 備考 |
|---|------|-----|------|
| L1 | Flask `lru_cache` | 60秒 | ワーカー間非共有（各ワーカー独立） |
| L2 | Redis | 5分（発走+30分で自動削除） | `maxmemory 256mb` 設定済み |
| L3 | PostgreSQL `race_cache` テーブル | ETL 更新まで有効 | 非同期書き戻し推奨 |
| L4 | GCS（`gcs_paths.py` の定数参照） | フォールバック | キャッシュミス時のみ参照 |

キャッシュミス時（推論未完了）は HTTP 503 + `"推論結果準備中"` を返す。

### 3-5. スクレイピング耐障害設計

- `tenacity` による指数バックオフリトライ（最大3回: 4秒→8秒→60秒）
- `pybreaker` による Circuit Breaker（5回失敗で OPEN / 60秒後 HALF-OPEN）
- Circuit OPEN 時: GCS 最終成功データからフォールバック表示 + フロントエンドに「データ更新中」バナー表示
- Circuit OPEN 検知時: Slack Webhook で5分以内アラート通知

### 3-6. AI 推論 2段階戦略

```
[通常時] VPS バッチ推論（cron 08:00、ETL 完了フラグ確認後）
  ├── ETL 完了フラグガード: Redis キー etl:complete:{race_date} 確認（最大10分待機）
  ├── LightGBM .pkl ロード（MODEL_CURRENT_PATH、起動時1回のみ）
  ├── Pandera スキーマバリデーション（距離 800〜3600m、頭数 2〜18頭、馬体重 380〜620kg）
  ├── predict_proba() → predictions テーブル + Redis 書き込み
  │      inference_source = 'local'
  └── SHAP TreeExplainer → 上位 10 頭のみ（時間差実行）

[フォールバック時] 外部 WebAPI 推論（モデル更新中 / VPS 推論失敗時）
  └── inference_source = 'external'（UI に明示表示）
```

---

## 4. 機能要件（確定版）

### 4-1. データパイプライン

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| DP-01 | GCS へのレースデータ取り込みをレース発走 T-90分前までに完了する | 高 |
| DP-02 | スクレイピング失敗時は最大3回リトライ（指数バックオフ: 30s→60s→120s） | 高 |
| DP-03 | データ欠損時は `v_latest-1`（1世代前）にフォールバックし `data_status: stale` フラグを付与 | 高 |
| DP-04 | GCS Object Finalize イベントを Pub/Sub でサブスクライブし、Redis に `race_data_ready:{race_id}` フラグを立てる | 高 |
| DP-05 | ETL パイプライン失敗時、5分以内に Slack 通知を送信する | 中 |
| DP-06 | `v_latest` + `v_latest-1` を常時保持し、それ以前は7日後に GCS Lifecycle Policy で自動削除する | 中 |

**データステータス定義**（API・UI 全層で共有）:

| `data_status` | 意味 | UI バナー |
|--------------|------|----------|
| `fresh` | 最新データ取得成功 | 非表示 |
| `stale` | 前バージョンデータ使用中 | ⚠️ 「前回データを表示しています（更新: {timestamp}）」黄背景 |
| `unavailable` | データ取得不可 | 🔴 「データ取得不可。しばらくお待ちください。」赤背景 |
| 推論計算中 | — | ⏳ 「予測計算中（T-60分に表示予定）」グレー背景 |

### 4-2. AI 予測モデル

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| PM-01 | 単勝予測的中率（推奨1位馬の実際勝率）初期目標 ≥ 20%、中期目標 ≥ 30% | 高 |
| PM-02 | 予測 API 応答時間 ≤ 3秒（P95、Redis キャッシュヒット前提） | 高 |
| PM-03 | 各馬に SHAP 値に基づく予測根拠 TOP3 を自然言語テキストで表示 | 高 |
| PM-04 | 各馬に信頼スコア（0〜100）を付与。stale データ推論時は -10pt のペナルティを適用 | 中 |
| PM-05 | モデル再学習は週次バッチ（毎週月曜 02:00 JST、Cloud Run Jobs） | 中 |
| PM-06 | Pub/Sub イベントトリガーで T-80分から推論バッチを起動し、T-60分までに結果を公開 | 高 |

### 4-3. データ分析機能

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| F-1 | 距離・馬場・クラス・季節・騎手・調教師でフィルタした勝率・連対率集計 | 高 |
| F-2 | F-1 と同条件に対応した単勝・複勝回収率の算出 | 高 |
| F-3 | 指定コース × 任意条件での種牡馬ランキング（`horses.sire_id` カラム必須） | 高 |
| F-4 | 開催場所・距離・馬場・天候・クラス・頭数の AND 条件指定 UI | 高 |
| F-5 | データ鮮度表示（スクレイピング最終更新タイムスタンプを UI に表示） | 中 |

### 4-4. フロントエンド / UX

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| FE-01 | モバイルファーストのレスポンシブデザイン（ブレークポイント: 375px / 768px / 1280px） | 高 |
| FE-02 | 初期ページロード LCP ≤ 2.5秒（モバイル） | 高 |
| FE-03 | トップ画面デフォルト表示は「今日のレース一覧」（開催場・レース番号・発走時刻・予測TOP3馬） | 高 |
| FE-04 | `data_status` の値に応じたバナーを UI 上部に出し分ける | 高 |
| FE-05 | 予測根拠（SHAP TOP3）はデフォルト折りたたみ・タップで展開 | 中 |
| FE-06 | 発走まで T-10分未満のレースはカウントダウンを赤字表示 | 中 |
| FE-07 | レース一覧ページに Next.js ISR（revalidate: 300秒）を適用 | 中 |
| FE-08 | レース詳細ページに Skeleton UI を実装 | 中 |
| FE-09 | 予測結果表示を「推奨馬券種別 + 上位3頭カード形式」に刷新 | 中 |
| FE-10 | PWA 対応（ホーム画面追加・オフラインキャッシュ） | 中 |
| FE-11 | `win_probability`（勝つ確率）と `confidence_score`（予測確信度）の違いをツールチップで説明 | 中 |

### 4-5. ユーザー認証・パーソナライゼーション

| 要件ID | 内容 | `auth_required` | 優先度 |
|--------|------|-----------------|--------|
| US-01 | Flask-Login セッション認証（ログイン・ログアウト） | No（ログインページ自体） | 高 |
| US-02 | 全分析 API に `@login_required` 適用 | Yes | 高 |
| US-03 | ゲスト（未ログイン）でも当日の予測 TOP3 は閲覧可 | No（制限あり） | 高 |
| US-04 | 馬券種別の優先設定を保存し表示最適化 | Yes | 低 |
| US-05 | 「追いかけ馬」最大10頭の登録と出走時プッシュ通知 | Yes | 低 |
| US-06 | 予測履歴・的中履歴を直近30日間保存 | Yes | 低 |

### 4-6. API エンドポイント一覧（確定版）

| メソッド | エンドポイント | `auth_required` | 説明 |
|--------|--------------|-----------------|------|
| GET | `/health` | No | ヘルスチェック |
| GET/POST | `/login` | No | ログイン |
| POST | `/logout` | Yes | ログアウト |
| GET | `/api/v1/races/today` | No（ゲスト制限あり） | 本日の全レース一覧 |
| GET | `/api/v1/races/{race_id}/entries` | Yes | 出馬表 + AI スコア統合 |
| GET | `/api/v1/predictions/{race_id}` | No（ゲスト制限あり） | 特定レースの予測結果 |
| GET | `/api/v1/horses/{id}` | Yes | 馬情報 |
| POST | `/api/v1/filter/stats` | Yes | フィルタ付き集計クエリ（勝率・回収率） |
| GET | `/api/v1/sires/ranking` | Yes | コース別種牡馬ランキング |
| GET | `/api/v1/shap/{race_id}` | Yes | SHAP 説明値 |
| GET | `/api/v1/health` | No | API ヘルスチェック |
| GET | `/admin/circuit-status` | Yes（IP 制限または Basic 認証で暫定保護） | Circuit Breaker 状態 |

**予測 API レスポンス仕様**:
```json
{
  "race_id": "R11_20240120",
  "data_status": "fresh",
  "inference_source": "local",
  "model_version": "v3",
  "predictions": [
    {
      "horse_id": "12345",
      "horse_name": "サンプルホース",
      "win_probability