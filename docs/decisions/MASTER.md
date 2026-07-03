# keiba-vpn — マスター仕様書
> 最終更新: 2026-07-03 | 参照DEC: DEC-001〜DEC-011

---

## 1. プロジェクト概要

keiba-vpn は、日本競馬（JRA）のレースデータをスクレイピングして LightGBM モデルで予測し、Next.js フロントエンドに表示するマルチユーザー競馬予測・データ分析 Web アプリケーションである。

- **主要機能**: AI予測スコア付き出馬表表示、条件別勝率・単勝回収率集計、コース別種牡馬ランキング
- **インフラ基盤**: ConoHa VPS（メモリ2GB / SSD 100GB）固定費モデル（契約済み）
- **設計原則**: VPS 2GB メモリ制約内での安定稼働、スクレイピング・推論・API配信の3プロセス完全分離、クラウド従量課金によるコスト爆発リスクの排除

---

## 2. 技術スタック

| レイヤー | 技術・サービス | 備考 |
|---------|--------------|------|
| フロントエンド | TypeScript / Next.js 14 (App Router, SSG+ISR+CSR) | DEC-001確定 |
| フロントエンドホスティング | Vercel（Hobbyプラン・無料枠） | 月額¥0、CDN配信、商用化時はProへ移行 |
| バックエンドAPI | Python 3.11 / Flask 3.x + Gunicorn 21.x | DEC-008確定（FastAPI移行はADR-001に基づき同時接続500超で再検討） |
| ASGI/WSGIワーカー | Gunicorn + gevent（`workers=2, worker-connections=500`） | DEC-011確定 |
| リバースプロキシ | Nginx（静的ファイル直配信・keepalive・gzip） | |
| キャッシュ層 | Redis 7（VPS内同居、`maxmemory 256mb`, `allkeys-lru`） | |
| データストア（主系） | VPS ローカル `/data`（SSD 100GB） + PostgreSQL | |
| データストア（長期保存） | Google Cloud Storage（Parquet形式） | DEC-002確定 |
| ETL/スクレイピング | Python 3.11 / requests + BeautifulSoup4、cron on VPS | |
| AI推論 | Python 3.11 / LightGBM + scikit-learn（バッチ専用） | |
| SHAP説明 | SHAP TreeExplainer（上位10頭のみ、時間差実行） | |
| モデル学習 | Cloud Run Jobs（2vCPU / 4GB、週次）| VPS上の学習実行は禁止（DEC-010） |
| 非同期タスク | Celery（Redis Broker、3キュー分離） | DEC-011確定 |
| VPN/認証 | WireGuard on VPS | スクレイピング出口IP管理 |
| CDN/DDoS防御 | Cloudflare（無料プラン）| DEC-011確定 |
| アラート | Slack Webhook | |
| 監視（初期） | UptimeRobot（無料）外形監視 | ユーザー増加後にPrometheus追加 |
| GCSパス管理 | `keiba-vpn/src/scraper/gcs_paths.py`（Single Source of Truth） | DEC-010確定 |

> **DEC-001 vs DEC-008矛盾解消**: DEC-001はバックエンドをFastAPIと提案したが、DEC-008（新しい決定）でFlask 3.x継続が全エージェント合意により確定。FastAPIへの移行はADR-001に従い同時接続500超過時に再検討する。

---

## 3. アーキテクチャ設計

### 3-1. システム全体図

```
[ユーザー（モバイル/PC）]
        │ HTTPS
[Cloudflare] ← DDoS防御・SSL終端・レート制限
        │
[Vercel CDN] ← Next.js 14 静的/ISR配信
        │ API calls（HTTPS）
[Nginx on VPS]
        │ リバースプロキシ
[Gunicorn + gevent（workers=2, connections=500）]
        │
[Flask API]
    ├── L1: Flask lru_cache（TTL 60秒）
    ├── L2: Redis（TTL 1分〜24時間）
    ├── L3: PostgreSQL race_cache テーブル
    └── L4: GCS フォールバック（gcs_paths.py 経由）

[VPS 内バッチプロセス（cron・systemd分離）]
    ├── Scraper Worker（cron 06:00, 12:00 JST）
    │    └── Circuit Breaker + 指数バックオフリトライ
    │    └── GCS書き込み: src/scraper/gcs_paths.py 参照
    └── Inference Worker（cron 08:10 JST、ETL完了フラグ確認後）
         └── LightGBM バッチ推論 → Redis + PostgreSQL 書き込み

[Google Cloud Storage]
    └── raw/race_card/{date}/ ← スクレイパー書き込み
    └── features/static/dt={date}/
    └── features/dynamic/dt={date}/race_id={race_id}/
    └── models/current/model.lgb
    └── models/v{N}/model.lgb（ロールバック用）

[Cloud Run Jobs（週次・学習専用）]
    └── 2vCPU / 4GB、GCSパスはgcs_paths.py参照
```

### 3-2. VPSメモリ予算（2GB）

| コンポーネント | 予算 |
|--------------|------|
| OS + systemd | 256 MB |
| Nginx | 50 MB |
| Flask / Gunicorn（workers=2、gevent） | 300 MB |
| LightGBM Booster（CoW共有、バッチ実行時） | 200 MB |
| Celery Workers（推論分離） | 200 MB |
| PostgreSQL（shared_buffers 128MB） | 200 MB |
| Redis（maxmemory 256MB） | 256 MB |
| スクレイパー（時間分離済み、ピーク時） | 300 MB |
| OOM Killerバッファ | 約 238 MB |
| **通常時合計** | **約 1,462 MB ✅** |

> Celeryワーカーとスクレイパーは時間分離（同時起動禁止）。cron間隔は90分以上確保。

### 3-3. GCSパス設計（`src/scraper/gcs_paths.py`）

```python
GCS_BUCKET              = "keiba-vpn-data"  # 環境変数で上書き可
RAW_RACE_CARD_PREFIX    = "raw/race_card/{date}/"
RAW_ODDS_PREFIX         = "raw/odds/{date}/{race_id}/"
FEATURE_STATIC_PREFIX   = "features/static/dt={date}/"
FEATURE_DYNAMIC_PREFIX  = "features/dynamic/dt={date}/race_id={race_id}/"
MODEL_PREFIX            = "models/v{version}/"
MODEL_CURRENT_PATH      = "models/current/model.lgb"
MODEL_ROLLBACK_PATH     = "models/v{version}/model.lgb"
```

> **禁止事項**: 上記以外の場所へのGCSパスハードコード禁止。CI/lintで自動検出。

### 3-4. データフロー（レース当日）

```
06:00 JST  → Scraper Worker（出馬表スクレイピング）→ GCS raw/
             mark_etl_complete(race_date) → Redis etl:complete:{date}
08:00 JST  → ETL/特徴量変換 → GCS features/
08:10 JST  → Inference Worker（ETL完了フラグ確認後起動）
             LightGBM predict_proba → Redis pred:{race_id} + PostgreSQL predictions
12:00 JST  → オッズ追加スクレイピング（差分再推論: ±15%変動時のみ）
レース後   → 結果スクレイピング → GCS results/
毎週月曜   → Cloud Run Jobs（モデル再学習）→ GCS models/vN/
```

---

## 4. 機能要件（確定版）

### 4-A. データパイプライン

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| DP-01 | GCSへのレースデータ取り込みをレース当日06:00までに完了 | 高 |
| DP-02 | スクレイピング失敗時は最大3回リトライ（指数バックオフ: 4秒→8秒→60秒） | 高 |
| DP-03 | データ欠損時はGCS前回データにフォールバックし `data_status: stale` フラグを付与 | 高 |
| DP-04 | GCS書き込み完了後にRedisキー `etl:complete:{race_date}` を設定（TTL 3600秒） | 高 |
| DP-05 | ETLパイプライン失敗時、5分以内にSlack通知 | 中 |
| DP-06 | Circuit Breaker（`pybreaker`）: 5回失敗でOPEN、60秒後HALF-OPEN、1回成功でCLOSED | 高 |
| DP-07 | Circuit OPEN時は最終成功GCSデータからフォールバック表示。フロントに「データ更新中」バナー表示 | 高 |

### 4-B. AI予測モデル

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| PM-01 | 単勝予測的中率（推薦1位馬）: 初期リリース目標 ≥20%、中期目標 ≥30% | 高 |
| PM-02 | AI推論はWebプロセスと完全分離したCeleryバッチワーカー専用（Gunicornワーカー内での`predict()`呼び出し禁止） | 高 |
| PM-03 | Inference Worker起動時に `MODEL_CURRENT_PATH` を参照してモデルを1回のみロード | 高 |
| PM-04 | Panderaスキーマバリデーション（距離800〜3600m、頭数2〜18頭、馬体重380〜620kg等）を推論前に実施 | 高 |
| PM-05 | ETL完了フラグガード: Redisキー `etl:complete:{race_date}` が存在しない場合、最大10分待機後に停止 | 高 |
| PM-06 | SHAP TreeExplainerによる上位5特徴量を上位10頭のみに適用（時間差実行） | 中 |
| PM-07 | モデルバージョン管理: `models/current/model.lgb` + 1世代前（`models/v{N-1}/model.lgb`）を保持 | 高 |
| PM-08 | 予測キャッシュペイロードにモデルバージョン情報を含める（後からのトレース用） | 中 |
| PM-09 | オッズが±15%を超えて変動した場合のみ差分再推論を実行（1日上限5回） | 低 |
| PM-10 | モデル週次再学習はCloud Run Jobs（2vCPU / 4GB）で実施。VPS上での学習実行禁止 | 高 |

### 4-C. フロントエンド/UX

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| FE-01 | モバイルファーストレスポンシブデザイン（ブレークポイント: 375px / 768px / 1280px） | 高 |
| FE-02 | 初期ページロードLCP ≤2.5秒 | 高 |
| FE-03 | トップ画面デフォルト: 今日のレース一覧（開催場・レース番号・発走時刻・予測TOP3馬） | 高 |
| FE-04 | `data_status`値に応じたバナーをUI上部に出し分け（fresh: 非表示 / stale: 黄背景警告 / unavailable: 赤背景） | 高 |
| FE-05 | レース一覧ページにNext.js ISR（revalidate: 300秒）を適用 | 中 |
| FE-06 | 当日出馬表・当日レースページはCSR（直前変更対応） | 高 |
| FE-07 | AI予測結果ページはISR（revalidate: 3600秒） | 中 |
| FE-08 | レース詳細ページにSkeleton UIを実装 | 中 |
| FE-09 | 予測結果表示を「推奨馬券種別 + 上位3頭カード形式」で表示。推奨馬ハイライト（色分け・ランク番号） | 中 |
| FE-10 | 仮想スクロール（`react-window`）をレース一覧・馬柱テーブルに適用 | 中 |
| FE-11 | 発走T-10分未満のレースはカウントダウンを赤字表示 | 中 |
| FE-12 | PWA対応: ホーム画面追加・オフラインキャッシュ有効化 | 中 |

### 4-D. APIエンドポイント（確定版）

| メソッド | パス | 認証 | 説明 |
|--------|------|------|------|
| GET | `/health` | ❌ No | ヘルスチェック |
| GET/POST | `/login` | ❌ No | ログイン（Flask-Login） |
| POST | `/logout` | ✅ Yes | ログアウト |
| GET | `/api/v1/races/today` | ❌ Public | 本日の全レース一覧 |
| GET | `/api/v1/predictions/{race_id}` | ✅ Yes | 特定レースのAI予測スコア一覧（4段キャッシュ経由） |
| POST | `/api/v1/filter/stats` | ✅ Yes | 条件別勝率・単勝回収率集計 |
| GET | `/api/v1/sires/ranking` | ✅ Yes | コース別種牡馬ランキング |
| GET | `/api/v1/shap/{race_id}` | ✅ Yes | SHAP説明値 |
| GET | `/api/v1/races/{race_id}/entries` | ✅ Yes | 出馬表 + AIスコア統合 |
| POST | `/api/predict` | ✅ Yes | 非同期推論タスク起動（Celery）→ task_id返却 |
| GET | `/api/inference/{task_id}` | ✅ Yes | 推論タスク状態（pending/done/failed） |
| GET | `/admin/circuit-status` | ✅ Yes（IP制限） | Circuit Breaker状態・失敗回数・最終失敗時刻 |

**予測APIレスポンス仕様（確定）**
```json
{
  "race_id": "R11_20240120",
  "data_status": "fresh",
  "model_version": "v3",
  "predictions": [
    {
      "horse_id": "12345",
      "horse_name": "サンプルホース",
      "win_probability": 0.34,
      "confidence_score": 82,
      "rank_prediction": 1,
      "reasons": ["過去成績: 直近3走平均着順 1.8位", "馬場適性: 良馬場勝率 61%"],
      "inference_source": "local"
    }
  ],
  "generated_at": "2024-01-20T09:15:00+09:00"
}
```

> タイムスタンプはISO 8601 + JST（+09:00）で統一。

### 4-E. キャッシュ設計（4段キャッシュ）

| レベル | 実装 | TTL | 対象 |
|--------|------|-----|------|
| L1 | Flask `lru_cache`（ワーカー内メモリ） | 60秒 | ホットデータ |
| L2 | Redis | 出走表10分 / AI予測60分 / オッズ1分 / 過去結果24時間 | 全APIレスポンス |
| L3 | PostgreSQL `race_cache` テーブル | ETL更新時に自動更新 | 出走表・予測 |
| L4 | GCS（フォールバック） | — | 全データ（`gcs_paths.py`経由） |

Thundering Herd防止: `SET NX`ミューテックスロックをRedisに実装。

### 4-F. 認証・ユーザー機能

| 要件ID | 内容 | 優先度 |
|--------|------|--------|
| US-01 | Flask-Loginセッション認証（ログイン・ログアウト） | 高 |
| US-02 | 全分析API（F-1〜F-10対応エンドポイント）に`@login_required`デコレータを適用 | 高 |
| US-03 | ゲスト（未ログイン）でも `/api/v1/races/today` は閲覧可 | 高 |
| US-04 | 予測履歴・的中履歴を直近30日間保存 | 低 |
| US-05 | 「追いかけ馬」最大10頭の登録と出走時プッシュ通知 | 低 |

---

## 5. 非機能要件（確定版）

| 要件ID | 分類 | 内容 | 目標値 |
|--------|------|------|--------|
| NFR-01 | 可用性 | サービス月間稼働率 | ≥ 99.5%（ダウンタイム ≤ 3.6h/月） |
| NFR-02 | 性能 | API P50レイテンシ（Redis HIT時） | ≤ 50ms |
| NFR-03 | 性能 | API P99レイテンシ（Redis HIT時） | ≤ 200ms |
| NFR-04 | 性能 | API P99レイテンシ（GCSフェッチ時） | ≤ 1,000ms |
| NFR-05 | 性能 | E2E P95レスポンスタイム（国内ブラウザ） | ≤ 1,200ms |
| NFR-06 | 性能 | 同時接続処理数 | 瞬間1,000 req（タイムアウトエラー率 < 0.1%） |
| NFR-07 | 性能 | AI推論バッチ全レース処理時間 | ≤ 10分（08:00〜08:10完了） |
| NFR-08 | 性能 | モバイルLCP | ≤ 2.5秒 |
| NFR-09 | 性能 | Lighthouse Performanceスコア（モバイル） | ≥ 85 |
| NFR-10 | メモリ | VPS総メモリ使用量（通常時） | ≤ 1,536MB（バッファ確保） |
| NFR-11 | メモリ | Redisメモリ上限 | 256MB（`maxmemory` 強制上限） |
| NFR-12 | メモリ | LightGBM Booster RSS増加量 | ≤ 250MB（超過時は`num_leaves` 63→31で再学習） |
| NFR-13 | メモリ | LightGBM `num_threads` 上限 | `max(1, vCPU // workers)` = 1（2vCPU環境） |
| NFR-14 | キャッシュ | Redisキャッシュヒット率 | ≥ 90%（レース日） |
| NFR-15 | キャッシュ | GCS読み取りリクエスト削減率 | ≥ 95%（Redis導入後比） |
| NFR-16 | スケーラビリティ | Gunicornワーカー数 | 2（`max_requests=500`, `max_requests_jitter=50`） |
| NFR-17 | セキュリティ | レート制限: `POST /api/predict` | 10回/分/