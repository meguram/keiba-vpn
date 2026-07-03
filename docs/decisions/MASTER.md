# keiba-vpn — マスター仕様書
> 最終更新: 2026-07-03 | 参照DEC: DEC-001〜DEC-009

---

## 1. プロジェクト概要

keiba-vpn は、日本競馬（JRA）のレースデータをスクレイピングして LightGBM モデルで勝率・着順を予測し、Next.js フロントエンドに出馬表と並べて表示するマルチユーザー競馬予測・データ分析 Web アプリである。

ユーザーは以下を利用できる。
- 距離・馬場・クラス・騎手等のフィルタ条件を指定した**勝率/単勝回収率の統計分析**
- バッチ事前計算済みの**AI 予測スコア（LightGBM + SHAP 説明）**の出馬表横表示
- レース後の**予測精度フィードバックダッシュボード**

インフラは **ConoHa VPS 固定費モデル（月額 ¥1,520〜1,820）** を採用し、クラウド従量課金によるコスト爆発リスクを排除する。フロントエンドは Vercel（無料枠）で CDN 配信する。

---

## 2. 技術スタック

| レイヤー | 技術・サービス | 根拠 |
|---------|--------------|------|
| **フロントエンド** | TypeScript / Next.js 14 (App Router) | Core Web Vitals 準拠、SSG/ISR/CSR 使い分け（DEC-001, DEC-008） |
| **フロントエンドホスティング** | Vercel（Hobby プラン・無料枠） | CDN 配信・VPS 負荷ゼロ。商用化時は Pro ($20/月) へ移行 |
| **スタイリング/グラフ** | Tailwind CSS + Recharts | — |
| **バックエンド API** | Python 3.11 / Flask 3.x | AI/ML ライブラリとの親和性、既存コード資産。同時接続 500 超過時に FastAPI 移行を検討（ADR-001） |
| **WSGI** | Gunicorn（worker=2, threads=4, worker_class=gthread）| 2vCPU VPS 向けスレッドモデル。`preload_app=True` で CoW メモリ共有 |
| **リバースプロキシ** | Nginx | — |
| **認証** | Flask-Login（セッション認証） | — |
| **キャッシュ** | Redis 7 on VPS（maxmemory 256MB, allkeys-lru） | 超低レイテンシ、追加コストゼロ（DEC-008） |
| **データストア（主系）** | VPS ローカル `/data`（SSD 100GB）+ SQLite / PostgreSQL | — |
| **データストア（バックアップ）** | GCS Parquet（`keiba-vpn-data/` バケット） | 長期保存・Parquet 形式（DEC-002） |
| **ETL / スクレイピング** | Python（requests / BeautifulSoup4）+ VPS cron | DEC-004 バッチ分離方針。VPS 追加コストゼロ（ADR-004） |
| **AI モデル** | scikit-learn / LightGBM + SHAP TreeExplainer | GPU 不要、実測 ~12ms/レース、VPS 2GB 制約内（DEC-006, DEC-008） |
| **VPN** | WireGuard on VPS | スクレイピング出口 IP 管理 |
| **監視** | UptimeRobot（無料・外形監視）| 初期フェーズ。ユーザー増加後に Prometheus on VPS を追加（ADR-D3） |
| **アラート** | Slack Webhook（無料プラン） | — |

> **[DEC-001 vs DEC-007/DEC-008 の矛盾解決]**: DEC-001 は FastAPI を選定したが、DEC-008（より新しく全エージェント合意済み）が Flask 3.x on VPS を採用。Flask 移行コストの最小化と既存コード資産を根拠とする。DEC-008 が勝つ。

---

## 3. アーキテクチャ設計

### 3-1. システム構成図

```
[Vercel CDN]
  Next.js 14 (TypeScript)
  ├── 出馬表 + AI予測スコア表示
  ├── 条件別データ分析フィルタ UI
  └── SSG（過去レース） / ISR（予測結果） / CSR（当日出馬表）
       │ HTTPS REST API
       ▼
[ConoHa VPS — 常駐プロセス]
  Nginx（リバースプロキシ）
  └── Flask 3.x + Gunicorn (workers=2, threads=4, preload_app=True)
       ├── GET  /api/v1/races/{race_id}/entries  （出馬表 + AI スコア統合）
       ├── GET  /api/v1/predictions/{race_id}     （AI 予測スコア一覧）
       ├── POST /api/v1/filter/stats              （条件別勝率・回収率）
       ├── GET  /api/v1/sires/ranking             （コース別種牡馬ランキング）
       ├── GET  /api/v1/shap/{race_id}            （SHAP 説明値）
       ├── GET  /health                           （ヘルスチェック）
       └── GET/POST /login、POST /logout
  Redis 7（maxmemory 256MB, allkeys-lru）
  PostgreSQL（users/auth/results、shared_buffers 128MB）
  WireGuard VPN

[ConoHa VPS — バッチプロセス（cron・時刻分離）]
  Scraper Worker       → GCS + VPS ローカル /data へ書き込み
  ETL Pipeline         → 特徴量変換、完了時 Redis に etl:complete:{date} フラグ
  Inference Worker     → LightGBM バッチ推論 → predictions テーブル + Redis
  SHAP Worker          → 推論完了後に時間差実行（上位 10 頭のみ）
  Model Retraining Job → 週次（毎週月曜 02:00 JST）

[GCS: keiba-vpn-data/]
  raw/html/YYYY-MM-DD/{race_id}.html
  normalized/entries/dt=YYYY-MM-DD/{race_id}.parquet
  results/dt=YYYY-MM-DD/{race_id}.parquet
  inference/dt=YYYY-MM-DD/{race_id}.parquet
  models/v{N}/lgbm.pkl + feature_metadata_v{N}.json
```

### 3-2. スクレイピングスケジュール

| データ種別 | 取得タイミング | 頻度 |
|-----------|--------------|------|
| 出馬表 | 毎朝 06:00（レース3日前〜前日） | 日次 |
| オッズ | 毎 10 分（当日 09:00〜発走） | 高頻度（当日のみ） |
| レース結果 | 毎 30 分（当日 15:00〜18:00） | レース後 |
| AI バッチ推論 | 出馬表確定後 23:00（cron、`nice -n 10`） | 日次 |
| SHAP 計算 | 推論完了後・時間差実行（上位 10 頭） | 日次 |
| モデル再学習 | 毎週月曜 02:00 | 週次 |

> **スクレイパーと推論バッチは同時起動禁止**。cron の時刻を 90 分以上離すこと。

### 3-3. VPS メモリ予算（2GB）

| コンポーネント | 予算 |
|--------------|------|
| OS + systemd | 300 MB |
| Gunicorn ワーカー × 2（preload_app CoW 共有） | 200 MB |
| LightGBM Booster（CoW = 物理 1 コピー） | 200 MB |
| PostgreSQL（shared_buffers 128MB + work_mem 等） | 200 MB |
| Redis（予測結果キャッシュ） | 64 MB |
| スクレイパー（レース日ピーク・時間分離済み） | 300 MB |
| OOM Killer 回避バッファ | 736 MB |
| **ピーク合計（時間分離後）** | **~1,264 MB ✅（2GB の 62%）** |

> LightGBM RSS 増加量 < 250MB、モデルファイルサイズ < 50MB を CI で継続計測。

### 3-4. 主要データモデル

```sql
races      (race_id, date, venue, course, distance, surface, weather, class)
horses     (horse_id, name, sire_id, ...)         -- sire_id: 種牡馬分析に必須
entries    (entry_id, race_id, horse_id, jockey_id, trainer_id, weight, odds)
results    (result_id, entry_id, rank, time, prize)
predictions (pred_id, race_id, horse_id, win_prob, place_prob,
             model_version, inference_source ENUM('local','external'), created_at)

-- インデックス
CREATE INDEX idx_horses_sire_id    ON horses(sire_id);
CREATE INDEX idx_entries_race      ON entries(race_id);
CREATE INDEX idx_predictions_race  ON predictions(race_id);
```

### 3-5. GCS モデルディレクトリ設計

```
gs://keiba-vpn-data/models/
  ├── v{N}/lgbm.pkl                  # LightGBM モデル本体
  ├── v{N}/feature_metadata_v{N}.json # 特徴量スキーマ・バージョン・学習日時
  └── version.json                   # {"current":"v3","previous":"v2"}
```

### 3-6. AI 推論 2 段階戦略

```
[通常時] VPS バッチ推論（cron 23:00）
  ├── LightGBM .pkl ロード（/models/v{N}/lgbm.pkl）
  ├── ETL 完了フラグ確認（Redis キー etl:complete:{date} が必須）
  ├── Pandera スキーマバリデーション
  ├── predict_proba() → predictions テーブル書き込み（inference_source='local'）
  └── SHAP TreeExplainer → 上位 10 頭のみ（時間差実行）

[フォールバック時] 外部 WebAPI（モデル更新中 / VPS 推論失敗時）
  └── inference_source='external' で predictions テーブルに記録
```

### 3-7. フォールバック戦略

```
スクレイピング失敗: 最大3回リトライ（指数バックオフ 30s→60s→120s）
                  → 全失敗時: 前日データで serving 継続（data_freshness: stale）
                  → Slack #ops-alerts 通知

Redis 障害: GCS 直読取にフォールバック（P95 ≤ 200ms → ≤ 400ms に劣化）

AI 推論未完了: HTTP 503 + "推論結果準備中"
              → ETL 未完了: 最大 10 分待機後 TimeoutError + Slack アラート
```

---

## 4. 機能要件（確定版）

### 4-1. データ分析機能

| # | 機能名 | 内容 | 優先度 |
|---|-------|------|--------|
| F-1 | 条件別勝率分析 | 距離・馬場・クラス・季節・騎手・調教師 × 対象馬のフィルタリング集計（勝率・連対率） | 高 |
| F-2 | 単勝回収率分析 | F-1 と同条件に対応した単勝・複勝回収率の算出 | 高 |
| F-3 | コース別好成績種牡馬 | 指定コース × 任意条件での種牡馬ランキング（`horses.sire_id` 必須） | 高 |
| F-4 | レース条件複合フィルタ | 開催場所・距離・馬場・天候・クラス・頭数の AND 条件指定 UI | 高 |
| F-5 | データ鮮度表示 | スクレイピング最終更新タイムスタンプを UI に表示（`data_freshness` フィールド） | 中 |

### 4-2. AI モデル結果表示機能

| # | 機能名 | 内容 | 優先度 |
|---|-------|------|--------|
| F-6 | AI 予測スコア一覧 | バッチ計算済み JSON を出馬表と並列表示（win_probability + confidence_score） | 高 |
| F-7 | 推奨馬ハイライト | 上位スコア馬を色分け・ランク番号で視覚強調 | 高 |
| F-8 | SHAP 説明可能 AI | LightGBM SHAP TreeExplainer による上位 5 特徴量・寄与度を簡易表示（上位 10 頭のみ適用） | 中 |
| F-9 | 予測精度フィードバック | レース後実結果と予測の照合・的中率追跡ダッシュボード | 低 |
| F-10 | 推論ソース表示 | バッチ推論 / 外部 WebAPI フォールバック区別を UI に表示（`inference_source` フラグ） | 中 |

### 4-3. 認証・共通機能

| # | 機能名 | 内容 | auth_required |
|---|-------|------|--------------|
| F-11 | ユーザ認証 | Flask-Login セッション認証（ログイン・ログアウト） | No（ログインページ） |
| F-12 | 全分析 API 認証ガード | F-1〜F-10 すべてに `@login_required` デコレータ適用 | Yes（全 API） |
| F-13 | ヘルスチェック | `/health` エンドポイント | No |

### 4-4. インフラ・パイプライン機能

| # | 機能名 | 内容 | 優先度 |
|---|-------|------|--------|
| F-14 | スクレイピング ETL パイプライン | JRA サイトから出馬表・オッズ・結果を収集し GCS + ローカル DB に保存 | 高 |
| F-15 | バッチ推論パイプライン | 毎夜 LightGBM でバッチ推論を実行し predictions テーブルに保存 | 高 |
| F-16 | ETL 完了フラグガード | Redis キー `etl:complete:{race_date}` を使い推論バッチの早期起動を防止 | 高 |
| F-17 | Pandera バリデーション | 特徴量の型・値域（距離 800〜3600m、頭数 2〜18 頭、馬体重 380〜620kg 等）を検証 | 高 |
| F-18 | モデルバージョン管理 | GCS `version.json` で current/previous を管理し、ロールバックを可能にする | 中 |
| F-19 | スクレイピング失敗アラート | Slack Webhook でリアルタイム通知 | 中 |
| F-20 | オッズ変動差分再推論 | ±15% 超変動時のみ差分再推論（1 日最大 5 回のハードリミット） | 低 |

### 4-5. 確定済み API エンドポイント

| メソッド | エンドポイント | 機能 | auth_required |
|--------|-------------|------|--------------|
| GET | `/health` | ヘルスチェック | No |
| GET/POST | `/login` | ログイン | No |
| POST | `/logout` | ログアウト | Yes |
| POST | `/api/v1/filter/stats` | 条件別勝率・回収率 | Yes |
| GET | `/api/v1/sires/ranking` | コース別種牡馬ランキング | Yes |
| GET | `/api/v1/predictions/{race_id}` | AI 予測スコア一覧 | Yes |
| GET | `/api/v1/shap/{race_id}` | SHAP 説明値 | Yes |
| GET | `/api/v1/races/{race_id}/entries` | 出馬表 + AI スコア統合 | Yes |

> **[DEC-005 vs DEC-008 の矛盾解決]**: DEC-005 は `/api/v1/races/today` と `/api/v1/predictions/{race_id}` を Public と仮設定し認証要否を未解決 (REVIEW-001) とした。DEC-008 が「全分析 API に @login_required 適用」と明確化した。DEC-008 が勝つ。

---

## 5. 非機能要件（確定版）

### 5-1. パフォーマンス SLO

| # | 要件 | 目標値 | 計測方法 |
|---|------|--------|---------|
| N-1 | API P95 レイテンシ（Redis キャッシュ HIT） | ≤ 200ms | Nginx access log |
| N-2 | API P99 レイテンシ | ≤ 500ms | Nginx access log |
| N-3 | E2E P95 レスポンスタイム（国内ブラウザ） | < 1,200ms | Synthetic Monitor |
| N-4 | データ分析 API レスポンス（キャッシュ未使用） | < 1.5 秒 | アプリログ |
| N-5 | AI スコア取得レスポンス | < 200ms | アプリログ |
| N-6 | AI 推論 P99（オンデマンド読み取り） | ≤ 30ms（Redis から） | アプリ内計測 |
| N-7 | Web API レスポンス（Redis 読み取り） | P50 < 10ms、P99 < 30ms | Redis INFO stats |
| N-8 | フロントエンド初期表示 LCP | < 2.5 秒 | Vercel Analytics |
| N-9 | バッチ推論完了時間（全レース・約 200 頭） | < 5 分 / 夜間バッチ | cron ログ |
| N-10 | ETL 処理完了時間 | レース開始 2 時間前 | cron ログ |
| N-11 | スクレイピング成功率 | 月次 ≥ 98% | `scrape_runs` テーブル |

### 5-2. リソース・可用性

| # | 要件 | 目標値 |
|---|------|--------|
| N-12 | VPS メモリ使用率（全プロセス合計） | ≤ 85%（1.7GB / 2GB）、ピーク < 1,264MB |
| N-13 | CPU 使用率（バッチ推論実行中） | ≤ 50%（`num_threads = vCPU // workers = 1`） |
| N-14 | Redis キャッシュ HIT 率 | ≥ 80%（レース日） |
| N-15 | LightGBM RSS 増加量 | < 250MB（CI で毎 PR 計測） |
| N-16 | モデルファイルサイズ | < 50MB（CI PR ゲート） |
| N-17 | Gunicorn ワーカー数 | 2（2vCPU 環境）、`max_requests=500 / max_requests_jitter=50` |
| N-18 | サービス月間稼働率 | ≥ 99.5%（ダウンタイム ≤ 3.6h/月） |
| N-19 | ETL → 推論最大待機タイムアウト | 10 分（超過時 Slack アラート + ジョブ停止） |
| N-20 | GCS データ鮮度 | 最大 24 時間遅延 |
| N-21 | API データ欠損率 | ≤ 0.5% |

### 5-3. モデル品質

| # | 要件 | 目標値 |
|---|------|--------|
| N-22 | 単勝予測的中率（初期リリース目標） | ≥ 20%（中期目標 ≥ 30%） |
| N-23 | Logloss（確率品質） | ≤ 2.2 |
| N-24 | Calibration Error | ≤ 0.05 |
| N-25 | ROI（単勝） | ≥ −15% |
| N-26 | 時系列リーク防止 | validation レースの日付より厳密に前の cutoff を強制（CI バリデーション必須） |

### 5-4. セキュリティ・レート制限

| # | 要件 | 目標値 |
|---|------|--------|
| N-27 | 予測系エンドポイントのレート制限 | 60 req/min/IP |
| N-28 | 集計クエリエンドポイントのレート制限 | 20 req/min/user |
| N-29 | Redis `maxmemory-policy` | allkeys-lru（OOM 自動 eviction） |
| N-30 | systemd 自動再起動 | `Restart=on-failure`, `RestartSec=5s` |

---

## 6. AI / ML パイプライン

### 6-1. モデル概要

- **ベースモデル**: LightGBM（`predict_proba`、勝率・連対率スコアリング）
- **SHAP 説明**: TreeExplainer（上位 10 頭に適用、上位 5 特徴量を自然言語表示）
- **ModelRegistry**: `RWLock` パターン。プロセス起動時に 1 回のみモデルをロード。GCS は `generation` メタデータ比較による差分ダウンロード

### 6-2. SHAP 自然言語マッピング（最低限定義）

```python
SHAP_TO_TEXT = {
    "recent_avg_rank":  "過去成績: 直近{n}走平均着順 {val}位",
    "track_win_rate":   "馬場適性: {surface}馬場勝率 {val:.0%}",
    "jockey_combo_wr":  "騎手相性: このコンビ過去{n}走勝率 {val:.0%}",
    "weight_diff":      "斤量変化: 前走比 {val:+.1f}kg",
    "odds_rank":        "オッズ: 現在{val}番人気",
}
```

### 6-3. CI 必須チェック

- `assert model_size_mb < 50`
- LightGBM RSS delta < 250MB（`memory_profiler` で計測）
- 時系列リークバリデーション（cutoff 厳密確認）
- Pandera スキーマバリデーション（距離 800〜3600m、頭数 2〜18 頭、馬体重 380〜620kg）

### 6-4. ModelRegistry 抜粋

```python
# inference/model_registry.py — バッチワーカープロセス専用（Web プロセスから使用禁止）
LGBM_THREADS = max(1, VCPU_COUNT // WORKER_COUNT)  # 2vCPU → 1

class ModelRegistry:
    def __init__(self, model_path: str): ...
    def reload(self, new_path=None) -> int: ...      # ダウンタイムゼロ更新（RWLock）
    def predict(self, features: np.ndarray) -> np.ndarray: ...
```

---

## 7. 運用コスト

### 7-1. 月次コスト確定値（VPS 固定費モデル）

| # | サービス | 用途 | 月額概算 |
|---|---------|------|---------|
| C-1 | **ConoHa VPS 2GB / 100GB SSD** | Flask API・AI 推論バッチ・cron・Redis・WireGuard 全同居 | **¥1,320** |
| C-2 | **Google Cloud Storage** | Parquet 過去データ長期保存・バックアップ | **¥200〜500** |
| C-3 | **Vercel（Next.js）** | フロントエンドホスティング | **¥0**（Hobby プラン） |
| C-4 | WireGuard VPN | スクレイピング出口 IP 管理 | ¥0（VPS 内同居） |
| C-5 | Redis 7 | 分析クエリキャッシュ（maxmemory 256MB） | ¥0（VPS 内同居） |
| C-6 | Slack Webhook | 障害・ETL 失敗アラート | ¥0（無料プラン） |
| — | **合計（概算）** | — | **¥1,520〜1,820 / 月** |

### 7-2. 将来スケール時のコスト移行パス

| フェーズ | 移行トリガー | 追加コスト目安 |
|---------|------------|--------------|
| Phase 2 | 同時ユーザー 50 人超 → 推論を Cloud Run へ移行 | +¥0〜¥375/月（Cloud Run 無料枠内） |
| Phase 3 | GPU モデル移行 / 同時ユーザー 500 人超 → Vertex AI へ | +¥750〜¥22,500/月 |
| フロント商用化 | トラフィック増大 → Vercel Pro | +¥2,500/月 |

> **Vertex AI / SageMaker 常時起動エンドポイントは現規模では禁止**（最低 ¥7,500/月〜が最低ライン）。Phase 3 条件を満たすまで使用しないこと（DEC-006）。

---

## 8. 未解決事項・Human 判断待ち

以下はすべての DEC ファイルから抽出した、まだ解決されていない open question である。

| # | 事項 | 重要度 | 参照 DEC |
|---|------|--------|---------|
| **Q-1** | **JRA/netkeiba スクレイピングの利用規約・robots.txt との法的適合確認**（ETL パイプライン実装前に必須） | 高 | DEC-004, DEC-005, DEC-008 |
| **Q-2** | **VPS 2GB メモリ割り当て表の実測値検証**（SHAP 計算の時間差実行設計が実際の運用で成立するか、Phase 2 開始前に負荷テスト実施を推奨） | 高 | DEC-008 |
| **Q-3** | **PostgreSQL `shared_buffers` の実設定値確認**（予算表は 128MB を想定しているが、実設定に依存。低い場合は 64MB に削減して調整） | 中 | DEC-009 |
| **Q-4** | **ETL 遅延時のフォールバック運用ポリシー**（前日データ暫定表示 vs. 503 一律返却 — どちらを採用するか未確定） | 中 | DEC-009 |
| **Q-5** | **Vercel Hobby プラン将来リスクへの対応方針**（トラフィック増加時の Vercel 課金ポリシー変更への対応。代替: Cloudflare Pages） | 低 | DEC-004 |
| **Q-6** | **AI モデル精度基準の実績ベースライン確認**（N-22: 複勝的中率 ≥ 35% の目標値が現実的か、初回モデル訓練後に検証） | 低 | DEC-007 |
| **Q-7** | **スクレイピング対象の出口 IP（WireGuard VPN）に関する ISP ポリシー確認** | 低 | DEC-008 |

---

## 9. 参照 DEC 一覧

| DEC | 日付 | 一行サマリー |
|-----|------|------------|
| DEC-001 | 2026-07-02 | フロントエンド TypeScript/Next.js 14・バックエンド Python/FastAPI・DB PostgreSQL+Redis の技術スタックを選定 |
| DEC-002 | 2026-07-02 | GCS(原本)・Cloud SQL(OLTP)・BigQuery(分析) の3層データアーキテクチャを確定（Firestore 廃止） |
| DEC-003 | 2026-07-02 | 2GB VPS 制約下でのサーバ負荷最小化設計（GCS 署名付き URL・Redis キャッシュ・SSG/CSR/ISR 使い分け）を定義 |
| DEC-004 | 2026-07-02 | スクレイパー・推論バッチを Web サーバから完全分離し VPS アイドルメモリを ~543MB に削減する Phase 1 最優先実施事項を確定 |
| DEC-005 | 2026-07-02 | 26 機能要件・15 非機能要件を統合した要件定義書 v1.0 を確定（API 認証要否 REVIEW-001 は後続フェーズへ） |
| DEC-006 | 2026-07-02 | AI 推論は VPS 内 Celery バッチ（月額 ¥10 未満）で実装し、外部 API（Vertex AI 等）は現規模では採用しないと決定 |
| DEC-007 | 2026-07-03 | ConoHa VPS 固定費モデルを前提にコスト・パフォーマンス選択肢を整理し、プラン 2（Vercel+VPS 内 Redis+cron）を推奨 |
| DEC-007（続） | 2026-07-03 | トークン上限で切断された DASHBOARD.md と thinking-logs を補完し、要件定義書一式を完成 |
| DEC-008 | 2026-07-03 | Flask+LightGBM バッチ+Redis を VPS 一台に同居させる ConoHa VPS 固定費モデル（¥1,520〜1,820/月）を全エージェント合意で確定 |
| DEC-009 | 2026-07-03 | LightGBM 推論の Web プロセス完全分離・ETL 完了フラグガード・num_threads 上限固定・preload_app CoW・Pandera バリデーションを必須実装と確定 |
