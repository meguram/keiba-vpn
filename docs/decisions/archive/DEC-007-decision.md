# DEC-007: decision

**Date**: 2026-07-03
**Agent**: web-search-agent, decisions-context-agent, proposal-agent, backend-engineer, data-engineer, ai-model-engineer, quality-reviewer
**Task**: TASK-025
**Status**: ACCEPTED

---

## Context

修正内容を確認してください: conoha vpsをすでに契約中ですので、そのコストはここで計算に入れなくて大丈夫です。cpuメモリ2GB / SSD100GB のvpsを使用しています。それ以外のコストとパフォーマンスの選択肢を複数提示するようにしてください。そこから最適案を選びます。

---

## Decision

<response>
# 要件定義書: keiba-vpn — 競馬予測 Web アプリケーション

> **バージョン**: v1.1  
> **作成日**: 2025-01-31  
> **ステータス**: `PENDING_REVIEW`  
> **前提**: ConoHa VPS（メモリ 2GB / SSD 100GB）は契約済み固定費のため、追加コスト計算から除外する

---

## サマリー

keiba-vpn は日本競馬（JRA）の出馬表・オッズ・AI予測スコアをマルチユーザーへ提供する Web アプリケーションである。スクレイピング → ETL → AI推論 → API → UI というデータパイプラインを運用し、**月額追加コスト最小化・P95 API レイテンシ ≤ 200ms** を非機能目標の基準とする。VPS 既契約を前提に、**各ドメインで「VPS 上で動かす案」と「クラウドサービスを使う案」を複数提示**し、最適案をここで選択できるようにする。

---

## 1. ドメイン別技術スタック定義表

> VPS 内で実行する選択肢は「追加コスト ¥0」として扱う。GCP 外部サービスの利用分のみコスト計上する。

| ドメイン | 言語 / ランタイム | 実行環境 | レイテンシ目標 | 追加月額コスト | 選定根拠 |
|---|---|---|---|---|---|
| **フロントエンド** | TypeScript / Next.js 14 (SSR+SSG) | ※選択肢あり（後述） | FCP ≤ 1.5s / CDN HIT ≤ 50ms | ¥0〜¥1,500 | Core Web Vitals 準拠 |
| **API サーバー** | Python 3.11 / Flask 3.x + Gunicorn 21.x | **VPS 内**（Nginx リバースプロキシ） | P50 ≤ 80ms / P95 ≤ 200ms | **¥0** | 既契約 VPS で賄える・既存コード資産 |
| **キャッシュ層** | — | ※選択肢あり（後述） | GET P50 ≤ 1ms / P99 ≤ 5ms | ¥0〜¥4,500 | GCS 毎回読取回避 |
| **データストア** | — | GCS Standard / us-central1 | 読取 P50 ≤ 50ms / P99 ≤ 200ms | Storage ~¥200 / Ops ~¥50 / Egress ~¥1,000 | フルマネージド・構造化不要（DEC-002 確定） |
| **ETL / スクレイピング** | Python 3.11 / Scrapy 2.x + httpx | ※選択肢あり（後述） | バッチ処理（SLA 対象外） / JRA 応答実測 800ms〜2s | ¥0〜¥300 | DEC-004 バッチ分離方針 |
| **AI 推論** | Python 3.11 / LightGBM + scikit-learn | **VPS 内**（夜間バッチ、メモリ ~200MB） | バッチ全量 ≤ 10分 | **¥0** | GPU 不要・実測 ~12ms/レース・VPS 2GB 制約内 |
| **認証 / VPN** | — | WireGuard on VPS | VPN トンネル遅延 ≤ 50ms | **¥0** | VPS 上で完結 |
| **監視 / 観測性** | — | ※選択肢あり（後述） | アラート発火 ≤ 60s | ¥0〜¥700 | — |

---

### 1-1. Gunicorn Worker 設定（API サーバー on VPS）

```
worker_class : gthread         # I/O バウンドな GCS / Redis アクセスに適合
workers      : 2               # VPS 2コア想定 × 1
threads      : 4               # スレッドで並行 I/O 処理
timeout      : 30              # GCS P99 200ms に対して十分なマージン
```

### 1-2. Redis キャッシュ対象と収容試算

| 優先度 | エンドポイント / データ | TTL | 想定サイズ |
|---|---|---|---|
| 1 | `/api/v1/races/today`（レース一覧） | 300s | ~50KB/レース |
| 2 | `/api/v1/horses/{id}`（馬情報） | 3600s | ~5KB/頭 |
| 3 | `/api/v1/predictions`（AI推論結果） | 600s | ~10KB/レース |

> **収容試算**: レース(5MB) + 馬(5MB) + 推論(1MB) ≈ **11MB** → どの構成でも余裕あり

### 1-3. AI 推論 Cadence 区別

| 種別 | タイミング | 目標 | SLA |
|---|---|---|---|
| バッチ推論 | 夜間1回（出馬表確定後） | 全レース処理完了 ≤ 10分 | 対象外 |
| API 推論（オンデマンド） | リクエスト時 | P50 ≤ 30ms / P99 ≤ 80ms | 対象（キャッシュ前提） |

---

## 2. ドメイン別 選択肢比較表

> 以下の各ドメインで **複数の選択肢とトレードオフ** を示す。「**推奨案**」は現時点の試算に基づく最適案だが、最終決定はここで行う。

---

### 2-A. フロントエンド配信

| 選択肢 | 構成 | 追加月額コスト | P95 FCP | メリット | デメリット |
|---|---|---|---|---|---|
| **A-1 VPS 直接配信** | Nginx on VPS で Next.js ビルド済み静的ファイルを配信 | **¥0** | ~300ms（VPS〜ブラウザ依存） | 追加コストゼロ | CDN なし・VPS 負荷が増える |
| **A-2 Vercel 無料枠** | Vercel Hobby プラン（個人・非商用） | **¥0** | ~100ms（CDN エッジ） | CDN 配信・デプロイ自動化 | 商用利用制限・SSR は VPS API に依存 |
| **A-3 Cloud CDN + Cloud Run** | Next.js SSR を Cloud Run で実行・CDN キャッシュ | ~¥800 | ~80ms（CDN HIT） | フルマネージド SSR | コスト発生・構成複雑 |
| ⭐ **A-4 Vercel Pro** | Vercel Pro プラン | ~¥2,500 | ~80ms（CDN エッジ） | 商用利用可・帯域制限なし | 月額固定費発生 |

> **推奨案**: **A-2（Vercel 無料枠）** — 個人・小規模利用であれば追加コスト¥0 で CDN 配信が可能。商用化時は A-4 へ移行。

---

### 2-B. キャッシュ層（Redis）

| 選択肢 | 構成 | 追加月額コスト | GET P99 | メリット | デメリット |
|---|---|---|---|---|---|
| **B-1 Redis on VPS** | VPS 内に Redis 7 をインストール（メモリ ~100MB 消費） | **¥0** | ≤ 2ms（ローカル） | 追加コストゼロ・超低レイテンシ | VPS メモリを 100MB 消費（残 ~1.9GB）・運用負荷 |
| **B-2 Cloud Memorystore** | GCP マネージド Redis 7 / 1GB / Basic tier | ~¥4,500 | ≤ 5ms（同一リージョン） | フルマネージド・高可用性 | 月¥4,500 は全体コストの大半を占める |
| **B-3 Upstash Redis** | サーバーレス Redis（無料枠: 10,000 req/日） | ¥0〜¥800 | ~10ms（HTTP API 経由） | 無料枠で小規模対応・従量課金 | HTTP オーバーヘッドあり・大量リクエスト時コスト増 |
| **B-4 キャッシュなし** | 毎リクエスト GCS 直読取 | **¥0** | 50〜200ms（GCS 依存） | 構成シンプル | GCS Ops コスト増・レイテンシ高 |

> **推奨案**: **B-1（Redis on VPS）** — VPS メモリ 2GB のうち Redis が消費するのは ~100MB であり問題なし。追加コスト¥0 で最低レイテンシを実現。VPS 障害 = キャッシュ障害となるリスクは B-4（GCS フォールバック）で吸収する。

---

### 2-C. ETL / スクレイピングバッチ実行基盤

| 選択肢 | 構成 | 追加月額コスト | 実行遅延 | メリット | デメリット |
|---|---|---|---|---|---|
| **C-1 cron on VPS** | VPS 上の cron + Systemd でバッチ実行 | **¥0** | ≤ 1分（cron 精度） | 追加コストゼロ・シンプル | VPS リソースを消費（スクレイプ中 CPU/メモリ高騰） |
| **C-2 Cloud Run Jobs + Scheduler** | バッチを Cloud Run Jobs に分離・Cloud Scheduler でトリガー | ~¥300 | コールドスタート ~30s | VPS 負荷ゼロ・独立スケール | 毎回コールドスタート・Playwright 使用時は起動 ~60s |
| **C-3 Cloud Run Jobs（min-instances=1）** | C-2 に加え最小1インスタンス常時起動 | ~¥800 | ~5s（ウォームスタート） | コールドスタート解消 | 常時起動コスト増加 |

> **推奨案**: **C-1（cron on VPS）** — DEC-004 の「バッチをWebサーバから独立プロセスとして切り離す」方針はプロセス分離であり、必ずしも別インフラを要しない。VPS 内で API サーバーと別プロセスとして cron 実行すれば追加コスト¥0。CPU 高騰が問題になった場合に C-2 へ移行する（ADR-004 見直しトリガー）。

---

### 2-D. 監視 / 観測性

| 選択肢 | 構成 | 追加月額コスト | アラート発火 | メリット | デメリット |
|---|---|---|---|---|---|
| **D-1 自前監視（VPS）** | Prometheus + Grafana on VPS（メモリ ~200MB 消費） | **¥0** | ≤ 30s | 追加コストゼロ・カスタマイズ自由 | VPS メモリ消費・初期構築コスト高 |
| **D-2 Cloud Monitoring** | GCP マネージド監視・カスタムメトリクス 5指標 | ~¥700 | ≤ 60s | フルマネージド・GCP サービスと統合 | 月¥700 の追加コスト |
| **D-3 UptimeRobot 無料** | 外形監視のみ（URL 死活・5分間隔） | **¥0** | ≤ 5分 | 追加コストゼロ・設定簡単 | 内部メトリクス不可・アラート粒度粗い |
| **D-4 D-3 + D-1 組合せ** | 外形監視(UptimeRobot) + 内部メトリクス(Prometheus on VPS) | **¥0** | 外形 ≤ 5分 / 内部 ≤ 30s | フル可視化・追加コストゼロ | VPS メモリ ~200MB 追加消費 |

> **推奨案**: **D-3（UptimeRobot 無料）＋ 必要に応じ D-1 追加** — 初期フェーズは外形監視のみで十分。ユーザー増加後に Prometheus を追加する段階的アプローチ。

---

## 3. コストシナリオ（VPS 追加コストのみ・選択肢別）

> **前提**: ConoHa VPS（¥1,000〜¥2,000/月程度）は固定費として計算外。GCS は DEC-002 確定のため全案共通で計上。

### 3-A. GCS（全構成共通・変動なし）

| 項目 | 月額コスト | 備考 |
|---|---|---|
| GCS Storage | ~¥200 | 10GB Standard |
| GCS Class A Ops | ~¥50 | 10万回/月 |
| GCS Egress | ~¥1,000 | 50GB/月（キャッシュ有効時は減少） |
| **GCS 小計** | **~¥1,250** | 全構成共通 |

---

### 3-B. 構成パターン別 追加コスト比較

| 構成パターン | フロントエンド | キャッシュ | ETL/バッチ | 監視 | GCS 共通 | **月額追加合計** |
|---|---|---|---|---|---|---|
| **プラン 1: フル VPS 内完結** | A-1 VPS 直接 ¥0 | B-1 Redis on VPS ¥0 | C-1 cron ¥0 | D-3 UptimeRobot ¥0 | ¥1,250 | **~¥1,250** |
| **プラン 2: フロントのみ Vercel** | A-2 Vercel 無料 ¥0 | B-1 Redis on VPS ¥0 | C-1 cron ¥0 | D-3 UptimeRobot ¥0 | ¥1,250 | **~¥1,250** |
| **プラン 3: バランス型** | A-2 Vercel 無料 ¥0 | B-1 Redis on VPS ¥0 | C-1 cron ¥0 | D-2 Cloud Monitoring ¥700 | ¥1,250 | **~¥1,950** |
| **プラン 4: バッチ分離型** | A-2 Vercel 無料 ¥0 | B-1 Redis on VPS ¥0 | C-2 Cloud Run ¥300 | D-2 Cloud Monitoring ¥700 | ¥1,250 | **~¥2,250** |
| **プラン 5: フルクラウド型** | A-3 Cloud CDN+Run ¥800 | B-2 Memorystore ¥4,500 | C-2 Cloud Run ¥300 | D-2 Cloud Monitoring ¥700 | ¥1,250 | **~¥7,550** |

> **ユーザー規模別スケール**: 上記コストは ~1,000 セッション/月の試算。10,000 セッション超過時は GCS Egress (~¥5,000)・Cloud Run (該当プランのみ) が増加。

---

### 3-C. 各プランのトレードオフサマリー

| プラン | 追加月額 | VPS メモリ消費 | 運用負荷 | レイテンシ品質 | スケール余力 |
|---|---|---|---|---|---|
| プラン 1 | ~¥1,250 | 高（Redis+API+バッチ同居） | 高（全自己管理） | 中（CDN なし） | 低 |
| **プラン 2** ⭐ | ~¥1,250 | 中（Redis+API+バッチ同居） | 中 | **高（Vercel CDN）** | 中 |
| プラン 3 | ~¥1,950 | 中 | 低（監視マネージド） | 高 | 中 |
| プラン 4 | ~¥2,250 | 低（バッチ分離） | 低 | 高 | 高 |
| プラン 5 | ~¥7,550 | 低（全分離） | 最低 | 最高 | 最高 |

> ⭐ **現時点の推奨案: プラン 2**  
> - 追加コスト ¥1,250/月（GCS 費用のみ）で運用可能  
> - Vercel 無料枠で CDN 配信・デプロイ自動化を確保  
> - Redis・API・バッチはすべて VPS 内で完結  
> - 商用化・ユーザー増加時はプラン 3→4 へ段階移行  

---

## 4. エンドツーエンド レイテンシ予算

**E2E 目標: P95 < 1,200ms（PC ブラウザ・国内回線）**

```
┌──────────────────────────────────────────────────────────────┐
│ レイヤー              │ 上限(P95)  │ 計測方法                │
├───────────────────────┼────────────┼─────────────────────────┤
│ CDN → HTML/JS 配信    │   80 ms    │ Vercel Analytics        │
│ Next.js SSR           │  150 ms    │ Server Timing API       │
│ VPN トンネル          │   50 ms    │ WireGuard latency       │
│ Flask API 処理        │  200 ms    │ Nginx access log        │
│ Redis キャッシュ取得  │    2 ms    │ Redis INFO stats        │
│ AI 推論（キャッシュ時）│    5 ms    │ アプリ内計測            │
│ GCS データ読み込み    │  150 ms    │ GCS Access Log          │
│ ネットワーク往復      │  170 ms    │ Synthetic Monitor       │
├───────────────────────┼────────────┼─────────────────────────┤
│ 合計（概算）          │  807 ms    │                         │
│ バジェット余裕        │  393 ms    │ P95 < 1,200ms に対して  │
└───────────────────────┴────────────┴─────────────────────────┘
```

---

## 5. 機能要件

| # | 要件 | 優先度 | 担当 |
|---|---|---|---|
| F-1 | 出馬表（レース・馬・騎手・オッズ）の一覧表示 | 高 | frontend-engineer |
| F-2 | レース・馬名・日付によるフィルタリング機能 | 高 | frontend-engineer |
| F-3 | AI 予測スコアの出馬表横表示 | 高 | frontend-engineer + ai-model-engineer |
| F-4 | JRA サイトからのスクレイピングバッチ（1日2回以上） | 高 | data-engineer |
| F-5 | ETL / 前処理パイプライン（スクレイピングデータ → 特徴量変換） | 高 | data-engineer |
| F-6 | LightGBM モデルによる勝率・連対率スコアリング | 高 | ai-model-engineer |
| F-7 | REST API `/api/v1/races` `/api/v1/predictions` `/api/v1/horses` | 高 | backend-engineer |
| F-8 | マルチユーザー認証（WireGuard VPN アクセス制御） | 高 | backend-engineer |
| F-9 | レース統計サマリ（ユーザー別集計・フィルタ保存） | 中 | fullstack-integrator |
| F-10 | データ鮮度インジケーター（最終更新時刻・staleness 表示） | 中 | frontend-engineer |
| F-11 | モデルバージョン切替（環境変数 `MODEL_VERSION` 制御） | 中 | ai-model-engineer |
| F-12 | スクレイピング失敗時の Slack 通知 | 中 | operations-engineer |
| F-13 | 管理者向けパイプライン状態ダッシュボード | 低 | fullstack-integrator |

---

## 6. 非機能要件

| # | 要件 | 目標値 | 計測方法 | 担当 |
|---|---|---|---|---|
| N-1 | API P95 レイテンシ | ≤ 200ms（キャッシュ HIT 前提） | Nginx access log | operations-engineer |
| N-2 | API P99 レイテンシ | ≤ 500ms | Nginx access log | operations-engineer |
| N-3 | E2E P95 レスポンスタイム | < 1,200ms（国内ブラウザ） | Synthetic Monitor | operations-engineer |
| N-4 | AI 推論 P99（オンデマンド） | ≤ 80ms | アプリ内計測 | ai-model-engineer |
| N-5 | スクレイピング成功率 | 月次 ≥ 98% | `scrape_runs` テーブル監視 | data-engineer |
| N-6 | ETL 処理完了時間 | レース開始 2 時間前 | cron ログ | data-engineer |
| N-7 | GCS データ鮮度 | 最大 24 時間遅延 | GCS メタデータ | data-engineer |
| N-8 | API データ欠損率 | ≤ 0.5% | アプリログ | backend-engineer |
| N-9 | 月額追加運用コスト（〜1,000 セッション） | ≤ ¥2,000（プラン2〜3基準） | GCP Billing Alert | cost-optimizer |
| N-10 | アラート発火まで | ≤ 5分（UptimeRobot）/ ≤ 30s（Prometheus導入後） | 外形監視 | operations-engineer |
| N-11 | バッチ推論完了時間（全レース） | ≤ 10 分/夜間バッチ | cron ログ | ai-model-engineer |
| N-12 | Redis キャッシュ HIT 率 | ≥ 80%（レース日） | Redis INFO stats | backend-engineer |
| N-13 | VPS メモリ使用率（全プロセス合計） | ≤ 85%（1.7GB / 2GB） | Prometheus / top | operations-engineer |

---

## 7. データパイプライン SLA & フォールバック戦略

```
【スクレイピング失敗時】
  → cron でリトライ（最大3回、指数バックオフ + ジッター）
  → 全リトライ失敗: GCS の前回データで serving 継続（stale-while-revalidate）
  → ユーザー向けレスポンス: 200 OK + { "data_freshness": "stale", "last_updated": "<timestamp>" }
  → Slack 通知 #ops-alerts チャンネル

【Redis 障害時】
  → GCS 直読取にフォールバック（レイテンシ一時悪化を許容）
  → ユーザー影響: P95 ≤ 200ms → P95 ≤ 400ms 程度に劣化

【GCS 読み込み失敗時】
  → Redis キャッシュからフォールバック提供
  → Redis も失敗: 503 + ステータスページに表示

【AI 推論タイムアウト時（P99 > 80ms）】
  → ルールベース簡易スコア（人気順 × 騎手勝率）にデグレード
  → フロントエンドに "simplified_score" フラグで明示

【JRA 外部依存のサーキットブレーカー】
  → 5xx 連続 3 回 → OPEN 状態（60 秒）
  → OPEN 中は GCS キャッシュデータのみ返却
```

---

## 8. ADR（アーキテクチャ決定記録）サマリー

| ADR-ID | 決定内容 | 採用しなかった選択肢 | 見直しトリガー |
|---|---|---|---|
| ADR-001 | バックエンド: Flask 3.x on VPS | FastAPI（移行コスト大）, Django（オーバースペック） | 同時接続 500 超過 → FastAPI 移行検討 |
| ADR-002 | データストア: GCS | BigQuery（コスト過剰）, Firestore（構造制約） | 月次クエリ分析ニーズ発生 → BigQuery 追加 |
| ADR-003 | AI 推論: LightGBM on VPS | XGBoost（同等）, Deep Learning（GPU コスト大） | 特徴量 100 超過 → Neural ODE 検討 |
| ADR-004 | バッチ実行: cron on VPS | Cloud Run Jobs（VPS 負荷増大 or コスト許容時に移行） | ETL 処理時間 30 分超過 → Cloud Run Jobs 移行 |
| ADR-005 | フロントエンド: Next.js 14 + Vercel | Vite + React SPA（SEO 不利）, Remix（学習コスト） | 商用化 → Vercel Pro ($20/月) へ移行 |

---

## 9. 実装ロードマップ

### Phase 1 — コア機能（優先度: 高）

| タスク | 担当 | 成果物 |
|---|---|---|
| VPS 環境構築（Nginx + Gunicorn + Flask） | backend-engineer | `nginx.conf` + `systemd` サービス定義 |
| GCS バケット + IAM 設計 | backend-engineer | セットアップスクリプト |
| Scrapy スクレイピングバッチ実装（cron） | data-engineer | cron ジョブ + Python スクリプト |
| ETL パイプライン（特徴量変換） | data-engineer | Python スクリプト + GCS 書込み |
| Redis on VPS セットアップ | backend-engineer | Redis 設定ファイル + キャッシュ層実装 |
| LightGBM 初期モデル訓練・デプロイ | ai-model-engineer | モデル `.pkl` + GCS `/models/v1/` |
| API エンドポイント実装 (`/races`, `/predictions`) | backend-engineer | OpenAPI 仕様書 |
|

---

## Conclusion

****

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-03_
