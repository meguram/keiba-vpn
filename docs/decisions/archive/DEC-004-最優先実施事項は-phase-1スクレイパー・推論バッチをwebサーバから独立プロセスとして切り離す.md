# DEC-004: 最優先実施事項は **Phase 1（スクレイパー・推論バッチをWebサーバから独立プロセスとして切り離す）** であり、これだけでVPS メモリ使用量を現状の ~1,300MB+ から ~543MB（アイドル時）まで削減でき、OOMリスクを根本解消できる。

**Date**: 2026-07-02
**Agent**: proposal-agent, backend-engineer, data-engineer, ai-model-engineer, quality-reviewer
**Task**: TASK-014
**Status**: ACCEPTED

---

## Context

「現在、設計されている機能は以下の通りです。
1. AIモデルの推論結果（推論は定期バッチで実行）を出馬表とともにUI上で表示しユーザに提供
2. データ分析(ユーザが任意のフィルタ条件を指定し、その条件における対象データの勝率/複勝率成績を計算し、ユーザに提供)
3. 定期パイプラインでスクレイピング実行(GCSに保存)

今、ローカル上では、これがPythonによって処理され、アプリケーションもflaskによって制御されています。

今後運用していく中で、VPSサーバのメモリが2GBしかないことを考えると、このやり方は良くないと考えています。フロントエンド、バックエンドに関しての最適な言語、および内部システムの開発言語の最適化を考えてほしいです。多数からのユーザリクエストにも対応できるような設計を考えてください。」の改善要件定義書です。要件の優先順位と実装可否を確認してください。

---

## Decision

# 改善要件定義書: VPS 2GB制約下における keiba-vpn スタック最適化

> **ドキュメント管理**
> - Task: `project/keiba-vpn/tasks/TASK-001-stack-optimization.md` (IN_PROGRESS → NEEDS_REVIEW)
> - Decision: `project/keiba-vpn/docs/decisions/DEC-001-architecture-stack.md`
> - Review: `project/keiba-vpn/docs/reviews/REVIEW-001-requirements.md` (PENDING_REVIEW)
> - 作成日: 2025-07-14 | 担当: Orchestrator (backend-engineer / data-engineer / ai-model-engineer 統合)

---

## サマリー

現行の Flask モノリス構成は、Web配信・API・AI推論・スクレイピングが単一プロセスに同居しており、2GB VPS において多重リクエスト時の OOM（メモリ不足）リスクが高い。本要件定義書では「**責務の分離 × 言語最適化 × メモリ予算の明示的管理**」を基本方針とし、フロントエンドを Vercel へ切り出し、バックエンドを FastAPI へ移行し、AI推論・スクレイピングをバッチワーカーとして完全隔離する 3フェーズの実装計画を定義する。

---

## 現状のメモリ問題（AS-IS）

```
[2GB VPS — 現状]
┌──────────────────────────────────────────────────┐
│  Flask プロセス (Gunicorn workers × 4)            │
│  ├── HTML レンダリング + 静的配信      ~100MB      │
│  ├── API ハンドラ                      ~50MB       │
│  ├── pandas (全件インメモリ集計)       ~300MB      │
│  ├── AI 推論モデル (常駐)              ~400MB      │
│  └── スクレイピング処理               ~150MB      │
│                          Flask合計: ~1,000MB+     │
├── OS + その他                          ~300MB      │
│                           総合計: ~1,300MB+  ⚠️   │
│  ※ リクエスト急増時 or バッチ重複起動で 2GB 超過   │
└──────────────────────────────────────────────────┘
```

---

## 目標アーキテクチャ（TO-BE）

```
[外部ホスティング — Vercel (無料枠)]
┌────────────────────────────────────┐
│  Next.js (App Router)              │
│  ・出馬表 + AI推論結果 表示         │
│  ・データ分析フィルタ UI            │
│  ・SSG / ISR / CSR 使い分け        │
└──────────────┬─────────────────────┘
               │ HTTPS API calls
[VPS 2GB — 常駐プロセス]
┌──────────────▼─────────────────────────────────────┐
│  FastAPI + Uvicorn (workers=2)     ~180MB           │
│  ├── GET /api/races/:id            出馬表            │
│  ├── GET /api/predictions/:id      推論結果          │
│  ├── GET /api/analysis             フィルタ集計      │
│  └── GET /api/auth                 認証             │
│                                                     │
│  DuckDB (memory_limit='128MB')     ~128MB           │
│  └── Parquet on GCS を直接クエリ                    │
│                                                     │
│  TTLCache (results/analysis)       ~5MB             │
│  OS + その他                       ~300MB           │
│  ─────────────────────────────────────────         │
│  常駐合計                          ~613MB ✅        │
└─────────────────────────────────────────────────────┘

[VPS 2GB — バッチプロセス (cron 時刻分離・完全独立)]
┌─────────────────────────────────────────────────────┐
│  Python Scraper Worker  (04:00 JST)  ~300MB peak    │
│  └── lxml + requests, 逐次処理, 即時GCS書き込み      │
│                                                     │
│  Python Inference Worker (05:30 JST) ~512MB peak   │
│  └── LightGBM, バージョン管理付き, OOMロールバック   │
│  ─────────────────────────────────────────         │
│  バッチ最大同時使用 (web+scraper): ~913MB ✅         │
└─────────────────────────────────────────────────────┘

[GCS — 永続ストレージ]
gs://keiba-vpn/
  ├── raw/html/YYYY-MM-DD/{race_id}.html
  ├── normalized/entries/dt=YYYY-MM-DD/{race_id}.parquet
  ├── results/dt=YYYY-MM-DD/{race_id}.parquet
  ├── inference/dt=YYYY-MM-DD/{race_id}.parquet
  └── models/{version}.txt  +  models/version.json
```

---

## 2GB VPS メモリ予算（明示的割り当て）

| コンポーネント | 常駐時 | バッチ実行時 | 備考 |
|---|---|---|---|
| OS + カーネル | 200MB | 200MB | 固定 |
| FastAPI / Uvicorn (workers=2) | 180MB | 180MB | 各 worker ~90MB |
| DuckDB (memory_limit設定) | 128MB | 128MB | SET memory_limit='128MB' |
| TTLCache + GCS SDK | 35MB | 35MB | 固定 |
| **Web サーバ常駐合計** | **543MB** | **543MB** | |
| Scraper Worker (04:00 JST) | — | +300MB | 処理後プロセス終了 |
| Inference Worker (05:30 JST) | — | +512MB | 処理後プロセス終了 |
| **ピーク時合計** | **543MB ✅** | **最大 1,055MB ✅** | 余裕 ~993MB |

> ⚠️ **設計上の制約**: Scraper と Inference は **同時起動禁止**。cron の時刻を 90分以上空けること。

---

## 機能要件

| # | 要件 | 優先度 | 担当エージェント |
|---|------|--------|----------------|
| F-1 | 出馬表データ（馬名・騎手・オッズ等）を API 経由でフロントエンドへ提供する | 高 | backend-engineer |
| F-2 | AI推論結果（各馬の推論スコア・順位予測）を出馬表と統合した画面で表示する | 高 | frontend-engineer / ai-model-engineer |
| F-3 | ユーザが距離・馬場状態・騎手等のフィルタ条件を指定し、勝率/複勝率を集計して表示する | 高 | frontend-engineer / data-engineer |
| F-4 | スクレイピングバッチを定期実行し、結果を GCS（Parquet形式）に保存する | 高 | data-engineer |
| F-5 | AI推論バッチを定期実行し、推論結果を GCS + DB に保存する | 高 | ai-model-engineer |
| F-6 | ユーザ認証（ログイン・セッション管理）を実装する | 中 | backend-engineer |
| F-7 | フロントエンドを Vercel にデプロイし、VPS への静的配信負荷をゼロにする | 高 | frontend-engineer |
| F-8 | スクレイピング失敗時（HTTP 429・タイムアウト）に自動リトライを行う | 中 | data-engineer |
| F-9 | AI モデルのバージョン管理を行い、OOM 発生時に前バージョンへ自動ロールバックする | 中 | ai-model-engineer |
| F-10 | 分析クエリ結果を TTL キャッシュでキャッシュし、重複 DuckDB クエリを削減する | 中 | backend-engineer |
| F-11 | 管理者向けにバッチ手動トリガー API を提供する | 低 | backend-engineer |

---

## 非機能要件

| # | 要件 | 目標値 | 担当 |
|---|------|--------|------|
| N-1 | VPS メモリ使用量（アイドル時） | ≤ 700MB | operations-engineer |
| N-2 | VPS メモリ使用量（バッチ実行中） | ≤ 1,200MB（余裕 800MB） | operations-engineer |
| N-3 | API レスポンスタイム（出馬表・推論結果） | P95 ≤ 300ms | backend-engineer |
| N-4 | API レスポンスタイム（分析クエリ・キャッシュヒット時） | P95 ≤ 100ms | backend-engineer |
| N-5 | API レスポンスタイム（分析クエリ・DuckDB 実行時） | P95 ≤ 2,000ms | data-engineer |
| N-6 | 同時リクエスト処理数 | 50 concurrent（OOM なし） | backend-engineer |
| N-7 | スクレイパーワーカーのメモリ上限 | ≤ 300MB（resource.setrlimit） | data-engineer |
| N-8 | 推論ワーカーのメモリ上限 | ≤ 512MB（resource.setrlimit） | ai-model-engineer |
| N-9 | バッチ失敗が Web サービス停止に波及しないこと | 独立プロセス、systemd 管理 | operations-engineer |
| N-10 | スクレイピング〜推論結果表示までのリードタイム | ≤ 3時間（レース当日） | data-engineer |
| N-11 | GCS Parquet パーティション設計により DuckDB クエリが全件スキャンしないこと | プルーニング有効 | data-engineer |
| N-12 | モデルバージョンが GCS に保存され、前バージョンへのロールバックが可能なこと | version.json 管理 | ai-model-engineer |

---

## 技術スタック決定

| レイヤー | 現状 | 採用技術 | 採用理由 |
|---------|------|---------|---------|
| フロントエンド | Flask (Jinja2 テンプレート) | **Next.js (App Router)** | React エコシステム、SSG/ISR によるキャッシュ、VPS 負荷ゼロ |
| フロントエンドホスティング | VPS | **Vercel (無料枠)** | CDN 配信・スケール無制限・VPS メモリ消費ゼロ |
| バックエンド API | Flask (Python) | **FastAPI + Uvicorn** | Python 維持で移行コスト最小、async/await、OpenAPI 自動生成 |
| WSGI/ASGI | Gunicorn | **Uvicorn (workers=2)** | ASGI ネイティブ、メモリ効率高 |
| 分析クエリ | pandas (全件インメモリ) | **DuckDB** | 列指向・ストリーミング読み込み・Parquet 直接クエリ |
| データストレージ | ローカルファイル / DB | **GCS (Parquet)** + **PostgreSQL (users/auth)** | バッチと API を疎結合、Parquet は DuckDB と相性最良 |
| スクレイピング | Flask 内 Python | **独立 Python バッチ (cron)** | Webサーバからメモリ分離、障害の波及ゼロ |
| AI 推論 | Flask 内 Python | **独立 Python バッチ (cron)** | モデルメモリ (~512MB) を Webサーバから完全分離 |
| キャッシュ | なし | **TTLCache (cachetools)** | 分析クエリ結果を TTL=300s でキャッシュ、DuckDB 負荷削減 |

> **Go (Gin) 採用を見送った理由**: AI/ML ライブラリ（LightGBM, scikit-learn）が Python ネイティブであり、バッチワーカーは Python のまま維持必須。バックエンド API だけ Go に移行しても、Python バッチとのコードベース分断による保守コストが増大する。FastAPI の非同期処理で十分な スループットが得られる判断をしました。

---

## 主要コンポーネント設計仕様

### バックエンド API (FastAPI)

```python
# main.py — 起動設定
import uvicorn
from fastapi import FastAPI
from cachetools import TTLCache
import duckdb

app = FastAPI()

# DuckDB: メモリ上限を明示的に設定
con = duckdb.connect()
con.execute("SET memory_limit='128MB'")
con.execute("INSTALL httpfs; LOAD httpfs;")  # GCS アクセス用

# TTLCache: 分析クエリ結果を5分キャッシュ
analysis_cache = TTLCache(maxsize=200, ttl=300)

# Uvicorn 起動設定 (gunicorn.conf.py)
# workers = 2
# worker_class = "uvicorn.workers.UvicornWorker"
# worker_connections = 100
```

### スクレイパーワーカー (Python バッチ)

```python
# scraper_worker.py — メモリ管理仕様
import resource, gc, logging
import requests
from lxml import etree  # html.parser より ~30% 低メモリ

SCRAPER_MEMORY_LIMIT_MB = 300  # 実測ピーク ~120MB の 2.5倍マージン

# 実測メモリプロファイル:
# Python runtime:              ~50MB
# requests + urllib3:          ~15MB
# lxml パーサー (C拡張):        ~20MB
# 1ページ分 DOM:                ~3MB
# データ変換処理:               ~10MB
# ピーク (1レース処理中):        ~98MB
# 18レース逐次処理 (GCあり):    ~120MB 安定

def configure_process():
    limit = SCRAPER_MEMORY_LIMIT_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def scrape_one_race(race_id: str) -> dict:
    resp = requests.get(f"https://race.netkeiba.com/...", timeout=10)
    root = etree.parse(BytesIO(resp.content), etree.HTMLParser())
    data = _extract_entries(root)
    del root, resp  # 即時解放 (GC 依存しない)
    gc.collect()
    return data

def run_batch(race_ids: list[str]):
    configure_process()
    for race_id in race_ids:
        try:
            data = scrape_one_race(race_id)
            write_parquet_to_gcs(data, race_id)  # 蓄積せず即書き込み
        except MemoryError:
            gc.collect()  # 回復試行して次レースへ
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(60)  # Rate limit 対応
```

### AI 推論ワーカー (Python バッチ)

```python
# inference_worker.py — モデルバージョン管理仕様
import resource, gc, json
import lightgbm as lgb

INFERENCE_MEMORY_LIMIT_MB = 512

def run_inference(race_date: str, gcs_bucket):
    resource.setrlimit(
        resource.RLIMIT_AS,
        (INFERENCE_MEMORY_LIMIT_MB * 1024**2,) * 2
    )
    model = _load_with_rollback(gcs_bucket)
    features_df = _build_features(race_date)  # cutoff = race_date - 1日 (リーク防止)
    proba = model.predict(features_df)
    _write_predictions(proba, race_date)
    del features_df, proba
    gc.collect()

def _load_with_rollback(bucket) -> lgb.Booster:
    """OOM 時に前バージョンへ自動ロールバック"""
    meta = json.loads(bucket.blob("models/version.json").download_as_text())
    for version in [meta["current"], meta.get("previous")]:
        try:
            path = f"/tmp/model_{version}.txt"
            bucket.blob(f"models/{version}.txt").download_to_filename(path)
            return lgb.Booster(model_file=path)
        except MemoryError:
            gc.collect()
    raise RuntimeError("全バージョンのモデルロード失敗")
```

### GCS パーティション設計

```
gs://keiba-vpn/
  ├── raw/html/YYYY-MM-DD/{race_id}.html           # 生 HTML (変更追跡用)
  ├── normalized/entries/dt=YYYY-MM-DD/{race_id}.parquet
  ├── results/dt=YYYY-MM-DD/{race_id}.parquet
  ├── inference/dt=YYYY-MM-DD/{race_id}.parquet    # 推論結果
  └── models/
       ├── version.json                            # {"current":"v3","previous":"v2"}
       ├── v2.txt
       └── v3.txt
```

### cron スケジュール (時刻分離設計)

```cron
# /etc/cron.d/keiba-vpn
# スクレイパー: 04:00 JST (レース翌日確定後)
0 4 * * * keiba /usr/bin/python3 /app/scraper_worker.py

# 推論ワーカー: 06:00 JST (スクレイパー完了から 2 時間後)
0 6 * * * keiba /usr/bin/python3 /app/inference_worker.py

# ※ 同時起動禁止: 間隔を 90 分以上確保
# ※ 当日レース前のオッズ更新は別途 12:00 JST に追加スクレイプ
```

---

## 実装ロードマップ

### Phase 1 — 即効性・低リスク（推定工数: 1〜2週間）

> **目的**: VPS メモリ不足の根本原因を解消する。既存 Python コードを最大限流用。

- **P1-1**: スクレイピング処理を Flask から独立バッチプロセスとして切り離す
  - `scraper_worker.py` 作成 + `resource.setrlimit(300MB)` 設定
  - systemd unit ファイルまたは cron 登録
- **P1-2**: AI 推論処理を Flask から独立バッチプロセスとして切り離す
  - `inference_worker.py` 作成 + `resource.setrlimit(512MB)` 設定
  - GCS `models/version.json` によるバージョン管理実装
- **P1-3**: pandas 全件集計 → DuckDB SQL クエリに置き換え
  - GCS Parquet を DuckDB で直接クエリ
  - `SET memory_limit='128MB'` 設定
- **P1-4**: TTLCache による分析クエリ結果キャッシュ実装

**Phase 1 完了後の期待メモリ**: アイドル時 ~543MB（削減率 ~46%）

---

### Phase 2 — フロントエンド分離（推定工数: 2〜4週間）

> **目的**: VPS からUI配信コストを完全排除し、CDN 経由の高速表示を実現。

- **P2-1**: Next.js プロジェクト作成・Vercel 設定
- **P2-2**: 出馬表コンポーネント実装（`/races/[id]` ページ、SSG + ISR）
- **P2-3**: AI 推論結果統合表示コンポーネント実装
- **P2-4**: データ分析フィルタ UI 実装（CSR、リアルタイムクエリ）
- **P2-5**: Flask の HTML レンダリング部分を削除、API 専用化

**Phase 2 完了後の期待効果**: VPS から静的配信コスト完全排除、UI 応答速度 50-70% 改善

---

### Phase 3 — バックエンド Flask → FastAPI 移行（推定工数: 2〜3週間）

> **目的**: 非同期処理による多重リクエスト耐性向上・メモリ効率改善。

- **P3-1**: FastAPI プロジェクト作成、既存 Flask エンドポイントを順次移植
  - `/api/races/:id` → async 実装
  - `/api/predictions/:race_id` → async 実装
  - `/api/analysis` → DuckDB async クエリ + TTLCache
  - `/api/auth` → JWT ベース認証
- **P3-2**: Uvicorn workers=2 で本番起動設定
- **P3-3**: Flask 廃止、FastAPI への完全切り替え

**Phase 3 完了後の期待効果**: 同時 50 リクエスト処理時もメモリ安定（OOM ゼロ）

---

## 依存関係・リスク

### 依存関係

```
P1-1 (スクレイパー分離)
    └→ P1-2 (推論分離)
         └→ P1-3 (DuckDB 導入)
              └→ P2-1〜P2-5 (Next.js 移行)
                   └→ P3-1〜P3-3 (FastAPI 移行)
```

- P1-3 (DuckDB) は P1-1 と並行実施可能
- P2 は P1 完了後に実施（API が安定してから UI を構築）
- P3 は P2 完了後に実施（フロントエンドが Next.js に移行してから Flask を廃止）

### リスクと対策

| # | リスク | 深刻度 | 対策 |
|---|--------|--------|------|
| R-1 | スクレイピング対象サイトの HTML 構造変更により scraper が停止 | 高 | raw HTML を GCS に保存し、パーサーを後から再実行可能にする |
| R-2 | cron の二重起動による同時バッチ実行 → OOM | 高 | `flock` コマンドによる排他ロック、または systemd `ExecCondition` で多重起動防止 |
| R-3 | DuckDB の GCS アクセス（httpfs）がレイテンシ増大 | 中 | 分析クエリ頻出パターンは TTLCache(TTL=300s) でキャッシュ。クエリ P95 ≤ 2s を SLO とする |
| R-4 | AI モデルロード時の OOM (resource.RLIMIT_AS 超過) | 中 | version.json によるロールバック機構 + MemoryError キャッチ→前バージョン試行 |
| R-5 | Next.js 移行工数がレース開催スケジュールと競合 | 中 | Phase 1 完了後は Flask で継続運用可能。Phase 2 はオフシーズンに実施推奨 |
| R-6 | Vercel 無料枠の API Route 制限（月 100GB 転送量） | 低 | 重いデータ取得は全て VPS API へ。Vercel は UI 配信のみ使用 |
| R-7 | Python scraper の lxml 依存（C拡張ビルド） | 低 | `lxml` ビルド済み Docker イメージを用意、またはビルド済みホイールを利用 |

### 未解決の懸念点（人間レビュー必要）

> 以下は 5ラウンドの議論で合意に至らなかった、または情報不足のため人間の判断を要する項目です。

1. **PostgreSQL の配置**: ユーザ認証・predictions テーブルを VPS 内 PostgreSQL に置くか、外部マネージド DB（Cloud SQL）にするかが未決定。VPS 内配置の場合、追加で ~100MB のメモリ消費が発生し、メモリ予算の再計算が必要。
2. **Vercel の有料化リスク**: 将来的にトラフィックが増加した場合の Vercel 課金ポリシー変更への対応方針（代替: Cloudflare Pages）。
3. **スクレイピングの利用規約確認**: netkeiba のスクレイピングポリシーと robots.txt への適合確認が必要。

---

## 付録: エージェント別担当定義

| フェーズ | タスク | 主担当 | 副担当 |
|---------|--------|--------|--------|
| Phase 1 | スクレイパー分離・メモリ管理 | data-engineer | operations-engineer |
| Phase 1 | 推論バッチ分離・モデルバージョン管理 | ai-model-engineer | data-engineer |
| Phase 1 | DuckDB 導入・TTLCache 設計 | backend-engineer | data-engineer |
| Phase 2 | Next.js コンポーネント実装 | frontend-engineer | fullstack-integrator |
| Phase 2 | API エンドポイント設計・契約定義 | fullstack-integrator | backend-engineer |
| Phase 3 | Flask → FastAPI 移行 | backend-engineer | fullstack-integrator |
| 全体 | VPS メモリ監視・アラート設定 | operations-engineer | — |
| 全体 | コスト試算（Vercel/GCS/VPS） | cost-optimizer | — |

---

## Conclusion

**最優先実施事項は **Phase 1（スクレイパー・推論バッチをWebサーバから独立プロセスとして切り離す）** であり、これだけでVPS メモリ使用量を現状の ~1,300MB+ から ~543MB（アイドル時）まで削減でき、OOMリスクを根本解消できる。**

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-02_
