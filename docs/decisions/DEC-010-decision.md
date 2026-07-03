# DEC-010: decision

**Date**: 2026-07-03
**Agent**: web-search-agent, decisions-context-agent, proposal-agent, backend-engineer, data-engineer, ai-model-engineer, integration-synthesizer, quality-reviewer
**Task**: TASK-029
**Status**: ACCEPTED

---

## Context

修正内容を確認してください: keiba-vpn/src/scraper/ gitディレクトリでGCSに保存するファイルパスを管理しています。これを考慮してください。

---

## Decision

# 改善要件定義書: keiba-vpn アプリ設計

> **作成日**: 2025-01-27
> **ステータス**: 確定（全エージェント APPROVE 済）
> **合意ラウンド**: Round 3 Complete
> **参照決定**: DEC-002 / DEC-004 / DEC-005 / DEC-007 / DEC-008 / DEC-009

---

## サマリー

本要件定義書は、日本競馬予測 Web アプリ **keiba-vpn** の設計改善方針を定めるものである。ConoHa VPS 2GB という制約環境下での安定稼働を最優先とし、「キャッシュ・スクレイピング耐障害性・MLパイプライン分離・フロントエンドUX・オブザーバビリティ」の5領域において段階的な改善を実施する。全改善は既存アーキテクチャ決定（DEC-002〜009）との整合性を維持しながら、推論レイテンシ200ms以下・スクレイピング障害5分以内検知・モバイルLCP 3秒以内を目標値として設定する。

**GCSファイルパス管理方針**: スクレイパーが書き込む GCS のファイルパス（バケット名・プレフィックス・命名規則）は **`keiba-vpn/src/scraper/`** ディレクトリ内で一元管理する。ETL・Feature Store・モデルバージョン管理など下流の全コンポーネントは、このディレクトリが定義するパス定数をインポートして参照し、パスのハードコードを禁止する。

---

## アーキテクチャ概観

```
┌────────────────────────────────────────────────────────────────────┐
│  ユーザー (モバイル / PC)                                           │
│  PWA対応 Next.js フロントエンド                                     │
└──────────────────────────┬─────────────────────────────────────────┘
                            │ HTTPS
┌──────────────────────────▼─────────────────────────────────────────┐
│  ConoHa VPS 2GB RAM                                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │ Flask API     │  │ Inference Worker  │  │ Redis 256MB          │ │
│  │ systemd unit1 │  │ systemd unit2     │  │ allkeys-lru          │ │
│  │ Gunicorn 4w   │  │ Gunicorn 1w       │  └──────────────────────┘ │
│  └──────┬───────┘  └────────┬─────────┘                           │
│         │                   │           ┌──────────────────────────┐│
│         └──────────┬────────┘           │ PostgreSQL 512MB         ││
│                    │                    │ race_cache / features    ││
│                    └────────────────────└──────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
             ↑ スクレイピング                   ↑ モデル・特徴量
┌────────────┴────────────┐     ┌──────────────┴────────────────────┐
│ 外部サイト               │     │ Google Cloud Storage (GCS)        │
│ JRA / NAR               │     │ パス定義: src/scraper/ で管理      │
│ Circuit Breaker + Retry  │     │ /models/vN/model.pkl              │
│ GCSパス: src/scraper/    │     │ /features/static / dynamic        │
│ gcs_paths.py で解決      │     │ /raw/{YYYY-MM-DD}/...             │
└─────────────────────────┘     └───────────────────────────────────┘
                                          ↑ 週次学習
                                 ┌────────┴──────────────────────────┐
                                 │ Cloud Run Jobs                    │
                                 │ 2vCPU / 4GB — 学習専用            │
                                 │ GCSパス: src/scraper/ から参照     │
                                 └───────────────────────────────────┘
```

---

## GCSファイルパス管理設計

スクレイパーが GCS に書き込むパスは `keiba-vpn/src/scraper/` 配下の設定モジュール（例: `gcs_paths.py`）で定数として定義し、ETL・Feature Store・Inference Worker・Cloud Run Jobs の全コンポーネントがこれを参照する。

```
keiba-vpn/src/scraper/
  ├── gcs_paths.py          ← GCS パス定数の唯一の定義場所（Single Source of Truth）
  ├── scraper.py
  ├── circuit_breaker.py
  └── ...
```

```python
# keiba-vpn/src/scraper/gcs_paths.py（パス定義の例）

GCS_BUCKET = "keiba-vpn-data"

# スクレイパーが書き込む生データ
RAW_RACE_CARD_PREFIX  = "raw/race_card/{date}/"          # 例: raw/race_card/2025-01-27/
RAW_ODDS_PREFIX       = "raw/odds/{date}/{race_id}/"

# ETL 済み Feature Store
FEATURE_STATIC_PREFIX = "features/static/dt={date}/"    # 週次バッチ更新
FEATURE_DYNAMIC_PREFIX= "features/dynamic/dt={date}/race_id={race_id}/"  # レース30分前更新

# 学習済みモデル
MODEL_PREFIX          = "models/v{version}/"             # 例: models/v3/
MODEL_CURRENT_PATH    = "models/current/model.lgb"       # 現行モデルへのポインタ
MODEL_ROLLBACK_PATH   = "models/v{version}/model.lgb"    # ロールバック用
```

**参照ルール（全コンポーネント共通）**:

| コンポーネント | 参照方法 |
|--------------|---------|
| ETL パイプライン | `from keiba_vpn.scraper.gcs_paths import RAW_RACE_CARD_PREFIX` |
| Feature Store 生成バッチ | `from keiba_vpn.scraper.gcs_paths import FEATURE_STATIC_PREFIX` |
| Cloud Run Jobs（学習） | 同上。環境変数でバケット名のみ上書き可（本番 / ステージング切替用） |
| Inference Worker | `from keiba_vpn.scraper.gcs_paths import MODEL_CURRENT_PATH` |
| Flask API（L4 フォールバック） | 同上 |

> ⚠️ **禁止事項**: `gcs_paths.py` 以外の場所（Flask アプリ・Cloud Run ジョブ・Inference Worker コード内）に GCS パスをハードコードしてはならない。変更は `gcs_paths.py` の1ファイルで完結させること。

---

## 機能要件

| # | 要件 | 優先度 | 担当エージェント |
|---|------|--------|----------------|
| **F-1** | VPSメモリ割当計画（OS:256MB / PostgreSQL:512MB / Flask:512MB / Redis:256MB / バッファ:512MB）を実施し、プロセスごとに上限を設定する | 高 | backend-engineer |
| **F-2** | Redis に `maxmemory 256mb` / `maxmemory-policy allkeys-lru` を設定し、OOMを防止する | 高 | backend-engineer |
| **F-3** | 4段キャッシュ（L1: Flask lru_cache 60秒 / L2: Redis 5分TTL / L3: PostgreSQL race_cache / L4: GCS）を `GET /api/v1/predictions/{race_id}` に実装する。L4フォールバック時のGCSパスは `src/scraper/gcs_paths.py` の定数を参照すること | 高 | backend-engineer / data-engineer |
| **F-4** | `race_cache` テーブルをPostgreSQLに新規作成し、ETL完了後フックで自動更新する | 高 | data-engineer |
| **F-5** | レース開始+30分経過をトリガーとして、当該レースのRedisキャッシュを自動削除（TTL expiry）する | 高 | backend-engineer |
| **F-6** | `tenacity` による指数バックオフリトライ（最大3回 / 4秒→8秒→60秒）をスクレイピング処理に実装する。スクレイピング先URLおよびGCS書き込みパスは `src/scraper/gcs_paths.py` から取得する | 高 | data-engineer |
| **F-7** | `pybreaker` による Circuit Breaker（5回失敗でOPEN / 60秒後HALF-OPEN）をスクレイピング処理に実装する | 高 | data-engineer |
| **F-8** | Circuit OPEN 時は `src/scraper/gcs_paths.py` で定義された最終成功データのGCSパスからフォールバック表示し、フロントエンドに「データ更新中」バナーを表示する | 高 | data-engineer / frontend-engineer |
| **F-9** | Circuit OPEN 検知時に Slack Webhook で5分以内のアラート通知を送信する | 高 | operations-engineer |
| **F-10** | Flask API と LightGBM 推論を systemd unit として分離し（unit1: API / unit2: Inference Worker）、プロセス間のメモリ共有を禁止する | 高 | backend-engineer / ai-model-engineer |
| **F-11** | Inference Worker 起動時に `src/scraper/gcs_paths.py` の `MODEL_CURRENT_PATH` を参照してGCSからモデルを1回ロードし、リクエスト毎の再ロードを行わない設計とする | 高 | ai-model-engineer |
| **F-12** | 学習処理を Cloud Run Jobs（2vCPU / 4GB）に完全移管し、VPS上での学習実行を禁止する（週次スケジュール）。Cloud Run Jobs は `src/scraper/gcs_paths.py` を参照して Feature Store パスとモデル保存先を解決する | 高 | ai-model-engineer / operations-engineer |
| **F-13** | モデルバージョン管理を `src/scraper/gcs_paths.py` の `MODEL_ROLLBACK_PATH` 定数（`GCS/models/vN/model.lgb`）で実施し、1世代前（v(N-1)）をロールバック用に保持する | 高 | ai-model-engineer |
| **F-14** | Feature Store（GCS + Parquet）を構築し、静的特徴量（週次更新）と動的特徴量（レース30分前更新）を分離管理する。GCSパスは `src/scraper/gcs_paths.py` の `FEATURE_STATIC_PREFIX` / `FEATURE_DYNAMIC_PREFIX` を使用する | 中 | data-engineer / ai-model-engineer |
| **F-15** | Next.js ISR（revalidate: 300秒）をレース一覧ページに適用し、静的生成によるレスポンス高速化を実現する | 中 | frontend-engineer |
| **F-16** | レース詳細ページに Skeleton UI を実装し、データロード中の体感速度を改善する | 中 | frontend-engineer |
| **F-17** | 予測結果表示を「推奨馬券種別 + 上位3頭カード形式」に刷新し、モバイルでの視認性を優先する | 中 | frontend-engineer |
| **F-18** | PWA（Progressive Web App）対応を実施し、ホーム画面追加・オフラインキャッシュを有効化する | 中 | frontend-engineer |
| **F-19** | `structlog` による構造化JSON ログを全Pythonプロセスに導入し、Cloud Logging / GCSへ出力する | 中 | operations-engineer |
| **F-20** | `opentelemetry-sdk` による分散トレーシングをFlaskに導入し、GCSアクセス〜APIレスポンスの全スパンを計測する | 中 | operations-engineer |
| **F-21** | `/admin/circuit-status` エンドポイントを実装し、サーキット状態・失敗回数・最終失敗時刻を返却する | 中 | backend-engineer |
| **F-22** | `scrape_runs` テーブルと Circuit Breaker 状態を統合し、スクレイピング実行履歴（成功/失敗/circuit_open）をPostgreSQLに記録する | 中 | data-engineer |
| **F-23** | データ鮮度チェック関数（`FreshnessStatus`: FRESH / STALE / STALE_CIRCUIT_OPEN / UNKNOWN）を実装し、APIレスポンスに最終スクレイプ時刻を含める | 低 | data-engineer |
| **F-24** | Feature Store スキーマの定義・管理権限をデータエンジニアに集約し、AIモデルエンジニアとの担当境界を明文化する。GCSパス定数の変更はデータエンジニアが `src/scraper/gcs_paths.py` を通じて実施する | 低 | data-engineer / ai-model-engineer |
| **F-25** | 予測結果キャッシュのペイロードにモデルバージョン情報を含め、後からどのバージョンの予測か追跡可能にする | 低 | ai-model-engineer |
| **F-26** | `src/scraper/gcs_paths.py` をGCSパス定数の Single Source of Truth として確立し、全コンポーネントからのパスハードコードを静的解析（lintルール or テスト）で検出・禁止する | 低 | backend-engineer / data-engineer |

---

## 非機能要件

| # | 要件 | 目標値 | 担当 |
|---|------|--------|------|
| **N-1** | API レスポンスタイム（P99） | ≤ 1,000ms | backend-engineer |
| **N-2** | 予測推論レイテンシ（安定値） | ≤ 200ms（現状500ms〜2秒を排除） | ai-model-engineer |
| **N-3** | モバイル LCP（Largest Contentful Paint） | ≤ 3秒 | frontend-engineer |
| **N-4** | Lighthouse Performance スコア（Mobile） | ≥ 85 | frontend-engineer |
| **N-5** | スクレイピング成功率 | ≥ 95%（下回ったらSlackアラート） | operations-engineer |
| **N-6** | 障害検知〜Slackアラート到達時間 | ≤ 5分（現状: 手動発見） | operations-engineer |
| **N-7** | ボトルネック特定時間 | ≤ 30分（現状: 数時間） | operations-engineer |
| **N-8** | 予測モデル呼び出しエラー率 | ≤ 1%（超過時アラート） | ai-model-engineer / operations-engineer |
| **N-9** | VPS総メモリ使用量 | ≤ 1,536MB（バッファ512MB確保） | backend-engineer / cost-optimizer |
| **N-10** | Redis メモリ使用量 | ≤ 256MB（`maxmemory` 強制上限） | backend-engineer |
| **N-11** | GCS 読み取りリクエスト削減率 | ≥ 80%（キャッシュ導入後比） | data-engineer |
| **N-12** | モデルロールバック完了時間 | ≤ 10分（GCS vN-1 → Inference Worker 再起動） | ai-model-engineer |
| **N-13** | セッション継続率向上（モバイル） | +20〜30%（モバイルUX改善後比） | frontend-engineer |

---

## 実装ロードマップ

### Phase 1 — 安定化基盤（優先度: 高）
**目安期間: 2〜4週間**

> 障害耐性・リソース管理・プロセス分離の確立。本番稼働の前提条件。

- **[F-26]** `src/scraper/gcs_paths.py` の作成・既存ハードコードパスの移行（**Phase 1 の先頭タスク**）
- **[F-1, F-2]** VPSメモリ割当計画の実施 + Redis `maxmemory 256mb` 設定
- **[F-6, F-7]** スクレイピング Circuit Breaker + 指数バックオフリトライの実装
- **[F-8, F-9]** フォールバック表示 + Slack Webhook アラート通知
- **[F-10, F-11]** Flask API / Inference Worker の systemd unit 分離
- **[F-13]** GCS モデルバージョン管理（`gcs_paths.py` の `MODEL_ROLLBACK_PATH` 準拠）+ ロールバック手順書作成
- **[F-12]** Cloud Run Jobs への学習処理移管（VPS上の学習処理削除）
- **[F-4]** `race_cache` テーブルのマイグレーション実行

```sql
-- Phase 1 マイグレーション
CREATE TABLE race_cache (
    race_id     VARCHAR(20) PRIMARY KEY,
    payload     JSONB        NOT NULL,
    cached_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    source      VARCHAR(10)  NOT NULL  -- 'gcs' | 'etl' | 'recompute'
);
CREATE INDEX idx_race_cache_cached_at ON race_cache(cached_at);
```

---

### Phase 2 — パフォーマンス最適化（優先度: 中）
**目安期間: 3〜5週間（Phase 1 完了後）**

> レイテンシ削減・UX改善・可観測性の確立。

- **[F-3, F-5]** 4段キャッシュ実装（L1〜L4） + レース終了後TTL自動削除
- **[F-15, F-16, F-17]** Next.js ISR + Skeleton UI + 予測カード形式UI刷新
- **[F-19]** `structlog` 構造化JSONログの全プロセス導入
- **[F-20]** `opentelemetry-sdk` 分散トレーシング導入
- **[F-21, F-22]** `/admin/circuit-status` エンドポイント + `scrape_runs` 統合
- **[F-14]** Feature Store（GCS + Parquet）構築・静的/動的特徴量分離
- **[F-18]** PWA 対応（オフラインキャッシュ + ホーム画面追加）

---

### Phase 3 — 高度化・品質保証（優先度: 低）
**目安期間: Phase 2 完了後・継続的改善**

> モデル品質維持・運用成熟度向上。

- **[F-23]** データ鮮度チェック関数 + APIレスポンスへの鮮度情報付与
- **[F-24]** Feature Store スキーマ担当境界の明文化（docs/decisions/ に記録）
- **[F-25]** 予測キャッシュへのモデルバージョン情報埋め込み
- Grafana / Google Cloud Monitoring ダッシュボード構築
- Lighthouse CI を CI/CD パイプラインに組み込み（Performance ≥ 85 の自動チェック）
- `gcs_paths.py` パスハードコード検出の lint ルール整備（F-26 の継続的強制）

---

## 依存関係・リスク

### 依存関係

```
F-26      ──→  F-3, F-6, F-8, F-11, F-12, F-13, F-14
               （gcs_paths.py 確立後に全コンポーネントがパスを参照可能になる）
F-1, F-2  ──→  F-3（メモリ割当確定後にキャッシュ設計を実装）
F-4       ──→  F-3（race_cacheテーブル作成後にL3フォールバック有効化）
F-10      ──→  F-11（プロセス分離後にモデルロード設計を実施）
F-12      ──→  F-13（Cloud Run Jobs 移管完了後にバージョン管理を確立）
F-14      ──→  F-12（Feature Store 完成後に Cloud Run Jobs の学習入力が安定）
F-7       ──→  F-21（Circuit Breaker 実装後に状態APIを公開）
F-19, F-20 ──→ Phase 3（ログ基盤確立後にダッシュボード構築）
```

### リスクと対策

| # | リスク | 影響度 | 対策 |
|---|--------|--------|------|
| **R-1** | Redis `maxmemory` 未設定でOOM発生 | 🔴 高 | Phase 1 先頭タスクとして即時設定。設定ファイルをGitで管理し変更を防止 |
| **R-2** | Gunicorn マルチワーカー構成で Circuit Breaker 状態がワーカー間で不一致 | 🟡 中 | 暫定: Gunicorn `-w 1` で単一ワーカー運用。Phase 2 以降で Redis ベースのカウンタ管理へ移行 |
| **R-3** | `/admin/circuit-status` エンドポイントが認証なしで本番公開 | 🟡 中 | Phase 2 の認証実装（DEC-005 `auth_required`）完了まで IP制限または Basic認証で暫定保護 |
| **R-4** | Cloud Run Jobs の学習コストが想定超過 | 🟡 中 | `cost-optimizer` エージェントによる週次コスト計測。スポットインスタンス（Spot VM）の利用を検討 |
| **R-5** | JRA/NAR サイトの HTML 構造変更によるスクレイピング完全停止 | 🟡 中 | Circuit Breaker のOPEN検知でSlack即時通知。スクレイピングコードをモジュール化し差し替え容易に設計 |
| **R-6** | Feature Store 設計遅延による ML パイプライン後退 | 🟢 低 | Phase 2 でデータエンジニアが Feature Store スキーマを先行定義。AI モデルエンジニアと週次レビュー |
| **R-7** | `race_cache` テーブルの ETL 更新タイミング競合（二重書き込み） | 🟢 低 | ETL 完了後フックを排他制御（PostgreSQL `pg_advisory_lock`）で実装 |
| **R-8** | `gcs_paths.py` 以外の場所に GCS パスがハードコードされ、パス不整合が発生する | 🟡 中 | F-26 の lint ルールで継続的に検出。CI パイプラインに組み込み、違反時はマージをブロック |

---

## 実装上の注意事項（エージェント合意済み）

### GCSパス管理（全コンポーネント共通）
- GCS のバケット名・プレフィックス・ファイル命名規則はすべて `keiba-vpn/src/scraper/gcs_paths.py` に集約する
- 下流コンポーネント（ETL・Inference Worker・Cloud Run Jobs・Flask API）は同ファイルの定数をインポートして参照し、パス文字列のハードコードを行わないこと
- バケット名のみ環境変数（`GCS_BUCKET`）で本番 / ステージング切替を許容する。その他のパス構造は `gcs_paths.py` で固定する

### backend-engineer 確認事項
- Flask `lru_cache`（L1）はGunicorn マルチワーカー構成ではワーカー間共有されない。L1ヒット率は各ワーカー独立であることを前提に計測すること
- PostgreSQL L3 への書き戻しは非同期化（`asyncio` またはバックグラウンドスレッド）を推奨し、書き込みレイテンシをAPIレスポンスに乗せないこと

### data-engineer 確認事項
- ETL → Feature Store の書き込み方向: `GCS raw（gcs_paths.RAW_*）→ 正規化テーブル → MATERIALIZED VIEW refresh → race_cache → Redis setex`
- `scrape_runs` テーブルと Circuit Breaker を統合し、`circuit_open` ステータスをスクレイピング実行ログに記録すること
- `gcs_paths.py` に定義されたパス定数の変更はデータエンジニアが主担当として管理し、変更時は下流コンポーネントへの影響を確認すること

### ai-model-engineer 確認事項
- LightGBM の `predict` はスレッドセーフ。Inference Worker は `-w 1` 単一プロセスで運用し、モデルオブジェクトを起動時1回のみロードすること（`MODEL_CURRENT_PATH` を参照）
- 予測キャッシュペイロードには必ずモデルバージョン（例: `"model_version": "v3"`）を含め、後からの予測品質トレースを可能にすること
- Cloud Run Jobs の学習スクリプトは `gcs_paths.py` から `FEATURE_STATIC_PREFIX` / `FEATURE_DYNAMIC_PREFIX` / `MODEL_PREFIX` を参照し、パスをハードコードしないこと

---

## ドキュメント管理

| ドキュメント | パス |
|------------|------|
| 本要件定義書 | `project/keiba-vpn/docs/decisions/DEC-010-requirements.md` |
| GCSパス定数定義 | `keiba-vpn/src/scraper/gcs_paths.py` |
| アーキテクチャ図 | `project/keiba-vpn/architecture/` |
| タスクトラッキング | `project/keiba-vpn/tasks/` |
| ステータスダッシュボード | `project/keiba-vpn/docs/status/DASHBOARD.md` |

---

ConoHa VPS 2GB 制約下での安定稼働を最優先とし、「Phase 1: `src/scraper/gcs_paths.py` によるGCSパス一元化 → Redis maxmemory 設定 → Circuit Breaker → プロセス分離」を最初に確立することが、全改善施策の前提条件となる最重要決定事項である。

---

## Conclusion

****

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-03_
