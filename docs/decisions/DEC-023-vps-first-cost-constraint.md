# DEC-023: VPS ファーストアーキテクチャ & 月額予算上限

**Status**: ACCEPTED
**Date**: 2026-07-06
**Author**: Orchestrator（TASK-056 エスカレーション確認）
**Related**: DEC-001, DEC-003, DEC-006, DEC-010, AREA-05, AREA-06, AREA-10
**Supersedes**: DEC-006（デプロイターゲット選定）の GCP マネージドサービス前提部分

---

## Context

AREA-05〜07 のコスト試算は GCP マネージドサービス（Cloud Run / Cloud SQL / BigQuery）を前提に $100/月（〜15,000円）で作られていたが、実際のシステムは ConoHa VPS で稼働しており実態と乖離していた。また想定リクエスト数が最大 50,000 req/日（1,000 req/レース × 36レース）であることが確認され、月額予算を 1,000〜3,000円以内とする要件が明確化された。

## Decision

### インフラ方針: VPS ファースト

| サービス | 旧仕様書（GCP前提） | 採用構成 |
|---------|------------------|--------|
| Web サーバ | Cloud Run | **ConoHa VPS（Flask + Nginx + Gunicorn）** |
| データベース | Cloud SQL（PostgreSQL） | **VPS 上のローカル PostgreSQL** |
| タスクキュー | Cloud Run Jobs + Cloud Scheduler | **VPS 上の Celery + cron** |
| 分析基盤 | BigQuery | **廃止 → PostgreSQL に集約** |
| オブジェクトストレージ | GCS（全データ） | **GCS は冷ストレージのみ**（HybridStorage で L1メモリ→L2ディスク→L3 GCS） |
| モニタリング | Google Cloud Monitoring | **Prometheus + Grafana on VPS**（AREA-10 準拠） |

### 予算上限

- **目標**: 1,000円/月以内
- **上限（絶対守る）**: 3,000円/月

### 月額コスト試算（VPS ベース）

| 項目 | 月額 |
|------|------|
| ConoHa VPS 2GB RAM / 2vCPU | ~1,030円 |
| GCS ストレージ（50〜100GB 冷ストレージ） | ~170〜350円 |
| GCS API 呼び出し（HybridStorage キャッシュ活用） | ~100〜200円 |
| **合計** | **~1,300〜1,580円** ✅ |

### コスト施策

1. **BigQuery を廃止** — PostgreSQL で特徴量・分析クエリをすべて処理
2. **Cloud Run / Cloud SQL / Cloud Scheduler を廃止** — VPS cron + ローカル DB
3. **GCS 読み取りを最小化** — HybridStorage（L1メモリ→L2ディスク→L3 GCS）で重複 GCS アクセスを防ぐ
4. **ML 特徴量を VPS ディスクにキャッシュ** — 毎回 GCS から読まない
5. **スクレイピングは逐次処理** — 同サイトへの並列リクエスト禁止（後述）

### スクレイピング制約

netkeiba.com および JRA公式サイトへの**並列リクエストは禁止**とする（逐次処理のみ）。
- `NETKEIBA_MAX_CONCURRENT_REQUESTS=1` を `.env` で強制
- 利用規約確認済み（2026-07-06）

## Consequences

- DEC-006（Cloud Run）の GCP マネージドサービス選定は本決定で実態ベースに上書きされる
- DEC-010（$100/月）の予算上限は本決定の 3,000円（約 $20）に更新される
- DEC-003（Cloud Run Jobs）のバッチ実行基盤は VPS cron に変更される
- AREA-05 のコスト試算を VPS ベース構成に全面改訂する
- AREA-06 の BigQuery テーブル定義（ml_features 等）を PostgreSQL スキーマに変更する
- 50,000 req/日は HybridStorage キャッシュにより VPS 2GB で対応可能

---
*Last updated: 2026-07-06*
