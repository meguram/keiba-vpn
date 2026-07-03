# AREA-04 — 運用最適化要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-003(サーバ負荷), DEC-004(プロセス分離), DEC-009(メモリ), DEC-010(systemd/Circuit Breaker)

---

## 1. プロセス分離設計

| systemd ユニット | プロセス | スケジュール |
|---|---|---|
| `keiba-api.service` | Flask API（Gunicorn） | 常時起動 |
| `keiba-inference.service` | LightGBM バッチワーカー | cron 08:10 JST |
| `keiba-scraper.service` | ETL スクレイパー | cron 06:00, 12:00 JST |

**共通設定**: `Restart=on-failure`, `/healthz` エンドポイント監視

**同時実行禁止**: スクレイパーと推論バッチは **90 分以上の間隔を強制**（OOM 防止）

---

## 2. VPS メモリバジェット（2GB = 2,048MB）

| コンポーネント | 割当 | 備考 |
|---|---|---|
| OS + systemd | 256 MB | |
| Nginx | 50 MB | |
| Flask/Gunicorn × 2（CoW） | 300 MB | preload_app=True でモデル共有 |
| LightGBM Booster（CoW, 物理 1 コピー） | 200 MB | |
| PostgreSQL（shared_buffers=128MB） | 200 MB | |
| Redis（maxmemory=256MB） | 256 MB | |
| スクレイパー（時間分離ピーク） | 300 MB | 推論と同時実行禁止 |
| OOM バッファ | 238 MB | |
| **通常時合計** | **~1,462 MB** | |
| **ピーク（スクレイパー or 推論実行時）** | **~1,762 MB（86%）** | |

**LightGBM 最適化**:
- `num_threads = max(1, vCPU // workers)` = 1（2vCPU 環境）
- `num_leaves` 上限 63（RSS 増加 ≤250MB）
- Booster RSS 超過時 → `num_leaves` 31 で再学習

---

## 3. スクレイピング耐障害設計（Circuit Breaker）

```
tenacity リトライ: max_attempts=3, wait=4s→8s→60s（exponential）
      │
      ├─ 成功 → 通常処理
      └─ 連続 5 回失敗 → pybreaker Circuit Breaker OPEN
             │
             ├─ GCS 最終成功データ配信（フォールバック）
             ├─ "データ更新中" バナー（FreshnessStatus: STALE_CIRCUIT_OPEN）
             ├─ Slack アラート（5 分以内）
             └─ 60 秒後 HALF-OPEN → 回復試行 → 成功で CLOSED
```

### Circuit Breaker 管理

- `scrape_runs` テーブルで実行履歴を記録
- `/admin/circuit-status`（IP 制限）で状態確認
- FreshnessStatus enum: `FRESH` / `STALE` / `STALE_CIRCUIT_OPEN` / `UNKNOWN`

---

## 4. モニタリング

| ツール | 用途 | コスト |
|---|---|---|
| UptimeRobot（無料） | 外部 HTTP 死活監視 | ¥0 |
| Slack Webhook | ETL 失敗 / Circuit Breaker OPEN アラート | ¥0 |
| Prometheus + Grafana（VPS 内） | CPU/メモリ/リクエスト数（Phase 2） | ¥0 |
| opentelemetry | 分散トレーシング（Phase 3） | ¥0 |
| structlog（JSON ロギング） | GCS / Cloud Logging への構造化ログ（Phase 2）| ¥0 |

### アラート閾値

| メトリクス | 閾値 | アクション |
|---|---|---|
| VPS メモリ使用率 | >85% | Slack 警告 |
| Redis メモリ | >220MB（256MB 上限の 86%） | Slack 警告 |
| API P99 レイテンシ | >500ms | Slack 警告 |
| スクレイピング失敗 | 1 回以上 | Slack 通知 |
| Circuit Breaker OPEN | — | 即座に Slack アラート |

---

## 5. デプロイ設計

```bash
# VPS デプロイ（git pull フック）
git pull origin main
pip install -r requirements.txt --no-cache-dir
sudo systemctl restart keiba-api.service
# Inference Worker は次回 cron で自動起動
```

### Cloudflare（DDoS 防御 / SSL 終端）

- 無料プランで DDoS 防御・SSL 終端・レート制限
- VPS への直接アクセスは Nginx で拒否（Cloudflare IP のみ許可）

---

## 6. ロールバック手順

| 項目 | 手順 | 目標時間 |
|---|---|---|
| コードロールバック | `git revert` + `systemctl restart` | <5 分 |
| モデルロールバック | `MODEL_ROLLBACK_PATH` からリストア | <10 分 |
| Redis フラッシュ | `redis-cli FLUSHDB` → 再キャッシュ | <2 分 |

---

## 7. 非機能要件（運用系）

| ID | 要件 |
|---|---|
| NFR-OPS-01 | 可用性 ≥99.5%（月間ダウンタイム ≤3.6h） |
| NFR-OPS-02 | VPS 総メモリ ≤1,536MB（通常時）|
| NFR-OPS-03 | Redis ≤256MB（allkeys-lru 強制）|
| NFR-OPS-04 | GCS 読み取りリクエスト削減 ≥95%（Redis 導入後比）|
| NFR-OPS-05 | アラート配信 ≤5 分（障害発生から）|
| NFR-OPS-06 | ボトルネック特定 ≤30 分（トレース情報から）|

---

## 8. Human Review 依頼事項

| 項目 | 優先度 |
|---|---|
| PostgreSQL shared_buffers 実際値の確認（メモリバジェット最終確定） | 高 |
| SHAP 時間分離のロードテスト（Phase 2 移行前） | 中 |
| ETL 遅延フォールバックポリシー: 503 全件 vs 前日データ配信 | 高 |
