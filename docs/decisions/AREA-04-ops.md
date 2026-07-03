# AREA-07 — 運用最適化要件（プロセス分離 / VPS メモリバジェット / Circuit Breaker / 監視・アラート / デプロイ / ロールバック）
**Status**: FINAL | **Last Updated**: 2026-07-03 | **Consolidates**: DEC-001, DEC-008, DEC-009, DEC-010, DEC-011

---

## 1. 本仕様書の位置づけ

DEC-001 は競馬予測システム全体の要件定義書であり、運用最適化（プロセス分離・VPS メモリバジェット・Circuit Breaker・監視・アラート・デプロイ・ロールバック）に関する独立した決定文書は現時点で **DEC-001 のみ** である。以下は、DEC-001 の非機能要件（Section 5）・機能要件（Section 4）・実装ロードマップ（Section 6）・リスク管理（Section 7）から運用最適化に関連する記述を抽出し、体系化したものである。

---

## 2. プロセス分離

DEC-001 では以下のプロセスを論理的に分離することが前提とされている。

| プロセス種別 | 役割 | 備考 |
|---|---|---|
| スクレイパープロセス | netkeiba.com からのデータ収集（Layer 1〜5 格納） | シングルワーカー (`concurrent_workers: 1`) |
| スナップショット集計バッチ | `horse_stats_snapshot` / `jockey_stats_snapshot` / `trainer_stats_snapshot` の生成 | `as_of_race_id` 紐付け、results 収集完了後に起動 |
| オッズ収集スケジューラ | 発走当日 08:00〜発走時刻まで 5分/2分/1分 間隔で収集 | タイムウィンドウ制御が必要 |
| 推論バッチプロセス | Stage 1 → Stage 2 の順次推論、`prediction_results` / `prediction_lap_times` 書き込み | 発走 3 時間前までに完了（N-9） |
| API サーバー | REST API 提供、Redis キャッシュ参照 | キャッシュヒット ≤ 200 ms（N-1）、ミス ≤ 2,000 ms（N-2） |
| DDL マイグレーションプロセス | Alembic によるスキーマバージョン管理 | デプロイ時に独立実行（N-11） |

**シングル IP 環境制約**: スクレイパーは `concurrent_workers: 1` を厳守し、並列リクエストによる IP ブロックを防止する。

---

## 3. VPS メモリバジェット

DEC-001 には VPS のメモリ上限や具体的なメモリ割り当て数値の明示的な記述は存在しない。

> **要対応**: 後続の DEC（運用インフラ決定文書）で以下を確定する必要がある。
> - VPS スペック（RAM 上限）
> - 各プロセス（API サーバー・推論バッチ・Redis・PostgreSQL）へのメモリ上限割り当て
> - LightGBM / LSTM モデルのロード時メモリ見積もり

---

## 4. Circuit Breaker

DEC-001 には Circuit Breaker パターンの明示的な記述は存在しないが、以下のリトライ・バックオフ設定がその代替機能を一部担っている。

### 4-1. スクレイパーのリトライ制御（SCRAPING_CONFIG）

```python
SCRAPING_CONFIG = {
    "request_interval_sec": 2.0,
    "jitter_sec": (0.5, 1.5),
    "concurrent_workers": 1,
    "session_rotate_interval": 50,   # 50 リクエスト毎にセッション更新
    "retry_on_429": True,
    "retry_backoff_base_sec": 30,    # 429 受信時のバックオフ基底秒数
    "user_agent_rotate": True,
}
```

### 4-2. 結果スクレイパーのリトライ

```yaml
results:
  trigger: "発走予定時刻 + 35分"
  retry: "5分間隔 × 最大6回"  # 合計最大 30 分間リトライ
```

### 4-3. 要対応事項

> 後続 DEC で以下を確定する必要がある。
> - Circuit Breaker ライブラリの選定（例: `pybreaker`・`tenacity`）
> - 閾値定義: 連続失敗 N 回でオープン状態遷移、クールダウン時間
> - Circuit Breaker 適用対象: netkeiba.com HTTP クライアント・Redis・PostgreSQL 接続

---

## 5. 監視・アラート

### 5-1. スクレイピング成功率監視（N-6、R-2）

| 項目 | 目標値 | アラート条件 |
|---|---|---|
| スクレイピング成功率 | ≥ 99% / 月 | 週次で閾値以下になった場合に通知（R-2 対策） |
| DB 反映遅延 | ≤ 10 分 | 超過時アラート（N-7） |
| オッズスナップショット欠損率（発走前 5 分以内） | ≤ 1% | 超過時アラート（N-8） |

### 5-2. 実行ログ管理（F-5）

```sql
-- scrape_runs テーブル（スクレイプ実行ログ）
-- カラム: target_type, status, retry_count
-- 監視基盤はこのテーブルを参照して成功率・失敗率を集計すること
```

### 5-3. API パフォーマンス監視（N-1、N-2）

| 項目 | SLO |
|---|---|
| キャッシュヒット時レスポンスタイム | ≤ 200 ms |
| キャッシュミス時レスポンスタイム | ≤ 2,000 ms |

### 5-4. 推論バッチ完了監視（N-9）

- 発走 3 時間前までに推論バッチが完了していない場合はアラートを発報する。

### 5-5. モデル品質モニタリング（Phase 4 以降）

- 特徴量重要度のモニタリング
- データドリフト検知
- 障害・エラー通知アラートの整備

### 5-6. テンポラルリーク検知（N-10）

- CI パイプラインにおいてテストデータ時系列分割によるリーク検知テストを自動実行する。
- `as_of_race_id` 紐付けの単体テストを必須化する（F-3 実装時）。

---

## 6. デプロイ

### 6-1. スキーママイグレーション（N-11）

- DDL 変更は **Alembic** でバージョン管理し、全スキーマ変更をマイグレーションファイルとして記録する。
- デプロイ時はマイグレーションプロセスを API サーバー起動前に独立実行する。

### 6-2. デプロイ順序（依存関係）

```
1. DDL マイグレーション実行（Alembic）
2. スクレイパープロセス起動
3. オッズ収集スケジューラ起動
4. 推論バッチプロセス起動
5. API サーバー起動
```

### 6-3. フェーズ別リリース計画

| Phase | 主要デプロイ対象 | 完了条件 |
|---|---|---|
| Phase 0 | scrape_runs テーブル・基本スキーマ | ラップデータ可用性確認完了 |
| Phase 1 | Layer 1〜5 全テーブル DDL・スクレイパー群・集計バッチ | 過去 2 年分データ格納済み |
| Phase 2 | 特徴量パイプライン・Stage 1 モデル・回収率計算ロジック | 勝率 Log Loss ベースライン比 −5% 改善 |
| Phase 3 | Stage 2 モデル・REST API・Redis キャッシュ・UI | 全予測ターゲット T-1〜T-11 が API 経由で取得可能 |
| Phase 4 | LSTM ラップモデル・自動再学習スケジューラ | 継続運用 |

### 6-4. キャッシュ設定（F-12、N-12）

```
キャッシュキー: prediction:{race_id}:{model_version}
キャッシュキー: lap:prediction:{race_id}:{model_version}
TTL: 発走時刻まで有効 / 発走後 60 秒で自動失効
```

---

## 7. ロールバック

### 7-1. モデルロールバック（F-16）

- 学習済みモデルはバージョン管理基盤（MLflow 等）で管理し、古いバージョンへのロールバック機能を提供する（Phase 2 で基盤整備）。
- `prediction_results` テーブルの `model_version` カラムにより、どのモデルバージョンによる推論結果かを追跡可能とする。

### 7-2. スキーマロールバック

- Alembic のダウングレード機能を用いてスキーマ変更を巻き戻す。
- Layer 2〜5 テーブルは **追記型・不変（削除不可）** 設計のため、データ自体のロールバックは行わない。

### 7-3. データロールバック非対応の設計原則

| テーブル種別 | 更新ポリシー | ロールバック可否 |
|---|---|---|
| `race_results`（Layer 2） | 追記のみ | ✗ 不変 |
| `*_stats_snapshot`（Layer 3） | 追記のみ（UNIQUE 制約） | ✗ 不変 |
| `race_odds_snapshot`（Layer 5） | 追記のみ・削除不可 | ✗ 不変 |
| `prediction_results` | UNIQUE(race_id, horse_id, model_version) | ✓ 旧 model_version を参照切替 |

---

## 8. 確定済み運用設計事項（DEC-008/009/010/011 統合）

以下の事項は後続の DEC により確定済みである。

### OP-1: VPS メモリバジェット（DEC-009 確定）

ConoHa VPS 2GB 環境でのプロセス別メモリ割り当て上限（ピーク時・時間分離後）:

```
┌──────────────────────────────────────────────────┬──────────┐
│ コンポーネント                                    │ 割当予算 │
├──────────────────────────────────────────────────┼──────────┤
│ OS + systemd                                      │  300 MB  │
│ Gunicorn ワーカー × 2（preload_app CoW 共有）     │  200 MB  │
│ LightGBM Booster（CoW = 物理 1 コピー）           │  200 MB  │
│ PostgreSQL（shared_buffers 128MB + work_mem 等）  │  200 MB  │
│ Redis（予測結果キャッシュ maxmemory 256MB）        │   64 MB  │
│ スクレイパー（レース日ピーク・時間分離済み）      │  300 MB  │
│ OOM Killer 回避バッファ                           │  736 MB  │
└──────────────────────────────────────────────────┴──────────┘
ピーク合計（スクレイパーと推論を時間分離後）: 約 1,264 MB ✅（2GB の 62%）
アイドル時（常駐プロセスのみ）:              約 543 MB ✅
```

**設計上の制約**: スクレイパー（04:00 JST）と推論ワーカー（06:00 JST）は **同時起動禁止**。cron の時刻を 90 分以上空けること。

### OP-2: Circuit Breaker（DEC-010 確定）

- **ライブラリ**: `pybreaker`
- **適用対象**: netkeiba.com HTTP クライアント（スクレイピング処理）
- **閾値**:
  - 連続失敗 5 回 → OPEN 状態遷移
  - OPEN 後 60 秒 → HALF-OPEN 自動遷移（N-13 DEC-011）
  - HALF-OPEN で 1 回成功 → CLOSED に復帰
- **OPEN 時の挙動**: `gcs_paths.py` で定義された最終成功データの GCS パスからフォールバック表示（DEC-010 F-8）
- **通知**: Circuit OPEN 検知時に Slack Webhook で 5 分以内にアラート送信（DEC-010 F-9）
- **リトライ**: `tenacity` による指数バックオフ（最大3回 / 4s→8s→60s）

### OP-3: 監視基盤ツール（DEC-007 確定）

| 監視層 | ツール | 費用 |
|---|---|---|
| 外形監視（プライマリ） | **UptimeRobot 無料**（URL 死活・5分間隔） | ¥0 |
| 内部メトリクス（フェーズ2以降） | **Prometheus + Grafana on VPS**（メモリ ~200MB 追加消費） | ¥0 |
| 構造化ログ | `structlog`（JSON ログ → Cloud Logging / GCS 出力） | ¥0 |
| 分散トレーシング（Phase 2） | `opentelemetry-sdk` on Flask | ¥0 |

### OP-4: アラート通知チャネル（DEC-008/010 確定）

- **Slack Webhook** を主チャネルとして採用
- 通知対象: スクレイピング失敗、Circuit OPEN、ETL 遅延（T-90 分前未完了）、OOM 発生
- 通知先: `#ops-alerts` チャンネル（DEC-005 確定）

### OP-5: プロセス管理・デプロイ方式（DEC-009/010 確定）

**systemd ユニット構成**:
```
unit1: keiba-api.service       ← Flask API（Gunicorn）
unit2: keiba-inference.service ← Inference Worker（バッチ推論）
```

共通設定:
```ini
[Service]
Restart=on-failure
RestartSec=5s
```

**Gunicorn 設定**（DEC-007/009 確定）:
```python
# gunicorn.conf.py
worker_class         = "gthread"        # I/O バウンドな GCS/Redis アクセスに適合
workers              = 2
threads              = 4
timeout              = 30
max_requests         = 500
max_requests_jitter  = 50               # 定期再起動でメモリリーク防止
preload_app          = True             # fork 前にモデルをロード → CoW でメモリ共有
```

**高負荷対応（DEC-011 確定）**:
- 1,000 同時接続が必要な場合: `--worker-class gevent --worker-connections 500`
- `monkey.patch_all()` を `app.py` 全インポートより先頭に適用必須
- `time.sleep()` → `gevent.sleep()` に統一

**バッチ cron スケジュール**（DEC-009 確定）:
```cron
# スクレイパー: 04:00 JST（レース翌日確定後）
00 4 * * * nice -n 10 /usr/bin/python3 -m etl.pipeline --date $(date +%Y-%m-%d)
# 推論バッチ: 06:00 JST（スクレイパー完了から 2 時間後）
00 6 * * * nice -n 10 /usr/bin/python3 -m inference.batch --date $(date +%Y-%m-%d)
# ※ 同時起動禁止: 間隔 90 分以上確保
```

**モデル週次再学習**:
- Cloud Run Jobs（2vCPU / 4GB）で週次（月曜 02:00 JST）実行（DEC-010/012 確定）
- VPS 上での学習実行は **禁止**（メモリ不足のため）

### OP-6: モデルレジストリ（DEC-009 確定）

- `ModelRegistry` クラス（`RWLock` パターン）をバッチワーカー専用に実装
- GCS からのモデル取得は差分チェック（`generation` メタデータ比較）で最小化
- モデルバージョン管理パス: `models/current/model.lgb`（現行）+ `models/vN/model.lgb`（ロールバック用）
- ロールバック完了時間目標: ≤ 10 分（DEC-010 N-12）
- MLflow は将来の Phase 2 以降での評価を継続する

### OP-7: セキュリティ・DDoS 対策（DEC-011 確定）

- **Cloudflare**: DNS プロキシとして設定、DDoS 防御・SSL 終端（無料プラン）
- **Flask-Limiter**: IP 別レート制限

| エンドポイント | 制限値 |
|---|---|
| `POST /api/predict` | 10 回 / 分 / IP |
| `GET /api/race/*` | 60 回 / 分 / IP |
| `GET /api/odds/*` | 30 回 / 分 / IP |

- **PostgreSQL 接続プール**: `max_connections=50`、`pool_size=5`、`max_overflow=10`（DEC-011 N-9）
- **Redis Thundering Herd 防止**: `SET NX` ミューテックスロック実装（DEC-011 F-4）