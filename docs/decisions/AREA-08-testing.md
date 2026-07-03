# AREA-08: テスト戦略

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂

---

## 1. テストフレームワーク

| ツール | 用途 | バージョン |
|--------|------|------------|
| `unittest` | 主要テストランナー | Python 標準 |
| `pytest` | 補助（将来移行候補） | ≥7.0 |
| `pytest-cov` | カバレッジ計測 | ≥4.0 |

---

## 2. ディレクトリ構造

```
tests/
├── api/               ← FastAPI エンドポイントのテスト
│   └── test_*.py
├── pipeline/          ← 特徴量ストア・学習・推論のテスト
│   └── test_*.py
├── scraper/           ← スクレイパー・パーサーのテスト
│   ├── test_*.py
│   └── manual/        ← 実 HTML を叩く手動スモーク（unittest 対象外）
│       ├── netkeiba_horse_page_smoke.py
│       └── netkeiba_speed_index_smoke.py
├── research/          ← 研究系・血統分析のテスト
│   └── manual/        ← 手動検証（unittest 対象外）
│       └── verify_*.py
├── fixtures/          ← テスト用固定データ
│   ├── html/          ← HTML サンプルファイル
│   └── *.json
└── phase0/            ← Phase 0 の 10 レースサンプル
```

---

## 3. テスト実行

### 全テスト

```bash
# リポジトリルートで実行
python3 -m unittest discover -s tests -t . -p 'test_*.py' -v
```

### 特定テスト

```bash
# 騎手・調教師統計のマージキー検証
python3 -m unittest tests.pipeline.test_jockey_trainer_stats -v

# scraper 系
python3 -m unittest discover -s tests/scraper -t . -p 'test_*.py' -v

# pipeline 系
python3 -m unittest discover -s tests/pipeline -t . -p 'test_*.py' -v
```

### 手動スモークテスト（実 netkeiba HTML を叩く）

```bash
# ユニットテスト対象外（CI では実行しない）
python3 tests/scraper/manual/netkeiba_horse_page_smoke.py
python3 tests/scraper/manual/netkeiba_speed_index_smoke.py
```

---

## 4. CI ゲート（すべてブロッキング）

| ゲート | 閾値 | テスト |
|--------|------|--------|
| **テンポラルリーク禁止** | 0 件 | `test_no_temporal_leak_in_snapshot()` |
| **カバレッジ** | ≥ 80% | `scraper/`, `pipeline/`, `api/` |
| **Log Loss 改善** | ベースライン比 −5% 以上 | ML テスト |
| **Spearman ρ** | ≥ 0.55 | ML テスト |
| **ラップ MAE** | ≤ 0.3 秒 | ML テスト |
| **API レイテンシ** | キャッシュヒット ≤200ms / ミス ≤2000ms | E2E テスト |
| **シャッフル禁止** | `shuffle=True` 使用禁止 | lint チェック |

---

## 5. テストレベル別要件

### 5-1. Unit テスト（必須カバレッジ対象）

**パーサー（`scraper/`）**:
- `race_list` パーサー: date_key 解析、races 配列抽出
- `race_result` パーサー: 着順・タイム・馬名・馬体重抽出
- `race_shutuba` パーサー: 出走馬・騎手・斤量抽出
- `race_odds` パーサー: 単勝・複勝・馬連パース

**特徴量（`pipeline/`）**:
- `FeatureEncoder`: 各 MAP の変換テスト
- `FeatureStore`: save/load の整合性
- `calculate_recovery_rate()`: 境界値テスト

**スキーマ（`scraper/schemas.py`）**:
- 各カテゴリの必須フィールド検証
- N/A スタブ JSON の検証通過

### 5-2. Integration テスト

**`HybridStorage`**:
- GCS 保存 → ローカル L2 キャッシュ → 読み込みの一貫性
- `exists()` / `batch_check_keys()` の整合

**`ScrapeJobQueue`**:
- `add_job` → `claim_pending_jobs_batch` → `update_job_status` のフロー
- `precheck` → `pending` → `running` → `completed` の状態遷移
- `dedupe_key` による重複排除

**`date_coverage`**:
- `update_date_coverage()` → `/api/coverage-calendar` のレスポンス
- N/A インデックス記録 → `/api/date-race-matrix` での `null` 返却

### 5-3. E2E テスト

| エンドポイント | テスト内容 |
|-------------|----------|
| `GET /api/health` | 200 OK、レイテンシ ≤100ms |
| `GET /api/date-race-matrix` | `true`/`false`/`null` の型正当性 |
| `POST /api/scrape-missing` | キュー投入件数の正当性 |
| `GET /api/scrape-queue/status` | `pending`/`running`/`completed`/`failed` の件数整合 |

### 5-4. ML テスト

**テンポラルリーク検知（`tests/ml/test_temporal_leak.py`）**:
```python
def test_no_temporal_leak_in_snapshot():
    """
    全特徴量スナップショットで as_of_race_id 以降のデータが
    含まれていないことを検証する
    """
```

**データ分割検証**:
- `train_test_split(shuffle=True)` を使用していないことをチェック
- GroupKFold でのグループ（`race_id`）リークがないこと

---

## 6. 100% カバレッジ必須パス

| コード | 説明 |
|--------|------|
| `as_of_race_id` フィルタ処理 | テンポラルリーク防止の核心 |
| `calculate_recovery_rate()` | 回収率計算ロジック |
| N/A マーカー判定ロジック | `_not_available` フラグ処理 |
| スキーマ検証エラーパス | `SchemaValidationError` の発生・伝播 |

---

## 7. テストデータ管理

| データ | 場所 | 管理方法 |
|--------|------|---------|
| HTML サンプル | `tests/fixtures/html/` | 実際に取得した HTML のスナップショット |
| JSON フィクスチャ | `tests/fixtures/` | 手動管理 |
| Phase 0 サンプル | `tests/phase0/` | 10 レース分の固定データ |
| 学習用データ分割 | `tests/data/split/` | 2〜3 年分のサブセット |

**方針**:
- 実際の netkeiba HTML は `tests/fixtures/html/` に保存し、テスト実行時にファイルから読む
- GCS や netkeiba への実 HTTP リクエストは Unit/Integration テストで行わない（手動スモーク専用）

---

## 8. 禁止事項

| 禁止 | 理由 |
|------|------|
| `train_test_split(shuffle=True)` | 時系列リークが発生する |
| GCS パスのハードコード（`gcs_paths.py` 経由以外） | パス変更時の影響範囲が拡大する |
| テスト内での実 netkeiba HTTP リクエスト | CI 環境で不安定・netkeiba に負荷 |
| テスト内での `.env` の本番 GCS 接続 | テストが本番データを書き換える可能性 |
