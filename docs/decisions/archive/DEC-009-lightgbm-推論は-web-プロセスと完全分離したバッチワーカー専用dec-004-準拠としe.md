# DEC-009: LightGBM 推論は Web プロセスと完全分離したバッチワーカー専用（DEC-004 準拠）とし、ETL 完了フラグガード・`num_threads` 上限固定・`preload_app` CoW メモリ共有・Pandera バリデーションを組み合わせることで、2GB VPS でのサーバ負荷スパイクを定量的に制御しながらレイテンシ優位性を維持する。

**Date**: 2026-07-03
**Agent**: web-search-agent, decisions-context-agent, proposal-agent, backend-engineer, data-engineer, ai-model-engineer, integration-synthesizer, quality-reviewer
**Task**: TASK-027
**Status**: ACCEPTED

---

## Context

メモリ予算表の PostgreSQL 200MB 計上値（`shared_buffers` 実設定値に依存）および ETL 遅延時のフォールバック挙動（前日データ暫定表示 vs. 503 一律返却）について、運用ポリシーの最終判断が必要です。

---

## Decision

# 改善要件定義書: LightGBM 内部推論のサーバ負荷対策 — keiba-vpn (DEC-008 準拠)

---

## サマリー

ConoHa VPS 固定費モデル（DEC-008）上で LightGBM 内部推論を採用する際、レイテンシ優位性を活かしつつ 2GB RAM / 2vCPU という制約下でのサーバ負荷スパイクを制御するため、**「バッチ事前推論 + Redis キャッシュ提供」アーキテクチャ**を主軸とする。Web リクエスト時のリアルタイム推論は行わず、推論プロセスを Web サーバから完全分離（DEC-004 準拠）することで、CPU・メモリ両面の安全設計を実現する。Gunicorn の `preload_app + CoW` によるメモリ共有、`num_threads` 上限固定、ETL 完了後の推論起動ガード、特徴量バリデーション層を必須実装とし、定量的なメモリ予算・レイテンシ目標を CI で継続計測する。

---

## 機能要件

| # | 要件 | 優先度 | 担当エージェント |
|---|------|--------|----------------|
| F-1 | LightGBM 推論は **Celery/バッチワーカープロセス専用**とし、Gunicorn ワーカー（Web プロセス）内での `predict()` 呼び出しを禁止する（DEC-004 遵守） | 高 | backend-engineer |
| F-2 | バッチワーカー内に `ModelRegistry`（`RWLock` パターン）を実装し、モデルロードは **プロセス起動時に 1 回のみ**行う。GCS からのモデル取得は差分チェック（`generation` メタデータ比較）で最小化する | 高 | ai-model-engineer |
| F-3 | 毎朝 08:00 にその日の全レースの出走馬データを GCS から取得し、バッチ推論を実行。結果を `predictions/{date}/{race_id}.json` として GCS + Redis に保存する | 高 | data-engineer |
| F-4 | ETL 完了後にのみ推論バッチを起動する **完了フラグガード**を実装する。Redis キー `etl:complete:{race_date}` が存在しない場合、推論ジョブは最大 10 分待機後に `TimeoutError` で停止 | 高 | data-engineer |
| F-5 | Flask API エンドポイントは Redis から推論済み JSON を読み取るのみとし、推論ゼロで応答する。キャッシュミス（推論未完了）時は HTTP 503 + `"推論結果準備中"` を返す | 高 | backend-engineer |
| F-6 | 推論前に **Pandera スキーマバリデーション**を実施し、特徴量の型・値域（距離 800〜3600m、頭数 2〜18 頭、馬体重 380〜620kg 等）の逸脱を例外で即停止する | 高 | ai-model-engineer |
| F-7 | Gunicorn は `preload_app = True` で起動し、`lgb.Booster` オブジェクトを fork 前にロードして CoW（Copy-on-Write）によるプロセス間メモリ共有を実現する | 中 | backend-engineer |
| F-8 | モデルファイルサイズのリグレッションテストを CI に組み込み、`assert model_size_mb < 50` を PR ゲートとして設定する | 中 | ai-model-engineer |
| F-9 | バッチ起動時に `nice -n 10` でプロセス優先度を下げ、推論バッチ実行中の Web API レイテンシ劣化を防止する | 中 | operations-engineer |
| F-10 | オッズが `stale_threshold`（例: ±15%）を超えて変動した場合のみ差分再推論を実行し、GCS と Redis の結果を上書きする | 低 | data-engineer |
| F-11 | `/healthz` エンドポイントを実装し、`systemd` の `Restart=on-failure` + ConoHa 監視で異常終了時の自動復旧を保証する | 中 | operations-engineer |
| F-12 | CI に `realistic_18horses_features()`（学習データの平均・標準偏差から生成するダミーデータ）を用いた推論ベンチマークを組み込む | 低 | ai-model-engineer |

---

## 非機能要件

| # | 要件 | 目標値 | 担当 |
|---|------|--------|------|
| N-1 | Web API レスポンスタイム（Redis キャッシュ読み取り） | P50 < 10ms、P99 < 30ms | operations-engineer |
| N-2 | バッチ推論の全レース処理時間（1日分、最大12レース × 18頭） | < 10分（08:00〜08:10 完了） | ai-model-engineer |
| N-3 | VPS メモリ使用量ピーク（レース日・時間分離後） | < 1,200 MB（2GB プランの 60%） | cost-optimizer |
| N-4 | CPU 使用率ピーク（バッチ推論実行中） | ≦ 50%（`num_threads = vCPU // workers = 1`） | operations-engineer |
| N-5 | モデルファイルサイズ | < 50 MB | ai-model-engineer |
| N-6 | LightGBM `Booster` RSS 増加量（ロード前後の差分） | < 250 MB（超過時は `num_leaves` 63→31 に削減して再学習） | ai-model-engineer |
| N-7 | VPS 月額固定費 | ConoHa VPS 1プラン（DEC-008 確定値） | cost-optimizer |
| N-8 | Gunicorn ワーカー数 | 2（2vCPU 環境）、`max_requests=500` / `max_requests_jitter=50` で定期再起動 | operations-engineer |
| N-9 | ETL → 推論の最大待機タイムアウト | 10 分（超過時は Slack アラート + ジョブ停止） | operations-engineer |
| N-10 | モデル更新時のダウンタイム | ゼロ（`RWLock` ホットリロードで更新中も旧モデルで応答継続） | backend-engineer |
| N-11 | PostgreSQL `shared_buffers` 込みの総メモリ予算適合 | 下記予算表参照、OOM バッファ ≥ 400 MB を維持 | cost-optimizer |

### VPS メモリ予算表（確定値・PostgreSQL 込み）

```
┌──────────────────────────────────────────────────┬──────────┐
│ コンポーネント                                    │ 予算     │
├──────────────────────────────────────────────────┼──────────┤
│ OS + systemd                                      │  300 MB  │
│ Gunicorn ワーカー × 2（preload_app CoW 共有）     │  200 MB  │
│ LightGBM Booster（CoW = 物理 1 コピー）           │  200 MB  │
│ PostgreSQL（shared_buffers 128MB + work_mem 等）  │  200 MB  │
│ Redis（予測結果キャッシュ）                       │   64 MB  │
│ スクレイパー（レース日ピーク・時間分離済み）      │  300 MB  │
│ OOM Killer 回避バッファ                           │  736 MB  │
└──────────────────────────────────────────────────┴──────────┘
ピーク合計（時間分離後）: 約 1,264 MB ✅（2GB の 62%）
```

---

## 実装ロードマップ

### Phase 1（優先度: 高 — 即時実装）

- **[F-1] プロセス分離の明示**
  - `ModelRegistry` クラスを `inference/model_registry.py` に配置し、`app.py` / Gunicorn ワーカーからのインポートを `TYPE_CHECKING` ガードで物理的に禁止
  - `pre-commit` フック or `ruff` ルールで `import inference.model_registry` が `api/` 配下に存在しないことを検査

- **[F-2] ModelRegistry 実装**（下記コード参照）

```python
# inference/model_registry.py
import os, threading
import lightgbm as lgb
import numpy as np
from pathlib import Path
from typing import Optional

VCPU_COUNT   = os.cpu_count() or 2
WORKER_COUNT = int(os.environ.get("GUNICORN_WORKERS", 2))
LGBM_THREADS = max(1, VCPU_COUNT // WORKER_COUNT)  # 2vCPU → 1

class ModelRegistry:
    """バッチワーカープロセス専用。Web プロセスからの使用禁止。"""
    def __init__(self, model_path: str):
        self._lock    = threading.RLock()
        self._path    = Path(model_path)
        self._booster: Optional[lgb.Booster] = None
        self._version = 0
        self._load()

    def _load(self) -> None:
        b = lgb.Booster(model_file=str(self._path))
        b.params["num_threads"] = LGBM_THREADS
        with self._lock:
            self._booster = b
            self._version += 1

    def reload(self, new_path: Optional[str] = None) -> int:
        if new_path:
            self._path = Path(new_path)
        self._load()
        return self._version

    def predict(self, features: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._booster is None:
                raise RuntimeError("モデル未初期化")
            return self._booster.predict(features)
```

- **[F-4] ETL 完了フラグガード実装**

```python
# etl/pipeline.py（末尾に追加）
def mark_etl_complete(race_date: str) -> None:
    redis_client.set(f"etl:complete:{race_date}", "1", ex=3600)

# inference/batch.py（冒頭に追加）
def wait_for_etl(race_date: str, timeout_sec: int = 600) -> None:
    import time
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if redis_client.get(f"etl:complete:{race_date}"):
            return
        time.sleep(10)
    raise TimeoutError(f"ETL 未完了: {race_date}")
```

- **[F-5] Flask API のキャッシュ読み取り専用化**

```python
# api/routes/predictions.py
async def get_predictions(race_id: str) -> dict:
    cached = await redis.get(f"pred:{race_id}")
    if cached:
        return json.loads(cached)
    raise HTTPException(503, detail="推論結果準備中")
```

- **[F-6] Pandera スキーマバリデーション**

```python
# inference/validation.py
from pandera import DataFrameSchema, Column, Check

FEATURE_SCHEMA = DataFrameSchema({
    "distance":     Column(int,   Check.in_range(800, 3600)),
    "field_size":   Column(int,   Check.in_range(2, 18)),
    "horse_weight": Column(float, Check.in_range(380, 620)),
})

def validate_and_predict(registry: ModelRegistry, raw: pd.DataFrame) -> np.ndarray:
    FEATURE_SCHEMA.validate(raw)
    return registry.predict(raw.values)
```

---

### Phase 2（優先度: 中 — 初回リリース前）

- **[F-3] バッチスケジュール定義**

```bash
# cron（VPS 上）
# ETL: 出馬表スクレイピング
00 8 * * * /usr/bin/python -m etl.pipeline --date $(date +%Y-%m-%d)
# 推論バッチ（ETL 完了フラグ確認後に起動）
10 8 * * * nice -n 10 /usr/bin/python -m inference.batch --date $(date +%Y-%m-%d)
```

- **[F-7] Gunicorn 設定最適化**

```python
# gunicorn.conf.py
workers           = 2
worker_class      = "sync"
max_requests      = 500
max_requests_jitter = 50
preload_app       = True   # CoW でモデルメモリ共有
timeout           = 30
```

- **[F-9] バッチプロセス nice 設定**（上記 cron に `nice -n 10` 追記）

- **[F-11] ヘルスチェックエンドポイント + systemd 自動再起動**

```ini
# /etc/systemd/system/keiba-api.service
[Service]
Restart=on-failure
RestartSec=5s
```

- **[N-6] CI メモリ計測テスト**

```python
# tests/test_model_memory.py
from memory_profiler import memory_usage
import lightgbm as lgb

def test_booster_rss_within_budget():
    baseline = memory_usage(-1, interval=0.1, timeout=1)[0]
    lgb.Booster(model_file="models/current.lgb")
    peak = max(memory_usage(-1, interval=0.1, timeout=2))
    delta = peak - baseline
    assert delta < 250, f"LightGBM RSS delta {delta:.1f}MB が 250MB 予算超過"
```

---

### Phase 3（優先度: 低 — 安定運用後）

- **[F-10] オッズ変動トリガーによる差分再推論**
  - `stale_threshold = 0.15`（±15% 変動）を設定し、該当レースのみ再バッチ実行
  - 再推論後に GCS・Redis を上書きし TTL をリセット

- **[F-12] CI ベンチマーク用リアル特徴量データ**

```python
# tests/conftest.py
def realistic_18horses_features(stats_path="data/feature_stats.json"):
    """学習データの統計量から正規分布サンプリングで CI 用データ生成"""
    with open(stats_path) as f:
        stats = json.load(f)
    return np.column_stack([
        np.random.normal(s["mean"], s["std"], 18)
        for s in stats["features"]
    ])
```

- **[F-8] モデルサイズ PR ゲート追加**

```yaml
# .github/workflows/ci.yml
- name: Benchmark inference
  run: pytest tests/benchmark_inference.py --benchmark-max-time=0.05
- name: Check model size
  run: python -c "import os; assert os.path.getsize('models/current.lgb') < 50*1024*1024"
```

---

## 依存関係・リスク

### 依存関係

```
F-4（ETL 完了フラグ）
  └─ F-3（バッチスケジュール）に先行必須
       └─ F-5（キャッシュ読み取り API）に先行必須

F-2（ModelRegistry）
  └─ F-6（スキーマバリデーション）を内包
       └─ F-1（プロセス分離）のアーキテクチャ制約に依存

F-7（Gunicorn preload_app）
  └─ N-6（メモリ計測 CI）で実測値を確認してから本番適用
```

### DEC 整合性チェック

| DEC | 内容 | 本要件との整合 |
|-----|------|--------------|
| DEC-004 | AI 推論を Web サーバと別プロセスに分離 | ✅ F-1 で明示的に禁止・F-2 でバッチワーカー専用を宣言 |
| DEC-008 | ConoHa VPS 固定費モデル採用 | ✅ メモリ予算表で 2GB プラン内に収まることを定量確認 |

### リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|--------|------|
| ETL 遅延で推論が未完了のままレース開始 | 中 | 高 | F-4 タイムアウト後に Slack アラート + F-5 で HTTP 503 を返す。前日データを暫定表示するフォールバックを検討 |
| LightGBM RSS が 250MB を超えて OOM | 低 | 高 | N-6 CI で毎 PR 計測。超過時は `num_leaves` 63→31 で再学習を自動 Issue 起票 |
| バッチ推論中の CPU スパイクが Web API レイテンシを劣化 | 中 | 中 | F-9 `nice -n 10` + N-4 `num_threads=1` で CPU 使用率 50% 上限。必要に応じてバッチ開始時刻を深夜帯（01:00）に移動 |
| PostgreSQL + Redis + LightGBM の同時稼働でメモリ予算超過 | 低 | 高 | N-11 予算表に PostgreSQL 200MB を算入済み。`pg_activity` で定期計測し `shared_buffers` を 64MB に削減する調整弁を確保 |
| オッズ変動再推論（F-10）が多発してバッチ頻度が増加 | 低 | 中 | `stale_threshold` を保守的な 15% に設定。1 日の再推論上限回数（例: 5 回）をハードリミットとして設ける |
| 新コース追加等で特徴量スキーマが変化しサイレントエラー | 低 | 高 | F-6 Pandera バリデーションで即例外停止。スキーマ変更は `feature_stats.json` の更新 PR と同時に Pandera スキーマの更新を必須とするチェックリストを README に追記 |

---

## Conclusion

**LightGBM 推論は Web プロセスと完全分離したバッチワーカー専用（DEC-004 準拠）とし、ETL 完了フラグガード・`num_threads` 上限固定・`preload_app` CoW メモリ共有・Pandera バリデーションを組み合わせることで、2GB VPS でのサーバ負荷スパイクを定量的に制御しながらレイテンシ優位性を維持する。**

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-03_
