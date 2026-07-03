# AREA-07 — モデリング管理要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-004(分離設計), DEC-006(コスト比較), DEC-009(詳細実装), DEC-010(Feature Store)

---

## 1. 設計原則

1. **Web プロセス内での `predict()` 呼び出し禁止**（ruff lint CI ゲート）
2. バッチ事前推論 + Redis キャッシュのみで API リクエストに応答
3. モデル学習は Cloud Run Jobs 専用（**VPS での学習は禁止**）
4. モデルは 1 プロセス内で 1 回のみロード（CoW で worker 間共有）

---

## 2. バッチ推論パイプライン（cron 08:10 JST）

```
ETL 完了フラグ確認: Redis key etl:complete:{race_date}
    │ max 10 分待機 → タイムアウト: Slack アラート + TimeoutError
    ▼
GCS から最新モデル取得（generation metadata diff-check で不要な再 fetch を回避）
    ▼
Pandera スキーマバリデーション
  - distance: 800〜3,600m
  - field_size: 2〜18
  - horse_weight: 380〜620kg
    ▼
LightGBM predict_proba（~12ms/レース）
    ▼
SHAP TreeExplainer（上位 10 頭のみ、時間分離実行）
    ▼
results: predictions/{date}/{race_id}.json → GCS + Redis
  inference_source = 'local'
  model_version = 現行バージョン
```

---

## 3. ModelRegistry 設計

```python
class ModelRegistry:
    """モデルを一度のみロードし RWLock でゼロダウンタイム hot-reload を実現"""

    def load(self, path: str) -> None:
        """GCS から fetch → /tmp にキャッシュ → lgb.Booster をロード"""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """RWLock read ロックでスレッドセーフに predict_proba"""

    def hot_reload(self, new_path: str) -> None:
        """RWLock write 取得 → 新モデルロード → 旧モデル解放"""
        # ゼロダウンタイムでモデル更新
```

**Gunicorn 設定（CoW 活用）**:
```python
preload_app = True  # 親プロセスでモデルロード → fork で worker 間共有
```

---

## 4. モデル学習（Cloud Run Jobs）

| 項目 | 値 |
|---|---|
| スケジュール | 毎週月曜 02:00 JST |
| リソース | 2vCPU / 4GB RAM |
| データソース | GCS `features/static/` + `features/dynamic/` |
| テンポラルリーク防止 | バリデーション日より前で **strict カットオフ** |
| 成果物 | `models/current/model.lgb` に upload |
| ロールバック保持 | N-1 世代（`models/v{N-1}/model.lgb`）|

---

## 5. モデルバージョニング

```
GCS パス（gcs_paths.py 定義）:
  MODEL_CURRENT_PATH  = "models/current/model.lgb"   ← 現行モデル
  MODEL_ROLLBACK_PATH = "models/v{version}/model.lgb" ← N-1 世代保持
  MODEL_PREFIX        = "models/v{version}/"

ロールバック: MODEL_ROLLBACK_PATH からリストア → ≤10 分で完了
予測キャッシュに model_version を含める（後からのトレース用）
```

---

## 6. SHAP 設計

| 項目 | 決定内容 |
|---|---|
| ライブラリ | SHAP TreeExplainer |
| 対象 | 上位 10 頭のみ（全頭は計算コスト大） |
| 実行タイミング | 推論バッチと **時間分離**（OOM 防止） |
| API | `/api/v1/shap/{race_id}` で取得（要認証） |
| 表示 | 上位 5 特徴量を自然言語で説明（FE-07 折りたたみ） |

---

## 7. 差分再推論

オッズが ±15% 超変動した際、該当レースのみ再推論:
- 1 日上限 5 回
- `inference_source = 'local'`（再推論後も）

---

## 8. 外部 WebAPI フォールバック

VPS 障害またはモデル更新中は外部 WebAPI を呼び出し:
- `inference_source = 'external'` を predictions に記録
- UI に推論ソースを表示（FE-06 カード）
- 応答 ≤3s 要件

---

## 9. CI ゲート

```python
# モデルサイズ上限
assert model_size_mb < 50

# Gunicorn ワーカー内での predict() 呼び出し禁止
# ruff rule: no-predict-in-gunicorn

# リアリスティック推論ベンチマーク
result = batch_predict(realistic_18horses_features())
assert result.latency_p99_ms < 200
```

---

## 10. 非機能要件（モデリング系）

| ID | 要件 |
|---|---|
| NFR-ML-01 | 単勝的中率: 初期 ≥20%, 中期 ≥30% |
| NFR-ML-02 | Logloss ≤2.2 |
| NFR-ML-03 | Calibration Error ≤0.05 |
| NFR-ML-04 | ROI ≥-15% |
| NFR-ML-05 | モデルファイル <50MB（CI 強制） |
| NFR-ML-06 | 推論バッチ ≤10 分/日（~200 頭） |
| NFR-ML-07 | Booster RSS 増加 ≤250MB（超過時 num_leaves 31 で再学習） |
| NFR-ML-08 | ロールバック ≤10 分 |
| NFR-ML-09 | テンポラルリーク防止（バリデーション日 strict カットオフ） |

---

## 11. Human Review 依頼事項

| 項目 | 優先度 |
|---|---|
| JRA スクレイピング ToS / robots.txt の法的確認（**ETL 開始前に必須**） | 緊急 |
| SHAP 時間分離設計のロードテスト（Phase 2 移行前） | 中 |
| ETL 遅延フォールバックポリシー: 503 全件 vs 前日データ配信 | 高 |
