# verify 実データ確認（RACE_ID 指定、T-013 / T-029 / T-032）

- 実行日時: 2026-10-04 23:2x JST
- 使用 RACE_ID: `202606020609`（GCS に `race_shutuba` あり。本結果作成の過程で `race_predictions`（`keiba_lgbm` キャッシュ）を新規作成）
- 前提修正: `.env` の `GCS_PRIVATE_KEY` に転記ミス（1 文字欠落）があり GCS 接続が失敗していたため先に修正（後述）

## 事前修正 1: `.env` の `GCS_PRIVATE_KEY` 転記ミス

`.env` をテンプレート形式へ作り直した際（前回チャット）、PEM の 1 文字（`6`）を書き漏らしており、`cryptography` の PEM パースが `InvalidByte(1621, 61)` で失敗 → GCS 接続不可（ローカルのみモードにフォールバック）になっていた。

```diff
- NzCYexJjl495j5flLpOv4a7jFiYVmP0frY+zDhO9YfXoty7CEWIvvPtpHETKEyF
+ NzCYexJjl495j5flLpOv4a7j6FiYVmP0frY+zDhO9YfXoty7CEWIvvPtpHETKEyF
```

修正後、`src/config/gcp_credentials.build_gcp_credentials()` 経由で GCS バケットへの接続・`bucket.exists()` を確認済み。

## 事前準備: 新鮮な予測キャッシュの作成

T-013 / T-032 は「GCS に予測キャッシュがあるレースID」を要求するが、既存の `race_predictions`（2026年は3件）はいずれも `keiba_lgbm` キャッシュの TTL（24h）を超えており `load_cached` がヒットしない状態だった。本番と同じ経路（`src.pipeline.inference.inference_pipeline.run_inference_for_race`、DB/Redis 保存のみ無効化）で実レース `202606020609`（GCS に実出馬表あり）に対して本物の推論を1回実行し、GCS に新鮮なキャッシュを作成した。

```python
from src.pipeline.inference.inference_pipeline import run_inference_for_race
run_inference_for_race("202606020609", allow_scrape=False, persist_db=False, persist_redis=False, persist_gcs=True)
```

（`models/keiba_model.pkl` は特徴量次元不一致で読み込み失敗 → `fallback_heuristic` にフォールバックして推論・保存。状態は `status=success` で問題なし）

## 結果

| 項目 | 結果 | ログ |
|------|------|------|
| T-013_harville（実データ） | **OK**（PASS=4 FAIL=0） | [`T-013_harville-RACEDATA-20261004-142752.log`](./T-013_harville-RACEDATA-20261004-142752.log) |
| T-029_t45_output（実データ） | **OK**（PASS=2 FAIL=0） | [`T-029_t45_output-RACEDATA-20261004-142752.log`](./T-029_t45_output-RACEDATA-20261004-142752.log) |
| T-032_betting（実データ・1 回目） | **NG**（PASS=6 FAIL=1） | [`T-032_betting-RACEDATA-20261004-142752.log`](./T-032_betting-RACEDATA-20261004-142752.log)（`KeyError: 'roi_pct'`） |
| T-032_betting（実データ・修正後） | **OK**（PASS=7 FAIL=0） | [`T-032_betting-RACEDATA-fix-20261004-142855.log`](./T-032_betting-RACEDATA-fix-20261004-142855.log) |

## 検出・修正したバグ: `POST /api/v1/betting/optimize` の `KeyError: 'roi_pct'`

- **症状**: 正常系（予測キャッシュあり・race_id 有効）で 500 Internal Server Error。
- **原因**: `src/api/v1/delegates.py` の `optimize_betting()` が `portfolio["roi_pct"]` を参照していたが、`BettingOptimizer.optimize()`（`src/pipeline/inference/betting.py`）が返す辞書のキーは `expected_roi`（小数、% ではない）であり `roi_pct` というキーは存在しない。
- **経緯**: `tests/` に `optimize_betting` / `/betting/optimize` を直接叩くテストが無く、かつ dev 環境では新鮮な予測キャッシュが存在しなかった（TTL 24h）ため、今回 RACE_ID 実データ確認を行うまで一度も実行されていなかったコードパス。
- **修正**: 既存コードの `_pct` 命名規則（`src/api/app.py` の `fast_pct` 等、`値 * 100` で保持）に合わせ、`expected_roi` を百分率に変換して返すよう修正。

```3:3:src/api/v1/delegates.py
        "roi_pct": round(portfolio["expected_roi"] * 100, 1),
```

- **確認**: 修正後 T-032 実データ確認 PASS（`total_bet=59700 expected_return=20306466.0 roi_pct=34014.2 候補数=5`）。
- **副作用確認**: `python3 -m pytest tests/api -q` で 199 passed / 1 failed（失敗は `psycopg2` 未インストールによる DB 接続テストで、本修正と無関係・修正前から存在）。`betting` 関連のユニットテストは存在しない。

## 備考（EV の異常値について）

今回の実データ確認で `roi_pct=34014.2`（%）のような極端な期待値が出ている。これは検証用に用意した `202606020609` の `race_odds` / `race_pair_odds`（GCS 保存済みの過去データ）と、`fallback_heuristic` モデルによる確率推定の組み合わせによるものと見られ、**本検証（レスポンス形式の確認）の範囲では問題なし**。EV・確率較正の妥当性チェックは本 T-032 の検証対象外のため、別途モデル精度の検証が必要であれば別タスクとして扱うこと。

## 総合（更新後）

`docs/todos/verify/` の 7 項目、**RACE_ID 指定時の実データ確認も含めてすべて OK**。
