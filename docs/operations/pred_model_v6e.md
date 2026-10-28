# 予測モデルを v6e（最新の親モデル）に切り替える

画面・API・集計（予測成績）・推奨馬券が参照する予測を、v4/v5 から v6e（`bundle_v6_2025`、前段 S3 e）に切り替える。決定ログ §72 以降・系譜監査参照。

## 仕組み
| 項目 | 内容 |
|---|---|
| 切替スイッチ | 環境変数 `KEIBA_PRED_MODEL`（既定 `v6e`）。`legacy`（または `v4`/`v5`/`off`）で従来（GCS > shadow=v4 > バックフィル）に戻る。`src/api/local_predictions.py` の `primary_model()` |
| 参照の優先順位（v6e） | ローカルの v6e（当日 `predictions_v6/<日>/<race_id>.parquet` > 過去分 `predictions_backfill/bundle_v6_2025/<日>/…`）> GCS の `race_predictions`（古いモデルが残っている）> v4 の shadow（`predictions_v3`）> 他のバックフィル。v6e がローカルに無いレースだけ従来の経路 |
| 反映される場所 | `/api/race/{id}/predictions`（画面の予測・印・期待値）、予測成績の集計（`prediction_results.load_predictions`）、推奨馬券 `optimize`/`target`/有料ゾーン（`delegates.py`）、レース一覧チップ・race-values（推奨を通す）、`pipeline_status` の AI 予測の時刻 |
| 当日の生成 | `weight_triggered_predict` が v4/v5 と並走して `predictions_v6` に書く。`KEIBA_V6_SHADOW` が未設定なら、`KEIBA_PRED_MODEL` が v6e のときに自動で有効（`v3_serving.v6_enabled()`）。明示は `KEIBA_V6_SHADOW=1/0` |
| レスポンス | `model_type=ensemble_v6e`、`model_version=bundle_v6_2025` |
| v6e の成果物 | バンドル `data/local/modeling/training_pipeline/final_v6e/bundle_v6_2025`、前段 `final_stage/v6_e_2025`、RB の履歴状態 `data/local/inference/state/rb_v6`（日次 `scripts/cron/rb_state_daily.sh`）、追い切り基準表 `tp35/tl_reference_v2.parquet` |

## 前段 S3B の入力（重要）
v6e の前段は、追走難度 v2・ラップ予想・位置取り予想の保存値（`tracking_v2/<日>/…`・`lap_forecast/<日>/…`）を入力にする。保存が無い日の行は欠損になる（学習では約 17% が欠損）。過去分のバックフィルでは、先に `python -m src.scripts.data.tracking_v2_backfill` と `tp55_backfill` で保存値を作ってから v6e を算出する。

## 過去分（2026 年）の再生成と差し替え
1. `tracking_v2_backfill --from 20260104 --to <直近>`、`tp55_backfill --dates …`（保存済みは飛ばす）。
2. 別の場所に v6e を算出: `KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.backfill_predictions --model v6e --all --out-root data/local/inference/predictions_backfill_v6e_staging --from 20260101 --to <直近> --max-load 20`（既存の予測があるレースも算出。出力済みは飛ばす＝再開可）。
3. 件数・確率の整合を確認してから、ホストで `sudo -n bash scripts/ops/swap_pred_v6e.sh` を実行する（staging の件数確認 → 既存の `bundle_v6_2025` を退避 → 1 回の `mv` で差し替え → 予測成績の日次記録を退避して作り直し → キャッシュの世代を更新。戻し方は実行の最後に表示される）。

## 切替・ロールバック
- 切替: コードを反映して FastAPI を再起動（キューが空なことを確認。`PORT=23220` を必ず指定）。`.env` の変更は要らない（既定が v6e）。
- ロールバック: `.env` に `KEIBA_PRED_MODEL=legacy` を足して再起動。当日の `predictions_v6` が作られるのを止めるには、あわせて `KEIBA_V6_SHADOW=0`。
- 2026 年は学習に使っていない（テストセット②）。再生成は推論のみ。

## 監視
- 凍結した馬券規則の追跡（`frozen_tracking.py`）は v4（`predictions_v3`）を入力にする設計のまま。モデル別の比較・ドリフト診断（`drift_monitor.py`、`model_perf_cache.py`）は v4/v5 の系列。v6e の系列は `predictions_v6` を足して見る。
