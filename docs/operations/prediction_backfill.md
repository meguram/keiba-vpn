# 過去のレースの予測の算出（バックフィル）と、画面での出し方

先週までのレースで、GCS に予測が無いものの予測を、推論経路（`v3_serving`）で後から算出して画面に出す。

## 画面（API）
- `GET /api/race/{id}/predictions` の優先順位: **GCS の `race_predictions` > shadow（`data/local/inference/predictions_v3/<日>/`）> バックフィル（`.../predictions_backfill/<モデル名>/<日>/`）**。
- shadow / バックフィルのときは、応答に `prediction_source`（`shadow` / `backfill`）と `prediction_note` が付き、週間予測・レース詳細に「算出済み（参考）」「検証用（参考）」が出る。
- 確率はモデルの出力。期待値（EV）は保存済みの最新（過去のレースは確定）オッズから `勝率×単勝オッズ`、`複勝率×複勝オッズ平均` で計算（レース当日に画面へ出た値ではない）。
- コード: `src/api/local_predictions.py`、画面: `frontend/components/PredictionSourceBadge.tsx`。

## 算出
```
# 対象の数（計算しない）
sudo -n env KEIBA_ENV=stg PYTHONPATH=. KEIBA_INFERENCE_ENGINE=full python3 -m src.scripts.data.backfill_predictions --list --from 20260101 --to 20261018
# 実行（新しい日から。出力済みはスキップ＝止めても続きから）
sudo -n nohup env KEIBA_ENV=stg PYTHONPATH=. KEIBA_INFERENCE_ENGINE=full OMP_NUM_THREADS=2 nice -n 19 python3 -m src.scripts.data.backfill_predictions --from 20260101 --to 20261018 > /tmp/backfill.log 2>&1 &
# shadow（当時の bundle エンジン）との突き合わせ確認（10/18 の先頭 6 レース）
... backfill_predictions --check-shadow 20261018 --max-races 6
# 確定着順との簡易検証
python3 -m src.scripts.data.eval_backfill_predictions --kind backfill --label bundle_v4_2025
```
- 方式は `verify_v3_serving` と同じ: **その日より前の履歴 + 同日で先に発走したレースの結果だけ**（full エンジン。バンドルの Elo は現在値なので使わない）。モデルは 2020〜2025 年で学習したバンドル（`.env` の `KEIBA_V3_BUNDLE`）なので、**2026 年のレースだけが未学習の期間**（既定は 2026 年以降のみ。`--allow-in-sample` で解除できるが、学習期間の予測は参考にならない）。
- 1 レース約 30〜50 秒（ホストの負荷により変動。v4 のメンバー構成の群が約 30 秒）。36 レースの日で約 20〜30 分。負荷が `--max-load`（既定 18）を超える間は待つ。
- 止める: `kill <PID>`（途中のレースは捨てられ、出力済みのレースは残る）。
- 注意: 馬主・産地などのプロフィールは現在の値（所有者が後から変わった馬は当時と違う可能性）。出馬表は後の取得で上書きされた内容（馬体重は確定後の値）。
