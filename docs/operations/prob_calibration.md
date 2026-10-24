# 確率の市場補正（KEIBA_PROB_CALIBRATION）の運用

仕様・検証: `docs/html/modeling/chapters/prob_calibration_gap.html`（tp-70）。市場オッズは**モデルの特徴量ではなく**、確率の較正と馬券の判断にだけ使う。

## 何が変わるか
- フラグ `KEIBA_PROB_CALIBRATION=1`（`.env`、**既定オフ**）のときだけ、実オッズ（確定/直前）が全頭に揃う 8 頭以上のレースで、`win_prob / top2_prob / top3_prob` を
  `logit(p_cal) = a + w1·logit(p_model) + w2·logit(p_market)`（勝率=ブレンド、連対・複勝=人気区分別ブレンド）に置き換える。レース内の合計は 1 / 2 / 3。
- 元のモデルの確率は各馬の `win_prob_model / top2_prob_model / top3_prob_model`、レースごとのメタは `prob_calibration: {applied, reason, version, odds_phase, method}`。
  適用しない理由（オッズ未確定・予測オッズ・8 頭未満・設定なし）は `reason` に入る。適用しないレースは従来の確率・印のまま。
- EV（ev_win / ev_top2 / ev_place）・合成スコアと順位・印（◎○▲★△消）・買い推奨度・組合せ券の確率・推奨ポートフォリオは補正後の確率を使う。
  保存済みの予測（`predictions_v3`、GCS の `race_predictions`）は書き換えず、**読み出し時に計算**する。
- 二重補正を避けるため、補正したレースは `config/ev_calibration_pc.json`（補正後の確率から作った EV に当てはめ直した版）を使う（`ev_calibration.active_config`）。
  ★の EV 下限（0.80 / 0.88）・買い推奨度の閾値・消印のしきい値（`config/kesi_mark.json` の `p3_max_pc` = 0.06）も補正後の確率用に別にしてある。
- 予測成績（`prediction_results`）の日次集計は、補正したレースで元のモデルの勝ち馬 LogLoss（`logloss_model`）と補正後（`sum_calibrated`）を並べて持つ。

## 有効化（ホスト）
1. 反映ファイル: `src/utils/prob_calibration.py`、`src/utils/ev_calibration.py`、`src/utils/race_probabilities.py`、`src/api/local_predictions.py`、`src/api/app.py`、`src/api/v1/delegates.py`、
   `src/pipeline/inference/prediction_results.py`、`config/prob_calibration.json`、`config/ev_calibration_pc.json`、`config/kesi_mark.json`、（監視）`src/scripts/data/drift_monitor.py`。
   ホストは継ぎはぎなので、ファイル単位でバックアップ（`data/local/meta/cron_backups/deploy_*`）してから置く。
2. キュー（`/api/scrape-jobs` の running/queued）が 0 のとき API を再起動してから、`.env` に `KEIBA_PROB_CALIBRATION=1` を追記して再起動。
3. 確認: 実レースの `/api/race/<id>/predictions` に `prob_calibration.applied: true` と `win_prob_model` が付くこと。オッズ未確定のレースは `applied: false` と理由。
4. フロントを反映（`frontend_build.sh stg` → `--switch-frontend`）。

## ロールバック
`.env` から `KEIBA_PROB_CALIBRATION` を外して API を再起動（コードは何も変えない設計。フラグを外せばすべて従来の挙動）。

## 係数の再当てはめ（定期）
- 契機: 週次の drift_monitor の `pop_calibration`（人気区分別の較正誤差）が警報、または V5 への切り替え後。
- 手順（ホスト、`/tmp/heavy_run.sh` 経由）:
  1. `python -m src.scripts.data.tp70_diag prep --frame <frame_all> --rank-dir <rank_tbl> --out DIR`（V5 の LOYO OOF が揃ったら frame を V5 の確率に差し替える。直前オッズ T-10/T-5/T-2 が貯まったら時点別の frame も作る）
  2. `python -m src.scripts.data.tp70_fit compare --base DIR/base.parquet --out DIR`（前年までのウォークフォワードで LogLoss・ECE を確認）
  3. `python -m src.scripts.data.tp70_fit export --base DIR/base.parquet --out config/prob_calibration.json`
  4. `python -m src.scripts.data.tp70_pc ev / star / kesi --preds DIR/tp70_cal_preds.parquet`（補正後の確率の EV の当てはめ、★の下限、消印のしきい値を作り直す）
  5. 推奨ポートフォリオ・`cand_*`・`levels`・年別の範囲の作り直し（約 1.2 万レースで数分。ホストの betting データ `payoff.parquet`・`odds_pair.parquet`・`betting_2026/*` を手元にコピーして実行しても同じ）:
     `python -m src.scripts.data.tp70_portfolio build --preds DIR/tp70_cal_preds.parquet --bet <betting> --bet26 <betting_2026> --mode raw --out DIR/pf` と `--mode pc`（補正後の確率）→
     `python -m src.scripts.data.tp70_portfolio fit --out DIR/pf --config config/ev_calibration_pc.json --hot-off`（方式選択は 2024・25 年、2026 年は最終確認。標本が少なく検証も通らない定数は市場並みに置く。補正後はレースの期待回収率が券種構成でほぼ決まり予測の高い順に実現が上がらないため hot を出さない設定）。
  6. `race_value` は補正したレースとしないレースで段の閾値（`ev_calibration_pc.json` / `ev_calibration.json`）と当日の順位を別に付ける。応答の `prob_calibration` と `levels_prob_calibrated` を参照。
- 注意: 係数は確定オッズで当てはめた。T-60/T-15 の早いオッズでは市場の情報量が少ないぶん市場側の係数が大きすぎる可能性がある。時点別の比較は直前オッズの蓄積後。
