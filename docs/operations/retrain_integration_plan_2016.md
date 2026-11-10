# 2016〜 の表で作ったモデルを本番の推論に組み込む計画（案、2026-11-10）

再学習（`docs/operations/retrain_trials/README.md`）で合格したものを、本番の予測（推論）に入れるための計画。
**承認をもらってから実装する**（CLAUDE.md: 複数ファイルにまたがる設計の変更）。段 11（推論との一致の確認）の前提。

## 1. 何が変わるか（学習側で合格したもの）

| 部品 | 今の本番 | 新しい版 | 学習側の成果物 |
|---|---|---|---|
| 馬の履歴（self8 など） | 2020 年からの履歴、係数（斤量 β など）は 2020〜 で当てはめ | 2016 年からの履歴、係数は 2016〜2021 で当てはめ | `retrain_out/base_2016/serving_bundle_hh`・`features/history_params.pkl` |
| めぐ指数 v3a の基準 | `config/megu_v3a`（2020〜2025 で当てはめ） | 年ごとの as-of（推論の基準は直近の前年までで当てはめ） | `retrain_out/megu_v3a_ref2016-2024` ほか |
| RB（めぐ指数の履歴） | 2020〜 の走 | 2016〜 の走（as-of のめぐ指数） | `base_2016/rb_state`（学習側と完全一致を確認済み） |
| 補助ヘッド（mtl_） | 2020〜2025 で学習 | 2016〜2025 で学習（乱数 5 本） | `base_2016/final_aux/model_s*.pkl` |
| 前段 S3 | variant e（L2・300 本） | 試行 7（L1・相対値 40/80 列・位置取りのメンバー内の関係・乱数 3＋1 の平均・着順のレース内中心化） | 推論用の最終モデルは未作成（作る必要あり） |
| 追走難度 v2 | `config/tracking_difficulty_v2`（旧版） | 試行 6（LAP・LAPD 群・位置取りの関係・過去の終いの落ちの組み合わせ） | 推論用の最終モデルは未作成 |
| ラップ予想 | tp55（2020〜） | 2016〜 の LAP 群、直接予測（LAPD）を追加 | 推論用の最終モデルは未作成 |
| 位置取り予想 | 馬柱の集計（`position_forecast.py`） | 学習モデル（`pos_model.py`、試行 6） | 推論用の最終モデルは未作成 |
| 親モデル | `final_v6e/bundle_v6_2025` | `base_2016/final_v6e16/bundle_v6_2025`（2016〜2025 の全行） | 学習中 |
| オッズ予測 | `config/odds_pred` | 試行 6（学習し直し・L2 とソフトマックスの平均・複勝の写像と半々） | 推論用の最終モデルは未作成 |

## 2. 組み込み先（scout で確認）

- 前段 S3: `src/pipeline/features/serving_stage_rel.py:110-128`（StageModels）、`src/pipeline/inference/v3_serving.py:83,120,195`（KEIBA_V6_STAGE）
- 追走難度: `src/pipeline/models/tracking_difficulty_v2.py:24,46,83,245`、`src/pipeline/inference/tracking_v2_service.py:139-219`
- ラップ予想: `src/pipeline/inference/lap_forecast_service.py:20-65`、`src/scripts/data/tp55_backfill.py:32-50`
- 位置取り: `src/pipeline/inference/position_forecast.py:113,196,232`、`tracking_difficulty_v2.py:245`
- オッズ予測: `src/pipeline/inference/target_bets.py:35,69,111,164,296`（KEIBA_ODDS_PRED_CONFIG）、`final_odds_predictor.py:19`
- 馬の履歴: `src/pipeline/features/horse_history/accumulate.py:51,141`（`range(2020, …)`）、`common.py:14`（TRAINING_YEARS）
- 補助ヘッド: `src/pipeline/inference/v3_serving.py:50-54`（KEIBA_V3_AUX）
- RB: `src/pipeline/features/serving_v6.py:115-128`（KEIBA_V6_RB_STATE）
- 親モデル: `src/pipeline/inference/v3_serving.py:79-87,116-118`（KEIBA_V6_BUNDLE）

## 3. 進め方（案）

1. **学習年の定数を 2016〜2025 にする**（`training_years.py`）。履歴の蓄積（`accumulate.py`）の `range(2020, …)` を定数から読むようにする。
2. **推論用の最終モデルを作る**（2016〜2025 の全行。学習側と同じ作り方）: 前段 S3（試行 7 の作り方を `fit_final_stage_v6` に足す）、追走難度、ラップ予想（直接予測を含む）、位置取り予想、オッズ予測。
3. **推論の作り方を学習に合わせる**: 前段 S3 の相対値・中心化、位置取りの学習モデル、追走難度の新しい入力、オッズの平均と複勝の写像。
4. **めぐ指数の基準**: 推論の基準を「直近の前年まで」で当てはめた版に替え、年が替わるときに当てはめ直す運用にする。
5. **段 11**: 過去の日付を選んで、推論で作った特徴量・予測が学習側の値と一致するかを確かめる（`v6_parity`・`readiness_v6_sample`・`test_serving_v6.py`）。
6. **段 12〜18**: 過去分の予測の差し替え（★要 OK）、較正・印・馬券の再確認、stg → 本番の配備（★要 OK）。

## 4. 費用（CLAUDE.md の上限）

- 推論の処理は今と同じ場所（Cloud Run・Jobs）で動き、回数は変えない。履歴が 2016 年からになる分、日次の蓄積の入力が約 1.7 倍に増える（ホストの手元の表から作るので GCS の転送料は増えない）。
- 状態のファイル（履歴の束・RB の状態）の大きさが約 1.6〜1.7 倍になる。束は起動時に 1 回読むだけ（Cloud Run は同じ地域で転送料なし）。見積もりは実装時にコミットへ残す。

## 5. 決めてほしいこと

- この順で進めてよいか（特に、位置取り予想を馬柱の集計から学習モデルに替えること、前段 S3 の作り方を変えること）。
- 段 12（過去分の予測の差し替え）と段 18（配備）は、それぞれ実行の前に改めて確認する。
