# horse_pre_race_dataset 仕様

## 目的
`horse_pre_race_dataset` は、レース直前確率モデルのための学習母表である。  
`1行 = 1レース1頭` の粒度で、`T-10分` 時点までに観測できる情報だけを保持する。

本仕様では以下の 2 つを定義する。

- `horse_pre_race_dataset.parquet`
  学習・推論で共通利用する特徴量母表
- `horse_labels.parquet`
  学習・評価用の結果ラベル表

## 設計思想
- 最初から全特徴量を一括投入するのではなく、`ベースライン母表` と `拡張母表` を分けて比較可能にする
- ベースライン段階では、既存テーブルで coverage が高く安定している列だけを使う
- 拡張段階で、`race_oikiri`、`race_paddock`、`race_trainer_comment`、血統補助、クッション値 proxy を順次追加する
- 予測精度だけでなく、`確率の信頼性` と `後続のベット最適化で使える一貫性` を重視する
- Kaggle 実戦系の tabular テクニックを積極採用するが、必ず `OOF` と `時系列リーク防止` を前提にする
- 単段モデルだけでなく、`位置取り → ペース → 当日馬場タイプ → 血統文脈適合 → 最終確率` の多段階パイプラインに拡張できる形で保存する

## 機械学習テクニックの利用方針

## 1. カテゴリ変数エンコーディング
カテゴリ変数は 1 手法に固定せず、次のレイヤで比較する。

### Tier1: 低リスクで常設
- frequency encoding
- count encoding
- rare label grouping
- simple label encoding

### Tier2: OOF 必須
- target encoding
- CatBoost-style ordered target encoding
- expanding mean encoding
- group target encoding

### Tier3: 比較実験用
- target std / target skew encoding
- weight of evidence 風 encoding
- cross feature target encoding

## 2. target encoding の原則
- 全データでまとめて平均を計算しない
- `OOF` で作る
- unseen category は train fold global mean にフォールバックする
- rare category は global mean へ shrink する
- `jockey_name`, `trainer_name`, `sire`, `dam_sire`, `venue_surface_distance_key` などを主対象にする

## 3. CatBoost-style encoding の原則
- 時系列順に並べて、同一行より未来を参照しない ordered target mean を使う
- fold 内の train 部分だけで cumulative に更新する
- race 単位 group をまたいで leakage しない

## 4. 交差特徴
Kaggle の tabular 競技で効きやすい交差特徴を、過剰に増やさず上位から順に採用する。
- `jockey x venue`
- `trainer x venue`
- `sire x surface`
- `sire x distance_bucket`
- `sex x distance_bucket`
- `bracket_number x venue_surface_distance_key`
- `venue_era x surface`

## 5. OOF 集約特徴
- OOF でしか作らない
- train fold 内で集計し valid fold に map する
- 例
  - `jockey_win_rate_oof`
  - `trainer_top3_rate_oof`
  - `sire_win_rate_oof`
  - `jockey_venue_top3_rate_oof`
  - `trainer_surface_distance_win_rate_oof`

## 6. 数値列の実戦処理
- log transform
- rank-gauss / quantile transform
- clipping / winsorization
- race 内 rank / percentile / z-score
- high-cardinality group 内の robust scaling

## 7. モデル側で想定する比較軸
- LightGBM
- CatBoost
- XGBoost
- 線形 baseline

最初の本命は LightGBM / CatBoost の 2 本柱とし、OOF 確率をアンサンブル可能な形で保存する。

## ベースライン母表と拡張母表

## BaselineV1
最初に必ず作る比較用母表。coverage と保守性を優先する。

### 採用対象
- `race_shutuba_flat`
- `race_shutuba_past_flat`
- `race_index_flat`
- GCS 由来 horse bundle の履歴・血統基礎集約

### 一時的に外すもの
- `race_oikiri_flat`
- `race_paddock_flat`
- `race_trainer_comment_flat`
- クッション値の高度な proxy

### 目的
- まず安定したベースラインを作り、追加特徴量の寄与を差分で検証する

## ExtendedV1
BaselineV1 に直前状態・馬場 proxy を追加した母表。

### 追加対象
- `race_oikiri_flat` 集約
- `race_paddock_flat`
- `race_trainer_comment_flat`
- クッション値・含水率・proxy
- `venue_era_fit`, `seasonal_surface_fit`, `field_strength_proxy`

### 目的
- 直前情報を加えたときの `log_loss`, `Brier`, `ECE`, `ROI` の改善幅を測る

## 粒度とキー

## 主キー
- `race_id`
- `horse_number`

## 補助キー
- `horse_id`
- `venue_code`
- `race_date`

## レコード粒度
- `horse_pre_race_dataset`
  - 1レース1頭
- `horse_labels`
  - 1レース1頭

## ソース

## 直接ソース
- `race_shutuba_flat`
- `race_shutuba_past_flat`
- `race_index_flat`
- `race_oikiri_flat`（集約後）
- `race_paddock_flat`
- `race_trainer_comment_flat`
- `race_odds_flat`（LayerB 用）
- GCS 由来 horse bundle
  - 履歴
  - 馬プロフィール
  - 血統
  - 適性系補助
- JRA クッション値・含水率

## ラベルソース
- `race_result_flat`

## 明示除外
- `race_result_lap_flat`
- `race_barometer_flat`
- `smartrc_race_flat`

## 出力 1: `horse_pre_race_dataset.parquet`

## セクション A. キー・識別子
- `race_id`
- `horse_number`
- `horse_id`
- `horse_name`
- `venue_code`
- `venue`
- `race_date`

## セクション B. レース条件
- `surface`
- `distance`
- `direction`
- `grade`
- `race_class`
- `weather`
- `track_condition`
- `start_time`
- `field_size`
- `weight_rule`
- `course_type`
- `race_round`
- `venue_era`

## セクション C. 開催・季節文脈
- `month`
- `quarter`
- `season`
- `week_of_year`
- `venue_meeting_index`
- `days_since_meet_start`
- `days_since_last_race_for_venue`
- `recent_meet_density_14d`
- `recent_meet_density_30d`
- `surface_season_key`
- `venue_season_key`

## セクション D. 馬の基本属性
- `bracket_number`
- `sex`
- `age`
- `jockey_weight`
- `jockey_name`
- `jockey_id`
- `trainer_name`
- `trainer_id`
- `body_weight`
- `body_weight_change`
- `body_weight_missing_flag`
- `body_weight_change_missing_flag`

## セクション E. 指数・能力
- `speed_max`
- `speed_avg`
- `speed_distance`
- `speed_course`
- `speed_recent_1`
- `speed_recent_2`
- `speed_recent_3`
- `speed_index_missing_flag`
- `speed_rank_in_race`
- `speed_percentile_in_race`
- `speed_course_minus_race_mean`
- `speed_distance_minus_race_mean`

## セクション F. 直近走・戦績集計
- `prev1_finish` 〜 `prev5_finish`
- `prev1_last_3f` 〜 `prev5_last_3f`
- `prev1_weight` 〜 `prev5_weight`
- `prev1_weight_change` 〜 `prev5_weight_change`
- `prev1_distance` 〜 `prev5_distance`
- `prev1_surface` 〜 `prev5_surface`
- `prev1_track_cond` 〜 `prev5_track_cond`
- `prev1_time_sec` 〜 `prev5_time_sec`
- `prev1_pass_first` 〜 `prev5_pass_first`
- `prev1_pass_last` 〜 `prev5_pass_last`
- `prev1_field_size` 〜 `prev5_field_size`
- `prev1_finish_diff` 〜 `prev5_finish_diff`
- `avg_finish_5`
- `min_finish_5`
- `win_count_5`
- `top3_count_5`
- `avg_last_3f_5`
- `min_last_3f_5`
- `std_last_3f_5`
- `avg_time_sec_5`
- `days_since_last`
- `avg_interval`
- `position_trend`
- `last_3f_trend`
- `career_runs`
- `career_win_rate`
- `career_top3_rate`
- `same_surface_runs`
- `same_surface_win_rate`
- `same_dist_runs`
- `same_dist_win_rate`
- `same_venue_runs`
- `same_venue_win_rate`
- `career_avg_finish`
- `career_avg_last_3f`

## セクション G. 調教・直前状態
- `training_count`
- `best_training_rank`
- `training_time_4f`
- `training_impression_score`
- `oikiri_count`
- `oikiri_latest_days_before_race`
- `oikiri_latest_course`
- `oikiri_best_evaluation`
- `oikiri_mean_evaluation`
- `oikiri_best_lap`
- `oikiri_mean_lap`
- `oikiri_lap_count`
- `oikiri_has_comment`
- `oikiri_positive_keyword_score`
- `oikiri_negative_keyword_score`
- `oikiri_rider_is_jockey_flag`

## セクション H. パドック・厩舎コメント
- `has_paddock`
- `has_paddock_comment`
- `paddock_rank`
- `paddock_score`
- `paddock_positive_keyword_score`
- `paddock_negative_keyword_score`
- `paddock_rank_in_race`
- `paddock_rank_percentile`
- `has_trainer_comment`
- `trainer_comment_presence`
- `trainer_comment_eval_score`
- `trainer_comment_sentiment_score`
- `trainer_comment_positive_keyword_score`
- `trainer_comment_negative_keyword_score`

## セクション I. 血統・適性
- `sire`
- `dam_sire`
- `sire_cluster`
- `dam_sire_cluster`
- `sire_mstn_c`
- `sire_mstn_t`
- `dam_sire_mstn_c`
- `dam_sire_mstn_t`
- `mstn_cc_prob`
- `mstn_ct_prob`
- `mstn_tt_prob`
- `mstn_speed_index`
- `mstn_distance_affinity`
- `note_apt_dist_fit`
- `note_apt_l2`
- `surface_aptitude_proxy`
- `distance_band_aptitude_proxy`
- `heavy_track_aptitude_proxy`
- `firm_track_aptitude_proxy`

## セクション J. 独自追加特徴量
- `field_strength_proxy`
- `pace_pressure_proxy`
- `travel_burden_proxy`
- `freshness_cluster`
- `seasonal_surface_fit`
- `venue_era_fit`
- `draw_bias_adjusted`
- `market_independent_score_inputs_count`

## セクション K. 多段階パイプライン用中間特徴
- `expected_first_corner_pos`
- `expected_first_corner_pos_norm`
- `tracking_difficulty_score`
- `front_run_probability`
- `stalker_probability`
- `closer_probability`
- `pred_lap_1f`
- `pred_lap_3f`
- `pred_pace_class`
- `pred_energy_distribution`
- `day_bias_front`
- `day_bias_closing`
- `day_bias_inner`
- `day_bias_outer`
- `day_bias_stamina`
- `day_bias_burst`
- `day_bias_mud`
- `day_bias_speed`
- `contextual_bloodline_fit`
- `contextual_bloodline_speed_fit`
- `contextual_bloodline_stamina_fit`
- `contextual_bloodline_mud_fit`
- `contextual_bloodline_burst_fit`
- `same_day_sire_cluster_bias`

## セクション L. 欠損・カバレッジフラグ
- `has_race_index`
- `has_recent_5_runs`
- `has_training_summary`
- `has_oikiri`
- `has_oikiri_lap`
- `has_paddock`
- `has_trainer_comment`
- `has_cushion_value`
- `has_turf_moisture`
- `has_dirt_moisture`

## セクション M. LayerB 用市場特徴量
`horse_pre_race_dataset` は LayerA を基本とするが、後続の市場統合用に以下をオプション列として持てる構造にする。

- `win_odds`
- `place_odds_min`
- `place_odds_max`
- `place_odds_avg`
- `market_popularity`
- `market_implied_win_prob_raw`
- `market_rank_by_win_odds`
- `market_rank_by_place_odds`
- `market_gap_to_race_median`
- `market_gap_to_race_favorite`
- `win_place_spread`

## 出力 2: `horse_labels.parquet`

## キー
- `race_id`
- `horse_number`
- `horse_id`

## ラベル列
- `finish_position`
- `is_win`
- `is_top2`
- `is_top3`
- `field_size`
- `finish_position_pct`
- `margin_seconds_proxy`
- `race_has_official_result`

## メタ列
- `race_date`
- `venue`
- `surface`
- `distance`

## ラベル定義
- `is_win = 1[finish_position == 1]`
- `is_top2 = 1[finish_position <= 2]`
- `is_top3 = 1[finish_position <= 3]`
- `finish_position_pct = finish_position / field_size`

## 加工ルール

## 1. join ルール
- 基本 join key は `race_id + horse_number`
- `race_oikiri_flat` は `race_id + horse_number` に集約してから join
- `horse_id` は補助キーとして保持するが、`race_odds_flat` には存在しないため join 主キーにしない

## 1.5 集約ルール
- `race_oikiri_flat` は複数行を持ちうるため、必ず `race_id + horse_number` に集約する
- `race_shutuba_past_flat.past_races` は `対象レース日以前` の履歴に切ってから集約する
- `horse bundle` 由来集計も、対象レース日以前だけで計算する
- レース内相対指標は、同一 `race_id` の母表完成後に最後に計算する

## 2. 時点ルール
- 各特徴は対象 `race_date` より前の情報だけで計算する
- `days_since_last` や戦績集計は、当該レース当日以前の履歴だけに限定する
- クッション値は対象開催日朝時点の公表値を使う

## 3. テキスト列ルール
- 生テキストは原則保存しない
- 保存する場合も `*_raw_text` として分離し、学習投入対象から外す
- 初期版では、コメントはスコア化と presence flag のみを正式採用とする

## 4. リーク防止ルール
- 結果確定後テーブルは特徴量母表に入れない
- `LayerA` に当日オッズを入れない
- target encoding や集計は split 後・fold 内で作る
- `finish_position` を説明変数に戻さない
- レース単位で train / valid / test を分け、同一 `race_id` をまたぐ分割を禁止する
- 推論時点に存在しない future master や future encoded mapping を読み込まない
- ordered / target encoding は `purged_group_time_series_split` 前提で作る
- calibration は train データではなく valid もしくは OOF 予測のみで学習する

## 4.5 EDA・品質監査ルール
参考資料では、学習前に欠損・外れ値・分布確認を明示的に行う重要性が強調されていた。  
そのため、母表生成時に以下の監査成果物を必須にする。

- `dataset_profile.json`
  - 行数、列数、欠損率、主要数値列の分位点
- `feature_missingness.csv`
  - 列ごとの欠損率
- `segment_coverage.csv`
  - `year x venue x surface` 単位の件数
- `label_balance.json`
  - `is_win`, `is_top2`, `is_top3` の陽性率
- `outlier_watchlist.csv`
  - 明示的に clip / winsorize 候補とした列

## 4.6 外れ値・ノイズ方針
- 取消・除外・競走中止・失格は `label` 側で明示フラグ化し、Baseline 学習対象から外す候補として管理する
- 極端なタイム差・明らかな異常値は、単純除外ではなく `outlier_flag` を立てた上で比較検証する
- `race_oikiri` のようなノイズが強いテーブルは、最初は集約統計だけを使用し、生の列を直接広げない

## 5. 推奨データ型
- キー: string / int
- カテゴリ: string
- 数値集計: float32 もしくは float64
- presence flag: int8 / bool
- ラベル: int8

## 6. 期待する成果物
- `horse_pre_race_dataset.parquet`
- `horse_labels.parquet`
- `horse_pre_race_dataset_schema.json`
- `feature_dictionary.md`
- `dataset_profile.json`
- `feature_missingness.csv`
- `label_balance.json`
- `horse_latent_state.parquet`
- `positioning_features.parquet`
- `pace_scenario_features.parquet`
- `day_track_bias_features.parquet`
- `contextual_bloodline_features.parquet`

## 7. 成果物の版管理
継続的な改善を前提に、母表とラベルはファイルだけでなく `生成条件` も保存する。

実装時のメモリ節約・`polars` 中心の処理方針は  
`docs/modeling/modeling_coding_strategy.md` を正本とする。

### 必須メタデータ
- `data_version`
- `code_version`
- `source_table_versions`
- `venue_era_definition_version`
- `split_manifest_version`
- `created_at`
- `row_count`
- `feature_count`
- `baseline_or_extended`
- `encoding_recipe_version`
- `cv_policy_version`
- `calibration_recipe_version`
- `staged_pipeline_version`
- `upstream_stage_versions`
- **`mlflow_experiment_id`** / **`mlflow_run_id`**（学習・データ生成ジョブを追跡する。`modeling_coding_strategy.md` の MLflow 節と一致させる）

### MLflow との対応

- 母表ビルド・fold 学習・校正・（将来）馬券方策学習の **各ジョブで Run を分ける**か、**親子 Run** で階層化するかをプロジェクトで統一する。
- `dataset_build_manifest.json` に **`mlflow_run_id` を必ず書き戻す**と、ドキュメント上の `data_version` と実験が一対一で辿れる。

### 保存形
- `data/meta/modeling/dataset_build_manifest.json`
- `data/meta/modeling/dataset_profiles/<dataset_name>.json`

## 8. 実装順
1. `race_shutuba_flat` を母表に固定
2. `race_shutuba_past_flat`・`race_index_flat` を結合
3. `race_oikiri_flat` を馬単位に集約
4. `race_paddock_flat`・`race_trainer_comment_flat` を left join
5. GCS 由来 horse bundle 集約を追加
6. クッション値・開催文脈・`venue_era` を付与
7. LayerA と LayerB の列集合を分離
8. `race_result_flat` から `horse_labels` を生成
9. `BaselineV1` と `ExtendedV1` を同時に出力
10. 監査成果物を保存
11. `encoding recipe` ごとの OOF 特徴量を分離保存
12. `calibration input` を別出力
13. `tracking_difficulty` 出力を中間成果物として保存
14. `pace_scenario` 出力を中間成果物として保存
15. `day_track_bias` と `contextual_bloodline` を別成果物として保存
16. `polars scan` ベースの builder で fold 単位に materialize する
