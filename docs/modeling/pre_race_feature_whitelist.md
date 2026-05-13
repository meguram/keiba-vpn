# レース直前特徴量ホワイトリスト

## 目的
このドキュメントは、`T-10分` 時点で利用可能な情報だけを使って、`勝率`・`連対率`・`複勝率` モデルを構築するための入力列ホワイトリストを定義する。

方針は次の 2 層に分ける。

- `LayerA`
  市場から独立した実力・状態推定モデル用。`単勝/複勝/人気/連系オッズ` は入れない。
- `LayerB`
  市場統合・馬券最適化用。`LayerA` の馬単位確率に加えて、直前オッズ・連系オッズを追加する。

## 共通ルール

### 採用原則
- 母表は `race_shutuba_flat`
- 主キーは `race_id + horse_number`
- 結果確定後にしか得られない情報は不採用
- 疎な情報は捨てず、`presence_flag` とセットで残す
- 生のテキストはそのまま学習に入れず、集約・辞書化・スコア化して使う

### 明示的に除外するテーブル
- `race_result_flat`
- `race_result_lap_flat`
- `race_barometer_flat`
- `smartrc_race_flat`

### 注意
- `race_shutuba_flat` と `race_index_flat` に含まれる `odds` / `popularity` は使わない
- オッズは必ず `race_odds_flat` と `race_pair_odds_flat` から扱う
- `smartrc_race` は `runners` だけなら前情報化できる余地があるが、現時点では結果混入リスクが高いため最初のデータセットでは除外する

## ベースライン優先原則
参考資料では、最初に「指数中心のベースライン」を作ってから特徴量を段階的に足す戦略が有効だった。  
そのため、本プロジェクトでも最初の正式比較対象は次のように固定する。

- `BaselineV1`
  - `race_shutuba_flat`
  - `race_shutuba_past_flat`
  - `race_index_flat`
  - horse bundle 由来の戦績・血統集約
- `ExtendedV1`
  - BaselineV1
  - `race_oikiri_flat` 集約
  - `race_paddock_flat`
  - `race_trainer_comment_flat`
  - クッション値・開催文脈の強化特徴

この 2 つを常に同時出力し、追加特徴量の寄与を差分で評価する。

## テーブル別ホワイトリスト

## 1. `race_shutuba_flat`
### 粒度
- `race_id + horse_number`

### LayerA でそのまま採用する列
- `race_id`
- `date`
- `venue`
- `surface`
- `distance`
- `direction`
- `grade`
- `race_class`
- `weather`
- `track_condition`
- `start_time`
- `field_size`
- `race_name`
- `venue_code`
- `round`
- `weight_rule`
- `course_type`
- `horse_number`
- `bracket_number`
- `horse_name`
- `horse_id`
- `sex_age`
- `jockey_weight`
- `jockey_name`
- `jockey_id`
- `trainer_name`
- `trainer_id`
- `weight`
- `weight_change`
- `sire`
- `dam_sire`

### LayerA で派生に変換して使う列
- `date`
  - `month`, `quarter`, `season`, `weekday`, `days_since_year_start`
- `start_time`
  - `post_time_bucket`, `is_night_like`
- `venue`, `surface`, `distance`, `direction`
  - `venue_surface_distance_key`, `distance_bucket`, `is_inner_outer_proxy`
- `sex_age`
  - `sex`, `age`
- `weight`, `weight_change`
  - `body_weight_missing_flag`, `abs_weight_change`, `weight_change_bucket`
- `jockey_name`, `trainer_name`
  - target encoding は禁止しないが、必ず fold 内集計で作る
- `sire`, `dam_sire`
  - 血統カテゴリ、血統クラスタ、距離帯適性派生

### 除外する列
- `odds`
- `popularity`

### 必須フラグ
- `has_weight`
- `has_weight_change`
- `has_grade`
- `has_course_type`

## 2. `race_odds_flat`
### 粒度
- `race_id + horse_number`

### LayerA
- 不採用

### LayerB で採用する列
- `win_odds`
- `place_odds_min`
- `place_odds_max`
- `popularity`

### LayerB で派生に変換して使う列
- `place_odds_avg = (place_odds_min + place_odds_max) / 2`
- `market_implied_win_prob_raw = 1 / win_odds`
- `market_rank_by_win_odds`
- `market_rank_by_place_odds`
- `market_gap_to_race_median`
- `market_gap_to_race_favorite`
- `win_place_spread`

### 必須フラグ
- `has_win_odds`
- `has_place_odds`
- `has_market_snapshot`

## 3. `race_shutuba_past_flat`
### 粒度
- `race_id + horse_number`

### LayerA で採用する元列
- `past_races`
- `training`

### LayerA で集約して使う特徴量
- 直近 1-5 走の着順・距離・馬場・上がり
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
- `same_surface_runs`
- `same_surface_win_rate`
- `same_dist_runs`
- `same_dist_win_rate`
- `same_venue_runs`
- `same_venue_win_rate`
- `career_runs`
- `career_win_rate`
- `career_top3_rate`
- `career_avg_finish`
- `career_avg_last_3f`

### Market 派生で使う列
- `prev{i}_odds`
- `prev{i}_popularity`

### Market 派生としてのみ保持する特徴量
- `prev{i}_pop_rank_diff`
- `prev{i}_upset_flag`
- `avg_prev_pop_rank_diff`
- `popularity_trend_slope`
- `upset_count`

### 除外する生値
- `prev{i}_odds`
- `prev{i}_popularity`

### 必須フラグ
- `has_past_races`
- `has_recent_5_runs`
- `has_training_summary`

## 4. `race_index_flat`
### 粒度
- `race_id + horse_number`

### LayerA で採用する列
- `speed_max`
- `speed_avg`
- `speed_distance`
- `speed_course`
- `speed_recent`（結合時に `speed_recent_1` / `speed_recent_2` / `speed_recent_3` に展開。元リストのプレースホルダ `0` は null 扱い）

### LayerA で採用しない列（理由）
- `time_index_m` — 指数表 HTML 上の位置推定で抽出しており、欠損が数値 `0` と区別しづらい。特徴量ストアにも載せない。

### LayerA で派生に変換して使う特徴量
- レース内 `z-score`
- レース内 `rank`
- レース内 `percentile`
- `speed_course_minus_race_mean`
- `speed_distance_minus_race_mean`
- `speed_recent_trend_proxy`（派生は `speed_recent_1`〜`3` から計算）

### 条件付き・後回し列
- `all_txt_c`
  - 初期版では未採用
  - テキスト特徴として別系統で扱う場合のみ使用

### 除外する列
- `odds`
- `popularity`

### 必須フラグ
- `has_race_index`
- `has_speed_recent`

## 5. `race_oikiri_flat`
### 粒度
- 1頭1行ではない可能性があるため、そのまま join しない

### LayerA の採用元列
- `training_date`
- `course`
- `condition`
- `rider`
- `lap_times`
- `impression`
- `evaluation`
- `comment`

### LayerA で馬単位に集約する特徴量
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

### 必須フラグ
- `has_oikiri`
- `has_oikiri_lap`
- `has_oikiri_comment`

## 6. `race_paddock_flat`
### 粒度
- `race_id + horse_number`

### LayerA の採用元列
- `paddock_rank`
- `paddock_comment`

### LayerA で派生に変換して使う特徴量
- `paddock_score`
- `paddock_positive_keyword_score`
- `paddock_negative_keyword_score`
- `paddock_rank_in_race`
- `paddock_rank_percentile`

### 必須フラグ
- `has_paddock`
- `has_paddock_comment`
- `has_paddock_rank`

## 7. `race_trainer_comment_flat`
### 粒度
- `race_id + horse_number`

### LayerA の採用元列
- `comment`
- `evaluation`
- `trainer_name`
- `questioner`

### LayerA で派生に変換して使う特徴量
- `trainer_comment_presence`
- `trainer_comment_sentiment_score`
- `trainer_comment_positive_keyword_score`
- `trainer_comment_negative_keyword_score`
- `trainer_comment_eval_score`

### 必須フラグ
- `has_trainer_comment`
- `has_trainer_eval`

## 8. `race_pair_odds_flat`
### 粒度
- `race_id + odds_type + pair`

### LayerB で採用する列
- `odds_type`
- `pair`
- `odds`
- `popularity`
- `odds_min`
- `odds_max`

### LayerB で派生に変換して使う特徴量
- `pair_market_rank`
- `pair_implied_prob_raw`
- `wide_mid_odds`
- `pair_gap_to_market_best`

### 用途
- 馬券最適化専用
- 馬単位確率モデルの学習には入れない

## 9. GCS 由来の馬情報バンドル
`feature_builder` が参照している `horse_profile`, `race_history`, `pedigree`, `myostatin`, `aptitude` 系は LayerA の中心情報として採用する。

### LayerA で採用する派生の中心
- 戦績集計
- 直近走推移
- 距離帯適性
- 芝/ダート適性
- コース相性
- 血統適性
- ミオスタチン由来の距離適性 proxy

### ルール
- 生の post-race 結果をそのまま連結しない
- 対象レース日以前の履歴だけで集計する
- 集計時点基準を必ず明記する

## LayerA 最終採用対象
- `race_shutuba_flat` の非市場列
- `race_shutuba_past_flat` の履歴集約
- `race_index_flat` の指数列
- `race_oikiri_flat` の馬単位集約
- `race_paddock_flat` のスコア化列
- `race_trainer_comment_flat` のスコア化列
- GCS horse bundle 由来の履歴・血統・適性集約
- JRA クッション値・含水率・proxy 系

## LayerA の追加ルール
- `odds` を直接使わなくても、オッズと強く相関するリーキー派生を後から混ぜない
- カテゴリ encoding は必ず train fold 内で作る
- ベースライン段階では、生テキスト埋め込みや大型外部特徴量は入れない

## LayerB 最終採用対象
- LayerA の最終馬単位確率
- `race_odds_flat`
- `race_pair_odds_flat`
- 市場由来の差分特徴量

## LayerB の追加ルール
- 市場を使うのは `確率推定器` ではなく `意思決定器` 側と明確に区別する
- 高オッズ帯の期待値は誤差に弱いため、`安全域付き期待値` を前提に使う
- 最終オッズを使った評価は `simulation_only` として管理する

## 学習前チェックリスト
- `race_result`, `race_result_lap`, `race_barometer`, `smartrc_race` が混入していない
- `odds` / `popularity` が LayerA に入っていない
- `race_oikiri_flat` を 1頭1行に集約している
- sparse テーブルに `presence_flag` を付与している
- 未来日付の履歴が混ざっていない
- split 前に target encoding を作っていない
- BaselineV1 と ExtendedV1 の両方を比較可能な形で保存している
