# レース直前確率モデルに使う実戦的 ML テクニック

## 目的
このドキュメントは、競馬 tabular モデルの次フェーズで活用する機械学習テクニックを、実戦投入前提で整理したものである。  
特に Kaggle 系の tabular 開発で一般的な手法を、競馬データに適用する際のルールまで含めて定義する。

## 基本方針
- テクニックは多用するが、`リーク防止` を最優先にする
- まず `BaselineV1` を成立させ、その後にテクニックを段階追加する
- 追加したテクニックは、必ず **train 内 OOF** と **2024 valid** で比較・選定し、設定凍結後にのみ **2025 test** で最終報告する（**test を見ながらチューニングしない**）。`data/meta/modeling/dataset_split_manifest.json` の一次プロトコルに合わせる。
- 実装基盤は `polars` 中心で組み、OOF / stage 出力は fold 単位の Parquet に保存する

実装時のデータ処理・メモリ戦略は  
`docs/modeling/modeling_coding_strategy.md` を参照。

**ハイパーパラメータ探索・inner/outer 評価・試行予算の配分**は  
`docs/modeling/hyperparameter_tuning_workflow.md` を参照する（本書はテクニック集であり SOP ではない）。

**新規特徴量の探索候補**は `docs/modeling/feature_engineering_exploration.md`（LayerA 前提・検証プロトコル付き）を参照する。

## 1. カテゴリ変数エンコーディング戦略

## 1.1 基本カテゴリ処理
まずは低リスクの定番を常設する。

- frequency encoding
- count encoding
- rare label grouping
- label encoding

主対象:
- `jockey_name`
- `trainer_name`
- `sire`
- `dam_sire`
- `venue_surface_distance_key`
- `venue_era`

## 1.2 target encoding
Kaggle 実戦で最も有効な高カードカテゴリ処理のひとつ。  
ただし、**全件平均を直接流し込むのは leakage** なので禁止する。

### ルール
- `OOF` で作る
- fold ごとに train 側だけで category mean を作る
- valid 側にはその train 側統計だけを map する
- test 側は全 train 統計だけを使う
- rare category は global mean へ shrink する

### 主な候補
- `jockey_name -> is_win`
- `trainer_name -> is_top3`
- `sire -> is_win`
- `dam_sire -> is_top3`
- `jockey_name x venue_code -> is_top3`
- `trainer_name x surface x distance_bucket -> is_win`

### 保存する派生列の例
- `te_jockey_win`
- `te_trainer_top3`
- `te_sire_win`
- `te_jockey_venue_top3`

## 1.3 CatBoost-style ordered target encoding
Kaggle / tabular 実戦で安定しやすい target-based encoder の代表。

### 使いどころ
- 出走順・開催日順に累積できるカテゴリ
- `jockey_name`, `trainer_name`, `sire`, `horse_id`

### ルール
- `race_date` 昇順で並べる
- 同一サンプルより未来のターゲットを使わない
- レース単位 group をまたがないように注意する
- OOF TE と ordered TE の両方を試し、valid で比較する

## 1.4 複数統計量 target encoding
Kaggle 系ライブラリや実戦コードでは、mean だけでなく `std`, `skew`, `count` を合わせることが多い。

### 候補
- target mean
- target std
- target count
- top3 rate
- win/top2/top3 の multi-target encoding

### 例
- `te_jockey_win_mean`
- `te_jockey_win_std`
- `te_trainer_top3_mean`
- `te_sire_top2_mean`

## 2. 数値特徴量の実戦加工

## 2.1 log / clipping / winsorization
競馬データは裾が重い列が多いので、そのまま入れず比較する。

対象候補:
- `career_runs`
- `days_since_last`
- `avg_interval`
- `training_time_4f`
- `oikiri_best_lap`

## 2.2 race 内相対特徴
競馬は同一レース内比較が本質なので、tabular の絶対値より race 内相対化が効きやすい。

### 基本
- rank
- percentile
- z-score
- gap to best
- gap to median

対象候補:
- `speed_max`
- `speed_avg`
- `speed_course`
- `body_weight`
- `oikiri_best_evaluation`
- `paddock_score`

## 2.3 interaction / crosses
Kaggle 系で定番の交差特徴を、ドメインに沿って限定導入する。

### 優先候補
- `jockey x venue`
- `trainer x venue`
- `sire x surface`
- `distance_bucket x surface`
- `venue_era x surface`
- `bracket_number x venue_surface_distance_key`

## 3. 学習・検証テクニック

## 3.1 Purged Group Time Series Split
通常の KFold では leakage が起こりやすいので、時系列 + group split を前提にする。

### ルール
- split unit は `race_id`
- 時間キーは `race_date`
- purge window を設ける
- embargo を設ける

### 目的
- 近接時点の統計量リーク防止
- OOF target encoding の過大評価防止

## 3.2 adversarial validation
一次プロトコル（`dataset_split_manifest.json`）に合わせ、次の **二系統**を用途別に使い分ける。

1. **`train(2020-2023)` vs `valid(2024)`** … 選定年が過去 train とどれだけ分布がずれるか（**特徴選抜・校正前の sanity**）。
2. **`train(2020-2023)` vs `test(2025)`** … 将来ホールドアウトが過去とどれだけ離れているか（**汎化ギャップの早期警告**）。  
   ※ `2020-2024` を train に含めた上で 2025 と比較すると、**2024 が valid 設計と混線**するため、上記の切り方を推奨する。

### 使いどころ
- 2024 / 2025 が過去 train とどれだけ異なるかの確認
- venue_era, cushion, paddock などの drift 発見

### 期待する出力
- `adversarial_auc`
- drift 上位特徴量
- drift が強い segment の一覧

## 3.3 calibration
確率モデルでは最重要。  
Kaggle でも leaderboard より実運用を重視する場合、後段校正が効くことが多い。

### 候補
- temperature scaling
- isotonic regression
- Platt scaling

### ルール
- calibration は OOF または valid の予測に対してのみ fit
- test では fit しない
- `p_win`, `p_top2`, `p_top3` それぞれで比較する

## 4. モデル戦略

## 4.1 単体モデル
- LightGBM
- CatBoost
- XGBoost

### 推奨順
1. LightGBM baseline
2. CatBoost baseline
3. 2者の OOF blending

## 4.2 multi-model ensemble
Kaggle 実戦では、単体勝負より OOF アンサンブルの方が安定しやすい。

### 初期案
- `blend_p_win = 0.5 * lgbm + 0.5 * catboost`
- `blend_p_top2 = weighted average`
- `blend_p_top3 = weighted average`

### その後
- stacker を入れる場合も、stacker 学習は OOF 予測だけで行う

## 4.3 rank + probability の二層化
競馬では race 内順位が重要なので、確率モデルと ranking モデルを併用する余地がある。

### 構成案
- `ModelA`: `is_win`, `is_top2`, `is_top3` の確率
- `ModelB`: race 内 ranking score
- 最終出力では、`ModelA` を主、`ModelB` を tie-break / 安定化補助に使う

## 4.4 多段階パイプライン
単一の tabular 学習器に全部入れるだけでなく、中間状態を別モデルで出してから最終確率へ渡す。

### 推奨ステージ
1. `HorseLatentState`
2. `PositioningModel`
3. `PaceScenarioModel`
4. `DayTrackBiasModel`
5. `ContextualBloodlineFitModel`
6. `FinalProbabilityModel`

### ルール
- 各 stage の出力は `OOF` で生成し、次段に渡す
- 次段学習時に train fold へ valid fold の stage 出力を混ぜない
- stacking 時も `OOF only` を守る

## 4.5 疑似ターゲット / 補助タスク
Kaggle 実戦では、最終目的だけでなく中間タスクを補助学習すると安定しやすい。

### 候補
- `first_corner_rank_norm`
- `position_deviation`
- `lap_time_1f`
- `lap_time_3f`
- `pace_class`
- `day_bias_front_like`
- `day_bias_closing_like`

### 目的
- 最終着順確率の前段構造をモデルに学ばせる
- 特徴量ブロックの寄与を分解しやすくする

## 5. 特徴量選択・解釈

## 5.1 SHAP
- 重要特徴量の確認
- drift 時の原因分析
- venue 別の効き方差分確認

## 5.2 permutation importance
- SHAP と順位がズレる列の確認
- noisy feature の削除判断

## 5.3 ablation study
以下のブロックごとに ON/OFF 比較する。
- base race info
- history aggregate
- race_index
- pedigree
- oikiri
- paddock
- trainer_comment
- cushion / seasonal context
- target encoding block

## 6. 期待値最適化向けの追加テクニック

## 6.1 safe expected value
高オッズ帯では誤差に弱いので、`safe_ev` を使う。

- `raw_ev = p * odds - 1`
- `safe_ev = calibrated_ev - buffer`

## 6.2 policy search
Kaggle 的にはモデルとルールを分けて探索するのが扱いやすい。

### 探索対象
- `min_safe_ev`
- `max_odds`
- `min_model_confidence`
- `min_rank_gap`
- `max_tickets_per_race`

### 最初の比較単位
- 単勝
- 複勝

## 7. 実装時の優先順
1. `frequency/count encoding`
2. `OOF target encoding`
3. `CatBoost-style ordered encoding`
4. `purged group time series split`
5. `calibration`
6. `positioning / pace` の補助モデル
7. `day bias / contextual bloodline` の補助モデル
8. `LightGBM + CatBoost OOF ensemble`
9. `safe_ev` と policy search

## 8. やらないこと
- split 前の target encoding
- test を見ながらエンコーディングを作る
- market odds を LayerA に混ぜる
- 高次交差特徴を無制限に増やす
- ROI だけでモデル採択する
