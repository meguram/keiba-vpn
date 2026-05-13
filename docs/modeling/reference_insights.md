# 参考資料から反映した設計メモ

## 目的
このメモは、レース直前確率モデル向けデータ準備企画書をブラッシュアップする際に参照した外部資料と、その反映ポイントを整理したものである。

## 参照資料
- [法政大学 経営数理工学研究室 2023年度卒論要旨 PDF](https://syslab.k.hosei.ac.jp/abst/2023-MN-RM.pdf)
- [Qiita: 機械学習で競馬の回収率140%超を達成](https://qiita.com/umaro_ai/items/d1e0b61f90098ee7fbcb)
- [Qiita: 「機械学習で競馬予想」をガチで作る](https://qiita.com/dijzpeb/items/db74aa9726aaf55201eb)
- [Speaker Deck: 競馬で学ぶ機械学習の基本と実践](https://speakerdeck.com/shoheimitani/machine-learning-with-horse-racing)
- [How to Do Target Encoding Without Data Leakage](https://medium.com/@prathik.codes/how-to-do-target-encoding-without-data-leakage-the-right-way-280bd24fbc81)
- [amerob/kaggle-for-ml-engineers](https://github.com/amerob/kaggle-for-ml-engineers)
- [autofepg](https://github.com/thomastschinkel/autofepg)
- [Benchmarking Categorical Encoders](https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8)
- [Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [Background: Time Series Cross-Validation (Purged CV)](https://www.waylandz.com/quant-book-en/Time-Series-Cross-Validation-Purged-CV)

## 既存コードから反映した点
- `src/pipeline/tracking_difficulty.py`
- `src/pipeline/pace_predictor.py`
- `src/research/bloodline_distance.py`
- `src/research/race_quality_model.py`

## 実装戦略の補足
- `docs/modeling/modeling_coding_strategy.md`
- `docs/modeling/race_performance_rating_system.md`

## 反映したポイント

## 1. ベースラインを先に固定する
Speaker Deck と法政大の資料では、まずは指数中心のシンプルなベースモデルを作り、その後に特徴量を段階的に追加して比較する流れが明確だった。

これを受けて、今回の設計では以下を明示した。
- `BaselineV1` を定義する
- `ExtendedV1` を別出力する
- 追加特徴量の効果は差分で評価する

反映先:
- `docs/modeling/pre_race_feature_whitelist.md`
- `docs/modeling/horse_pre_race_dataset_spec.md`

## 2. 確率の信頼性を ROI より先に評価する
Speaker Deck では、期待値が 1 を超えても高オッズ帯では小さな確率誤差で簡単に負けに転ぶ点が強調されていた。また `logloss` や `Brier score` のような確率の信頼性指標が重要だと示されていた。

これを受けて、今回の設計では以下を明示した。
- 一次評価は `log_loss`, `Brier score`, `ECE`
- ROI は二次評価
- `safe_ev` と `buffer` を前提にする
- 高オッズ帯は安全域付きで扱う

反映先:
- `docs/modeling/betting_backtest_dataset_spec.md`
- `data/meta/modeling/dataset_split_manifest.json`

## 3. TimeSeriesSplit / 時系列分割を採用する
法政大資料と Speaker Deck の双方で、競馬モデルでは時間順の検証が重要であることが示されていた。

これを受けて、今回の設計では以下を採用した。
- `2020-2023 train / 2024 valid / 2025 test`
- `rolling-origin` fold
- 行単位ランダム split 禁止
- train fold 内での集計・encoding

反映先:
- `data/meta/modeling/dataset_split_manifest.json`

## 4. 1着・2着内・3着内を分けて持つ
Qiita の 140% 記事では `1着指数`, `2着以内指数`, `3着以内指数` を併せて扱う思想があり、今回のゴールにも合致していた。

これを受けて、今回の設計ではラベルと出力を最初から 3 本持つ形にした。
- `is_win`
- `is_top2`
- `is_top3`
- `p_win`
- `p_top2`
- `p_top3`

反映先:
- `docs/modeling/horse_pre_race_dataset_spec.md`
- `docs/modeling/betting_backtest_dataset_spec.md`

## 5. MLOps / 継続開発前提で成果物を版管理する
Qiita の `dijzpeb` 記事では、単発 notebook で終わらせずに、ソースコード・データ・モデル・実験結果を整理して継続改善できる形にする重要性が強調されていた。

これを受けて、今回の設計では以下を追加した。
- `data_version`, `code_version`, `source_table_versions` を成果物に保存
- `dataset_profile.json`, `feature_missingness.csv`, `label_balance.json` を監査成果物として出す
- 閾値ルールを `bet_policy_thresholds.json` のような外部設定で管理する

反映先:
- `docs/modeling/horse_pre_race_dataset_spec.md`
- `docs/modeling/betting_backtest_dataset_spec.md`

## 6. EDA と欠損・外れ値監査を必須化する
法政大資料と Speaker Deck では、分布・欠損・相関・外れ値を先に点検することが、特徴量設計より前に重要とされていた。

これを受けて、今回の設計では以下を必須にした。
- `dataset_profile.json`
- `feature_missingness.csv`
- `segment_coverage.csv`
- `outlier_watchlist.csv`

反映先:
- `docs/modeling/horse_pre_race_dataset_spec.md`

## 7. 市場情報はモデル本体と意思決定層で分離する
Qiita の 140% 記事は「オッズを特徴量に使わない能力モデル」を示しつつ、Speaker Deck は「期待値計算には市場価格が必要」と示していた。

この緊張関係を解くため、今回の設計では次のように分けた。
- `LayerA`: 市場非依存の能力・状態モデル
- `LayerB`: 市場価格を使う意思決定・最適化レイヤ

反映先:
- `docs/modeling/pre_race_feature_whitelist.md`
- `docs/modeling/betting_backtest_dataset_spec.md`

## 8. OOF target encoding を正式採用する
Kaggle 系の実戦では、target encoding は強力だが leakage を起こしやすいため、OOF 前提で使うのが定石とされている。  
また、rare category を global mean に shrink し、unseen category に fallback を持つ実装が一般的である。

これを受けて、今回の設計では以下を追加した。
- `OOF target encoding`
- `CatBoost-style ordered target encoding`
- `OOF aggregate features`
- `encoding_recipe_version`

反映先:
- `docs/modeling/horse_pre_race_dataset_spec.md`
- `docs/modeling/advanced_modeling_techniques.md`

## 9. Purged / embargo 付き CV を採用する
Kaggle の時系列・非IIDタスクでは、通常の KFold より `PurgedGroupTimeSeriesSplit` が安全であるという知見が多い。

これを受けて、今回の設計では以下を追加した。
- `cv_policy.primary_scheme = purged_group_time_series_split`
- `purge_window_days`
- `embargo_days`
- OOF 特徴量も purge / embargo に従う制約

反映先:
- `data/meta/modeling/dataset_split_manifest.json`
- `docs/modeling/advanced_modeling_techniques.md`

## 10. CatBoost / LightGBM の二本柱を比較対象にする
Kaggle の tabular 実戦では、カテゴリ処理に強い CatBoost と、高速で拡張しやすい LightGBM の比較・ブレンドが定番である。

これを受けて、今回の設計では以下を追加した。
- `LightGBM baseline`
- `CatBoost baseline`
- `OOF blending`

反映先:
- `docs/modeling/advanced_modeling_techniques.md`

## 11. 期待値安全域をルール化する
Speaker Deck と Qiita の実践例では、高オッズ帯の期待値判定はわずかな確率誤差に弱く、そのまま買うと破綻しやすい点が繰り返し示されていた。

これを受けて、今回の設計では以下を追加した。
- `safe_ev`
- `buffer`
- `bet_policy_thresholds.json`

反映先:
- `docs/modeling/betting_backtest_dataset_spec.md`

## 12. Kaggle 的な比較可能性を設計に含める
Kaggle 実戦では、「手法を追加したら必ず OOF / holdout で差分比較する」流れが徹底される。

これを受けて、今回の設計では以下を追加した。
- `BaselineV1` / `ExtendedV1`
- ablation study
- feature block ごとの ON/OFF 比較
- adversarial validation の導入

反映先:
- `docs/modeling/horse_pre_race_dataset_spec.md`
- `docs/modeling/advanced_modeling_techniques.md`

## 13. 位置取りを前段モデル化する
既存の `src/pipeline/tracking_difficulty.py` には、脚質プロファイル・隣枠影響・頭数・コース条件から `position_deviation` を学習する設計がある。

これを受けて、今回の設計では以下を追加した。
- `expected_first_corner_pos`
- `tracking_difficulty_score`
- `front_run_probability`
- `stalker_probability`
- `closer_probability`
- 位置取りモデルを FinalProbability の前段に置く

反映先:
- `docs/modeling/staged_probability_pipeline.md`
- `docs/modeling/horse_pre_race_dataset_spec.md`

## 14. ペースを前段モデル化する
既存の `src/pipeline/pace_predictor.py` には、距離・馬場・頭数・前に行きたい馬の比率から `lap_time_1f` と `lap_time_3f` を予測する実装がある。

これを受けて、今回の設計では以下を追加した。
- `pred_lap_1f`
- `pred_lap_3f`
- `pred_pace_class`
- `pred_energy_distribution`
- ペースシナリオを Stage3 として独立

反映先:
- `docs/modeling/staged_probability_pipeline.md`
- `docs/modeling/horse_pre_race_dataset_spec.md`

## 15. 血統を当日文脈と組み合わせて使う
既存の `src/research/bloodline_distance.py` は sire / dam_sire と距離適性の関係を整理しており、`src/research/race_quality_model.py` は「そのレースでどのタイプが好走したか」を segment prior と結果整合で推定している。

この 2 つを組み合わせることで、
- 血統の静的距離適性
- その日の馬場タイプ / race-day bias

の相互作用として血統を扱う設計に広げた。

追加した考え方:
- `day prior + same-day posterior`
- `contextual_bloodline_fit`
- `same_day_sire_cluster_bias`
- `bloodline_day_fit_speed/stamina/mud/burst`

反映先:
- `docs/modeling/staged_probability_pipeline.md`
- `docs/modeling/horse_pre_race_dataset_spec.md`

## 16. Polars 中心の省メモリ実装を前提にする
現在のデータ量と WSL 環境を考えると、全年・全列を `pandas` で一括処理する実装は不安定である。

そのため、今回の設計では以下を追加した。
- `polars scan_parquet()` を第一選択にする
- fold 単位で materialize する
- OOF / stage 出力を都度 Parquet 保存する
- `float32/int8` への型削減を前提にする

反映先:
- `docs/modeling/modeling_coding_strategy.md`
- `docs/modeling/horse_pre_race_dataset_spec.md`
- `docs/modeling/advanced_modeling_techniques.md`

## 結論
今回のブラッシュアップで、企画書は「何を作るか」だけでなく、
- どう比較するか
- どの指標で良し悪しを判断するか
- どこでリークや過大評価が起きやすいか
- どう継続改善できる形にするか

まで含む設計に強化された。
