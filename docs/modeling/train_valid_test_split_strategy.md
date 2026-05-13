# Train / Valid / Test 分割戦略（レース直前確率モデル）

## 1. 目的

本ドキュメントは、**通常の i.i.d. 想定の機械学習タスクとは異なる**競馬データにおいて、

- どの期間を **train / valid / test** に割り当てるか
- **時系列リーク**・**同一開催週の相関**・**改修・制度変更**をどう扱うか
- 分割を決める前に、**母集団の分布**をどう把握すべきか

を、`dataset_split_manifest.json` および既存のモデリング設計書と整合する形で整理する。

参照ドキュメント:

- `docs/modeling/advanced_modeling_techniques.md` — Purged CV / adversarial validation
- `docs/modeling/betting_backtest_dataset_spec.md` — 二次評価・馬券テーブル
- `docs/modeling/horse_pre_race_dataset_spec.md` — 母表・ラベル・OOF 方針
- `docs/modeling/reference_insights.md` — 時系列分割・運用成果物
- `docs/modeling/race_performance_rating_system.md` — 特徴モジュール単体の検証プロトコル（本書の一次分割とは別用途あり）
- `docs/modeling/hyperparameter_tuning_workflow.md` — **ハイパラ探索・inner/outer・試行予算**（valid/test の使い方と整合）
- `docs/modeling/feature_engineering_exploration.md` — **追加特徴の探索**（同一 split で ablation）
- `data/meta/modeling/dataset_split_manifest.json` — **機械上の正**（バージョン付きマニフェスト）

---

## 2. なぜ「ランダム split」が通用しないか

| 課題 | 説明 |
|------|------|
| **時間の因果** | 未来のレース・未来の成績が、過去の行に混入するとリークになる。 |
| **グループ相関** | 同一 `race_id` 内の馬行は独立ではない。**分割単位は必ず race 単位**。 |
| **開催週の連続性** | 近接日付のレースは馬場・天候・メンバー強度が似る。train と valid を日付で「接しすぎ」ないよう **purge / embargo** が必要。 |
| **分布シフト** | コロナ以降の開催密度、データ取得カバレッジ、指数・パドックの提供範囲、京都競馬場改修後の `venue_era` など、**年によって特徴の意味が変わる**。 |
| **希少カテゴリ** | 重賞・特定コースの head が小さい。ランダムだと valid に偶然偏りが出る。 |

以上より、**行単位ランダム split は禁止**（マニフェスト `constraints.no_random_row_split` と一致）。

---

## 3. 一次プロトコル（モデル選定・本番前評価）

`data/meta/modeling/dataset_split_manifest.json` の **`protocols.model_selection_primary`** を採用する。

| 役割 | 年（暦年・`race_date` 基準） | 用途 |
|------|------------------------------|------|
| **Train** | 2020, 2021, 2022, 2023 | 学習本体・（fold 内）TE / 集約特徴の学習 |
| **Valid** | 2024 | ハイパラ・特徴選抜・キャリブレーション fit（**test は見ない**） |
| **Test** | 2025 | **一度きりの最終報告**・本番前の汎化チェック |

- train / valid / test は **`race_id` 単位で完全分離**。
- **Calibration**（温度スケール・アイソトニック等）は **OOF 予測または valid のみ**に fit。test では fit しない。

### 3.1 本番再学習（モデル確定後のみ）

`protocols.production_refit_after_selection` に従い、選定完了後の本番用再学習では **2020–2025 を train に含めうる**。  
これは「test でチューニングした後」の手順であり、**test 指標を報告する前に実行しない**。

---

## 4. 交差検証（モデル開発の主戦場）

一次プロトコルの valid は **2024 年 1 本**のため、開発中の安定推定には **rolling-origin + purged group time series** を併用する。

マニフェスト `cv_policy`:

- `group_key`: `race_id`
- `time_key`: `race_date`
- `purge_window_days`: **7**（同一開催週付近の情報漏れを抑える保守的初期値）
- `embargo_days`: **3**（validation 直前の train を除外し、TE / 集約特徴のリークを抑える）

`rolling_origin_folds` は `2020-01-01` 起点で valid を半年ずつずらす定義が入っている。  
実装時は、母表の `race_date` と突き合わせ、**fold ごとに race が片側にのみ属すること**を自動テストする。

---

## 5. 補助評価セット（セグメント・ドリフト）

マニフェスト `special_sets` および `segment_evaluations` を活用する。

- **京都改修後** (`kyoto_post_renovation_*`): 現行コース条件への追従性
- **クッション観測あり** (`cushion_observed_test`): 数値馬場特徴の効き
- **パドック有無** (`with_paddock_test` / `without_paddock_test`): ExtendedV1 の依存度

一次の logloss / Brier / ECE に加え、**セグメント別**で性能とキャリブレーションを見る。

---

## 6. 他ドキュメントの「2020–2024 vs 2025」との関係

`race_performance_rating_system.md` では、**レースパフォーマンス特徴量単体**の寄与検証として、

- 履歴構築・学習側: 2020–2024
- ホールドアウト: 2025（かつ 2025 は online-safe 算出）

という **モジュールベンチ用**の切り方が記載されている。

**一次の確率モデル分割（2020–2023 / 2024 / 2025）とは目的が異なる**ため、両方を併存させる:

- **確率モデル本線**: マニフェスト §3
- **パフォーマンス信号の感度分析**: 上記ベンチ（長い train で信号の安定性を見る用途）

混同しないよう、実験ログに `protocol_name` を必ず残す。

---

## 7. 分布把握（EDA）— 実データに基づくサマリ

`horse_pre_race_dataset` 完成前でも、**同一カレンダー範囲・JRA 中央・race 単位**の派生表として  
`data/features/race_performance/20xx.parquet`（馬行 = レース内出走頭数の行）が揃っている。  
ここから **年次の規模・ラベルバランス・主要カテゴリ・数値列のドリフト**を把握した（集計日: リポジトリ上の Parquet 更新時点）。

### 7.1 規模（年 × 行数 × レース数）

| 年 | 行数（馬行） | ユニーク `race_id` |
|----|-------------|-------------------|
| 2020 | 47,876 | 3,456 |
| 2021 | 47,476 | 3,456 |
| 2022 | 46,840 | 3,456 |
| 2023 | 47,273 | 3,456 |
| 2024 | 46,752 | 3,454 |
| 2025 | 47,497 | 3,455 |

→ **年次の母数はほぼ一定**。valid(2024) と test(2025) のサンプルサイズ感は対称に近い。

### 7.2 ラベル（行加重の参考値）

| 年 | 勝ち (1着) 行割合 | 3着以内 行割合 |
|----|------------------|----------------|
| 2020 | 7.24% | 21.67% |
| 2021 | 7.30% | 21.86% |
| 2022 | 7.39% | 22.14% |
| 2023 | 7.32% | 21.95% |
| 2024 | 7.40% | 22.18% |
| 2025 | 7.29% | 21.85% |

→ **クラス比率は年間で大きくは揺れない**。時系列分割でも極端な imbalance シフトは起きにくい一方、**レース内相関**は依然として最大の設計制約。

### 7.3 カテゴリ分布（全期間合算の上位）

全期間で観測された主なカテゴリのシェア（**分割前の母集団把握用**）:

- **surface**: ダート > 芝 >> 障害
- **distance_band**: mile > sprint >> intermediate > long > extended
- **pace_shape_class**: balanced が大半、次いで grind（unknown / burst は少数）
- **class_group**: 未勝利・1勝クラスが中心、G・OP は少数だが重要セグメント
- **track_condition**: 良が大半、稍重・重・不良は少数だが馬場補正の検証に必要

重賞など head の小さいセグメントは、**全体指標だけに依存せず** `segment_evaluations` で切る。

### 7.4 数値列の年次ドリフト（例: パフォーマンス派生列）

`run_performance_final` の **平均**は年によって数ポイント程度変化（例: 2020 平均約 -21 → 2024–2025 平均約 -18 付近）。  
**標準偏差は年とともにやや縮小**（分布が少し締まっている可能性）。  
`race_level_pre_race` の平均も年次で緩やかに低下傾向。→ **adversarial validation（train vs test の分離容易度）**で確認することが `advanced_modeling_techniques.md` と整合。

`track_variant_online` は平均は 0 近辺だが、**最大値側に外れ値**（極端に速い日補正）が存在。学習時は **winsorize / robust scaling** を valid で調整するのが安全。

**2025 の `run_performance_final` 最小付近に極端値**が見える → 母表結合後も、**結合キー誤り・欠損埋め・単位崩れ**の監査対象とし、`dataset_profile` / `outlier_watchlist` に載せる。

---

## 8. 推奨ワークフロー（実装順）

1. `race_date` でソートした母表を構築（`horse_pre_race_dataset_spec.md`）。
2. マニフェストの **primary split** で holdout を固定し、**2025 test を触らない**。
3. **Purged + embargo 付き rolling CV** で開発（TE・OOF 中間特徴すべてに同一ポリシー）。
4. **2024 valid** でモデル・特徴・キャリブレーションを確定。
5. **2025 test** で最終報告 + `special_sets` / segment で補助診断。
6. 成果物: `dataset_profile.json`, `feature_missingness.csv`, `label_balance.json`（`reference_insights.md` の監査リスト）。

---

## 9. まとめ

| 項目 | 方針 |
|------|------|
| 分割単位 | **`race_id`（レース単位）** |
| 一次 train / valid / test | **2020–2023 / 2024 / 2025**（マニフェスト準拠） |
| 開発時 CV | **Purged group time series + rolling-origin** |
| ランダム行分割 | **禁止** |
| 分布・ドリフト | **年次 EDA + adversarial + セグメント** |
| 他用途の split | **race_performance 単体ベンチ**は別プロトコルとして明示 |

マニフェストのバージョンを上げる際は、本書の該当節と `reference_insights.md` の相互リンクを更新すること。
