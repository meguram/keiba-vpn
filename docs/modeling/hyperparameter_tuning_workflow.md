# ハイパーパラメータ・モデル選定のチューニング工程

## 0. 現状認識：議論は「終わっている」か

**終わっていない（未整理だった）領域である。**

- `train_valid_test_split_strategy.md` では **2024 valid でハイパラ・特徴選抜**とあるが、
  **探索手順・打ち切り条件・再現性・計算予算の配分**は別ドキュメント化されていなかった。
- `advanced_modeling_techniques.md` はテクニック集であり、**チューニング運用の SOP** ではない。
- 本書は **「どのデータで・どの指標で・どこまで探索し、いつ凍結するか」** と、
  **工程そのものの最適化（試行回数・壁時計時間・メモリ）** を定義する。

関連: `data/meta/modeling/dataset_split_manifest.json`、`train_valid_test_split_strategy.md`、`modeling_coding_strategy.md`。

---

## 1. 絶対ルール（リークと報告の潔さ）

| ルール | 内容 |
|--------|------|
| **test 禁止** | **2025** の指標を見ながらハイパラ・特徴列・校正器をいじらない。 |
| **一次の選定面** | **2024 valid**（または inner CV の OOF 集約）だけでチューニング判断。 |
| **校正** | 温度スケール等は **2024 valid または train 内 OOF** のみ fit。**2025 では fit しない**。 |
| **データ版の固定** | 1 つの tuning study では `data_version`・母表列集合・TE recipe を **凍結**。変えたら別 study として記録。 |

---

## 2. 二層構造：inner CV と outer valid

### 2.1 Inner（開発の主戦場）

- `rolling_origin_folds` + **purge / embargo**（manifest 準拠）で **複数 valid 窓**を回す。
- **各 trial** は「1 つのハイパラ設定」を **全 inner fold で学習し、集約スコア**（平均 logloss や平均 NDCG@3）を返す。
- 目的: **2024 1 年だけに過適合した設定**を早期に弾く。

### 2.2 Outer（最終ゲート）

- inner で上位 K 件（例: 3〜5）に絞ったうえで、**同一パイプライン・同一 feature** のまま **2024 valid 全体**で再学習・再評価。
- **2025 test** は、その設定が凍結されたあと **一度だけ**報告用に実行。

### 2.3 いつ nested が必要か

- 試行数が少ない・探索空間が狭い → **flat valid（2024 のみ）**でも可。
- 試行数が多い・Optuna 等で自動探索 → **inner CV スコアを primary** にし、2024 は **最終 sanity** として使うと安全。

---

## 3. 最適化する「対象」の順序（凍結の順）

過剰な同時探索を避け、**下流ほど探索空間を狭くする**。

1. **母表・TE recipe・欠損方針**を固定（BaselineV1 または ExtendedV1 のどちらか一方）。
2. **単一モデル族**（例: LightGBM pointwise）で木・正則化・学習率を tuning。
3. **別モデル族**（CatBoost）を同じグリッド思想で tuning → **アンサンブル重みは valid のみ**。
4. **ランキング枝**（lambdarank / YetiRank）は pointwise が安定してから **別 study**。
5. **多段 stage** のハイパラは **段ごと**に区切る（前段を毎 trial 再学習しないよう、前段は checkpoint 再利用）。
6. **L2 メタ**は L1 の OOF が揃ってから、**極小探索空間**（線形ブレンド係数など）。

---

## 4. チューニング「工程」の最適化（計算コスト・壁時計）

### 4.1 粗から細へ（Coarse-to-fine）

- **第 0 相**: ランダム探索を **年サブセット**（例: 2022 のみ）または **レース数サンプリング**で実施し、明らかに悪い領域を切る。
- **第 1 相**: 残候補を **全 train 年** + inner CV で再評価。
- **第 2 相**: 上位のみ **木本数を増やす** / early stopping round を伸ばす本番寄り設定。

### 4.2 多保真度（Multi-fidelity）

- **少ない boosting round** や **縮小特徴集合**で順位付けし、上位だけフル budget。
- LightGBM / XGBoost / CatBoost いずれも **early stopping** を primary にし、**同じ valid fold** で比較する。

### 4.3 探索アルゴリズム

- 次元が低〜中: **Bayesian optimization（TPE / Gaussian Process）** — Optuna 等。
- ベースライン比較: **RandomizedSearch** で十分なことが多い（競馬はノイズが大きい）。
- **グリッド全探索**は次元爆発しやすいので、**重要次元だけグリッド**＋他はランダム推奨。

### 4.4 並列化と I/O

- **trial 間は独立** → 並列 worker。ただし **WSL メモリ**に合わせて同時 trial 数を制限（`modeling_coding_strategy.md`）。
- **fold 行列の materialize は 1 回キャッシュ**し、同一 fold で複数 trial が読み替えるだけにする（Polars → 学習直前 numpy）。

### 4.5 早期打ち切り（Study レベル）

- **ASHA / Successive Halving** 的運用: 中間エポック（または round 10 のスコア）が下位分位以下なら trial 打ち切り。
- **Pruner**（Optuna）と inner fold の平均を組み合わせると試行数対効果が良くなる。

---

## 5. 目的関数の設計（単一指標の宣言）

- **1 study につき primary メトリックを 1 つ**決める（例: `mean_logloss` または `mean_ndcg@3`）。
- 副次指標（ECE、重賞セグメント）は **ログに残すが primary にはしない**（多目的最適化は複雑化の割に得が薄いことが多い）。
- **ランキング枝**と **pointwise 枝**は **別 primary** で別 study にする。

---

## 6. アンチパターン

- train+valid を結合して early stopping や探索に使う。
- test を見たあとに「もう一度だけ」グリッドを広げる。
- TE recipe と木の深さを **同一 trial で毎回変えて**比較する（要因分解不能）。
- inner fold を無視して **2024 のたまたま良い日付**にだけ合う設定を採用する。

---

## 7. 成果物（study ごとに保存）

- `study_id`, `git_commit`, `split_manifest_version`, `feature_list_hash`
- 各 trial: `params`, `inner_fold_scores`, `aggregated_score`, `wall_time_sec`, `gpu_mem_peak`（任意）
- **採用 config** の YAML または JSON（本番 refit でそのまま読める形）

推奨パス例: `data/meta/modeling/tuning_studies/<study_id>/`

### 7.1 MLflow での必須記録

各 study・各 trial（少なくとも **採用 trial と最終 refit**）について、**MLflow Run** を切る。

- **Parameters**: ハイパラ全集、`split_manifest_version`、`feature_list_hash`
- **Metrics**: inner fold 集約スコア、2024 valid 再評価スコア、壁時計、`n_trees` 等
- **Tags**: `study_id`, `trial_number`, `git_commit`, `primary_metric`
- **Artifacts**: 採用 `config.yaml`、特徴量リスト、（任意）OOF 予測サンプル

`horse_pre_race_dataset_spec.md` の必須メタデータと **`mlflow_run_id` を相互参照**できるようにする。

---

## 8. チェックリスト（2025 報告前の凍結）

- [ ] primary metric が study 冒頭で文書化されている  
- [ ] 2025 を一切参照していない  
- [ ] 採用モデルの `model_version` / `data_version` / `encoding_recipe_version` が一意に決まる  
- [ ] 校正器は valid または OOF のみ fit  
- [ ] ランキング枝を併用する場合、**馬券用確率**はどの head から出すか明記されている  
- [ ] **MLflow** に採用 run が記録され、`mlflow_run_id` が成果物マニフェストと一致する  

---

## 9. まとめ

チューニング工程の議論は、**split・リーク・実装メモリ**とは別軸で、本書まで含めて初めて **運用上の輪郭が閉じる**。  
今後の変更は **study 単位**でバージョンを切り、`runs.jsonl` または tuning ディレクトリに追記すること。

---

## 10. 現行実装（`pipeline/`）とのギャップ — 実装を踏まえた議論

ドキュメント（`dataset_split_manifest.json`・本書・`train_valid_test_split_strategy.md`）と、リポジトリ内の学習コードは **2026 時点で完全には一致していない**。  
モデル選定 SOP をコードに落とすときは、次の差分を前提にマイルストーンを切る。

### 10.1 `pipeline/trainer.py`（`ModelTrainer`）

| ドキュメント側の意図 | 実装の現状 | リスク |
|----------------------|------------|--------|
| **暦年・race 単位**で train / valid / test を分離 | `split_idx = int(len(X) * (1 - test_ratio))` で **行の先頭からの割合分割**（DataFrame の行順依存） | 行順が日付順でないと **時系列 split にならない**。同一レースが train と test に跨る可能性。 |
| 2024 valid / 2025 test など manifest 準拠 | `years` は `_load_or_build` の一部経路でしか効かず、分割は **年ではない** | 報告用 test を「一度きり」に固定できない。 |
| relevance による LambdaRank | `y` は **複勝圏の二値** `(1<=着順<=3)` をランキングラベルに使用 | 着順の強弱が損失に反映されず、**順位本来の listwise 意図とずれる**（当面のベースラインとしてはあり得るが、ドキュメントの LTR 節とは別物として明示すべき）。 |
| Optuna によるハイパラ探索 | `n_optuna_trials` は **MLflow へのタグ記録**中心で、`train()` 本体に **探索ループは無い** | ドキュメントの「study・pruner」は **未接続**。 |

**やれること（実装）**: `race_date`（または `race_id` 先頭年）でソート → `dataset_split_manifest.json` を読み、**eval 用 race_id 集合**でマスク分割するモジュールを `ModelTrainer` 前段に挿入する。LambdaRank 用は `relevance = f(finish_position)` を別列で渡すオプションを追加する。

### 10.2 `pipeline/ensemble_trainer.py`（`EnsembleTrainer`）

| ドキュメント側の意図 | 実装の現状 | リスク |
|----------------------|------------|--------|
| Purged + embargo 付き時系列 CV | **GroupKFold(race_id)** は train 内の **同一レース同一 fold** に閉じるが、**日付に基づく purge は無い** | 近接開催の情報リークはドキュメントより緩い。 |
| メタ学習は valid のみ fit 等 | メタは **train 区間の OOF** に対して `LogisticRegression` fit（設計としては標準的） | **最終報告用の 2024 ブロックを完全にホールドアウト**する設計にはなっていない（train の一部の OOF 上でメタが学習される）。 |

**やれること（実装）**: `GroupKFold` を維持しつつ、fold 境界を **`race_date` + purge_window_days** で切る `PurgedGroupTimeSeriesSplit` 相当のイテレータに差し替える。メタ用に **2024 行を学習から除外した OOF** を別パスで生成するオプションを検討する。

### 10.3 `pipeline/feature_store.py`

- 列指向・スナップショット・`_registry.json` は **`modeling_coding_strategy.md` の方針と整合**しやすい。
- 学習行列の **行順・重複 race** がスナップショット生成側でどう保証されるかは、`build_training_matrix` 呼び出し規約で明示する必要がある（trainer の分割と連動）。

### 10.4 `research/evaluate_race_performance_signal.py` 等

- ベンチ用スクリプトは **独自のデータ結合・特徴**を持つ。**本番 `ModelTrainer` と同じ split とは限らない**。
- 指標比較は「信号検証」としてログに残し、**本番選定スコアは manifest 準拠の runner 一本**に寄せるのがよい。

### 10.5 推奨マイルストーン（実装順）

1. **Split モジュール**: manifest 読込 + `race_id` / `race_date` による train/valid/test マスク（**trainer から切り出し**）。  
2. **trainer**: 上記マスクに差し替え、`test_ratio` ベース分割を **deprecated** または「デバッグ専用」と明記。  
3. **relevance 列**（任意）と二値 top3 の **切り替えフラグ**。  
4. **Optuna**: `train()` から `study.optimize` を呼ぶオプション、または CLI `pipeline/cli.py` から study 専用コマンド。  
5. **ensemble**: purge 対応 fold → メタの fit 対象データをドキュメントの outer valid 方針に揃えるオプション。

以上を進めると、**議論（md）と実装（pipeline）の二重管理**が減り、`study_id` と `split_manifest_version` を同一 run で記録できるようになる。
