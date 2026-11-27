# 多層のアンサンブル（stacking）段 0 の計画と調べたこと

担当 stacking。2026-11-27 に始め、同じ日に統括役の指示で止めた（めぐ AI の再モデリングは、ここ数日 model-relabel のラベルの拡張を主にする）。
再開は、ラベルの拡張の結果が出た後に統括役から相談する。学習・集計はまだ 1 本も流していない。本番の経路は変えていない。

## 1. 結論

- 段 1（今の 7 部品だけで 2 層目を学ぶ）は、手元の材料だけで始められる。重い学習は要らない。
- 段 2（部品 20〜30 本の 3 層）は、2016 年起点の表で組める。1 層目の OOF は 2020〜24 年（t 年は 2016〜t−1 年で学ぶ）。
- 補助ヘッド（mtl_）の 2016 年の値だけは年の中の 4 分割の予測で、未来の月が混ざる。2016 年は 1 層目の学習の行にだけ入れ、評価の行には入れない。

## 2. 調べた材料

| 材料 | 置き場 | 中身 |
|---|---|---|
| 本番 v6e の 7 部品の OOF | `data/local/modeling/training_pipeline_v4/tp27_v6e/oof/*__v4.parquet` | 2022〜24 年・140,865 行・p_win・p_top2・p_top3。年ごとに前へ進める予測 |
| 本番 v6e の合成の予測 | `tp27_v6e/ens/v6.parquet`・`v6.json` | LOYO（1 年を外して重みを決める）の予測。勝率の LogLoss 0.20851・2 着内 0.32719・3 着内 0.41143 |
| 候補 v6e16 の 7 部品の OOF | `/home/jovyan/work/retrain_out/base_2016/tp27_v6e/oof/` | 2022〜24 年・140,867 行。2016 年起点で学んだ版 |
| 2016 年起点の特徴量の表 | `/home/jovyan/work/retrain_out/cycles16_fix3/X_xgb.parquet`（1.1GB）・`meta.parquet` | 1,231 列・477,671 行・2016〜2025 年。meta に field_size・race_class |
| 前段 S3B の列 | `base_2016/oof/stage_features_expanding.parquet` | 2016 年は空・2017 年以降は全部埋まる |
| 補助ヘッドの列 | `base_2016/final_aux/aux_train.parquet`（`rebuild_groups_2016.py:333`） | 2017 年以降は「その年の 10 日前まで」で学ぶ。2016 年は年の中の 4 分割 |
| 本番の学習表 | `training_pipeline_v4/features/train_table.parquet` | 2020〜2025 年。条件の列（field_size・surface・distance・race_class・grade・venue_code） |

注意:
- 学習表と 2016 年起点の表には 2025 年の行が入っている。読み込みの直後に落とす。
- `ens/v6.parquet` は LOYO なので、2022 年の重みに 2023・24 年が入っている。前へ進める評価の比べ先にそのまま使わない。
- `tp27_v6/ens/v6.json` は本番の束と値が違う。比べ先に使わない。
- Python は `/opt/conda/bin/python`（lightgbm 4.6・xgboost 3.2・catboost 1.2.10・torch 2.6 CUDA 可）。`/usr/bin/python3` には pandas が無い。

## 3. 3 層の設計

### 1 層目（部品 20〜30 本）
- 学習器: LightGBM・XGBoost・CatBoost（列の組と seed を替える）・NN・線形・LambdaRank（レースの中の順位）・近傍法。
- 列の組は 3 つ並べる。本番 v6e と同じ 1,231 列、めぐの AI v5bag の除外の規則（`mc_data.py` の `RICH_EXCLUDE_RE`）を通した列、学習と推論のずれの列を外した列。
- ずれの列は `tl_w1_`・`tl_w2_`・`wc_`・`course_type`・`f1_sw`・`f1_sim_res`（docs/operations/train_serve_skew_20261126.md）。部品が増えるほど、ずれの列を拾う部品も増えるため。
- 予測は年ごとに前へ進める入れ子で作る（`yearly_expanding_folds`）。t 年（2020〜24）は 2016〜t−1 年で学ぶ。無作為の k 分割は使わない。

### 2 層目（合わせ役 3〜5 本）
- ロジスティック回帰（stack_lr）・浅い LightGBM・小さな NN。
- 入力は 1 層目の logit（部品 × 勝ち・2 着内・3 着内）、レースの中の相対値（logit のレース内 z・順位）、レースの条件（頭数・クラス・芝ダ・距離）。
- t 年（2022〜24）は、1 層目の OOF の 2020〜t−1 年だけで学ぶ。

### 3 層目（仕上げ）
- 合わせ役の重み付きの平均。
- レースの中で勝率の和 1・2 着内の和 2・3 着内の和 3 にそろえる（`win_models.normalize_to_total`）。
- 順の制約（勝率 ≤ 2 着内 ≤ 3 着内）を入れる。今のコードは順を保証していない（統括役の調べ、11/27）。案は 2 つ。2・3 着内を勝率から割引 Harville で作る案と、max でそろえてから和を直す案。
- 帯の較正を入れる。今の v6e は勝率 0.1〜0.2 の帯で出し過ぎ、3 着内 0.5〜0.7 の帯で出し足りない（統括役の調べ・ECE 0.005〜0.013）。

## 4. 評価の決まり（結果を見る前に固定する）

- 評価する行は 2022〜24 年だけ。2025・2026 年は使わない。
- 指標は本物の 1 着・3 着内の対数損失・AUC・較正（ECE と帯ごとの表）・1 番手の勝率。回収率は記述だけにする（買い方に使うなら rules-council を通す）。
- 比べ方は 2 本並べる。
  - (a) LOYO 同士。v6e の採用と同じ物差しで、`ensemble.loyo_eval` に stack 系を通す。
  - (b) 前へ進める評価。2022 年で学んで 2023 年を測り、2022〜23 年で学んで 2024 年を測る。v6e の weighted_logit・greedy も同じ窓で当て直す（比べ先の窓をそろえる）。
- 採る基準は 2 つとも満たすこと。`feature_bakeoff.compare_oof` の上側の限界（`_hi`）が負。2023・2024 年の両方で同じ向き。v6 と v5 の差は約 0.0003 だったので、上積みも同じ桁と見る。
- 比べ先は本番 v6e を主にし、候補 v6e16 とめぐの AI v5bag（勝率だけ）を並べる。
- 新しい seed の分けは作らない（rules-council 1 号への連絡は不要）。

## 5. 段取りと負荷

| 段 | 中身 | 負荷 |
|---|---|---|
| 段 1 | 本番 7 部品の OOF で 2 層目を学び、(a)(b) で v6e と比べる | 14 万行のロジスティック回帰と浅い木。軽いが heavy.lock を通す |
| 段 2a | 部品 10 本（多様性が効くかを見る） | 1 本あたり 5 折 × 1 seed。XGBoost は GPU で 1 折 2〜3 分、LightGBM は CPU のスレッドを絞って 1 折 5〜10 分の見込み |
| 段 2b | 部品 20〜30 本・合わせ役 3〜5 本・仕上げ | 夜に 1 本ずつ。10 時間前後の見込み |
| 段 3 | 影で並べる | 本番に入れるかは統括役とユーザーの承認。市場の段を足すかは rules-council |

- 重い処理は `flock /tmp/claude-1000/heavy.lock nice -n 19 <命令>`。流す前に load 8 以下・空きメモリ 20GB 以上を確かめる。
- 開催日の 07:30〜17:00 は学習を流さない。
- 出力は `/home/jovyan/work/retrain_out/stacking/`。成果のページは docs/html/modeling/final/stacking.html（本と棚に登録）。
- 本番で全部の部品が同じ入力で動くかを、段 3 の前に確かめる（学習と推論のずれを作らない）。
