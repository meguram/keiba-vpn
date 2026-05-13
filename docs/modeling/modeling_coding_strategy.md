# モデリング実装時のコーディング戦略

## 目的
このドキュメントは、`2020-2025` の年次ローカルテーブルと GCS 由来補助データを用いて、  
メモリを極力抑えながら、再現性のあるモデリング基盤を実装するためのコーディング戦略を定義する。

特に以下を重視する。
- `WSL` 上でも安定するメモリ使用量
- `OOF` / `purged CV` / 多段 stage に対応できる中間成果物管理
- `polars` を中心に据えた lazy 処理
- 速度よりもまず「落ちない」「再現できる」「比較できる」実装

競馬の「レース内での相対順位」予測には **ランキング学習（Learning-to-Rank, LTR）** がよく採用される。  
本書では従来の **点推定（pointwise: 各馬を独立行として logloss 等）** を主軸にしつつ、**LambdaRank 系・最新の LTR・深いアンサンブル・NN** を精度探索の軸として明示する（実装は段階的にブランチを切る）。

## 基本方針

### 1. DataFrame の第一選択は `polars`
- 学習母表の組み立て
- 特徴量結合
- 集約
- fold ごとの OOF 特徴量生成

は原則 `polars` で行う。

### 2. `pandas` は境界面だけで使う
以下のような最小限の場面に限定する。
- LightGBM / CatBoost / XGBoost へ最終入力する直前
- SHAP や一部可視化ライブラリが `pandas` を要求する場面
- 単純な小サイズ CSV 出力

### 3. 入力は原則 `Parquet + scan`
`read_parquet()` ではなく、まず `scan_parquet()` を使う。  
必要列だけ `select()` し、必要行だけ `filter()` してから `collect()` する。

### 4. 中間成果物を積極保存する
OOF や stage 出力を毎回再計算しない。

保存対象:
- `horse_latent_state.parquet`
- `positioning_features.parquet`
- `pace_scenario_features.parquet`
- `day_track_bias_features.parquet`
- `contextual_bloodline_features.parquet`
- `horse_probability_table.parquet`

## 推奨ライブラリ構成

## データ処理
- `polars`
- `pyarrow`
- `duckdb`（必要時のみ）

## 学習
- `lightgbm`（回帰・分類に加え **`lambdarank`**）
- `catboost`（**`YetiRank`** / **`StochasticRank`** 等のランキングモード）
- `xgboost`（**`rank:ndcg`**, **`rank:pairwise`**, **`rank:map`**）
- `scikit-learn`（メタ学習・校正・補助回帰）

## ランキング・深層（任意・精度探索ブランチ）
- `pytorch` または `jax`（**レース単位 listwise**、スコアヘッド＋ softmax、小規模 MLP / 埋め込み）
- 大規模 tabular NN（例: **FT-Transformer** 系）は実装コストとメモリが重いため、**ベースライン確立後**の比較実験として位置づける

## 補助
- `numpy`
- `scipy`

## ライブラリごとの役割

### `polars`
- lazy query
- 列選択
- join
- groupby 集約
- fold ごとの特徴量作成

### `pyarrow`
- schema 管理
- Parquet metadata の取得
- row group / file layout 管理

### `duckdb`
以下のときのみ使う。
- 複数年 Parquet をまたぐ ad-hoc 集計
- SQL 的に書いた方が明らかに読みやすい分析
- 大規模 segment coverage チェック

## ディレクトリ戦略

## ソース
- `data/local/tables/<year>/*.parquet`

## モデリング中間成果物
- `data/modeling/base/`
- `data/modeling/features/`
- `data/modeling/oof/`
- `data/modeling/calibration/`
- `data/modeling/backtest/`

## 実験メタデータ
- `data/meta/modeling/`

## 推奨ファイル分割
- `base_horse_dataset_2020_2025.parquet`
- `baseline_v1.parquet`
- `extended_v1.parquet`
- `oof_fold_1_target_encoding.parquet`
- `oof_fold_1_positioning.parquet`
- `oof_fold_1_final_probabilities.parquet`

## メモリ節約の原則

## 1. 必要列だけ読む
悪い例:

```python
df = pl.read_parquet("race_shutuba_flat.parquet")
```

良い例:

```python
df = (
    pl.scan_parquet("race_shutuba_flat.parquet")
    .select([
        "race_id", "date", "venue_code", "surface", "distance",
        "horse_number", "horse_id", "jockey_id", "trainer_id",
        "weight", "weight_change"
    ])
    .collect()
)
```

## 2. 必要年だけ読む
年をまたぐ時も全件 `collect()` しない。

```python
lf = pl.concat([
    pl.scan_parquet(f"data/local/tables/{y}/race_shutuba_flat.parquet")
    for y in years
])
```

## 3. 必要 fold だけ materialize する
- 全 fold 分の OOF を一度に作らない
- `fold_1` を作って保存
- メモリ解放
- `fold_2` を作る

## 4. 型を落とす
- 確率・連続量: `Float32`
- count・flag: `Int8 / Int16 / UInt16`
- rank: `UInt16`
- カテゴリ ID: `Int32`
- 長い文字列は可能なら辞書化 or id 化

## 5. Python ループを避ける
- `groupby().agg()` を優先
- `with_columns()` で複数列をまとめて作る
- `map_elements()` は最後の手段

## 6. 巨大 join を避ける
母表に全部一気に join しない。

順番:
1. 母表の最小列を作る
2. `race_index` を join
3. 保存
4. `past aggregate` を join
5. 保存

## 7. 生テキストを展開しない
- `comment` や `paddock_comment` を全文保持した巨大 DataFrame を学習用に持たない
- スコア化・presence flag 化してから捨てる

## `polars` 実装パターン

## 1. 母表生成

```python
base = (
    pl.concat([
        pl.scan_parquet(f"data/local/tables/{y}/race_shutuba_flat.parquet")
        for y in years
    ])
    .select([
        "race_id", "date", "venue_code", "venue", "surface", "distance",
        "direction", "grade", "race_class", "field_size",
        "horse_number", "horse_id", "horse_name",
        "jockey_id", "trainer_id", "weight", "weight_change",
        "sire", "dam_sire"
    ])
)
```

## 2. sparse テーブル結合

```python
paddock = (
    pl.concat([
        pl.scan_parquet(f"data/local/tables/{y}/race_paddock_flat.parquet")
        for y in years
    ])
    .select(["race_id", "horse_number", "paddock_rank", "paddock_comment"])
    .with_columns([
        pl.lit(1).alias("has_paddock"),
    ])
)

joined = base.join(
    paddock,
    on=["race_id", "horse_number"],
    how="left",
)
```

## 3. race 内 rank / z-score

```python
df = df.with_columns([
    pl.col("speed_max")
      .rank("ordinal")
      .over("race_id")
      .alias("speed_rank_in_race"),
    (
        (pl.col("speed_max") - pl.col("speed_max").mean().over("race_id")) /
        (pl.col("speed_max").std().over("race_id") + 1e-6)
    ).alias("speed_z_in_race"),
])
```

## OOF 特徴量の生成戦略

## 1. target encoding
方針:
- split manifest に従って fold ごとに train / valid を分ける
- train 側だけでカテゴリ平均を作る
- valid 側へ map
- 結果を `oof_fold_x_*.parquet` に保存

## 2. stage モデル出力
`PositioningModel`, `PaceScenarioModel`, `DayTrackBiasModel`, `ContextualBloodlineFitModel` の出力は、必ず OOF 保存する。

例:
- `data/modeling/oof/fold_1_positioning.parquet`
- `data/modeling/oof/fold_1_pace.parquet`
- `data/modeling/oof/fold_1_final_prob.parquet`

## 3. calibration
- OOF 予測を集めた 1 本の calibration 用テーブルを作る
- calibration 用テーブルだけ `pandas` / `numpy` に落としてもよい

## 学習器ごとの実装戦略

## LightGBM
- 最終入力は `float32` / `int32`
- `categorical_feature` を活かせる列は integer id にして渡す
- ただし OOF target encoding 後の列は数値扱い
- 学習用 matrix は fold 単位で生成し、不要になったら即解放
- **ランキング**: `objective="lambdarank"` に **`label_gain`**（relevance レベルごとの gain）と **`group`**（各レースの行数）を指定。監視に **`eval_at=[3,5]`** 等で NDCG@k を併記すると比較が楽

## CatBoost
- 高カードカテゴリは CatBoost の強み
- 文字列カテゴリをそのまま渡すか、辞書 id を渡す
- ただし巨大 DataFrame を `pandas` に一気に変換しない
- fold 単位で `Pool` を組み立てる
- **ランキング**: `loss_function` に **`YetiRank`** / **`StochasticRank`** / **`QueryCrossEntropy`** 等（公式の ranking 系一覧を参照）、**`group_id=race_id`** でクエリを指定

## XGBoost
- `QuantileDMatrix` を優先
- dense `float64` 行列を避ける
- **ランキング**では `objective=rank:ndcg`（または `rank:pairwise` / `rank:map`）と、クエリごとの行数 **`group`** を必ず指定する

---

## ランキング学習（LTR）と競馬タスクの相性

- **クエリ（query）** = 1 レース `race_id` に属する出走馬の集合（可変長）。
- **ドキュメント** = 各馬の特徴ベクトル（`T-10分` までに観測可能な列のみ）。
- **関連度（relevance）** = 確定着順や複勝圏から定義した非負整数／実数（例: 1着を最高、失格・除外は別ルールで 0 扱い等）。

**Pointwise**（各行に `is_win` を当てる）は実装が単純で馬券用の確率校正とも親和性が高い。  
一方で **Listwise / Pairwise** は「同レース内の順序」を直接損失に入れるため、**上位捕捉（NDCG@3）や順位形状**に強いことが多い。

本プロジェクトでは両方を **同一の purged CV / race 単位 split** の下で比較可能にする（`train_valid_test_split_strategy.md` と整合）。

---

## 古典〜標準: LambdaRank と GBDT ランキング

### LambdaRank / LambdaMART
- **LambdaRank**: 勾配を NDCG の変化に重み付けした近似（ペアを動かしたときの評価指標改善を強調）。
- **LambdaMART**: その勾配で MART（GBDT）を学習する枠組み。
- **LightGBM**: `objective="lambdarank"` と **`label_gain`**（relevance レベルごとの gain 配列）、**`group`** で listwise 学習が可能。
- メモリ: `group` は `numpy` 配列で渡すことが多い → **レース単位で fold materialize** し、学習直前だけ行列化する。

### XGBoost のランキング目的関数
- `rank:ndcg` … NDCG を意識した pairwise 近似（報告しやすい）。
- `rank:pairwise` … シンプルなペアワイズ（高速・ベースライン比較用）。
- `rank:map` … 平均精度（上位重視の別解釈）。

### CatBoost のランキング
- **`YetiRank`**、**`StochasticRank`**、**`QueryCrossEntropy`**、**`LambdaMart`** など（CatBoost 公式 *Ranking: objectives and metrics* 参照）。高カードカテゴリと併用しやすい。
- **`group_id=race_id`** でクエリを指定し、同一レース内でペア／リスト損失を構成させる。

### relevance 設計の例（実装メモ）
- 着順ベース: `relevance = max_rank + 1 - finish_position`（着外は一定の低値にクリップ）。
- 複勝重視: 3着以内を同タイルに寄せる、または top3 にボーナスを付けた多段 relevance。
- **将来情報の混入禁止**: relevance は **確定結果ラベル**のみで定義し、特徴側に結果由来の列を入れない（二重チェック）。

---

## 拡張: 目的関数・学習枠の幅を広げる試み

精度向上の探索として、次を **同一 CV プロトコル**で ablation しやすい順に並べる。

1. **ListNet / ListMLE 型（NN または近似）**  
   レース内スコア `s_i` に対し `P(順序) ∝ softmax` の負の対数尤度（ListMLE）や確率分布 KL（ListNet）。  
   小規模なら **レース単位バッチ**（`batch_size=1 race`、馬数可変）の MLP で試せる。

2. **Pairwise 深層（RankNet 系）**  
   馬ペア `(i,j)` を同一レース内で生成し、`σ(s_i - s_j)` と真の勝敗ラベルの cross-entropy。  
   メモリ注意: ペア数は `O(field^2)` → **上位馬同士・サンプリング**で削る。

3. **Multi-task**  
   同一バックボーンから **pointwise head**（BCE / focal）と **ranking head**（pairwise hinge）を併用し、損失を `L = L_point + λ L_rank` で結合。馬券用確率は pointwise 側を校正しやすい。

4. **Tabular 深層（NN 線を残す）**  
   - **埋め込み**: `horse_id` / `jockey_id` / `trainer_id` を hash embedding（出現頻度で次元調整）。  
   - **FT-Transformer / TabTransformer**: 列をトークン化して self-attention。精度ポテンシャルは高いが **fold ごとの学習コスト**が大きい → `extended_v1` 確定後の「伸ばし」候補。  
   - **蒸留**: 強いランカーまたは大きい NN のスコアを、LightGBM pointwise / lambdarank に **知識蒸留**（教師スコアを追加ターゲットまたはソフトラベル）。

5. **確率とランキングの橋渡し**  
   - ランカー出力スコア → レース内 **softmax で疑似確率** → 温度スケールで校正（valid のみ fit）。  
   - **Plackett–Luce** 等の順序モデルは実装コストが高いが理論的に listwise と整合。まずは softmax 近似で十分なことが多い。

6. **研究寄り・長期オプション**（優先度は低め）  
   - **LambdaLoss** 一般形や、オークション理論に基づく損失などは文献単位で検討。  
   - **Determinantal Point Process (DPP)** で多様性を入れる等は馬券最適化層に近い。

---

## アンサンブルの「層」を深くする（許容設計）

浅い平均ブレンドだけでなく、**層状スタッキング**を許容する。例:

| 層 | 中身 | 備考 |
|----|------|------|
| L1a | LightGBM / CatBoost / XGBoost **pointwise**（`is_win` / `is_top3`） | 現行方針の主戦力 |
| L1b | LightGBM **lambdarank**、CatBoost **YetiRank**、XGBoost **rank:ndcg** | 同一特徴でも順序目的で別視点 |
| L1c | **NN**（listwise または score + softmax、multi-task） | メモリ・再現性のコストが高いので fold 単位 checkpoint |
| L2 | **OOF 上のメタ特徴**（L1 各モデルの予測・順位・レース内 z-score）を入力とした **浅い GBDT / ロジスティック** | valid でのみ学習、test では L1 の OOF 手順を固定 |
| L3 | **ランキング再ブレンド**（Borda / 順位平均 + L2 出力の凸結合）や **セグメント別重み**（芝・ダ・頭数帯） | 過学習に注意し、重みは valid のみ |

実装原則:
- **各層の OOF を Parquet 保存**し、上の層は下の層の保存物だけを読む（再計算禁止に近い運用）。
- **test には一切メタを fit しない**（`advanced_modeling_techniques.md` と同じ）。

---

## LTR 導入時の `polars` / メモリとの両立

- relevance・`group`（各 race の行数）列は **`polars` で母表に付与**してから、学習直前に `numpy` に落とす。
- **可変長クエリ**は、学習 API ごとに `group` 配列または `group_id` 列で表現。巨大なペア展開テーブルはディスクに書かず、**DataLoader 内で on-the-fly 生成**（NN）を推奨。
- **型**: relevance は `Int16` 程度で足りることが多い。`label_gain` は事前に固定長配列として定義しておくと LightGBM 側が単純。

---

## ランキング用の評価指標（ログに必ず残す）

- **NDCG@k**（k=1,3,5 など）、**MAP**、レース内 **Kendall τ**（順序一致度）。
- 馬券・校正との接続のため、**pointwise の LogLoss / Brier / ECE** も併記（LayerA 確率の品質監査）。

## 校正の実装戦略
- `OOF predictions + labels` だけを集めた小さな calibration テーブルを作る
- `isotonic` はメモリより overfit に注意
- 初期版は `temperature scaling` と `Platt` を優先

## stage ごとの実装指針

## Stage0 / Stage1
- `polars` で母表を作る
- まず `BaselineV1` を保存
- 次に `ExtendedV1` を差分生成

## Stage2 / Stage3
- 学習データ構築は `polars`
- モデル学習直前だけ `pandas` / `numpy`
- 出力は fold 単位保存

## Stage4 / Stage5
- 同日先行レース集計は `polars.groupby`
- 同日 posterior はレースごとに更新するが、1 日単位の小さいテーブルで扱う
- 血統ベクトルは事前計算して Parquet 化しておく

## Stage6
- 入力は stage 出力を join した最終特徴量テーブル
- `OOF final probabilities` を正式成果物にする（**pointwise 主系**）
- **（任意）** 同一特徴から **ランキング OOF**（`lambdarank` / `YetiRank` / `rank:ndcg`）を別 Parquet で保存し、メタ層または softmax 疑似確率の比較材料にする

## MLflow による学習記録（必須運用）

確率モデル・ランキング枝・アンサンブル・（将来の）校正前後の比較まで、**学習ジョブごとに MLflow に記録**する。ローカル運用でも **Tracking URI** を固定し、実験名でモデル種別を分ける。

### 最低限ログに残すタグ / パラメータ
- `split_manifest_version`, `data_version`, `encoding_recipe_version`, `feature_list_hash`（または列数）
- `git_commit` / `code_version`（CI ならビルド ID）
- `primary_metric` 名と `study_id`（Optuna 等を接続したとき）
- `model_family`（例: `lgbm_pointwise`, `lgbm_lambdarank`, `ensemble_stack`）

### アーティファクト
- 学習に用いた **特徴量リスト**（テキスト or YAML）
- **採用ハイパラ** JSON、**重要度**（gain）が取れるモデルでは CSV
- 可能なら **OOF 予測のサンプル Parquet**（容量に応じて）

既存コードでは `pipeline/trainer.py` / `pipeline/ensemble_trainer.py` / `pipeline/pace_predictor.py` 等が MLflow 連携を持つ。**新規 runner を追加するときも同パターンに揃える**こと。

## 再現性・比較可能性

## 必ず保存するもの
- 使用列一覧
- dtypes
- split manifest version
- encoding recipe version
- stage model version
- calibration recipe version
- feature count
- row count
- **mlflow_experiment_name** / **mlflow_run_id**（追跡 URL とセット）

## 命名規則
- `baseline_v1_fold1.parquet`
- `extended_v1_fold1.parquet`
- `oof_te_recipe_a_fold1.parquet`
- `final_prob_lgbm_catboost_blend_fold1.parquet`
- `oof_fold1_lgbm_lambdarank_ndcg3.parquet`
- `oof_fold1_catboost_yetirank.parquet`
- `meta_stack_l2_from_l1a_l1b_oof.parquet`
- `nn_listwise_fold1_checkpoint.pt`

## やらないこと
- 全年全列を `pandas` に一括ロード
- `read_parquet()` だけで巨大テーブルを都度 materialize
- 全 fold を同時にメモリ上に保持
- stage 出力を保存せず毎回再計算
- 生テキストや nested list をそのまま最終学習行列へ入れる
- **L2/L3 メタ学習やランキング→確率の温度スケールを test で fit する**（valid または OOF のみ）

## 推奨実装順
1. `polars` ベースの母表 builder
2. fold 単位 OOF runner
3. target encoding recipe runner
4. stage 出力保存 runner
5. final probability runner（**pointwise 主系**で安定版を確保）
6. **（任意）** LTR ブランチ: relevance / `group` 生成 → LightGBM `lambdarank` と CatBoost `YetiRank` の **同一 CV 比較**
7. **（任意）** L2 メタ学習（L1 pointwise + L1 rank の OOF のみ入力）
8. **（任意）** NN listwise / multi-task（`extended_v1` と purged CV が固まってから）
9. calibration runner
10. backtest dataset runner
