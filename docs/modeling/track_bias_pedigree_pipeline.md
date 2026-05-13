# 血統×馬場傾向（track_bias_pedigree）パイプライン

## 目的

当日・前日の馬場傾向（芝/ダ・場・発走順）と、出馬表の父・母父の芝/ダ別累計勝率を組み合わせた **リークなし** 特徴を LayerA に付与する。  
**5世代血統グラフや GCS の `horse_pedigree_5gen` は本パイプラインの必須ではない**（後述）。

## データフロー（GCS 負荷を抑える）

| 段階 | データ源 | アクセス |
|------|----------|----------|
| 特徴計算 | `data/local/tables/<年>/race_result_flat.parquet` | **ローカルのみ** |
| 血統名 | `race_shutuba_flat` の `sire` / `dam_sire` | **ローカルのみ** |
| 母表 | `pipeline/layer_a_dataset.py` → `data/modeling/layer_a_train.parquet` | ローカル |

- **学習・チューニング・評価**はすべて上記 Parquet を読むだけで完結する。推論時も同様に、当日以前の `race_result` と出馬表で同じ式を再現する。
- **5世代血統（HTML/GCS）**を使うのは `research/pedigree_similarity.py` など別系統であり、馬場傾向予測の必須入力ではない。必要な場合は **一度だけバッチ取得 → `data/local/` またはローカルキャッシュに保存し、以降は再フェッチしない** 方針を推奨する。

## 重みチューニング（当日 vs 前日 + 馬場区分 + 距離帯）

加重合成は次の通り（`pipeline/track_bias_pedigree.py`）。

- **当日/前日ブレンド**: `w_same × 当日該当レース前 + w_prev × 前日`（`w_same + w_prev = 1`）。既定は 0.75 / 0.25。
- **馬場適性（3 区分）**: `race_result.track_condition` を **良** → `good`、**稍重・重** → `yielding_soft`、**不良** → `heavy_bad` に割り当て。`track_bias_winner_3f_same_day_prior` / `prev_day` に **区分別の乗算重み**（チューニングで決定、グループ内で平均 1 に正規化）。
- **舞台（競馬場 + 芝/ダ/障 + 距離帯）**: `venue` + `surface`（芝/ダ/`障`→ obstacle）+ `distance` から距離帯（≤1400 sprint、≤1800 mile、≤2200 middle、それ以外 long）を付与。傾向集計は **日×場×芝ダ障×馬場3区分×距離4帯** の細キーで計算し、欠損は **日×場×芝ダ障** の粗キーでフォールバック。`track_bias_*` 列への距離帯別乗算重みもチューニング対象（4 帯で平均 1 に正規化、`unknown` は 1.0）。

チューニング出力 JSON は **`version`: 2** で、`best_same_day_weight` / `best_prev_day_weight` に加え `cond_multipliers` / `dist_multipliers` を含む。v1（当日/前日のみ）も `load_track_bias_weights_config` で読み込める。

1. **馬場区分・距離帯列付き**で LayerA を再ビルドする（古い Parquet には列が無い）。  
   `python3 -m pipeline.build_layer_a_dataset`
2. Optuna で valid の **NDCG@3** を最大化（GCS 不要）。  
   `python3 -m research.tune_track_bias_weights --n-trials 60`
3. 出力 `data/meta/modeling/track_bias_weight_best.json` を母表ビルドに渡す。  
   `python3 -m pipeline.build_layer_a_dataset --track-bias-weights-json data/meta/modeling/track_bias_weight_best.json`
4. 血統×馬場のみの検証（任意）。  
   `python3 -m research.evaluate_track_bias_pedigree_only --weights-json data/meta/modeling/track_bias_weight_best.json`

**test メトリクス**（`metrics_test_at_best`）を JSON に載せたい場合は、マニフェストの test 年（例: 2025）を `--years` に含めて母表をビルドすること。train+valid のみの Parquet では test はスキップされる。

チューニングの目的関数は **valid のみ**とし、test は最終報告用（リーク回避）。

## 手法の伸ばし方（優先度の目安）

1. **重みの最適化**（上記）— 当日/前日・馬場 3 区分・距離 4 帯のデータ駆動調整。
2. **5世代血統**— 父・母父名の一致を超えた類似度が必要な場合のみ、ローカルキャッシュ済みの系譜グラフを別特徴として結合（本モジュールとは独立に管理し、GCS をホットパスに置かない）。

## 関連モジュール

- `pipeline/track_bias_pedigree.py` — 特徴定義・加重ブレンド・`recompute_weighted_interactions`
- `research/tune_track_bias_weights.py` — Optuna
- `research/evaluate_track_bias_pedigree_only.py` — 単独特徴群のベースライン評価
