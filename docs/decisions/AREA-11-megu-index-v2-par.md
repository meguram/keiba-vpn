# AREA-11 補足 — 実測めぐ指数 v2（クラス別 par_time）

**Status**: DECIDED | **Date**: 2026-07-11

## 背景

v1 の `par_time` は距離×芝ダ×方向×馬場の **全クラス混在** 2着平均のため、未勝利戦でも field 平均めぐが 100 超えになるなど、レースクラスと実測指数が乖離していた。

## 戦略会議の結論

| 項目 | v1 | **v2（採用）** |
|------|-----|----------------|
| par_time 軸 | 距離×芝ダ×方向×馬場 | **+ class_bucket** |
| 基準走 | 2着補正済みタイム平均 | 同左（2着なしは1着代用） |
| 補正（par 学習時） | Δpace + Δtrack + Δweight | 同左（**Δlevel 除外**） |
| 斤量基準 | 牡55 / 牝53 × dist/2000 | 維持 |
| クラス差の吸収 | FQ（β₄=0 で未稼働） | **par_time の class 分割**（β₄ は 0 維持） |
| 競馬場（venue） | なし | **v2 では見送り**（サンプル不足。v3 候補） |
| model_version | v1 | **v2** |

### class_bucket（6段階）

| bucket | 含むグレード |
|--------|-------------|
| 未勝利 | 新馬・未勝利 |
| 1勝 | 1勝 |
| 2勝 | 2勝 |
| 3勝 | 3勝 |
| OP | L・OP |
| 重賞 | G2・G3 |
| G1 | G1 |

### par_time フォールバック（サンプル不足時）

1. `distance × surface × direction × track_cat × class_bucket`
2. `distance × surface × track_cat × class_bucket`（方向プール）
3. `distance × surface × direction × track_cat`（クラスプール `class_bucket=''`）
4. `distance × surface × track_cat`（方向・クラスプール）

最低サンプル数: **20** レース（2着基準）

## 実装

- `src/pipeline/megu_index/class_bucket.py` — クラスバケット付与
- `src/pipeline/megu_index/build_par_time.py` — par_time v2 推定・DB 投入
- `src/pipeline/megu_index/par_time_resolve.py` — 計算時の階層マージ
- `megu_par_time.class_bucket` 列（migration 006）

## 再計算手順

```bash
KEIBA_ENV=stg python -m src.pipeline.megu_index.build_par_time --model-version v2
KEIBA_ENV=stg python -m src.pipeline.megu_index.compute --year 2024 --model-version v2
# 対象年を繰り返し
```

## 想定めぐ指数への影響

実測 `megu_index` が v2 で再計算されるため、想定指数（過去走平均）も連動して更新される。

### 想定めぐ (megu_final) v2.1（2026-07）

| 項目 | 内容 |
|------|------|
| base | 過去走 megu → 各走 par で補正済みタイムに戻し加重平均 → **今回 par に換算** |
| 条件転換 | 芝↔ダート・距離±600m 超で `megu_condition_transfer` を加算（final に反映） |
| 斤量 | 今回 `jockey_weight` の `weight_megu_delta` を **final に反映** |
| 実測との比較 | ともにペース・馬場・斤量・レベル補正後。1点=0.1秒で着差想定可能 |

### 学習済みチューニング（`config/megu_predict_params.json`）

2024–2025 v2 実測との MAE 最小化（**1点=0.1秒は不変**。調整は秒空間のみ）。

```bash
KEIBA_ENV=stg python3 -m src.pipeline.megu_index.optimize_predict --year-start 2024 --year-end 2025
```

| パラメータ | 最適値（例） | 意味 |
|-----------|-------------|------|
| `par_blend` | 0.0 | 過去 megu 加重平均（par 換算ブレンドなし） |
| `ability_bias_sec` | 1.0 | 能力推定に +1.0秒補正（指数 +10点） |
| `transfer_strength` | 0.0 | 条件転換 Δ をオフ（MAE 悪化のため） |
| `history_weights` | 0.50/0.25/0.15/0.07/0.03 | 直近走重視 |
| full MAE | **17.5** | ベースライン 22.5 から改善（67k ペア） |
