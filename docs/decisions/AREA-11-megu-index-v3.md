# AREA-11 — めぐ指数 v3（実測めぐ指数・設計仕様）
**Status**: DECIDED | **Date**: 2026-07-13

---

## 0. 前バージョンとの関係

| バージョン | ファイル | 状態 |
|---|---|---|
| v1 | AREA-11-megu-index.md | **ARCHIVED**（本仕様に置換） |
| v2 par補足 | AREA-11-megu-index-v2-par.md | **ARCHIVED**（本仕様に統合） |
| **v3（本文書）** | AREA-11-megu-index-v3.md | **CURRENT** |

---

## 1. 基本原則

| 項目 | 内容 |
|---|---|
| スケール | **1点 = 0.1秒差** |
| 基準値 | **50 = 1勝クラス2着馬相当のパフォーマンス** |
| すべての補正 | 秒単位で統一し、等価性を保つ |
| 障害レース | 除外（平地レースのみ対象） |

---

## 2. 指数の種類

| 指数名 | 計算タイミング | 内容 | UI表示 |
|---|---|---|---|
| **実測めぐ指数** | レース確定後（事後） | ペース・馬場・斤量・クラスを補正した実走パフォーマンス | ○ |
| **想定めぐ指数** | レース前 | 過去の実測めぐ指数を集計した出走時の予測値 | ○ |

---

## 3. 全体計算フロー

```
raw_time（走破タイム）
    ↓ 斤量補正（NB-01計算済み）
adjusted_time_sec
    ↓ Δpace（ペース補正）
    ↓ Δtrack（馬場差補正）
corrected_time = adjusted_time_sec − Δpace − Δtrack
    ↓
実測めぐ指数 = 50 + (par_time_class − corrected_time) × 10
```

---

## 4. par_time の構築（クラス別線形回帰）

### 4-1. 使用データ

| 項目 | 仕様 |
|---|---|
| 対象馬 | **2着馬のみ** |
| 対象レース | 3歳以上・4歳以上の**混合条件戦のみ**（2歳のみ・3歳のみの世代戦を除外） |
| 使用タイム | `adjusted_time_sec`（斤量補正後タイム） |

### 4-2. クラス階層

| rank | クラス |
|---|---|
| 1 | 未勝利 |
| 2 | 1勝クラス |
| 3 | 2勝クラス |
| 4 | 3勝クラス |
| 5 | OP |
| 6 | 重賞（G2/G3） |
| 7 | G1 |

### 4-3. 回帰モデル

グループ単位：**開催場 × コース(芝/ダ) × 距離**

```
avg_2nd_time(venue, surface, distance, class_rank)
    = α(venue, surface, distance)  +  β(venue, surface, distance) × class_rank
```

- β は負（クラスが上がるほど平均タイムが速くなる）
- 基準点：1勝クラス（rank=2）→ `α + β × 2` = 実測めぐ指数50点に対応

### 4-4. par_time の算出

```python
par_time_class(venue, surface, distance, class_rank) = α + β × class_rank
```

### 4-5. Sparse対策

| 状況 | 対処 |
|---|---|
| セルのクラスデータが2水準未満 | `surface × 距離帯` のプーリングβを使用 |
| それも不足 | Ridge回帰でグローバルβへ縮小 |

---

## 5. Δpace（ペース補正）

| 項目 | 仕様 |
|---|---|
| 目的 | ペース差による最終タイムの変動を除去 |
| 係数推定条件 | **開催場 × コース(芝/ダ) × 距離** |
| 推定方法 | セル内OLS：`front_split_dev`（実スプリット − par_front_split）→ `adjusted_time_sec` |
| 適用式 | `Δpace = coeff_pace(venue, surface, distance) × front_split_dev` |
| Sparse対策 | 30件未満 → `surface × 距離帯` → `surface` 全体の順でフォールバック |
| データ欠損時 | `front_split_sec` が NULL → `Δpace = 0` |

---

## 6. Δtrack（馬場差補正）

| 項目 | 仕様 |
|---|---|
| 目的 | 当日の時計水準の偏りを除去（同一馬場状態区分内での日別ズレ） |
| 係数算出条件 | **開催日 × 開催場 × コース(芝/ダ)** |
| 対象レース | **未勝利〜G3**（G1を除外） |
| 算出式 | `Δtrack = mean(adjusted_time_sec − Δpace) − mean(par_time_class)` |
| 符号の直感 | 速い馬場の日 → Δtrack < 0 → corrected_time から引くと上昇（時計的優位を除去） |
| 計算タイミング | 当日の対象レース全確定後の**事後計算** |
| Sparse対策 | 当日対象レースが3件未満 → `Δtrack = 0` |

---

## 7. データ品質フィルタ

### out_of_range フラグ

| 項目 | 仕様 |
|---|---|
| 条件 | 3着以降 かつ `finish_time_sec > time_2nd + 2.0秒` |
| 処理 | `megu_index = NULL`、`computation_status = 'out_of_range'` |
| 対象外 | 1着・2着は無条件に計算対象 |
| 根拠 | 着差2秒超は競走不成立相当（出遅れ・故障・明らかな競走意欲喪失）であり、補正後タイムの信頼性が著しく低い |

---

## 8. 指数の読み方

| 値 | 意味 |
|---|---|
| **50** | 出走クラスの2着馬相当（= par） |
| 55 | parより0.5秒速い |
| 60 | parより1.0秒速い |
| 45 | parより0.5秒遅い |
| 40 | parより1.0秒遅い |

実走1着馬は概ね50〜55、G1好走馬で60台が想定される範囲。

---

## 9. ノートブック実装計画

| NB | 目的 | 主要入力 | 主要出力 |
|---|---|---|---|
| NB-01 | データ前処理・斤量補正 | race_result_flat, cushion | adjusted_time_sec, par_splits |
| NB-02 | Δpace係数推定 & par_time回帰 | NB-01出力 | coeff_pace, par_time_class |
| NB-03 | Δtrack算出 | NB-02出力 | Δtrack per (date, venue, surface) |
| NB-04 | 実測めぐ指数の計算・検証 | NB-02/03出力 | megu_index 全件 |
| NB-05 | 想定めぐ指数の集計・検証 | NB-04出力 | megu_final |
| NB-06 | 有効性検証 | NB-04/05出力 | 着順相関・回収率レポート |

---

## 10. 未決定事項

| # | 項目 | 備考 |
|---|---|---|
| T-1 | 想定めぐ指数の集計ロジック詳細 | 別途設計 |
| T-2 | Δtrack の馬場状態カテゴリ（良/稍重/重・不良）による分離の要否 | データ密度と要相談 |
