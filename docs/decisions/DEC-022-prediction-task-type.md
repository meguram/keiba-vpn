# DEC-022: 予測タスク種別（二値分類 + Harville 導出）

**Status**: ACCEPTED
**Date**: 2026-07-06
**Author**: Orchestrator（TASK-056 エスカレーション確認）
**Related**: DEC-005, DEC-015, DEC-016, AREA-06, AREA-07

---

## Context

AREA-07 の単勝予測タスク定義が「回帰問題（連続値スコア）」と「二値分類（binary classification）」の両方で記述されており内部矛盾が発生していた。また連対率・複勝率の確率計算ロジックが未定義だった。

## Decision

**二値分類（Binary Classification）+ softmax 正規化 + Harville 式導出** を採用する。

### ステップ 1: 基礎スコア算出
LightGBM `objective='binary'` で各馬の生スコアを出力する。
ラベル: 1着 = 1、それ以外 = 0

### ステップ 2: 勝率（単勝確率）
レース内の全馬スコアを **softmax 正規化** して勝率 $p_i$ を得る。

$$p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \quad \sum_i p_i = 1$$

### ステップ 3: 連対率（top-2確率）— Harville 式

$$P_i^{(top2)} = p_i + \sum_{j \neq i} p_j \cdot \frac{p_i}{1 - p_j}, \quad \sum_i P_i^{(top2)} = 2$$

### ステップ 4: 複勝率（top-3確率）— Harville 式

$$P_i^{(top3)} = P_i^{(top2)} + \sum_{j \neq i} \sum_{\substack{k \neq i \\ k \neq j}} p_j \cdot \frac{p_k}{1-p_j} \cdot \frac{p_i}{1-p_j-p_k}, \quad \sum_i P_i^{(top3)} = 3$$

### オッズ
最終オッズは発走直前まで確定しないため、予測時点の暫定オッズを特徴量として使用する。

## Consequences

- AREA-07 の「回帰問題」記述を削除し、本決定に統一する
- AREA-06 の `tansho_label` を `BOOLEAN`（1着/否）として維持
- AREA-02 の `prediction_score` フィールドは `win_probability: float (0.0〜1.0)` として定義
- DEC-016 の「AI上位3頭ハイライト」は `place_probability`（複勝率）の降順で選定する
- 18頭立てでの Harville 計算量は現実的（$O(n^2)$ 〜 $O(n^3)$）

---
*Last updated: 2026-07-06*
