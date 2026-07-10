# AREA-08 — エンティティ統計特徴量管理
**Status**: ACTIVE | **Last Updated**: 2026-07-10

---

## 1. 概要

本ドキュメントは、競馬予測モデルで使用する **騎手・調教師・種牡馬・母父** の
パフォーマンス統計特徴量の定義・計算方法・パイプライン設計・本番運用を定義する。

### 設計の3原則

1. **時系列リーク防止**: 各レース行に付与する統計は「そのレースより前」の成績のみ使用
2. **スパース対策 (Bayesian Shrinkage)**: 少サンプルの条件別統計は上位粒度に引き寄せ
3. **本番再現性**: 学習時と同一のロジックで推論スナップショットを生成できる

---

## 2. 特徴量グループ一覧

### グループ A/B: 騎手・調教師 (既存 `jt_race_features`)

ファイル: `data/local/features/race_horse_tbl/<YYYY>/jt_race_features.parquet`  
生成コマンド: `python -m src.pipeline.build_jockey_trainer_stats`

| カラムパターン | 内容 |
|---|---|
| `jk_prior_all_*` / `tr_prior_all_*` | 通算成績（直前累計）|
| `jk_roll5/10/30_*` / `tr_roll5/10/30_*` | 直近 5/10/30 走 |
| `jk_cal90/365_*` / `tr_cal90/365_*` | カレンダー 90日/1年 |
| `jk_at_venue_*` / `tr_at_venue_*` | 競馬場別直前累計 |
| `jk_at_surface_*` / `tr_at_surface_*` | 馬場別直前累計 |
| `jk_at_dist_*` / `tr_at_dist_*` | 距離帯別直前累計 |
| `jk_at_grade_*` / `tr_at_grade_*` | グレード別直前累計 |
| `jk_at_season_*` / `tr_at_season_*` | 季節別直前累計 |
| `jk_at_track_cond_*` / `tr_at_track_cond_*` | 馬場状態別直前累計 |
| `jk_at_weight_bin_*` / `tr_at_weight_bin_*` | 斤量帯別直前累計 |
| `jk_same_day_*` / `tr_same_day_*` | 当日直前（同日複数レース） |
| `jk_prior_all_avg_pass_*` / `tr_prior_all_avg_pass_*` | 位置取り平均（先頭・区間・正規化） |

計 174 列。各統計には `_starts`, `_wins`, `_top3`, `_win_rate`, `_top3_rate`, `_avg_finish` を含む。

### グループ C/D: 騎手・調教師 7 日間ローリング (NEW)

モジュール: `src/pipeline/features/stallion_target_encoder.py` > `build_jt_extended_stats()`

| カラム | 内容 |
|---|---|
| `jk_last7d_starts` | 直近 7 日間の出走数 |
| `jk_last7d_wins` | 直近 7 日間の勝利数 |
| `jk_last7d_top3` | 直近 7 日間の複勝圏内数 |
| `tr_last7d_*` | 調教師版（同上） |

**用途**: 週末の乗り替わり直後の好調・不調を捉える。`jk_cal90` より細粒度な短期フォーム指標。

### グループ E/F: 騎手・調教師 venue×surface Bayesian 勝率 (NEW)

モジュール: `src/pipeline/features/stallion_target_encoder.py` > `build_jt_extended_stats()`

| カラム | 内容 | スパース対策 |
|---|---|---|
| `jk_venue_sf_win_rate_bayes` | 競馬場×馬場別勝率 | Bayesian C=50 (surface 勝率ベース) |
| `jk_venue_sf_starts` | 競馬場×馬場の累計出走数 | — |
| `tr_venue_sf_win_rate_bayes` | 調教師版 | Bayesian C=50 |
| `tr_venue_sf_starts` | 調教師版出走数 | — |

**用途**: 既存の `jk_at_venue_*` + `jk_at_surface_*` を組み合わせたインタラクション特徴量。
例: 「Aという騎手は東京芝が特に得意」という情報を単一特徴量で捉える。

### グループ G/H: 種牡馬・母父統計 (NEW)

モジュール: `src/pipeline/features/stallion_target_encoder.py` > `build_all_sire_stats()`

| カラム | 内容 | C (平滑化) |
|---|---|---|
| `sire_prior_starts` | 産駒直前累計出走数 | — |
| `sire_prior_win_rate` | 全体勝率 (Bayesian) | 30 |
| `sire_prior_top3_rate` | 全体複勝率 | 30 |
| `sire_prior_avg_finish_norm` | 平均着順 / 頭数 (0=1着, 1=最下位) | 30 |
| `sire_prior_avg_pass` | 平均先頭コーナー通過順 (走法傾向) | — |
| `sire_prior_win_rate_surface` | 馬場別勝率 | 20 |
| `sire_prior_top3_rate_surface` | 馬場別複勝率 | 20 |
| `sire_prior_avg_finish_norm_surface` | 馬場別平均着順正規化 | 20 |
| `sire_prior_avg_pass_surface` | 馬場別走法傾向 | — |
| `sire_prior_win_rate_dist` | 距離帯別勝率 | 25 |
| `sire_prior_top3_rate_dist` | 距離帯別複勝率 | 25 |
| `sire_prior_win_rate_venue_sf` | 競馬場×馬場別勝率（高 Bayesian） | 50 |
| `sire_prior_top3_rate_venue_sf` | 競馬場×馬場別複勝率 | 50 |
| `sire_prior_win_rate_grade` | グレード別勝率 | 30 |
| `sire_prior_top3_rate_grade` | グレード別複勝率 | 30 |
| `dam_sire_*` | 母父版（全て同上） | 同上 |

計 30 列（sire 15 + dam_sire 15）。

---

## 3. Bayesian Shrinkage 式

```
encoded = (C × μ_prior + Σ_local) / (C + n_local)
```

- `C`: 平滑化定数（表中の値）
- `μ_prior`: 上位粒度の推定値（階層フォールバック）
- `n_local`: そのストラタのサンプル数

### 階層フォールバック

```
venue × surface × entity  (C=50)
       ↓ n_local が小さいとき
surface × entity           (C=20)
       ↓
entity × global            (C=30)
       ↓
データ全体の global mean
```

### サンプル数ゼロの扱い

`starts=0` の行（初産駒 or 新種牡馬）は `rate` 系を **NaN** にする（global_mean を入れない）。
モデルには `is_new_sire = (sire_prior_starts == 0)` フラグを別途付与して
「新種牡馬であること」自体を特徴量として扱う設計を推奨。

---

## 4. リーク防止実装

### cumsum - self パターン

```python
g = df.sort_values(["entity_id", "race_id"]).groupby("entity_id")
cs_win = g["win"].cumsum()
prior_wins = cs_win - df["win"]  # 自身を除く直前累計
```

- `race_id = YYYYMMDDVVRR` なのでアルファベット昇順 = 日付昇順
- 同日複数レースは `race_id` の小さい（= 発走の早い）ものが先になる
- → 同日同時刻の参照はされない（厳密には会場コードが小さい方が先だが、同会場の複数レースは race_number で分離）

### 7 日間ローリング (O(n²) アルゴリズム)

各騎手グループ内でループし、各行 `i` に対して `i-7days <= date < date[i]` の範囲を集計。
計算量が問題になる場合は年単位でチャンク分割して実行。

---

## 5. ファイル構成

```
data/local/
├── features/
│   ├── race_horse_tbl/<YYYY>/jt_race_features.parquet  (グループ A/B)
│   └── entity_stats_snapshot.parquet                   (本番推論用スナップショット)
└── modeling/
    └── entity_stats/
        ├── entity_stats_full.parquet     (全期間)
        ├── entity_stats_train.parquet
        ├── entity_stats_valid.parquet
        ├── entity_stats_test.parquet
        └── entity_stats.meta.json
```

---

## 6. 生成コマンド

### 学習用（全年）

```bash
# Step 1: 既存 jt_race_features を再生成（グループ A/B）
python -m src.pipeline.build_jockey_trainer_stats

# Step 2: 種牡馬統計 + 拡張統計 + マスターデータセット結合（グループ C〜H）
jupyter nbconvert --to notebook --execute notebooks/modeling/nb-02-entity-stats.ipynb
```

### 本番推論用スナップショット更新

```bash
# scripts/cron/update_entity_stats_snapshot.sh
python -m src.scripts.data.update_entity_stats_snapshot
```

スクリプトは毎朝 6:00 に cron で実行し、`entity_stats_snapshot.parquet` を更新する。
推論 API 起動時にこのファイルをメモリにロードして使用する。

---

## 7. 本番推論での使用方法

```python
from src.pipeline.features.stallion_target_encoder import (
    build_all_sire_stats,
    build_jt_extended_stats,
)

# 予測対象レース当日以前の全レース結果
rr_history = load_race_results(until_date="2025-07-06")
rr_with_sire = rr_history.merge(shutuba_history[["race_id","horse_number","sire","dam_sire"]],
                                  on=["race_id","horse_number"], how="left")

# スナップショット生成
sire_snap = build_all_sire_stats(rr_with_sire)
jt_ext    = build_jt_extended_stats(rr_with_sire)

# 当日出馬表に結合
prediction_input = shutuba_today.merge(sire_snap, on=["race_id","horse_number"], how="left")
prediction_input = prediction_input.merge(jt_ext,   on=["race_id","horse_number"], how="left")
```

---

## 8. 関連ドキュメント

- `AREA-07-modeling.md` §4 — 特徴量ホワイトリスト
- `src/pipeline/features/jockey_trainer_stats.py` — グループ A/B 実装
- `src/pipeline/features/stallion_target_encoder.py` — グループ C〜H 実装
- `notebooks/modeling/nb-02-entity-stats.ipynb` — 実装デモ・動作確認
