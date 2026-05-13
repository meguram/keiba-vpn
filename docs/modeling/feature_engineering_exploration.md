# 特徴量エンジニアリング — 探索メモ（新要素の発見）

## 目的

既存の `BaselineV1` / `ExtendedV1`・`horse_pre_race_dataset_spec.md`・`pre_race_feature_whitelist.md` を前提に、  
**追加で検証に値する特徴量候補**を列挙する。ここに書いた項目はすべて **LayerA（T-10 時点で観測可能）** を優先し、  
**結果ラベル由来のリーク**にならないものだけを対象とする（オッズ由来は LayerB として別枠）。

実装時は `src/pipeline/feature_store.py` の列登録・`data/features/_registry.json` 方針に従い、  
**OOF + purged CV** でのみ寄与を測る。

HTML 要約（ナビ用）: `docs/html/feature_engineering/index.html`

---

## 1. レース内「圧力場」の量化（出走メンバー依存）

| 候補 | 定義イメージ | データソース例 |
|------|--------------|------------------|
| `field_speed_spread` | 同一レース内の `speed_max` / 指数の **標準偏差・IQR** | `race_index_flat`, shutuba past |
| `field_form_dispersion` | 近走好走率や着順の **分散**（メンバーが揃いか荒れか） | past flat |
| `favorite_clustering` | 人気が集中しているか（分散が小さい＝堅い）※LayerB なら T-10 オッズ | LayerB のみ |
| `ability_entropy` | レース内の能力指標を rank 化し **エントロピー**（混戦度） | 指数・past 集約 |

**仮説**: 混戦レースでは絶対スコアより **相対順位特徴** の効きが変わる → `race_rank` / `z-score` との **交互作用** を試す。

---

## 2. 時系列・リズム（馬・厩舎・騎手）

| 候補 | 定義イメージ |
|------|----------------|
| `days_since_last_same_surface` | 前走同馬場からの日数（芝/ダ別） |
| `rotation_pressure` | 短期間での連闘回数（開催単位でのカウント） |
| `trainer_recent_volatility` | 厩舎の直近 N 走の着順分散（同一 trainer_id） |
| `jockey_venue_momentum` | 騎手×場の直近 K 走の勝率変化率（OOF TE と別軸の raw 統計） |

**仮説**: 「仕上げ周期」と「陣営の波」は **距離帯×季節** と交差しやすい。

---

## 3. コース・馬場文脈（venue_era 以外の次元）

| 候補 | 定義イメージ |
|------|----------------|
| `rail_side_bucket` | 直線の内外 / 向きと枠の交差（取得できる場合のみ） |
| `weather_transition` | 前日〜当日の天候変化カテゴリ（同開催内） |
| `cushion_z_in_meet` | 同一開催内でクッション値の **Z 化**（Extended） |
| `heavy_track_sire_rate` | 父系統の重・不良別成績（血統 flat から集約） |

**仮説**: `venue_era` 改修フラグと **馬場極端日** の交差が重賞で効く。

---

## 4. ペース・位置取りの「事前合成」特徴

| 候補 | 定義イメージ |
|------|----------------|
| `expected_pace_pressure` | 全馬の逃げ・先行指標の和／最大（レース全体の前圧） |
| `draw_bias_x_running_style` | 枠 × 脚質の one-hot 交差（過去統計で埋める） |
| `late_charge_density` | 差し・追込馬の頭数と距離バンドの組 |

**仮説**: Stage2/3 のモデル出力がなくても、**ルールベース合成**で一部を近似できる（後段モデルへの橋渡し）。

---

## 5. レースパフォーマンス信号の「文脈付き」取り込み

`race_performance` 系は **別モジュール**だが、LayerA に載せるなら **当日以前の履歴のみ**。

| 候補 | 定義イメージ |
|------|----------------|
| `rp_final_ma3_same_distance_band` | 同馬の `run_performance_final` の移動平均（過去のみ） |
| `rp_pct_vs_field_hist` | 過去レースでの `run_performance_final_pct` の平均 |
| `race_level_pre_race_ma_field` | 当該馬が過去に出たレースの `race_level_pre_race` 平均（対戦強度履歴） |

**仮説**: 単発のパフォーマンス値より **帯・コースとの組み合わせ** で安定する。

---

## 6. 欠損・カバレッジを特徴にする（メタ特徴）

| 候補 | 定義イメージ |
|------|----------------|
| `n_past_races_used` | past から参照できた本数 |
| `has_speed_index_any` | 指数欠損の反転フラグ |
| `feature_coverage_score` | 既に betting 仕様にあるが **LayerA でも** 学習に入れ欠損パターンを学習させる |

**仮説**: 2020 前半など **欠損が構造的**な年では、メタ特徴がドリフト検知に寄与する。

---

## 7. 検証プロトコル（新要素を「発見」したと言える条件）

1. **同一 `split_manifest_version`** で Baseline と **+1 特徴群**のみ差分。  
2. **primary metric**（例: valid logloss）が統計的に改善、または **セグメントで一貫して**改善。  
3. 改善があれば `feature_dictionary.md` と `_registry.json` に登録し、`mlflow_run_id` を紐付ける。  
4. 効かない候補は本書に **却下理由** を 1 行追記（負の知見の蓄積）。

---

## 8. 次アクション（優先度）

1. **field 内分散系**（セクション1）— 実装コスト低・解釈容易。  
2. **メタカバレッジ**（セクション6）— データ品質年との交互作用確認。  
3. **race_performance 履歴**（セクション5）— 既存パイプラインとの join 設計が必要。

---

## 9. 実測メモ（自動探索・2026-04）

実行スクリプト: `notebooks/feature_engineering/run_10_fe_cycles.py`（10 サイクル）、`run_extended_fe_analysis.py`（拡張）。  
成果物: `notebooks/feature_engineering/_run_output/cycles_report.md` / `extended_analysis.md`。

### 10 サイクル（要約）

- **混戦度系**（`field_speed_std` / `field_speed_iqr` / CV クリップ後）は相互に **Spearman ρ≈0.91–0.98**（レース単位）。**本番は 1 本＋ablation** が現実的。`speed_max/distance` のレース内 std は同一レースで距離一定のため **ほぼ `field_speed_std` と同一情報**。
- **`ability_entropy`（粗いビン）**は欠損・飽和があり、現定義では判別力が弱い → **ビン数・最小頭数の再設計**が必要。
- `field_weight_std` は指数系と **低冗長**。`trainer_top_share` は `speed_pct` と弱い正の相関があり **交絡の疑い**。
- `field_bracket_std` と `trainer_top_share` は **強い負相関** → 主効果を両方入れず **交互作用または合成 1 本**を推奨。

### 拡張分析で出た新規候補（数値はローカル flat Parquet 時点）

| 候補 | 解釈 |
|------|------|
| `field_std_minus_iqr` | std−IQR。**ρ(特徴, field_std)≈0.25** で「外れ値混戦」成分として残りやすい。 |
| `bracket_trader_interaction` = `bracket_std * (1 - trainer_share)` | 枠×厩舎の合成。**ρ(特徴, field_std)≈0.21**。 |
| `field_weight_cv` | 斤量のレース内変動係数。**ρ(特徴, field_std)≈0.15** で指数系と最も直交に近い。 |
| `field_speed_gap_std` | `speed_max−speed_avg` のレース内 std。**ρ(特徴, field_std)≈0.32** — 付加価値は限定的になりやすい。 |
| `field_std_surface_z` | 年×surface 内 z 化。**ρ(特徴, field_std)≈0.92** — 依然として混戦度と強く重複。 |
| `field_speed_std_per_sqrt_n` | 頭数補正。**ρ** ほぼ **1** — 実質 `field_speed_std` の単調変換。 |

---

## 参照

- `horse_pre_race_dataset_spec.md`
- `pre_race_feature_whitelist.md`
- `advanced_modeling_techniques.md`
- `race_performance_rating_system.md`
- `modeling_coding_strategy.md`
