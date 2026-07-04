# 拡張分析 — 新規特徴候補の即席検証

指標: 2024 vs 2025 KS、Spearman(特徴, speed_pct_in_race)、Spearman(特徴, field_speed_std)。

## `field_speed_gap_std`

- レース内 (speed_max-speed_avg) の std — 指数の形のばらつき
- 欠損率 0.00% | KS=0.0625 (p=3.00e-40) | ρ(特徴,speed_pct)=-0.0431 | ρ(特徴,field_std)=0.3246

## `field_std_minus_iqr`

- std - IQR — 外れ値由来の「追加混戦」成分
- 欠損率 0.00% | KS=0.0317 (p=1.08e-10) | ρ(特徴,speed_pct)=0.0143 | ρ(特徴,field_std)=0.2470

## `bracket_trader_interaction`

- bracket_std * (1 - trainer_share) — 枠分散と厩舎集中度の合成
- 欠損率 0.00% | KS=0.0215 (p=3.71e-05) | ρ(特徴,speed_pct)=-0.1557 | ρ(特徴,field_std)=0.2053

## `field_std_surface_z`

- 同一 race_year×surface 内での field_std の z（相対化）
- 欠損率 0.00% | KS=0.0325 (p=3.65e-11) | ρ(特徴,speed_pct)=-0.0568 | ρ(特徴,field_std)=0.9198

## `field_speed_std_per_sqrt_n`

- field_std / sqrt(field_size) — 頭数補正混戦度
- 欠損率 0.00% | KS=0.0644 (p=1.16e-42) | ρ(特徴,speed_pct)=-0.0492 | ρ(特徴,field_std)=0.9911

## `field_weight_cv`

- 斤量の変動係数（レース内）
- 欠損率 0.00% | KS=0.0147 (p=1.27e-02) | ρ(特徴,speed_pct)=-0.0422 | ρ(特徴,field_std)=0.1539

---

## 解釈メモ（自動）

- **field_speed_gap_std**: max/avg ギャップのばらつき。field_std と **ρ≈0.32** なら追加価値は限定的になりやすい。
- **field_std_minus_iqr**: std と IQR の差。**ρ(特徴,field_std)** が中程度なら「外れ値混戦」単独信号の可能性。
- **bracket_trader_interaction**: 枠と厩舎の合成。speed_pct との |ρ| が枠単独より下がれば **交絡低減**の兆候（要数値確認）。
- **field_std_surface_z**: field_std との相関が理論上低くなるよう設計。ρ(特徴,field_std) が低ければ **相対混戦**として別軸。
- **field_speed_std_per_sqrt_n**: 頭数補正。大レースの混戦度を比較可能に。**ρ(特徴,field_std)** が高ければ単調変換に近い。
- **field_weight_cv**: 斤量スケール正規化。**field_weight_std** より年次ドリフトが小さければ採用しやすい。
