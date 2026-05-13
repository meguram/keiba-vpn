#!/usr/bin/env python3
"""
10 サイクル後の拡張分析: 新規特徴候補の即席検証（KS / Spearman / 冗長度）。
出力: _run_output/extended_analysis.md, extended_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import os

os.chdir(REPO)
from pipeline.feature_store import FeatureStore  # noqa: E402

OUT = Path(__file__).resolve().parent / "_run_output"


def ks_24_25(s: pd.Series, y: pd.Series) -> tuple[float, float]:
    a = s[y == "2024"].dropna()
    b = s[y == "2025"].dropna()
    if len(a) < 50 or len(b) < 50:
        return float("nan"), float("nan")
    r = stats.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def sp(x: pd.Series, y: pd.Series, n: int = 40000) -> float:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) > n:
        d = d.sample(n, random_state=0)
    return float(d["x"].corr(d["y"], method="spearman")) if len(d) > 50 else float("nan")


def main() -> None:
    store = FeatureStore(base_dir=REPO)
    years = store.available_years()
    ri = store.load_source(
        "race_index",
        years=years,
        columns=[
            "race_id",
            "horse_number",
            "speed_max",
            "speed_avg",
            "distance",
            "field_size",
            "surface",
        ],
    )
    for c in ("speed_max", "speed_avg", "distance", "field_size"):
        if c in ri.columns:
            ri[c] = pd.to_numeric(ri[c], errors="coerce")
    rs = store.load_source(
        "race_shutuba",
        years=years,
        columns=["race_id", "horse_number", "weight", "bracket_number", "trainer_id"],
    )
    for c in ("weight", "bracket_number"):
        if c in rs.columns:
            rs[c] = pd.to_numeric(rs[c], errors="coerce")

    m = ri.merge(rs, on=["race_id", "horse_number"], how="inner")
    m["race_year"] = m["race_id"].astype(str).str[:4]
    m["speed_pct_in_race"] = m.groupby("race_id")["speed_max"].rank(pct=True)

    # ベース特徴（レース単位）
    g = m.groupby("race_id", sort=False)
    fs_std = g["speed_max"].transform("std")
    fs_iqr = g["speed_max"].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
    m["field_speed_std"] = fs_std
    m["field_speed_iqr"] = fs_iqr

    def max_trainer_share(s: pd.Series) -> float:
        vc = s.value_counts(normalize=True)
        return float(vc.iloc[0]) if len(vc) else np.nan

    tr = m.groupby("race_id")["trainer_id"].apply(max_trainer_share)
    m["trainer_top_share"] = m["race_id"].map(tr)
    m["field_bracket_std"] = g["bracket_number"].transform("std")
    m["field_weight_std"] = g["weight"].transform("std")
    wmean = g["weight"].transform("mean")
    m["field_weight_cv"] = m["field_weight_std"] / (wmean.abs() + 1e-6)

    # 新規アイデア A: max-avg のレース内ばらつき（指数の「形」の差）
    m["speed_gap"] = m["speed_max"] - m["speed_avg"]
    m["field_speed_gap_std"] = g["speed_gap"].transform("std")

    # 新規アイデア B: std と IQR の差分（外れ値に敏感な成分の残差）
    m["field_std_minus_iqr"] = fs_std - fs_iqr

    # 新規アイデア C: 枠×厩舎の交互作用（強い負相関の両方を 1 本に）
    m["bracket_trader_interaction"] = m["field_bracket_std"] * (1.0 - m["trainer_top_share"])

    # 新規アイデア D: surface 内で field_std を年別 z 化（ドリフト吸収の素案）
    m["_surf"] = m["surface"].fillna("unknown").astype(str)
    m["field_std_surface_z"] = (
        m.groupby(["race_year", "_surf"])["field_speed_std"]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    )

    # 新規アイデア E: field_size で割った混戦度（頭数補正）
    fsz = g["field_size"].transform("first")
    m["field_speed_std_per_sqrt_n"] = fs_std / (np.sqrt(fsz.clip(lower=2)) + 1e-9)

    candidates = {
        "field_speed_gap_std": "レース内 (speed_max-speed_avg) の std — 指数の形のばらつき",
        "field_std_minus_iqr": "std - IQR — 外れ値由来の「追加混戦」成分",
        "bracket_trader_interaction": "bracket_std * (1 - trainer_share) — 枠分散と厩舎集中度の合成",
        "field_std_surface_z": "同一 race_year×surface 内での field_std の z（相対化）",
        "field_speed_std_per_sqrt_n": "field_std / sqrt(field_size) — 頭数補正混戦度",
        "field_weight_cv": "斤量の変動係数（レース内）",
    }

    rows = []
    lines = [
        "# 拡張分析 — 新規特徴候補の即席検証",
        "",
        "指標: 2024 vs 2025 KS、Spearman(特徴, speed_pct_in_race)、Spearman(特徴, field_speed_std)。",
        "",
    ]

    for col, desc in candidates.items():
        if col not in m.columns:
            continue
        s = m[col]
        ks_s, ks_p = ks_24_25(s, m["race_year"])
        r_pct = sp(s, m["speed_pct_in_race"])
        r_fs = sp(s, m["field_speed_std"])
        rows.append(
            {
                "feature": col,
                "description_ja": desc,
                "null_pct": float(s.isna().mean() * 100),
                "ks_24_25": ks_s,
                "ks_p": ks_p,
                "spearman_vs_speed_pct": r_pct,
                "spearman_vs_field_std": r_fs,
            }
        )
        lines.append(f"## `{col}`")
        lines.append("")
        lines.append(f"- {desc}")
        lines.append(
            f"- 欠損率 {s.isna().mean()*100:.2f}% | KS={ks_s:.4f} (p={ks_p:.2e}) | ρ(特徴,speed_pct)={r_pct:.4f} | ρ(特徴,field_std)={r_fs:.4f}"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 解釈メモ（自動）")
    lines.append("")
    # 簡易ルールでコメント生成
    for row in rows:
        col = row["feature"]
        rfs = row["spearman_vs_field_std"]
        rpv = row["spearman_vs_speed_pct"]
        if col == "field_std_surface_z":
            lines.append(
                f"- **{col}**: field_std との相関が理論上低くなるよう設計。ρ(特徴,field_std) が低ければ **相対混戦**として別軸。"
            )
        elif col == "field_std_minus_iqr":
            lines.append(
                f"- **{col}**: std と IQR の差。**ρ(特徴,field_std)** が中程度なら「外れ値混戦」単独信号の可能性。"
            )
        elif col == "bracket_trader_interaction":
            lines.append(
                f"- **{col}**: 枠と厩舎の合成。speed_pct との |ρ| が枠単独より下がれば **交絡低減**の兆候（要数値確認）。"
            )
        elif col == "field_speed_gap_std":
            lines.append(
                f"- **{col}**: max/avg ギャップのばらつき。field_std と **ρ≈{rfs:.2f}** なら追加価値は限定的になりやすい。"
            )
        elif col == "field_speed_std_per_sqrt_n":
            lines.append(
                f"- **{col}**: 頭数補正。大レースの混戦度を比較可能に。**ρ(特徴,field_std)** が高ければ単調変換に近い。"
            )
        elif col == "field_weight_cv":
            lines.append(
                f"- **{col}**: 斤量スケール正規化。**field_weight_std** より年次ドリフトが小さければ採用しやすい。"
            )
    lines.append("")

    (OUT / "extended_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "extended_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote", OUT / "extended_analysis.md")
    for row in rows:
        print(
            f"{row['feature']}: KS={row['ks_24_25']:.4f} sp_pct={row['spearman_vs_speed_pct']:.4f} sp_fs={row['spearman_vs_field_std']:.4f}"
        )


if __name__ == "__main__":
    main()
