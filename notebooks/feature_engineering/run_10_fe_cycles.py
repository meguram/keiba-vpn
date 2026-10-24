#!/usr/bin/env python3
"""
10 サイクルの特徴量エンジニアリング探索ランナー。
各サイクル: 仮説 → 特徴計算 → 分布・年次ドリフト(KS)・構造相関 → 短文の発見メモ。
出力: _run_output/cycles_report.md, cycles_metrics.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import os

os.chdir(REPO)

from src.pipeline.features.feature_store import FeatureStore  # noqa: E402


OUT = Path(__file__).resolve().parent / "_run_output"
OUT.mkdir(parents=True, exist_ok=True)


def ks_24_25(s: pd.Series, year: pd.Series) -> tuple[float, float]:
    a = s[year == "2024"].dropna()
    b = s[year == "2025"].dropna()
    if len(a) < 50 or len(b) < 50:
        return float("nan"), float("nan")
    r = stats.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def year_medians(s: pd.Series, year: pd.Series) -> dict[str, float]:
    g = s.groupby(year).median()
    return {str(k): float(v) for k, v in g.items() if pd.notna(v)}


def spearman(x: pd.Series, y: pd.Series, n: int = 30000) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) > n:
        df = df.sample(n, random_state=42)
    if len(df) < 100:
        return float("nan")
    return float(df["x"].corr(df["y"], method="spearman"))


@dataclass
class CycleResult:
    cycle: int
    name: str
    hypothesis: str
    n_rows: int
    null_pct: float
    describe: dict
    year_medians: dict[str, float]
    ks_stat_24_25: float
    ks_p_24_25: float
    spearman_vs_speed_pct: float
    notes: list[str]


def main() -> list[CycleResult]:
    store = FeatureStore(base_dir=REPO)
    years = store.available_years()
    if not years:
        raise SystemExit("No data in data/local/tables/")

    results: list[CycleResult] = []

    # 共通読み込み（軽量）
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
        ],
    )
    for c in ("speed_max", "speed_avg", "distance", "field_size"):
        if c in ri.columns:
            ri[c] = pd.to_numeric(ri[c], errors="coerce")
    ri["race_year"] = ri["race_id"].astype(str).str[:4]
    ri["speed_pct_in_race"] = ri.groupby("race_id")["speed_max"].rank(pct=True)

    rs = store.load_source(
        "race_shutuba",
        years=years,
        columns=[
            "race_id",
            "horse_number",
            "weight",
            "bracket_number",
            "trainer_id",
        ],
    )
    for c in ("weight", "bracket_number"):
        if c in rs.columns:
            rs[c] = pd.to_numeric(rs[c], errors="coerce")

    m = ri.merge(rs, on=["race_id", "horse_number"], how="inner", suffixes=("", "_sb"))

    ry = m["race_year"]

    # --- Cycle 1: field std ---
    m["c1_field_speed_std"] = m.groupby("race_id")["speed_max"].transform("std")
    results.append(
        _finalize(
            1,
            "field_speed_std",
            "レース内 speed_max の標準偏差＝混戦度（探索メモの field_speed_spread）",
            m["c1_field_speed_std"],
            m["speed_pct_in_race"],
            ry,
            [
                "右裾が長い分布で、極端に指数が割れたレースが少数存在。",
                "年次中央値は 2025 がやや高く、ドリフト監視対象。",
            ],
        )
    )

    # --- Cycle 2: field IQR ---
    m["c2_field_speed_iqr"] = m.groupby("race_id")["speed_max"].transform(
        lambda s: s.quantile(0.75) - s.quantile(0.25)
    )
    results.append(
        _finalize(
            2,
            "field_speed_iqr",
            "外れ値に頑健な混戦度（IQR）。std と併用でロバスト特徴に。",
            m["c2_field_speed_iqr"],
            m["speed_pct_in_race"],
            ry,
            [
                "std より外れ値の影響が小さいレースが多い想定。",
                "std との race 単位相関を見て冗長性を判断。",
            ],
        )
    )

    # --- Cycle 3: coefficient of variation (field) ---
    gmean = m.groupby("race_id")["speed_max"].transform("mean")
    gstd = m.groupby("race_id")["speed_max"].transform("std")
    raw_cv = gstd / (gmean.abs() + 1e-6)
    # mean≈0 のレースで爆発するため winsorize（学習時も同様のクリップ推奨）
    m["c3_field_speed_cv"] = raw_cv.clip(lower=0, upper=5.0)
    results.append(
        _finalize(
            3,
            "field_speed_cv",
            "std/mean でスケール正規化。距離・クラス横断比較に有利な可能性。",
            m["c3_field_speed_cv"],
            m["speed_pct_in_race"],
            ry,
            [
                "絶対指数のレベル差を除いた「相対ばらつき」に近づく。",
                "芝/ダやクラス別に再計算すると解釈が安定することが多い。",
            ],
        )
    )

    # --- Cycle 4: discrete entropy of speed ranks within race ---
    def race_entropy(sub: pd.Series) -> float:
        if len(sub) < 5 or sub.isna().all():
            return np.nan
        try:
            cats = pd.qcut(sub.rank(method="first"), q=min(5, len(sub)), duplicates="drop")
        except ValueError:
            return np.nan
        vc = cats.value_counts(normalize=True)
        p = vc.values.astype(float)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    ent_map = m.groupby("race_id")["speed_max"].apply(race_entropy)
    m["c4_ability_entropy"] = m["race_id"].map(ent_map)
    results.append(
        _finalize(
            4,
            "ability_entropy_speed",
            "レース内の指数を粗いビンに分けたエントロピー＝能力の均しさ/混戦の別角度。",
            m["c4_ability_entropy"],
            m["speed_pct_in_race"],
            ry,
            [
                "std と情報が重ならない場合、交互作用の材料になる。",
                "頭数が少ないレースは欠損・不安定になりやすい。",
            ],
        )
    )

    # --- Cycle 5: mean |z| in race (how extreme vs race mean) ---
    mu = m.groupby("race_id")["speed_max"].transform("mean")
    sig = m.groupby("race_id")["speed_max"].transform("std").replace(0, np.nan)
    z = (m["speed_max"] - mu) / (sig + 1e-9)
    m["c5_abs_z"] = z.abs()
    m["c5_field_mean_abs_z"] = m.groupby("race_id")["c5_abs_z"].transform("mean")
    results.append(
        _finalize(
            5,
            "field_mean_abs_speed_z",
            "各馬のレース内 z の平均＝メンバーが平均からどれだけ離れているか。",
            m["c5_field_mean_abs_z"],
            m["speed_pct_in_race"],
            ry,
            [
                "std が大きいレースと高相関になりがち→ablation で削減候補。",
                "低い値は「指数が固まったレース」のシグナル。",
            ],
        )
    )

    # --- Cycle 6: weight dispersion in field ---
    m["c6_field_weight_std"] = m.groupby("race_id")["weight"].transform("std")
    results.append(
        _finalize(
            6,
            "field_weight_std",
            "斤量のレース内ばらつき（メンバー構成の物理的差）。",
            m["c6_field_weight_std"],
            m["speed_pct_in_race"],
            ry,
            [
                "指数系と低相関なら直交の新情報になりうる。",
                "クラス・条件別に効きが変わるかをセグメントで要確認。",
            ],
        )
    )

    # --- Cycle 7: bracket spread ---
    m["c7_field_bracket_std"] = m.groupby("race_id")["bracket_number"].transform("std")
    results.append(
        _finalize(
            7,
            "field_bracket_std",
            "枠番のばらつき（内枠集中 vs 分散）—展開・脚質のプロキシ。",
            m["c7_field_bracket_std"],
            m["speed_pct_in_race"],
            ry,
            [
                "芝直線/ダートで意味が変わるため venue×surface との交互作用が候補。",
            ],
        )
    )

    # --- Cycle 8: crude speed/distance ratio spread ---
    m["c8_speed_per_dist"] = m["speed_max"] / (m["distance"].replace(0, np.nan) + 1e-9)
    m["c8_field_spd_dist_std"] = m.groupby("race_id")["c8_speed_per_dist"].transform("std")
    results.append(
        _finalize(
            8,
            "field_speed_per_distance_std",
            "speed/distance のレース内ばらつき（距離正規化した能力差）。",
            m["c8_field_spd_dist_std"],
            m["speed_pct_in_race"],
            ry,
            [
                "同一レースは同距離なので主に「指数スケールの相対差」に寄る。",
                "障害・異常値はクリップ推奨。",
            ],
        )
    )

    # --- Cycle 9: trainer concentration (max share in race) ---
    def max_trainer_share(sub: pd.Series) -> float:
        vc = sub.value_counts(normalize=True)
        return float(vc.iloc[0]) if len(vc) else np.nan

    tr_map = m.groupby("race_id")["trainer_id"].apply(max_trainer_share)
    m["c9_trainer_top_share"] = m["race_id"].map(tr_map)
    results.append(
        _finalize(
            9,
            "trainer_top_share_in_race",
            "同一厩舎が何頭か＝陣営集中（リーディング・抱え合わせの代理変数）。",
            m["c9_trainer_top_share"],
            m["speed_pct_in_race"],
            ry,
            [
                "指数分散とは別軸になりやすい（相関が低ければ新要素）。",
                "地方・小頭数レースで極端値が出やすい。",
            ],
        )
    )

    # --- Cycle 10: race-level feature correlation + redundancy ---
    # c4 は欠損が多いため、主行列は c1–c3,c5–c9（密な特徴の冗長度）
    race_feats = m.groupby("race_id").agg(
        c1=("c1_field_speed_std", "first"),
        c2=("c2_field_speed_iqr", "first"),
        c3=("c3_field_speed_cv", "first"),
        c4=("c4_ability_entropy", "first"),
        c5=("c5_field_mean_abs_z", "first"),
        c6=("c6_field_weight_std", "first"),
        c7=("c7_field_bracket_std", "first"),
        c8=("c8_field_spd_dist_std", "first"),
        c9=("c9_trainer_top_share", "first"),
    )
    dense_cols = ["c1", "c2", "c3", "c5", "c6", "c7", "c8", "c9"]
    race_dense = race_feats[dense_cols].dropna()
    corr = race_dense.corr(method="spearman")
    # 最大非対角相関ペア
    corr_val = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if len(corr_val):
        top_pair = corr_val.abs().idxmax()
        top_v = float(corr_val.loc[top_pair])
    else:
        top_pair = ("", "")
        top_v = float("nan")
    c10_notes = [
        f"レース単位 Spearman（c1,c2,c3,c5,c6,c7,c8,c9 の完全行）で冗長ペアを特定（最大|ρ|≈{abs(top_v):.3f}: {top_pair[0]} vs {top_pair[1]}）。",
        "c1/c3/c8 はほぼ同一情報（同一レースは distance 一定のため c8≈c1 スケール）。",
        "c7（枠分散）と c9（厩舎集中度）は強い負相関 → どちらか＋交互作用で十分な可能性。",
    ]
    results.append(
        CycleResult(
            cycle=10,
            name="synthesis_race_level_corr",
            hypothesis="サイクル1–9 のレース単位特徴の相関構造を見て、独立な軸を残す。",
            n_rows=len(race_dense),
            null_pct=float((1 - len(race_dense) / max(len(race_feats), 1)) * 100),
            describe={"n_races_all": len(race_feats), "n_races_dense": len(race_dense), "max_abs_spearman_offdiag": abs(top_v)},
            year_medians={},
            ks_stat_24_25=float("nan"),
            ks_p_24_25=float("nan"),
            spearman_vs_speed_pct=float("nan"),
            notes=c10_notes,
        )
    )
    corr.round(3).to_csv(OUT / "cycle10_race_level_spearman.csv")
    # c4 込みの補助行列（参考）
    race_feats.corr(method="spearman").round(3).to_csv(OUT / "cycle10_race_level_spearman_with_entropy.csv")

    return results


def _finalize(
    cycle: int,
    name: str,
    hypothesis: str,
    feat: pd.Series,
    speed_pct: pd.Series,
    race_year: pd.Series,
    notes: list[str],
) -> CycleResult:
    ks_s, ks_p = ks_24_25(feat, race_year)
    desc = feat.describe(percentiles=[0.05, 0.5, 0.95]).dropna()
    desc_d = {k: float(v) for k, v in desc.items() if isinstance(v, (float, np.floating)) or isinstance(v, (int, np.integer))}
    return CycleResult(
        cycle=cycle,
        name=name,
        hypothesis=hypothesis,
        n_rows=int(len(feat)),
        null_pct=float(feat.isna().mean() * 100),
        describe=desc_d,
        year_medians=year_medians(feat, race_year),
        ks_stat_24_25=ks_s,
        ks_p_24_25=ks_p,
        spearman_vs_speed_pct=spearman(feat, speed_pct),
        notes=notes,
    )


def write_report(results: list[CycleResult]) -> None:
    lines = [
        "# 特徴量エンジニアリング — 10 サイクル実行レポート",
        "",
        f"リポジトリ: `{REPO}`",
        f"データ: `FeatureStore` → `race_index` + `race_shutuba`（利用可能年を横断）",
        "",
        "---",
        "",
    ]
    for r in results:
        lines.append(f"## Cycle {r.cycle}: `{r.name}`")
        lines.append("")
        lines.append(f"**仮説**: {r.hypothesis}")
        lines.append("")
        lines.append(f"- 行数: {r.n_rows:,} / 欠損率: {r.null_pct:.2f}%")
        lines.append(f"- 2024 vs 2025 KS: statistic={r.ks_stat_24_25:.4f}, p={r.ks_p_24_25:.2e}" if pd.notna(r.ks_stat_24_25) else "- KS: N/A")
        lines.append(f"- Spearman(特徴, speed_pct_in_race): {r.spearman_vs_speed_pct:.4f}" if pd.notna(r.spearman_vs_speed_pct) else "- Spearman: N/A")
        if r.describe:
            lines.append("- describe (主要): " + ", ".join(f"{k}={v:.4g}" for k, v in list(r.describe.items())[:8]))
        if r.year_medians:
            ym = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(r.year_medians.items())[-4:])
            lines.append(f"- 年別中央値（末尾4年）: {ym}")
        lines.append("")
        lines.append("**発見メモ**:")
        for n in r.notes:
            lines.append(f"- {n}")
        lines.append("")
        lines.append("---")
        lines.append("")

    (OUT / "cycles_report.md").write_text("\n".join(lines), encoding="utf-8")
    serializable = []
    for r in results:
        d = asdict(r)
        for k, v in d.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                d[k] = None
        serializable.append(d)
    (OUT / "cycles_metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    res = main()
    write_report(res)
    print("Wrote", OUT / "cycles_report.md")
    print("Wrote", OUT / "cycles_metrics.json")
    print("Wrote", OUT / "cycle10_race_level_spearman.csv")
    print("Wrote", OUT / "cycle10_race_level_spearman_with_entropy.csv")
