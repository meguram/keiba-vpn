"""
レースパフォーマンス指数 (perf_index)
======================================

定義:
    perf_index = 50 + (daily_mean_z - race_z) × baseline_std × 10

- 50 = 基準値（standard）
- daily_mean_z: その日の同会場・同surface全レースのz値平均（馬場速度）
- race_z:       個別レースのz値（ペース補正済み）→ 列名 `z`
- baseline_std: そのレースのベースライン標準偏差 → 列名 `baseline_std`
- 1単位 = 0.1秒、高いほどパフォーマンスが高い

レベル分類（正規分布基準）:
    ハイレベル: perf_index ≥ 56.5 (50 + 0.5σ, 上位~31%)
    平均的    : 43.5 < perf_index < 56.5
    低レベル  : perf_index ≤ 43.5 (50 - 0.5σ, 下位~31%)
"""

from __future__ import annotations

import glob
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("research.perf_index")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RACES_DIR = PROJECT_ROOT / "data" / "analysis" / "track_speed"
BASELINES_PATH = PROJECT_ROOT / "data" / "knowledge" / "track_speed_baselines.parquet"

# baseline_std の上限キャップ
# distance=0 のレースなど集約ベースラインは std が20〜47と異常に大きく、
# そのまま使うとperf_indexが数百〜数千に膨張する。
# 距離既知レースのstdは 0.55〜2.88 (p99≈2.5) なので 3.0 でキャップする。
BASELINE_STD_CAP = 3.0

# 2000m基準化の参照距離
REF_DISTANCE_STD2000 = 2000

# sigma参照テーブルのキャッシュ
_sigma_ref_cache: dict | None = None

PERF_INDEX_BASE = 50.0
PERF_STD_APPROX = 13.0          # バリデーション実測値
PERF_LEVEL_HI_THRESH = PERF_INDEX_BASE + 0.5 * PERF_STD_APPROX  # 56.5
PERF_LEVEL_LO_THRESH = PERF_INDEX_BASE - 0.5 * PERF_STD_APPROX  # 43.5


def invalidate_sigma_cache() -> None:
    """ベースライン再構築後に呼び出してsigmaキャッシュをリセットする。"""
    global _sigma_ref_cache
    _sigma_ref_cache = None


def load_sigma_reference() -> dict:
    """(surface, distance, cond_pool) → median baseline_std（JRA会場のみ）。"""
    global _sigma_ref_cache
    if _sigma_ref_cache is not None:
        return _sigma_ref_cache
    if not BASELINES_PATH.exists():
        _sigma_ref_cache = {}
        return {}
    try:
        df = pd.read_parquet(BASELINES_PATH)
        if "std" not in df.columns:
            _sigma_ref_cache = {}
            return {}
        ref: dict = {}
        work = df[df["venue"] != "ALL"]
        for (surf, dist, pool), g in work.groupby(["surface", "distance", "cond_pool"]):
            ref[(str(surf), int(dist), str(pool))] = float(g["std"].median())
        _sigma_ref_cache = ref
    except Exception:
        _sigma_ref_cache = {}
    return _sigma_ref_cache


def _sigma_at(sigma_ref: dict, surface: str, distance: int, cond_pool: str) -> float | None:
    """(surface, distance, cond_pool) のsigmaをフォールバック付きで返す。"""
    v = sigma_ref.get((surface, distance, cond_pool))
    if v is not None:
        return v
    v = sigma_ref.get((surface, distance, "firm_yielding"))
    if v is not None:
        return v
    # 最近傍距離（同surface同cond_pool）
    same_surf_pool = [(d, s) for (su, d, p), s in sigma_ref.items() if su == surface and p == cond_pool]
    if same_surf_pool:
        return min(same_surf_pool, key=lambda t: abs(t[0] - distance))[1]
    same_surf_fw = [(d, s) for (su, d, p), s in sigma_ref.items() if su == surface and p == "firm_yielding"]
    if same_surf_fw:
        return min(same_surf_fw, key=lambda t: abs(t[0] - distance))[1]
    return None


def _std2000_scale(
    sigma_ref: dict,
    surface: str,
    distance: int,
    cond_pool: str,
) -> float:
    """perf_index の 2000m良基準スケール係数 = σ_2000_good / σ_actual。"""
    sigma_2000 = _sigma_at(sigma_ref, surface, REF_DISTANCE_STD2000, "firm_yielding")
    sigma_actual = _sigma_at(sigma_ref, surface, distance, cond_pool)
    if not sigma_2000 or not sigma_actual or sigma_actual < 1e-6:
        return 1.0
    scale = sigma_2000 / sigma_actual
    # スケールを合理的な範囲にクリップ（±3倍まで）
    return max(0.33, min(3.0, scale))


def classify_perf_level(perf_index: float) -> tuple[str, str]:
    """(label, key): ハイレベル/hi  平均的/mid  低レベル/lo"""
    if perf_index >= PERF_LEVEL_HI_THRESH:
        return "ハイレベル", "hi"
    if perf_index <= PERF_LEVEL_LO_THRESH:
        return "低レベル", "lo"
    return "平均的", "mid"


def add_perf_level(df: pd.DataFrame) -> pd.DataFrame:
    """perf_index列からperf_level/perf_level_label列を追加して返す。"""
    if df.empty or "perf_index" not in df.columns:
        return df
    out = df.copy()
    labels, keys = [], []
    for v in out["perf_index"]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            labels.append(None)
            keys.append(None)
        else:
            lbl, key = classify_perf_level(float(v))
            labels.append(lbl)
            keys.append(key)
    out["perf_level_label"] = labels
    out["perf_level"] = keys
    return out


def _robust_mean_z(zs: np.ndarray) -> tuple[float, dict]:
    """
    IQR フェンス法による外れ値除去後のロバスト平均を返す。

    n <= 3 : 単純平均（サンプル不足のため robust 化しない）
    n >= 4 : Tukey フェンス [Q1-1.5IQR, Q3+1.5IQR] 外を除外後に平均。
             除外後の残りが 2 未満の場合は中央値にフォールバック。

    Returns:
        (mean_z, stats_dict)
        stats_dict keys: n_total, n_inliers, n_outliers, z_std, z_spread, confidence
    """
    n = len(zs)
    if n == 0:
        return 0.0, {}
    if n <= 3:
        return float(np.mean(zs)), {
            "n_total": n, "n_inliers": n, "n_outliers": 0,
            "z_std": float(np.std(zs)) if n > 1 else 0.0,
            "z_spread": float(np.max(zs) - np.min(zs)) if n > 1 else 0.0,
            "confidence": "low",
        }

    q1, q3 = float(np.percentile(zs, 25)), float(np.percentile(zs, 75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (zs >= lo) & (zs <= hi)
    inliers = zs[mask]
    n_out = int((~mask).sum())

    if len(inliers) >= 2:
        mean_z = float(np.mean(inliers))
        z_std = float(np.std(inliers))
    else:
        mean_z = float(np.median(zs))
        z_std = float(np.std(zs))

    z_spread = float(np.max(inliers) - np.min(inliers)) if len(inliers) > 1 else 0.0

    # 信頼度: std < 0.5 かつ outlier_rate < 0.2 → high
    outlier_rate = n_out / n
    if z_std < 0.5 and outlier_rate < 0.2 and len(inliers) >= 4:
        confidence = "high"
    elif z_std < 1.0 and outlier_rate < 0.4:
        confidence = "mid"
    else:
        confidence = "low"

    return mean_z, {
        "n_total": n,
        "n_inliers": int(len(inliers)),
        "n_outliers": n_out,
        "z_std": round(z_std, 3),
        "z_spread": round(z_spread, 3),
        "confidence": confidence,
    }


def build_daily_mean_z_map(df: pd.DataFrame) -> dict[str, float]:
    """
    {date|venue|surface: robust_mean_z}

    IQR フェンス法による外れ値除去後のロバスト平均 z-score を返す。
    """
    if df.empty or "z" not in df.columns:
        return {}

    work = df[["date", "venue", "surface", "z"]].copy()
    work = work[work["z"].notna()]
    if work.empty:
        return {}

    work["date"] = work["date"].astype(str).str.replace("-", "", regex=False)

    result: dict[str, float] = {}
    for (date, venue, surface), g in work.groupby(["date", "venue", "surface"]):
        zs = g["z"].dropna().values.astype(float)
        if len(zs) == 0:
            continue
        mean_z, _ = _robust_mean_z(zs)
        result[f"{date}|{venue}|{surface}"] = mean_z

    return result


def build_daily_z_stats(df: pd.DataFrame) -> dict[str, dict]:
    """
    {date|venue|surface: stats_dict}

    ロバスト mean_z に加えて n_total/n_outliers/z_std/confidence 等の詳細統計を返す。
    confidence: "high" | "mid" | "low"
    """
    if df.empty or "z" not in df.columns:
        return {}

    work = df[["date", "venue", "surface", "z"]].copy()
    work = work[work["z"].notna()]
    if work.empty:
        return {}

    work["date"] = work["date"].astype(str).str.replace("-", "", regex=False)

    result: dict[str, dict] = {}
    for (date, venue, surface), g in work.groupby(["date", "venue", "surface"]):
        zs = g["z"].dropna().values.astype(float)
        if len(zs) == 0:
            continue
        mean_z, stats = _robust_mean_z(zs)
        result[f"{date}|{venue}|{surface}"] = {"mean_z": round(mean_z, 3), **stats}

    return result


# 時間的プーリングの重み付け係数
# 分析結果: 同条件r=0.551(土→日), r=0.516(前週→当週), 異条件r=0.2-0.4
TEMPORAL_WEIGHTS = {
    "current": 1.0,
    "prior_day": 0.35,   # 同週前日(同条件)
    "prior_week": 0.25,  # 前週同曜日(同条件)
}


def _modal_condition(df_group: pd.DataFrame) -> str | None:
    """グループ内で最多の馬場状態を返す。"""
    if df_group.empty or "track_condition" not in df_group.columns:
        return None
    counts = df_group["track_condition"].value_counts()
    return str(counts.index[0]) if len(counts) > 0 else None


def _compute_temporal_pooled_stats(
    groups: list[tuple[np.ndarray, float]],
) -> tuple[float, dict]:
    """
    複数時間グループの加重平均z-scoreを計算する。
    各グループに対して独立にロバスト平均を計算後、重み付け平均する。
    重みはdeclared_weight × n_inliers（レース数多いほど信頼性高）。
    """
    group_means: list[float] = []
    group_weights_eff: list[float] = []
    n_current = 0
    n_inliers_curr = 0
    n_outliers_curr = 0
    confidence_curr = "low"
    z_std_curr = 0.0
    n_prior_sources = 0

    for i, (zs, w) in enumerate(groups):
        if len(zs) == 0:
            continue
        mean_z, stats = _robust_mean_z(zs)
        effective_n = stats.get("n_inliers", len(zs))
        group_means.append(mean_z)
        group_weights_eff.append(w * max(effective_n, 1))

        if i == 0:
            n_current = len(zs)
            n_inliers_curr = stats.get("n_inliers", len(zs))
            n_outliers_curr = stats.get("n_outliers", 0)
            confidence_curr = stats.get("confidence", "mid")
            z_std_curr = stats.get("z_std", 0.0)
        else:
            n_prior_sources += 1

    if not group_means:
        return 0.0, {}

    total_w = sum(group_weights_eff)
    pooled_mean = sum(m * w for m, w in zip(group_means, group_weights_eff)) / total_w

    return pooled_mean, {
        "n_total": n_current,
        "n_inliers": n_inliers_curr,
        "n_outliers": n_outliers_curr,
        "z_std": z_std_curr,
        "z_spread": 0.0,
        "confidence": confidence_curr,
        "n_prior_sources": n_prior_sources,
    }


def build_all_temporal_z_maps(
    df: pd.DataFrame,
    extra_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    dfに含まれる全日付に対してテンポラルプーリングz-mapを一括構築する。

    extra_dfは前日・前週データを補完するための追加DataFrame（例: 前年分）。
    Returns: {YYYYMMDD|venue|surface: pooled_mean_z}
    """
    if df.empty or "z" not in df.columns:
        return {}

    work = df.copy()
    work["date"] = work["date"].astype(str).str.replace("-", "", regex=False)

    if extra_df is not None and not extra_df.empty:
        extra = extra_df.copy()
        extra["date"] = extra["date"].astype(str).str.replace("-", "", regex=False)
        lookup_df = pd.concat(
            [work, extra[extra.columns.intersection(work.columns)]],
            ignore_index=True,
        )
    else:
        lookup_df = work

    result: dict[str, float] = {}
    for target_date in sorted(work["date"].unique()):
        stats = build_temporal_pooled_z_stats(lookup_df, str(target_date))
        for key, s in stats.items():
            result[key] = s["mean_z"]

    return result


def build_temporal_pooled_z_stats(
    df: pd.DataFrame,
    target_date: str,
) -> dict[str, dict]:
    """
    目標日に対して時間的プーリングを適用した日別z-statsを計算する。

    プーリング対象:
        当日 (weight=1.0): 常に含む
        前日 (weight=0.35): 同会場・同面・同馬場状態の場合のみ
        前週同曜 (weight=0.25): 同会場・同面・同馬場状態の場合のみ

    Args:
        df: 当日 + 前日 + 前週のレースデータを含むDataFrame
        target_date: YYYYMMDD形式の目標日

    Returns:
        {"YYYYMMDD|venue|surface": stats_dict} 形式の辞書
    """
    if df.empty or "z" not in df.columns:
        return {}

    work = df[["date", "venue", "surface", "z", "track_condition"]].copy()
    work = work[work["z"].notna()]
    if work.empty:
        return {}

    work["date"] = work["date"].astype(str).str.replace("-", "", regex=False)
    target_d = target_date.replace("-", "")

    # 前日・前週の日付を計算
    dt = datetime.strptime(target_d, "%Y%m%d")
    prior_day_str = (dt - timedelta(days=1)).strftime("%Y%m%d")
    prior_week_str = (dt - timedelta(days=7)).strftime("%Y%m%d")

    target_rows = work[work["date"] == target_d]
    if target_rows.empty:
        return {}

    result: dict[str, dict] = {}
    for (venue, surface), curr_g in target_rows.groupby(["venue", "surface"]):
        curr_zs = curr_g["z"].dropna().values.astype(float)
        if len(curr_zs) == 0:
            continue

        curr_cond = _modal_condition(curr_g)
        groups: list[tuple[np.ndarray, float]] = [(curr_zs, TEMPORAL_WEIGHTS["current"])]

        # 前日（同週）
        prior_day_g = work[
            (work["date"] == prior_day_str)
            & (work["venue"] == venue)
            & (work["surface"] == surface)
        ]
        if not prior_day_g.empty and _modal_condition(prior_day_g) == curr_cond:
            pz = prior_day_g["z"].dropna().values.astype(float)
            if len(pz) > 0:
                groups.append((pz, TEMPORAL_WEIGHTS["prior_day"]))

        # 前週同曜
        prior_week_g = work[
            (work["date"] == prior_week_str)
            & (work["venue"] == venue)
            & (work["surface"] == surface)
        ]
        if not prior_week_g.empty and _modal_condition(prior_week_g) == curr_cond:
            pwz = prior_week_g["z"].dropna().values.astype(float)
            if len(pwz) > 0:
                groups.append((pwz, TEMPORAL_WEIGHTS["prior_week"]))

        pooled_mean_z, stats = _compute_temporal_pooled_stats(groups)
        result[f"{target_d}|{venue}|{surface}"] = {
            "mean_z": round(pooled_mean_z, 3),
            **stats,
        }

    return result


def add_perf_index(
    df: pd.DataFrame,
    daily_mean_z_map: dict | None = None,
) -> pd.DataFrame:
    """
    DataFrameにperf_index / perf_index_std2000 列を追加して返す。

    perf_index      = 50 + (daily_mean_z - race_z) × baseline_std × 10
                      コース内での生の評価（1pt=0.1秒）

    perf_index_std2000 = 50 + (daily_mean_z - race_z) × baseline_std × 10
                             × (σ_2000m_良 / σ_distance_cond)
                      全コース横断比較用（2000m良馬場相当スケール）

    daily_mean_z_map が None の場合は df から自動構築する。
    baseline_std が 0 または NaN の場合は None。
    """
    if df.empty:
        return df

    out = df.copy()
    out["date"] = out["date"].astype(str).str.replace("-", "", regex=False)

    if daily_mean_z_map is None:
        daily_mean_z_map = build_daily_mean_z_map(out)

    sigma_ref = load_sigma_reference()

    perf_values: list[float | None] = []
    perf_std_values: list[float | None] = []
    for _, row in out.iterrows():
        z = row.get("z")
        bstd = row.get("baseline_std")
        date_str = str(row.get("date", "")).replace("-", "")
        venue = str(row.get("venue", ""))
        surface = str(row.get("surface", ""))

        if z is None or (isinstance(z, float) and math.isnan(z)):
            perf_values.append(None)
            perf_std_values.append(None)
            continue
        if bstd is None or (isinstance(bstd, float) and math.isnan(bstd)) or float(bstd) == 0:
            perf_values.append(None)
            perf_std_values.append(None)
            continue

        key = f"{date_str}|{venue}|{surface}"
        daily_z = daily_mean_z_map.get(key)
        if daily_z is None:
            perf_values.append(None)
            perf_std_values.append(None)
            continue

        bstd_eff = min(float(bstd), BASELINE_STD_CAP)
        delta = (float(daily_z) - float(z)) * bstd_eff * 10.0

        perf = PERF_INDEX_BASE + delta
        if math.isnan(perf) or math.isinf(perf):
            perf_values.append(None)
            perf_std_values.append(None)
            continue
        perf_values.append(round(perf, 2))

        # 2000m良基準スケール補正
        distance = int(row.get("distance") or 0)
        cond_pool = str(row.get("cond_pool") or "firm_yielding")
        scale = _std2000_scale(sigma_ref, surface, distance, cond_pool) if distance > 0 else 1.0
        perf_std = PERF_INDEX_BASE + delta * scale
        if math.isnan(perf_std) or math.isinf(perf_std):
            perf_std_values.append(None)
        else:
            perf_std_values.append(round(perf_std, 2))

    out["perf_index"] = perf_values
    out["perf_index_std2000"] = perf_std_values
    return out


def run_validation(n_race_threshold: int = 100) -> dict[str, Any]:
    """
    4つのバリデーションテストを実行してdictで結果を返す。

    Parquetファイルは RACES_DIR/races_*.parquet を使用。

    Tests:
        T1: 分布妥当性 - |mean(perf_index)| < 5 かつ 5 ≤ std ≤ 60
        T2: クラス中立性 - 各class_bandの平均がglobal_meanから±8以内 (n≥50)
        T3: 馬場独立性 - |corr(perf_index, daily_baba_index)| < 0.35
        T4: 時系列安定性 - 年別mean範囲 < 8.0

    Returns dict with keys:
        - passed: bool (all tests passed)
        - tests: dict of test results
        - n_total: total number of races
    """
    # Load all parquet files
    parquet_files = sorted(RACES_DIR.glob("races_*.parquet"))
    if not parquet_files:
        return {
            "passed": False,
            "error": f"No parquet files found in {RACES_DIR}",
            "tests": {},
        }

    frames: list[pd.DataFrame] = []
    for p in parquet_files:
        try:
            df_year = pd.read_parquet(p)
            frames.append(df_year)
        except Exception as e:
            logger.warning("Failed to load %s: %s", p, e)

    if not frames:
        return {
            "passed": False,
            "error": "Failed to load any parquet files",
            "tests": {},
        }

    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = all_df["date"].astype(str).str.replace("-", "", regex=False)

    if len(all_df) < n_race_threshold:
        return {
            "passed": False,
            "error": f"Not enough races: {len(all_df)} < {n_race_threshold}",
            "tests": {},
        }

    # Build perf_index across all data
    daily_mean_z_map = build_daily_mean_z_map(all_df)
    all_df = add_perf_index(all_df, daily_mean_z_map)

    valid = all_df[all_df["perf_index"].notna()].copy()
    if valid.empty:
        return {
            "passed": False,
            "error": "No valid perf_index values computed",
            "tests": {},
        }

    # Build daily_baba_index = mean baba_index per date×venue×surface
    # We use z values to compute it (baba_index = 50 - z * 10)
    valid["daily_baba_index"] = valid.apply(
        lambda row: 50 - daily_mean_z_map.get(
            f"{row['date']}|{row['venue']}|{row['surface']}", float("nan")
        ) * 10,
        axis=1,
    )

    pi = valid["perf_index"]
    pi_mean = float(pi.mean())
    pi_std = float(pi.std())

    # ── T1: Distribution validity ─────────────────────────────────────
    t1_mean_ok = abs(pi_mean - PERF_INDEX_BASE) < 5
    t1_std_ok = 5 <= pi_std <= 60
    t1_passed = t1_mean_ok and t1_std_ok
    t1 = {
        "passed": t1_passed,
        "mean": round(pi_mean, 3),
        "std": round(pi_std, 3),
        "criterion": f"|mean - {PERF_INDEX_BASE}| < 5 かつ 5 ≤ std ≤ 60",
        "details": {
            "mean_ok": t1_mean_ok,
            "std_ok": t1_std_ok,
        },
    }

    # ── T2: Class neutrality ─────────────────────────────────────────
    global_mean = pi_mean
    t2_bands: dict[str, Any] = {}
    t2_passed = True
    for band, g in valid.groupby("class_band"):
        if len(g) < 50:
            continue
        band_mean = float(g["perf_index"].mean())
        diff = abs(band_mean - global_mean)
        ok = diff <= 8.0
        if not ok:
            t2_passed = False
        t2_bands[str(band)] = {
            "n": len(g),
            "mean": round(band_mean, 3),
            "diff_from_global": round(diff, 3),
            "ok": ok,
        }
    t2 = {
        "passed": t2_passed,
        "global_mean": round(global_mean, 3),
        "criterion": "各class_bandの平均がglobal_meanから±8以内（n≥50のみ）",
        "bands": t2_bands,
    }

    # ── T3: Track condition independence ─────────────────────────────
    t3_passed = False
    t3_corr = None
    dbi = valid["daily_baba_index"].dropna()
    pi_aligned = valid.loc[dbi.index, "perf_index"].dropna()
    common_idx = dbi.index.intersection(pi_aligned.index)
    if len(common_idx) >= 30:
        dbi_arr = dbi.loc[common_idx].values.astype(float)
        pi_arr = pi_aligned.loc[common_idx].values.astype(float)
        # Remove inf/nan
        mask = np.isfinite(dbi_arr) & np.isfinite(pi_arr)
        if mask.sum() >= 30:
            corr_matrix = np.corrcoef(dbi_arr[mask], pi_arr[mask])
            t3_corr = float(corr_matrix[0, 1])
            if not math.isnan(t3_corr):
                t3_passed = abs(t3_corr) < 0.35
    t3 = {
        "passed": t3_passed,
        "corr": round(t3_corr, 4) if t3_corr is not None else None,
        "criterion": "|corr(perf_index, daily_baba_index)| < 0.35",
        "n_pairs": int(mask.sum()) if t3_corr is not None else 0,
    }

    # ── T4: Time series stability ─────────────────────────────────────
    valid["year"] = valid["date"].astype(str).str[:4]
    year_means: dict[str, float] = {}
    for yr, g in valid.groupby("year"):
        if len(g) >= 30:
            year_means[str(yr)] = round(float(g["perf_index"].mean()), 3)

    t4_passed = False
    t4_range = None
    if len(year_means) >= 2:
        vals = list(year_means.values())
        t4_range = round(max(vals) - min(vals), 3)
        t4_passed = t4_range < 8.0
    t4 = {
        "passed": t4_passed,
        "year_means": year_means,
        "range": t4_range,
        "criterion": "年別mean範囲 < 8.0",
    }

    all_passed = t1_passed and t2_passed and t3_passed and t4_passed

    return {
        "passed": all_passed,
        "n_total": len(all_df),
        "n_valid": len(valid),
        "tests": {
            "T1_distribution": t1,
            "T2_class_neutrality": t2,
            "T3_track_independence": t3,
            "T4_time_stability": t4,
        },
    }
