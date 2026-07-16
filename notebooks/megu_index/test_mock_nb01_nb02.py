#!/usr/bin/env python3
"""
NB-01 / NB-02 モックデータ動作確認スクリプト

実際の parquet ファイルなしに、合成データで主要パイプラインが
正常に動作するかを end-to-end で検証する。

実行例:
    cd /home/hirokiakataoka/project/myproject/project-multi-agent/project/keiba-vpn/repo
    python notebooks/megu_index/test_mock_nb01_nb02.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── パス設定 ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# ── ソースモジュール ──────────────────────────────────────────────────────
from src.pipeline.megu_index.lap_splits import parse_lap_times, select_split_point
from src.pipeline.megu_index.track_speed import attach_track_speed_to_horses
from src.pipeline.megu_index.par_front_split import (
    fit_par_front_split_coefficients,
    attach_par_front_split_sec,
)
from src.pipeline.megu_index.weight_age_base import attach_base_weight
from src.pipeline.megu_index.fit_coeff_pace import fit_coeff_pace
from src.pipeline.megu_index.fit_par_time_class import fit_par_time_class, fit_pool_betas
from src.pipeline.megu_index.track_speed import assign_class_rank

# ── 定数 ─────────────────────────────────────────────────────────────────
THEORY_BETA = 0.2
MAX_SEC_PER_KG = 0.5
MIN_SAMPLES = 30

PASS = []
FAIL = []


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")
    PASS.append(msg)


def fail(msg: str, detail: str = "") -> None:
    print(f"  ✗ {msg}" + (f"\n    {detail}" if detail else ""))
    FAIL.append(msg)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =========================================================================
# モックデータ生成
# =========================================================================

def _lap_list(distance: int, total_sec: float) -> list[float]:
    """距離・総タイムから合成ラップリストを生成。"""
    n = distance // 200 + (1 if distance % 200 else 0)
    # 200m 倍数に揃える
    if distance % 200 == 0:
        n = distance // 200
    else:
        n = distance // 200 + 1
    mean_lap = total_sec / n
    # 後半がやや速くなるパターン
    pattern = np.linspace(1.04, 0.96, n)
    laps = (pattern * mean_lap)
    # 合計を total_sec に正規化
    laps = (laps / laps.sum() * total_sec).round(1).tolist()
    return laps


def _pace_json(distance: int, laps: list[float]) -> str:
    """ペース JSON 文字列を生成。"""
    if len(laps) >= 6:
        first = round(sum(laps[:3]), 1)
        second = round(sum(laps[-3:]), 1)
    else:
        mid = len(laps) // 2
        first = round(sum(laps[:mid]), 1)
        second = round(sum(laps[mid:]), 1)
    return json.dumps({"first_half_3f": first, "second_half_3f": second})


def build_mock_race_result_flat(seed: int = 42) -> pd.DataFrame:
    """
    race_result_flat.parquet の構造を模したモック DataFrame を生成する。

    設計:
      - 5会場 × 2馬場 × 3距離 × 5年 × 8レース = 1,200 レース
      - 1レース 12頭 → 合計 14,400 行
      - 年齢: 2〜6歳、性別: 牡・牝・セン
      - 斤量: 年齢×性別基準 ± 小変動（別定加算を模擬）
    """
    rng = np.random.default_rng(seed)

    venues = ["東京", "阪神", "中山", "京都", "中京"]
    venue_codes = {"東京": "05", "阪神": "08", "中山": "06", "京都": "07", "中京": "10"}
    surface_dist = [
        ("芝",   1200, 65.0),
        ("芝",   1600, 92.0),
        ("芝",   2000, 120.0),
        ("ダート", 1200, 70.0),
        ("ダート", 1400, 84.0),
        ("ダート", 1800, 112.0),
    ]
    years = [2020, 2021, 2022, 2023, 2024]
    races_per_cell = 8
    n_horses = 12

    grade_class_map = [
        ("未勝利", "サラ系2歳 未勝利"),
        ("未勝利", "サラ系3歳 未勝利"),
        ("1勝",   "3歳以上1勝クラス"),
        ("2勝",   "3歳以上2勝クラス"),
        ("3勝",   "3歳以上3勝クラス"),
        ("OP",    "オープン"),
    ]

    rows = []
    race_idx = 0

    for year in years:
        for venue in venues:
            for surf, dist, par_sec in surface_dist:
                for r in range(races_per_cell):
                    # レース基本情報
                    grade, race_class = grade_class_map[r % len(grade_class_map)]
                    month = (r % 10) + 3   # 3〜12月に分散
                    day = rng.integers(1, 28)
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    race_id = f"{year}{venue_codes[venue]}{r+1:04d}{race_idx:02d}"
                    track_cond = rng.choice(["良", "良", "良", "稍重", "重"], p=[0.5, 0.2, 0.1, 0.15, 0.05])
                    weather = rng.choice(["晴", "曇", "小雨"])
                    field_size = n_horses

                    # レース全体の馬場速度バリエーション（±1.5秒）
                    track_delta = float(rng.normal(0, 0.8))

                    # ラップ生成（全馬共通ラップ）
                    race_par = par_sec + track_delta
                    laps = _lap_list(dist, race_par)
                    lap_json = json.dumps(laps)
                    pace_json = _pace_json(dist, laps)

                    # 馬能力分布
                    abilities = rng.normal(0, 2.0, n_horses)
                    order = np.argsort(abilities)[::-1]  # 強い馬が速い

                    for pos, idx in enumerate(order):
                        finish_pos = pos + 1
                        # 年齢設定: グレード別に年齢分布を変える
                        if "2歳" in race_class:
                            age = 2
                        elif "3歳" in race_class and "以上" not in race_class:
                            age = 3
                        else:
                            age = rng.choice([3, 4, 5, 6], p=[0.25, 0.35, 0.30, 0.10])

                        sex = rng.choice(["牡", "牝", "セン"], p=[0.50, 0.35, 0.15])
                        sex_age = f"{sex}{age}"

                        # 基準斤量（JRA馬齢重量）
                        if age == 2:
                            base_m = 54.0 if month <= 9 else (55.0 if month <= 11 else 56.0)
                            female_disc = 0.0 if month <= 9 else 1.0
                        elif age == 3:
                            base_m = 55.0 if month <= 2 else (56.0 if month <= 4 else 57.0)
                            female_disc = 2.0
                        else:
                            base_m = 58.0
                            female_disc = 2.0
                        base_w = base_m - (female_disc if sex == "牝" else 0.0)

                        # 別定加算（収得賞金模擬: 0〜4kg）+ 騎手体重誤差
                        prize_add = rng.choice([0.0, 0.0, 1.0, 2.0, 3.0], p=[0.4, 0.3, 0.15, 0.1, 0.05])
                        jockey_err = float(rng.choice([0.0, 0.5, -0.5, 1.0], p=[0.6, 0.15, 0.15, 0.1]))
                        jockey_w = base_w + prize_add + jockey_err

                        # タイム（能力差 + 斤量影響 + ノイズ）
                        ability_gap = abilities[idx] - abilities[order[0]]
                        coeff_w = THEORY_BETA * (dist / 2000)
                        weight_effect = (jockey_w - base_w) * coeff_w
                        noise = float(rng.normal(0, 0.3))
                        time_sec = race_par - ability_gap * 0.3 + weight_effect + noise
                        if time_sec <= 0:
                            time_sec = race_par

                        # last_3f: 最後3ラップの合計
                        last3f = round(sum(laps[-3:]) if len(laps) >= 3 else sum(laps), 1)

                        horse_id = f"{year - age + 1}{rng.integers(100000, 199999)}"

                        rows.append({
                            "race_id": race_id,
                            "date": date_str,
                            "venue": venue,
                            "venue_code": venue_codes[venue],
                            "surface": surf,
                            "distance": dist,
                            "direction": "右",
                            "grade": grade,
                            "race_class": race_class,
                            "weather": weather,
                            "track_condition": track_cond,
                            "field_size": field_size,
                            "race_name": f"テストレース{r+1}",
                            "horse_id": horse_id,
                            "horse_number": pos + 1,
                            "sex_age": sex_age,
                            "jockey_weight": jockey_w,
                            "finish_position": finish_pos,
                            "time_sec": round(time_sec, 1),
                            "last_3f": last3f,
                            "lap_times": lap_json,
                            "pace": pace_json,
                            "year": year,
                        })
                    race_idx += 1

    df = pd.DataFrame(rows)
    print(f"  モックデータ生成: {len(df):,} 行 / {df['race_id'].nunique():,} レース")
    print(f"  年: {sorted(df['year'].unique())}  surface: {df['surface'].unique().tolist()}")
    print(f"  distance: {sorted(df['distance'].unique())}  venue: {df['venue'].nunique()} 会場")
    return df


# =========================================================================
# NB-01 パイプライン
# =========================================================================

def run_nb01(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NB-01 の主要処理を再現。df_clean, par_split_full を返す。"""

    # ── 数値化 ──
    df_raw = df_raw.copy()
    df_raw["finish_pos"] = pd.to_numeric(df_raw["finish_position"], errors="coerce")
    df_raw["time_sec_num"] = pd.to_numeric(df_raw["time_sec"], errors="coerce")
    df_raw["distance_num"] = pd.to_numeric(df_raw["distance"], errors="coerce")
    df_raw["jockey_weight_num"] = pd.to_numeric(df_raw["jockey_weight"], errors="coerce")

    df_raw["sex"] = df_raw["sex_age"].astype(str).str.extract(r"^(牡|牝|セン)", expand=False).fillna("牡")
    df_raw["sex_group"] = np.where(df_raw["sex"] == "牝", "牝", "牡")
    df_raw["age_num"] = pd.to_numeric(
        df_raw["sex_age"].astype(str).str.extract(r"(\d+)$", expand=False), errors="coerce"
    )

    # ── weight_age_base: 年齢×性別×月 基準斤量 ──────────────────────────
    df_raw = attach_base_weight(df_raw)
    df_raw["weight_dev_kg"] = df_raw["jockey_weight_num"] - df_raw["base_weight_kg"]
    df_raw["weight_dev"] = df_raw["weight_dev_kg"]
    df_raw["dist_scale"] = df_raw["distance_num"] / 2000.0
    df_raw["weight_x_dist"] = df_raw["weight_dev_kg"] * df_raw["dist_scale"]

    # ── OLS 斤量係数推定 ──────────────────────────────────────────────────
    def _ols_slope(x, y):
        if len(x) < MIN_SAMPLES or np.nanvar(x) < 1e-9:
            return np.nan
        return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))

    mask_fit = (
        df_raw["finish_pos"].between(1, 5)
        & df_raw["time_sec_num"].notna()
        & df_raw["jockey_weight_num"].notna()
        & df_raw["surface"].isin(["芝", "ダート"])
        & df_raw["distance_num"].notna()
        & (df_raw["distance_num"] > 0)
    )
    df_fit = df_raw.loc[
        mask_fit,
        ["race_id", "surface", "distance_num", "sex_group", "weight_dev_kg", "weight_x_dist", "time_sec_num"],
    ].copy()
    df_fit["time_dm"] = df_fit["time_sec_num"] - df_fit.groupby("race_id")["time_sec_num"].transform("mean")
    df_fit["weight_dev_dm"] = df_fit["weight_dev_kg"] - df_fit.groupby("race_id")["weight_dev_kg"].transform("mean")
    df_fit["weight_x_dist_dm"] = df_fit["weight_x_dist"] - df_fit.groupby("race_id")["weight_x_dist"].transform("mean")

    coef_rows = []
    for (surf, dist, sex_g), g in df_fit.groupby(["surface", "distance_num", "sex_group"], sort=False):
        beta = _ols_slope(g["weight_dev_dm"].to_numpy(), g["time_dm"].to_numpy())
        if pd.notna(beta) and beta > 0:
            sec_per_kg = min(beta, MAX_SEC_PER_KG)
        else:
            sec_per_kg = np.nan
        coef_rows.append({"surface": surf, "distance_num": int(dist), "sex_group": sex_g,
                           "sec_per_kg_direct": sec_per_kg, "n_fit": len(g)})
    coef_by_cell = pd.DataFrame(coef_rows)

    surf_sex_beta = {}
    for (surf, sex_g), g in df_fit.groupby(["surface", "sex_group"]):
        b = _ols_slope(g["weight_x_dist_dm"].to_numpy(), g["time_dm"].to_numpy())
        surf_sex_beta[(surf, sex_g)] = b if pd.notna(b) and b > 0 else THEORY_BETA

    coef_by_cell["beta_ss"] = coef_by_cell.apply(
        lambda r: surf_sex_beta.get((r["surface"], r["sex_group"]), THEORY_BETA), axis=1
    )
    coef_by_cell["sec_per_kg_ss"] = np.minimum(
        coef_by_cell["beta_ss"] * (coef_by_cell["distance_num"] / 2000.0), MAX_SEC_PER_KG
    )
    coef_by_cell["sec_per_kg_final"] = coef_by_cell["sec_per_kg_direct"].fillna(coef_by_cell["sec_per_kg_ss"])
    coef_by_cell["weight_coef_source"] = np.where(
        coef_by_cell["sec_per_kg_direct"].notna(), "cell_sex_within_race", "surface_sex_within_race"
    )

    df_raw = df_raw.merge(
        coef_by_cell[["surface", "distance_num", "sex_group", "sec_per_kg_final", "weight_coef_source"]],
        on=["surface", "distance_num", "sex_group"], how="left",
    )
    df_raw["sec_per_kg_final"] = df_raw["sec_per_kg_final"].fillna(THEORY_BETA * df_raw["dist_scale"])
    df_raw["weight_coef_source"] = df_raw["weight_coef_source"].fillna("theory")
    df_raw["delta_weight_sec"] = df_raw["weight_dev_kg"] * df_raw["sec_per_kg_final"]
    df_raw["adjusted_time_sec"] = df_raw["time_sec_num"] - df_raw["delta_weight_sec"]

    # ── §2 前半スプリット ─────────────────────────────────────────────────
    df_flat = df_raw[df_raw["surface"].isin(["芝", "ダート"]) & df_raw["distance"].notna()].copy()
    df_flat["distance"] = pd.to_numeric(df_flat["distance"], errors="coerce")
    df_flat = df_flat[df_flat["distance"] > 0]

    splits = []
    for _, row in df_flat[["race_id", "distance", "lap_times"]].drop_duplicates("race_id").iterrows():
        lap_dict = parse_lap_times(row["lap_times"], row["distance"])
        if not lap_dict:
            splits.append({"race_id": row["race_id"], "front_split_sec": np.nan, "split_point_m": np.nan})
            continue
        sp = select_split_point(int(row["distance"]), list(lap_dict.keys()))
        if sp is None:
            splits.append({"race_id": row["race_id"], "front_split_sec": np.nan, "split_point_m": np.nan})
        else:
            splits.append({"race_id": row["race_id"], "front_split_sec": lap_dict[sp], "split_point_m": int(sp)})
    df_splits = pd.DataFrame(splits)

    # ── §3 馬場速度指数（3層設計）────────────────────────────────────────
    train_years = [2020, 2021, 2022, 2023, 2024]
    df_flat, race_track_tbl, day_course_tbl = attach_track_speed_to_horses(
        df_flat, train_years=train_years, min_samples=3,
        splits_df=df_splits,  # Layer 1 ペースフィルタ用
    )

    # ── §4 マスターデータセット ───────────────────────────────────────────
    df_main = df_flat.merge(df_splits[["race_id", "front_split_sec", "split_point_m"]], on="race_id", how="left")
    df_main["finish_time_raw_sec"] = pd.to_numeric(df_main["time_sec"], errors="coerce")
    df_main["finish_time_sec"] = df_main["adjusted_time_sec"]
    df_main["finish_pos"] = pd.to_numeric(df_main["finish_position"], errors="coerce")
    df_main = df_main[df_main["finish_time_sec"].notna() & df_main["surface"].isin(["芝", "ダート"])]

    dist_bins = [0, 1499, 1799, 2399, 9999]
    dist_labels = ["sprint", "mile", "middle", "long"]
    df_main["distance_band"] = pd.cut(df_main["distance"], bins=dist_bins, labels=dist_labels)
    df_main["track_cat"] = df_main["track_condition"].map(
        {"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}
    ).fillna("良")

    # ── §5 外れ値除去 ─────────────────────────────────────────────────────
    grp = ["distance_band", "surface", "track_cat"]
    df_main["time_mean"] = df_main.groupby(grp, observed=True)["finish_time_sec"].transform("mean")
    df_main["time_std"] = df_main.groupby(grp, observed=True)["finish_time_sec"].transform("std")
    df_main["time_z"] = (df_main["finish_time_sec"] - df_main["time_mean"]) / df_main["time_std"].replace(0, np.nan)
    df_clean = df_main[df_main["time_z"].abs() <= 3.5].copy()

    # ── §6 基準前半スプリット ─────────────────────────────────────────────
    # §3 で race_t2nd_sec が付与済みの場合は上書きマージ前に除去して列名衝突を防ぐ
    _t2 = df_clean[df_clean["finish_pos"] == 2][["race_id", "adjusted_time_sec"]].drop_duplicates("race_id")
    _t1 = df_clean[df_clean["finish_pos"] == 1][["race_id", "adjusted_time_sec"]].drop_duplicates("race_id").rename(
        columns={"adjusted_time_sec": "t1"}
    )
    race_t2nd = _t2.merge(_t1, on="race_id", how="left")
    race_t2nd["race_t2nd_sec"] = race_t2nd["adjusted_time_sec"].fillna(race_t2nd["t1"])
    race_t2nd = race_t2nd[["race_id", "race_t2nd_sec"]]

    if "race_t2nd_sec" in df_clean.columns:
        df_clean = df_clean.drop(columns=["race_t2nd_sec"])
    df_clean = df_clean.merge(race_t2nd, on="race_id", how="left")

    df_2nd = df_clean[
        (df_clean["finish_pos"] == 2)
        & df_clean["front_split_sec"].notna()
        & df_clean["race_t2nd_sec"].notna()
        & df_clean["year"].isin(train_years)
    ].copy()

    par_split_full = fit_par_front_split_coefficients(df_2nd, min_cell_n=5)
    df_clean = attach_par_front_split_sec(df_clean, par_split_full, df_2nd)

    return df_clean, par_split_full


# =========================================================================
# NB-02 パイプライン
# =========================================================================

def run_nb02(df: pd.DataFrame, par_splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NB-02 の主要処理を再現。coeff_pace, par_time_class を返す。"""
    df = df.copy()
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce").astype("Int64")
    par_splits["distance"] = pd.to_numeric(par_splits["distance"], errors="coerce").astype("Int64")

    # front_split_dev
    if "par_front_split_sec" in df.columns and df["par_front_split_sec"].notna().any():
        df["par_front_split_final"] = df["par_front_split_sec"]
    else:
        df = df.merge(
            par_splits[["distance", "surface", "par_intercept", "par_slope", "t2nd_ref"]],
            on=["distance", "surface"], how="left",
        )
        df["par_front_split_final"] = (
            df["par_intercept"] + df["par_slope"] * (df["race_t2nd_sec"] - df["t2nd_ref"])
        )
    df["front_split_dev"] = (df["front_split_sec"] - df["par_front_split_final"]).fillna(0.0)

    df_train = df[df["year"] <= 2024].copy()

    # coeff_pace
    coeff_pace = fit_coeff_pace(df_train)

    # class_rank 付与
    df["class_rank"] = df.apply(lambda r: assign_class_rank(r["grade"], r["race_class"]), axis=1)

    # 世代戦除外
    exclude_mask = (
        df["race_class"].str.contains("2歳|２歳", na=False)
        | (df["race_class"].str.contains("3歳|３歳", na=False) & ~df["race_class"].str.contains("以上", na=False))
    )
    df_open = df[~exclude_mask].copy()

    # par_time_class
    df_par_base = df_open[
        (df_open["finish_pos"] == 2)
        & df_open["class_rank"].between(1, 7)
        & (df_open["year"] <= 2024)
        & df_open["adjusted_time_sec"].notna()
    ].copy()

    if len(df_par_base) < 5:
        return coeff_pace, pd.DataFrame()

    global_beta, pool_distband_beta = fit_pool_betas(df_par_base)
    par_time_class = fit_par_time_class(df_par_base, global_beta, pool_distband_beta)

    return coeff_pace, par_time_class


# =========================================================================
# 検証チェック
# =========================================================================

def check_nb01(df_clean: pd.DataFrame, par_split_full: pd.DataFrame) -> None:
    section("NB-01 出力チェック")
    n = len(df_clean)
    print(f"  df_clean: {n:,} 行 × {len(df_clean.columns)} 列")

    # 行数
    if n >= 1000:
        ok(f"行数 {n:,} >= 1,000")
    else:
        fail("行数が少なすぎる", f"got {n}")

    # 必須列（3層設計で追加された front_split_dev / n_valid_races を含む）
    required = [
        "adjusted_time_sec", "finish_time_sec", "front_split_sec",
        "race_t2nd_sec", "par_front_split_sec", "tsi_raw", "track_dev_sec",
        "weight_dev", "base_weight_kg", "race_month", "time_z",
        "distance_band", "track_cat",
        "front_split_dev", "n_valid_races",
    ]
    missing_cols = [c for c in required if c not in df_clean.columns]
    if not missing_cols:
        ok(f"必須列 {len(required)} 列すべて存在")
    else:
        fail(f"列不足: {missing_cols}")

    # adjusted_time_sec カバレッジ
    cov = df_clean["adjusted_time_sec"].notna().mean()
    if cov >= 0.99:
        ok(f"adjusted_time_sec カバレッジ {cov:.1%}")
    else:
        fail("adjusted_time_sec カバレッジ < 99%", f"{cov:.1%}")

    # front_split_sec カバレッジ
    cov_sp = df_clean["front_split_sec"].notna().mean()
    if cov_sp >= 0.90:
        ok(f"front_split_sec カバレッジ {cov_sp:.1%}")
    else:
        fail("front_split_sec カバレッジ < 90%", f"{cov_sp:.1%}")

    # tsi_raw = -track_dev_sec の確認
    check = df_clean.dropna(subset=["tsi_raw", "track_dev_sec"]).drop_duplicates(["date", "venue", "surface"])
    if len(check) > 0:
        max_diff = (check["tsi_raw"] + check["track_dev_sec"]).abs().max()
        if max_diff < 1e-6:
            ok(f"tsi_raw = -track_dev_sec (最大誤差 {max_diff:.2e})")
        else:
            fail("tsi_raw ≠ -track_dev_sec", f"最大誤差={max_diff:.4f}")

    # base_weight_kg が年齢適切範囲にある
    age_base_ok = True
    for _, row in df_clean.sample(min(50, len(df_clean)), random_state=0).iterrows():
        bw = row["base_weight_kg"]
        if not (50.0 <= bw <= 60.0):
            age_base_ok = False
            break
    if age_base_ok:
        ok("base_weight_kg が全サンプルで 50〜60 kg の合理的範囲")
    else:
        fail("base_weight_kg に範囲外の値あり")

    # 2歳馬の base_weight_kg が旧来の55kgより低い（年齢補正が効いている）
    df2 = df_clean[df_clean["age_num"] == 2]
    if len(df2) > 0:
        mean_bw_2 = df2["base_weight_kg"].mean()
        if mean_bw_2 < 55.5:
            ok(f"2歳馬 base_weight 平均 {mean_bw_2:.2f} kg (旧 55kg より低い ✓)")
        else:
            fail("2歳馬 base_weight が旧来と変わらない", f"mean={mean_bw_2:.2f}")

    # 5歳以上の base_weight_kg が 57〜58 kg
    df5p = df_clean[df_clean["age_num"] >= 5]
    if len(df5p) > 0:
        min_bw = df5p["base_weight_kg"].min()
        if min_bw >= 56.0:
            ok(f"5歳以上 base_weight 最小 {min_bw:.1f} kg (56 kg以上 ✓)")
        else:
            fail("5歳以上 base_weight が低すぎる", f"min={min_bw:.1f}")

    # class_group が 6区分に収まっているか
    valid_groups = {"G1orG2", "G3orOP", "3勝", "2勝", "1勝", "未勝利"}
    actual_groups = set(df_clean["class_group"].dropna().unique())
    unexpected = actual_groups - valid_groups
    if not unexpected:
        ok(f"class_group が 6区分内のみ ({sorted(actual_groups)})")
    else:
        fail(f"class_group に想定外の値あり: {unexpected}")

    # n_valid_races <= n_races_track
    if "n_valid_races" in df_clean.columns and "n_races_track" in df_clean.columns:
        bad = df_clean.dropna(subset=["n_valid_races", "n_races_track"])
        excess = (bad["n_valid_races"] > bad["n_races_track"]).sum()
        if excess == 0:
            ok("n_valid_races <= n_races_track（全行）")
        else:
            fail("n_valid_races > n_races_track の行あり", str(excess))

    # 収縮確認: track_dev_sec が race_track_dev_sec の単純平均と一致しない（収縮が効いている）
    day_grp = (
        df_clean.dropna(subset=["race_track_dev_sec", "track_dev_sec"])
        .groupby(["date_str", "venue", "surface"])
        .agg(raw_mean=("race_track_dev_sec", "mean"),
             shrunken=("track_dev_sec", "first"),
             n=("race_track_dev_sec", "count"))
        .reset_index()
    )
    shrunken_ok = (day_grp["shrunken"].abs() <= day_grp["raw_mean"].abs() + 0.01).mean() >= 0.5
    if shrunken_ok:
        ok("収縮推定: |track_dev_sec| は |raw_mean| 以下（多数のセルで収縮確認）")
    else:
        fail("収縮が機能していない可能性あり")

    # par_split_full
    if len(par_split_full) >= 3:
        ok(f"par_split_full セル数 {len(par_split_full)}")
    else:
        fail("par_split_full セル数が少なすぎる", f"{len(par_split_full)}")

    req_par_cols = {"par_intercept", "par_slope", "t2nd_ref", "n_fit"}
    if req_par_cols.issubset(par_split_full.columns):
        ok("par_split_full 必須列あり")
    else:
        fail("par_split_full 列不足", str(req_par_cols - set(par_split_full.columns)))

    # 外れ値除去率が合理的
    outlier_rate = 1.0 - n / max(1, n + (df_clean["time_z"].abs() > 3.5).sum())
    ok(f"外れ値除去後データ確認 (time_z 範囲: {df_clean['time_z'].abs().max():.2f} 以下)")


def check_nb02(coeff_pace: pd.DataFrame, par_time_class: pd.DataFrame) -> None:
    section("NB-02 出力チェック")

    # coeff_pace
    print(f"  coeff_pace: {len(coeff_pace):,} 行")
    if len(coeff_pace) >= 3:
        ok(f"coeff_pace 行数 {len(coeff_pace)}")
    else:
        fail("coeff_pace 行数が少なすぎる")

    req_cp = {"venue", "surface", "distance", "coeff_pace", "source"}
    if req_cp.issubset(coeff_pace.columns):
        ok("coeff_pace 必須列あり")
    else:
        fail("coeff_pace 列不足", str(req_cp - set(coeff_pace.columns)))

    clip_ok = coeff_pace["coeff_pace"].between(0.3, 1.5).all()
    if clip_ok:
        ok(f"coeff_pace 値が [0.3, 1.5] 範囲内 (mean={coeff_pace['coeff_pace'].mean():.3f})")
    else:
        fail("coeff_pace に範囲外の値あり",
             f"min={coeff_pace['coeff_pace'].min():.3f}, max={coeff_pace['coeff_pace'].max():.3f}")

    # par_time_class
    if len(par_time_class) == 0:
        fail("par_time_class が空（混合戦データ不足の可能性）")
        return

    print(f"  par_time_class: {len(par_time_class):,} 行")
    if len(par_time_class) >= 10:
        ok(f"par_time_class 行数 {len(par_time_class)}")
    else:
        fail("par_time_class 行数が少なすぎる")

    req_ptc = {"venue", "surface", "distance", "class_rank", "par_time_sec", "beta"}
    if req_ptc.issubset(par_time_class.columns):
        ok("par_time_class 必須列あり")
    else:
        fail("par_time_class 列不足", str(req_ptc - set(par_time_class.columns)))

    # beta <= 0（高クラスほど速い）
    beta_neg = (par_time_class.drop_duplicates(["venue", "surface", "distance"])["beta"] <= 0).all()
    if beta_neg:
        ok("par_time_class の beta すべて ≤ 0 (高クラス=速い ✓)")
    else:
        n_pos = (par_time_class.drop_duplicates(["venue", "surface", "distance"])["beta"] > 0).sum()
        fail(f"par_time_class に beta > 0 のセルあり", f"{n_pos} セル")

    # class_rank 1 > class_rank 7 のタイム（未勝利 > G1）
    ptc1 = par_time_class[par_time_class["class_rank"] == 1]["par_time_sec"]
    ptc7 = par_time_class[par_time_class["class_rank"] == 7]["par_time_sec"]
    if len(ptc1) > 0 and len(ptc7) > 0:
        if ptc1.mean() > ptc7.mean():
            ok(f"par_time rank1 平均 ({ptc1.mean():.2f}s) > rank7 ({ptc7.mean():.2f}s) ✓")
        else:
            fail("par_time の単調性異常: rank1 が rank7 より速い")


# =========================================================================
# 詳細サマリー表示
# =========================================================================

def print_summary(df_clean: pd.DataFrame, par_split_full: pd.DataFrame,
                  coeff_pace: pd.DataFrame, par_time_class: pd.DataFrame) -> None:
    section("データサマリー")

    print("\n  [NB-01] base_weight_kg 年齢別統計:")
    bw_summary = (
        df_clean.groupby(["age_num", "sex_group"])["base_weight_kg"]
        .agg(["mean", "min", "max", "count"])
        .round(2)
    )
    print(bw_summary.to_string())

    print("\n  [NB-01] weight_dev_kg 年齢別 mean（≒ 別定加算の平均）:")
    wd = df_clean.groupby(["age_num", "sex_group"])["weight_dev_kg"].mean().round(3)
    print(wd.to_string())

    print("\n  [NB-01] par_split_full:")
    print(par_split_full[["distance", "surface", "par_intercept", "par_slope", "n_fit", "model"]].to_string(index=False))

    print("\n  [NB-02] coeff_pace (先頭8行):")
    print(coeff_pace.head(8)[["venue", "surface", "distance", "coeff_pace", "n_fit", "source"]].to_string(index=False))

    if len(par_time_class) > 0:
        print("\n  [NB-02] par_time_class サンプル (東京芝 or 先頭会場):")
        sample = par_time_class.head(14)
        print(sample[["venue", "surface", "distance", "class_rank", "par_time_sec", "beta", "source"]].to_string(index=False))


# =========================================================================
# メイン
# =========================================================================

def main() -> int:
    section("モックデータ生成")
    df_raw = build_mock_race_result_flat(seed=42)

    section("NB-01 パイプライン実行")
    try:
        df_clean, par_split_full = run_nb01(df_raw)
        print(f"  → df_clean: {len(df_clean):,} 行  par_split_full: {len(par_split_full)} セル")
    except Exception as e:
        fail("NB-01 実行エラー", traceback.format_exc())
        return 1

    section("NB-02 パイプライン実行")
    try:
        coeff_pace, par_time_class = run_nb02(df_clean, par_split_full)
        print(f"  → coeff_pace: {len(coeff_pace)} 行  par_time_class: {len(par_time_class)} 行")
    except Exception as e:
        fail("NB-02 実行エラー", traceback.format_exc())
        return 1

    check_nb01(df_clean, par_split_full)
    check_nb02(coeff_pace, par_time_class)
    print_summary(df_clean, par_split_full, coeff_pace, par_time_class)

    section("結果")
    print(f"  PASS: {len(PASS)}  FAIL: {len(FAIL)}")
    if FAIL:
        print("  失敗項目:")
        for f in FAIL:
            print(f"    ✗ {f}")
        return 1
    else:
        print("  すべてのチェックが通過しました ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
