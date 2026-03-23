"""
コース特性 × 血統適性 統合研究エンジン

目的:
  1. 競馬場の物理特性ドメインナレッジ (course_profiles.json) を活用
  2. 血統グループの実成績を競馬場×距離×馬場状態で集計
  3. コース特性ベクトルと血統パフォーマンスの相関を分析
  4. 血統ごとの「最適コース条件」を推定

分析:
  A. 競馬場別 × 種牡馬別 複勝率/勝率マトリクス
  B. コース特性ベクトルと血統成績の相関分析
  C. 種牡馬の「コース適性スコア」算出
  D. 血統×コース条件から最適距離を逆算
  E. 馬場状態×血統の交互作用

Usage:
  python -m research.course_bloodline --years 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("research.course_bloodline")

COURSE_PROFILES_PATH = Path(__file__).parent.parent / "data" / "knowledge" / "course_profiles.json"

TRAIT_KEYS = [
    "stamina_demand", "power_demand", "speed_sustain",
    "acceleration", "agility", "track_bias_inner",
    "front_runner_advantage",
]

MIN_SAMPLES = 15


class CourseBloodlineAnalyzer:
    """コース特性×血統の統合分析エンジン"""

    def __init__(self, output_dir: str = "data/research/course_bloodline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.DataFrame()
        self.profiles: dict[str, Any] = {}
        self._load_course_profiles()

    def _load_course_profiles(self):
        if not COURSE_PROFILES_PATH.exists():
            logger.error("コースプロファイルが見つかりません: %s", COURSE_PROFILES_PATH)
            return
        with open(COURSE_PROFILES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.profiles = data.get("venues", {})
        self.surface_modifiers = data.get("surface_modifiers", {})
        self.dist_bands = data.get("distance_band_characteristics", {})
        logger.info("コースプロファイル読込: %d場", len(self.profiles))

    def _venue_name_to_code(self, name: str) -> str:
        for code, prof in self.profiles.items():
            if prof["name"] == name:
                return code
        return ""

    def _get_trait_vector(self, venue_code: str) -> np.ndarray:
        prof = self.profiles.get(venue_code, {})
        traits = prof.get("traits", {})
        return np.array([traits.get(k, 5) for k in TRAIT_KEYS], dtype=np.float64)

    # ── データロード ──

    def load_from_gcs(self, years: list[str] | None = None):
        from scraper.storage import HybridStorage
        storage = HybridStorage(".")
        if not storage.gcs_enabled:
            logger.error("GCS 未接続")
            return

        if years is None:
            years = storage.list_years("race_result")
        logger.info("対象年度: %s", years)

        rows: list[dict] = []

        for year in years:
            race_ids = storage.list_keys("race_result", year)
            logger.info("  %s: %d レース", year, len(race_ids))

            for race_id in race_ids:
                result_data = storage.load("race_result", race_id)
                if not result_data:
                    continue

                venue_code = race_id[4:6] if len(race_id) >= 6 else ""
                venue_name = result_data.get("venue", "")
                if not venue_code and venue_name:
                    venue_code = self._venue_name_to_code(venue_name)

                distance = result_data.get("distance", 0)
                surface = result_data.get("surface", "")
                track_cond = result_data.get("track_condition", "")
                direction = result_data.get("direction", "")

                shutuba_data = storage.load("race_shutuba", race_id)
                shutuba_map = {}
                if shutuba_data:
                    for e in shutuba_data.get("entries", []):
                        hn = e.get("horse_number", 0)
                        if hn:
                            shutuba_map[hn] = e

                field_size = result_data.get("field_size", 0) or len(
                    result_data.get("entries", [])
                )

                for entry in result_data.get("entries", []):
                    hid = entry.get("horse_id", "")
                    fp = entry.get("finish_position", 0)
                    hn = entry.get("horse_number", 0)
                    if not hid or fp == 0:
                        continue

                    sire = entry.get("sire", "")
                    dam_sire = entry.get("dam_sire", "")

                    se = shutuba_map.get(hn, {})
                    if not sire:
                        sire = se.get("sire", "")
                    if not dam_sire:
                        dam_sire = se.get("dam_sire", "")

                    if not sire:
                        horse_data = storage.load("horse_result", hid)
                        if horse_data:
                            sire = horse_data.get("sire", "")
                            dam_sire = dam_sire or horse_data.get("dam_sire", "")

                    if not sire:
                        continue

                    bracket = entry.get("bracket_number", 0) or se.get("bracket_number", 0)

                    race_month = 0
                    race_date = result_data.get("date", "")
                    if not race_date and len(race_id) >= 4:
                        race_date = race_id[:4]
                    if race_date and len(race_date) >= 6:
                        try:
                            race_month = int(race_date[4:6])
                        except ValueError:
                            pass

                    rows.append({
                        "race_id": race_id,
                        "horse_id": hid,
                        "horse_number": hn,
                        "bracket_number": bracket,
                        "finish_position": fp,
                        "sire": sire,
                        "dam_sire": dam_sire or "不明",
                        "venue_code": venue_code,
                        "venue": venue_name or self.profiles.get(venue_code, {}).get("name", ""),
                        "distance": distance,
                        "surface": surface,
                        "track_condition": track_cond,
                        "direction": direction,
                        "field_size": field_size,
                        "race_month": race_month,
                        "is_top3": 1 if fp <= 3 else 0,
                        "is_win": 1 if fp == 1 else 0,
                    })

        self.df = pd.DataFrame(rows)
        if not self.df.empty:
            self.df["dist_cat"] = self.df["distance"].apply(self._categorize_distance)
            self.df["draw_zone"] = self.df.apply(self._classify_draw_zone, axis=1)
            self.df["grass_type_est"] = self.df.apply(
                lambda r: self._estimate_grass_type(r["venue_code"], r["race_month"], r["surface"]),
                axis=1,
            )
            self.df["first_corner_m"] = self.df.apply(
                lambda r: self._get_first_corner_distance(r["venue_code"], r["distance"], r["surface"]),
                axis=1,
            )
            self.df["fc_band"] = self.df["first_corner_m"].apply(self._classify_first_corner_band)
        logger.info("データ構築完了: %d 出走", len(self.df))

    @staticmethod
    def _categorize_distance(d: int) -> str:
        if d < 1400:
            return "sprint"
        if d < 1800:
            return "mile"
        if d < 2200:
            return "intermediate"
        if d < 2800:
            return "long"
        return "extended"

    @staticmethod
    def _classify_draw_zone(row) -> str:
        hn = row.get("horse_number", 0)
        fs = row.get("field_size", 0)
        if hn <= 0 or fs <= 0:
            return "unknown"
        ratio = hn / fs
        if ratio <= 0.33:
            return "inner"
        if ratio <= 0.66:
            return "middle"
        return "outer"

    def _get_first_corner_distance(self, venue_code: str, distance: int, surface: str) -> int | None:
        """course_profiles.json の draw_bias からレース条件に合致する初角距離を取得"""
        prof = self.profiles.get(venue_code, {})
        draw_bias = prof.get("draw_bias", {})
        if not draw_bias:
            return None

        surface_prefix = "dirt" if surface in ("ダ", "ダート") else "turf"
        best_match: int | None = None
        best_diff = 99999

        for key, entry in draw_bias.items():
            if not key.startswith(surface_prefix):
                continue
            try:
                parts = key.replace(surface_prefix + "_", "").split("_")
                key_dist = int(parts[0])
            except (ValueError, IndexError):
                continue
            diff = abs(key_dist - distance)
            if diff < best_diff:
                best_diff = diff
                fc = entry.get("first_corner_m")
                if fc is not None:
                    best_match = fc
                    best_diff = diff

        if best_diff <= 200:
            return best_match
        return None

    def _classify_first_corner_band(self, fc_m: int | None) -> str:
        """初角距離をバンドに分類"""
        if fc_m is None:
            return "unknown"
        if fc_m <= 250:
            return "very_short"
        if fc_m <= 400:
            return "short"
        if fc_m <= 550:
            return "medium"
        if fc_m <= 700:
            return "long"
        return "very_long"

    def _estimate_grass_type(self, venue_code: str, month: int, surface: str) -> str:
        if surface in ("ダ", "ダート"):
            return "dirt"
        prof = self.profiles.get(venue_code, {})
        gi = prof.get("grass_info", {})
        if not gi:
            return "unknown"

        base = gi.get("base_type", "")
        if base == "洋芝":
            return "洋芝"
        if base == "エクイターフ":
            return "エクイターフ"

        if not gi.get("seasonal_variation", False) or month == 0:
            return base

        for sched in gi.get("schedule", []):
            if month in sched.get("months", []):
                t = sched.get("type", "")
                if "オーバーシード" in t:
                    return "オーバーシード"
                return "野芝"
        return base or "野芝"

    # ── 分析A: 競馬場別×種牡馬別 成績マトリクス ──

    def analyze_venue_sire_matrix(self) -> pd.DataFrame:
        """種牡馬×競馬場の複勝率/勝率ピボット"""
        grp = self.df.groupby(["sire", "venue"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
            win_rate=("is_win", "mean"),
        ).reset_index()
        grp = grp[grp["count"] >= MIN_SAMPLES]

        top_sires = (
            self.df.groupby("sire")["is_top3"].size()
            .nlargest(50).index.tolist()
        )
        filtered = grp[grp["sire"].isin(top_sires)]

        pivot = filtered.pivot_table(
            index="sire", columns="venue",
            values="top3_rate", fill_value=np.nan,
        )

        venue_order = [
            self.profiles[c]["name"]
            for c in sorted(self.profiles.keys())
            if self.profiles[c]["name"] in pivot.columns
        ]
        pivot = pivot[[c for c in venue_order if c in pivot.columns]]

        path = self.output_dir / "venue_sire_top3rate.csv"
        pivot.to_csv(path, encoding="utf-8-sig")
        logger.info("保存: %s", path)
        return pivot

    # ── 分析B: コース特性ベクトル×血統成績の相関 ──

    def analyze_trait_bloodline_correlation(self) -> pd.DataFrame:
        """
        各種牡馬の「競馬場別複勝率」と「コース特性値」の相関を計算。
        相関が高い特性 = その種牡馬が得意/苦手とする要素。
        """
        venue_sire = self.df.groupby(["sire", "venue_code"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
        ).reset_index()
        venue_sire = venue_sire[venue_sire["count"] >= MIN_SAMPLES]

        top_sires = (
            self.df.groupby("sire")["is_top3"].size()
            .nlargest(80).index.tolist()
        )
        venue_sire = venue_sire[venue_sire["sire"].isin(top_sires)]

        trait_matrix = {}
        for vc in sorted(self.profiles.keys()):
            trait_matrix[vc] = self._get_trait_vector(vc)

        results = []
        for sire in top_sires:
            sire_data = venue_sire[venue_sire["sire"] == sire]
            if len(sire_data) < 3:
                continue

            rates = []
            traits_list = []
            for _, row in sire_data.iterrows():
                vc = row["venue_code"]
                if vc in trait_matrix:
                    rates.append(row["top3_rate"])
                    traits_list.append(trait_matrix[vc])

            if len(rates) < 3:
                continue

            rates_arr = np.array(rates)
            traits_arr = np.array(traits_list)

            corr_row = {"sire": sire, "n_venues": len(rates)}
            for j, trait_name in enumerate(TRAIT_KEYS):
                trait_col = traits_arr[:, j]
                if np.std(trait_col) > 0 and np.std(rates_arr) > 0:
                    corr = np.corrcoef(rates_arr, trait_col)[0, 1]
                    corr_row[trait_name] = round(corr, 3)
                else:
                    corr_row[trait_name] = 0.0
            results.append(corr_row)

        corr_df = pd.DataFrame(results)

        path = self.output_dir / "sire_trait_correlation.csv"
        corr_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path)
        return corr_df

    # ── 分析C: 種牡馬コース適性スコア ──

    def compute_sire_course_aptitude(self) -> pd.DataFrame:
        """
        種牡馬ごとに、各コース特性への適性スコアを算出。
        特性値×複勝率の加重平均で「得意な特性」を定量化。
        """
        venue_sire = self.df.groupby(["sire", "venue_code"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
            win_rate=("is_win", "mean"),
        ).reset_index()
        venue_sire = venue_sire[venue_sire["count"] >= MIN_SAMPLES // 2]

        top_sires = (
            self.df.groupby("sire")["is_top3"].size()
            .nlargest(100).index.tolist()
        )

        results = []
        for sire in top_sires:
            sire_data = venue_sire[venue_sire["sire"] == sire]
            if sire_data.empty:
                continue

            weighted_traits = np.zeros(len(TRAIT_KEYS))
            total_weight = 0
            total_count = 0
            overall_top3 = 0.0

            for _, row in sire_data.iterrows():
                vc = row["venue_code"]
                if vc not in self.profiles:
                    continue
                traits = self._get_trait_vector(vc)
                weight = row["count"]
                performance = row["top3_rate"]

                weighted_traits += traits * performance * weight
                total_weight += weight * performance
                total_count += row["count"]
                overall_top3 += row["top3_rate"] * row["count"]

            if total_weight == 0 or total_count == 0:
                continue

            aptitude = weighted_traits / total_weight
            overall_top3 /= total_count

            row_data = {
                "sire": sire,
                "total_runs": total_count,
                "overall_top3_rate": round(overall_top3, 4),
            }
            for j, tk in enumerate(TRAIT_KEYS):
                row_data[f"apt_{tk}"] = round(aptitude[j], 2)

            best_trait_idx = np.argmax(aptitude)
            row_data["best_trait"] = TRAIT_KEYS[best_trait_idx]
            row_data["best_trait_score"] = round(aptitude[best_trait_idx], 2)

            results.append(row_data)

        apt_df = pd.DataFrame(results).sort_values("total_runs", ascending=False)

        path = self.output_dir / "sire_course_aptitude.csv"
        apt_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path)
        return apt_df

    # ── 分析D: 血統×コース条件 → 最適距離推定 ──

    def estimate_optimal_conditions(self) -> pd.DataFrame:
        """
        種牡馬ごとに、コース特性を加味した「最適レース条件」を推定。
        距離×競馬場×馬場の複勝率から、最も適した条件を逆算する。
        """
        grp = self.df.groupby([
            "sire", "venue", "dist_cat", "surface",
        ]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
        ).reset_index()
        grp = grp[grp["count"] >= MIN_SAMPLES // 2]

        top_sires = (
            self.df.groupby("sire")["is_top3"].size()
            .nlargest(60).index.tolist()
        )
        grp = grp[grp["sire"].isin(top_sires)]

        results = []
        for sire in top_sires:
            sire_data = grp[grp["sire"] == sire]
            if sire_data.empty:
                continue

            best = sire_data.loc[sire_data["top3_rate"].idxmax()]

            total = self.df[self.df["sire"] == sire]
            dist_grp = total.groupby("dist_cat").agg(
                count=("is_top3", "size"),
                top3_rate=("is_top3", "mean"),
            ).reset_index()
            if not dist_grp.empty:
                best_dist = dist_grp.loc[dist_grp["top3_rate"].idxmax()]
            else:
                best_dist = pd.Series({"dist_cat": "", "top3_rate": 0})

            venue_grp = total.groupby("venue").agg(
                count=("is_top3", "size"),
                top3_rate=("is_top3", "mean"),
            ).reset_index()
            venue_grp = venue_grp[venue_grp["count"] >= MIN_SAMPLES]
            if not venue_grp.empty:
                best_venue = venue_grp.loc[venue_grp["top3_rate"].idxmax()]
            else:
                best_venue = pd.Series({"venue": "", "top3_rate": 0})

            cond_grp = total.groupby("track_condition").agg(
                count=("is_top3", "size"),
                top3_rate=("is_top3", "mean"),
            ).reset_index()
            cond_grp = cond_grp[cond_grp["count"] >= MIN_SAMPLES // 2]
            if not cond_grp.empty:
                best_cond = cond_grp.loc[cond_grp["top3_rate"].idxmax()]
            else:
                best_cond = pd.Series({"track_condition": "", "top3_rate": 0})

            avg_dist = total[total["is_top3"] == 1]["distance"].mean()
            avg_dist = round(avg_dist) if not np.isnan(avg_dist) else 0

            results.append({
                "sire": sire,
                "total_runs": len(total),
                "overall_top3": round(total["is_top3"].mean(), 4),
                "best_distance_cat": best_dist.get("dist_cat", ""),
                "best_distance_top3": round(float(best_dist.get("top3_rate", 0)), 4),
                "weighted_best_dist_m": int(avg_dist),
                "best_venue": best_venue.get("venue", ""),
                "best_venue_top3": round(float(best_venue.get("top3_rate", 0)), 4),
                "best_track_cond": best_cond.get("track_condition", ""),
                "best_cond_top3": round(float(best_cond.get("top3_rate", 0)), 4),
                "best_combo": f"{best.get('venue', '')} {best.get('surface', '')} {best.get('dist_cat', '')}",
                "best_combo_top3": round(float(best.get("top3_rate", 0)), 4),
            })

        opt_df = pd.DataFrame(results)
        path = self.output_dir / "sire_optimal_conditions.csv"
        opt_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path)
        return opt_df

    # ── 分析E: 馬場状態×血統の交互作用 ──

    def analyze_track_condition_interaction(self) -> pd.DataFrame:
        """馬場状態ごとの種牡馬成績変動を分析。"""
        grp = self.df.groupby(["sire", "surface", "track_condition"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
        ).reset_index()
        grp = grp[grp["count"] >= MIN_SAMPLES]

        top_sires = (
            self.df.groupby("sire")["is_top3"].size()
            .nlargest(40).index.tolist()
        )
        grp = grp[grp["sire"].isin(top_sires)]

        baseline = self.df.groupby("sire")["is_top3"].mean().to_dict()

        grp["baseline_top3"] = grp["sire"].map(baseline)
        grp["delta"] = grp["top3_rate"] - grp["baseline_top3"]
        grp["condition_label"] = grp["surface"] + "・" + grp["track_condition"]

        path = self.output_dir / "track_condition_interaction.csv"
        grp.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path)
        return grp

    # ── 分析F: 枠順×血統の交互作用 ──

    def analyze_draw_bloodline_interaction(self) -> pd.DataFrame:
        """
        枠順ゾーン (内/中/外) × 種牡馬の複勝率。
        コースごとに枠順バイアスを受けやすい/受けにくい血統を検出する。
        """
        df = self.df[self.df["draw_zone"] != "unknown"].copy()

        grp = df.groupby(["sire", "venue", "dist_cat", "draw_zone"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
        ).reset_index()
        grp = grp[grp["count"] >= MIN_SAMPLES // 2]

        top_sires = (
            df.groupby("sire")["is_top3"].size()
            .nlargest(50).index.tolist()
        )
        grp = grp[grp["sire"].isin(top_sires)]

        baseline = df.groupby(["sire", "venue", "dist_cat"])["is_top3"].mean()
        baseline_map = baseline.to_dict()

        grp["baseline_top3"] = grp.apply(
            lambda r: baseline_map.get((r["sire"], r["venue"], r["dist_cat"]), 0), axis=1
        )
        grp["delta"] = grp["top3_rate"] - grp["baseline_top3"]

        path = self.output_dir / "draw_bloodline_interaction.csv"
        grp.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path)

        summary = df.groupby(["sire", "draw_zone"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
        ).reset_index()
        summary = summary[
            (summary["count"] >= MIN_SAMPLES) &
            (summary["sire"].isin(top_sires))
        ]

        pivot = summary.pivot_table(
            index="sire", columns="draw_zone",
            values="top3_rate", fill_value=np.nan,
        )
        for zone in ["inner", "middle", "outer"]:
            if zone not in pivot.columns:
                pivot[zone] = np.nan
        pivot = pivot[["inner", "middle", "outer"]]
        pivot["inner_outer_gap"] = pivot["inner"] - pivot["outer"]
        pivot = pivot.sort_values("inner_outer_gap", ascending=False)

        path2 = self.output_dir / "sire_draw_bias_summary.csv"
        pivot.to_csv(path2, encoding="utf-8-sig")
        logger.info("保存: %s", path2)

        return pivot

    # ── 分析H: 初角距離×枠順×血統 交互作用 ──

    def analyze_first_corner_draw_interaction(self) -> pd.DataFrame:
        """
        初角までの距離帯 (very_short/short/medium/long/very_long)
        × 枠順ゾーン (inner/middle/outer)
        × 種牡馬 の複勝率を分析。

        初角距離が短いコースで外枠でも走れる血統 = 先行力・機動力が高い。
        初角距離が長いコースでしか外枠で走れない = 追込み専科の証拠。
        """
        df = self.df[
            (self.df["fc_band"] != "unknown") &
            (self.df["draw_zone"] != "unknown")
        ].copy()

        if df.empty:
            logger.warning("初角距離データなし")
            return pd.DataFrame()

        grp = df.groupby(["sire", "fc_band", "draw_zone"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
            win_rate=("is_win", "mean"),
        ).reset_index()
        grp = grp[grp["count"] >= MIN_SAMPLES // 2]

        top_sires = (
            df.groupby("sire")["is_top3"].size()
            .nlargest(50).index.tolist()
        )
        grp = grp[grp["sire"].isin(top_sires)]

        path = self.output_dir / "first_corner_draw_interaction.csv"
        grp.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path)

        fc_bands = ["very_short", "short", "medium", "long", "very_long"]
        fc_labels = {
            "very_short": "~250m", "short": "250~400m",
            "medium": "400~550m", "long": "550~700m", "very_long": "700m~",
        }

        summary_rows = []
        for sire in top_sires:
            sire_data = df[df["sire"] == sire]
            if len(sire_data) < MIN_SAMPLES:
                continue

            row: dict[str, Any] = {"sire": sire, "total": len(sire_data)}

            for band in fc_bands:
                for zone in ["inner", "outer"]:
                    subset = sire_data[
                        (sire_data["fc_band"] == band) &
                        (sire_data["draw_zone"] == zone)
                    ]
                    key = f"{band}_{zone}"
                    if len(subset) >= MIN_SAMPLES // 3:
                        row[f"{key}_rate"] = round(subset["is_top3"].mean(), 4)
                        row[f"{key}_n"] = len(subset)
                    else:
                        row[f"{key}_rate"] = np.nan
                        row[f"{key}_n"] = len(subset)

            for band in fc_bands:
                inner_r = row.get(f"{band}_inner_rate")
                outer_r = row.get(f"{band}_outer_rate")
                if inner_r is not None and outer_r is not None and not (
                    np.isnan(inner_r) or np.isnan(outer_r)
                ):
                    row[f"{band}_gap"] = round(inner_r - outer_r, 4)
                else:
                    row[f"{band}_gap"] = np.nan

            vs_gap = row.get("very_short_gap")
            l_gap = row.get("long_gap")
            if vs_gap is not None and l_gap is not None and not (
                np.isnan(vs_gap) or np.isnan(l_gap)
            ):
                row["draw_sensitivity"] = round(vs_gap - l_gap, 4)
            else:
                row["draw_sensitivity"] = np.nan

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty and "draw_sensitivity" in summary_df.columns:
            summary_df = summary_df.sort_values("draw_sensitivity", ascending=False, na_position="last")

        path2 = self.output_dir / "first_corner_draw_summary.csv"
        summary_df.to_csv(path2, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path2)

        return summary_df

    # ── 分析G: 芝種別×血統パフォーマンス ──

    def analyze_grass_type_bloodline(self) -> pd.DataFrame:
        """
        芝種 (洋芝/野芝/オーバーシード/エクイターフ) 別の種牡馬成績。
        同じ種牡馬でも芝種によってパフォーマンスが大きく変わるケースを検出。
        """
        df = self.df[
            (self.df["grass_type_est"] != "unknown") &
            (self.df["grass_type_est"] != "dirt")
        ].copy()

        if df.empty:
            logger.warning("芝データなし")
            return pd.DataFrame()

        grp = df.groupby(["sire", "grass_type_est"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
            win_rate=("is_win", "mean"),
        ).reset_index()
        grp = grp[grp["count"] >= MIN_SAMPLES]

        top_sires = (
            df.groupby("sire")["is_top3"].size()
            .nlargest(60).index.tolist()
        )
        grp = grp[grp["sire"].isin(top_sires)]

        baseline = df.groupby("sire")["is_top3"].mean().to_dict()
        grp["baseline_top3"] = grp["sire"].map(baseline)
        grp["delta"] = grp["top3_rate"] - grp["baseline_top3"]

        path = self.output_dir / "grass_type_bloodline.csv"
        grp.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", path)

        pivot = grp.pivot_table(
            index="sire", columns="grass_type_est",
            values="top3_rate", fill_value=np.nan,
        )
        for gt in ["洋芝", "野芝", "オーバーシード", "エクイターフ"]:
            if gt not in pivot.columns:
                pivot[gt] = np.nan

        existing = [c for c in ["洋芝", "野芝", "オーバーシード", "エクイターフ"] if c in pivot.columns]
        pivot = pivot[existing]

        if "洋芝" in pivot.columns and "野芝" in pivot.columns:
            pivot["洋芝_野芝_gap"] = pivot["洋芝"] - pivot["野芝"]
        if "オーバーシード" in pivot.columns and "野芝" in pivot.columns:
            pivot["OS_野芝_gap"] = pivot["オーバーシード"] - pivot["野芝"]

        path2 = self.output_dir / "sire_grass_type_summary.csv"
        pivot.to_csv(path2, encoding="utf-8-sig")
        logger.info("保存: %s", path2)

        return pivot

    # ── 統合レポート ──

    def generate_report(self) -> str:
        logger.info("=" * 60)
        logger.info("コース特性 × 血統適性 統合分析")
        logger.info("=" * 60)

        logger.info("[1/7] 競馬場×種牡馬 成績マトリクス")
        venue_sire = self.analyze_venue_sire_matrix()

        logger.info("[2/7] コース特性×血統 相関分析")
        trait_corr = self.analyze_trait_bloodline_correlation()

        logger.info("[3/7] 種牡馬コース適性スコア")
        aptitude = self.compute_sire_course_aptitude()

        logger.info("[4/7] 最適レース条件推定")
        optimal = self.estimate_optimal_conditions()

        logger.info("[5/7] 馬場状態×血統 交互作用")
        track_cond = self.analyze_track_condition_interaction()

        logger.info("[6/8] 枠順×血統 交互作用")
        draw_bl = self.analyze_draw_bloodline_interaction()

        logger.info("[7/8] 芝種別×血統パフォーマンス")
        grass_bl = self.analyze_grass_type_bloodline()

        logger.info("[8/8] 初角距離×枠順×血統 交互作用")
        fc_draw = self.analyze_first_corner_draw_interaction()

        profiles_out = []
        for code in sorted(self.profiles.keys()):
            p = self.profiles[code]
            gi = p.get("grass_info", {})
            row = {"code": code, "name": p["name"], "direction": p["direction"]}
            row["grass_base"] = gi.get("base_type", p.get("grass_type", ""))
            row["seasonal_variation"] = gi.get("seasonal_variation", False)
            row["straight_m"] = p.get("turf_straight_m", 0)
            row["circumference_m"] = p.get("turf_circumference_m", 0)
            row["slope_height_m"] = p.get("slope_height_m", 0)
            row["slope_type"] = p.get("slope_type", "")
            row["course_shape"] = p.get("course_shape", "")
            rp = p.get("rail_positions", {})
            row["rail_positions_available"] = ",".join(rp.get("available", []))
            db = p.get("draw_bias", {})
            fc_list = []
            for dk, dv in db.items():
                fc_val = dv.get("first_corner_m")
                if fc_val is not None:
                    fc_list.append(f"{dk}:{fc_val}m")
            row["first_corner_distances"] = "; ".join(fc_list)
            for tk in TRAIT_KEYS:
                row[tk] = p.get("traits", {}).get(tk, 5)
            profiles_out.append(row)

        profiles_df = pd.DataFrame(profiles_out)
        profiles_df.to_csv(
            self.output_dir / "course_profiles_summary.csv",
            index=False, encoding="utf-8-sig",
        )

        report_path = self.output_dir / "course_bloodline_report.html"
        report_path.write_text(
            self._build_report(venue_sire, trait_corr, aptitude, optimal, track_cond),
            encoding="utf-8",
        )
        logger.info("レポート: %s", report_path)
        return str(report_path)

    def _build_report(self, venue_sire, trait_corr, aptitude, optimal, track_cond) -> str:
        n_records = len(self.df)
        n_sires = self.df["sire"].nunique() if not self.df.empty else 0
        n_venues = self.df["venue"].nunique() if not self.df.empty else 0

        # --- venue radar data ---
        venue_profiles_js = json.dumps([
            {
                "name": self.profiles[c]["name"],
                "traits": {k: self.profiles[c]["traits"][k] for k in TRAIT_KEYS},
                "straight": self.profiles[c].get("turf_straight_m", 0),
                "slope": self.profiles[c].get("slope_height_m", 0),
                "shape": self.profiles[c].get("course_shape", ""),
            }
            for c in sorted(self.profiles.keys())
        ], ensure_ascii=False)

        # --- venue sire heatmap ---
        hm_rows = ""
        venues_list = list(venue_sire.columns) if not venue_sire.empty else []
        for sire in list(venue_sire.index)[:35]:
            cells = f"<td><strong>{sire}</strong></td>"
            for v in venues_list:
                val = venue_sire.loc[sire, v]
                if pd.isna(val):
                    cells += "<td>-</td>"
                else:
                    pct = round(val * 100, 1)
                    cls = "clr-g" if pct >= 35 else ("clr-y" if pct >= 25 else "clr-r")
                    cells += f'<td><span class="hcell {cls}">{pct}%</span></td>'
            hm_rows += f"<tr>{cells}</tr>\n"

        venue_headers = "".join(f"<th>{v}</th>" for v in venues_list)

        # --- aptitude table ---
        apt_rows = ""
        trait_labels = {
            "stamina_demand": "スタミナ", "power_demand": "パワー",
            "speed_sustain": "持続力", "acceleration": "瞬発力",
            "agility": "器用さ", "track_bias_inner": "内枠○",
            "front_runner_advantage": "先行○",
        }
        for _, r in aptitude.head(40).iterrows():
            best_t = trait_labels.get(r.get("best_trait", ""), "")
            cells = (
                f"<td><strong>{r['sire']}</strong></td>"
                f"<td>{int(r['total_runs']):,}</td>"
                f"<td>{r['overall_top3_rate']*100:.1f}%</td>"
            )
            for tk in TRAIT_KEYS:
                v = r.get(f"apt_{tk}", 5)
                w = v / 10 * 100
                cells += f'<td><div class="mini-bar"><div style="width:{w}%;background:var(--accent)"></div></div><span class="mini-val">{v:.1f}</span></td>'
            cells += f"<td><span class='best-badge'>{best_t}</span></td>"
            apt_rows += f"<tr>{cells}</tr>\n"

        # --- optimal conditions ---
        opt_rows = ""
        for _, r in optimal.head(40).iterrows():
            opt_rows += (
                f"<tr><td><strong>{r['sire']}</strong></td>"
                f"<td>{int(r['total_runs']):,}</td>"
                f"<td>{r['overall_top3']*100:.1f}%</td>"
                f"<td>{r['best_venue']}</td>"
                f"<td>{r['best_venue_top3']*100:.1f}%</td>"
                f"<td>{r['best_distance_cat']}</td>"
                f"<td>{r['weighted_best_dist_m']:,}m</td>"
                f"<td>{r['best_track_cond']}</td>"
                f"<td style='color:var(--accent)'>{r['best_combo']}</td></tr>\n"
            )

        trait_th = "".join(f"<th title='{tk}'>{trait_labels.get(tk, tk)}</th>" for tk in TRAIT_KEYS)

        return f"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>コース特性×血統 | ML-AutoPilot Keiba</title>
<style>
:root{{--bg:#0a0e17;--sf:#131926;--sf2:#1a2235;--sf3:#212d42;--bd:#243049;--tx:#c8d6e5;--txd:#6b7d95;--ac:#a78bfa;--acd:rgba(167,139,250,.15);--bl:#3b82f6;--gn:#22c55e;--yl:#f59e0b;--rd:#ef4444;--cy:#06b6d4;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--tx);padding:24px 32px;max-width:1600px;margin:0 auto;}}
a{{color:var(--ac);text-decoration:none;}}a:hover{{text-decoration:underline;}}
.back{{font-size:13px;color:var(--txd);margin-bottom:16px;display:inline-block;}}
h1{{font-size:26px;font-weight:700;margin-bottom:6px;}}
h2{{font-size:18px;font-weight:600;margin:28px 0 12px;border-bottom:1px solid var(--bd);padding-bottom:6px;}}
h3{{font-size:14px;color:var(--txd);margin-bottom:8px;}}
.sub{{font-size:13px;color:var(--txd);margin-bottom:20px;}}
.card{{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:20px;margin-bottom:16px;}}
.stats{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px;}}
.sbox{{background:var(--sf2);border-radius:10px;padding:14px 18px;flex:1;min-width:120px;}}
.sbox .l{{font-size:10px;color:var(--txd);text-transform:uppercase;letter-spacing:.5px;}}
.sbox .v{{font-size:20px;font-weight:700;}}
table{{width:100%;border-collapse:collapse;font-size:12px;}}
th{{text-align:left;font-size:10px;color:var(--txd);text-transform:uppercase;letter-spacing:.5px;padding:8px 6px;border-bottom:1px solid var(--bd);}}
td{{padding:7px 6px;border-bottom:1px solid rgba(36,48,73,.3);}}
tr:hover{{background:rgba(167,139,250,.03);}}
.hcell{{display:inline-block;min-width:44px;padding:2px 6px;border-radius:4px;font-weight:600;text-align:center;font-size:11px;}}
.clr-g{{background:rgba(34,197,94,.12);color:var(--gn);}}
.clr-y{{background:rgba(245,158,11,.1);color:var(--yl);}}
.clr-r{{background:rgba(239,68,68,.08);color:var(--rd);}}
.mini-bar{{height:10px;background:var(--sf3);border-radius:3px;overflow:hidden;width:60px;display:inline-block;vertical-align:middle;}}
.mini-bar>div{{height:100%;border-radius:3px;}}
.mini-val{{font-size:10px;color:var(--txd);margin-left:4px;}}
.best-badge{{font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;background:var(--acd);color:var(--ac);}}
.radar-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:16px;}}
.radar-card{{background:var(--sf2);border-radius:10px;padding:16px;}}
.radar-card h4{{font-size:14px;margin-bottom:8px;}}
.bar-row{{display:flex;align-items:center;gap:6px;margin:3px 0;}}
.bar-row .bl{{font-size:10px;color:var(--txd);min-width:50px;}}
.bar-bg{{flex:1;height:12px;background:var(--sf3);border-radius:3px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:3px;transition:width .4s;}}
.bar-row .bv{{font-size:10px;min-width:24px;text-align:right;}}
.scroll-x{{overflow-x:auto;}}
</style></head><body>
<a href="/bloodline" class="back">← 血統研究</a>
<h1>コース特性 × 血統適性</h1>
<p class="sub">出走数: {n_records:,} | 種牡馬: {n_sires} | 競馬場: {n_venues}</p>

<div class="stats">
<div class="sbox"><div class="l">出走データ</div><div class="v">{n_records:,}</div></div>
<div class="sbox"><div class="l">種牡馬数</div><div class="v">{n_sires}</div></div>
<div class="sbox"><div class="l">分析対象場</div><div class="v">{n_venues}</div></div>
</div>

<h2>1. 競馬場プロファイル — コース特性レーダー</h2>
<div class="card">
<h3>各競馬場の物理特性を7軸で定量化</h3>
<div class="radar-grid" id="radarGrid"></div>
</div>

<h2>2. 競馬場 × 種牡馬 複勝率ヒートマップ</h2>
<div class="card"><div class="scroll-x">
<table><thead><tr><th>種牡馬</th>{venue_headers}</tr></thead>
<tbody>{hm_rows}</tbody></table>
</div></div>

<h2>3. 種牡馬コース適性スコア</h2>
<div class="card">
<h3>コース特性×複勝率の加重平均で「得意な特性」を定量化</h3>
<div class="scroll-x">
<table><thead><tr><th>種牡馬</th><th>出走</th><th>複勝率</th>{trait_th}<th>最適特性</th></tr></thead>
<tbody>{apt_rows}</tbody></table>
</div></div>

<h2>4. 最適レース条件推定</h2>
<div class="card">
<h3>距離×競馬場×馬場状態の最適組み合わせ</h3>
<div class="scroll-x">
<table><thead><tr><th>種牡馬</th><th>出走</th><th>複勝率</th><th>最適場</th><th>場複勝</th><th>最適距離</th><th>加重距離</th><th>最適馬場</th><th>最強条件</th></tr></thead>
<tbody>{opt_rows}</tbody></table>
</div></div>

<script>
const venues = {venue_profiles_js};
const traitLabels = {{stamina_demand:'スタミナ',power_demand:'パワー',speed_sustain:'持続力',acceleration:'瞬発力',agility:'器用さ',track_bias_inner:'内枠○',front_runner_advantage:'先行○'}};
const traitColors = {{stamina_demand:'#ef4444',power_demand:'#f59e0b',speed_sustain:'#22c55e',acceleration:'#3b82f6',agility:'#a78bfa',track_bias_inner:'#06b6d4',front_runner_advantage:'#ec4899'}};

const grid = document.getElementById('radarGrid');
for (const v of venues) {{
  let bars = '';
  for (const [k, label] of Object.entries(traitLabels)) {{
    const val = v.traits[k] || 0;
    const w = val / 10 * 100;
    const color = traitColors[k] || 'var(--ac)';
    bars += `<div class="bar-row"><span class="bl">${{label}}</span><div class="bar-bg"><div class="bar-fill" style="width:${{w}}%;background:${{color}}"></div></div><span class="bv">${{val}}</span></div>`;
  }}
  grid.innerHTML += `<div class="radar-card"><h4>${{v.name}}</h4><div style="font-size:11px;color:var(--txd);margin-bottom:8px">${{v.shape}} / 直線${{v.straight}}m / 坂${{v.slope}}m</div>${{bars}}</div>`;
}}
</script>
</body></html>"""


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="コース特性×血統適性 研究")
    parser.add_argument("--years", nargs="*", default=None)
    parser.add_argument("--output-dir", default="data/research/course_bloodline")
    args = parser.parse_args()

    analyzer = CourseBloodlineAnalyzer(output_dir=args.output_dir)
    analyzer.load_from_gcs(years=args.years)

    if analyzer.df.empty:
        logger.error("データが空です")
        sys.exit(1)

    report_path = analyzer.generate_report()
    print(f"\nレポート: {report_path}")


if __name__ == "__main__":
    main()
