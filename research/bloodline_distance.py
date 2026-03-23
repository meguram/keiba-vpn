"""
血統 × 距離適性 研究コード

目的:
  種牡馬 (sire) および母父 (dam_sire) の血統構成が距離適性にどの程度影響するかを
  統計的に検証し、可視化する。

分析内容:
  1. 種牡馬別・距離カテゴリ別の複勝率ヒートマップ
  2. 種牡馬の「ベスト距離」推定 (加重平均)
  3. 父×母父の組み合わせによる距離適性マトリクス
  4. 距離カテゴリ間の血統類似度 (コサイン類似度)
  5. 血統ベクトルによる距離クラスタリング
  6. 血統特徴量の予測寄与度分析

Usage:
  # GCS データから分析 (推奨)
  python -m research.bloodline_distance --source gcs --years 2020 2021 2022 2023 2024 2025

  # CSV 特徴量ファイルから分析
  python -m research.bloodline_distance --source csv --features-dir data/features

  # 結果は data/research/bloodline/ に保存される
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("research.bloodline")

DISTANCE_CATEGORIES = {
    "sprint": (0, 1400),
    "mile": (1400, 1800),
    "intermediate": (1800, 2200),
    "long": (2200, 2800),
    "extended": (2800, 9999),
}

DIST_CAT_LABELS = {
    "sprint": "短距離 (〜1400m)",
    "mile": "マイル (1400〜1800m)",
    "intermediate": "中距離 (1800〜2200m)",
    "long": "長距離 (2200〜2800m)",
    "extended": "超長距離 (2800m〜)",
}

MIN_SAMPLE_SIZE = 20


def categorize_distance(dist: int) -> str:
    for cat, (lo, hi) in DISTANCE_CATEGORIES.items():
        if lo <= dist < hi:
            return cat
    return "extended"


class BloodlineDistanceAnalyzer:
    """血統と距離適性の関係を分析する。"""

    def __init__(self, output_dir: str = "data/research/bloodline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df: pd.DataFrame = pd.DataFrame()

    def load_from_gcs(self, years: list[str] | None = None):
        """GCS からレース結果 + 馬データを読み込み、分析用 DataFrame を構築する。"""
        from scraper.storage import HybridStorage
        storage = HybridStorage(".")
        if not storage.gcs_enabled:
            logger.error("GCS 未接続")
            return

        if years is None:
            years = storage.list_years("race_result")
        logger.info("対象年度: %s", years)

        rows: list[dict] = []
        total_races = 0

        for year in years:
            race_ids = storage.list_keys("race_result", year)
            logger.info("  %s: %d レース", year, len(race_ids))

            for race_id in race_ids:
                result_data = storage.load("race_result", race_id)
                if not result_data:
                    continue

                shutuba_data = storage.load("race_shutuba", race_id)

                distance = result_data.get("distance", 0)
                surface = result_data.get("surface", "")
                venue = result_data.get("venue", "")

                shutuba_map = {}
                if shutuba_data:
                    for e in shutuba_data.get("entries", []):
                        hn = e.get("horse_number", 0)
                        if hn:
                            shutuba_map[hn] = e

                for entry in result_data.get("entries", []):
                    hn = entry.get("horse_number", 0)
                    hid = entry.get("horse_id", "")
                    fp = entry.get("finish_position", 0)
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

                    field_size = result_data.get("field_size", 0) or len(
                        result_data.get("entries", [])
                    )

                    rows.append({
                        "race_id": race_id,
                        "horse_id": hid,
                        "horse_number": hn,
                        "finish_position": fp,
                        "sire": sire,
                        "dam_sire": dam_sire or "不明",
                        "distance": distance,
                        "distance_cat": categorize_distance(distance),
                        "surface": surface,
                        "venue": venue,
                        "field_size": field_size,
                        "is_top3": 1 if fp <= 3 else 0,
                        "is_win": 1 if fp == 1 else 0,
                    })

                total_races += 1

        self.df = pd.DataFrame(rows)
        logger.info(
            "データ構築完了: %d レース, %d 出走, %d ユニーク種牡馬, %d ユニーク母父",
            total_races, len(self.df),
            self.df["sire"].nunique(), self.df["dam_sire"].nunique(),
        )

    def load_from_csv(self, features_dir: str = "data/features"):
        """既存の特徴量 CSV から読み込む。"""
        p = Path(features_dir)
        csvs = list(p.glob("*.csv"))
        if not csvs:
            logger.error("CSV ファイルが見つかりません: %s", features_dir)
            return

        frames = [pd.read_csv(c) for c in csvs]
        df = pd.concat(frames, ignore_index=True)

        required = ["sire", "distance", "finish_position"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("必要なカラムがありません: %s", missing)
            return

        df["distance_cat"] = df["distance"].apply(categorize_distance)
        df["is_top3"] = (df["finish_position"].between(1, 3)).astype(int)
        df["is_win"] = (df["finish_position"] == 1).astype(int)
        if "dam_sire" not in df.columns:
            df["dam_sire"] = "不明"

        self.df = df
        logger.info("CSV データ読込: %d 行", len(df))

    # ── 分析1: 種牡馬別・距離カテゴリ別の複勝率ヒートマップ ──

    def analyze_sire_distance_top3rate(self, min_samples: int = MIN_SAMPLE_SIZE) -> pd.DataFrame:
        """種牡馬 × 距離カテゴリの複勝率 pivot を作成する。"""
        grp = self.df.groupby(["sire", "distance_cat"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
            win_rate=("is_win", "mean"),
        ).reset_index()

        grp = grp[grp["count"] >= min_samples]

        total_by_sire = self.df.groupby("sire")["is_top3"].agg(["size", "mean"])
        total_by_sire.columns = ["total_count", "overall_top3_rate"]
        total_by_sire = total_by_sire[total_by_sire["total_count"] >= min_samples * 2]

        top_sires = total_by_sire.nlargest(50, "total_count").index.tolist()
        grp_top = grp[grp["sire"].isin(top_sires)]

        pivot = grp_top.pivot_table(
            index="sire", columns="distance_cat",
            values="top3_rate", fill_value=0,
        )
        cat_order = [c for c in DISTANCE_CATEGORIES if c in pivot.columns]
        pivot = pivot[cat_order]

        csv_path = self.output_dir / "sire_distance_top3rate.csv"
        pivot.to_csv(csv_path, encoding="utf-8-sig")
        logger.info("保存: %s", csv_path)

        return pivot

    # ── 分析2: 種牡馬のベスト距離推定 ──

    def estimate_sire_best_distance(self, min_samples: int = MIN_SAMPLE_SIZE) -> pd.DataFrame:
        """種牡馬ごとの「ベスト距離」を加重平均で推定する。"""
        df = self.df.copy()
        df["weighted_dist"] = df["distance"] * df["is_top3"]

        grp = df.groupby("sire").agg(
            total_runs=("is_top3", "size"),
            total_top3=("is_top3", "sum"),
            top3_rate=("is_top3", "mean"),
            win_rate=("is_win", "mean"),
            avg_distance=("distance", "mean"),
            weighted_dist_sum=("weighted_dist", "sum"),
        ).reset_index()

        grp = grp[grp["total_runs"] >= min_samples]
        grp["best_distance"] = np.where(
            grp["total_top3"] > 0,
            grp["weighted_dist_sum"] / grp["total_top3"],
            grp["avg_distance"],
        )
        grp["best_distance"] = grp["best_distance"].round(0).astype(int)
        grp["best_dist_cat"] = grp["best_distance"].apply(categorize_distance)
        grp = grp.sort_values("total_runs", ascending=False)

        csv_path = self.output_dir / "sire_best_distance.csv"
        grp.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("保存: %s", csv_path)

        return grp

    # ── 分析3: 父×母父の距離適性マトリクス ──

    def analyze_sire_damsire_matrix(
        self, min_samples: int = 10, top_n: int = 30,
    ) -> pd.DataFrame:
        """父×母父の組み合わせごとの複勝率と出走数。"""
        grp = self.df.groupby(["sire", "dam_sire"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
            avg_distance=("distance", "mean"),
        ).reset_index()

        grp = grp[grp["count"] >= min_samples]

        top_sires = (
            self.df.groupby("sire")["is_top3"].size()
            .nlargest(top_n).index.tolist()
        )
        top_damsires = (
            self.df.groupby("dam_sire")["is_top3"].size()
            .nlargest(top_n).index.tolist()
        )

        filtered = grp[
            grp["sire"].isin(top_sires) & grp["dam_sire"].isin(top_damsires)
        ]

        pivot_rate = filtered.pivot_table(
            index="sire", columns="dam_sire",
            values="top3_rate", fill_value=np.nan,
        )

        csv_path = self.output_dir / "sire_damsire_top3rate.csv"
        pivot_rate.to_csv(csv_path, encoding="utf-8-sig")
        logger.info("保存: %s", csv_path)

        pivot_dist = filtered.pivot_table(
            index="sire", columns="dam_sire",
            values="avg_distance", fill_value=np.nan,
        )
        csv_path2 = self.output_dir / "sire_damsire_avg_distance.csv"
        pivot_dist.to_csv(csv_path2, encoding="utf-8-sig")
        logger.info("保存: %s", csv_path2)

        return filtered

    # ── 分析4: 距離カテゴリ間の血統類似度 ──

    def compute_distance_bloodline_similarity(
        self, min_sire_count: int = 10,
    ) -> pd.DataFrame:
        """
        距離カテゴリごとに「複勝した種牡馬の分布ベクトル」を構成し、
        カテゴリ間のコサイン類似度を計算する。
        血統分布が近い距離カテゴリ = 適性の近い距離帯。
        """
        top3_df = self.df[self.df["is_top3"] == 1]

        sire_counts = top3_df.groupby("sire")["is_top3"].size()
        valid_sires = sire_counts[sire_counts >= min_sire_count].index.tolist()

        filtered = top3_df[top3_df["sire"].isin(valid_sires)]

        dist_sire = filtered.groupby(["distance_cat", "sire"]).size().unstack(fill_value=0)

        # L1 正規化 → コサイン類似度
        norms = dist_sire.values.sum(axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = dist_sire.values / norms

        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(normalized)

        cats = dist_sire.index.tolist()
        sim_df = pd.DataFrame(sim_matrix, index=cats, columns=cats)

        cat_order = [c for c in DISTANCE_CATEGORIES if c in sim_df.index]
        sim_df = sim_df.loc[cat_order, cat_order]

        csv_path = self.output_dir / "distance_bloodline_similarity.csv"
        sim_df.to_csv(csv_path, encoding="utf-8-sig")
        logger.info("保存: %s", csv_path)

        return sim_df

    # ── 分析5: 血統ベクトルクラスタリング ──

    def cluster_sires_by_distance_profile(
        self, min_samples: int = MIN_SAMPLE_SIZE, n_clusters: int = 6,
    ) -> pd.DataFrame:
        """
        種牡馬を距離適性プロファイル (各距離カテゴリの複勝率ベクトル) で
        K-means クラスタリングする。
        """
        grp = self.df.groupby(["sire", "distance_cat"]).agg(
            count=("is_top3", "size"),
            top3_rate=("is_top3", "mean"),
        ).reset_index()

        total = self.df.groupby("sire")["is_top3"].size()
        valid = total[total >= min_samples * 2].index
        grp = grp[grp["sire"].isin(valid)]

        pivot = grp.pivot_table(
            index="sire", columns="distance_cat",
            values="top3_rate", fill_value=0,
        )
        cat_order = [c for c in DISTANCE_CATEGORIES if c in pivot.columns]
        pivot = pivot[cat_order]

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = pivot.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        actual_clusters = min(n_clusters, len(X_scaled))
        km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        result = pivot.copy()
        result["cluster"] = labels

        cluster_summary = result.groupby("cluster")[cat_order].mean()
        cluster_summary["n_sires"] = result.groupby("cluster").size()

        cluster_names = {}
        for cl in cluster_summary.index:
            profile = cluster_summary.loc[cl, cat_order]
            best_cat = profile.idxmax()
            cluster_names[cl] = DIST_CAT_LABELS.get(best_cat, best_cat)

        result["cluster_name"] = result["cluster"].map(cluster_names)
        cluster_summary["cluster_name"] = cluster_summary.index.map(cluster_names)

        csv_path = self.output_dir / "sire_clusters.csv"
        result.to_csv(csv_path, encoding="utf-8-sig")
        logger.info("保存: %s", csv_path)

        summary_path = self.output_dir / "cluster_summary.csv"
        cluster_summary.to_csv(summary_path, encoding="utf-8-sig")
        logger.info("保存: %s", summary_path)

        return result

    # ── 分析6: 血統特徴量の予測寄与度 ──

    def analyze_pedigree_predictive_power(
        self, min_samples: int = MIN_SAMPLE_SIZE,
    ) -> dict[str, Any]:
        """
        血統関連特徴量のみで LightGBM を学習し、
        距離カテゴリ別の予測寄与度を測定する。
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM が必要です")
            return {}

        from sklearn.metrics import roc_auc_score

        df = self.df.copy()

        sire_counts = df.groupby("sire")["is_top3"].size()
        valid_sires = set(sire_counts[sire_counts >= min_samples].index)
        df["sire_code"] = df["sire"].apply(
            lambda s: hash(s) % 100000 if s in valid_sires else -1
        )

        ds_counts = df.groupby("dam_sire")["is_top3"].size()
        valid_ds = set(ds_counts[ds_counts >= min_samples // 2].index)
        df["dam_sire_code"] = df["dam_sire"].apply(
            lambda s: hash(s) % 100000 if s in valid_ds else -1
        )

        sire_top3 = df.groupby("sire")["is_top3"].mean().to_dict()
        ds_top3 = df.groupby("dam_sire")["is_top3"].mean().to_dict()
        df["sire_enc"] = df["sire"].map(sire_top3).fillna(0.2)
        df["dam_sire_enc"] = df["dam_sire"].map(ds_top3).fillna(0.2)

        results_by_cat: dict[str, dict] = {}

        for cat in DISTANCE_CATEGORIES:
            cat_df = df[df["distance_cat"] == cat]
            if len(cat_df) < min_samples * 5:
                continue

            features = cat_df[["sire_enc", "dam_sire_enc", "distance"]].fillna(0)
            target = cat_df["is_top3"]

            split = int(len(features) * 0.8)
            X_tr, X_te = features.iloc[:split], features.iloc[split:]
            y_tr, y_te = target.iloc[:split], target.iloc[split:]

            params = {
                "objective": "binary", "metric": "auc",
                "learning_rate": 0.05, "num_leaves": 31,
                "verbose": -1,
            }
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)

            model = lgb.train(
                params, dtrain, num_boost_round=100,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )

            preds = model.predict(X_te)
            auc = roc_auc_score(y_te, preds) if len(y_te.unique()) > 1 else 0.5

            importance = dict(zip(
                features.columns,
                model.feature_importance(importance_type="gain").tolist(),
            ))

            results_by_cat[cat] = {
                "auc": round(auc, 4),
                "n_samples": len(cat_df),
                "feature_importance": importance,
            }
            logger.info(
                "  %s: AUC=%.4f (n=%d)", DIST_CAT_LABELS.get(cat, cat),
                auc, len(cat_df),
            )

        # 全距離を対象
        features_all = df[["sire_enc", "dam_sire_enc", "distance"]].fillna(0)
        target_all = df["is_top3"]
        split = int(len(features_all) * 0.8)

        params["objective"] = "binary"
        dtrain = lgb.Dataset(features_all.iloc[:split], label=target_all.iloc[:split])
        dval = lgb.Dataset(
            features_all.iloc[split:], label=target_all.iloc[split:],
            reference=dtrain,
        )
        model_all = lgb.train(
            params, dtrain, num_boost_round=100,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
        preds_all = model_all.predict(features_all.iloc[split:])
        auc_all = roc_auc_score(
            target_all.iloc[split:], preds_all,
        ) if len(target_all.iloc[split:].unique()) > 1 else 0.5

        results_by_cat["all"] = {
            "auc": round(auc_all, 4),
            "n_samples": len(df),
            "feature_importance": dict(zip(
                features_all.columns,
                model_all.feature_importance(importance_type="gain").tolist(),
            )),
        }

        json_path = self.output_dir / "pedigree_predictive_power.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_by_cat, f, ensure_ascii=False, indent=2)
        logger.info("保存: %s", json_path)

        return results_by_cat

    # ── 可視化 (HTML レポート) ──

    def generate_report(self) -> str:
        """分析結果を統合した HTML レポートを生成する。"""
        logger.info("=" * 60)
        logger.info("血統 × 距離適性 分析レポート生成")
        logger.info("=" * 60)

        logger.info("[1/6] 種牡馬×距離 複勝率")
        sire_dist = self.analyze_sire_distance_top3rate()

        logger.info("[2/6] 種牡馬ベスト距離")
        best_dist = self.estimate_sire_best_distance()

        logger.info("[3/6] 父×母父 マトリクス")
        cross = self.analyze_sire_damsire_matrix()

        logger.info("[4/6] 距離間 血統類似度")
        sim = self.compute_distance_bloodline_similarity()

        logger.info("[5/6] 血統クラスタリング")
        clusters = self.cluster_sires_by_distance_profile()

        logger.info("[6/6] 血統予測寄与度")
        pred_power = self.analyze_pedigree_predictive_power()

        html = self._build_html_report(
            sire_dist, best_dist, cross, sim, clusters, pred_power,
        )

        report_path = self.output_dir / "bloodline_distance_report.html"
        report_path.write_text(html, encoding="utf-8")
        logger.info("レポート保存: %s", report_path)

        return str(report_path)

    def _build_html_report(
        self,
        sire_dist: pd.DataFrame,
        best_dist: pd.DataFrame,
        cross: pd.DataFrame,
        sim: pd.DataFrame,
        clusters: pd.DataFrame,
        pred_power: dict,
    ) -> str:
        """HTML レポートを構築する。"""

        heatmap_data = []
        for sire in sire_dist.index:
            for cat in sire_dist.columns:
                val = sire_dist.loc[sire, cat]
                heatmap_data.append({
                    "sire": sire, "cat": DIST_CAT_LABELS.get(cat, cat),
                    "value": round(float(val) * 100, 1),
                })

        best_dist_top30 = best_dist.head(30)
        best_data = []
        for _, row in best_dist_top30.iterrows():
            best_data.append({
                "sire": row["sire"],
                "runs": int(row["total_runs"]),
                "top3_rate": round(float(row["top3_rate"]) * 100, 1),
                "best_dist": int(row["best_distance"]),
                "best_cat": DIST_CAT_LABELS.get(row["best_dist_cat"], ""),
            })

        sim_data = []
        for cat1 in sim.index:
            for cat2 in sim.columns:
                sim_data.append({
                    "cat1": DIST_CAT_LABELS.get(cat1, cat1),
                    "cat2": DIST_CAT_LABELS.get(cat2, cat2),
                    "value": round(float(sim.loc[cat1, cat2]), 3),
                })

        cluster_summary_path = self.output_dir / "cluster_summary.csv"
        cluster_data = []
        if cluster_summary_path.exists():
            cs = pd.read_csv(cluster_summary_path, index_col=0)
            for idx, row in cs.iterrows():
                cluster_data.append({
                    "cluster": int(idx),
                    "name": row.get("cluster_name", ""),
                    "n_sires": int(row.get("n_sires", 0)),
                    "profile": {
                        DIST_CAT_LABELS.get(c, c): round(float(row.get(c, 0)) * 100, 1)
                        for c in DISTANCE_CATEGORIES if c in row.index
                    },
                })

        pred_data = {}
        for cat, info in pred_power.items():
            label = DIST_CAT_LABELS.get(cat, "全体") if cat != "all" else "全体"
            pred_data[label] = {
                "auc": info.get("auc", 0),
                "n": info.get("n_samples", 0),
            }

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>血統×距離適性 研究レポート</title>
<style>
  :root {{
    --bg: #0a0e17; --surface: #131926; --surface2: #1a2235;
    --border: #243049; --text: #c8d6e5; --text-dim: #6b7d95;
    --accent: #3b82f6; --ok: #22c55e; --warn: #f59e0b; --err: #ef4444;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg); color: var(--text); padding: 32px;
    max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
  h2 {{ font-size: 20px; font-weight: 600; margin: 32px 0 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  h3 {{ font-size: 16px; font-weight: 600; color: var(--text-dim); margin-bottom: 8px; }}
  .subtitle {{ font-size: 14px; color: var(--text-dim); margin-bottom: 24px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 20px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    text-align: left; font-size: 11px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.5px;
    padding: 8px; border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 8px; border-bottom: 1px solid rgba(36,48,73,0.4); }}
  tr:hover {{ background: rgba(59,130,246,0.05); }}
  .heat {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; min-width: 48px; text-align: center; }}
  .heat-high {{ background: rgba(34,197,94,0.2); color: var(--ok); }}
  .heat-mid {{ background: rgba(245,158,11,0.15); color: var(--warn); }}
  .heat-low {{ background: rgba(239,68,68,0.12); color: var(--err); }}
  .sim-cell {{ font-weight: 600; }}
  .stat {{ display: inline-flex; flex-direction: column; align-items: center; background: var(--surface2); border-radius: 8px; padding: 12px 20px; min-width: 120px; }}
  .stat-val {{ font-size: 20px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: var(--text-dim); margin-top: 2px; }}
  .stats-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
  .bar {{ height: 16px; border-radius: 4px; background: var(--accent); transition: width 0.3s; }}
  .bar-bg {{ height: 16px; border-radius: 4px; background: var(--surface2); width: 100%; overflow: hidden; }}
  .cluster-badge {{
    display: inline-block; font-size: 11px; font-weight: 700;
    padding: 2px 8px; border-radius: 12px;
    background: rgba(59,130,246,0.15); color: var(--accent);
  }}
</style>
</head>
<body>
<h1>血統 × 距離適性 研究レポート</h1>
<p class="subtitle">
  データ: {len(self.df):,} 出走 |
  種牡馬: {self.df['sire'].nunique()} |
  母父: {self.df['dam_sire'].nunique()} |
  距離帯: {len(DISTANCE_CATEGORIES)}
</p>

<h2>1. 種牡馬 × 距離カテゴリ 複勝率ヒートマップ</h2>
<div class="card">
<h3>上位50種牡馬 (出走数順)</h3>
<table>
<thead><tr><th>種牡馬</th>{''.join(f'<th>{DIST_CAT_LABELS.get(c, c)}</th>' for c in DISTANCE_CATEGORIES if c in sire_dist.columns)}</tr></thead>
<tbody>
{''.join(self._render_heatmap_row(sire, sire_dist) for sire in sire_dist.index[:30])}
</tbody>
</table>
</div>

<h2>2. 種牡馬ベスト距離推定 (TOP30)</h2>
<div class="card">
<table>
<thead><tr><th>種牡馬</th><th>出走数</th><th>複勝率</th><th>ベスト距離</th><th>距離帯</th></tr></thead>
<tbody>
{''.join(f'<tr><td><strong>{d["sire"]}</strong></td><td>{d["runs"]:,}</td><td>{d["top3_rate"]}%</td><td>{d["best_dist"]:,}m</td><td>{d["best_cat"]}</td></tr>' for d in best_data)}
</tbody>
</table>
</div>

<h2>3. 距離カテゴリ間 血統類似度</h2>
<div class="card">
<h3>コサイン類似度: 複勝種牡馬分布の類似性を測定</h3>
<p style="color:var(--text-dim);font-size:12px;margin-bottom:12px">
  値が 1.0 に近いほど、同じ血統の馬が活躍する距離帯。
</p>
<table>
<thead><tr><th></th>{''.join(f'<th>{DIST_CAT_LABELS.get(c, c)}</th>' for c in DISTANCE_CATEGORIES if c in sim.columns)}</tr></thead>
<tbody>
{self._render_sim_table(sim)}
</tbody>
</table>
</div>

<h2>4. 血統クラスタリング</h2>
<div class="card">
<h3>種牡馬を距離適性プロファイルで {len(cluster_data)} クラスタに分類</h3>
{''.join(self._render_cluster(cl) for cl in cluster_data)}
</div>

<h2>5. 血統特徴量の予測寄与度</h2>
<div class="card">
<h3>血統情報のみ (sire_enc, dam_sire_enc, distance) で複勝を予測した AUC</h3>
<div class="stats-row">
{''.join(f'<div class="stat"><span class="stat-val" style="color:{("var(--ok)" if v["auc"]>=0.55 else "var(--warn)")}">{v["auc"]:.4f}</span><span class="stat-label">{k} (n={v["n"]:,})</span></div>' for k, v in pred_data.items())}
</div>
<p style="font-size:12px;color:var(--text-dim)">
  AUC 0.50 = ランダム、0.55+ で血統に有意な予測力あり。
  距離帯によって血統の寄与度が異なることが確認できれば、距離×血統の交互作用特徴量が有効。
</p>
</div>

<h2>6. 考察・示唆</h2>
<div class="card">
<ul style="font-size:14px;line-height:1.8;padding-left:20px">
  <li><strong>血統類似度マトリクス</strong>: 隣接する距離帯ほど血統構成が似る傾向があれば、距離適性は連続的</li>
  <li><strong>クラスタ分析</strong>: スプリンター血統/ステイヤー血統の明確な分離は、血統特徴量の有効性を裏付ける</li>
  <li><strong>予測寄与度</strong>: AUC が特定距離帯で高い場合、その距離帯では血統が重要な予測因子</li>
  <li><strong>実践的示唆</strong>: 父×母父の交互作用特徴量や、距離×血統クラスタの交差特徴量を feature_builder に追加することで精度向上の可能性</li>
</ul>
</div>

</body>
</html>"""

    def _render_heatmap_row(self, sire: str, pivot: pd.DataFrame) -> str:
        cells = [f"<td><strong>{sire}</strong></td>"]
        for cat in DISTANCE_CATEGORIES:
            if cat not in pivot.columns:
                continue
            val = float(pivot.loc[sire, cat])
            pct = round(val * 100, 1)
            cls = "heat-high" if pct >= 35 else ("heat-mid" if pct >= 25 else "heat-low")
            cells.append(f'<td><span class="heat {cls}">{pct}%</span></td>')
        return f"<tr>{''.join(cells)}</tr>"

    def _render_sim_table(self, sim: pd.DataFrame) -> str:
        rows = []
        for cat1 in sim.index:
            cells = [f'<td><strong>{DIST_CAT_LABELS.get(cat1, cat1)}</strong></td>']
            for cat2 in sim.columns:
                val = float(sim.loc[cat1, cat2])
                color = "var(--ok)" if val >= 0.9 else ("var(--warn)" if val >= 0.7 else "var(--text-dim)")
                cells.append(f'<td class="sim-cell" style="color:{color}">{val:.3f}</td>')
            rows.append(f"<tr>{''.join(cells)}</tr>")
        return "\n".join(rows)

    def _render_cluster(self, cl: dict) -> str:
        profile_bars = ""
        max_val = max(cl["profile"].values()) if cl["profile"] else 1
        for label, val in cl["profile"].items():
            w = val / max_val * 100 if max_val > 0 else 0
            profile_bars += f"""
            <div style="display:flex;align-items:center;gap:8px;margin:2px 0">
              <span style="font-size:11px;min-width:140px;color:var(--text-dim)">{label}</span>
              <div class="bar-bg" style="flex:1"><div class="bar" style="width:{w}%"></div></div>
              <span style="font-size:11px;min-width:40px">{val}%</span>
            </div>"""
        return f"""
        <div style="margin-bottom:16px;padding:12px;background:var(--surface2);border-radius:8px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <span class="cluster-badge">Cluster {cl['cluster']}</span>
            <strong>{cl['name']}</strong>
            <span style="color:var(--text-dim);font-size:12px">({cl['n_sires']} 種牡馬)</span>
          </div>
          {profile_bars}
        </div>"""


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="血統×距離適性 研究")
    parser.add_argument("--source", choices=["gcs", "csv"], default="gcs")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--years", nargs="*", default=None)
    parser.add_argument("--output-dir", default="data/research/bloodline")
    args = parser.parse_args()

    analyzer = BloodlineDistanceAnalyzer(output_dir=args.output_dir)

    if args.source == "gcs":
        analyzer.load_from_gcs(years=args.years)
    else:
        analyzer.load_from_csv(features_dir=args.features_dir)

    if analyzer.df.empty:
        logger.error("データが空です。終了します。")
        sys.exit(1)

    report_path = analyzer.generate_report()
    print(f"\nレポート生成完了: {report_path}")


if __name__ == "__main__":
    main()
