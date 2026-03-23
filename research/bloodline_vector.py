"""
血統ベクトル空間解析

種牡馬をベクトル空間に埋め込み、血統構成の類似度を計算する。

ベクトルの構成要素:
  1. Sireライン系統エンコーディング (大系統 + 中系統)
  2. 主要祖先の出現パターン (父, 母父, 母母父をキーに再帰探索)
  3. レース成績統計 (距離分布, 馬場適性, コース適性)
  4. ミオスタチン遺伝子アレル確率

Usage:
  python -m research.bloodline_vector --years 2020 2021 2022 2023 2024 2025 2026
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

logger = logging.getLogger("research.bloodline_vector")

# ── 大系統分類 ─────────────────────────────────────

SIRE_LINES: dict[str, list[str]] = {
    "サンデーサイレンス系": [
        "サンデーサイレンス", "ディープインパクト", "ステイゴールド", "ハーツクライ",
        "ダイワメジャー", "フジキセキ", "アグネスタキオン", "マンハッタンカフェ",
        "ゴールドアリュール", "スペシャルウィーク", "ネオユニヴァース",
        "ゼンノロブロイ", "ブラックタイド",
        "キズナ", "コントレイル", "サトノダイヤモンド", "マカヒキ",
        "シルバーステート", "ミッキーアイル", "ダノンプレミアム",
        "リアルスティール", "ワグネリアン", "シャフリヤール",
        "オルフェーヴル", "ゴールドシップ", "ナカヤマフェスタ",
        "ジャスタウェイ", "スワーヴリチャード",
        "キタサンブラック", "イスラボニータ", "ダノンザキッド",
        "ディープブリランテ", "トーセンラー", "サトノアラジン",
        "ダノンキングリー", "ダノンバラード", "シュヴァルグラン",
        "ラブリーデイ", "リアルインパクト", "フィエールマン",
        "エイシンヒカリ", "エイシンフラッシュ", "ロゴタイプ",
        "インディチャンプ", "ストロングリターン", "ロジャーバローズ",
        "アドマイヤマーズ", "グレーターロンドン", "ファインニードル",
        "キンシャサノキセキ", "ビッグアーサー", "ダノンスマッシュ",
        "ウインブライト", "ヴィクトワールピサ", "ディーマジェスティ",
        "コパノリッキー", "エスポワールシチー", "リオンディーズ",
        "ゴールドドリーム", "ルヴァンスレーヴ", "クリソベリル",
        "マテラスカイ", "フィレンツェファイア", "モズアスコット",
        "レッドファルクス", "タワーオブロンドン",
        "アルアイン", "ミスターメロディ",
    ],
    "キングカメハメハ系": [
        "キングカメハメハ", "ロードカナロア", "ルーラーシップ",
        "ドゥラメンテ", "レイデオロ", "ホッコータルマエ",
        "サートゥルナーリア", "キセキ",
    ],
    "ロベルト系": [
        "シンボリクリスエス", "エピファネイア", "スクリーンヒーロー",
        "モーリス", "グラスワンダー",
    ],
    "ノーザンダンサー系": [
        "Northern Dancer", "Nearctic", "ノーザンテースト",
        "サトノクラウン", "デクラレーションオブウォー",
    ],
    "デインヒル系": [
        "Danehill", "ハービンジャー", "Siyouni", "Kingman",
        "ハービンジャーHarbinger(英)", "ノーブルミッションNoble Mission(英)",
        "シスキンSiskin(米)", "マクフィMakfi(英)",
        "ポエティックフレアPoetic Flare(愛)", "ヴァンゴッホ",
        "タリスマニックTalismanic(英)",
    ],
    "サドラーズウェルズ系": [
        "Sadler's Wells", "Galileo", "Frankel", "New Approach",
        "Cracksman", "Dawn Approach",
        "バゴBago(仏)", "ベンバトルBenbatl(英)",
        "サンダースノーThunder Snow(愛)",
    ],
    "ストームキャット系": [
        "Storm Cat", "ヘニーヒューズ", "ディスクリートキャット",
        "ブリックスアンドモルタル",
        "ヘニーヒューズHenny Hughes(米)",
        "ディスクリートキャットDiscreet Cat(米)",
        "ブリックスアンドモルタルBricks and Mortar(米)",
        "シャンハイボビーShanghai Bobby(米)",
        "Into Mischief",
        "ミスチヴィアスアレックスMischevious Alex(米)",
        "マインドユアビスケッツMind Your Biscuits(米)",
        "ナダルNadal(米)", "フォーウィールドライブFour Wheel Drive(米)",
    ],
    "ミスタープロスペクター系": [
        "Mr. Prospector", "Kingmambo", "Dubawi",
        "ドレフォン", "サウスヴィグラス",
        "ドレフォンDrefong(米)", "カレンブラックヒル",
        "カリフォルニアクロームCalifornia Chrome(米)",
        "ニューイヤーズデイ", "アメリカンペイトリオット",
    ],
    "エーピーインディ系": [
        "パイロ", "パイロPyro(米)",
        "マジェスティックウォリアーMajestic Warrior(米)",
        "ロージズインメイRoses in May(米)",
        "ビーチパトロールBeach Patrol(米)",
        "シニスターミニスターSinister Minister(米)",
    ],
    "デピュティミニスター系": [
        "クロフネ", "フレンチデピュティ", "アニマルキングダムAnimal Kingdom(米)",
        "ダノンレジェンド",
    ],
    "その他": [
        "トニービン", "ジャングルポケット", "メジロマックイーン",
        "タイキシャトル", "サクラバクシンオー",
        "モーニン", "アジアエクスプレス",
    ],
}

_SIRE_TO_LINE: dict[str, str] = {}
for line, sires in SIRE_LINES.items():
    for s in sires:
        _SIRE_TO_LINE[s] = line

ALL_LINE_NAMES = sorted(SIRE_LINES.keys())

DISTANCE_BINS = [
    ("sprint", 0, 1400),
    ("mile", 1400, 1800),
    ("middle", 1800, 2200),
    ("long", 2200, 9999),
]

SURFACE_KEYS = ["芝", "ダート"]

VENUE_KEYS = [
    "札幌", "函館", "福島", "新潟", "東京",
    "中山", "中京", "京都", "阪神", "小倉",
]

TRACK_COND_KEYS = ["良", "稍重", "重", "不良"]


def classify_sire_line(sire_name: str) -> str:
    if sire_name in _SIRE_TO_LINE:
        return _SIRE_TO_LINE[sire_name]
    return "不明"


def _dist_bin(distance: int) -> str:
    for label, lo, hi in DISTANCE_BINS:
        if lo <= distance < hi:
            return label
    return "long"


class BloodlineVectorBuilder:
    """レースデータから種牡馬の血統ベクトルを構築する。"""

    def __init__(self, output_dir: str = "data/research/bloodline_vector"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.sire_profiles: dict[str, dict] = {}
        self.vectors: pd.DataFrame = pd.DataFrame()

    # ── データ収集 ──────────────────────────────────

    def load_from_gcs(self, years: list[str] | None = None) -> None:
        from scraper.storage import HybridStorage
        storage = HybridStorage(".")
        if not storage.gcs_enabled:
            logger.error("GCS 未接続")
            return

        if years is None:
            years = storage.list_years("race_result")
        logger.info("対象年度: %s", years)

        rows: list[dict] = []
        sire_parent_cache: dict[str, dict] = {}

        for year in sorted(years):
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
                track_cond = result_data.get("track_condition", "")
                field_size = result_data.get("field_size", 0) or len(
                    result_data.get("entries", []))

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
                    sire = sire or se.get("sire", "")
                    dam_sire = dam_sire or se.get("dam_sire", "")

                    if not sire:
                        horse_data = storage.load("horse_result", hid)
                        if horse_data:
                            sire = horse_data.get("sire", "")
                            dam_sire = dam_sire or horse_data.get("dam_sire", "")

                    if not sire:
                        continue

                    rows.append({
                        "race_id": race_id,
                        "horse_id": hid,
                        "sire": sire,
                        "dam_sire": dam_sire or "不明",
                        "finish_position": fp,
                        "distance": distance,
                        "dist_bin": _dist_bin(distance),
                        "surface": surface,
                        "venue": venue,
                        "track_condition": track_cond,
                        "field_size": field_size,
                        "is_top3": 1 if fp <= 3 else 0,
                        "is_win": 1 if fp == 1 else 0,
                    })

        self.raw_df = pd.DataFrame(rows)
        logger.info("データ構築完了: %d 出走, %d 種牡馬, %d 母父",
                     len(self.raw_df),
                     self.raw_df["sire"].nunique(),
                     self.raw_df["dam_sire"].nunique())

        self._sire_parents: dict[str, dict] = {}

    # ── ベクトル構築 ────────────────────────────────

    def build_vectors(self, min_runners: int = 30) -> pd.DataFrame:
        """種牡馬ごとの特徴ベクトルを構築する。"""
        df = self.raw_df
        if df.empty:
            logger.error("データが空です")
            return pd.DataFrame()

        sire_groups = df.groupby("sire")
        profiles: list[dict] = []

        for sire_name, group in sire_groups:
            n = len(group)
            if n < min_runners:
                continue

            profile: dict[str, Any] = {"sire": sire_name, "n_runners": n}

            profile["sire_line"] = classify_sire_line(sire_name)
            valid_dist = group.loc[group["distance"] > 0, "distance"]
            profile["avg_distance"] = valid_dist.mean() if len(valid_dist) > 0 else 1800

            has_dist = group["distance"] > 0
            n_with_dist = has_dist.sum()
            for dbin in ["sprint", "mile", "middle", "long"]:
                mask = (group["dist_bin"] == dbin) & has_dist
                total = mask.sum()
                profile[f"dist_{dbin}_rate"] = total / n_with_dist if n_with_dist else 0
                profile[f"dist_{dbin}_top3"] = (
                    group.loc[mask, "is_top3"].mean() if total >= 5 else 0.0)

            for surf in SURFACE_KEYS:
                mask = group["surface"] == surf
                total = mask.sum()
                profile[f"surf_{surf}_rate"] = total / n if n else 0
                profile[f"surf_{surf}_top3"] = (
                    group.loc[mask, "is_top3"].mean() if total >= 5 else 0.0)

            for tc in TRACK_COND_KEYS:
                mask = group["track_condition"] == tc
                total = mask.sum()
                profile[f"tc_{tc}_rate"] = total / n if n else 0
                profile[f"tc_{tc}_top3"] = (
                    group.loc[mask, "is_top3"].mean() if total >= 5 else 0.0)

            for venue in VENUE_KEYS:
                mask = group["venue"] == venue
                total = mask.sum()
                profile[f"venue_{venue}_rate"] = total / n if n else 0
                profile[f"venue_{venue}_top3"] = (
                    group.loc[mask, "is_top3"].mean() if total >= 5 else 0.0)

            profile["overall_top3_rate"] = group["is_top3"].mean()
            profile["overall_win_rate"] = group["is_win"].mean()

            try:
                from research.myostatin import get_lookup
                mstn = get_lookup()
                ac, at = mstn.get_allele_probs(sire_name)
                profile["mstn_allele_c"] = ac
                profile["mstn_allele_t"] = at
            except Exception:
                profile["mstn_allele_c"] = 0.4
                profile["mstn_allele_t"] = 0.6

            for line_name in ALL_LINE_NAMES:
                profile[f"line_{line_name}"] = (
                    1.0 if profile["sire_line"] == line_name else 0.0)

            profiles.append(profile)

        self.sire_profiles = {p["sire"]: p for p in profiles}
        self.vectors = pd.DataFrame(profiles).set_index("sire")
        logger.info("ベクトル構築: %d 種牡馬 (min_runners=%d)",
                     len(self.vectors), min_runners)
        return self.vectors

    def get_feature_columns(self) -> list[str]:
        """次元削減に使う数値カラムを返す。"""
        exclude = {"sire_line", "n_runners", "avg_distance"}
        return [c for c in self.vectors.columns
                if c not in exclude and self.vectors[c].dtype in ("float64", "int64")]

    # ── 次元削減 ──────────────────────────────────

    def compute_embeddings(self, method: str = "umap", n_components: int = 2) -> pd.DataFrame:
        """UMAP or PCA で2次元に埋め込む。"""
        feat_cols = self.get_feature_columns()
        X = self.vectors[feat_cols].fillna(0).values

        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(15, len(X) - 1),
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )
                embedding = reducer.fit_transform(X_scaled)
            except ImportError:
                logger.warning("umap-learn 未インストール。PCA にフォールバック")
                method = "pca"

        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=42)
            embedding = pca.fit_transform(X_scaled)
            logger.info("PCA 寄与率: %s",
                        [f"{v:.2%}" for v in pca.explained_variance_ratio_])

        if method == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(
                n_components=n_components, perplexity=min(30, len(X) - 1),
                random_state=42, metric="cosine")
            embedding = tsne.fit_transform(X_scaled)

        emb_df = pd.DataFrame(
            embedding,
            index=self.vectors.index,
            columns=[f"dim_{i}" for i in range(n_components)],
        )
        for col in ["sire_line", "n_runners", "avg_distance",
                     "overall_top3_rate", "overall_win_rate",
                     "mstn_allele_c", "mstn_allele_t",
                     "dist_sprint_rate", "dist_mile_rate",
                     "dist_middle_rate", "dist_long_rate",
                     "surf_芝_rate", "surf_ダート_rate",
                     "surf_芝_top3", "surf_ダート_top3"]:
            if col in self.vectors.columns:
                emb_df[col] = self.vectors[col]

        return emb_df

    # ── 類似度 ────────────────────────────────────

    def similarity(self, sire_a: str, sire_b: str) -> float:
        """2種牡馬間のコサイン類似度。"""
        feat_cols = self.get_feature_columns()
        if sire_a not in self.vectors.index or sire_b not in self.vectors.index:
            return 0.0
        va = self.vectors.loc[sire_a, feat_cols].fillna(0).values
        vb = self.vectors.loc[sire_b, feat_cols].fillna(0).values
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(dot / norm) if norm > 0 else 0.0

    def most_similar(self, sire_name: str, top_n: int = 10) -> list[tuple[str, float]]:
        """指定種牡馬に最も類似した種牡馬を返す。"""
        feat_cols = self.get_feature_columns()
        if sire_name not in self.vectors.index:
            return []

        from sklearn.metrics.pairwise import cosine_similarity
        X = self.vectors[feat_cols].fillna(0).values
        idx = list(self.vectors.index).index(sire_name)
        sims = cosine_similarity(X[idx:idx+1], X)[0]

        results = []
        for i, s in enumerate(sims):
            name = self.vectors.index[i]
            if name != sire_name:
                results.append((name, float(s)))
        results.sort(key=lambda x: -x[1])
        return results[:top_n]

    # ── 出力 ──────────────────────────────────────

    def save_results(self, emb_df: pd.DataFrame) -> None:
        self.vectors.to_csv(self.output_dir / "sire_vectors.csv", encoding="utf-8-sig")
        emb_df.to_csv(self.output_dir / "sire_embeddings.csv", encoding="utf-8-sig")

        vis_data = []
        for sire_name in emb_df.index:
            row = emb_df.loc[sire_name]
            vis_data.append({
                "name": sire_name,
                "x": round(float(row["dim_0"]), 4),
                "y": round(float(row["dim_1"]), 4),
                "sire_line": row.get("sire_line", "不明"),
                "n_runners": int(row.get("n_runners", 0)),
                "avg_distance": round(float(row.get("avg_distance", 0)), 0),
                "top3_rate": round(float(row.get("overall_top3_rate", 0)), 4),
                "win_rate": round(float(row.get("overall_win_rate", 0)), 4),
                "mstn_c": round(float(row.get("mstn_allele_c", 0)), 2),
                "mstn_t": round(float(row.get("mstn_allele_t", 0)), 2),
                "sprint_rate": round(float(row.get("dist_sprint_rate", 0)), 4),
                "mile_rate": round(float(row.get("dist_mile_rate", 0)), 4),
                "middle_rate": round(float(row.get("dist_middle_rate", 0)), 4),
                "long_rate": round(float(row.get("dist_long_rate", 0)), 4),
                "turf_rate": round(float(row.get("surf_芝_rate", 0)), 4),
                "dirt_rate": round(float(row.get("surf_ダート_rate", 0)), 4),
                "turf_top3": round(float(row.get("surf_芝_top3", 0)), 4),
                "dirt_top3": round(float(row.get("surf_ダート_top3", 0)), 4),
            })

        vis_path = self.output_dir / "sire_embeddings.json"
        vis_path.write_text(
            json.dumps(vis_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("結果保存: %s", self.output_dir)

    # ── フル実行 ──────────────────────────────────

    def run(self, years: list[str] | None = None, min_runners: int = 30,
            method: str = "pca") -> pd.DataFrame:
        self.load_from_gcs(years)
        self.build_vectors(min_runners)
        emb_df = self.compute_embeddings(method=method)
        self.save_results(emb_df)
        return emb_df


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="血統ベクトル空間解析")
    parser.add_argument("--years", nargs="*", default=None)
    parser.add_argument("--min-runners", type=int, default=30)
    parser.add_argument("--method", choices=["pca", "umap", "tsne"], default="pca")
    args = parser.parse_args()

    builder = BloodlineVectorBuilder()
    emb_df = builder.run(args.years, args.min_runners, args.method)

    print(f"\n=== 結果: {len(emb_df)} 種牡馬 ===")
    print(emb_df.head(20))

    if not emb_df.empty:
        sample = emb_df.index[0]
        print(f"\n{sample} に最も近い種牡馬:")
        for name, sim in builder.most_similar(sample, 5):
            print(f"  {name}: {sim:.4f}")


if __name__ == "__main__":
    main()
