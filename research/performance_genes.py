"""
競走馬パフォーマンス関連遺伝子（ミオスタチン以外）の解析モジュール

DMRT3（歩様）、PDK4（代謝）、COX4I2（持久力）、BIEC2-808543（距離適性）等の
遺伝子マーカーに関するナレッジベース検索と推論機能を提供。

Usage:
  from research.performance_genes import PerformanceGenes
  pg = PerformanceGenes()
  pg.get_gene_info("DMRT3")
  pg.predict_pdk4_distance_preference("GG", "AA")
  pg.get_combined_distance_profile(mstn="CT", pdk4="GA", biec2="CT")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge" / "performance_genes.json"


class PerformanceGenes:
    """ミオスタチン以外の競走馬パフォーマンス遺伝子ナレッジベースの検索・推論エンジン。"""

    def __init__(self, kb_path: str | Path | None = None):
        self._kb_path = Path(kb_path) if kb_path else _KB_PATH
        self._genes: dict[str, dict] = {}
        self._interactions: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self._kb_path.exists():
            logger.warning("パフォーマンス遺伝子KBが見つかりません: %s", self._kb_path)
            return
        data = json.loads(self._kb_path.read_text(encoding="utf-8"))
        for gene in data.get("genes", []):
            gene_id = gene.get("gene_id", "")
            if gene_id:
                self._genes[gene_id] = gene
        self._interactions = data.get("gene_interactions", [])

    @property
    def gene_count(self) -> int:
        return len(self._genes)

    def list_genes(self) -> list[str]:
        """登録されている遺伝子IDのリストを取得。"""
        return sorted(self._genes.keys())

    def get_gene_info(self, gene_id: str) -> dict | None:
        """遺伝子IDで基本情報を取得。"""
        return self._genes.get(gene_id)

    def get_genotype_info(self, gene_id: str, genotype: str) -> dict | None:
        """遺伝子型（例: "CC", "GA"）の詳細情報を取得。"""
        gene = self._genes.get(gene_id)
        if not gene:
            return None
        genotypes = gene.get("genotypes", {})
        return genotypes.get(genotype)

    # ─── DMRT3（歩様遺伝子）──────────────────────────

    def dmrt3_gait_type(self, genotype: str) -> str:
        """
        DMRT3遺伝子型から歩様タイプを返す。

        Args:
            genotype: "AA", "CA", "CC"

        Returns:
            "pace_trot" (速歩・側対歩), "intermediate", "gallop" (襲歩)
        """
        genotype = genotype.upper()
        if genotype == "AA":
            return "pace_trot"
        elif genotype in ("CA", "AC"):
            return "intermediate"
        elif genotype == "CC":
            return "gallop"
        return "unknown"

    def dmrt3_thoroughbred_suitability(self, genotype: str) -> float:
        """
        DMRT3遺伝子型のサラブレッド競走への適合度（0.0〜1.0）。

        CC型 = 1.0（最適）、CA型 = 0.7、AA型 = 0.3（不適）
        """
        genotype = genotype.upper()
        if genotype == "CC":
            return 1.0
        elif genotype in ("CA", "AC"):
            return 0.7
        elif genotype == "AA":
            return 0.3
        return 0.5

    # ─── PDK4（エネルギー代謝遺伝子）──────────────────

    def pdk4_distance_preference(self, genotype: str) -> dict[str, Any]:
        """
        PDK4遺伝子型から距離適性プロファイルを返す。

        Returns:
            {
                "optimal_distance": int (m),
                "distance_range": (min, max),
                "metabolism_type": str,
                "stamina_score": float (0-1)
            }
        """
        genotype = genotype.upper()
        info = self.get_genotype_info("PDK4", genotype)
        if not info:
            return {
                "optimal_distance": 1800,
                "distance_range": (1000, 3000),
                "metabolism_type": "unknown",
                "stamina_score": 0.5
            }

        # 距離適性文字列を数値化
        dist_apt = info.get("distance_aptitude", "")
        if "短距離" in dist_apt or "1000-1600" in dist_apt:
            opt_dist = 1300
            dist_range = (1000, 1600)
            stamina = 0.3
            metabolism = "glucose"
        elif "長距離" in dist_apt or "1800m以上" in dist_apt:
            opt_dist = 2200
            dist_range = (1800, 3000)
            stamina = 0.9
            metabolism = "fat_oxidation"
        else:  # 中距離
            opt_dist = 1800
            dist_range = (1400, 2200)
            stamina = 0.6
            metabolism = "mixed"

        return {
            "optimal_distance": opt_dist,
            "distance_range": dist_range,
            "metabolism_type": metabolism,
            "stamina_score": stamina
        }

    def predict_pdk4_offspring(
        self,
        sire_genotype: str,
        dam_genotype: str | None = None
    ) -> dict[str, float]:
        """
        父母のPDK4遺伝子型から産駒の遺伝子型確率を推定。

        母の遺伝子型が不明の場合は集団平均（GG:35%, GA:45%, AA:20%）を使用。

        Returns:
            {"GG": p, "GA": p, "AA": p}
        """
        sire_gt = sire_genotype.upper()

        # アレル確率を計算
        def allele_probs(gt: str) -> tuple[float, float]:
            if gt == "GG":
                return (1.0, 0.0)
            elif gt in ("GA", "AG"):
                return (0.5, 0.5)
            elif gt == "AA":
                return (0.0, 1.0)
            return (0.5, 0.5)  # unknown

        sire_g, sire_a = allele_probs(sire_gt)

        if dam_genotype:
            dam_g, dam_a = allele_probs(dam_genotype.upper())
        else:
            # 母不明の場合、集団平均から推定
            # GG: 35%, GA: 45%, AA: 20% → G頻度 = 0.35*1 + 0.45*0.5 = 0.575
            dam_g = 0.575
            dam_a = 0.425

        # メンデル遺伝
        gg = sire_g * dam_g
        aa = sire_a * dam_a
        ga = sire_g * dam_a + sire_a * dam_g

        total = gg + ga + aa
        if total > 0:
            gg /= total
            ga /= total
            aa /= total

        return {"GG": round(gg, 4), "GA": round(ga, 4), "AA": round(aa, 4)}

    # ─── 統合的距離適性スコア ──────────────────────────

    def get_combined_distance_profile(
        self,
        mstn: str | None = None,
        pdk4: str | None = None,
        biec2: str | None = None,
        distance_m: int = 0,
    ) -> dict[str, Any]:
        """
        複数遺伝子マーカーを統合した距離適性プロファイル。

        Args:
            mstn: ミオスタチン遺伝子型（CC/CT/TT）
            pdk4: PDK4遺伝子型（GG/GA/AA）
            biec2: BIEC2-808543遺伝子型（CC/CT/TT）
            distance_m: レース距離（指定時はその距離での適合度も返す）

        Returns:
            {
                "optimal_distance": int (統合推定値),
                "distance_range": (min, max),
                "distance_score": float (指定距離での適合度, 0-1),
                "speed_index": float (0=ステイヤー, 1=スプリンター),
                "stamina_index": float (0=スプリント, 1=ステイヤー),
                "contributors": dict (各遺伝子の寄与)
            }
        """
        # 各遺伝子からの推定値
        distances = []
        weights = []
        speed_scores = []
        stamina_scores = []
        contributors = {}

        # MSTN（最重要）
        if mstn:
            mstn = mstn.upper()
            if mstn == "CC":
                distances.append(1200)
                speed_scores.append(1.0)
                stamina_scores.append(0.0)
                weights.append(1.0)
                contributors["MSTN"] = {"genotype": mstn, "distance": 1200, "weight": 1.0}
            elif mstn == "CT":
                distances.append(1700)
                speed_scores.append(0.5)
                stamina_scores.append(0.5)
                weights.append(1.0)
                contributors["MSTN"] = {"genotype": mstn, "distance": 1700, "weight": 1.0}
            elif mstn == "TT":
                distances.append(2200)
                speed_scores.append(0.0)
                stamina_scores.append(1.0)
                weights.append(1.0)
                contributors["MSTN"] = {"genotype": mstn, "distance": 2200, "weight": 1.0}

        # PDK4（中程度の重要性）
        if pdk4:
            pdk4_profile = self.pdk4_distance_preference(pdk4)
            distances.append(pdk4_profile["optimal_distance"])
            stamina_scores.append(pdk4_profile["stamina_score"])
            speed_scores.append(1.0 - pdk4_profile["stamina_score"])
            weights.append(0.4)
            contributors["PDK4"] = {
                "genotype": pdk4,
                "distance": pdk4_profile["optimal_distance"],
                "weight": 0.4
            }

        # BIEC2-808543（中程度の重要性、MSTNと相関）
        if biec2:
            biec2 = biec2.upper()
            biec2_info = self.get_genotype_info("BIEC2-808543", biec2)
            if biec2_info:
                biec2_dist = biec2_info.get("best_distance_peak", 1600)
                # 文字列の場合は数値化
                if isinstance(biec2_dist, str):
                    if "1200" in biec2_dist:
                        biec2_dist = 1200
                    elif "1600" in biec2_dist:
                        biec2_dist = 1600
                    elif "2400" in biec2_dist:
                        biec2_dist = 2400
                    else:
                        biec2_dist = 1800
                distances.append(biec2_dist)
                weights.append(0.3)
                contributors["BIEC2"] = {"genotype": biec2, "distance": biec2_dist, "weight": 0.3}

        # 統合推定
        if not distances:
            return {
                "optimal_distance": 1800,
                "distance_range": (1000, 3000),
                "distance_score": 0.5,
                "speed_index": 0.5,
                "stamina_index": 0.5,
                "contributors": {}
            }

        # 加重平均
        total_weight = sum(weights)
        optimal_dist = int(sum(d * w for d, w in zip(distances, weights)) / total_weight)

        # スピード・スタミナ指数
        if speed_scores:
            speed_idx = sum(s * w for s, w in zip(speed_scores, weights[:len(speed_scores)])) / sum(weights[:len(speed_scores)])
        else:
            speed_idx = 0.5

        if stamina_scores:
            stamina_idx = sum(s * w for s, w in zip(stamina_scores, weights[:len(stamina_scores)])) / sum(weights[:len(stamina_scores)])
        else:
            stamina_idx = 0.5

        # 距離範囲（±20%）
        dist_min = int(optimal_dist * 0.8)
        dist_max = int(optimal_dist * 1.2)

        # 指定距離での適合度
        distance_score = 0.5
        if distance_m > 0:
            # ガウス分布的スコア
            sigma = optimal_dist * 0.25  # 標準偏差は最適距離の25%
            distance_score = _gaussian(float(distance_m), float(optimal_dist), sigma)

        return {
            "optimal_distance": optimal_dist,
            "distance_range": (dist_min, dist_max),
            "distance_score": round(distance_score, 4),
            "speed_index": round(speed_idx, 4),
            "stamina_index": round(stamina_idx, 4),
            "contributors": contributors
        }

    # ─── 遺伝子相互作用の取得 ──────────────────────────

    def get_interaction(self, gene1: str, gene2: str) -> dict | None:
        """2つの遺伝子間の相互作用情報を取得。"""
        for interaction in self._interactions:
            genes = interaction.get("genes", [])
            if gene1 in genes and gene2 in genes:
                return interaction
        return None

    def check_extreme_phenotype(
        self,
        mstn: str | None = None,
        pdk4: str | None = None
    ) -> dict[str, Any]:
        """
        MSTN×PDK4の組み合わせで極端な表現型（超スプリント、超ステイヤー）を判定。

        Returns:
            {
                "is_extreme": bool,
                "phenotype": "ultra_sprinter" | "ultra_stayer" | "balanced",
                "note": str
            }
        """
        if not mstn or not pdk4:
            return {"is_extreme": False, "phenotype": "unknown", "note": ""}

        mstn = mstn.upper()
        pdk4 = pdk4.upper()

        # 超スプリンター（MSTN CC + PDK4 GG）
        if mstn == "CC" and pdk4 == "GG":
            return {
                "is_extreme": True,
                "phenotype": "ultra_sprinter",
                "note": "筋肉質体型 + グルコース代謝優位 = 極短距離特化（1000-1200m）"
            }

        # 超ステイヤー（MSTN TT + PDK4 AA）
        if mstn == "TT" and pdk4 == "AA":
            return {
                "is_extreme": True,
                "phenotype": "ultra_stayer",
                "note": "スリム体型 + 脂肪酸酸化優位 = 極長距離特化（2400m+）"
            }

        # その他の極端な組み合わせ
        if mstn == "CC" and pdk4 == "AA":
            return {
                "is_extreme": False,
                "phenotype": "conflicted",
                "note": "MSTN（スプリント）とPDK4（スタミナ）が矛盾。中間的能力の可能性"
            }

        if mstn == "TT" and pdk4 == "GG":
            return {
                "is_extreme": False,
                "phenotype": "conflicted",
                "note": "MSTN（スタミナ）とPDK4（スプリント）が矛盾。中距離寄りの可能性"
            }

        return {"is_extreme": False, "phenotype": "balanced", "note": ""}


# ─── ヘルパー関数 ────────────────────────────────

def _gaussian(x: float, mu: float, sigma: float) -> float:
    """ガウス分布（正規化なし、ピークで1.0）"""
    import math
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ─── シングルトン ──────────────────────────────────

_instance: PerformanceGenes | None = None


def get_lookup() -> PerformanceGenes:
    """モジュールレベルのシングルトンインスタンスを取得。"""
    global _instance
    if _instance is None:
        _instance = PerformanceGenes()
    return _instance


# ─── CLI ───────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="競走馬パフォーマンス遺伝子解析ツール")
    sub = parser.add_subparsers(dest="cmd")

    list_p = sub.add_parser("list", help="登録遺伝子一覧")

    gene_p = sub.add_parser("gene", help="遺伝子情報表示")
    gene_p.add_argument("gene_id", help="遺伝子ID (例: DMRT3, PDK4)")

    dmrt3_p = sub.add_parser("dmrt3", help="DMRT3遺伝子型の適合度")
    dmrt3_p.add_argument("genotype", choices=["AA", "CA", "CC"], help="遺伝子型")

    pdk4_p = sub.add_parser("pdk4", help="PDK4遺伝子型の距離適性")
    pdk4_p.add_argument("genotype", choices=["GG", "GA", "AA"], help="遺伝子型")

    combined_p = sub.add_parser("combined", help="統合的距離適性プロファイル")
    combined_p.add_argument("--mstn", choices=["CC", "CT", "TT"], help="ミオスタチン遺伝子型")
    combined_p.add_argument("--pdk4", choices=["GG", "GA", "AA"], help="PDK4遺伝子型")
    combined_p.add_argument("--biec2", choices=["CC", "CT", "TT"], help="BIEC2-808543遺伝子型")
    combined_p.add_argument("--distance", type=int, default=0, help="レース距離(m)")

    extreme_p = sub.add_parser("extreme", help="極端表現型チェック")
    extreme_p.add_argument("--mstn", choices=["CC", "CT", "TT"], required=True)
    extreme_p.add_argument("--pdk4", choices=["GG", "GA", "AA"], required=True)

    args = parser.parse_args()
    pg = PerformanceGenes()

    if args.cmd == "list":
        print("登録遺伝子:")
        for gene_id in pg.list_genes():
            info = pg.get_gene_info(gene_id)
            print(f"  {gene_id:20s} - {info.get('gene_name', '')}")

    elif args.cmd == "gene":
        info = pg.get_gene_info(args.gene_id)
        if info:
            print(f"遺伝子ID: {info['gene_id']}")
            print(f"遺伝子名: {info['gene_name']}")
            print(f"染色体: {info.get('chromosome', '不明')}")
            print(f"機能: {info.get('function', '')}")
            print("\n遺伝子型:")
            for gt, gt_info in info.get("genotypes", {}).items():
                print(f"  {gt}: {gt_info.get('description', '')}")
        else:
            print(f"遺伝子 '{args.gene_id}' は登録されていません")

    elif args.cmd == "dmrt3":
        gait = pg.dmrt3_gait_type(args.genotype)
        suit = pg.dmrt3_thoroughbred_suitability(args.genotype)
        print(f"DMRT3遺伝子型: {args.genotype}")
        print(f"歩様タイプ: {gait}")
        print(f"サラブレッド適合度: {suit:.2f}")

    elif args.cmd == "pdk4":
        profile = pg.pdk4_distance_preference(args.genotype)
        print(f"PDK4遺伝子型: {args.genotype}")
        print(f"最適距離: {profile['optimal_distance']}m")
        print(f"距離範囲: {profile['distance_range'][0]}-{profile['distance_range'][1]}m")
        print(f"代謝タイプ: {profile['metabolism_type']}")
        print(f"スタミナスコア: {profile['stamina_score']:.2f}")

    elif args.cmd == "combined":
        profile = pg.get_combined_distance_profile(
            mstn=args.mstn,
            pdk4=args.pdk4,
            biec2=args.biec2,
            distance_m=args.distance
        )
        print("統合距離適性プロファイル:")
        print(f"最適距離: {profile['optimal_distance']}m")
        print(f"距離範囲: {profile['distance_range'][0]}-{profile['distance_range'][1]}m")
        print(f"スピード指数: {profile['speed_index']:.4f}")
        print(f"スタミナ指数: {profile['stamina_index']:.4f}")
        if args.distance > 0:
            print(f"{args.distance}mでの適合度: {profile['distance_score']:.4f}")
        print("\n寄与遺伝子:")
        for gene, info in profile['contributors'].items():
            print(f"  {gene}: {info['genotype']} (推定距離={info['distance']}m, 重み={info['weight']})")

    elif args.cmd == "extreme":
        result = pg.check_extreme_phenotype(mstn=args.mstn, pdk4=args.pdk4)
        print(f"MSTN: {args.mstn}, PDK4: {args.pdk4}")
        print(f"極端表現型: {'はい' if result['is_extreme'] else 'いいえ'}")
        print(f"表現型: {result['phenotype']}")
        if result['note']:
            print(f"備考: {result['note']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
