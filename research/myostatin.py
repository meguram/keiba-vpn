"""
ミオスタチン遺伝子 (MSTN) 解析モジュール

ミオスタチン遺伝子の型 (CC/CT/TT) をナレッジベースから検索し、
子馬の遺伝子型確率を推論し、距離適性スコアを算出する。

遺伝ルール:
  - 各馬は 2つのアレル (C or T) を持つ → CC / CT / TT
  - 親から子へ各 1アレルずつ遺伝
  - TT: 長距離向き, CT: マイル〜中距離, CC: スプリント向き

Usage:
  from research.myostatin import MyostatinLookup
  mstn = MyostatinLookup()
  mstn.get_sire_info("ディープインパクト")
  mstn.predict_offspring("キタサンブラック", "サンデーサイレンス")
  mstn.distance_affinity("ディープインパクト", "ロードカナロア", 2400)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge" / "myostatin_genes.json"


class MyostatinLookup:
    """競走馬・種牡馬ミオスタチン遺伝子ナレッジベースの検索・推論エンジン。"""

    def __init__(self, kb_path: str | Path | None = None):
        self._kb_path = Path(kb_path) if kb_path else _KB_PATH
        self._stallions: dict[str, dict] = {}
        self._en_index: dict[str, str] = {}
        self._defaults: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self._kb_path.exists():
            logger.warning("ミオスタチンKBが見つかりません: %s", self._kb_path)
            return
        data = json.loads(self._kb_path.read_text(encoding="utf-8"))
        self._defaults = data.get("population_defaults", {})
        for s in data.get("stallions", []):
            name = s.get("name", "")
            if name:
                self._stallions[name] = s
            name_en = s.get("name_en", "")
            if name_en:
                self._en_index[name_en.lower()] = name

    @property
    def stallion_count(self) -> int:
        """登録馬数（種牡馬・競走馬）を返す。"""
        return len(self._stallions)

    def list_stallions(self) -> list[str]:
        """登録馬名のリストを返す（種牡馬・競走馬）。"""
        return sorted(self._stallions.keys())

    def get_sire_info(self, name: str) -> dict | None:
        """馬名（日本語 or 英語）でナレッジベースを検索。種牡馬・競走馬どちらも対象。"""
        if name in self._stallions:
            return self._stallions[name]
        jp_name = self._en_index.get(name.lower())
        if jp_name:
            return self._stallions.get(jp_name)
        return None

    def get_allele_probs(self, name: str) -> tuple[float, float]:
        """
        馬のアレル確率 (allele_c, allele_t) を返す。
        KBに未登録の場合は集団デフォルト値を返す。
        """
        info = self.get_sire_info(name)
        if info:
            return (info.get("allele_c", 0.5), info.get("allele_t", 0.5))
        defaults = self._defaults.get("japanese_thoroughbred_avg", {})
        return (defaults.get("allele_c", 0.4), defaults.get("allele_t", 0.6))

    def predict_offspring(
        self,
        sire_name: str,
        dam_sire_name: str,
    ) -> dict[str, float]:
        """
        父と母父の遺伝子型から、子馬の遺伝子型確率を推定する。

        母の遺伝子型は不明なため、母父のアレル確率 + 母の母側デフォルト値を
        組み合わせて母のアレル確率を推定する。

        Returns: {"CC": p, "CT": p, "TT": p}
        """
        sire_c, sire_t = self.get_allele_probs(sire_name)

        ds_c, ds_t = self.get_allele_probs(dam_sire_name)
        dam_dam_defaults = self._defaults.get("unknown_mare_avg", {})
        dd_c = dam_dam_defaults.get("allele_c", 0.45)
        dd_t = dam_dam_defaults.get("allele_t", 0.55)

        dam_c = ds_c * 0.5 + dd_c * 0.5
        dam_t = ds_t * 0.5 + dd_t * 0.5

        cc = sire_c * dam_c
        tt = sire_t * dam_t
        ct = sire_c * dam_t + sire_t * dam_c

        total = cc + ct + tt
        if total > 0:
            cc /= total
            ct /= total
            tt /= total

        return {"CC": round(cc, 4), "CT": round(ct, 4), "TT": round(tt, 4)}

    def distance_affinity(
        self,
        sire_name: str,
        dam_sire_name: str,
        distance_m: int,
    ) -> float:
        """
        推定ミオスタチン遺伝子型とレース距離の適合度を算出する。

        各遺伝子型の「理想距離帯」をガウス分布でモデル化し、
        子馬の推定遺伝子型確率で加重平均した値を返す。

        Returns: 0.0〜1.0 のスコア (1.0 = 完全適合)
        """
        probs = self.predict_offspring(sire_name, dam_sire_name)
        return _genotype_distance_score(probs, distance_m)

    def reverse_infer(
        self,
        offspring_genotype: str,
        sire_name: str | None = None,
        dam_sire_name: str | None = None,
    ) -> dict[str, Any]:
        """
        産駒の確定遺伝子型から親の遺伝子型を逆推論する。

        @Plucky_Liege のツイート等で産駒の遺伝子型が判明した際に使う。
        例: 産駒がCC → 父も母もCアレル保有確定。
            産駒がTT → 父も母もTアレル保有確定。

        Returns: dict with inferred constraints on sire/dam
        """
        result: dict[str, Any] = {
            "offspring_genotype": offspring_genotype,
            "sire_constraints": {},
            "dam_constraints": {},
        }

        if offspring_genotype == "CC":
            result["sire_constraints"]["has_c"] = True
            result["sire_constraints"]["possible_genotypes"] = ["CC", "CT"]
            result["sire_constraints"]["excludes"] = ["TT"]
            result["dam_constraints"]["has_c"] = True
        elif offspring_genotype == "TT":
            result["sire_constraints"]["has_t"] = True
            result["sire_constraints"]["possible_genotypes"] = ["TT", "CT"]
            result["sire_constraints"]["excludes"] = ["CC"]
            result["dam_constraints"]["has_t"] = True
        elif offspring_genotype == "CT":
            result["sire_constraints"]["note"] = "CTからは親の型を限定できない"

        if sire_name:
            sire_info = self.get_sire_info(sire_name)
            if sire_info:
                current = sire_info.get("genotype", "")
                if offspring_genotype == "CC" and current == "TT":
                    result["sire_constraints"]["conflict"] = (
                        f"{sire_name}は現在TT推定だがCC産駒が出現→CT以上に修正が必要"
                    )
                elif offspring_genotype == "TT" and current == "CC":
                    result["sire_constraints"]["conflict"] = (
                        f"{sire_name}は現在CC推定だがTT産駒が出現→CT以上に修正が必要"
                    )

        return result

    def sire_allele_features(self, sire_name: str) -> dict[str, float]:
        """パイプライン向け: 父馬のアレル特徴量を dict で返す。"""
        c, t = self.get_allele_probs(sire_name)
        return {"sire_mstn_c": c, "sire_mstn_t": t}

    def dam_sire_allele_features(self, dam_sire_name: str) -> dict[str, float]:
        """パイプライン向け: 母父のアレル特徴量を dict で返す。"""
        c, t = self.get_allele_probs(dam_sire_name)
        return {"dam_sire_mstn_c": c, "dam_sire_mstn_t": t}

    def offspring_features(
        self,
        sire_name: str,
        dam_sire_name: str,
        distance_m: int = 0,
    ) -> dict[str, float]:
        """
        パイプライン向け: ミオスタチン関連の全特徴量を一括生成。

        生成される特徴量:
          - sire_mstn_c, sire_mstn_t         : 父のアレル確率
          - dam_sire_mstn_c, dam_sire_mstn_t  : 母父のアレル確率
          - mstn_cc_prob, mstn_ct_prob, mstn_tt_prob : 子馬遺伝子型確率
          - mstn_distance_affinity            : 距離適合度 (距離が指定された場合)
          - mstn_speed_index                  : スプリント寄り度 (0=完全ステイヤー, 1=完全スプリンター)
        """
        feats: dict[str, float] = {}
        feats.update(self.sire_allele_features(sire_name))
        feats.update(self.dam_sire_allele_features(dam_sire_name))

        probs = self.predict_offspring(sire_name, dam_sire_name)
        feats["mstn_cc_prob"] = probs["CC"]
        feats["mstn_ct_prob"] = probs["CT"]
        feats["mstn_tt_prob"] = probs["TT"]

        feats["mstn_speed_index"] = probs["CC"] * 1.0 + probs["CT"] * 0.5 + probs["TT"] * 0.0

        if distance_m > 0:
            feats["mstn_distance_affinity"] = _genotype_distance_score(probs, distance_m)
        else:
            feats["mstn_distance_affinity"] = 0.0

        return feats


# ─── 距離適合度ヘルパー ────────────────────────────

_GENOTYPE_DIST_PARAMS: dict[str, tuple[float, float]] = {
    "CC": (1200.0, 250.0),
    "CT": (1700.0, 350.0),
    "TT": (2200.0, 400.0),
}


def _gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _genotype_distance_score(probs: dict[str, float], distance_m: int) -> float:
    score = 0.0
    for gtype, (mu, sigma) in _GENOTYPE_DIST_PARAMS.items():
        score += probs.get(gtype, 0.0) * _gaussian(float(distance_m), mu, sigma)
    return round(min(score, 1.0), 4)


# ─── シングルトン ──────────────────────────────────

_instance: MyostatinLookup | None = None


def get_lookup() -> MyostatinLookup:
    """モジュールレベルのシングルトンインスタンスを取得。"""
    global _instance
    if _instance is None:
        _instance = MyostatinLookup()
    return _instance


# ─── CLI ───────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ミオスタチン遺伝子検索・推論ツール")
    sub = parser.add_subparsers(dest="cmd")

    lookup_p = sub.add_parser("lookup", help="馬の遺伝子型を検索")
    lookup_p.add_argument("name", help="馬名（種牡馬・競走馬）")

    predict_p = sub.add_parser("predict", help="子馬の遺伝子型確率を推定")
    predict_p.add_argument("sire", help="父名")
    predict_p.add_argument("dam_sire", help="母父名")
    predict_p.add_argument("--distance", type=int, default=0, help="距離(m)")

    sub.add_parser("list", help="登録馬一覧（種牡馬・競走馬）")

    sub.add_parser("stats", help="ナレッジベース統計")

    infer_p = sub.add_parser("infer", help="産駒の遺伝子型から親を逆推論")
    infer_p.add_argument("genotype", choices=["CC", "CT", "TT"], help="産駒の遺伝子型")
    infer_p.add_argument("--sire", help="父名")
    infer_p.add_argument("--dam-sire", help="母父名")

    args = parser.parse_args()
    mstn = MyostatinLookup()

    if args.cmd == "lookup":
        info = mstn.get_sire_info(args.name)
        if info:
            print(f"馬名: {info['name']} ({info.get('name_en', '')})")
            print(f"遺伝子型: {info.get('genotype', '不明')}")
            print(f"信頼度: {info.get('confidence', '不明')}")
            print(f"アレル確率: C={info.get('allele_c', '?')}, T={info.get('allele_t', '?')}")
            print(f"平均勝距離: {info.get('avg_win_dist', '?')}m")
            print(f"根拠: {info.get('source', '')}")
        else:
            print(f"'{args.name}' はナレッジベースに未登録です（種牡馬・競走馬データ）")
            c, t = mstn.get_allele_probs(args.name)
            print(f"デフォルト値を適用: C={c}, T={t}")

    elif args.cmd == "predict":
        probs = mstn.predict_offspring(args.sire, args.dam_sire)
        print(f"父: {args.sire}")
        print(f"母父: {args.dam_sire}")
        print(f"子馬遺伝子型確率: CC={probs['CC']:.1%}, CT={probs['CT']:.1%}, TT={probs['TT']:.1%}")
        if args.distance > 0:
            aff = mstn.distance_affinity(args.sire, args.dam_sire, args.distance)
            print(f"距離適合度 ({args.distance}m): {aff:.4f}")

    elif args.cmd == "list":
        for name in mstn.list_stallions():
            info = mstn.get_sire_info(name)
            gt = info.get("genotype", "?") if info else "?"
            conf = info.get("confidence", "?") if info else "?"
            print(f"  {name:20s} {gt:3s} ({conf})")

    elif args.cmd == "infer":
        result = mstn.reverse_infer(args.genotype, args.sire, args.dam_sire)
        print(f"産駒遺伝子型: {result['offspring_genotype']}")
        print("父への制約:")
        for k, v in result["sire_constraints"].items():
            print(f"  {k}: {v}")
        print("母(系)への制約:")
        for k, v in result["dam_constraints"].items():
            print(f"  {k}: {v}")

    elif args.cmd == "stats":
        by_type: dict[str, int] = {}
        by_conf: dict[str, int] = {}
        for name in mstn.list_stallions():
            info = mstn.get_sire_info(name)
            if info:
                gt = info.get("genotype", "unknown")
                conf = info.get("confidence", "unknown")
                by_type[gt] = by_type.get(gt, 0) + 1
                by_conf[conf] = by_conf.get(conf, 0) + 1
        print(f"登録馬数（種牡馬・競走馬）: {mstn.stallion_count}")
        print("遺伝子型別:")
        for k, v in sorted(by_type.items()):
            print(f"  {k}: {v}頭")
        print("信頼度別:")
        for k, v in sorted(by_conf.items()):
            print(f"  {k}: {v}頭")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
