"""全祖先種牡馬 (~4,773 + 主流外) の L2_fine 割り当てを構築する。

主流 133 種牡馬は ``unified.parquet`` に L2_fine ラベルが付いている。
それ以外の祖先 (海外祖先含む) は、その種牡馬の **子孫の L2 分布** を集計して
最頻 L2 を割り当てる (= 父系継承による L2 推定)。

子孫の L2 分布が取れない場合は、4 大主流系統 → デフォルト L2 にマッピング:
    Turn-To系          → L2=8 (持久力長持続力中型, ディープ系)
    Native Dancer系    → L2=4 (東京型, Kingmambo 系)
    Northern Dancer系  → L2=1 (阪神ダート型, Storm Cat 系)
    Nasrullah系        → L2=8 (持久力長持続力中型)
    非主流             → L2=10 (小倉札幌型)
これはあくまでフォールバック。

入力:
    data/research/bloodline_meta_cluster/unified.parquet
    data/research/bloodline_meta_cluster/sid_to_main.json
    data/research/pedigree_race_index/horse_pedigree_cats.parquet  (祖先→子孫マップ)
    data/research/bloodline_meta_cluster/sid_to_name_full.json

出力:
    data/research/bloodline_meta_cluster/ancestor_to_l2.json
        {stallion_id: {"L2": int, "method": "direct"|"descendant_vote"|"main_group_default",
                       "name": str, "main_group": str, "n_votes": int}}

Usage:
    python -m src.research.pedigree.build_ancestor_l2_index
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "data/page_reference/note_aptitude_race"
CATS_PATH = ROOT / "data/page_reference/pedigree_race_index/horse_pedigree_cats.parquet"

# main_group → デフォルト L2 (centroid 最近接 + 解釈的整合性)
MAIN_GROUP_DEFAULT_L2 = {
    "Turn-To系":         8,   # 持久力長持続力中型 (Deep Impact 系)
    "Native Dancer系":   4,   # 東京型 (Kingmambo 系)
    "Northern Dancer系": 1,   # 阪神ダート型 (Storm Cat 系)
    "Nasrullah系":       8,
    "非主流":             10,
}


def main() -> int:
    print("[load] unified.parquet ...", flush=True)
    uni = pd.read_parquet(ART / "unified.parquet")
    sid_to_main = json.loads((ART / "sid_to_main.json").read_text(encoding="utf-8"))
    name_full = json.loads((ART / "sid_to_name_full.json").read_text(encoding="utf-8"))

    # 1) 主流 (133頭): unified に L2 ラベルあり
    direct: dict[str, int] = {}
    for _, r in uni[uni["entity_type"] == "stallion"].iterrows():
        sid = str(r["entity_id"])
        L2 = int(r["L2"])
        if L2 >= 0:
            direct[sid] = L2
    print(f"  direct (主流): {len(direct)} 頭", flush=True)

    # 2) horse_pedigree_cats から「祖先 → 子孫 (主流馬) → L2」 票決
    print("[load] horse_pedigree_cats.parquet ...", flush=True)
    cats = pd.read_parquet(CATS_PATH, columns=["horse_id", "stallion_id"])
    cats["stallion_id"] = cats["stallion_id"].astype(str)
    cats["horse_id"] = cats["horse_id"].astype(str)
    print(f"  ancestor records: {len(cats):,}", flush=True)

    # horse_id (= 子孫) が unified 主流種牡馬と一致する場合のみ集計
    # しかし unified の主流は stallion entity_id だが、horse_pedigree_cats の horse_id は
    # 競走馬 ID (子孫) なので、直接マッチしない。
    # → 競走馬 horse_id ベースでは適性が分からない (L2 ラベルは種牡馬につく)
    # → 代わりに、各馬の **父** stallion_id が主流 L2 を持つかでみる
    # それでも各祖先 (Ancestors) → 子孫の父 L2 という関係になる

    # 各祖先について、その祖先を持つ馬たちの「父」が何の L2 か集計
    print("[group] 各祖先 -> 該当競走馬を集約 ...", flush=True)
    anc_to_horses = cats.groupby("stallion_id")["horse_id"].apply(list).to_dict()
    print(f"  unique ancestors: {len(anc_to_horses):,}", flush=True)

    # 各競走馬の「父 stallion_id」 (gen=1, path_fm=F) を取得
    sires = cats[(cats.get("gen", 0) == 1) | (cats["stallion_id"].str.len() == 10)].copy() if False else None
    # シンプルに: 各 horse_id の最初のレコード (gen=1) を父とする
    cats_gen = pd.read_parquet(CATS_PATH, columns=["horse_id", "stallion_id", "gen"])
    cats_gen["horse_id"] = cats_gen["horse_id"].astype(str)
    cats_gen["stallion_id"] = cats_gen["stallion_id"].astype(str)
    horse_to_sire = (
        cats_gen[cats_gen["gen"] == 1].drop_duplicates("horse_id")
        .set_index("horse_id")["stallion_id"].to_dict()
    )
    print(f"  horse_to_sire entries: {len(horse_to_sire):,}", flush=True)

    # 子孫の父 L2 集計
    descendant_vote: dict[str, dict] = {}
    n_unknown = 0
    for anc, horses in anc_to_horses.items():
        votes = Counter()
        for h in horses:
            sire = horse_to_sire.get(h)
            if sire and sire in direct:
                votes[direct[sire]] += 1
        if votes:
            top_l2, top_n = votes.most_common(1)[0]
            descendant_vote[anc] = {
                "L2": int(top_l2),
                "method": "descendant_vote",
                "n_votes": int(top_n),
                "vote_counts": dict(votes),
            }
        else:
            n_unknown += 1
    print(f"  descendant_vote 成功: {len(descendant_vote):,} / unknown: {n_unknown:,}", flush=True)

    # 3) フォールバック: main_group → デフォルト L2
    fallback: dict[str, dict] = {}
    for sid in anc_to_horses:
        if sid in direct or sid in descendant_vote:
            continue
        mg = sid_to_main.get(sid, "非主流")
        fallback[sid] = {
            "L2": MAIN_GROUP_DEFAULT_L2.get(mg, 10),
            "method": "main_group_default",
            "main_group": mg,
        }

    # 4) 統合 (direct → descendant_vote → fallback)
    result: dict[str, dict] = {}
    for sid, L2 in direct.items():
        result[sid] = {
            "L2": int(L2),
            "method": "direct",
            "name": name_full.get(sid, ""),
            "main_group": sid_to_main.get(sid, "非主流"),
        }
    for sid, info in descendant_vote.items():
        if sid in result:
            continue
        result[sid] = {
            **info,
            "name": name_full.get(sid, ""),
            "main_group": sid_to_main.get(sid, "非主流"),
        }
    for sid, info in fallback.items():
        result[sid] = {
            **info,
            "name": name_full.get(sid, ""),
        }

    (ART / "ancestor_to_l2.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[save] {ART/'ancestor_to_l2.json'} ({len(result):,} entries)", flush=True)

    # 集計サマリ
    by_method = Counter(v["method"] for v in result.values())
    print("\n=== method 別 ===")
    for k, v in by_method.most_common():
        print(f"  {k:24}: {v:,}")
    by_l2 = Counter(v["L2"] for v in result.values())
    print("\n=== L2 別 ===")
    for k, v in sorted(by_l2.items()):
        print(f"  L2={k:2}: {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
