"""
全種牡馬の父系（サイアーライン）ツリーを構築し、系統グループを自動検出して保存するスクリプト。

アルゴリズム
-----------
1. horse_pedigree_5gen の全 JSON から sire_of[child_id] = sire_id を収集
2. 分析 parquet (horse_pedigree_cats) に登場する種牡馬を「分析対象」とする
3. 各種牡馬の総出現頻度を horse_pedigree_cats から集計し、上位 MAX_ANCHORS を「系統アンカー」に設定
4. 各分析対象種牡馬 → サイアーチェーンを上下に探索して最近接アンカーに割り当て
   上探索 (↑) : 自分の先祖方向（父 → 祖父 → ... ）
   下探索 (↓) : 自分の子孫方向（ BFS、最初に出会うアンカーへ）
5. stallion_lineage.parquet に保存

なぜ頻度ベースか
----------------
集計で最頻出の Pharos, Phalaris, Nearco, Gainsborough 等が自然にアンカーになる。
これらは gen6-10 でほぼすべての現代競走馬の配合に現れるため、
「○○系かどうか」という視覚的な色分けに最適。

出力
----
data/research/pedigree_race_index/stallion_lineage.parquet
  stallion_id, stallion_name, anchor_id, anchor_name,
  depth_to_anchor, group_id (0..N-1, -1=その他)

data/research/pedigree_race_index/stallion_lineage_meta.json
  [{group_id, anchor_name, anchor_id, count}]

Usage
-----
python -m research.build_stallion_lineage
python -m research.build_stallion_lineage --max-anchors 30
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

PED_DIR = _ROOT / "data" / "local" / "horse_pedigree_5gen"
CATS_PARQUET = _ROOT / "data" / "research" / "pedigree_race_index" / "horse_pedigree_cats.parquet"
OUT_PATH = _ROOT / "data" / "research" / "pedigree_race_index" / "stallion_lineage.parquet"

_COUNTRY_SUFFIX = re.compile(r"\s*\([^)]{1,4}\)\s*$")


def _clean_name(s: str) -> str:
    return _COUNTRY_SUFFIX.sub("", s).strip()


def build(max_anchors: int = 30) -> None:
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ── 1. 全 JSON から sire_of / name_of を構築 ──
    logger.info("horse_pedigree_5gen 読み込み中...")
    sire_of: dict[str, str] = {}
    name_of: dict[str, str] = {}

    count = 0
    for f in sorted(PED_DIR.rglob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hid = str(rec.get("horse_id") or "").strip()
        if hid:
            rn = _clean_name(str(rec.get("horse_name") or rec.get("name") or ""))
            if rn:
                existing = name_of.get(hid, rn)
                name_of[hid] = min(existing, rn, key=len)
        for anc in rec.get("ancestors") or []:
            aid = str(anc.get("horse_id") or "").strip()
            if not aid:
                continue
            aname = _clean_name(str(anc.get("name") or ""))
            if aname:
                existing = name_of.get(aid, aname)
                name_of[aid] = min(existing, aname, key=len)
            if anc.get("generation") == 1 and anc.get("position") == 0 and hid:
                sire_of[hid] = aid
        count += 1

    logger.info("JSON %d 件 / sire関係: %d (%.1fs)", count, len(sire_of), time.time() - t0)

    # ── 2. 分析対象種牡馬を読み込み、頻度集計 ──
    if not CATS_PARQUET.exists():
        logger.error("horse_pedigree_cats.parquet が見つかりません。")
        return

    cats_df = pd.read_parquet(CATS_PARQUET, columns=["stallion_id"])
    cats_df["stallion_id"] = cats_df["stallion_id"].astype(str)
    analysis_stallions: set[str] = set(cats_df["stallion_id"].unique())
    freq_series = cats_df.groupby("stallion_id", observed=True).size()
    freq_map: dict[str, int] = {str(k): int(v) for k, v in freq_series.items()}
    logger.info("分析対象種牡馬: %d 種", len(analysis_stallions))

    # ── 3. 逆引き (children_of) ・トポロジカル順を構築 ──
    children_of: dict[str, list[str]] = {}
    all_nodes = set(sire_of) | set(sire_of.values())
    for child, parent in sire_of.items():
        children_of.setdefault(parent, []).append(child)

    # ── 4. アンカー選択: 集計頻度上位 max_anchors ──
    #
    # 頻度ベースで選ぶ理由:
    #   集計で最頻出の Pharos, Phalaris, Nearco, Gainsborough 等は
    #   gen6-10 で大半の現代馬に共通する「系統代表」として最適。
    #   各種牡馬は「サイアーチェーンを辿って最近接アンカー」へ割り当てられるので、
    #   Northern Dancer の子孫は Nearco (←アンカー) ではなく Nearco の 子孫 Northern Dancer が
    #   アンカーなら Northern Dancer グループに正しく割り当てられる。
    #
    sorted_by_freq = sorted(freq_map.items(), key=lambda x: -x[1])
    anchor_ids: list[str] = [sid for sid, _ in sorted_by_freq[:max_anchors]]
    anchor_set: set[str] = set(anchor_ids)
    anchor_to_group: dict[str, int] = {aid: i for i, aid in enumerate(anchor_ids)}

    logger.info("系統アンカー確定: %d 個", len(anchor_ids))
    for i, aid in enumerate(anchor_ids):
        logger.info("  [%02d] %s %-32s (freq: %d)", i, aid, name_of.get(aid, aid), freq_map.get(aid, 0))

    # ── 5. 各種牡馬に系統グループを割り当て ──
    #
    # 優先順位:
    #   (A) 自身がアンカー → depth=0
    #   (B) 上探索: サイアーチェーンを辿り最初のアンカー → depth=steps
    #   (C) 下探索 (BFS): 子孫方向で最近接アンカー → depth=steps (負の値で報告)
    #
    # (C) は (B) が失敗した場合のみ行う。
    # 最大探索深さ MAX_DEPTH を設けて計算量を抑える。
    #
    MAX_UP_DEPTH = 40
    MAX_DOWN_DEPTH = 10  # 子孫方向は浅く探索

    def _find_anchor_up(sid: str) -> tuple[str, int]:
        """サイアーチェーンを上に辿ってアンカーを探す。(anchor_id, depth) or ("", 0)"""
        if sid in anchor_set:
            return sid, 0
        cur = sire_of.get(sid)
        d = 1
        seen: set[str] = set()
        while cur and cur not in seen and d <= MAX_UP_DEPTH:
            seen.add(cur)
            if cur in anchor_set:
                return cur, d
            cur = sire_of.get(cur)
            d += 1
        return "", 0

    # 子孫方向探索: BFS で最近接アンカーを探す
    def _find_anchor_down(sid: str) -> tuple[str, int]:
        """子孫方向 BFS でアンカーを探す。(anchor_id, depth) or ("", 0)"""
        q: deque[tuple[str, int]] = deque([(sid, 0)])
        visited: set[str] = set([sid])
        while q:
            nid, d = q.popleft()
            if d > MAX_DOWN_DEPTH:
                break
            for child in children_of.get(nid, []):
                if child in visited:
                    continue
                visited.add(child)
                if child in anchor_set:
                    return child, d + 1
                q.append((child, d + 1))
        return "", 0

    rows: list[dict] = []
    n_up = n_down = n_none = 0

    for sid in sorted(analysis_stallions):
        anchor_id, depth = _find_anchor_up(sid)
        method = "up"
        if not anchor_id:
            anchor_id, depth = _find_anchor_down(sid)
            method = "down"
        if not anchor_id:
            method = "none"

        if anchor_id:
            group_id = anchor_to_group[anchor_id]
        else:
            group_id = -1

        if method == "up":
            n_up += 1
        elif method == "down":
            n_down += 1
        else:
            n_none += 1

        sname = name_of.get(sid, sid)
        aname = name_of.get(anchor_id, anchor_id) if anchor_id else ""
        rows.append({
            "stallion_id": sid,
            "stallion_name": sname,
            "anchor_id": anchor_id,
            "anchor_name": aname,
            "depth_to_anchor": depth,
            "group_id": group_id,
        })

    logger.info("割り当て結果: 上探索=%d, 下探索=%d, 未割り当て=%d", n_up, n_down, n_none)

    df = pd.DataFrame(rows)
    df["group_id"] = df["group_id"].astype("int8")
    df["depth_to_anchor"] = df["depth_to_anchor"].astype("int16")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="zstd")
    logger.info("保存: %s (%d 件, %.1f KB)", OUT_PATH, len(df), OUT_PATH.stat().st_size / 1024)

    # グループ別件数サマリ
    group_summary = (
        df.groupby(["group_id", "anchor_name"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    logger.info("系統グループ別件数:\n%s", group_summary.to_string(index=False))

    # meta JSON
    import json as _json
    groups_meta = []
    for _, row in group_summary.iterrows():
        gid = int(row["group_id"])
        groups_meta.append({
            "group_id": gid,
            "anchor_name": row["anchor_name"],
            "anchor_id": anchor_ids[gid] if 0 <= gid < len(anchor_ids) else "",
            "count": int(row["count"]),
        })
    meta_path = OUT_PATH.parent / "stallion_lineage_meta.json"
    meta_path.write_text(_json.dumps(groups_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("メタ保存: %s", meta_path)

    # ── 8. アンカー間の親子関係を計算して anchor_tree.json を保存 ──
    #
    # 各アンカーについて「自身を除くサイアーチェーンを上に辿り最初に出会うアンカー」が
    # ツリー上の親ノードになる。これにより max_anchors ノードの小さなツリーが構成できる。
    #
    logger.info("アンカーツリー（親子関係）構築中...")

    # group別割り当て件数（analysis stallion 数）
    from collections import Counter as _Counter
    assigned_counts: dict[str, int] = _Counter(
        r["anchor_id"] for r in rows if r["anchor_id"]
    )

    anchor_tree_nodes = []
    for gid, aid in enumerate(anchor_ids):
        parent_anchor_id = ""
        parent_gid = -1
        cur = sire_of.get(aid)
        d = 1
        seen_at: set[str] = set()
        while cur and cur not in seen_at and d <= MAX_UP_DEPTH:
            seen_at.add(cur)
            if cur in anchor_set:
                parent_anchor_id = cur
                parent_gid = anchor_to_group[cur]
                break
            cur = sire_of.get(cur)
            d += 1

        node = {
            "id": aid,
            "name": name_of.get(aid, aid),
            "group_id": gid,
            "freq": freq_map.get(aid, 0),
            "assigned_count": assigned_counts.get(aid, 0),
            "parent_id": parent_anchor_id,
            "parent_group_id": parent_gid,
            "depth_to_parent": d if parent_anchor_id else 0,
        }
        anchor_tree_nodes.append(node)

    anchor_tree_path = OUT_PATH.parent / "anchor_tree.json"
    anchor_tree_path.write_text(
        _json.dumps({"nodes": anchor_tree_nodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("アンカーツリー保存: %s (%d ノード)", anchor_tree_path, len(anchor_tree_nodes))
    logger.info("完了 (%.1fs)", time.time() - t0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-anchors", type=int, default=30,
                        help="表示する系統グループ数 (default: 30)")
    args = parser.parse_args()
    build(max_anchors=args.max_anchors)


if __name__ == "__main__":
    main()
