"""
全種牡馬 父系ツリー 構築スクリプト
===================================
horse_pedigree_5gen の全 JSON から、考えうる全ての「父(sire) → 子(child)」関係を抽出し、
全種牡馬の父系ツリーデータを構築・保存する。

【父-子 位置エンコーディング】
5世代血統表の各祖先は (generation, position) で一意に特定される。

  position 0 = 父系ルート（最左）
  generation=1, position=0 → 対象馬の父
  generation=1, position=1 → 対象馬の母

  generation=g, position=p の祖先は:
    ・父 = generation=g+1, position=p*2  （父系 = 偶数インデックス）
    ・母 = generation=g+1, position=p*2+1（母系 = 奇数インデックス）

よって「父-子」ペアとして抽出できる全ケース:
  1. (gen=1, pos=0).horse_id  →  horse_id（馬自身の父）
  2. gen=1..4, pos=0..2^g-1:
       (gen=g+1, pos=p*2).horse_id  →  (gen=g, pos=p).horse_id

【出力ファイル】
  data/research/pedigree_race_index/full_sire_tree.parquet
    columns: sire_id, sire_name, child_id, child_name
  data/research/pedigree_race_index/full_sire_tree_nodes.parquet
    columns: horse_id, name, n_children, n_descendants, depth_from_root
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

PED_DIR   = _ROOT / "data" / "local" / "horse_pedigree_5gen"
OUT_DIR   = _ROOT / "data" / "research" / "pedigree_race_index"
EDGES_OUT = OUT_DIR / "full_sire_tree.parquet"
NODES_OUT = OUT_DIR / "full_sire_tree_nodes.parquet"

_COUNTRY_SUFFIX = re.compile(r"\s*\([^)]{1,4}\)\s*$")


def _clean_name(s: str) -> str:
    """カナ・アルファベット後の国籍サフィックスを除去して正規化。"""
    return _COUNTRY_SUFFIX.sub("", s).strip()


from typing import Callable, Optional

_ProgressCallback = Optional[Callable[[str, float], None]]

# ── 強制ルートノード ──────────────────────────────────────────────────────────
# 歴史的に父系不明（基礎血統）であることが確認されている馬。
# netkeiba のポジションエンコードエラーにより誤った父が投票されるが、
# ここに列挙した horse_id は sire_of から強制削除してルートとして扱う。
#
# 判明している問題:
#   Godolphin Arabian (000a00f77b): Cade の母方祖父 Bald Galloway が
#     誤って pos=0 (父系ライン) に記録され「Bald Galloway の子」とされている。
#     実際は三大始祖の一頭で父系不明 = ROOT。
FORCED_ROOTS: frozenset[str] = frozenset({
    "000a00f77b",   # Godolphin Arabian（三大始祖）
})

# ── 強制削除エッジ（既知の誤りペア） ─────────────────────────────────────────
# (child_id, wrong_sire_id) の形式で列挙。投票後に sire_of から除去する。
# 問題:
#   Whynot (000a0159e0) が St.Victor's Barb の「父」として記録されているが、
#   実際は Whynot→Grey Whynot (牝馬)→Bald Galloway の母系関係であり、
#   St.Victor's Barb と Whynot は「Bald Galloway の父・母の父」であって父子ではない。
FORCED_REMOVE_EDGES: frozenset[tuple[str, str]] = frozenset({
    ("000a0159e3", "000a0159e0"),  # St.Victor's Barb の父 = Whynot（誤り）
})

# ── 牝馬名パターン ────────────────────────────────────────────────────────────
# 名前に「Mare」「Filly」「Dam」が単語として含まれる場合は牝馬とみなして除外する。
# ただし「Le Marechal」「Kenmare」「Damask」など部分一致のみの馬は対象外（単語境界で判定）。
import re as _re
_MARE_NAME_RE = _re.compile(
    r"(?:^|[\s\-'])(Mare|Filly|Broodmare)(?:$|[\s\-'])",
    _re.IGNORECASE,
)


def build(progress_cb: _ProgressCallback = None) -> dict:
    """
    ツリーを構築して保存する。

    Args:
        progress_cb: (message, fraction 0.0-1.0) を受け取るコールバック。
                     None の場合は logging のみ。

    Returns:
        統計情報 dict（n_nodes, n_edges, n_roots, elapsed_sec 等）
    """
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def _prog(msg: str, frac: float) -> None:
        logger.info(msg)
        if progress_cb:
            progress_cb(msg, frac)

    # ── 1. 全 JSON を走査して父-子ペアを抽出 ──────────────────────────────────
    # 【方針】各馬自身の JSON の gen=1,pos=0（直接の父）のみを使用する。
    #   - ケース1のみ: 各ファイルの主役馬の直父 (gen=1,pos=0) を採用
    #   - ケース2（祖先連鎖推定）は廃止: 5世代血統から間接的な父子関係を推定すると、
    #     ポジションエンコードエラー（牝馬が偶数 pos に誤記録される等）により
    #     誤った父系接続が生じる（例: Brimmer Mare → Lister Turk 等）
    _prog(f"horse_pedigree_5gen を全走査中 ({PED_DIR})...", 0.0)

    sire_of:  dict[str, str] = {}   # child_id -> sire_id  (ケース1のみ、1ファイル1票)
    name_of:  dict[str, str] = {}   # horse_id -> canonical name
    sex_of:   dict[str, str] = {}   # horse_id -> '牡'/'牝'/''

    def _register_name(hid: str, raw_name: str) -> None:
        if not hid or not raw_name:
            return
        cname = _clean_name(raw_name)
        if not cname:
            return
        prev = name_of.get(hid)
        if prev is None or len(cname) < len(prev):
            name_of[hid] = cname

    # ファイル総数を先読み（進捗率計算用）
    all_files = sorted(PED_DIR.rglob("*.json"))
    total_files = len(all_files)
    _prog(f"対象ファイル数: {total_files:,}", 0.02)

    file_count = 0
    for f in all_files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        hid = str(rec.get("horse_id") or "").strip()
        if not hid:
            file_count += 1
            continue

        # 馬名・性別の登録
        _register_name(hid, str(rec.get("horse_name") or rec.get("name") or ""))
        sex = str(rec.get("sex") or "").strip()
        if sex and hid not in sex_of:
            sex_of[hid] = sex

        # 祖先の名前登録（sex フィールドも収集）
        for a in rec.get("ancestors") or []:
            aid = str(a.get("horse_id") or "").strip()
            aname = str(a.get("name") or "")
            asex  = str(a.get("sex") or "").strip()
            if not aid:
                continue
            _register_name(aid, aname)
            if asex and aid not in sex_of:
                sex_of[aid] = asex

        # ── ケース1のみ: gen=1, pos=0 = 直接の父 ──────────────────────────
        sire_id = ""
        for a in rec.get("ancestors") or []:
            if a.get("generation") == 1 and a.get("position") == 0:
                sire_id = str(a.get("horse_id") or "").strip()
                break
        if sire_id and sire_id != hid:
            # 既に登録済みの場合は上書きしない（最初に見つかった直接の父を採用）
            if hid not in sire_of:
                sire_of[hid] = sire_id

        file_count += 1
        if file_count % 5000 == 0:
            frac = 0.02 + 0.55 * (file_count / max(total_files, 1))
            _prog(f"  {file_count:,} / {total_files:,} ファイル処理済み...", frac)

    _prog(f"全 {file_count:,} ファイル処理完了 ({time.time() - t0:.1f}s)", 0.58)
    _prog(f"  父-子ペア（ケース1のみ）: {len(sire_of):,}", 0.59)
    _prog(f"  名前登録: {len(name_of):,} 頭  / 性別登録: {len(sex_of):,} 頭", 0.60)

    # ── 2. 牝馬除去フィルタ ─────────────────────────────────────────────────
    # 判定優先順:
    #   a) sex_of[hid] == '牝' → 確実に牝馬（スクレイプ時に取得）
    #   b) 名前に '牝馬' パターン (Mare/Filly/Broodmare) が含まれる → 牝馬とみなす
    #      ただし Kenmare / Le Marechal / Damask 等の誤検知を避けるため単語境界で判定

    def _is_mare(hid: str) -> bool:
        if sex_of.get(hid) == "牝":
            return True
        nm = name_of.get(hid, "")
        return bool(_MARE_NAME_RE.search(nm))

    sire_of_filtered: dict[str, str] = {}
    mare_removed = 0
    for child_id, sire_id in sire_of.items():
        if _is_mare(child_id):
            mare_removed += 1
            continue           # 牝馬が child なら除去
        if _is_mare(sire_id):
            mare_removed += 1
            continue           # 牝馬が父に登録されている（エンコードエラー）なら除去
        sire_of_filtered[child_id] = sire_id
    sire_of = sire_of_filtered
    _prog(f"確定 父-子ペア（牝馬除去後）: {len(sire_of):,}  (除去 {mare_removed})", 0.65)

    # ── 2c. 強制ルート & 既知誤りエッジの除去 ───────────────────────────────────
    # FORCED_ROOTS: 歴史的に父系不明の基礎馬 → sire_of から強制削除してルートへ
    for hid in FORCED_ROOTS:
        if hid in sire_of:
            removed_sire = sire_of.pop(hid)
            logger.info(
                "FORCED_ROOT: %s (%s) の誤った父 %s (%s) を削除してルートとして扱う",
                name_of.get(hid, hid), hid,
                name_of.get(removed_sire, removed_sire), removed_sire,
            )
    # FORCED_REMOVE_EDGES: 既知の誤った父子ペアを除去
    for child_id, wrong_sire_id in FORCED_REMOVE_EDGES:
        if sire_of.get(child_id) == wrong_sire_id:
            sire_of.pop(child_id)
            logger.info(
                "FORCED_REMOVE: %s (%s) → 誤った父 %s (%s) を削除",
                name_of.get(child_id, child_id), child_id,
                name_of.get(wrong_sire_id, wrong_sire_id), wrong_sire_id,
            )

    _prog(f"確定 父-子ペア（強制補正後）: {len(sire_of):,}", 0.66)

    # ── 3. 全ノード収集 ──────────────────────────────────────────────────────
    all_ids: set[str] = set(sire_of.keys()) | set(sire_of.values())
    # name_of に無い ID はそのまま ID を名前として使う
    for hid in all_ids:
        if hid not in name_of:
            name_of[hid] = hid

    _prog(f"全ノード数: {len(all_ids):,}", 0.68)

    # ── 4. 子リスト構築 (children_of) ───────────────────────────────────────
    children_of: dict[str, list[str]] = {hid: [] for hid in all_ids}
    for child_id, sire_id in sire_of.items():
        children_of.setdefault(sire_id, []).append(child_id)

    # ── 5. ルートノード (父が不明) ────────────────────────────────────────────
    roots: list[str] = sorted(
        all_ids - set(sire_of.keys()),
        key=lambda x: -len(children_of.get(x, []))
    )
    _prog(f"ルートノード数: {len(roots):,}", 0.72)

    # ── 6. 各ノードの子孫数をトポロジカル BFS で計算 ─────────────────────────
    _prog("子孫数 計算中...", 0.75)
    n_descendants: dict[str, int] = {hid: 0 for hid in all_ids}
    depth_of:      dict[str, int] = {hid: -1 for hid in all_ids}

    # BFS (roots から)
    q: deque[tuple[str, int]] = deque()
    for rid in roots:
        q.append((rid, 0))
        depth_of[rid] = 0

    visited_bfs: set[str] = set()
    topo_order: list[str] = []
    while q:
        nid, d = q.popleft()
        if nid in visited_bfs:
            continue
        visited_bfs.add(nid)
        topo_order.append(nid)
        for ch in children_of.get(nid, []):
            if ch not in visited_bfs:
                if depth_of[ch] == -1:
                    depth_of[ch] = d + 1
                q.append((ch, d + 1))

    # 残り (ループ状のデータや孤立ノード)
    for nid in all_ids - visited_bfs:
        topo_order.append(nid)

    # トポロジカル逆順で子孫数を積み上げ
    for nid in reversed(topo_order):
        kids = children_of.get(nid, [])
        n_descendants[nid] = len(kids) + sum(n_descendants.get(k, 0) for k in kids)

    # ── 7. Parquet 保存 ──────────────────────────────────────────────────────
    _prog("Parquet 保存中...", 0.88)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # エッジリスト
    edge_rows = [
        {
            "sire_id":    sire_id,
            "sire_name":  name_of.get(sire_id, sire_id),
            "child_id":   child_id,
            "child_name": name_of.get(child_id, child_id),
        }
        for child_id, sire_id in sire_of.items()
    ]
    edges_df = pd.DataFrame(edge_rows)
    edges_df.to_parquet(EDGES_OUT, index=False, compression="zstd")
    _prog(f"エッジ保存: {len(edges_df):,} ペア  ({EDGES_OUT.stat().st_size / 1024:.1f} KB)", 0.92)

    # ノードリスト
    node_rows = [
        {
            "horse_id":        hid,
            "name":            name_of.get(hid, hid),
            "n_children":      len(children_of.get(hid, [])),
            "n_descendants":   n_descendants.get(hid, 0),
            "depth_from_root": depth_of.get(hid, -1),
            "is_root":         hid in set(roots),
        }
        for hid in all_ids
    ]
    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_parquet(NODES_OUT, index=False, compression="zstd")
    _prog(f"ノード保存: {len(nodes_df):,} 頭  ({NODES_OUT.stat().st_size / 1024:.1f} KB)", 0.96)

    # ── 8. サマリー ──────────────────────────────────────────────────────────
    max_depth = max((v for v in depth_of.values() if v >= 0), default=0)
    elapsed = time.time() - t0
    stats = {
        "n_nodes":   len(all_ids),
        "n_edges":   len(sire_of),
        "n_roots":   len(roots),
        "max_depth": max_depth,
        "n_files":   file_count,
        "elapsed_sec": round(elapsed, 1),
    }
    _prog(
        f"完了: ノード {stats['n_nodes']:,} / エッジ {stats['n_edges']:,} / "
        f"ルート {stats['n_roots']:,} / 最大深さ {stats['max_depth']} "
        f"({elapsed:.1f}s)",
        1.0,
    )
    return stats


def main() -> None:
    stats = build()
    logger.info("統計: %s", stats)


if __name__ == "__main__":
    main()
