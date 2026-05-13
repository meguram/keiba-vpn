"""
build_horse_bms_index.py
────────────────────────────────────────────────────────────────────────
non-000a/ 競走馬の「該当馬 × 母父（BMS）」ペアテーブルを構築して Parquet に保存する。
ツリーマップには表示しないが、データ分析用マスクデータとして利用する。

出力: data/research/pedigree_race_index/horse_bms.parquet
────────────────────────────────────────────────────────────────────────
スキーマ
  horse_id          : str  - 競走馬 ID
  sire_id           : str  - 父 (gen=1, pos=0)
  sire_name         : str
  bms_id            : str  - 母父 (gen=2, pos=2)
  bms_name          : str
  sire_root_id      : str  - 父の父系ルート horse_id
  sire_root_name    : str
  bms_root_id       : str  - 母父の父系ルート horse_id
  bms_root_name     : str
  sire_depth        : int  - 父の父系ツリー深さ（ルートからの距離）
  bms_depth         : int  - 母父の父系ツリー深さ
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_PED_DIR = _ROOT / "data" / "local" / "horse_pedigree_5gen"
_IDX_DIR = _ROOT / "data" / "research" / "pedigree_race_index"
_OUT_PATH = _IDX_DIR / "horse_bms.parquet"


def _build_sire_line_lookup(
    edges_path: Path,
    nodes_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """full_sire_tree から各種牡馬の父系ルートと深さを O(1) で引けるマップを構築。

    Returns:
        sire_of    : child_id → sire_id
        root_of    : horse_id → root_horse_id
        depth_of   : horse_id → depth_from_root
    """
    edges_df = pd.read_parquet(edges_path, columns=["sire_id", "child_id"])
    sire_of: dict[str, str] = dict(
        zip(edges_df["child_id"].astype(str), edges_df["sire_id"].astype(str))
    )

    nodes_df = pd.read_parquet(nodes_path)
    nodes_df["horse_id"] = nodes_df["horse_id"].astype(str)

    # is_root の馬は自身がルート
    root_nodes: set[str] = set(nodes_df.loc[nodes_df["is_root"] == True, "horse_id"])
    depth_of: dict[str, int] = dict(
        zip(nodes_df["horse_id"], nodes_df["depth_from_root"].astype(int))
    )
    name_of: dict[str, str] = dict(
        zip(nodes_df["horse_id"], nodes_df["name"].astype(str))
    )

    # 各ノードのルートをメモ化しながら辿る
    root_of: dict[str, str] = {}

    def _find_root(hid: str) -> str:
        if hid in root_of:
            return root_of[hid]
        visited = []
        cur = hid
        while cur not in root_of:
            if cur in root_nodes or cur not in sire_of:
                root_of[cur] = cur
                break
            visited.append(cur)
            cur = sire_of[cur]
        root = root_of.get(cur, cur)
        for v in visited:
            root_of[v] = root
        return root_of.get(hid, hid)

    for hid in nodes_df["horse_id"]:
        _find_root(hid)

    return sire_of, root_of, depth_of, name_of


def build(
    progress_cb: "Callable[[str, float], None] | None" = None,
) -> dict:
    """horse_bms.parquet を構築して保存する。

    Returns:
        stats dict (n_rows, elapsed_sec, etc.)
    """

    def _cb(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(msg, frac)

    t0 = time.time()

    # ── 1. 父系ルートマップを構築 ──────────────────────────────────────────
    _cb("父系ルートマップを構築中...", 0.0)
    edges_path = _IDX_DIR / "full_sire_tree.parquet"
    nodes_path = _IDX_DIR / "full_sire_tree_nodes.parquet"
    if not edges_path.exists() or not nodes_path.exists():
        raise FileNotFoundError(
            "full_sire_tree*.parquet が見つかりません。先にツリーを再構築してください。"
        )
    _, root_of, depth_of, name_of = _build_sire_line_lookup(edges_path, nodes_path)

    # ── 2. 競走馬 JSON を走査して (horse_id, sire, bms) を収集 ────────────────
    _cb("競走馬血統データを走査中...", 0.1)
    rows: list[dict] = []
    n_files = 0

    for dirpath, _dirs, files in os.walk(_PED_DIR):
        folder = os.path.basename(dirpath)
        if folder == "000a":
            continue
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue
            n_files += 1

            horse_id = str(rec.get("horse_id") or "").strip()
            if not horse_id:
                continue

            # 祖先を {(gen, pos): ancestor} に変換
            anc_map: dict[tuple, dict] = {}
            for a in rec.get("ancestors") or []:
                key = (int(a["generation"]), int(a["position"]))
                anc_map[key] = a

            sire_anc = anc_map.get((1, 0)) or {}
            bms_anc  = anc_map.get((2, 2)) or {}

            sire_id   = str(sire_anc.get("horse_id") or "").strip()
            sire_name = str(sire_anc.get("name") or "").strip()
            bms_id    = str(bms_anc.get("horse_id") or "").strip()
            bms_name  = str(bms_anc.get("name") or "").strip()

            rows.append({
                "horse_id":       horse_id,
                "sire_id":        sire_id,
                "sire_name":      sire_name,
                "bms_id":         bms_id,
                "bms_name":       bms_name,
            })

    _cb(f"{n_files:,} 件走査完了。父系情報を付与中...", 0.7)

    # ── 3. DataFrame 化して父系ルート・深さを付与 ──────────────────────────
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("有効なレコードが0件です。")

    # 父の父系ルート / 深さ
    df["sire_root_id"]   = df["sire_id"].map(lambda h: root_of.get(h, "") if h else "")
    df["sire_root_name"] = df["sire_root_id"].map(lambda h: name_of.get(h, h) if h else "")
    df["sire_depth"]     = df["sire_id"].map(lambda h: depth_of.get(h, -1) if h else -1)

    # 母父の父系ルート / 深さ
    df["bms_root_id"]    = df["bms_id"].map(lambda h: root_of.get(h, "") if h else "")
    df["bms_root_name"]  = df["bms_root_id"].map(lambda h: name_of.get(h, h) if h else "")
    df["bms_depth"]      = df["bms_id"].map(lambda h: depth_of.get(h, -1) if h else -1)

    # 列の順序を整理
    df = df[[
        "horse_id",
        "sire_id",   "sire_name",   "sire_root_id",   "sire_root_name",   "sire_depth",
        "bms_id",    "bms_name",    "bms_root_id",     "bms_root_name",    "bms_depth",
    ]]

    # ── 4. 保存 ────────────────────────────────────────────────────────────
    _cb("Parquet に保存中...", 0.9)
    _IDX_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT_PATH, index=False)

    elapsed = round(time.time() - t0, 1)
    n_with_bms  = int((df["bms_id"] != "").sum())
    n_with_sire = int((df["sire_id"] != "").sum())

    _cb(f"完了: {len(df):,} 件 ({elapsed}s)", 1.0)
    return {
        "n_rows":       len(df),
        "n_with_sire":  n_with_sire,
        "n_with_bms":   n_with_bms,
        "elapsed_sec":  elapsed,
        "out_path":     str(_OUT_PATH),
    }


if __name__ == "__main__":
    def _print_cb(msg: str, frac: float) -> None:
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg}", end="", flush=True)

    print("horse_bms.parquet を構築します...")
    stats = build(_print_cb)
    print()
    print(f"  総レコード数:         {stats['n_rows']:,}")
    print(f"  父あり:               {stats['n_with_sire']:,}")
    print(f"  母父あり:             {stats['n_with_bms']:,}")
    print(f"  出力先:               {stats['out_path']}")
    print(f"  所要時間:             {stats['elapsed_sec']}s")
