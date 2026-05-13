"""
rebuild_stallion_lineage.py
────────────────────────────────────────────────────────────────────────
日本競馬の「4大主流系統」に基づく階層グルーピングで stallion_lineage を再構築する。

大グループ（main_group）:
  1 = Turn-To系       (000a001042)
  2 = Northern Dancer系 (000a000e04)
  3 = Native Dancer系   (000a000f89)
  4 = Nasrullah系     (000a000f88)
  5 = 非主流

小グループ（sub_group）:
  各大グループ内で、horse_pedigree_cats での出現頭数が SUBGROUP_MIN_COUNT 以上の
  中間祖先ノードを「統計的に有意な小グループ」として採用する。
  閾値未満のノードは大グループの "その他" として扱う。

手法:
  horse_pedigree_5gen/ の全 JSON から gen=1 pos=0 の直父チェーンを構築し、
  各 stallion_id の祖先パスを辿って最初にマッチするアンカーで分類する。

出力:
  data/research/pedigree_race_index/stallion_lineage.parquet
  data/research/pedigree_race_index/stallion_lineage_meta.json
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

_ROOT    = Path(__file__).resolve().parents[3]
_IDX_DIR = _ROOT / "data" / "research" / "pedigree_race_index"
_PED_DIR = _ROOT / "data" / "local" / "horse_pedigree_5gen"

# サブグループを名付けるための最小出現頭数閾値
SUBGROUP_MIN_COUNT = 100

# ── 4大系統 大グループ定義 ─────────────────────────────────────────────────────
MAIN_GROUPS: dict[str, tuple[int, str]] = {
    "000a001042": (1, "Turn-To系"),
    "000a000e04": (2, "Northern Dancer系"),
    "000a000f89": (3, "Native Dancer系"),
    "000a000f88": (4, "Nasrullah系"),
}
MAIN_GROUP_OTHER = (5, "非主流")

# ── サブグループ候補アンカー（既知の重要中間祖先） ───────────────────────────────
# (horse_id, main_group_id, sub_group_label, priority)
# priority が小さいほど優先（子孫に近い方を優先）
SUB_ANCHORS: list[tuple[str, int, str, int]] = [
    # ━━ Turn-To系 (1) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Halo 系列 (優先度順: 孫 < 子 < Halo)
    ("000a00033a", 1, "Sunday Silence系",   10),   # Sunday Silence
    ("000a001a8f", 1, "Devil's Bag系",      10),   # Devil's Bag
    ("000a00031e", 1, "Southern Halo系",    10),   # サザンヘイロー
    ("000a0012bf", 1, "Halo系",            20),   # Halo (上記を束ねる)
    # Roberto 系列
    ("000a0019b4", 1, "Silver Hawk系",      10),   # Silver Hawk
    ("000a0016f2", 1, "Kris S.系",          10),   # Kris S.
    ("000a000082", 1, "Brian's Time系",     10),   # ブライアンズタイム
    ("000a0012cb", 1, "Roberto系",          20),   # Roberto (上記を束ねる)
    # Sir Gaylord 系列
    ("000a000dd9", 1, "Sir Ivor系",         10),   # Sir Ivor
    ("000a00184f", 1, "Sir Tristram系",     10),   # Sir Tristram
    ("000a000dda", 1, "Sir Gaylord系",      20),   # Sir Gaylord
    # Hail to Reason
    ("000a000f2b", 1, "Hail to Reason系",   30),   # Hail to Reason
    # ━━ Northern Dancer系 (2) ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Storm Bird 系列
    ("000a001a98", 2, "Storm Cat系",        10),   # Storm Cat
    ("000a011d00", 2, "War Front系",        10),   # War Front (Danzig 系)
    # Danzig 系列
    ("000a0000d3", 2, "Danzig系",           10),   # Danzig (correct id)
    # Sadler's Wells 系列
    ("000a00232b", 2, "Galileo系",          10),   # Galileo
    ("1996190001", 2, "Montjeu系",          10),   # Montjeu
    ("000a00185d", 2, "Sadler's Wells系",   20),   # Sadler's Wells (Galileo の親)
    # Nijinsky 系列
    ("000a000dfe", 2, "Nijinsky系",         10),   # Nijinsky
    # Lyphard / Nureyev 系列
    ("000a001676", 2, "Nureyev系",          10),   # Nureyev
    ("000a001205", 2, "Lyphard系",          20),   # Lyphard (Nureyev の親)
    # ━━ Native Dancer系 (3) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ("000a001d7e", 3, "Kingmambo系",        10),   # Kingmambo
    ("000a001cd0", 3, "Unbridled系",        10),   # Unbridled
    ("000a001be1", 3, "Gone West系",        10),   # Gone West
    ("000a001607", 3, "Mr. Prospector系",   20),   # Mr. Prospector (Kingmambo の親)
    ("000a0010e2", 3, "In Reality系",       10),   # In Reality
    # ━━ Nasrullah系 (4) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ("000a001d37", 4, "A.P. Indy系",        10),   # A.P. Indy
    ("000a0015fc", 4, "Seattle Slew系",     15),   # Seattle Slew (A.P. Indy の親)
    ("000a000ded", 4, "Secretariat系",      20),   # Secretariat (Seattle Slew の親)
    ("000a000e94", 4, "Bold Ruler系",       25),   # Bold Ruler (Secretariat の親)
    # ━━ 非主流 (5) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ("000a000fdf", 5, "Ribot系",            10),   # Ribot
]
# horse_id → (main_group_id, sub_group_label, priority)
_SUB_MAP: dict[str, tuple[int, str, int]] = {
    hid: (mg, sl, p) for hid, mg, sl, p in SUB_ANCHORS
}

# ── カラーパレット ─────────────────────────────────────────────────────────────
# 大グループごとに色相を固定、小グループで明暗を変える
_PALETTE: dict[tuple[int, str], str] = {
    # Turn-To系 (1) → 橙/黄系
    (1, "Sunday Silence系"):   "#f59e0b",
    (1, "Devil's Bag系"):      "#fb923c",
    (1, "Southern Halo系"):    "#fbbf24",
    (1, "Halo系"):             "#f97316",
    (1, "Silver Hawk系"):      "#d97706",
    (1, "Kris S.系"):          "#b45309",
    (1, "Brian's Time系"):     "#92400e",
    (1, "Roberto系"):          "#c2410c",
    (1, "Sir Ivor系"):         "#ea580c",
    (1, "Sir Tristram系"):     "#9a3412",
    (1, "Sir Gaylord系"):      "#c2410c",
    (1, "Hail to Reason系"):   "#e8520c",
    (1, "Turn-To系その他"):      "#f0a060",
    # Northern Dancer系 (2) → 青/紫系
    (2, "Storm Cat系"):        "#3b82f6",
    (2, "War Front系"):        "#1d4ed8",
    (2, "Danzig系"):           "#4f46e5",
    (2, "Galileo系"):          "#7c3aed",
    (2, "Montjeu系"):          "#8b5cf6",
    (2, "Sadler's Wells系"):   "#6d28d9",
    (2, "Nijinsky系"):         "#a855f7",
    (2, "Nureyev系"):          "#c084fc",
    (2, "Lyphard系"):          "#d8b4fe",
    (2, "Northern Dancer系その他"): "#60a5fa",
    # Native Dancer系 (3) → 緑系
    (3, "Kingmambo系"):        "#22c55e",
    (3, "Unbridled系"):        "#16a34a",
    (3, "Gone West系"):        "#15803d",
    (3, "Mr. Prospector系"):   "#166534",
    (3, "In Reality系"):       "#4d7c0f",
    (3, "Native Dancer系その他"): "#4ade80",
    # Nasrullah系 (4) → 赤/ピンク系
    (4, "A.P. Indy系"):        "#ef4444",
    (4, "Seattle Slew系"):     "#dc2626",
    (4, "Secretariat系"):      "#b91c1c",
    (4, "Bold Ruler系"):       "#991b1b",
    (4, "Nasrullah系その他"):    "#f87171",
    # 非主流 (5) → グレー系
    (5, "Ribot系"):            "#a78bfa",
    (5, "非主流"):             "#6b7280",
}

# 大グループ代表色
_MAIN_COLOR = {
    1: "#f97316",   # Turn-To系 (橙)
    2: "#3b82f6",   # Northern Dancer系 (青)
    3: "#22c55e",   # Native Dancer系 (緑)
    4: "#ef4444",   # Nasrullah系 (赤)
    5: "#9ca3af",   # 非主流 (グレー)
}


def build(progress_cb: "Callable[[str, float], None] | None" = None) -> dict:
    def _cb(msg: str, frac: float) -> None:
        if progress_cb:
            progress_cb(msg, frac)

    t0 = time.time()

    # ── 1. pedigree JSON から直父チェーン (sire_from_json) を構築 ──────────────
    _cb("全 pedigree JSON から直父チェーンを構築中...", 0.0)
    sire_from_json: dict[str, str] = {}   # horse_id → gen=1,pos=0 sire_id
    name_from_json: dict[str, str] = {}

    for dirpath, _, files in os.walk(_PED_DIR):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(dirpath, fname), encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue
            hid = str(rec.get("horse_id") or "").strip()
            if not hid:
                continue
            # 名前登録
            name = str(rec.get("horse_name") or rec.get("name") or "").strip()
            if name and hid not in name_from_json:
                name_from_json[hid] = name
            # 直父
            for a in rec.get("ancestors") or []:
                if a.get("generation") == 1 and a.get("position") == 0:
                    sid = str(a.get("horse_id") or "").strip()
                    if sid and hid not in sire_from_json:
                        sire_from_json[hid] = sid
                    # 祖先の名前も登録
                    aname = str(a.get("name") or "").strip()
                    aid   = str(a.get("horse_id") or "").strip()
                    if aname and aid and aid not in name_from_json:
                        name_from_json[aid] = aname
                    break

    _cb(f"直父チェーン {len(sire_from_json):,} 件構築完了", 0.25)

    # ── 2. 祖先パス探索関数 ───────────────────────────────────────────────────
    _cache: dict[str, list[str]] = {}

    def get_ancestor_path(hid: str, max_depth: int = 12) -> list[str]:
        """gen=1 pos=0 の父系ラインを辿った祖先リスト（自身を含まない）"""
        if hid in _cache:
            return _cache[hid]
        path: list[str] = []
        cur = hid
        seen: set[str] = {hid}
        for _ in range(max_depth):
            parent = sire_from_json.get(cur)
            if not parent or parent in seen:
                break
            path.append(parent)
            seen.add(parent)
            cur = parent
        _cache[hid] = path
        return path

    # ── 3. horse_pedigree_cats ロード ─────────────────────────────────────────
    _cb("horse_pedigree_cats をロード中...", 0.3)
    cats = pd.read_parquet(
        _IDX_DIR / "horse_pedigree_cats.parquet",
        columns=["stallion_id", "stallion_name", "horse_id"],
    )
    cats["stallion_id"]   = cats["stallion_id"].astype(str)
    cats["stallion_name"] = cats["stallion_name"].astype(str)

    canon_name = (
        cats.groupby("stallion_id", observed=True)["stallion_name"]
        .agg(lambda x: min(x, key=len))
    )
    horse_cnt = cats.groupby("stallion_id", observed=True)["horse_id"].nunique()
    unique_ids = [s for s in cats["stallion_id"].unique() if s and s != "nan"]

    # ── 4. 大グループ候補セット（self → 祖先 の中で最初にマッチするものを採用）──
    main_anchor_set = set(MAIN_GROUPS)
    sub_anchor_set  = set(_SUB_MAP)
    all_anchor_set  = main_anchor_set | sub_anchor_set

    def classify(hid: str) -> tuple[int, str, int, str, str]:
        """
        Returns:
            (main_gid, main_name, sub_gid, sub_name, anchor_id)

        優先順位:
          1. self が main_group アンカー → その他 として扱う
          2. self が sub_anchor → そのサブグループの "代表" として分類
          3. 祖先パスから最近の sub_anchor を探す
          4. 祖先パスから main_group アンカーを探す → その他
          5. 未分類 → 非主流
        """
        # 1. self が main group anchor (Turn-To, Northern Dancer 等) 自身
        if hid in MAIN_GROUPS:
            mg, mn = MAIN_GROUPS[hid]
            other_label = f"{mn}その他"
            return mg, mn, 0, other_label, hid

        # 2. self が sub_anchor (Roberto, Halo, Sunday Silence 等) 自身
        #    → 自分が定義するサブグループのルートとして分類
        if hid in _SUB_MAP:
            mg, sl, p = _SUB_MAP[hid]
            mn = _sub_to_main_name(mg)
            return mg, mn, mg * 100 + p, sl, hid

        # 3 & 4. 祖先パスを辿る
        path = get_ancestor_path(hid)
        best_anchor_id = ""

        for ancestor in path:
            if ancestor in _SUB_MAP:
                mg, sl, p = _SUB_MAP[ancestor]
                mn = _sub_to_main_name(mg)
                return mg, mn, mg * 100 + p, sl, ancestor

        for ancestor in path:
            if ancestor in MAIN_GROUPS:
                mg, mn = MAIN_GROUPS[ancestor]
                other_label = f"{mn}その他"
                return mg, mn, 0, other_label, ancestor

        # 5. 未分類 → 非主流
        return 5, "非主流", 500, "非主流", ""

    def _sub_to_main_name(mg: int) -> str:
        names = {1: "Turn-To系", 2: "Northern Dancer系",
                 3: "Native Dancer系", 4: "Nasrullah系", 5: "非主流"}
        return names.get(mg, "非主流")

    # ── 5. 全 stallion_id を分類 ──────────────────────────────────────────────
    _cb("系統分類中...", 0.5)
    rows: list[dict] = []
    for sid in unique_ids:
        mg, mn, sg, sl, anchor_id = classify(sid)
        rows.append({
            "stallion_id":    sid,
            "stallion_name":  str(canon_name.get(sid, name_from_json.get(sid, sid))),
            "anchor_id":      anchor_id,
            "anchor_name":    str(name_from_json.get(anchor_id, anchor_id)),
            "depth_to_anchor": len(get_ancestor_path(sid)),
            "group_id":       sg,
            "main_group_id":  mg,
            "main_group_name": mn,
            "sub_group_label": sl,
        })

    lin_df = pd.DataFrame(rows)

    # ── 6. 統計的有意性チェック: サブグループの出現頭数でフィルタ ────────────────
    # 出現頭数が SUBGROUP_MIN_COUNT 未満のサブグループは大グループ「その他」に格下げ
    _cb("統計フィルタ適用中...", 0.75)
    sub_horse_cnt = (
        lin_df.merge(
            horse_cnt.reset_index().rename(columns={"horse_id": "n_horses"}),
            on="stallion_id", how="left",
        )
        .groupby("sub_group_label")["n_horses"]
        .sum()
    )
    for sg_label, cnt_val in sub_horse_cnt.items():
        if cnt_val < SUBGROUP_MIN_COUNT and "その他" not in str(sg_label):
            # 大グループ「その他」に格下げ
            mask = lin_df["sub_group_label"] == sg_label
            mg_val = lin_df.loc[mask, "main_group_id"].iloc[0] if mask.any() else 5
            mn_val = lin_df.loc[mask, "main_group_name"].iloc[0] if mask.any() else "非主流"
            other_lbl = f"{mn_val}その他"
            lin_df.loc[mask, "sub_group_label"] = other_lbl
            lin_df.loc[mask, "group_id"] = 0

    # group_id を (main_group_id, sub_group_label) の組み合わせで再番号付け
    combo = lin_df[["main_group_id", "sub_group_label"]].drop_duplicates()
    combo = combo.sort_values(["main_group_id", "sub_group_label"])
    gid_map: dict[tuple, int] = {
        (int(r["main_group_id"]), r["sub_group_label"]): i
        for i, (_, r) in enumerate(combo.iterrows())
    }
    lin_df["group_id"] = lin_df.apply(
        lambda r: gid_map[(int(r["main_group_id"]), r["sub_group_label"])], axis=1
    )

    # ── 7. meta 構築 ──────────────────────────────────────────────────────────
    _cb("meta 構築中...", 0.85)
    merged = lin_df.merge(
        horse_cnt.reset_index().rename(columns={"horse_id": "n_horses"}),
        on="stallion_id", how="left",
    )
    grp_meta = (
        merged.groupby(["group_id", "main_group_id", "main_group_name", "sub_group_label"])
        ["n_horses"].sum()
        .reset_index(name="count")
        .sort_values(["main_group_id", "count"], ascending=[True, False])
    )
    meta: list[dict] = []
    for _, row in grp_meta.iterrows():
        gid  = int(row["group_id"])
        mg   = int(row["main_group_id"])
        mn   = str(row["main_group_name"])
        sl   = str(row["sub_group_label"])
        color = _PALETTE.get((mg, sl), _MAIN_COLOR.get(mg, "#888"))
        meta.append({
            "group_id":        gid,
            "main_group_id":   mg,
            "main_group_name": mn,
            "anchor_name":     sl,
            "count":           int(row["count"]),
            "color":           color,
            "main_color":      _MAIN_COLOR.get(mg, "#888"),
        })

    # ── 8. 保存 ────────────────────────────────────────────────────────────────
    _cb("保存中...", 0.92)
    _IDX_DIR.mkdir(parents=True, exist_ok=True)
    lin_df.to_parquet(_IDX_DIR / "stallion_lineage.parquet", index=False)
    with open(_IDX_DIR / "stallion_lineage_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    elapsed = round(time.time() - t0, 1)
    n_classified = int((lin_df["main_group_id"] < 5).sum())
    _cb(f"完了: {len(lin_df):,} 件  主流分類: {n_classified:,}  ({elapsed}s)", 1.0)
    return {
        "n_stallions":  len(lin_df),
        "n_classified": n_classified,
        "n_subgroups":  len(meta),
        "elapsed_sec":  elapsed,
    }


if __name__ == "__main__":
    def _cb(msg: str, frac: float) -> None:
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg}", end="", flush=True)

    print("stallion_lineage (4大系統階層) を再構築します...")
    stats = build(_cb)
    print()
    print(f"  stallion 総数:     {stats['n_stallions']:,}")
    print(f"  4大主流 分類済み:   {stats['n_classified']:,}")
    print(f"  サブグループ数:     {stats['n_subgroups']}")
    print(f"  所要時間:           {stats['elapsed_sec']}s")

    import json as _json
    meta = _json.loads((Path(__file__).resolve().parents[3] / "data/research/pedigree_race_index/stallion_lineage_meta.json").read_text())
    print("\n系統グループ一覧:")
    for g in meta:
        print(f"  [{g['main_group_name']:18s}]  {g['anchor_name']:25s}  count={g['count']:7,}  color={g['color']}")
