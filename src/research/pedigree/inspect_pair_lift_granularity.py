"""pair_lift の粒度別 件数 / 平均サンプル数を点検し、
「母父個体」 vs 「母父系統」のどちらが実用的かを判定する。

使用:
  python -m src.research.pedigree.inspect_pair_lift_granularity
"""
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "data/page_reference/note_aptitude_race"
IDX = ROOT / "data/page_reference/pedigree_race_index"

print("="*80)
print("【1】 現在の 3 階層 pair_lift プロファイル")
print("="*80)
for label, fn, keys in [
    ("PAIR_INDIV  (父個体 × 母父個体)",   "pair_lift_profiles_indiv.parquet",  ["sire_id", "bms_id"]),
    ("PAIR_GROUP  (父個体 × 母父5大系統)", "pair_lift_profiles_group.parquet",  ["sire_id", "bms_main"]),
    ("PAIR_GXG    (父5系統 × 母父5系統)",  "pair_lift_profiles_gxg.parquet",    ["sire_main", "bms_main"]),
]:
    p = ART / fn
    if not p.exists():
        print(f"\n{label}: 未生成")
        continue
    df = pd.read_parquet(p)
    n = len(df)
    nh_mean = df["n_horses"].mean() if "n_horses" in df.columns else None
    nr_mean = df["n_records"].mean() if "n_records" in df.columns else None
    nr_med = df["n_records"].median() if "n_records" in df.columns else None
    print(f"\n{label}")
    print(f"  ペア数: {n:,}")
    print(f"  n_horses 平均: {nh_mean:.1f}  / n_records 平均: {nr_mean:.0f} 中央値: {nr_med:.0f}")
    if "bms_main" in df.columns:
        print(f"  母父系統別 ペア数: {dict(df['bms_main'].value_counts().head())}")

# ────────────────────────────────────────────────────────────
# bms_root_id (=母父の父系ルート種牡馬) の粒度を調査
#   - これが "母父系" として中間粒度になる可能性
# ────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("【2】 母父 horse_bms.parquet の粒度別 件数")
print("="*80)
bms = pd.read_parquet(IDX / "horse_bms.parquet")
print(f"  全件: {len(bms):,}")
print(f"  ユニーク bms_id (母父個体):       {bms['bms_id'].nunique():,}")
if "bms_root_id" in bms.columns:
    print(f"  ユニーク bms_root_id (母父父系root): {bms['bms_root_id'].nunique():,}")
sid_to_main: dict[str, str] = json.loads((ART / "sid_to_main.json").read_text())
bms["bms_main"] = bms["bms_id"].map(sid_to_main)
print(f"  ユニーク bms_main (5大系統):       {bms['bms_main'].nunique():,}")
print(f"    → 内訳: {dict(bms['bms_main'].value_counts())}")

# 母父 root の上位 (主要な母父系統)
if "bms_root_id" in bms.columns and "bms_root_name" in bms.columns:
    print(f"\n  bms_root_name (母父父系) 上位 15:")
    top = bms.groupby(["bms_root_id", "bms_root_name"]).size().reset_index(name="n_horses").sort_values("n_horses", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"    {r['bms_root_name']:25s} {r['n_horses']:6,}頭")

# ────────────────────────────────────────────────────────────
# シミュレーション: 父個体 × 母父root でペアを組んだら何ペアになるか?
# ────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("【3】 提案: 父個体 × 母父 父系root  でペア化した場合のシミュレーション")
print("="*80)

if "bms_root_id" in bms.columns:
    bms["sire_id_str"] = bms["sire_id"].astype(str)
    bms["bms_root_id_str"] = bms["bms_root_id"].astype(str)

    # 各 (sire, bms_root) ペアに何頭の馬が紐付くか
    pairs = bms.groupby(["sire_id_str", "bms_root_id_str"]).size().reset_index(name="n_horses")

    print(f"  ユニーク (父個体 × 母父root) ペア数: {len(pairs):,}")
    for n_min in (3, 5, 10, 20):
        n_pairs = (pairs["n_horses"] >= n_min).sum()
        n_horses_total = pairs.loc[pairs["n_horses"] >= n_min, "n_horses"].sum()
        print(f"    n_horses>={n_min:2d}: {n_pairs:,} ペア, 紐付く産駒の総数 = {n_horses_total:,}")

# 比較: 既存の (sire_id × bms_main) と件数比較
print(f"\n  既存 PAIR_GROUP (父個体 × 母父5大系統) の n_records>=50 ペア数: 759 (実データから)")
print(f"  既存 PAIR_INDIV (父個体 × 母父個体)    の n_records>=80 ペア数: 442 (実データから)")
