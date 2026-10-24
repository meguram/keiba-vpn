"""dev_pct (±%pt) の計算定義を実数で精査する検証スクリプト。

確認したいこと:
  ★ profile.strengths/weaknesses の dev_pct の正体は何か?
  (a) 該当 L2 と「それ以外の種牡馬」を絶対勝率で比較した偏差?
  (b) 該当 L2 内の各種牡馬が「自身の通算勝率」と比べた条件別倍率の平均?
  (c) その他?

予測利用に重要なので、絶対勝率比較の数値も同時に算出して比較する。

リポジトリルートで: python tests/research/manual/verify_dev_pct_meaning.py
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "data/research/bloodline_meta_cluster"
TARGET_L2 = 4   # タイトルホルダーが属する「東京型」
TARGET_COND = "win_v_東京"  # +9.46pt と表示されている条件 (実カラム名)

uni = pd.read_parquet(ART / "unified.parquet")
print(f"unified.parquet shape={uni.shape}")
print(f"columns sample: {list(uni.columns)[:6]} ...")

stallions = uni[uni["entity_type"] == "stallion"].copy()
print(f"全種牡馬数: {len(stallions)}")

target = stallions[stallions["L2"] == TARGET_L2].copy()
others = stallions[stallions["L2"] != TARGET_L2].copy()
print(f"L2={TARGET_L2} に属する種牡馬数: {len(target)}, それ以外: {len(others)}")

# ────────────────────────────────────────────────────────────
# 1. 現在の dev_pct (= "L2 内の各種牡馬の自身平均比 lift の平均 - 1.0")
# ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("【現状の dev_pct の計算】")
print("="*70)

base_target = target["win_eb_total"].clip(lower=0.001)
rel_target  = target[TARGET_COND] / base_target  # 各種牡馬: 条件別 / 自身通算
mean_rel    = rel_target.mean()
dev_pct_now = (mean_rel - 1.0) * 100.0

print(f"\n  L2={TARGET_L2} の各種牡馬で計算:")
print(f"    rel_i = {TARGET_COND} / win_eb_total  (各種牡馬の自身平均比)")
print(f"    rel.mean() = {mean_rel:.4f}")
print(f"    (rel.mean() - 1.0) * 100 = {dev_pct_now:+.2f} pt")
print(f"\n  → これは UI に表示されている '+9.46pt' と一致するはず")

# 主要種牡馬の rel を表示
target_with_rel = target.copy()
target_with_rel["rel"] = rel_target
print(f"\n  L2={TARGET_L2} 内の代表種牡馬 (top 8 by win_eb_total):")
top = target_with_rel.nlargest(8, "win_eb_total")[["entity_label", "win_eb_total", TARGET_COND, "rel"]]
top["dev"] = (top["rel"] - 1.0) * 100
for _, r in top.iterrows():
    print(f"    {r['entity_label']:20s}  通算={r['win_eb_total']:.4f}  東京={r[TARGET_COND]:.4f}  rel={r['rel']:.3f}  dev={r['dev']:+.1f}pt")

# ────────────────────────────────────────────────────────────
# 2. もし「該当L2 vs それ以外L2」を絶対勝率で比較したら? (案A)
# ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("【参考: 該当 L2 vs それ以外L2 を 「絶対勝率」 で比較した場合】")
print("="*70)

mean_target_abs = target[TARGET_COND].mean()
mean_others_abs = others[TARGET_COND].mean()
abs_diff_pct    = (mean_target_abs - mean_others_abs) * 100  # 絶対勝率の差 (pt)
abs_lift_ratio  = (mean_target_abs / mean_others_abs - 1.0) * 100

print(f"\n  L2={TARGET_L2} の {TARGET_COND} 平均:    {mean_target_abs:.4f}  ({mean_target_abs*100:.2f}%)")
print(f"  L2!={TARGET_L2} の {TARGET_COND} 平均:    {mean_others_abs:.4f}  ({mean_others_abs*100:.2f}%)")
print(f"  絶対勝率の差 (target - others):  {abs_diff_pct:+.3f} pt")
print(f"  相対倍率 ((target/others - 1) * 100):  {abs_lift_ratio:+.2f} %")

# ────────────────────────────────────────────────────────────
# 3. もし「該当L2 vs 全体平均勝率」を絶対比較したら? (案A')
# ────────────────────────────────────────────────────────────
overall_path = ART / "overall_raw.json"
overall = json.loads(overall_path.read_text())
overall_cond = overall.get(TARGET_COND)
print("\n" + "="*70)
print("【参考: 該当 L2 vs 「全体勝率」 を絶対比較した場合】")
print("="*70)
if overall_cond is not None:
    diff_vs_overall = (mean_target_abs - overall_cond) * 100
    ratio_vs_overall = (mean_target_abs / overall_cond - 1.0) * 100
    print(f"\n  全体平均 {TARGET_COND}:                 {overall_cond:.4f}  ({overall_cond*100:.2f}%)")
    print(f"  L2={TARGET_L2} の {TARGET_COND} 平均:    {mean_target_abs:.4f}")
    print(f"  絶対勝率差 (L2_mean - overall):  {diff_vs_overall:+.3f} pt")
    print(f"  相対倍率 (L2_mean / overall - 1):    {ratio_vs_overall:+.2f} %")

# ────────────────────────────────────────────────────────────
# 4. まとめ
# ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("【まとめ - 3 つの '+pt' 系指標の比較】")
print("="*70)
print(f"\n  条件: {TARGET_COND}, L2={TARGET_L2}")
print(f"   ① 現UI dev_pct (= 自身平均比 lift の L2平均 - 1.0):  {dev_pct_now:+.2f} pt")
print(f"   ② 絶対勝率差 (L2 mean - others mean):                {abs_diff_pct:+.3f} pt")
if overall_cond is not None:
    print(f"   ③ 絶対勝率倍率 (L2 mean / overall - 1):              {ratio_vs_overall:+.2f} %")
print(f"\n  注意: 単位 (%pt) と (%) は別物!")
print(f"        ①は '個体差を消した条件適性' を抽出した相対指標")
print(f"        ②は L2 グループの '絶対的な強さ差' (種牡馬全体力含む)")
print(f"        ③は L2 が '全体平均と比べてどれくらい強いか' (絶対勝率比)")
