"""血統 prior + 個体実績 + ラベル付き戦績テーブルの検証。

リポジトリルートで: python tests/research/manual/verify_record_labels.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.api.bloodline_meta_cluster import analyze_horse  # noqa: E402

TARGETS = [
    "タイトルホルダー",
    "イクイノックス",
    "ドウデュース",
    "リバティアイランド",
    "コントレイル",
    "アーモンドアイ",
    "ジャスティンパレス",
    "スターズオンアース",
]


def _show(name: str):
    print(f"\n══════════════════════════════════════════════")
    print(f"  {name}")
    print(f"══════════════════════════════════════════════")
    res = analyze_horse(name)
    if res.get("status") != "ok":
        print(f"  status={res.get('status')}, message={res.get('message')}")
        return

    rec = res.get("record_with_labels")
    if not rec:
        print("  [record_with_labels なし]")
        return

    summary = rec["summary"]
    print(f"  通算: {summary['n_total']} 戦 {summary['win_total']} 勝 "
          f"({summary['win_rate_total']:.1%}, 95%CI {summary['win_rate_ci_low']:.1%}-{summary['win_rate_ci_high']:.1%})")
    print(f"  ▶ {rec['summary_text']}")

    print(f"\n  ── 戦績 (ラベル付き) ──")
    for grp_name, grp_label in [("venue","場所"),("surface","路面"),("distance","距離"),
                                 ("topography","地形"),("track_condition","馬場")]:
        rows = rec["groups"].get(grp_name, [])
        rows_with_data = [r for r in rows if r["n"] > 0]
        if not rows_with_data: continue
        print(f"\n  【{grp_label}】")
        print(f"    {'条件':<12}{'n':>4}{'勝':>3}{'勝率':>7}{'95%CI':>15}{'prior':>8}{'個体lift':>9}{'p値':>8}  ラベル")
        for r in sorted(rows_with_data, key=lambda x: -x["n"]):
            ci = f"{r['win_rate_ci_low']:.1%}-{r['win_rate_ci_high']:.1%}"
            indl = r["individual_lift"]
            indl_s = f"{indl:.2f}" if indl is not None else "-"
            pv_s = f"{r['p_value']:.3f}" if r["p_value"] < 1.0 else "  -  "
            print(f"    {r['label_jp']:<12}{r['n']:>4}{r['win']:>3}{r['win_rate']:>7.1%}{ci:>15}"
                  f"{r['prior_lift']:>8.2f}{indl_s:>9}{pv_s:>8}  {r['pattern_icon']} {r['pattern_short']}")

    if rec["notable_personal"]:
        print(f"\n  ⚡ 個体特異・突出パターン (Top 3):")
        for c in rec["notable_personal"][:3]:
            print(f"    {c['label_jp']:<14}  {c['n']}戦{c['win']}勝 ({c['win_rate']:.1%})  "
                  f"prior={c['prior_lift']:.2f}, 個体={c['individual_lift']:.2f}, p={c['p_value']:.4f}")
    if rec["notable_personal_low"]:
        print(f"\n  ⚠ 個体特異・低調パターン (Top 3):")
        for c in rec["notable_personal_low"][:3]:
            print(f"    {c['label_jp']:<14}  {c['n']}戦{c['win']}勝 ({c['win_rate']:.1%})  "
                  f"prior={c['prior_lift']:.2f}, 個体={c['individual_lift']:.2f}, p={c['p_value']:.4f}")
    if rec["notable_pedigree"]:
        print(f"\n  ★ 血統適性発揮パターン (Top 3):")
        for c in rec["notable_pedigree"][:3]:
            print(f"    {c['label_jp']:<14}  {c['n']}戦{c['win']}勝 ({c['win_rate']:.1%})  "
                  f"prior={c['prior_lift']:.2f}, 個体={c['individual_lift']:.2f}")


for n in TARGETS:
    _show(n)
