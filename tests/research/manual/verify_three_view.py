"""3 ビュー (父系/母系/統合) + 個体実績 vs 血統 prior の動作検証。

リポジトリルートで: python tests/research/manual/verify_three_view.py
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
    "オルフェーヴル",
]

def _show(name: str):
    print(f"\n══════════════════════════════════════════")
    print(f"  {name}")
    print(f"══════════════════════════════════════════")
    res = analyze_horse(name)
    if res.get("status") != "ok":
        print(f"  status={res.get('status')}, message={res.get('message')}")
        return
    print(f"  父: {res.get('sire_name')}  /  母父: {res.get('dam_sire')}")
    print(f"  L2: {res.get('L2')} ({(res.get('L2_meta') or {}).get('name','-')})")

    tv = res.get("three_view_profile")
    if not tv:
        print("  [three_view_profile なし]")
        return

    # 父系
    pat = tv.get("paternal")
    if pat:
        print(f"\n  ── 父系プロファイル ── ({pat.get('description','')})")
        print(f"    重み: {pat['weights_used']}, active: {pat['active_roles']}")
        print(f"    強み Top5:")
        for s in pat["strengths"][:5]:
            print(f"      {s['label']:<22}  lift={s['lift']:.3f} ({s['lift_pct']:+5.1f}%)")
    # 母系
    mat = tv.get("maternal")
    if mat:
        print(f"\n  ── 母系プロファイル ── ({mat.get('description','')})")
        print(f"    重み: {mat['weights_used']}, active: {mat['active_roles']}")
        print(f"    強み Top5:")
        for s in mat["strengths"][:5]:
            print(f"      {s['label']:<22}  lift={s['lift']:.3f} ({s['lift_pct']:+5.1f}%)")
    # 統合 (階層フォールバック)
    integ = tv["integrated"]
    print(f"\n  ── 統合プロファイル ──  source: {integ['source_layer']}")
    sm = integ.get("source_meta", {})
    print(f"    n_horses={sm.get('n_horses','?')}, n_records={sm.get('n_records','?')}")
    print(f"    強み Top5:")
    for s in integ["strengths"][:5]:
        print(f"      {s['label']:<22}  lift={s['lift']:.3f} ({s['lift_pct']:+5.1f}%)")
    print(f"    弱み Top5:")
    for s in integ["weaknesses"][:5]:
        print(f"      {s['label']:<22}  lift={s['lift']:.3f} ({s['lift_pct']:+5.1f}%)")

    # フォールバックチェーン
    print(f"\n  ── フォールバックチェーン ──")
    for fc in tv["fallback_chain"]:
        mark = "✓" if fc["available"] else "×"
        extra = ""
        if fc.get("bms_main"):
            extra += f" bms_main={fc['bms_main']}"
        if fc.get("sire_main"):
            extra += f" sire_main={fc['sire_main']}"
        print(f"    {mark} layer{fc['layer']} {fc['kind']:<12}  n_records={fc.get('n_records','?')}{extra}")

    # 個体実績 vs prior
    iv = res.get("individual_vs_prior")
    if iv:
        actual = iv["actual"]
        print(f"\n  ── 個体実績 vs 血統 prior ──")
        print(f"    本馬通算: {actual['n_total']} 戦 {actual['win_total']} 勝 ({actual['win_rate_total']:.1%})")
        print(f"    prior 採用 layer: {iv.get('prior_source_layer')}")
        if iv["top_overperform"]:
            print(f"    ▲ prior を大きく上回った条件:")
            for d in iv["top_overperform"][:5]:
                print(f"      {d['label']:<22}  個体 lift={d['actual_lift']:.2f} vs prior {d['prior_lift']:.2f}  "
                      f"(Δ={d['deviation_lift']:+.2f}, n={d['n_actual']})")
        if iv["top_underperform"]:
            print(f"    ▼ prior を下回った条件:")
            for d in iv["top_underperform"][:5]:
                print(f"      {d['label']:<22}  個体 lift={d['actual_lift']:.2f} vs prior {d['prior_lift']:.2f}  "
                      f"(Δ={d['deviation_lift']:+.2f}, n={d['n_actual']})")

for n in TARGETS:
    _show(n)
