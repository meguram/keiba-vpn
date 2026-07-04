"""role_blend 統合の動作検証: タイトルホルダー他で profile が妥当になったか確認。

リポジトリルートで: python tests/research/manual/verify_role_blend.py
"""
from __future__ import annotations
import json
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
    "ジェンティルドンナ",
]


def _show(name: str):
    print(f"\n══════════════════════════════════════════")
    print(f" {name}")
    print(f"══════════════════════════════════════════")
    res = analyze_horse(name)
    if res.get("status") != "ok":
        print(f"  status={res.get('status')}, message={res.get('message')}")
        return
    print(f"  父: {res.get('sire_name')}  /  母父: {res.get('dam_sire')}")
    print(f"  L2: {res.get('L2')} ({(res.get('L2_meta') or {}).get('name','-')})")

    rb = res.get("role_blend")
    if not rb:
        print("  [role_blend なし]")
        return

    print(f"\n  ── 採用された role / 重み ──")
    for role, prof in rb["role_profiles"].items():
        print(f"   {role:>4}: {prof['stallion_name']:<22}"
              f"  n_records={prof['n_records']:>5}  "
              f"weight={prof['weight_effective']:.3f}  source={prof['source']}")

    diag = rb.get("diagnostics", {})
    if diag.get("missing_roles"):
        print(f"   未取得: {diag['missing_roles']}")
    if diag.get("fallback_roles"):
        print(f"   fallback: {diag['fallback_roles']}")

    print("\n  ── ブレンド プロファイル: 強み Top5 ──")
    for s in rb["blended_strengths"][:5]:
        print(f"   {s['label']:<24}  lift={s['lift']:.3f}  ({s['lift_pct']:+.1f}%)")
    print("  ── ブレンド プロファイル: 弱み Top5 ──")
    for s in rb["blended_weaknesses"][:5]:
        print(f"   {s['label']:<24}  lift={s['lift']:.3f}  ({s['lift_pct']:+.1f}%)")

    # 阪神 / 東京 / 長距離 の詳細を role 別に見る
    print(f"\n  ── 注目条件の role 別 lift ──")
    print(f"  {'condition':<14} {'F':>8} {'MF':>8} {'MMF':>8} {'FF':>8} {'blended':>9}")
    for cond in ["win_v_阪神", "win_v_東京", "win_v_中山", "win_d_長距離", "win_d_短距離", "win_heavy"]:
        row = f"  {cond:<14}"
        for role in ["F", "MF", "MMF", "FF"]:
            prof = rb["role_profiles"].get(role)
            if prof and cond in prof["lift"]:
                row += f"  {prof['lift'][cond]:>6.2f}"
            else:
                row += f"  {'-':>6}"
        bl = rb["blended_lift"].get(cond)
        row += f"  {bl:>6.2f}" if bl is not None else f"  {'-':>6}"
        print(row)


for name in TARGETS:
    _show(name)
