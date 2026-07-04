"""L2_fine クラスタプロファイルから自動命名 (l2_names.json) を生成する。

各 L2_fine クラスタについて、相対 lift トップ条件と代表種牡馬から:
    - name        : 短名 (4-12 文字, 例「東京芝中距離主流型」)
    - subtitle    : 1 行説明 (例「東京 / 芝 / 中距離 / スロー瞬発で強い」)
    - color       : super-group 親色 + クラスタ ID で微調整
    - icon        : super-group 親アイコン
    - highlights  : 得意条件タグ (lift +0.10 以上)
    - weaknesses  : 不得意条件 (lift -0.10 以下)
    - rationale   : 命名根拠の自然文
    - top_members : 代表種牡馬名

入力:
    data/research/bloodline_meta_cluster/l2_profiles.json
    data/research/bloodline_meta_cluster/l2_super_groups.json
    data/research/bloodline_meta_cluster/unified.parquet

出力:
    data/research/bloodline_meta_cluster/l2_names.json (上書き)

Usage:
    python -m src.research.pedigree.generate_l2_fine_names
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "data/page_reference/note_aptitude_race"

COND_LABEL = {
    "win_v_東京": "東京", "win_v_中山": "中山", "win_v_阪神": "阪神", "win_v_京都": "京都",
    "win_v_中京": "中京", "win_v_新潟": "新潟", "win_v_小倉": "小倉", "win_v_福島": "福島",
    "win_v_札幌": "札幌", "win_v_函館": "函館",
    "win_s_芝": "芝", "win_s_ダート": "ダート",
    "win_d_短距離": "短距離", "win_d_マイル": "マイル", "win_d_中距離": "中距離", "win_d_長距離": "長距離",
    "win_pace_スロー瞬発力_短距離": "スロー瞬発短", "win_pace_スロー瞬発力_マイル": "スロー瞬発マイル",
    "win_pace_スロー瞬発力_中距離": "スロー瞬発中", "win_pace_スロー瞬発力_長距離": "スロー瞬発長",
    "win_pace_持続力勝負_短距離": "持続力短", "win_pace_持続力勝負_マイル": "持続力マイル",
    "win_pace_持続力勝負_中距離": "持続力中", "win_pace_持続力勝負_長距離": "持続力長",
    "win_pace_持久力勝負_短距離": "持久力短", "win_pace_持久力勝負_マイル": "持久力マイル",
    "win_pace_持久力勝負_中距離": "持久力中", "win_pace_持久力勝負_長距離": "持久力長",
    "win_steep": "急坂", "win_flat": "平坦", "win_heavy": "道悪",
    # 基礎スピード次元 (track_speed_races 由来)
    "win_fast": "高速馬場", "win_slow": "低速馬場",
    # コース形状次元
    "win_outer": "外回り", "win_inner": "内回り",
}

# 条件カテゴリ (重複削減用)
COND_CATEGORY = {
    **{c: "venue" for c in ["win_v_東京", "win_v_中山", "win_v_阪神", "win_v_京都",
                              "win_v_中京", "win_v_新潟", "win_v_小倉", "win_v_福島",
                              "win_v_札幌", "win_v_函館"]},
    "win_s_芝": "surface", "win_s_ダート": "surface",
    **{c: "distance" for c in ["win_d_短距離", "win_d_マイル", "win_d_中距離", "win_d_長距離"]},
    **{c: "pace" for c in COND_LABEL if c.startswith("win_pace_")},
    "win_steep": "course", "win_flat": "course", "win_heavy": "track",
    "win_fast": "track_speed", "win_slow": "track_speed",
    "win_outer": "course_shape", "win_inner": "course_shape",
}

# super-group → ベース色 (HSL 5 色, 視覚的に明確に区別)
SUPER_COLORS = [
    "#3b82f6",  # 青  (主流芝)
    "#a78bfa",  # 紫  (スタミナ / 道悪)
    "#b45309",  # 茶  (ダート系)
    "#10b981",  # 緑  (中距離スロー瞬発)
    "#ef4444",  # 赤  (急坂 / 個性派)
    "#f59e0b",  # 橙  (オールラウンダー)
    "#ec4899",  # 桃  (その他)
]
SUPER_ICONS = ["◆", "▲", "■", "●", "◇", "△", "□", "○"]


def _format_label(c: str) -> str:
    return COND_LABEL.get(c, c.replace("win_", ""))


# カテゴリ命名優先順位: 低い = より優先
# venue を surface より優先 → 特定コース種牡馬の命名がより具体的になる
_CAT_PRIORITY = {
    "distance": 0,
    "venue": 1,       # venue: 特定コースへの偏りは最重要
    "surface": 2,
    "pace": 3,
    "track": 4,
    "track_speed": 5, # 高速/低速馬場: speed index の指標として有用
    "course": 6,
    "course_shape": 8,  # 外回り/内回りは補助的に
    "other": 7,
}


def _pick_top(profile: dict[str, float], threshold: float, top_n: int) -> list[tuple[str, float]]:
    items = [(c, v) for c, v in profile.items() if v >= threshold]
    # カテゴリ優先度を考慮してソート: まず優先度 asc, 次に lift desc
    items.sort(key=lambda x: (_CAT_PRIORITY.get(COND_CATEGORY.get(x[0], "other"), 7), -x[1]))
    return items[:top_n]


def _pick_bottom(profile: dict[str, float], threshold: float, top_n: int) -> list[tuple[str, float]]:
    items = [(c, v) for c, v in profile.items() if v <= threshold]
    items.sort(key=lambda x: x[1])
    return items[:top_n]


def _dedupe_by_category(items: list[tuple[str, float]], max_per_cat: int = 2) -> list[tuple[str, float]]:
    """同じカテゴリの条件が並ぶのを抑える。course_shape/track_speed は 1 個まで。"""
    cat_max = {"course_shape": 1, "track_speed": 1}
    cat_count: dict[str, int] = {}
    out = []
    for c, v in items:
        cat = COND_CATEGORY.get(c, "other")
        limit = cat_max.get(cat, max_per_cat)
        if cat_count.get(cat, 0) >= limit:
            continue
        out.append((c, v))
        cat_count[cat] = cat_count.get(cat, 0) + 1
    return out


def _generate_name(top_conds: list[tuple[str, float]], top_members: list[str]) -> tuple[str, str]:
    """名前 + サブタイトル を生成。名前は各カテゴリから最大1件で2語以内。"""
    # 名前用: カテゴリ内重複は1件まで
    deduped = _dedupe_by_category(top_conds, max_per_cat=1)
    labels = [_format_label(c) for c, _ in deduped[:3]]
    if not labels:
        return "汎用型", "強み突出なし"

    # 主要ラベル (上位 2 個) で短名
    short = labels[0] + (labels[1] if len(labels) > 1 else "") + "型"
    if len(short) > 14:
        short = labels[0] + "型"

    subtitle_parts = labels[:3]
    if top_members:
        subtitle_parts.append(f"({top_members[0]} 型)")
    subtitle = " / ".join(subtitle_parts[:3])
    if top_members and len(top_members) > 0:
        subtitle += f" — {top_members[0]} 系"

    return short, subtitle


def main() -> int:
    profiles = json.loads((ART / "l2_profiles.json").read_text(encoding="utf-8"))
    super_groups = json.loads((ART / "l2_super_groups.json").read_text(encoding="utf-8"))
    uni = pd.read_parquet(ART / "unified.parquet")
    stallions = uni[uni["entity_type"] == "stallion"].copy()

    l2_to_super = super_groups.get("L2_to_super", {})

    names: dict[str, dict] = {}
    for k, prof in profiles.items():
        L2 = int(k)
        sg = int(l2_to_super.get(str(L2), 0))

        # 代表種牡馬 (n_horses 上位)
        sub = stallions[stallions["L2"] == L2].sort_values("n_horses", ascending=False)
        top_members = [str(r["entity_label"])[:24] for _, r in sub.head(5).iterrows()]

        # 命名用: 強シグナル (lift>0.10) を強度順に並べる (カテゴリ優先なし)
        # → 本当に際立った条件のみが名前に反映される
        top_for_name = sorted(
            [(c, v) for c, v in prof.items() if v >= 0.10],
            key=lambda x: -x[1],
        )
        if not top_for_name:
            # fallback: 0.03 以上から最大3件
            top_for_name = sorted(
                [(c, v) for c, v in prof.items() if v >= 0.03],
                key=lambda x: -x[1],
            )[:3]

        # 得意 / 不得意 highlights (閾値 0.03, カテゴリ優先度ソート)
        top_pos = _pick_top(prof, threshold=0.03, top_n=8)
        top_neg = _pick_bottom(prof, threshold=-0.03, top_n=5)

        name, subtitle = _generate_name(top_for_name, top_members)

        # ハイライト/不得意のラベル化 (max_per_cat 緩和 + 取得数増)
        highlights = [_format_label(c) for c, _ in _dedupe_by_category(top_pos, max_per_cat=3)[:7]]
        weaknesses = [_format_label(c) for c, _ in _dedupe_by_category(top_neg, max_per_cat=3)[:5]]
        # 「道悪」がある場合は実成績マッチング用に「重」「不良」「稍重」も付加
        if "道悪" in highlights:
            highlights.extend(["重", "不良", "稍重"])
        # 「平坦」は実成績では venue マッチングなのでそのまま
        # 「急坂」も同様 (venue「中山/阪神/中京」が含まれていれば実成績ヒット)
        # 距離区分の対応: 「中長距離」も追加 (中距離 → 中距離 + 中長距離)
        if "中距離" in highlights and "長距離" in highlights and "中長距離" not in highlights:
            highlights.append("中長距離")

        # 命名根拠
        if top_pos:
            top_labels = [f"{_format_label(c)}(+{v:.2f})" for c, v in top_pos[:3]]
            rationale = "得意条件 lift: " + ", ".join(top_labels)
        else:
            rationale = "突出した得意条件なし (全体平均的)"

        names[str(L2)] = {
            "L2": L2,
            "super_group": sg,
            "name": name,
            "subtitle": subtitle,
            "color": SUPER_COLORS[sg % len(SUPER_COLORS)],
            "icon": SUPER_ICONS[sg % len(SUPER_ICONS)],
            "n_members": int(len(sub)),
            "top_members": top_members,
            "highlights": highlights,
            "weaknesses": weaknesses,
            "rationale": rationale,
        }

    # super-group のサマリ
    super_summary: dict[str, dict] = {}
    for sg_id, sg_data in super_groups.get("super_groups", {}).items():
        sg = int(sg_id)
        members = sg_data.get("member_L2s", [])
        sg_l2_names = [names[str(m)]["name"] for m in members if str(m) in names]
        sg_top_horses = []
        for m in members:
            sub_m = stallions[stallions["L2"] == m].sort_values("n_horses", ascending=False)
            sg_top_horses.extend([str(r["entity_label"])[:24] for _, r in sub_m.head(2).iterrows()])
        # 共通する得意条件 (member 平均 lift トップ)
        avg_prof: dict[str, float] = {}
        for m in members:
            for c, v in profiles[str(m)].items():
                avg_prof[c] = avg_prof.get(c, 0) + v / max(1, len(members))
        top_avg = sorted(avg_prof.items(), key=lambda x: -x[1])[:5]
        super_summary[sg_id] = {
            "super_id": sg,
            "name": f"スーパー{sg}",
            "member_L2_names": sg_l2_names,
            "member_L2_ids": members,
            "n_L2": int(sg_data.get("n_L2", len(members))),
            "n_stallions": int(sg_data.get("n_stallions", 0)),
            "color": SUPER_COLORS[sg % len(SUPER_COLORS)],
            "icon": SUPER_ICONS[sg % len(SUPER_ICONS)],
            "top_stallions": sg_top_horses[:6],
            "top_conditions": [_format_label(c) for c, _ in top_avg],
        }

    # super-group に解釈的な名前を付ける
    # 戦略:
    #   1. member L2 数が 1 個なら、その L2 の name 由来
    #   2. 2 個以上なら、member 全員の highlights を集計した「共通テーマ」と
    #      頻出 top_conditions の組み合わせで命名
    #   3. 「○○系」だけでは抽象的なので、 「コース傾向 + 距離・脚質傾向」の 2 軸で
    cat_to_label = {
        "venue": "コース",
        "surface": "路面",
        "distance": "距離",
        "pace": "脚質",
        "track": "馬場",
    }
    for sg_id, info in super_summary.items():
        members = info["member_L2_ids"]
        if len(members) == 1:
            l2 = names[str(members[0])]
            info["name"] = l2["name"]
            info["subtitle"] = l2["subtitle"]
            continue
        # 共通する highlights を抽出 (member L2 のうち過半数で出現する条件)
        from collections import Counter
        h_count = Counter()
        for m in members:
            for h in names[str(m)].get("highlights", []):
                h_count[h] += 1
        common = [h for h, c in h_count.most_common() if c >= max(2, len(members) // 2 + 1)]
        # 命名: top member の name から代表テーマを 2 つ取る
        top_conds = info.get("top_conditions", [])
        if len(common) >= 2:
            theme = f"{common[0]}{common[1]}"
        elif len(top_conds) >= 2:
            theme = f"{top_conds[0]}{top_conds[1]}"
        elif top_conds:
            theme = top_conds[0]
        else:
            theme = f"SG{sg_id}"
        info["name"] = f"{theme}系"
        # subtitle: member L2 名を列挙して透明性確保
        member_names_short = [names[str(m)]["name"] for m in members if str(m) in names]
        info["subtitle"] = " · ".join(member_names_short)
        # 共通条件もメタとして保持
        info["common_highlights"] = common[:5]

    out = {
        "l2_names": names,
        "super_groups": super_summary,
        "version": "fine_2026_05",
    }
    (ART / "l2_names.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"saved: {ART / 'l2_names.json'}")
    print(f"  generated names for {len(names)} L2_fine clusters")
    print(f"  super-group summaries: {len(super_summary)}")
    print()
    print("=== L2_fine names ===")
    for k in sorted(names.keys(), key=int):
        v = names[k]
        print(f"  L2={k:>2} [SG={v['super_group']}] {v['icon']} {v['name']:14} | {v['subtitle']}")
    print()
    print("=== super-groups ===")
    for k in sorted(super_summary.keys(), key=int):
        v = super_summary[k]
        print(f"  SG={k} {v['icon']} {v['name']:14} (L2 {len(v['member_L2_ids'])}個, {v['n_stallions']}頭)")
        print(f"      L2 members: {v['member_L2_names']}")
        print(f"      top conds : {v['top_conditions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
