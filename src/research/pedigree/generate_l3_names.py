"""L3 クラスタプロファイルから自動命名候補を生成する。

各 (view, L2, L3) のプロファイルを読み、以下を出力:
    - short_name: 4-10 文字の短名 (例: 「ダート短距離パワー型」)
    - subtitle:   1 行説明 (例: 「ダート × 短距離 × 急坂で強い」)
    - color:      L2 親色をベースにバリエーション
    - icon:       L2 親アイコン
    - rationale:  選定根拠 (top lift conditions + top tags)
    - top_members: 代表種牡馬

命名ロジック:
    1. cond_lift_mean から「全体平均 +0.15 以上の lift」がある条件を上位 3 件抽出
    2. tag_freq から「30% 以上」が立つタグを上位 3 件抽出
    3. (条件 + タグ) のキーワード組合せから short_name を構成
    4. L2 親メタを l2_names.json から取得

Usage:
    python -m src.research.pedigree.generate_l3_names
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
L3DIR = ROOT / "data/page_reference/note_aptitude_race_l3"
ART_DIR = ROOT / "data/page_reference/note_aptitude_race"

# 条件名 → 短ラベル変換
COND_LABEL = {
    "win_v_東京": "東京", "win_v_中山": "中山", "win_v_阪神": "阪神", "win_v_京都": "京都",
    "win_v_中京": "中京", "win_v_新潟": "新潟", "win_v_小倉": "小倉", "win_v_福島": "福島",
    "win_v_札幌": "札幌", "win_v_函館": "函館",
    "win_s_芝": "芝", "win_s_ダート": "ダート",
    "win_d_短距離": "短距離", "win_d_マイル": "マイル", "win_d_中距離": "中距離", "win_d_長距離": "長距離",
    "win_pace_スロー瞬発力_短距離": "スロー瞬発短", "win_pace_スロー瞬発力_マイル": "スロー瞬発M",
    "win_pace_スロー瞬発力_中距離": "スロー瞬発中", "win_pace_スロー瞬発力_長距離": "スロー瞬発長",
    "win_pace_持続力勝負_短距離": "持続力短", "win_pace_持続力勝負_マイル": "持続力M",
    "win_pace_持続力勝負_中距離": "持続力中", "win_pace_持続力勝負_長距離": "持続力長",
    "win_pace_持久力勝負_短距離": "持久力短", "win_pace_持久力勝負_マイル": "持久力M",
    "win_pace_持久力勝負_中距離": "持久力中", "win_pace_持久力勝負_長距離": "持久力長",
    "win_steep": "急坂", "win_flat": "平坦", "win_heavy": "重馬場",
}

TAG_LABEL = {
    "dist_sprint": "短距離適性", "dist_mile": "マイル適性", "dist_mid": "中距離適性",
    "dist_stay": "ステイヤー",
    "course_flat": "平坦巧者", "course_steep": "急坂巧者",
    "course_small_turn": "小回り", "course_long_str": "長直線",
    "track_high_speed": "高速馬場", "track_heavy": "道悪",
    "pace_slow": "スロー瞬発", "pace_long": "持久力", "pace_sustain": "持続力",
    "runstyle_nige": "逃げ", "runstyle_senko": "先行", "runstyle_sashi": "差し",
    "runstyle_oikomi": "追込",
    "speed_top": "トップスピード", "speed_gear": "ギアチェンジ",
    "age_juvenile": "若駒巧者", "age_older": "古馬巧者",
    "draw_inner": "内枠巧者", "draw_outer": "外枠巧者",
    "rotate_repeat": "連闘", "rotate_long": "間隔広",
    "growth_late": "晩成", "growth_early": "早熟",
}

# L2 親色 のバリエーション (HSL を変えて L3 用)
L2_COLOR_VARIANT = {
    0: ["#a78bfa", "#8b5cf6", "#7c3aed", "#6d28d9", "#5b21b6", "#c4b5fd"],  # 紫系
    1: ["#b45309", "#92400e", "#d97706", "#f59e0b", "#fbbf24", "#ea580c"],  # 茶系
    2: ["#ef4444", "#dc2626", "#b91c1c", "#f87171", "#fca5a5", "#991b1b"],  # 赤系
    3: ["#10b981", "#059669", "#047857", "#34d399", "#6ee7b7", "#065f46"],  # 緑系
}


def _generate_short_name(top_conds: list[str], top_tags: list[str]) -> str:
    """上位特徴量から短名を生成。"""
    parts = []
    for c in top_conds[:2]:
        parts.append(c)
    for t in top_tags[:1]:
        parts.append(t)
    if not parts:
        return "汎用型"
    name = " ".join(parts)
    if len(name) > 12:
        name = name[:12]
    return name + "型"


def main():
    prof_path = L3DIR / "l3_profiles.json"
    meta_path = L3DIR / "l3_meta.json"
    if not prof_path.exists():
        raise FileNotFoundError(f"L3 profiles が無い: {prof_path}\n  -> build_l3_clusters を先に実行")

    prof = json.loads(prof_path.read_text(encoding="utf-8"))
    # L2 親メタ
    l2_names_path = ART_DIR / "l2_names.json"
    l2_meta = {}
    if l2_names_path.exists():
        l2_meta = json.loads(l2_names_path.read_text(encoding="utf-8"))

    # sire_id -> name
    uni = pd.read_parquet(ART_DIR / "unified.parquet")
    sid_to_name: dict[str, str] = {}
    for _, r in uni.iterrows():
        sid_to_name[str(r["entity_id"])] = str(r.get("entity_label") or r["entity_id"])

    l3_names: dict[str, dict] = {}
    for key, p in prof.items():
        # key 例: "father/L2=3/L3=1"
        try:
            view, l2_part, l3_part = key.split("/")
            L2 = int(l2_part.split("=")[1])
            L3 = int(l3_part.split("=")[1])
        except (ValueError, IndexError):
            continue

        cond_lift = p.get("cond_lift_mean", {})
        tag_freq = p.get("tag_freq", {})
        # 上位条件 (lift >= 1.15)
        top_conds = sorted(cond_lift.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
        top_cond_labels = []
        for c_key, v in top_conds:
            if v is None or v < 1.15:
                break
            short = c_key  # already short like "v_東京"
            full = f"win_{short}"
            label = COND_LABEL.get(full) or COND_LABEL.get(c_key) or short
            top_cond_labels.append(label)
            if len(top_cond_labels) >= 3:
                break

        # 上位タグ (>= 0.3)
        top_tag_labels = []
        for t_key, v in sorted(tag_freq.items(), key=lambda x: x[1] if x[1] else 0, reverse=True):
            if v is None or v < 0.3:
                break
            label = TAG_LABEL.get(t_key) or t_key
            top_tag_labels.append(label)
            if len(top_tag_labels) >= 3:
                break

        short_name = _generate_short_name(top_cond_labels, top_tag_labels)

        # subtitle
        sub_parts = []
        if top_cond_labels:
            sub_parts.append(" / ".join(top_cond_labels[:3]))
        if top_tag_labels:
            sub_parts.append(f"({', '.join(top_tag_labels[:2])})")
        subtitle = " ".join(sub_parts) or "—"

        # color
        palette = L2_COLOR_VARIANT.get(L2, ["#9ca3af"])
        color = palette[L3 % len(palette)]

        # icon: L2 親
        icon = l2_meta.get(str(L2), {}).get("icon", "●")

        # rationale
        rationale_parts = []
        for c_label, (c_key, v) in zip(top_cond_labels, top_conds):
            rationale_parts.append(f"{c_label} lift +{v-1.0:.2f}")
        for t_label, t_key in zip(top_tag_labels, sorted(tag_freq, key=lambda x: tag_freq[x] or 0, reverse=True)):
            rationale_parts.append(f"{t_label} freq {tag_freq[t_key]:.0%}")
        rationale = " / ".join(rationale_parts[:5])

        # top members (名前変換)
        top_member_ids = p.get("top_members", [])[:6]
        top_member_names = [sid_to_name.get(str(s), str(s))[:30] for s in top_member_ids]

        l3_names[key] = {
            "view": view,
            "L2": L2,
            "L3": L3,
            "name": short_name,
            "subtitle": subtitle,
            "color": color,
            "icon": icon,
            "highlights": top_cond_labels[:5],
            "tags": top_tag_labels[:5],
            "rationale": rationale,
            "n_members": p.get("n_members", 0),
            "silhouette": p.get("silhouette"),
            "top_members": top_member_names,
        }

    out_path = L3DIR / "l3_names.json"
    out_path.write_text(json.dumps(l3_names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[L3names] 書き出し: {out_path} ({len(l3_names)} L3 cluster)")

    # 視点・L2 別の集計サマリ
    summary: dict = {}
    for k, v in l3_names.items():
        view = v["view"]
        L2 = v["L2"]
        summary.setdefault(view, {}).setdefault(L2, []).append(
            f"  L3={v['L3']}: {v['name']:20} (n={v['n_members']:>3}) - {v['subtitle']}"
        )
    print("\n=== L3 階層構造 ===")
    for view in ["father", "dam_sire_line", "dam_dam_line"]:
        if view not in summary:
            continue
        print(f"\n[{view}]")
        for L2 in sorted(summary[view]):
            print(f" L2={L2}:")
            for line in summary[view][L2]:
                print(line)


if __name__ == "__main__":
    main()
