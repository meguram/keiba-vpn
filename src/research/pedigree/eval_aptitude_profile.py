"""血統適性プロファイルと実成績の整合性を統計的に検証する。

### 検証ロジック
各競走馬 (n_races >= 5) について:
  1. 適性プロファイル top-L2 を計算 (L2_scores 11 次元)
  2. L2 centroid (31D) の **lift 上位条件** を、その馬の予測得意条件とみなす
  3. その馬の **実成績ベスト条件** (オッズ<30 倍 + n>=2 で勝率最大の venue/distance/surface) と比較

### 整合性メトリクス
  - **Hit Rate**: L2 highlights のいずれかに 実成績 best がマッチした馬の割合
  - **Spearman**: 馬の L2_scores 上位 3 と、実成績ベスト条件カテゴリ別の勝率
                 ランキング相関 (per-horse Spearman の平均)
  - **Baseline**: ランダムに L2 を割り当てた場合の Hit Rate との比較

### 出力
標準出力にメトリクス + 比較表
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "data/page_reference/note_aptitude_race"
HORSE_AT = ROOT / "data/local/research/horse_aptitude"


def main(min_races: int = 5, sample_size: int = 3000, seed: int = 42) -> int:
    from src.research.pedigree.horse_aptitude_profile import (
        HorseAptitudeProfileCalc,
    )

    calc = HorseAptitudeProfileCalc(mode="label")
    l2_meta = calc.l2_info     # {L2: {name, highlights, weaknesses}}

    # 競走馬 (n_races >= min_races) を抽出
    idx = calc.idx
    cands = idx[idx["n_races"] >= min_races].copy()
    if sample_size and len(cands) > sample_size:
        cands = cands.sample(sample_size, random_state=seed)
    print(f"[eval] target horses: {len(cands)}", flush=True)

    # 各馬の実成績ベスト条件 (オッズ<30 倍, n>=2)
    perf = calc.perf
    perf_n2 = perf[perf["n_races"] >= 2].copy()

    venue_hits = surface_hits = dist_hits = track_hits = 0
    any_hits = 0
    total = 0
    venue_hit_baseline = 0

    # ベースライン: L2 highlights から「東京」「中山」「東京 中距離」...と均一に予測した場合
    all_highlights = []
    for L2, info in l2_meta.items():
        all_highlights.extend(info.get("highlights", []))
    label_freq = Counter(all_highlights)
    print(f"  L2 highlights 出現頻度 top10: {label_freq.most_common(10)}")

    L2_rank_records = []   # 各馬で top L2 ランクごとの実成績 best 命中

    horse_id_top_l2: dict[str, int] = {}
    horse_id_best: dict[str, dict] = {}

    for r in cands.itertuples(index=False):
        hid = str(r.horse_id)
        if hid not in calc.cats_by_horse:
            continue
        prof = calc.compute(horse_id=hid)
        if "error" in prof:
            continue
        l2_top = prof["L2_top"]
        if not l2_top:
            continue
        top_l2_id = int(l2_top[0]["L2"])
        horse_id_top_l2[hid] = top_l2_id

        # 実成績 best
        hp = perf_n2[perf_n2["horse_id"] == hid]
        if len(hp) == 0:
            continue
        best_venue = best_dist = best_surf = best_track = None
        for kind, var in [("venue", "best_venue"), ("distance_bin", "best_dist"),
                          ("surface", "best_surf"), ("track_condition", "best_track")]:
            sub = hp[hp["cond_kind"] == kind].sort_values("win_rate", ascending=False)
            if len(sub) > 0 and sub.iloc[0]["win_rate"] > 0:
                val = sub.iloc[0]["cond_value"]
                if kind == "venue": best_venue = val
                elif kind == "distance_bin": best_dist = val
                elif kind == "surface": best_surf = val
                else: best_track = val

        highlights_top = set()
        for tl in l2_top[:2]:    # top 1-2 まで
            L2 = int(tl["L2"])
            highlights_top.update(l2_meta.get(L2, {}).get("highlights", []))

        v_hit = best_venue in highlights_top
        s_hit = best_surf in highlights_top
        d_hit = best_dist in highlights_top
        t_hit = best_track in highlights_top

        venue_hits += int(v_hit)
        surface_hits += int(s_hit)
        dist_hits += int(d_hit)
        track_hits += int(t_hit)
        if v_hit or s_hit or d_hit or t_hit:
            any_hits += 1

        # ベースライン: ランダム L2 だった場合の venue 一致率を推定
        # → highlight 全体に出現する確率
        if best_venue:
            v_in_any = best_venue in all_highlights
            if v_in_any:
                # 全 L2 highlight 中の出現割合
                p = label_freq[best_venue] / max(1, sum(label_freq.values()))
                venue_hit_baseline += p

        horse_id_best[hid] = dict(
            v=best_venue, d=best_dist, s=best_surf, t=best_track,
            v_hit=v_hit, d_hit=d_hit, s_hit=s_hit, t_hit=t_hit,
        )

        total += 1

    if total == 0:
        print("  no valid horses")
        return 1

    def _pct(n): return f"{n/total*100:.1f}%"
    print()
    print(f"=== Hit Rate (top-2 L2 highlights に実成績 best が含まれる) ===")
    print(f"  競馬場 (venue)        : {venue_hits} / {total}  =  {_pct(venue_hits)}  "
          f"(baseline ≒ {venue_hit_baseline/total*100:.1f}%)")
    print(f"  距離区分 (dist_bin)   : {dist_hits} / {total}  =  {_pct(dist_hits)}")
    print(f"  路面 (surface)        : {surface_hits} / {total}  =  {_pct(surface_hits)}")
    print(f"  馬場 (track)         : {track_hits} / {total}  =  {_pct(track_hits)}")
    print(f"  Any  (いずれか命中)   : {any_hits} / {total}  =  {_pct(any_hits)}")

    # 命中分布: top L2 別
    print()
    print("=== Top L2 分布 (上位 5) ===")
    top_l2_count = Counter(horse_id_top_l2.values())
    for L2, n in top_l2_count.most_common(5):
        meta = l2_meta.get(L2, {})
        print(f"  L2={L2:>2} ({meta.get('name','?'):20}): {n} 馬 ({n/total*100:.1f}%)")

    # L2 別 venue 命中率
    print()
    print("=== L2 別 venue 命中率 ===")
    by_l2 = {}
    for hid, l2 in horse_id_top_l2.items():
        b = horse_id_best.get(hid)
        if b is None: continue
        by_l2.setdefault(l2, []).append(b["v_hit"])
    rows = []
    for L2 in sorted(by_l2):
        v = by_l2[L2]
        rows.append((L2, sum(v), len(v), sum(v)/max(1,len(v))*100))
    rows.sort(key=lambda r: -r[3])
    for L2, hits, n, rate in rows:
        meta = l2_meta.get(L2, {})
        hl = ",".join(meta.get("highlights",[])[:3])
        print(f"  L2={L2:>2} ({meta.get('name','?'):20}, hl={hl:25}): venue hit {hits}/{n} = {rate:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
