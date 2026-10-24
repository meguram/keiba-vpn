"""ラベル分布と妥当性のバックテスト。

n>=15 の馬を対象に、record_with_labels の各 cell ラベル分布と、
ラベル別の actual lift 平均を集計して、ラベルが意味のある分類になっているか確認。

期待される結果:
  - PERSONAL_BREAKTHROUGH cells の actual lift > prior lift (lift 差が大)
  - PERSONAL_UNDERPERFORM cells の actual lift < prior lift
  - ラベル別の lift 差は明確に分離されているはず
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.api.bloodline_meta_cluster import (  # noqa: E402
    _load_artifacts, compute_record_with_labels, find_horse_id, LABEL_DEFS,
)

ROOT = _ROOT
ANALYSIS_DIR = ROOT / "data/local/analysis/pedigree"
IDX = ROOT / "data/page_reference/pedigree_race_index"


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] load")
    art = _load_artifacts()
    if art is None:
        print("artifact load failed"); return

    race = pd.read_parquet(IDX / "race_result_slim.parquet", columns=["horse_id","finish_position","horse_name"])
    race["horse_id"] = race["horse_id"].astype(str)
    race = race[race["finish_position"].notna() & (race["finish_position"] > 0)]
    g = race.groupby("horse_id").size()
    targets = sorted(g[(g >= 15) & (g <= 60)].index, key=lambda h: -g[h])
    print(f"[{time.time()-t0:.1f}s] targets (n_runs in [15,60]) = {len(targets):,}")

    # サンプルとして最初の 1000 馬
    sample = targets[:1500]
    print(f"[{time.time()-t0:.1f}s] sample size = {len(sample)}")

    rows = []
    horse_summary = []
    n_done = 0
    for hid in sample:
        n_done += 1
        if n_done % 200 == 0:
            print(f"  ... {n_done}/{len(sample)}", flush=True)
        try:
            rec = compute_record_with_labels(hid, art=art)
        except Exception as e:
            continue
        if rec is None: continue
        # 馬サマリー
        n_personal = len(rec["notable_personal"])
        n_personal_low = len(rec["notable_personal_low"])
        n_pedigree = len(rec["notable_pedigree"])
        horse_summary.append({
            "horse_id": hid, "n_personal": n_personal,
            "n_personal_low": n_personal_low, "n_pedigree": n_pedigree,
        })
        # 全 cell
        for grp_name, cells in rec["groups"].items():
            for c in cells:
                rows.append({
                    "horse_id": hid, "grp": grp_name, **c,
                })
    df = pd.DataFrame(rows)
    hs = pd.DataFrame(horse_summary)
    print(f"[{time.time()-t0:.1f}s] cells = {len(df):,}, horses processed = {len(hs):,}")

    # ラベル分布
    print(f"\n========= ラベル分布 (全 cell) =========")
    pat_counts = df["pattern"].value_counts()
    for p, c in pat_counts.items():
        defs = LABEL_DEFS.get(p, {})
        print(f"  {defs.get('icon','?'):>2} {p:<25} {defs.get('ja_short','?'):<14} n={c:>5,} ({c/len(df):.1%})")

    # ラベル分布 (n>=5 のみ)
    print(f"\n========= ラベル分布 (n>=5, 暫定除外) =========")
    df_solid = df[df["n"] >= 5]
    pat_solid = df_solid["pattern"].value_counts()
    for p, c in pat_solid.items():
        defs = LABEL_DEFS.get(p, {})
        print(f"  {defs.get('icon','?'):>2} {p:<25} {defs.get('ja_short','?'):<14} n={c:>5,} ({c/len(df_solid):.1%})")

    # ラベル別の lift 差 (妥当性チェック)
    print(f"\n========= ラベル別 actual_lift / prior_lift 平均 (n>=5) =========")
    print(f"  {'pattern':<25} {'n':>5}  {'individual_lift_mean':>20}  {'prior_lift_mean':>17}  {'差':>6}")
    for p in df_solid["pattern"].unique():
        sub = df_solid[df_solid["pattern"] == p]
        sub2 = sub[sub["individual_lift"].notna()]
        if len(sub2) < 5: continue
        ail = sub2["individual_lift"].mean()
        pl = sub2["prior_lift"].mean()
        print(f"  {p:<25} {len(sub2):>5,}  {ail:>20.3f}  {pl:>17.3f}  {ail-pl:>+6.3f}")

    # 顕著な PERSONAL_BREAKTHROUGH 例
    print(f"\n========= 顕著な PERSONAL_BREAKTHROUGH (p<0.05, n>=5) =========")
    bt = df[(df["pattern"]=="PERSONAL_BREAKTHROUGH") & (df["p_value"]<0.05) & (df["n"]>=5)]
    bt = bt.sort_values("p_value").head(20)
    horse_name_map = race.dropna(subset=["horse_name"]).drop_duplicates("horse_id").set_index("horse_id")["horse_name"].to_dict()
    print(f"  {'馬名':<20}{'条件':<14}{'n':>4}{'勝':>4}{'勝率':>7}{'prior':>7}{'個体lift':>9}{'p値':>8}")
    for _, r in bt.iterrows():
        nm = horse_name_map.get(r["horse_id"], r["horse_id"])[:18]
        print(f"  {nm:<20}{r['label_jp']:<14}{r['n']:>4}{r['win']:>4}{r['win_rate']:>7.1%}"
              f"{r['prior_lift']:>7.2f}{r['individual_lift']:>9.2f}{r['p_value']:>8.4f}")

    print(f"\n========= 顕著な PERSONAL_UNDERPERFORM (p<0.05, n>=5) =========")
    up = df[(df["pattern"]=="PERSONAL_UNDERPERFORM") & (df["p_value"]<0.05) & (df["n"]>=5)]
    up = up.sort_values("p_value").head(20)
    print(f"  {'馬名':<20}{'条件':<14}{'n':>4}{'勝':>4}{'勝率':>7}{'prior':>7}{'個体lift':>9}{'p値':>8}")
    for _, r in up.iterrows():
        nm = horse_name_map.get(r["horse_id"], r["horse_id"])[:18]
        print(f"  {nm:<20}{r['label_jp']:<14}{r['n']:>4}{r['win']:>4}{r['win_rate']:>7.1%}"
              f"{r['prior_lift']:>7.2f}{r['individual_lift']:>9.2f}{r['p_value']:>8.4f}")

    # 馬ごとの集計 (個体特異が多い馬 = "血統からズレた個性派")
    print(f"\n========= 個体特異シグナルが多い馬 Top 15 =========")
    hs_sub = hs[hs["n_personal"] + hs["n_personal_low"] >= 1].copy()
    hs_sub["n_personal_total"] = hs_sub["n_personal"] + hs_sub["n_personal_low"]
    hs_sub = hs_sub.sort_values("n_personal_total", ascending=False).head(15)
    print(f"  {'馬名':<20}{'⚡突出':>7}{'⚠低調':>7}")
    for _, r in hs_sub.iterrows():
        nm = horse_name_map.get(r["horse_id"], r["horse_id"])[:18]
        print(f"  {nm:<20}{r['n_personal']:>7}{r['n_personal_low']:>7}")

    # ラベル × 出走数別 分布
    print(f"\n========= サンプルサイズ別 ラベル分布 =========")
    df["n_bin"] = pd.cut(df["n"], bins=[0,3,5,10,20,100], labels=["1-3","4-5","6-10","11-20","21+"])
    pivot = df.pivot_table(index="n_bin", columns="pattern", values="cond", aggfunc="count", fill_value=0, observed=True)
    print(pivot)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / "backtest_labels.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[{time.time()-t0:.1f}s] saved {out}")


if __name__ == "__main__":
    main()
