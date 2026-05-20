"""deep_role_evidence.parquet を分析して指標を表示する。

使用:
  python -m src.research.pedigree.deep_role_evidence   # 先に生成
  python -m src.research.pedigree.deep_role_analyze
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = ROOT / "data/local/analysis/pedigree"
DIST_LABELS = ["短距離", "マイル", "中距離", "長距離"]
AXIS_CONDS: dict[str, list[str]] = {
    "venue":    [f"win_v_{v}" for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]],
    "surface":  [f"win_s_{s}" for s in ["芝","ダート"]],
    "distance": [f"win_d_{d}" for d in DIST_LABELS],
    "topo":     ["win_steep","win_flat"],
    "track":    ["win_heavy"],
}
COND_COLS: list[str] = sum(AXIS_CONDS.values(), [])

df = pd.read_parquet(ANALYSIS_DIR / "deep_role_evidence.parquet")
print(f"=== 集計済み role × stallion: {len(df):,} 行 / {df['stallion_id'].nunique():,} 種牡馬 ===")
print(f"  role 別行数: {dict(df['role'].value_counts().sort_index())}")
print()

# ───────── 指標 1: 軸別 lift std ─────────
print("========= [指標 1] role 別 × 軸別の lift 分散 (種牡馬間中央値) =========")
print(f"  解釈: std 大 → その軸でメリハリのある適性が出やすい")
print(f"  {'role':<10} {'venue':>9} {'distance':>9} {'surface':>9} {'topo':>9} {'track*':>9}  {'overall':>9}")
print("  (* track は 1 列なので別馬間 std を表示)")
for role in ["F","MF","PatDeep","MatAny","MatDeep","MatG3","MatG4","MatG5+"]:
    sub = df[df["role"]==role]
    if sub.empty: continue
    line = f"  {role:<10}"
    sums = []
    for axis, conds in AXIS_CONDS.items():
        cols = [f"lift_{c}" for c in conds if f"lift_{c}" in sub.columns]
        if not cols:
            line += f" {'-':>9}"; continue
        if len(cols) == 1:
            # 軸内 std は計算不可。代わりに種牡馬間 std (=その軸 lift の馬間分散)
            v = sub[cols[0]].std()
            line += f"  {v:>8.3f}"
            sums.append(v)
        else:
            stds = sub[cols].std(axis=1)
            v = float(np.nanmedian(stds))
            line += f"  {v:>8.3f}"
            sums.append(v)
    mean_v = float(np.nanmean(sums)) if sums else float('nan')
    line += f"  {mean_v:>8.3f}"
    print(line)

# ───────── 指標 2: 最大変動軸の分布 ─────────
print("\n========= [指標 2] 各種牡馬で最も lift std が大きい軸の分布 =========")
print("  → どの軸 (venue/dist/surface/topo) で最も差がつきやすいか")
print(f"  {'role':<10} " + " ".join(f"{ax:>9}" for ax in ("venue","distance","surface","topo")))
for role in ["F","MF","PatDeep","MatAny","MatDeep"]:
    sub = df[df["role"]==role]
    if sub.empty: continue
    votes = {ax: 0 for ax in ("venue","distance","surface","topo")}
    for _, r in sub.iterrows():
        best_ax, best_s = None, -1
        for ax in votes:
            cols = [f"lift_{c}" for c in AXIS_CONDS[ax] if f"lift_{c}" in r.index]
            if len(cols) < 2: continue
            vals = r[cols].dropna()
            if len(vals) < 2: continue
            s = float(vals.std())
            if s > best_s:
                best_s, best_ax = s, ax
        if best_ax: votes[best_ax] += 1
    tot = sum(votes.values())
    line = f"  {role:<10}"
    for ax in ("venue","distance","surface","topo"):
        v = votes.get(ax, 0)
        pct = (v/tot*100) if tot else 0
        line += f"  {pct:>7.1f}%"
    print(line + f"  (n={tot})")

# ───────── 指標 3: F vs MatDeep 相関 ─────────
print("\n========= [指標 3] 同一種牡馬 F vs MatDeep の条件別 lift 相関 (Pearson) =========")
print("  解釈: 相関 ~0 → F と MatDeep が独立シグナル")
pivot_F  = df[df["role"]=="F"].set_index("stallion_id")
for other_role in ["MF","MatAny","MatDeep","MatG3","MatG4","MatG5+","PatDeep"]:
    pivot_O = df[df["role"]==other_role].set_index("stallion_id")
    common = list(set(pivot_F.index) & set(pivot_O.index))
    if not common: continue
    corrs = []
    for sid in common:
        f_v = pd.Series({c: pivot_F.loc[sid, f"lift_{c}"] for c in COND_COLS if f"lift_{c}" in pivot_F.columns}).dropna()
        o_v = pd.Series({c: pivot_O.loc[sid, f"lift_{c}"] for c in COND_COLS if f"lift_{c}" in pivot_O.columns}).dropna()
        cc = list(set(f_v.index) & set(o_v.index))
        if len(cc) < 5: continue
        corrs.append(float(np.corrcoef(f_v[cc], o_v[cc])[0,1]))
    if corrs:
        med = np.median(corrs); mean = np.mean(corrs)
        ind = sum(1 for r in corrs if abs(r)<0.3)/len(corrs)*100
        pos = sum(1 for r in corrs if r>0.5)/len(corrs)*100
        neg = sum(1 for r in corrs if r<-0.3)/len(corrs)*100
        print(f"  F vs {other_role:<10}  n={len(corrs):>4}  median r={med:>+5.2f}  mean r={mean:>+5.2f}  "
              f"|r|<0.3: {ind:>5.1f}%  r>0.5: {pos:>5.1f}%  r<-0.3: {neg:>5.1f}%")

# ───────── 指標 4: 大括り vs 細粒度 振幅 ─────────
print("\n========= [指標 4] 大括り (venue+dist+surface) vs 細粒度 (topo+track) 振幅 =========")
print("  種牡馬ごとに 軸内 max-min を測り、種牡馬間中央値")
print(f"  {'role':<10} {'BROAD振幅':>11} {'FINE振幅':>11} {'BROAD/FINE':>12}")
for role in ["F","MF","PatDeep","MatAny","MatDeep","MatG3","MatG4","MatG5+"]:
    sub = df[df["role"]==role]
    if sub.empty: continue
    bcols = sum([[f"lift_{c}" for c in AXIS_CONDS[a]] for a in ("venue","distance","surface")],[])
    fcols = sum([[f"lift_{c}" for c in AXIS_CONDS[a]] for a in ("topo","track")],[])
    bcols = [c for c in bcols if c in sub.columns]
    fcols = [c for c in fcols if c in sub.columns]
    b_amp = (sub[bcols].max(axis=1) - sub[bcols].min(axis=1)).median()
    f_amp = (sub[fcols].max(axis=1) - sub[fcols].min(axis=1)).median() if fcols else float('nan')
    ratio = float(b_amp/f_amp) if f_amp and f_amp>0 else float('nan')
    print(f"  {role:<10}  {b_amp:>10.3f}  {f_amp:>10.3f}  {ratio:>11.3f}")

# ───────── 指標 5: 母系全域 vs 父 で「lift が体系的に上回る」条件 ─────────
print("\n========= [指標 5] MaternalAny vs F: 条件別 lift 差 (中央値) =========")
print("  各種牡馬で lift_MA - lift_F を計算し、条件別に集計")
pivot_MA = df[df["role"]=="MatAny"].set_index("stallion_id")
common2 = list(set(pivot_F.index) & set(pivot_MA.index))
print(f"  対象種牡馬数: {len(common2)}")
print(f"  {'cond':<24} {'median(M-F)':>13} {'sign+ %':>9} {'n':>5}")
diffs = {c: [] for c in COND_COLS}
for sid in common2:
    for c in COND_COLS:
        col = f"lift_{c}"
        if col in pivot_F.columns and col in pivot_MA.columns:
            fv = pivot_F.loc[sid, col]; mv = pivot_MA.loc[sid, col]
            if pd.notna(fv) and pd.notna(mv):
                diffs[c].append(float(mv) - float(fv))
sorted_diffs = sorted(diffs.items(), key=lambda x: np.median(x[1]) if x[1] else 0, reverse=True)
for c, vals in sorted_diffs:
    if not vals: continue
    med = np.median(vals); pos = sum(1 for v in vals if v>0)/len(vals)*100
    print(f"  {c:<24} {med:>+13.3f} {pos:>8.1f}% {len(vals):>5}")

# ───────── 指標 5b: 同上、母系深部 (MatDeep) vs F ─────────
print("\n========= [指標 5b] MatDeep (gen>=3) vs F: 条件別 lift 差 (中央値) =========")
pivot_MD = df[df["role"]=="MatDeep"].set_index("stallion_id")
common2b = list(set(pivot_F.index) & set(pivot_MD.index))
print(f"  対象種牡馬数: {len(common2b)}")
print(f"  {'cond':<24} {'median(MD-F)':>14} {'sign+ %':>9} {'n':>5}")
diffs2 = {c: [] for c in COND_COLS}
for sid in common2b:
    for c in COND_COLS:
        col = f"lift_{c}"
        if col in pivot_F.columns and col in pivot_MD.columns:
            fv = pivot_F.loc[sid, col]; mv = pivot_MD.loc[sid, col]
            if pd.notna(fv) and pd.notna(mv):
                diffs2[c].append(float(mv) - float(fv))
sorted_diffs2 = sorted(diffs2.items(), key=lambda x: np.median(x[1]) if x[1] else 0, reverse=True)
for c, vals in sorted_diffs2:
    if not vals: continue
    med = np.median(vals); pos = sum(1 for v in vals if v>0)/len(vals)*100
    print(f"  {c:<24} {med:>+14.3f} {pos:>8.1f}% {len(vals):>5}")

# ───────── 指標 6: gen 別の振る舞いの推移 ─────────
print("\n========= [指標 6] 母系 gen 別の lift 振幅・分散 (深いほど何が出るか) =========")
print(f"  {'role':<10} {'n_stall':>8} {'overall_std':>12} {'venue_std':>11} {'dist_std':>10} {'topo_std':>10} {'track_std':>11}")
for role in ["F","MF","MatG3","MatG4","MatG5+"]:
    sub = df[df["role"]==role]
    if sub.empty: continue
    venue_cols = [f"lift_{c}" for c in AXIS_CONDS["venue"] if f"lift_{c}" in sub.columns]
    dist_cols  = [f"lift_{c}" for c in AXIS_CONDS["distance"] if f"lift_{c}" in sub.columns]
    topo_cols  = [f"lift_{c}" for c in AXIS_CONDS["topo"] if f"lift_{c}" in sub.columns]
    track_cols = [f"lift_{c}" for c in AXIS_CONDS["track"] if f"lift_{c}" in sub.columns]
    all_cols = venue_cols + dist_cols + topo_cols + track_cols + [f"lift_{c}" for c in AXIS_CONDS["surface"] if f"lift_{c}" in sub.columns]
    o_std  = sub[all_cols].std(axis=1).median()
    v_std  = sub[venue_cols].std(axis=1).median()
    d_std  = sub[dist_cols].std(axis=1).median()
    t_std  = sub[topo_cols].std(axis=1).median()
    tk_std = float(sub[track_cols[0]].std()) if track_cols else float('nan')
    print(f"  {role:<10} {int(len(sub)):>8} {o_std:>12.3f} {v_std:>11.3f} {d_std:>10.3f} {t_std:>10.3f} {tk_std:>11.3f}")

# ───────── 指標 7: 父 vs 母系 各「軸の説明力」 ─────────
print("\n========= [指標 7] 父 vs 母系 が説明する「条件分散」の割合 =========")
print("  解釈: 各軸の説明分散を父/母系で比較し、どの軸が父・母どちらに帰属するか")
common_all = list(set(pivot_F.index) & set(pivot_MA.index))
print(f"  対象種牡馬数: {len(common_all)}")
print(f"  {'axis':<12} {'F の軸内 std':>14} {'MatAny の軸内 std':>18} {'比 MA/F':>10}")
for axis, conds in AXIS_CONDS.items():
    f_stds = []
    m_stds = []
    for sid in common_all:
        f_vals = []
        m_vals = []
        for c in conds:
            col = f"lift_{c}"
            if col in pivot_F.columns and col in pivot_MA.columns:
                fv = pivot_F.loc[sid, col]; mv = pivot_MA.loc[sid, col]
                if pd.notna(fv): f_vals.append(float(fv))
                if pd.notna(mv): m_vals.append(float(mv))
        if len(f_vals) >= 2: f_stds.append(np.std(f_vals))
        if len(m_vals) >= 2: m_stds.append(np.std(m_vals))
    f_med = np.median(f_stds) if f_stds else float('nan')
    m_med = np.median(m_stds) if m_stds else float('nan')
    ratio = m_med / f_med if f_med > 0 else float('nan')
    print(f"  {axis:<12} {f_med:>14.3f} {m_med:>18.3f} {ratio:>9.3f}")
