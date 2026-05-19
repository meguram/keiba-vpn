"""父系 vs 母系深部の「情報レイヤー差」を大規模に検証する（高速版）。

設計の高速化:
  - 各馬について、各条件 (cond_col) の「出走数」「勝利数」を事前に集計しておく
    => race を毎ループ走査せず、馬単位の総和で済ませる
  - role × stallion ループは、その role の該当馬群について上記集計を sum するだけ

仮説:
  H_paternal_broad : F (父) は大括り (場所/距離/路面) のシグナルが出やすい
  H_maternal_specific : MaternalDeep (母系深部) は細粒度 (馬場/急坂など) が出やすい
  H_signature_independence : F と MatDeep は条件別 lift で独立シグナル
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = ROOT / "data/analysis/pedigree"
IDX_DIR = ROOT / "data/research/pedigree_race_index"
ART_DIR = ROOT / "data/research/bloodline_meta_cluster"

EB_PRIOR = 30
GLOBAL_WIN = 0.139
DIST_BINS = [0, 1400, 1800, 2200, 9999]
DIST_LABELS = ["短距離", "マイル", "中距離", "長距離"]
STEEP_VENUES = ("中山", "阪神", "中京")

AXIS_CONDS: dict[str, list[str]] = {
    "venue":    [f"win_v_{v}" for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]],
    "surface":  [f"win_s_{s}" for s in ["芝","ダート"]],
    "distance": [f"win_d_{d}" for d in DIST_LABELS],
    "topo":     ["win_steep","win_flat"],
    "track":    ["win_heavy"],
}
COND_COLS: list[str] = sum(AXIS_CONDS.values(), [])
MIN_N_RECORDS = 80
MIN_PER_COND_N = 10  # 条件別 lift 計算最低 n


def _t(t0: float) -> str:
    return f"[{time.time()-t0:6.1f}s]"


def _eb(w, n, prior_p=GLOBAL_WIN):
    if n < 5: return None
    return (w + EB_PRIOR * prior_p) / (n + EB_PRIOR)


def main():
    t0 = time.time()
    print(f"{_t(t0)} [load] start", flush=True)

    cats = pd.read_parquet(
        IDX_DIR / "horse_pedigree_cats.parquet",
        columns=["horse_id", "stallion_id", "path_fm", "gen"],
    )
    cats["horse_id"] = cats["horse_id"].astype(str)
    cats["stallion_id"] = cats["stallion_id"].astype(str)

    bms = pd.read_parquet(IDX_DIR / "horse_bms.parquet", columns=["horse_id","sire_id","bms_id"])
    bms["horse_id"] = bms["horse_id"].astype(str)
    bms["sire_id"] = bms["sire_id"].astype(str)
    bms["bms_id"] = bms["bms_id"].astype(str)

    race = pd.read_parquet(IDX_DIR / "race_result_slim.parquet")
    race["horse_id"] = race["horse_id"].astype(str)
    race = race[race["finish_position"].notna() & (race["finish_position"] > 0)].copy()
    race["win"] = (race["finish_position"] == 1).astype(int)
    race["dist_cat"] = pd.cut(race["distance"], bins=DIST_BINS, labels=DIST_LABELS).astype(str)
    race["is_steep"] = race["venue"].astype(str).isin(STEEP_VENUES)
    race["is_heavy"] = race["track_condition"].astype(str).isin(["重","不良"])
    print(f"{_t(t0)} [load] race={len(race):,} cats={len(cats):,} bms={len(bms):,}", flush=True)

    # ── 1) 各 horse について cond ごとの (n_runs, n_wins) を事前集計
    print(f"{_t(t0)} [pre-agg] horse × cond の総和を計算", flush=True)
    horse_n: dict[str, dict[str, int]] = {}  # horse_id -> {cond: n}
    horse_w: dict[str, dict[str, int]] = {}  # horse_id -> {cond: w}
    horse_total_n: dict[str, int] = {}
    horse_total_w: dict[str, int] = {}

    # 各 cond → series マスク
    def _mask(c):
        if c.startswith("win_v_"): return race["venue"].astype(str) == c.split("_")[-1]
        if c.startswith("win_s_"): return race["surface"].astype(str) == c.split("_")[-1]
        if c.startswith("win_d_"): return race["dist_cat"] == c.split("_")[-1]
        if c == "win_steep": return race["is_steep"]
        if c == "win_flat":  return ~race["is_steep"]
        if c == "win_heavy": return race["is_heavy"]
        raise ValueError(c)

    # 各馬の総出走/勝
    g_total = race.groupby("horse_id")["win"].agg(["count","sum"])
    horse_total_n = g_total["count"].to_dict()
    horse_total_w = g_total["sum"].to_dict()
    print(f"{_t(t0)}   ... total per horse: {len(g_total):,}", flush=True)

    # 各 cond の per horse 集計
    cond_horse_n: dict[str, dict[str, int]] = {}
    cond_horse_w: dict[str, dict[str, int]] = {}
    for ic, c in enumerate(COND_COLS):
        sub = race[_mask(c)]
        g = sub.groupby("horse_id")["win"].agg(["count","sum"])
        cond_horse_n[c] = g["count"].to_dict()
        cond_horse_w[c] = g["sum"].to_dict()
        print(f"{_t(t0)}   cond {ic+1}/{len(COND_COLS)}: {c} -> {len(g):,} horses", flush=True)

    # ── 2) role 別 horse_ids index
    print(f"{_t(t0)} [idx] building role -> horse_ids", flush=True)
    sid_to_main = json.loads((ART_DIR / "sid_to_main.json").read_text(encoding="utf-8"))
    sid_to_name: dict[str,str] = {}
    for fn in ("sid_to_name_full.json","sid_to_name.json"):
        p = ART_DIR / fn
        if p.exists():
            sid_to_name.update(json.loads(p.read_text(encoding="utf-8")))

    bms_by_sire = bms.groupby("sire_id")["horse_id"].apply(set).to_dict()
    bms_by_bms = bms.groupby("bms_id")["horse_id"].apply(set).to_dict()
    cats["is_pat_deep"] = cats["path_fm"].str.fullmatch(r"F+", na=False) & (cats["gen"] >= 2)
    cats["is_mat_any"]  = cats["path_fm"].str.startswith("M", na=False) & cats["path_fm"].str.endswith("F", na=False)

    def _idx_from_mask(mask):
        sub = cats[mask]
        return sub.groupby("stallion_id")["horse_id"].apply(set).to_dict()

    role_indices: dict[str, dict[str, set[str]]] = {
        "F":       bms_by_sire,
        "MF":      bms_by_bms,
        "PatDeep": _idx_from_mask(cats["is_pat_deep"]),
        "MatAny":  _idx_from_mask(cats["is_mat_any"]),
        "MatDeep": _idx_from_mask(cats["is_mat_any"] & (cats["gen"] >= 3)),
        "MatG3":   _idx_from_mask(cats["is_mat_any"] & (cats["gen"] == 3)),
        "MatG4":   _idx_from_mask(cats["is_mat_any"] & (cats["gen"] == 4)),
        "MatG5+":  _idx_from_mask(cats["is_mat_any"] & (cats["gen"] >= 5)),
    }
    print(f"{_t(t0)} [idx] role -> #stallions: " + ", ".join(f"{k}={len(v)}" for k,v in role_indices.items()), flush=True)

    # ── 3) role × stallion ループ (高速集計)
    print(f"{_t(t0)} [run] start aggregation", flush=True)
    candidates: set[str] = set(sid_to_main.keys())
    for v in role_indices.values(): candidates |= set(v.keys())
    print(f"{_t(t0)}   candidates={len(candidates):,}", flush=True)

    rows: list[dict] = []
    for ri, (role, idx) in enumerate(role_indices.items()):
        n_cand = 0
        n_acc = 0
        for sid, horses in idx.items():
            n_cand += 1
            if not horses: continue
            # 総 n / w
            n_tot = sum(horse_total_n.get(h,0) for h in horses)
            if n_tot < MIN_N_RECORDS: continue
            w_tot = sum(horse_total_w.get(h,0) for h in horses)
            eb_tot = _eb(w_tot, n_tot)
            if eb_tot is None or eb_tot <= 0: continue
            row = {
                "stallion_id": sid,
                "stallion_name": sid_to_name.get(sid, sid),
                "main_group": sid_to_main.get(sid),
                "role": role,
                "n_horses": len(horses),
                "n_records": n_tot,
                "eb_total": eb_tot,
            }
            for c in COND_COLS:
                cn = cond_horse_n.get(c, {})
                cw = cond_horse_w.get(c, {})
                n_c = sum(cn.get(h,0) for h in horses)
                if n_c < MIN_PER_COND_N: continue
                w_c = sum(cw.get(h,0) for h in horses)
                e = _eb(w_c, n_c)
                if e is None: continue
                row[f"lift_{c}"] = e / eb_tot
                row[f"n_{c}"] = n_c
            rows.append(row)
            n_acc += 1
        print(f"{_t(t0)}   role={role:<10} processed {n_cand:,} cand, accepted {n_acc:,}", flush=True)

    df = pd.DataFrame(rows)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "deep_role_evidence.parquet"
    df.to_parquet(out_path, index=False)
    print(f"{_t(t0)} [save] {out_path}: {len(df):,} rows / {df['stallion_id'].nunique():,} unique stallions")
    print(f"  role 別 行数: {dict(df['role'].value_counts().sort_index())}")

    # ───────── 指標 1: 軸別 lift std ─────────
    print(f"\n{_t(t0)} ========= [指標 1] role 別・軸別の lift 分散 =========")
    print(f"  解釈: std 大 → その軸でメリハリのある適性が出る (種牡馬間中央値)")
    print(f"  {'role':<10} {'venue':>9} {'distance':>9} {'surface':>9} {'topo':>9} {'track':>9} {'overall':>9}")
    for role in ["F","MF","PatDeep","MatAny","MatDeep","MatG3","MatG4","MatG5+"]:
        sub = df[df["role"] == role]
        if sub.empty: continue
        line = f"  {role:<10}"
        all_stds = []
        for axis, conds in AXIS_CONDS.items():
            cols = [f"lift_{c}" for c in conds if f"lift_{c}" in sub.columns]
            if not cols:
                line += f" {'-':>9}"; continue
            stds_per = sub[cols].std(axis=1)
            med = float(np.nanmedian(stds_per))
            line += f"  {med:>8.3f}"
            all_stds.append(med)
        line += f"  {np.mean(all_stds):>8.3f}"
        print(line)

    # ───────── 指標 2: 最大変動軸の分布 ─────────
    print(f"\n{_t(t0)} ========= [指標 2] role 別・最大変動軸の分布 =========")
    print(f"  各種牡馬で最も std の大きい軸を投票し、role 内で集計")
    print(f"  {'role':<10} " + " ".join(f"{ax:>9}" for ax in AXIS_CONDS))
    for role in ["F","MF","PatDeep","MatAny","MatDeep"]:
        sub = df[df["role"] == role]
        if sub.empty: continue
        votes = {ax: 0 for ax in AXIS_CONDS}
        for _, r in sub.iterrows():
            best_ax, best_s = None, -1
            for axis, conds in AXIS_CONDS.items():
                cols = [f"lift_{c}" for c in conds if f"lift_{c}" in r.index]
                if not cols: continue
                vals = r[cols].dropna()
                if len(vals) < 2: continue
                s = float(vals.std())
                if s > best_s:
                    best_s, best_ax = s, axis
            if best_ax: votes[best_ax] += 1
        tot = sum(votes.values())
        line = f"  {role:<10}"
        for ax in AXIS_CONDS:
            v = votes.get(ax, 0)
            pct = (v/tot*100) if tot else 0
            line += f"  {pct:>7.1f}%"
        print(line + f"  (n={tot})")

    # ───────── 指標 3: F vs MatDeep 相関 ─────────
    print(f"\n{_t(t0)} ========= [指標 3] 同一種牡馬 F vs MatDeep の条件別 lift 相関 =========")
    pivot_F  = df[df["role"]=="F"].set_index("stallion_id")
    pivot_MD = df[df["role"]=="MatDeep"].set_index("stallion_id")
    common = list(set(pivot_F.index) & set(pivot_MD.index))
    print(f"  共通種牡馬数: {len(common)}")
    corrs = []
    for sid in common:
        f_v = pd.Series({c: pivot_F.loc[sid, f"lift_{c}"] for c in COND_COLS if f"lift_{c}" in pivot_F.columns}).dropna()
        d_v = pd.Series({c: pivot_MD.loc[sid, f"lift_{c}"] for c in COND_COLS if f"lift_{c}" in pivot_MD.columns}).dropna()
        cc = list(set(f_v.index) & set(d_v.index))
        if len(cc) < 5: continue
        r = float(np.corrcoef(f_v[cc], d_v[cc])[0,1])
        corrs.append(r)
    if corrs:
        print(f"  個別相関: 中央値={np.median(corrs):.3f}, 平均={np.mean(corrs):.3f}, 25/75%ile={np.percentile(corrs,25):.3f}/{np.percentile(corrs,75):.3f}")
        print(f"  独立的 (|r|<0.3) の馬数: {sum(1 for r in corrs if abs(r)<0.3)}/{len(corrs)} ({sum(1 for r in corrs if abs(r)<0.3)/len(corrs)*100:.1f}%)")
        print(f"  強い順方向 r>0.5  : {sum(1 for r in corrs if r>0.5)}/{len(corrs)}")
        print(f"  強い逆方向 r<-0.3: {sum(1 for r in corrs if r<-0.3)}/{len(corrs)}")

    # ───────── 指標 4: 大括り vs 細粒度 振幅比 ─────────
    print(f"\n{_t(t0)} ========= [指標 4] 大括り vs 細粒度 軸の lift 振幅 =========")
    print("  大括り = venue/distance/surface, 細粒度 = topo/track")
    print(f"  {'role':<10} {'BROAD振幅':>11} {'FINE振幅':>11} {'BROAD/FINE':>12}")
    for role in ["F","MF","PatDeep","MatAny","MatDeep","MatG3","MatG5+"]:
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

    # ───────── 指標 5: 父系 vs 母系 で「lift が体系的に上回る」条件 ─────────
    print(f"\n{_t(t0)} ========= [指標 5] 父系 vs 母系 で lift が体系的に上回る条件 =========")
    print("  各種牡馬で (MatAny lift - F lift) を計算し、条件別の中央値を見る")
    pivot_MA = df[df["role"]=="MatAny"].set_index("stallion_id")
    common2 = list(set(pivot_F.index) & set(pivot_MA.index))
    print(f"  対象種牡馬数: {len(common2)}")
    diffs = {c: [] for c in COND_COLS}
    for sid in common2:
        for c in COND_COLS:
            col = f"lift_{c}"
            if col in pivot_F.columns and col in pivot_MA.columns:
                fv = pivot_F.loc[sid, col]
                mv = pivot_MA.loc[sid, col]
                if pd.notna(fv) and pd.notna(mv):
                    diffs[c].append(float(mv) - float(fv))
    sorted_diffs = sorted(diffs.items(), key=lambda x: np.median(x[1]) if x[1] else 0, reverse=True)
    print(f"  {'cond':<24} {'median(M-F)':>13} {'sign+ %':>9} {'n':>5}")
    for c, vals in sorted_diffs:
        if not vals: continue
        med = np.median(vals); pos = sum(1 for v in vals if v>0)/len(vals)*100
        print(f"  {c:<24} {med:>+13.3f} {pos:>8.1f}% {len(vals):>5}")

    # ───────── 指標 6: F と MatDeep の同方向/逆方向 ─────────
    print(f"\n{_t(t0)} ========= [指標 6] F と MatDeep の条件別シグナル方向の一致度 =========")
    common3 = list(set(pivot_F.index) & set(pivot_MD.index))
    same_count = {c: {"same":0,"opp":0} for c in COND_COLS}
    for sid in common3:
        for c in COND_COLS:
            col = f"lift_{c}"
            if col in pivot_F.columns and col in pivot_MD.columns:
                fv = pivot_F.loc[sid, col]; dv = pivot_MD.loc[sid, col]
                if pd.notna(fv) and pd.notna(dv):
                    if (fv > 1.05 and dv > 1.05) or (fv < 0.95 and dv < 0.95):
                        same_count[c]["same"] += 1
                    elif (fv > 1.05 and dv < 0.95) or (fv < 0.95 and dv > 1.05):
                        same_count[c]["opp"] += 1
    print(f"  対象種牡馬数: {len(common3)}")
    print(f"  {'cond':<24} {'同方向':>7} {'逆方向':>7} {'同方向率':>9}")
    for c in COND_COLS:
        s = same_count[c]["same"]; o = same_count[c]["opp"]; tot = s+o
        if tot < 5: continue
        rate = s/tot*100
        print(f"  {c:<24} {s:>7} {o:>7} {rate:>8.1f}%")

    print(f"\n{_t(t0)} [done]")


if __name__ == "__main__":
    main()
