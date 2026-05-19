"""父 × 母父 (個体 / 系統 / クロス) で lift がどう増幅・相殺されるか検証する。

設計:
  ターゲット集合 (馬群) を 3 通りの粒度で構築:
    A) 父=Sx ×  母父=Mx           (個体ペア, サンプル少)
    B) 父=Sx ×  母父系統=Mg       (個別父 × 母父大系統 5 つ)        ← メイン
    C) 父系統=Sg × 母父=Mx        (父大系統 × 個別母父)

  各セルで n_records >= 50 を満たすものを採用し、条件別 lift を計算。
  父単独 (Sx ×*) と母父単独 (* × Mx) を baseline に、ペア/クロスで「上振れ・下振れ」を観察。

  発見したいパターン:
    - クロス相互作用 (interaction) :  pair lift > both single lifts (positive synergy)
    - 相殺 (cancellation) : single の lift が逆方向のとき、pair でどちらが勝つか
    - 系統的バイアス : 「父 X は母父○系で阪神得意になる」のような体系的シフト
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
COND_COLS = (
    [f"win_v_{v}" for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]]
    + [f"win_s_{s}" for s in ["芝","ダート"]]
    + [f"win_d_{d}" for d in DIST_LABELS]
    + ["win_steep","win_flat","win_heavy"]
)

MIN_N_SINGLE = 200    # 父・母父 単体の最低出走数
MIN_N_PAIR = 80       # ペア (B/C) の最低出走数


def _t(t0): return f"[{time.time()-t0:6.1f}s]"
def _eb(w, n):
    if n < 5: return None
    return (w + EB_PRIOR * GLOBAL_WIN) / (n + EB_PRIOR)


def _compute_profile(horse_ids: set[str], race: pd.DataFrame, cond_mask: dict) -> dict | None:
    if not horse_ids: return None
    mask = race["horse_id"].isin(horse_ids)
    sub = race[mask]
    n_total = int(len(sub))
    if n_total < MIN_N_PAIR: return None
    eb_total = _eb(sub["win"].sum(), n_total)
    if eb_total is None or eb_total <= 0: return None
    out = {"_eb_total": eb_total, "_n_total": n_total, "_n_horses": int(len(horse_ids))}
    for c in COND_COLS:
        sc = race[mask & cond_mask[c]]
        e = _eb(sc["win"].sum(), len(sc))
        if e is not None:
            out[c] = e / eb_total
            out[f"n_{c}"] = int(len(sc))
    return out


def main():
    t0 = time.time()
    print(f"{_t(t0)} [load]", flush=True)
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

    sid_to_main = json.loads((ART_DIR / "sid_to_main.json").read_text(encoding="utf-8"))
    sid_to_name: dict[str, str] = {}
    for fn in ("sid_to_name_full.json","sid_to_name.json"):
        p = ART_DIR / fn
        if p.exists():
            sid_to_name.update(json.loads(p.read_text(encoding="utf-8")))

    bms["sire_main"] = bms["sire_id"].map(sid_to_main)
    bms["bms_main"] = bms["bms_id"].map(sid_to_main)
    print(f"{_t(t0)} race={len(race):,}, bms={len(bms):,}, sid_to_main={len(sid_to_main)}", flush=True)
    print(f"  bms.sire_main 分布: {dict(bms['sire_main'].value_counts(dropna=False).head(8))}")
    print(f"  bms.bms_main  分布: {dict(bms['bms_main'].value_counts(dropna=False).head(8))}")

    def _mask(c):
        if c.startswith("win_v_"): return race["venue"].astype(str) == c.split("_")[-1]
        if c.startswith("win_s_"): return race["surface"].astype(str) == c.split("_")[-1]
        if c.startswith("win_d_"): return race["dist_cat"] == c.split("_")[-1]
        if c == "win_steep": return race["is_steep"]
        if c == "win_flat":  return ~race["is_steep"]
        if c == "win_heavy": return race["is_heavy"]
        raise ValueError(c)
    cond_mask = {c: _mask(c) for c in COND_COLS}

    # 父単体 lift (n>=200 の主要父のみ)
    print(f"\n{_t(t0)} [single profile] 父単体・母父単体", flush=True)
    sire_horses = bms.groupby("sire_id")["horse_id"].apply(set).to_dict()
    bms_horses = bms.groupby("bms_id")["horse_id"].apply(set).to_dict()

    sire_profiles = {}
    for sid, horses in sire_horses.items():
        if len(horses) < 5: continue
        prof = _compute_profile(horses, race, cond_mask)
        if prof and prof["_n_total"] >= MIN_N_SINGLE:
            sire_profiles[sid] = prof
    bms_profiles = {}
    for sid, horses in bms_horses.items():
        if len(horses) < 5: continue
        prof = _compute_profile(horses, race, cond_mask)
        if prof and prof["_n_total"] >= MIN_N_SINGLE:
            bms_profiles[sid] = prof
    print(f"{_t(t0)}   父プロファイル: {len(sire_profiles)} 種牡馬, 母父プロファイル: {len(bms_profiles)} 種牡馬")

    # ───────────────────────────────────
    # B) 父個体 × 母父系統 (主要 5 系統)
    # ───────────────────────────────────
    print(f"\n{_t(t0)} [B] 父個体 × 母父系統 のクロス集計", flush=True)
    MAIN_GROUPS = ["Turn-To系", "Native Dancer系", "Northern Dancer系", "Nasrullah系", "非主流"]
    rows_B = []
    sires_with_data = sorted(sire_profiles, key=lambda s: -sire_profiles[s]["_n_total"])
    for sid in sires_with_data:
        s_prof = sire_profiles[sid]
        sub = bms[bms["sire_id"] == sid]
        for mg in MAIN_GROUPS:
            mg_sub = sub[sub["bms_main"] == mg]
            if mg_sub.empty: continue
            horses = set(mg_sub["horse_id"])
            prof = _compute_profile(horses, race, cond_mask)
            if prof is None: continue
            row = {
                "sire_id": sid,
                "sire_name": sid_to_name.get(sid, sid),
                "sire_main": sid_to_main.get(sid),
                "bms_main": mg,
                "n_horses": prof["_n_horses"],
                "n_records": prof["_n_total"],
                "eb_total": prof["_eb_total"],
                "sire_eb_total": s_prof["_eb_total"],
                "sire_n_records": s_prof["_n_total"],
            }
            for c in COND_COLS:
                if c in prof:
                    row[f"lift_{c}"] = prof[c]
                    row[f"sire_lift_{c}"] = s_prof.get(c)
            rows_B.append(row)
    df_B = pd.DataFrame(rows_B)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_b = ANALYSIS_DIR / "cross_term_B.parquet"
    df_B.to_parquet(out_b, index=False)
    print(f"{_t(t0)} [save] {out_b}  rows={len(df_B):,}, 父={df_B['sire_id'].nunique()}")
    print(f"  bms_main 分布: {dict(df_B['bms_main'].value_counts())}")

    # B 分析: ペア lift と父単体 lift の差を集計
    print(f"\n{_t(t0)} ========= [B 分析 1] 父個体 × 母父系統 のペア lift と 父単体 lift の差 =========")
    print("  各 (父, 母父系統) ペアで 'lift_pair - lift_sire_alone' を計算し、母父系統別に集計")
    target_conds = ["win_v_東京","win_v_中山","win_v_阪神","win_v_京都","win_d_長距離","win_d_短距離","win_steep","win_heavy","win_s_芝","win_s_ダート"]
    print(f"\n  {'母父系統':<24} " + " ".join(f"{c.split('_')[-1]:>7}" for c in target_conds))
    for mg in MAIN_GROUPS:
        sub = df_B[df_B["bms_main"] == mg]
        if sub.empty: continue
        line = f"  {mg:<24}"
        for c in target_conds:
            d = sub[f"lift_{c}"] - sub[f"sire_lift_{c}"]
            d = d.dropna()
            if len(d) < 5:
                line += f" {'-':>7}"
            else:
                line += f"  {d.median():>+6.3f}"
        print(line + f"  (n_pairs={len(sub)})")

    # B 分析: 父系統別の母父系統への振る舞い (4×4 matrix)
    print(f"\n{_t(t0)} ========= [B 分析 2] 父系統 × 母父系統 マトリクス: 阪神 lift 中央値 =========")
    print(f"  父系統 \\ 母父系統 → " + " ".join(f"{m[:6]:>8}" for m in MAIN_GROUPS))
    for sg in MAIN_GROUPS:
        line = f"  {sg:<22}"
        for mg in MAIN_GROUPS:
            sub = df_B[(df_B["sire_main"] == sg) & (df_B["bms_main"] == mg)]
            if sub.empty:
                line += f"  {'-':>8}"
                continue
            v = sub["lift_win_v_阪神"].median()
            n = len(sub)
            line += f"  {v:>5.2f}({n:>2})"
        print(line)

    print(f"\n{_t(t0)} ========= [B 分析 2b] 同様: 長距離 lift =========")
    print(f"  父系統 \\ 母父系統 → " + " ".join(f"{m[:6]:>8}" for m in MAIN_GROUPS))
    for sg in MAIN_GROUPS:
        line = f"  {sg:<22}"
        for mg in MAIN_GROUPS:
            sub = df_B[(df_B["sire_main"] == sg) & (df_B["bms_main"] == mg)]
            if sub.empty:
                line += f"  {'-':>8}"; continue
            v = sub["lift_win_d_長距離"].median()
            n = len(sub)
            line += f"  {v:>5.2f}({n:>2})"
        print(line)

    print(f"\n{_t(t0)} ========= [B 分析 2c] 同様: 重・不良 lift =========")
    print(f"  父系統 \\ 母父系統 → " + " ".join(f"{m[:6]:>8}" for m in MAIN_GROUPS))
    for sg in MAIN_GROUPS:
        line = f"  {sg:<22}"
        for mg in MAIN_GROUPS:
            sub = df_B[(df_B["sire_main"] == sg) & (df_B["bms_main"] == mg)]
            if sub.empty:
                line += f"  {'-':>8}"; continue
            v = sub["lift_win_heavy"].median()
            n = len(sub)
            line += f"  {v:>5.2f}({n:>2})"
        print(line)

    # B 分析: 個別の興味深い父について、母父系統で lift がどう動くか
    print(f"\n{_t(t0)} ========= [B 分析 3] 主要父 × 母父系統 の lift 推移 (代表例) =========")
    SHOW_SIRES = ["2002100816","2001103038","2008103552","2001103460","2007103143",
                  "2012104511","2008102636","1994108729","2010105827","2011100655"]
    show_conds = ["win_v_東京","win_v_阪神","win_v_中山","win_d_長距離","win_d_短距離","win_steep","win_heavy"]
    for sid in SHOW_SIRES:
        sub = df_B[df_B["sire_id"] == sid]
        if sub.empty: continue
        sname = sid_to_name.get(sid, sid)
        s_prof = sire_profiles.get(sid, {})
        print(f"\n  {sname} (n_父={s_prof.get('_n_total','-')}, eb={s_prof.get('_eb_total',0):.3f})")
        print(f"    {'母父系統':<22} {'n':>5} " + " ".join(f"{c.split('_')[-1]:>6}" for c in show_conds))
        # 父単体行
        line = f"    {'(父単体)':<22} {s_prof.get('_n_total',0):>5} "
        for c in show_conds:
            v = s_prof.get(c)
            line += f" {v:>6.2f}" if v is not None else f" {'-':>6}"
        print(line)
        for _, r in sub.sort_values("n_records", ascending=False).iterrows():
            line = f"    {r['bms_main']:<22} {r['n_records']:>5} "
            for c in show_conds:
                v = r.get(f"lift_{c}")
                line += f" {v:>6.2f}" if pd.notna(v) else f" {'-':>6}"
            print(line)

    # B 分析: 父-母父系統 で「相互作用」が顕著な条件
    print(f"\n\n{_t(t0)} ========= [B 分析 4] 父個体 ごとの 母父系統内 lift std (相互作用の強さ) =========")
    print("  std 大 → 母父系統で大きく振れる (相互作用強)")
    print(f"  {'父':<22} {'n_pairs':>8}  " + " ".join(f"{c.split('_')[-1]:>7}" for c in target_conds[:7]))
    inter_rows = []
    for sid, sub in df_B.groupby("sire_id"):
        if len(sub) < 3: continue
        sname = sid_to_name.get(sid, sid)
        line = f"  {sname[:20]:<22} {len(sub):>8}  "
        stds = []
        for c in target_conds[:7]:
            col = f"lift_{c}"
            if col in sub.columns:
                v = sub[col].std()
                line += f"  {v:>5.2f}"
                stds.append(v)
            else:
                line += f"  {'-':>5}"
        if any(s > 0.10 for s in stds if pd.notna(s)):
            print(line)
        inter_rows.append({"sire_id": sid, "sire_name": sname,
                           "mean_std": float(np.nanmean(stds))})

    inter_df = pd.DataFrame(inter_rows).sort_values("mean_std", ascending=False)
    print(f"\n  ── 母父系統間の lift 振れが大きい上位 父 ──")
    for _, r in inter_df.head(15).iterrows():
        print(f"    {r['sire_name']:<24}  mean_std={r['mean_std']:.3f}")

    # ───────────────────────────────────
    # C) 父系統 × 母父個体 (補完情報)
    # ───────────────────────────────────
    print(f"\n\n{_t(t0)} [C] 父系統 × 母父個体 のクロス集計 (n_pair>={MIN_N_PAIR})", flush=True)
    rows_C = []
    bms_with_data = sorted(bms_profiles, key=lambda s: -bms_profiles[s]["_n_total"])
    for bid in bms_with_data:
        b_prof = bms_profiles[bid]
        sub = bms[bms["bms_id"] == bid]
        for sg in MAIN_GROUPS:
            sg_sub = sub[sub["sire_main"] == sg]
            if sg_sub.empty: continue
            horses = set(sg_sub["horse_id"])
            prof = _compute_profile(horses, race, cond_mask)
            if prof is None: continue
            row = {
                "bms_id": bid,
                "bms_name": sid_to_name.get(bid, bid),
                "bms_main": sid_to_main.get(bid),
                "sire_main": sg,
                "n_horses": prof["_n_horses"],
                "n_records": prof["_n_total"],
                "eb_total": prof["_eb_total"],
                "bms_eb_total": b_prof["_eb_total"],
                "bms_n_records": b_prof["_n_total"],
            }
            for c in COND_COLS:
                if c in prof:
                    row[f"lift_{c}"] = prof[c]
                    row[f"bms_lift_{c}"] = b_prof.get(c)
            rows_C.append(row)
    df_C = pd.DataFrame(rows_C)
    out_c = ANALYSIS_DIR / "cross_term_C.parquet"
    df_C.to_parquet(out_c, index=False)
    print(f"{_t(t0)} [save] {out_c}  rows={len(df_C):,}, 母父={df_C['bms_id'].nunique()}")

    # ───────────────────────────────────
    # A) 父個体 × 母父個体 (n>=80 のペアのみ。ニッチ)
    # ───────────────────────────────────
    print(f"\n{_t(t0)} [A] 父個体 × 母父個体 ペア (n>={MIN_N_PAIR})", flush=True)
    pair_horses = bms.groupby(["sire_id","bms_id"])["horse_id"].apply(set).to_dict()
    rows_A = []
    for (sid, bid), horses in pair_horses.items():
        if len(horses) < 5: continue
        prof = _compute_profile(horses, race, cond_mask)
        if prof is None: continue
        row = {
            "sire_id": sid, "sire_name": sid_to_name.get(sid, sid),
            "bms_id": bid, "bms_name": sid_to_name.get(bid, bid),
            "sire_main": sid_to_main.get(sid),
            "bms_main": sid_to_main.get(bid),
            "n_horses": prof["_n_horses"],
            "n_records": prof["_n_total"],
            "eb_total": prof["_eb_total"],
        }
        for c in COND_COLS:
            if c in prof:
                row[f"lift_{c}"] = prof[c]
        rows_A.append(row)
    df_A = pd.DataFrame(rows_A)
    out_a = ANALYSIS_DIR / "cross_term_A.parquet"
    df_A.to_parquet(out_a, index=False)
    print(f"{_t(t0)} [save] {out_a}  rows={len(df_A):,}, ユニーク父={df_A['sire_id'].nunique()}, 母父={df_A['bms_id'].nunique()}")

    # A 分析: 高 lift 出現ペアの抽出
    print(f"\n{_t(t0)} ========= [A 分析] 父個体×母父個体ペア: 顕著なシグナル =========")
    print(f"  阪神得意ペア (lift_v_阪神 >= 1.20, n_records>=100, 上位 15)")
    aa = df_A[(df_A["n_records"] >= 100) & (df_A["lift_win_v_阪神"] >= 1.20)].sort_values("lift_win_v_阪神", ascending=False).head(15)
    print(f"  {'父':<22}{'母父':<22}{'n':>6}{'阪神':>7}{'長距離':>8}{'重':>7}")
    for _, r in aa.iterrows():
        print(f"  {r['sire_name'][:20]:<22}{r['bms_name'][:20]:<22}{r['n_records']:>6} "
              f"{r['lift_win_v_阪神']:>6.2f} {r.get('lift_win_d_長距離', float('nan')):>7.2f} {r.get('lift_win_heavy', float('nan')):>6.2f}")

    print(f"\n  長距離得意ペア (lift_d_長距離 >= 1.30, n_records>=100, 上位 15)")
    aa = df_A[(df_A["n_records"] >= 100) & (df_A["lift_win_d_長距離"] >= 1.30)].sort_values("lift_win_d_長距離", ascending=False).head(15)
    print(f"  {'父':<22}{'母父':<22}{'n':>6}{'長距離':>8}{'阪神':>7}{'東京':>7}")
    for _, r in aa.iterrows():
        print(f"  {r['sire_name'][:20]:<22}{r['bms_name'][:20]:<22}{r['n_records']:>6} "
              f"{r['lift_win_d_長距離']:>7.2f} {r.get('lift_win_v_阪神', float('nan')):>6.2f} {r.get('lift_win_v_東京', float('nan')):>6.2f}")

    print(f"\n  道悪得意ペア (lift_heavy >= 1.30, n_records>=100, 上位 15)")
    aa = df_A[(df_A["n_records"] >= 100) & (df_A["lift_win_heavy"] >= 1.30)].sort_values("lift_win_heavy", ascending=False).head(15)
    print(f"  {'父':<22}{'母父':<22}{'n':>6}{'重':>7}{'長距離':>8}{'阪神':>7}")
    for _, r in aa.iterrows():
        print(f"  {r['sire_name'][:20]:<22}{r['bms_name'][:20]:<22}{r['n_records']:>6} "
              f"{r['lift_win_heavy']:>6.2f} {r.get('lift_win_d_長距離', float('nan')):>7.2f} {r.get('lift_win_v_阪神', float('nan')):>6.2f}")

    print(f"\n{_t(t0)} [done]")


if __name__ == "__main__":
    main()
