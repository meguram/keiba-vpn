"""役割（role）別 lift プロファイルの仮説検証スクリプト。

unified.parquet にある主流 4 大系統の主要種牡馬（n_horses>=30）について、
F/MF/MMF/FF の各 role での条件別 lift を計算し、以下の仮説を実データで検証する。

H1: 同じ種牡馬の lift が、role 間で統計的に異なるか（Wilcoxon 検定）
H2: 母系経由（MF, MMF）で系統的に強く出る条件は何か
H3: F の lift と MF の lift は独立か（相関）
H4: サンプル数 N と lift 分散の関係
"""
from __future__ import annotations

import json
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
STEEP_VENUES = ["中山", "阪神", "中京"]
COND_COLS = (
    [f"win_v_{v}" for v in ["東京", "中山", "阪神", "京都", "中京", "新潟", "小倉", "福島", "札幌", "函館"]]
    + [f"win_s_{s}" for s in ["芝", "ダート"]]
    + [f"win_d_{d}" for d in DIST_LABELS]
    + ["win_steep", "win_flat", "win_heavy"]
)

ROLES_TO_TEST = ["F", "MF", "MMF", "FF"]
MIN_N_RECORDS = 30  # role を有効と認める最低出走数


def _eb(w: float, n: int) -> float | None:
    if n < 3:
        return None
    return (w + EB_PRIOR * GLOBAL_WIN) / (n + EB_PRIOR)


def main():
    print("[load] data ...", flush=True)
    cats = pd.read_parquet(
        IDX_DIR / "horse_pedigree_cats.parquet",
        columns=["horse_id", "stallion_id", "path_fm"],
    )
    cats["horse_id"] = cats["horse_id"].astype(str)
    cats["stallion_id"] = cats["stallion_id"].astype(str)

    bms = pd.read_parquet(IDX_DIR / "horse_bms.parquet", columns=["horse_id", "sire_id", "bms_id"])
    bms["horse_id"] = bms["horse_id"].astype(str)
    bms["sire_id"] = bms["sire_id"].astype(str)
    bms["bms_id"] = bms["bms_id"].astype(str)

    race = pd.read_parquet(IDX_DIR / "race_result_slim.parquet")
    race["horse_id"] = race["horse_id"].astype(str)
    race = race[race["finish_position"].notna() & (race["finish_position"] > 0)].copy()
    race["win"] = (race["finish_position"] == 1).astype(int)
    race["dist_cat"] = pd.cut(race["distance"], bins=DIST_BINS, labels=DIST_LABELS).astype(str)
    race["is_steep"] = race["venue"].astype(str).isin(STEEP_VENUES)
    race["is_heavy"] = race["track_condition"].astype(str).isin(["重", "不良"])

    unified = pd.read_parquet(ART_DIR / "unified.parquet")
    main_stallions = unified[unified["entity_type"] == "stallion"][
        ["entity_id", "entity_label", "main_group"]
    ].rename(columns={"entity_id": "stallion_id", "entity_label": "stallion_name"})
    print(f"[ok] {len(main_stallions)} 種牡馬 (4 大主流のみ) を対象", flush=True)

    # 条件フィルタを事前マスク化
    def _filter_for(col: str):
        if col.startswith("win_v_"):
            return race["venue"].astype(str) == col.split("_")[-1]
        if col.startswith("win_s_"):
            return race["surface"].astype(str) == col.split("_")[-1]
        if col.startswith("win_d_"):
            return race["dist_cat"] == col.split("_")[-1]
        if col == "win_steep":
            return race["is_steep"]
        if col == "win_flat":
            return ~race["is_steep"]
        if col == "win_heavy":
            return race["is_heavy"]
        raise ValueError(col)

    cond_mask = {c: _filter_for(c) for c in COND_COLS}

    # role × stallion → horse_ids
    print("[idx] building role -> horse_ids index", flush=True)
    bms_by_sire = bms.groupby("sire_id")["horse_id"].apply(set).to_dict()
    bms_by_bms = bms.groupby("bms_id")["horse_id"].apply(set).to_dict()
    cats_mmf = cats[cats["path_fm"] == "MMF"].groupby("stallion_id")["horse_id"].apply(set).to_dict()
    cats_ff = cats[cats["path_fm"] == "FF"].groupby("stallion_id")["horse_id"].apply(set).to_dict()

    def _profile_for(horse_ids: set) -> dict[str, float] | None:
        if not horse_ids:
            return None
        mask = race["horse_id"].isin(horse_ids)
        sub = race[mask]
        if len(sub) < MIN_N_RECORDS:
            return None
        eb_total = _eb(sub["win"].sum(), len(sub))
        if eb_total is None:
            return None
        out: dict[str, float] = {"_eb_total": eb_total, "_n_total": int(len(sub)), "_n_horses": len(horse_ids)}
        for c in COND_COLS:
            sub_c = race[mask & cond_mask[c]]
            n = len(sub_c)
            e = _eb(sub_c["win"].sum(), n)
            if e is None:
                continue
            out[c] = e / eb_total
            out[f"_{c}_n"] = n
        return out

    rows: list[dict] = []
    for _, sr in main_stallions.iterrows():
        sid = sr["stallion_id"]
        for role in ROLES_TO_TEST:
            if role == "F":
                horses = bms_by_sire.get(sid, set())
            elif role == "MF":
                horses = bms_by_bms.get(sid, set())
            elif role == "MMF":
                horses = cats_mmf.get(sid, set())
            elif role == "FF":
                horses = cats_ff.get(sid, set())
            else:
                continue
            prof = _profile_for(horses)
            if prof is None:
                continue
            row = {
                "stallion_id": sid,
                "stallion_name": sr["stallion_name"],
                "main_group": sr["main_group"],
                "role": role,
                "n_horses": prof["_n_horses"],
                "n_records": prof["_n_total"],
                "eb_total": prof["_eb_total"],
            }
            for c in COND_COLS:
                if c in prof:
                    row[f"lift_{c}"] = prof[c]
                    row[f"n_{c}"] = prof[f"_{c}_n"]
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n[result] role × stallion: {len(df)} rows")
    print(f"  role 分布: {dict(df['role'].value_counts())}")
    print()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "role_lift_evidence.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[save] {out_path}")

    # ------- H1: 同じ stallion で role 別 lift が違うか -------
    print("\n========== H1: 同じ種牡馬の role 別 lift の差（pair-wise wilcoxon）==========")
    from scipy.stats import wilcoxon
    paired_diffs: dict[tuple[str, str], dict[str, list[float]]] = {}
    pivot = df.pivot_table(index="stallion_id", columns="role", values=[f"lift_{c}" for c in COND_COLS])
    show_cond = ["win_v_東京", "win_v_阪神", "win_v_中山", "win_v_京都", "win_d_長距離", "win_d_短距離",
                 "win_pace_持久力勝負_短距離" if "win_pace_持久力勝負_短距離" in COND_COLS else None,
                 "win_steep", "win_heavy", "win_s_芝", "win_s_ダート"]
    show_cond = [c for c in show_cond if c is not None and c in COND_COLS]
    print(f"\n  {'条件':<14} {'F vs MF':<25} {'F vs MMF':<25} {'F vs FF':<25}")
    for c in show_cond:
        line = f"  {c.split('_')[-1] if c.startswith('win_') else c:<14}"
        for other in ["MF", "MMF", "FF"]:
            try:
                pair = pivot[(f"lift_{c}", "F")].dropna().align(pivot[(f"lift_{c}", other)].dropna(), join="inner")
                a, b = pair[0], pair[1]
                if len(a) >= 5:
                    stat, p = wilcoxon(a, b)
                    diff = float(b.mean() - a.mean())
                    line += f"  Δ={diff:+.3f} (n={len(a)}, p={p:.3f})"
                else:
                    line += "  (n<5)                  "
            except Exception:
                line += "  err                     "
        print(line)

    # ------- H2: 母系で系統的に伸びる条件 -------
    print("\n========== H2: 母父役（MF）vs 父役（F）の lift 差（条件別、平均と sample 数）==========")
    print(f"  {'条件':<24} {'mean(MF-F)':>11} {'std':>6} {'有意馬数':>9} {'n総種牡馬':>11}")
    for c in COND_COLS:
        try:
            pair = pivot[(f"lift_{c}", "F")].dropna().align(pivot[(f"lift_{c}", "MF")].dropna(), join="inner")
            a, b = pair[0], pair[1]
            if len(a) < 8:
                continue
            diff = b - a
            n_pos = int((diff > 0.10).sum())  # +10% 以上母父役で上回った馬
            print(f"  {c:<24} {diff.mean():>+11.3f} {diff.std():>6.2f} {n_pos:>9} {len(diff):>11}")
        except Exception:
            continue

    # ------- H3: F vs MF lift の相関 -------
    print("\n========== H3: F vs MF lift の Pearson 相関（同一種牡馬内）==========")
    print(f"  {'条件':<24} {'相関':>8} {'n':>6}")
    for c in COND_COLS:
        try:
            pair = pivot[(f"lift_{c}", "F")].dropna().align(pivot[(f"lift_{c}", "MF")].dropna(), join="inner")
            a, b = pair[0], pair[1]
            if len(a) < 8:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            print(f"  {c:<24} {r:>+8.3f} {len(a):>6}")
        except Exception:
            continue

    # ------- H4: サンプル数と lift 分散 -------
    print("\n========== H4: サンプル数 N と lift 分散の関係（条件別の lift 分散） ==========")
    print(f"{'role':<5} {'n_records 中央値':>16} {'n_records 平均':>14} {'lift_avg_std':>14}  ← role 内・条件横断的な std")
    for role in ROLES_TO_TEST:
        sub = df[df["role"] == role]
        lift_cols = [f"lift_{c}" for c in COND_COLS if f"lift_{c}" in sub.columns]
        std_avg = float(sub[lift_cols].std().mean())
        print(f"{role:<5} {sub['n_records'].median():>16.0f} {sub['n_records'].mean():>14.0f} {std_avg:>14.3f}")

    # 役割別 サンプル数分布
    print()
    print(f"  役割別 種牡馬数 (n_records >= {MIN_N_RECORDS} の有効種牡馬):")
    print(f"    F:   {(df['role']=='F').sum()}  種牡馬")
    print(f"    MF:  {(df['role']=='MF').sum()}")
    print(f"    MMF: {(df['role']=='MMF').sum()}")
    print(f"    FF:  {(df['role']=='FF').sum()}")


if __name__ == "__main__":
    main()
