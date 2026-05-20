"""v2 (複数 layer 加重ブレンド + shrinkage) の血統 prior をバックテスト。

v1 の単一層採用と同じ評価軸で実行し、改善の有無を確認する。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = ROOT / "data/local/analysis/pedigree"
sys.path.insert(0, str(ROOT))

IDX = ROOT / "data/page_reference/pedigree_race_index"
ART = ROOT / "data/page_reference/note_aptitude_race"

DIST_BINS = [0, 1400, 1800, 2400, 9999]
DIST_LABELS = ["短距離","マイル","中距離","長距離"]
STEEP_VENUES = ("中山","阪神","中京")
COND_COLS = (
    [f"win_v_{v}" for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]]
    + [f"win_s_{s}" for s in ["芝","ダート"]]
    + [f"win_d_{d}" for d in DIST_LABELS]
    + ["win_steep","win_flat","win_heavy"]
)

# v2 設定 (API と一致させる)
LAYER_BASE_WEIGHT = {
    "pair_indiv": 0.50,
    "pair_group": 0.30,
    "role_blend": 0.30,
    "pair_gxg":   0.10,
}
LAYER_PSEUDO_N = {
    "pair_indiv": 80,
    "pair_group": 200,
    "role_blend": 100,
    "pair_gxg":   500,
}
SHRINK_RATIO = 0.85


def _t(t0): return f"[{time.time()-t0:5.1f}s]"


def _cond_filter(rec, c):
    if c.startswith("win_v_"): return rec["venue"].astype(str) == c.split("_")[-1]
    if c.startswith("win_s_"): return rec["surface"].astype(str) == c.split("_")[-1]
    if c.startswith("win_d_"): return rec["dist_cat"] == c.split("_")[-1]
    if c == "win_steep": return rec["is_steep"]
    if c == "win_flat":  return ~rec["is_steep"]
    if c == "win_heavy": return rec["is_heavy"]
    raise ValueError(c)


def main():
    t0 = time.time()
    print(f"{_t(t0)} load")
    race = pd.read_parquet(IDX / "race_result_slim.parquet")
    race["horse_id"] = race["horse_id"].astype(str)
    race = race[race["finish_position"].notna() & (race["finish_position"] > 0)].copy()
    race["win"] = (race["finish_position"] == 1).astype(int)
    race["dist_cat"] = pd.cut(race["distance"], bins=DIST_BINS, labels=DIST_LABELS).astype(str)
    race["is_steep"] = race["venue"].astype(str).isin(STEEP_VENUES)
    race["is_heavy"] = race["track_condition"].astype(str).isin(["重","不良"])

    bms = pd.read_parquet(IDX / "horse_bms.parquet", columns=["horse_id","sire_id","bms_id"])
    bms["horse_id"] = bms["horse_id"].astype(str)
    bms["sire_id"] = bms["sire_id"].astype(str)
    bms["bms_id"] = bms["bms_id"].astype(str)
    sid_to_main = json.loads((ART/"sid_to_main.json").read_text(encoding="utf-8"))
    bms["sire_main"] = bms["sire_id"].map(sid_to_main)
    bms["bms_main"] = bms["bms_id"].map(sid_to_main)

    pair_indiv = pd.read_parquet(ART / "pair_lift_profiles_indiv.parquet")
    pair_group = pd.read_parquet(ART / "pair_lift_profiles_group.parquet")
    pair_gxg   = pd.read_parquet(ART / "pair_lift_profiles_gxg.parquet")
    pair_indiv["sire_id"] = pair_indiv["sire_id"].astype(str)
    pair_indiv["bms_id"] = pair_indiv["bms_id"].astype(str)
    pair_group["sire_id"] = pair_group["sire_id"].astype(str)

    role_lift = pd.read_parquet(ART / "role_lift_profiles.parquet")
    role_lift["stallion_id"] = role_lift["stallion_id"].astype(str)

    print(f"{_t(t0)} target prep")
    by_horse = race.groupby("horse_id")["win"].agg(["count","sum"]).reset_index()
    by_horse.columns = ["horse_id","n_total","win_total"]
    by_horse = by_horse[by_horse["n_total"] >= 8].copy()
    by_horse["rate_total"] = by_horse["win_total"] / by_horse["n_total"]
    target_set = set(by_horse["horse_id"])
    rec = race[race["horse_id"].isin(target_set)].copy()
    print(f"{_t(t0)} target horses = {len(by_horse):,}, records = {len(rec):,}")

    # ── 各 layer のインデックス ──
    pi_idx = {(r["sire_id"], r["bms_id"]): r for _, r in pair_indiv.iterrows()}
    pg_idx = {(r["sire_id"], r["bms_main"]): r for _, r in pair_group.iterrows()}
    gxg_idx = {(r["sire_main"], r["bms_main"]): r for _, r in pair_gxg.iterrows()}

    role_lift_dict: dict[tuple[str, str], dict[str, float]] = {}
    role_n_dict: dict[tuple[str, str], int] = {}
    for _, r in role_lift.iterrows():
        d = {c.replace("lift_", ""): r[c] for c in role_lift.columns
             if c.startswith("lift_") and pd.notna(r[c])}
        role_lift_dict[(str(r["stallion_id"]), str(r["role"]))] = d
        role_n_dict[(str(r["stallion_id"]), str(r["role"]))] = int(r.get("n_records", 0))

    bms_map = bms.set_index("horse_id")[["sire_id","bms_id","sire_main","bms_main"]].to_dict("index")

    def _row_lift_dict(row):
        return {c.replace("lift_", ""): float(row[c]) for c in row.index
                if isinstance(c, str) and c.startswith("lift_") and pd.notna(row[c])}

    def _role_blend_lift(sid, bid):
        f = role_lift_dict.get((str(sid), "F")) if sid else None
        m = role_lift_dict.get((str(bid), "MF")) if bid else None
        n_f = role_n_dict.get((str(sid), "F"), 0) if sid else 0
        n_m = role_n_dict.get((str(bid), "MF"), 0) if bid else 0
        if not f and not m: return None, 0
        keys = set(f or {}).union(set(m or {}))
        out = {}
        for k in keys:
            vals, ws = [], []
            if f and k in f: vals.append(f[k]); ws.append(0.6)
            if m and k in m: vals.append(m[k]); ws.append(0.4)
            if vals: out[k] = sum(v*w for v,w in zip(vals, ws)) / sum(ws)
        return out, n_f + n_m

    # ── v2 prior 構築 (馬ごと) ──
    print(f"{_t(t0)} build v2 priors per horse")
    rows_v2 = []
    n_done = 0
    for hid in by_horse["horse_id"]:
        n_done += 1
        if n_done % 5000 == 0:
            print(f"  ... {n_done}/{len(by_horse)}", flush=True)
        info = bms_map.get(hid)
        if not info: continue
        sid = info["sire_id"]; bid = info["bms_id"]
        sg = info["sire_main"]; mg = info["bms_main"]

        components = {}

        # pair_indiv
        if (sid, bid) in pi_idx:
            row = pi_idx[(sid, bid)]
            n = int(row.get("n_records", 0))
            components["pair_indiv"] = {
                "lift": _row_lift_dict(row), "n": n,
                "w": LAYER_BASE_WEIGHT["pair_indiv"] * n / (n + LAYER_PSEUDO_N["pair_indiv"]),
            }
        # pair_group
        if (sid, mg) in pg_idx:
            row = pg_idx[(sid, mg)]
            n = int(row.get("n_records", 0))
            components["pair_group"] = {
                "lift": _row_lift_dict(row), "n": n,
                "w": LAYER_BASE_WEIGHT["pair_group"] * n / (n + LAYER_PSEUDO_N["pair_group"]),
            }
        # role_blend
        rb_lift, rb_n = _role_blend_lift(sid, bid)
        if rb_lift:
            components["role_blend"] = {
                "lift": rb_lift, "n": rb_n,
                "w": LAYER_BASE_WEIGHT["role_blend"] * rb_n / (rb_n + LAYER_PSEUDO_N["role_blend"]),
            }
        # pair_gxg
        if (sg, mg) in gxg_idx:
            row = gxg_idx[(sg, mg)]
            n = int(row.get("n_records", 0))
            components["pair_gxg"] = {
                "lift": _row_lift_dict(row), "n": n,
                "w": LAYER_BASE_WEIGHT["pair_gxg"] * n / (n + LAYER_PSEUDO_N["pair_gxg"]),
            }

        if not components: continue
        w_sum = sum(c["w"] for c in components.values())
        if w_sum <= 0: continue
        weights_norm = {k: c["w"] / w_sum for k, c in components.items()}
        all_conds = set().union(*[c["lift"].keys() for c in components.values()])
        for cond in all_conds:
            num = 0.0; den = 0.0
            for k, c in components.items():
                if cond in c["lift"]:
                    w = weights_norm[k]
                    num += w * c["lift"][cond]; den += w
            if den > 0:
                raw = num / den
                shrunk = 1.0 + (raw - 1.0) * SHRINK_RATIO
                rows_v2.append({
                    "horse_id": hid, "cond": cond,
                    "prior_lift_raw": raw, "prior_lift": shrunk,
                    "n_layers": len(components),
                    "max_layer": max(components.keys(), key=lambda k: components[k]["w"]),
                })

    prior_df = pd.DataFrame(rows_v2)
    print(f"{_t(t0)} v2 prior rows = {len(prior_df):,}")

    # ── actual lift bulk ──
    print(f"{_t(t0)} compute actual")
    by_horse = by_horse.set_index("horse_id")
    actuals = []
    for c in COND_COLS:
        mask = _cond_filter(rec, c)
        sc = rec[mask]
        gc = sc.groupby("horse_id")["win"].agg(["count","sum"]).reset_index()
        gc.columns = ["horse_id","n_cond","win_cond"]
        gc = gc.merge(by_horse[["n_total","win_total","rate_total"]].reset_index(), on="horse_id", how="left")
        gc["rate_cond"] = gc["win_cond"] / gc["n_cond"]
        gc["actual_lift"] = gc["rate_cond"] / gc["rate_total"]
        gc["cond"] = c
        actuals.append(gc[["horse_id","cond","n_cond","win_cond","rate_cond","rate_total","actual_lift"]])
    act_df = pd.concat(actuals, ignore_index=True)

    df = prior_df.merge(act_df, on=["horse_id","cond"], how="inner")
    df = df[df["actual_lift"].notna() & (df["rate_total"] > 0) & (df["n_cond"] >= 3)].copy()
    print(f"{_t(t0)} joined rows = {len(df):,}")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ANALYSIS_DIR / "backtest_prior_v2.parquet", index=False)

    # ────────────────────────────────────────────────
    # キャリブレーション
    # ────────────────────────────────────────────────
    print(f"\n{_t(t0)} ========= [v2-1] キャリブレーション =========")
    bins = [0, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 5]
    df["prior_bin"] = pd.cut(df["prior_lift"], bins=bins)
    g = df.groupby("prior_bin", observed=True).agg(
        n_pairs=("actual_lift","count"),
        win_sum=("win_cond","sum"),
        n_sum=("n_cond","sum"),
        rate_total_w=("rate_total", lambda s: float(np.average(s, weights=df.loc[s.index,"n_cond"]))),
    )
    g["actual_rate_w"] = g["win_sum"] / g["n_sum"]
    g["actual_lift_w"] = g["actual_rate_w"] / g["rate_total_w"]
    print(g[["n_pairs","n_sum","actual_rate_w","rate_total_w","actual_lift_w"]])

    print(f"\n{_t(t0)} ========= [v2-2] 高/低 prior の actual 比較 =========")
    high = df[df["prior_lift"] >= 1.2]
    low  = df[df["prior_lift"] <= 0.8]
    print(f"  high (prior>=1.2): n={len(high):,}, actual_lift={high['actual_lift'].mean():.3f}, "
          f"win_rate={high['win_cond'].sum()/high['n_cond'].sum():.3f}")
    print(f"  low  (prior<=0.8): n={len(low):,}, actual_lift={low['actual_lift'].mean():.3f}, "
          f"win_rate={low['win_cond'].sum()/low['n_cond'].sum():.3f}")

    print(f"\n{_t(t0)} ========= [v2-3] n_layers 別精度 =========")
    for nl in sorted(df["n_layers"].unique()):
        sub = df[df["n_layers"] == nl]
        if len(sub) < 100: continue
        corr = sub["prior_lift"].corr(sub["actual_lift"])
        h = sub[sub["prior_lift"]>=1.2]["actual_lift"].mean()
        l = sub[sub["prior_lift"]<=0.8]["actual_lift"].mean()
        print(f"  n_layers={nl}  n={len(sub):>7,}  corr={corr:>+5.3f}  "
              f"actual(高)={h:>5.3f}  actual(低)={l:>5.3f}  Δ={h-l:>+5.3f}")

    print(f"\n{_t(t0)} ========= [v2-3b] dominant layer 別精度 =========")
    for layer in ["pair_indiv","pair_group","role_blend","pair_gxg"]:
        sub = df[df["max_layer"] == layer]
        if len(sub) < 100: continue
        corr = sub["prior_lift"].corr(sub["actual_lift"])
        h = sub[sub["prior_lift"]>=1.2]["actual_lift"].mean()
        l = sub[sub["prior_lift"]<=0.8]["actual_lift"].mean()
        print(f"  max_layer={layer:<14}  n={len(sub):>7,}  corr={corr:>+5.3f}  "
              f"actual(高)={h:>5.3f}  actual(低)={l:>5.3f}  Δ={h-l:>+5.3f}")

    print(f"\n{_t(t0)} ========= [v2-4] 条件別精度 (主要) =========")
    print(f"  {'cond':<22}{'n':>7}{'corr':>7}{'高prior':>9}{'低prior':>9}{'Δ':>7}")
    for c in ["win_v_東京","win_v_中山","win_v_阪神","win_v_京都","win_v_中京",
              "win_v_新潟","win_v_札幌","win_v_福島","win_v_小倉","win_v_函館",
              "win_d_短距離","win_d_マイル","win_d_中距離","win_d_長距離",
              "win_s_芝","win_s_ダート","win_steep","win_flat","win_heavy"]:
        sub = df[df["cond"] == c]
        if len(sub) < 200: continue
        corr = sub["prior_lift"].corr(sub["actual_lift"])
        h = sub[sub["prior_lift"]>=1.2]["actual_lift"].mean()
        l = sub[sub["prior_lift"]<=0.8]["actual_lift"].mean()
        print(f"  {c:<22}{len(sub):>7,}{corr:>+7.3f}{h:>9.3f}{l:>9.3f}{h-l:>+7.3f}")

    print(f"\n{_t(t0)} ========= [v2-5] 方向性一致率 =========")
    df["prior_sign"] = np.sign(df["prior_lift"] - 1.0)
    df["actual_sign"] = np.sign(df["actual_lift"] - 1.0)
    sig = df[(df["prior_lift"] - 1.0).abs() >= 0.10]
    agree = (sig["prior_sign"] == sig["actual_sign"]).mean()
    print(f"  全 (|prior-1|>=0.10): n={len(sig):,}, 一致率={agree:.3f}")

    # ────────────────────────────────────────────────
    # v1 vs v2 同条件比較
    # ────────────────────────────────────────────────
    print(f"\n{_t(t0)} ========= [v1 vs v2 比較] =========")
    prior_v1 = ANALYSIS_DIR / "backtest_prior.parquet"
    if prior_v1.exists():
        v1 = pd.read_parquet(prior_v1)
        # v1 は actual_lift も含むので、prior_lift だけリネームして merge
        v1 = v1[["horse_id","cond","prior_lift","layer"]].rename(
            columns={"prior_lift": "prior_lift_v1", "layer": "layer_v1"})
        v2_slim = df[["horse_id","cond","prior_lift","actual_lift","n_cond","rate_total","n_layers"]].rename(
            columns={"prior_lift": "prior_lift_v2"})
        cmp = v1.merge(v2_slim, on=["horse_id","cond"], how="inner")
        print(f"  共通 (horse,cond) ペア: {len(cmp):,}")
        # 全体 corr
        c1 = cmp["prior_lift_v1"].corr(cmp["actual_lift"])
        c2 = cmp["prior_lift_v2"].corr(cmp["actual_lift"])
        print(f"  全体 corr (prior vs actual):  v1={c1:+.4f}  v2={c2:+.4f}")
        # 高/低での差
        for thr in (1.2, 1.3, 1.4):
            h1 = cmp[cmp["prior_lift_v1"]>=thr]["actual_lift"].mean()
            h2 = cmp[cmp["prior_lift_v2"]>=thr]["actual_lift"].mean()
            l1 = cmp[cmp["prior_lift_v1"]<=2-thr]["actual_lift"].mean()
            l2 = cmp[cmp["prior_lift_v2"]<=2-thr]["actual_lift"].mean()
            print(f"  thr={thr:.1f}: v1 高={h1:.3f} 低={l1:.3f} (Δ={h1-l1:+.3f})  "
                  f"v2 高={h2:.3f} 低={l2:.3f} (Δ={h2-l2:+.3f})")
        # 方向性一致率
        for tag, lift in (("v1","prior_lift_v1"), ("v2","prior_lift_v2")):
            cmp[f"sign_{tag}"] = np.sign(cmp[lift] - 1.0)
            cmp[f"agree_{tag}"] = (cmp[f"sign_{tag}"] == np.sign(cmp["actual_lift"] - 1.0))
        sig_cmp = cmp[(cmp["prior_lift_v1"]-1).abs() >= 0.10]
        a1 = sig_cmp["agree_v1"].mean()
        sig_cmp = cmp[(cmp["prior_lift_v2"]-1).abs() >= 0.10]
        a2 = sig_cmp["agree_v2"].mean()
        print(f"  方向性一致率 (|prior-1|>=0.10): v1={a1:.3f}  v2={a2:.3f}")

        # 重み付き相関
        def wcorr(x, y, w):
            x_m = np.average(x, weights=w); y_m = np.average(y, weights=w)
            cov = np.average((x-x_m)*(y-y_m), weights=w)
            vx = np.average((x-x_m)**2, weights=w); vy = np.average((y-y_m)**2, weights=w)
            return cov / np.sqrt(vx*vy) if vx > 0 and vy > 0 else float("nan")
        w = cmp["n_cond"].values
        wc1 = wcorr(cmp["prior_lift_v1"].values, cmp["actual_lift"].values, w)
        wc2 = wcorr(cmp["prior_lift_v2"].values, cmp["actual_lift"].values, w)
        print(f"  重み付き相関:  v1={wc1:+.4f}  v2={wc2:+.4f}")

    print(f"\n{_t(t0)} done")


if __name__ == "__main__":
    main()
