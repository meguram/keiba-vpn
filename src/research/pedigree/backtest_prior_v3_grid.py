"""v3: ハイパーパラメータ (layer 重み, pseudo_n, shrinkage) のグリッド検索。

v2 の発見:
  - n_layers=2 (corr 0.324) > n_layers=3 (corr 0.221)
  - pair_gxg が混ざる時に精度低下
  - ダート (corr 0.143) が芝より弱い

v3 グリッド: pair_gxg 重みを下げる + shrinkage を試行
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
IDX = ROOT / "data/research/pedigree_race_index"
ART = ROOT / "data/research/bloodline_meta_cluster"

DIST_BINS = [0, 1400, 1800, 2200, 9999]
DIST_LABELS = ["短距離","マイル","中距離","長距離"]
STEEP_VENUES = ("中山","阪神","中京")
COND_COLS = (
    [f"win_v_{v}" for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]]
    + [f"win_s_{s}" for s in ["芝","ダート"]]
    + [f"win_d_{d}" for d in DIST_LABELS]
    + ["win_steep","win_flat","win_heavy"]
)


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

    by_horse = race.groupby("horse_id")["win"].agg(["count","sum"]).reset_index()
    by_horse.columns = ["horse_id","n_total","win_total"]
    by_horse = by_horse[by_horse["n_total"] >= 8].copy()
    by_horse["rate_total"] = by_horse["win_total"] / by_horse["n_total"]
    target_set = set(by_horse["horse_id"])
    rec = race[race["horse_id"].isin(target_set)].copy()

    pi_idx = {(r["sire_id"], r["bms_id"]): r for _, r in pair_indiv.iterrows()}
    pg_idx = {(r["sire_id"], r["bms_main"]): r for _, r in pair_group.iterrows()}
    gxg_idx = {(r["sire_main"], r["bms_main"]): r for _, r in pair_gxg.iterrows()}

    role_lift_dict = {}
    role_n_dict = {}
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

    # ── 各馬の各 layer 生 lift をキャッシュ ──
    print(f"{_t(t0)} cache per-horse layer lifts")
    horse_layers: dict[str, dict[str, dict]] = {}
    for hid in by_horse["horse_id"]:
        info = bms_map.get(hid)
        if not info: continue
        sid = info["sire_id"]; bid = info["bms_id"]
        sg = info["sire_main"]; mg = info["bms_main"]
        layers = {}
        if (sid, bid) in pi_idx:
            row = pi_idx[(sid, bid)]
            layers["pair_indiv"] = {"lift": _row_lift_dict(row), "n": int(row.get("n_records", 0))}
        if (sid, mg) in pg_idx:
            row = pg_idx[(sid, mg)]
            layers["pair_group"] = {"lift": _row_lift_dict(row), "n": int(row.get("n_records", 0))}
        rb_lift, rb_n = _role_blend_lift(sid, bid)
        if rb_lift:
            layers["role_blend"] = {"lift": rb_lift, "n": rb_n}
        if (sg, mg) in gxg_idx:
            row = gxg_idx[(sg, mg)]
            layers["pair_gxg"] = {"lift": _row_lift_dict(row), "n": int(row.get("n_records", 0))}
        if layers:
            horse_layers[hid] = layers
    print(f"{_t(t0)} horses with at least 1 layer: {len(horse_layers):,}")

    # ── actual table ──
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
    act_df = act_df[(act_df["rate_total"] > 0) & (act_df["n_cond"] >= 3) & act_df["actual_lift"].notna()].copy()
    actual_idx = {(r["horse_id"], r["cond"]): (r["actual_lift"], r["n_cond"]) for _, r in act_df.iterrows()}

    # ── prior 構築 + 評価 (パラメータ可変) ──
    def evaluate(base_w: dict, pseudo_n: dict, shrink: float) -> dict:
        prior_rows = []
        for hid, layers in horse_layers.items():
            comps = {}
            for k, info in layers.items():
                w = base_w.get(k, 0.0) * info["n"] / (info["n"] + pseudo_n.get(k, 100))
                if w > 0:
                    comps[k] = (info["lift"], w)
            if not comps: continue
            w_sum = sum(w for _, w in comps.values())
            if w_sum <= 0: continue
            all_conds = set().union(*[lift.keys() for lift, _ in comps.values()])
            for cond in all_conds:
                num = 0.0; den = 0.0
                for lift, w in comps.values():
                    if cond in lift:
                        num += w * lift[cond]; den += w
                if den > 0:
                    raw = num / den
                    shrunk = 1.0 + (raw - 1.0) * shrink
                    prior_rows.append((hid, cond, shrunk))
        # 評価
        priors = pd.DataFrame(prior_rows, columns=["horse_id","cond","prior_lift"])
        # join actual
        act = act_df[["horse_id","cond","actual_lift","n_cond"]]
        d = priors.merge(act, on=["horse_id","cond"], how="inner")
        if len(d) == 0: return {}
        # 全体相関 (重み付き)
        x = d["prior_lift"].values; y = d["actual_lift"].values; w = d["n_cond"].values
        x_m = np.average(x, weights=w); y_m = np.average(y, weights=w)
        cov = np.average((x-x_m)*(y-y_m), weights=w)
        vx = np.average((x-x_m)**2, weights=w); vy = np.average((y-y_m)**2, weights=w)
        wcorr = cov / np.sqrt(vx*vy) if vx > 0 and vy > 0 else 0
        # 高低差
        h = d[d["prior_lift"]>=1.3]
        l = d[d["prior_lift"]<=0.85]
        delta = (h["actual_lift"].mean() if len(h) else 0) - (l["actual_lift"].mean() if len(l) else 0)
        # 方向性一致率
        sig = d[(d["prior_lift"]-1).abs() >= 0.10]
        agree = (np.sign(sig["prior_lift"]-1) == np.sign(sig["actual_lift"]-1)).mean() if len(sig) else 0
        # ダート専用 corr
        dt = d[d["cond"]=="win_s_ダート"]
        dt_corr = dt["prior_lift"].corr(dt["actual_lift"]) if len(dt) > 100 else 0
        # 重不良 専用
        hv = d[d["cond"]=="win_heavy"]
        hv_corr = hv["prior_lift"].corr(hv["actual_lift"]) if len(hv) > 100 else 0
        return {
            "n_pairs": len(d), "wcorr": wcorr, "delta_high_low": delta, "agree": agree,
            "dt_corr": dt_corr, "hv_corr": hv_corr,
            "n_high": len(h), "n_low": len(l),
        }

    # ── ベースライン (v2) ──
    print(f"\n{_t(t0)} ========= [grid 検索] =========")
    print(f"  {'config':<60}{'wcorr':>8}{'Δhigh-low':>12}{'agree':>8}{'dt_corr':>9}{'hv_corr':>9}{'n_high':>9}{'n_low':>9}")

    configs = [
        # (label, base_w, pseudo_n, shrink)
        ("v2 baseline", {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.10},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
        ("v3a: pair_gxg=0.05", {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.05},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
        ("v3b: pair_gxg=0.0 (除外)", {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
        ("v3c: role_blend up=0.40", {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.40,"pair_gxg":0.05},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
        ("v3d: pair_indiv up=0.60", {"pair_indiv":0.60,"pair_group":0.25,"role_blend":0.30,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
        ("v3e: shrink 0.80",       {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.80),
        ("v3f: shrink 0.90",       {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.90),
        ("v3g: shrink 1.00 無し",  {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 1.00),
        ("v3h: pseudo_n 半減",     {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.0},
         {"pair_indiv":40,"pair_group":100,"role_blend":50,"pair_gxg":500}, 0.85),
        ("v3i: pseudo_n 倍",       {"pair_indiv":0.50,"pair_group":0.30,"role_blend":0.30,"pair_gxg":0.0},
         {"pair_indiv":160,"pair_group":400,"role_blend":200,"pair_gxg":500}, 0.85),
        ("v3j: pair_indiv up + role up", {"pair_indiv":0.60,"pair_group":0.20,"role_blend":0.40,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
        ("v3k: pure indiv+role",   {"pair_indiv":0.50,"pair_group":0.0,"role_blend":0.50,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
        ("v3l: pure indiv+group",  {"pair_indiv":0.50,"pair_group":0.50,"role_blend":0.0,"pair_gxg":0.0},
         {"pair_indiv":80,"pair_group":200,"role_blend":100,"pair_gxg":500}, 0.85),
    ]
    for label, bw, pn, sh in configs:
        r = evaluate(bw, pn, sh)
        if not r:
            print(f"  {label:<60}  -- empty --")
            continue
        print(f"  {label:<60}{r['wcorr']:>+8.4f}{r['delta_high_low']:>+12.3f}{r['agree']:>8.3f}"
              f"{r['dt_corr']:>+9.3f}{r['hv_corr']:>+9.3f}{r['n_high']:>9,}{r['n_low']:>9,}")

    print(f"\n{_t(t0)} done")


if __name__ == "__main__":
    main()
