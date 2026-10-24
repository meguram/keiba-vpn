"""血統 prior の予測力をバックテスト。

設計:
  1. 出走数 >= 8 の馬を全て対象 (~数千頭)
  2. 各馬について 統合 prior (階層FB, 但し本馬を含む集団=ほぼ leakage 無視可) を構築
  3. 各馬の各条件 (場所/距離/路面など) で実勝率と prior を照合
  4. キャリブレーション・layer 別精度・条件軸別予測力を集計

評価指標:
  (a) calibration: prior_lift bin ごとに actual_lift 平均が単調か
  (b) signal-to-noise: 高 prior と低 prior の actual 差
  (c) layer 別精度: pair_indiv / pair_group / role_blend / pair_gxg
  (d) 条件軸別: 阪神 / 東京 / 長距離 / 重 等で個別に
  (e) AUROC 風: prior が高い (>=1.2) 群と低い (<=0.8) 群で勝率差

leakage 対策:
  pair_group は集団規模 n_records>=50 を採用、本馬の影響は小さいため近似で OK。
  詳細検証時は本馬除外も実装可能。
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
GLOBAL_WIN = 0.139
EB_PRIOR_N = 30


def _t(t0): return f"[{time.time()-t0:5.1f}s]"


def _eb(w, n, prior=GLOBAL_WIN):
    if n < 1: return None
    return (w + EB_PRIOR_N * prior) / (n + EB_PRIOR_N)


def _build_actual_table(race: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """horse_id × cond の (n, win, win_rate, individual_lift) を返す。"""
    print(f"{_t(t0)} build actual table (min_n={min_n})", flush=True)
    by_horse = race.groupby("horse_id")["win"].agg(["count","sum"]).reset_index()
    by_horse.columns = ["horse_id","n_total","win_total"]
    by_horse = by_horse[by_horse["n_total"] >= min_n].copy()
    by_horse["rate_total"] = by_horse["win_total"] / by_horse["n_total"]
    target_set = set(by_horse["horse_id"])
    rec = race[race["horse_id"].isin(target_set)].copy()
    print(f"{_t(t0)} target horses = {len(by_horse):,}, records = {len(rec):,}", flush=True)
    return by_horse, rec


def _cond_filter(rec: pd.DataFrame, c: str) -> pd.Series:
    if c.startswith("win_v_"): return rec["venue"].astype(str) == c.split("_")[-1]
    if c.startswith("win_s_"): return rec["surface"].astype(str) == c.split("_")[-1]
    if c.startswith("win_d_"): return rec["dist_cat"] == c.split("_")[-1]
    if c == "win_steep": return rec["is_steep"]
    if c == "win_flat":  return ~rec["is_steep"]
    if c == "win_heavy": return rec["is_heavy"]
    raise ValueError(c)


def main():
    global t0
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
    print(f"{_t(t0)} pair_indiv={len(pair_indiv):,}, pair_group={len(pair_group):,}, pair_gxg={len(pair_gxg)}")

    # ── 役割別 lift (role_blend 用) も読込 ──
    role_lift = pd.read_parquet(ART / "role_lift_profiles.parquet")
    role_lift["stallion_id"] = role_lift["stallion_id"].astype(str)
    print(f"{_t(t0)} role_lift={len(role_lift):,}")

    # 5gen pedigree (FF, MMF 取得用)
    ped_path = IDX / "horse_pedigree_5gen.parquet"
    if ped_path.exists():
        ped = pd.read_parquet(ped_path, columns=["horse_id", "pedigree"])
        ped["horse_id"] = ped["horse_id"].astype(str)
        ped_map = dict(zip(ped["horse_id"], ped["pedigree"]))
    else:
        ped_map = {}
    print(f"{_t(t0)} pedigree map = {len(ped_map):,}")

    # ── 馬選定 ──
    by_horse, rec = _build_actual_table(race, min_n=8)

    # ── 各馬の prior 構築 ──
    print(f"{_t(t0)} build priors per horse", flush=True)
    pi_idx = {(r["sire_id"], r["bms_id"]): r for _, r in pair_indiv.iterrows()}
    pg_idx = {(r["sire_id"], r["bms_main"]): r for _, r in pair_group.iterrows()}
    gxg_idx = {(r["sire_main"], r["bms_main"]): r for _, r in pair_gxg.iterrows()}

    # 役割別 lift dict
    role_lift_dict: dict[tuple[str, str], dict[str, float]] = {}
    for _, r in role_lift.iterrows():
        d = {c.replace("lift_", ""): r[c] for c in role_lift.columns
             if c.startswith("lift_") and pd.notna(r[c])}
        role_lift_dict[(str(r["stallion_id"]), str(r["role"]))] = d

    bms_map = bms.set_index("horse_id")[["sire_id","bms_id","sire_main","bms_main"]].to_dict("index")

    # role_blend (簡易版): F + MF の重み平均 (MMF/FF は省略 - 親 5gen 解析が高コストのため)
    def _role_blend_lift(sid: str | None, bid: str | None) -> dict[str, float] | None:
        f = role_lift_dict.get((str(sid), "F")) if sid else None
        m = role_lift_dict.get((str(bid), "MF")) if bid else None
        if not f and not m: return None
        keys = set(f or {}).union(set(m or {}))
        out = {}
        for k in keys:
            vals = []
            ws = []
            if f and k in f:
                vals.append(f[k]); ws.append(0.6)
            if m and k in m:
                vals.append(m[k]); ws.append(0.4)
            if vals:
                out[k] = sum(v*w for v,w in zip(vals, ws)) / sum(ws)
        return out

    def _row_lift_dict(row) -> dict[str, float]:
        return {c.replace("lift_", ""): float(row[c]) for c in row.index
                if isinstance(c, str) and c.startswith("lift_") and pd.notna(row[c])}

    # 馬ごとに prior 確定 (layer + lift)
    prior_records = []
    n_done = 0
    for hid in by_horse["horse_id"]:
        n_done += 1
        if n_done % 5000 == 0:
            print(f"  ... {n_done}/{len(by_horse)}", flush=True)
        info = bms_map.get(hid)
        if not info: continue
        sid = info["sire_id"]
        bid = info["bms_id"]
        sg = info["sire_main"]
        mg = info["bms_main"]

        layer = None
        prior_lift: dict[str, float] | None = None
        n_records = None

        # layer1
        if (sid, bid) in pi_idx:
            row = pi_idx[(sid, bid)]
            prior_lift = _row_lift_dict(row)
            n_records = int(row.get("n_records", 0))
            layer = "pair_indiv"
        elif (sid, mg) in pg_idx:
            row = pg_idx[(sid, mg)]
            prior_lift = _row_lift_dict(row)
            n_records = int(row.get("n_records", 0))
            layer = "pair_group"
        else:
            blended = _role_blend_lift(sid, bid)
            if blended:
                prior_lift = blended
                n_records = -1  # サイズ不明
                layer = "role_blend"
            elif (sg, mg) in gxg_idx:
                row = gxg_idx[(sg, mg)]
                prior_lift = _row_lift_dict(row)
                n_records = int(row.get("n_records", 0))
                layer = "pair_gxg"
            else:
                continue

        for c, lift in prior_lift.items():
            prior_records.append({"horse_id": hid, "cond": c, "prior_lift": lift, "layer": layer, "n_pair": n_records})
    prior_df = pd.DataFrame(prior_records)
    print(f"{_t(t0)} prior records = {len(prior_df):,}, layer dist = {dict(prior_df.drop_duplicates('horse_id')['layer'].value_counts())}")

    # ── 各馬の actual lift (条件別) を bulk 計算 ──
    print(f"{_t(t0)} compute actual lift per (horse, cond)", flush=True)
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
    print(f"{_t(t0)} actual records = {len(act_df):,}")

    # ── join ──
    df = prior_df.merge(act_df, on=["horse_id","cond"], how="inner")
    df = df[df["actual_lift"].notna() & df["prior_lift"].notna()].copy()
    # ノイズ除去: 本馬通算勝率 0 や 条件出走 0 は除外
    df = df[(df["rate_total"] > 0) & (df["n_cond"] >= 3)].copy()
    print(f"{_t(t0)} joined rows = {len(df):,}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / "backtest_prior.parquet"
    df.to_parquet(out, index=False)
    print(f"{_t(t0)} saved {out}")

    # ────────────────────────────────────────────────
    # 集計 1: キャリブレーション (prior bin ごとの actual)
    # ────────────────────────────────────────────────
    print(f"\n{_t(t0)} ========= [1] キャリブレーション (全条件統合) =========")
    bins = [0, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 5]
    df["prior_bin"] = pd.cut(df["prior_lift"], bins=bins)
    # 重みつき (n_cond): sum(win_cond) / sum(rate_total * n_cond)
    g = df.groupby("prior_bin", observed=True).agg(
        n_pairs=("actual_lift","count"),
        actual_lift_mean=("actual_lift","mean"),
        win_sum=("win_cond","sum"),
        n_sum=("n_cond","sum"),
        rate_total_w=("rate_total", lambda s: float(np.average(s, weights=df.loc[s.index,"n_cond"]))),
    )
    g["actual_rate_w"] = g["win_sum"] / g["n_sum"]
    g["actual_lift_w"] = g["actual_rate_w"] / g["rate_total_w"]
    print(g[["n_pairs","n_sum","win_sum","actual_rate_w","rate_total_w","actual_lift_w","actual_lift_mean"]])
    print("\n  ▷ prior_bin と actual_lift_w が単調増加なら、prior は予測力あり")

    # ────────────────────────────────────────────────
    # 集計 2: 高 prior vs 低 prior の差分
    # ────────────────────────────────────────────────
    print(f"\n{_t(t0)} ========= [2] 高 prior vs 低 prior の actual 比較 =========")
    high = df[df["prior_lift"] >= 1.2]
    low = df[df["prior_lift"] <= 0.8]
    print(f"  high (prior>=1.2): n_pairs={len(high):,}, actual_lift={high['actual_lift'].mean():.3f}, "
          f"win_rate={high['win_cond'].sum()/high['n_cond'].sum():.3f}")
    print(f"  low  (prior<=0.8): n_pairs={len(low):,}, actual_lift={low['actual_lift'].mean():.3f}, "
          f"win_rate={low['win_cond'].sum()/low['n_cond'].sum():.3f}")

    # ────────────────────────────────────────────────
    # 集計 3: layer 別精度
    # ────────────────────────────────────────────────
    print(f"\n{_t(t0)} ========= [3] layer 別予測力 =========")
    for layer in ["pair_indiv","pair_group","role_blend","pair_gxg"]:
        sub = df[df["layer"] == layer]
        if len(sub) < 100: continue
        corr = sub["prior_lift"].corr(sub["actual_lift"])
        # 重み付きで相関
        w = sub["n_cond"].values
        x = sub["prior_lift"].values
        y = sub["actual_lift"].values
        x_mean = np.average(x, weights=w); y_mean = np.average(y, weights=w)
        cov = np.average((x-x_mean)*(y-y_mean), weights=w)
        var_x = np.average((x-x_mean)**2, weights=w)
        var_y = np.average((y-y_mean)**2, weights=w)
        wcorr = cov / np.sqrt(var_x*var_y) if var_x > 0 and var_y > 0 else float("nan")
        # 高低差
        h = sub[sub["prior_lift"]>=1.2]["actual_lift"].mean()
        l = sub[sub["prior_lift"]<=0.8]["actual_lift"].mean()
        print(f"  {layer:<14}  n={len(sub):>7,}  corr={corr:>+5.3f}  wcorr={wcorr:>+5.3f}  "
              f"actual(高prior)={h:>5.3f}  actual(低prior)={l:>5.3f}  Δ={h-l:>+5.3f}")

    # ────────────────────────────────────────────────
    # 集計 4: 条件軸別予測力
    # ────────────────────────────────────────────────
    print(f"\n{_t(t0)} ========= [4] 条件別予測力 =========")
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
        diff = (h or 0) - (l or 0)
        marker = " ★" if abs(corr) > 0.05 and (h-l) > 0.05 else ""
        print(f"  {c:<22}{len(sub):>7,}{corr:>+7.3f}{h:>9.3f}{l:>9.3f}{diff:>+7.3f}{marker}")

    # ────────────────────────────────────────────────
    # 集計 5: prior の方向性正解率
    # ────────────────────────────────────────────────
    print(f"\n{_t(t0)} ========= [5] prior 方向性 と actual 方向性の一致率 =========")
    df["prior_sign"] = np.sign(df["prior_lift"] - 1.0)
    df["actual_sign"] = np.sign(df["actual_lift"] - 1.0)
    # |prior - 1| >= 0.10 のもののみ評価
    sig = df[(df["prior_lift"] - 1.0).abs() >= 0.10]
    agree = (sig["prior_sign"] == sig["actual_sign"]).mean()
    print(f"  全 (|prior-1|>=0.10): n={len(sig):,}, 一致率={agree:.3f}")
    for layer in ["pair_indiv","pair_group","role_blend"]:
        s = sig[sig["layer"]==layer]
        if len(s) > 100:
            a = (s["prior_sign"] == s["actual_sign"]).mean()
            print(f"  {layer}: n={len(s):,}, 一致率={a:.3f}")

    print(f"\n{_t(t0)} done")


if __name__ == "__main__":
    main()
