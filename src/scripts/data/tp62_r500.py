"""tp-62 (2): 『的中率の下限 × レース回収率 500%』で買い目を組むルール族と、そのウォークフォワード評価。

python -m src.scripts.data.tp62_r500 wf    [--max-races N]               # 固定水準（レース的中率 20% 以上・的中時 500% 以上）で検証年 2023〜2026
python -m src.scripts.data.tp62_r500 sens  [--max-races N]               # 水準（的中率 15/20/25/30%、倍率 300/500/700%）の感度（参考。選択バイアスあり）
python -m src.scripts.data.tp62_r500 null  --reps 40 --seed 1            # 帰無（市場が正しい）で同じ手順を繰り返す
python -m src.scripts.data.tp62_r500 census                               # 水準を満たすレースの頻度と、較正後の確率 vs 実績

組み立て方: 各レース・各券種で、較正後の的中確率 p_cal が券種別の下限（PFLOORS）以上、期待値（p_cal × オッズ）が閾値以上の買い目を、
EV の降順（または p_cal の降順）に並べ、上位 K 点（K ≤ Kmax）を等額で買う。K 点買うとき『的中した買い目の払戻が、そのレースの総賭け金の M 倍以上』
（= すべての買い目のオッズ ≥ M × K）かつ『いずれかが的中する確率（較正後）が H 以上』を満たす K のうち、期待回収率が最大のものを選ぶ。満たす K が無いレースは買わない。
M × H = 1（損益分岐）なので、これは『期待値 ≥ 1 かつ的中率 ≥ H』の制約と同じ向きの条件になる。
較正後の p_cal は、検証年より前の年のチケット表で券種ごとに p_mix → 的中 を単調回帰して作る（検証年の結果は使わない）。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.scripts.data import tp62_lib as L

H_MAIN, M_MAIN = 0.20, 5.0
PFLOORS = (0.0, 0.03, 0.05, 0.08, 0.12, 0.20)
KMAXS = (1, 2, 3, 5)
ORDERS = ("ev", "p")
THS = (1.0, 1.2)
EXCLUSIVE = {0, 2, 4, 5}      # 同時に 2 つは的中しない券種（単勝・馬連・馬単・3連単）。複勝・ワイドは重なりうる
SLOTS = 5
TEST_YEARS = (2023, 2024, 2025, 2026)
NMIN = 20              # 各訓練年の最低購入レース数（この族は購入が少ないので EV 族の 40 より下げる）


def calibrate_hit(d: pd.DataFrame, train_years) -> np.ndarray:
    """券種ごとの p_mix → 的中 の単調回帰。全行の p_cal を返す。
    訓練年の行は『自分の入っていない半分のレース（レース番号の偶奇）』で当てはめた回帰で較正する（自分の結果で自分を較正すると、
    EV≥1 の制約が実力以上に満たされて見えるため）。検証年・それ以外の年の行は、訓練年すべてで当てはめた回帰を使う（検証年の結果は使わない）。"""
    from sklearn.isotonic import IsotonicRegression

    p0 = d["p_mix"].to_numpy(np.float64)
    pc = p0.copy()
    hit = (d["ret"].to_numpy() > 0).astype(float)
    yr = d["year"].to_numpy()
    par = d["ridx"].to_numpy() % 2
    for bt in sorted(d["bt"].unique()):
        m = d["bt"].to_numpy() == bt
        tr = m & np.isin(yr, list(train_years))
        if tr.sum() < 2000:
            continue
        full = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p0[tr], hit[tr])
        pc[m] = full.predict(p0[m])
        for h in (0, 1):
            fit_rows = tr & (par == h)
            if fit_rows.sum() < 1000:
                continue
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p0[fit_rows], hit[fit_rows])
            tgt = tr & (par != h)
            pc[tgt] = ir.predict(p0[tgt])
    return pc.astype(np.float32)


def build_rules_r500(d: pd.DataFrame, pcal: np.ndarray, H: float = H_MAIN, M: float = M_MAIN):
    """戻り値: meta, pair_rule(int32), pair_row(int32), pu[U, Rn]（選んだ買い目の的中確率の和/併合、較正後）"""
    ridx = d["ridx"].to_numpy()
    Rn = int(ridx.max()) + 1
    odds = d["odds"].to_numpy(np.float64)
    meta, pr, pw, pus = [], [], [], []
    rid = 0
    for bt in sorted(d["bt"].unique()):
        idx = np.flatnonzero(d["bt"].to_numpy() == bt)
        p_i = pcal[idx].astype(np.float64)
        o_i = odds[idx]
        ev_i = p_i * o_i
        for order in ORDERS:
            score = ev_i if order == "ev" else p_i
            ordr = np.lexsort((-score, ridx[idx]))
            rows = idx[ordr]
            ps, os_, evs, r_s = p_i[ordr], o_i[ordr], ev_i[ordr], ridx[idx][ordr]
            chg = np.flatnonzero(np.diff(r_s, prepend=r_s[0] - 1))
            cnts = np.diff(np.append(chg, len(r_s)))
            seg_start = np.repeat(chg, cnts)
            for th in THS:
                for pf in PFLOORS:
                    ok = (ps >= pf) & (evs >= th)
                    cs = np.cumsum(ok)
                    base = np.where(seg_start > 0, cs[np.maximum(seg_start - 1, 0)], 0)
                    rank = cs - base
                    take = ok & (rank <= SLOTS)
                    P = np.full((Rn, SLOTS), np.nan)
                    O = np.full((Rn, SLOTS), np.nan)
                    RW = np.full((Rn, SLOTS), -1, np.int64)
                    P[r_s[take], rank[take] - 1] = ps[take]
                    O[r_s[take], rank[take] - 1] = os_[take]
                    RW[r_s[take], rank[take] - 1] = rows[take]
                    best_ev = np.full((len(KMAXS), Rn), -np.inf)
                    best_k = np.zeros((len(KMAXS), Rn), np.int64)
                    best_pu = np.zeros((len(KMAXS), Rn))
                    for k in range(1, SLOTS + 1):
                        have = ~np.isnan(P[:, k - 1])
                        Pk, Ok = P[:, :k], O[:, :k]
                        with np.errstate(invalid="ignore"):
                            feas = have & (np.nanmin(np.where(np.isnan(Ok), np.inf, Ok), axis=1) >= M * k)
                        pu = np.minimum(1.0, np.nansum(Pk, 1)) if bt in EXCLUSIVE else 1.0 - np.prod(1.0 - np.nan_to_num(Pk, nan=0.0), 1)
                        evk = np.nansum(Pk * Ok, 1) / k
                        good = feas & (pu >= H)
                        for ki, km in enumerate(KMAXS):
                            if k > km:
                                continue
                            upd = good & (evk > best_ev[ki])
                            best_ev[ki, upd] = evk[upd]
                            best_k[ki, upd] = k
                            best_pu[ki, upd] = pu[upd]
                    for ki, km in enumerate(KMAXS):
                        kk = best_k[ki]
                        sel_r = np.flatnonzero(kk > 0)
                        rows_sel = []
                        for s in range(SLOTS):
                            m = kk[sel_r] > s
                            rows_sel.append(RW[sel_r[m], s])
                        rows_sel = np.concatenate(rows_sel) if rows_sel else np.array([], np.int64)
                        meta.append({"rule": rid, "bt": int(bt), "score": "p_cal", "ov": "none", "order": order, "theta": th, "pfloor": pf, "kmax": km,
                                     "theta_": th, "band": 0, "k": km})
                        pr.append(np.full(len(rows_sel), rid, np.int32))
                        pw.append(rows_sel.astype(np.int32))
                        pus.append(best_pu[ki].astype(np.float32))
                        rid += 1
    return pd.DataFrame(meta), np.concatenate(pr), np.concatenate(pw), np.stack(pus)


def describe(grid: L.Grid, u: int, f: int) -> str:
    i = u % grid.U
    m = grid.meta.iloc[i]
    mode = "定額" if u < grid.U else "レース均等"
    return (f"{L.BT_NAMES[int(m['bt'])]}|p_cal>={m['pfloor']:g}|EV_cal>={m['theta']:g}|{'EV順' if m['order'] == 'ev' else '確率順'}|"
            f"上位{int(m['kmax'])}点まで|{mode}|{grid.races.mask_names[f]}")


def make_fold_grid(d: pd.DataFrame, races, train_years, H=H_MAIN, M=M_MAIN):
    pcal = calibrate_hit(d, train_years)
    meta, pr, pw, pu = build_rules_r500(d, pcal, H, M)
    grid = L.Grid(d, races, meta, pr, pw)
    grid.describe = lambda u, f, _g=grid: describe(_g, u, f)
    grid.pu = pu
    grid.pcal = pcal
    return grid


def detail(grid: L.Grid, u: int, f: int, year: int) -> dict:
    """検証年の、選んだ戦略の買い目の実態: 購入レース数・1 レースの平均点数・レース的中率（実績 vs 較正後予測）・的中時の払戻倍率の中央値。"""
    ur = u % grid.U
    yr = grid.races.year == year
    mk = grid.races.masks[f]
    cnt = grid.cnt[ur]
    bet = (cnt > 0) & yr & mk
    if not bet.any():
        return {"bet_races": 0}
    rs = np.bincount(grid.key[grid.key // grid.Rn == ur] % grid.Rn, weights=grid.d["ret"].to_numpy()[grid.pair_row[grid.key // grid.Rn == ur]], minlength=grid.Rn)
    hit = (rs > 0) & bet
    stake_race = cnt[bet]
    mult = rs[hit] / np.maximum(cnt[hit], 1.0)
    return {"bet_races": int(bet.sum()), "mean_points": float(stake_race.mean()), "race_hit_rate": float(hit.sum() / bet.sum()),
            "pred_race_hit": float(grid.pu[ur][bet].mean()), "hit_multiple_median": float(np.median(mult)) if len(mult) else None,
            "hit_multiple_ge5": float((mult >= 5.0).mean()) if len(mult) else None}


def run_fold_all(d, races, H=H_MAIN, M=M_MAIN, modes=("A", "B", "C"), with_detail=True, grids=None):
    out = {}
    use = grids or {}
    for ty in TEST_YEARS:
        if ty not in sorted(set(races.year.tolist())):
            continue
        train = [y for y in sorted(set(races.year.tolist())) if 2022 <= y < ty]
        g = use.get(ty) or make_fold_grid(d, races, train, H, M)
        res = {"train_years": train, "n_rules": int(g.U), "insample": L.insample_pass_count(g, ty, NMIN)}
        for mode in modes:
            sel = L.select(g, train, mode, nmin=NMIN)
            if sel is None:
                res[mode] = None
                continue
            u, f, info = sel
            r = {"strategy": g.describe(u, f), **info, "test": L.test_result(g, u, f, ty)}
            if with_detail:
                r["detail"] = detail(g, u, f, ty)
            res[mode] = r
        out[str(ty)] = res
    return out


def cmd_wf(a):
    t0 = time.time()
    d = L.load_tickets(max_races=a.max_races)
    races = L.build_races(d)
    out = run_fold_all(d, races)
    L.OUT.mkdir(parents=True, exist_ok=True)
    (L.OUT / ("r500_wf_small.json" if a.max_races else "r500_wf.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for ty, res in out.items():
        print("==", ty, "訓練", res["train_years"], "事後(年内)で基準を満たした戦略数", res["insample"], f"({time.time() - t0:.0f}s)")
        for mode in ("A", "B", "C"):
            r = res[mode]
            if not r:
                print(" ", mode, "選べる戦略なし")
                continue
            t = r["test"]
            print(f"  {mode} {r['strategy']}\n     訓練 ROI={r['train_roi']:.3f} 下限={r['train_zlow']:.3f} 購入R={r['train_bets_races']:.0f} → 検証 ROI={t['roi']:.3f} "
                  f"購入R={t['bet_races']:.0f} 勝ち月={t['win_months']}/{t['months']}(購入月{t['bought_months']}) A={t['A']} B={t['B']} 実績レース的中={r['detail'].get('race_hit_rate')} 予測={r['detail'].get('pred_race_hit')}")


def cmd_sens(a):
    d = L.load_tickets(max_races=a.max_races)
    races = L.build_races(d)
    out = {}
    for H in (0.15, 0.20, 0.25, 0.30):
        for M in (3.0, 5.0, 7.0):
            out[f"H{H:g}_M{M:g}"] = run_fold_all(d, races, H, M, modes=("A",), with_detail=True)
            print(f"H={H} M={M}", {ty: (r["A"]["test"]["roi"] if r["A"] else None) for ty, r in out[f"H{H:g}_M{M:g}"].items()}, flush=True)
    L.OUT.mkdir(parents=True, exist_ok=True)
    (L.OUT / ("r500_sens_small.json" if a.max_races else "r500_sens.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_census(a):
    """水準 (H=20%, M=500%) を満たすレースの頻度。券種別・年別。『満たすレースは年に何R あるか』『較正後の予測と実績の的中率』。"""
    d = L.load_tickets(max_races=a.max_races)
    races = L.build_races(d)
    out = {}
    for ty in TEST_YEARS:
        years = sorted(set(races.year.tolist()))
        if ty not in years:
            continue
        train = [y for y in years if 2022 <= y < ty]
        g = make_fold_grid(d, races, train)
        m = g.meta
        rows = []
        for bt in sorted(d["bt"].unique()):
            for th in THS:
                sub = m[(m.bt == bt) & (m.order == "ev") & (m.theta == th) & (m.pfloor == 0.0) & (m.kmax == 5)]
                if len(sub) == 0:
                    continue
                u = int(sub.index[0])
                f_all = g.races.mask_names.index("all")
                dt = detail(g, u, f_all, ty)
                tr = L.test_result(g, u, f_all, ty) if dt.get("bet_races") else {}
                n_races_year = int((g.races.year == ty).sum())
                rows.append({"bt": L.BT_NAMES[bt], "theta": th, "n_races_in_year": n_races_year, **dt,
                             "roi": tr.get("roi"), "win_months": tr.get("win_months"), "months": tr.get("months"), "bought_months": tr.get("bought_months"),
                             "A": tr.get("A"), "AB": tr.get("AB"), "monthly_roi": tr.get("monthly_roi")})
        out[str(ty)] = rows
        print(ty, rows, flush=True)
    L.OUT.mkdir(parents=True, exist_ok=True)
    (L.OUT / "r500_census.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_power(a):
    """候補（各券種の『水準だけ・絞り込みなし』のルール）が、真の期待回収率 t のときに、その年に基準 A・B を満たす確率と、
    t = 105% を 90% の確からしさで 100% 超と言うために要る購入レース数・年数（実データのレース別払戻の分散から）。"""
    d = L.load_tickets(max_races=a.max_races)
    races = L.build_races(d)
    sim = L.NullSim(d)
    years = sorted(set(races.year.tolist()))
    out = {}
    ty = 2025
    g = make_fold_grid(d, races, [y for y in years if 2022 <= y < ty])
    m = g.meta
    f_all = g.races.mask_names.index("all")
    for bt in (2, 3, 0):
        sub = m[(m.bt == bt) & (m.order == "ev") & (m.theta == 1.0) & (m.pfloor == 0.0) & (m.kmax == 5)]
        u = int(sub.index[0])
        pc = L.power_curve(g, sim, u, f_all, ty, reps=a.reps, seed=a.seed)
        # 全年のレース別収益から、1 レースあたりの回収倍率（レース均等）の平均と標準偏差
        ur = u % g.U
        rs = np.bincount(g.key[g.key // g.Rn == ur] % g.Rn, weights=d["ret"].to_numpy()[g.pair_row[g.key // g.Rn == ur]], minlength=g.Rn)
        cnt = g.cnt[ur]
        bet = cnt > 0
        per_race = rs[bet] / cnt[bet]
        mu, sd = float(per_race.mean()), float(per_race.std(ddof=1))
        n_bet_year = float(np.mean([(g.cnt[ur][(g.races.year == y)] > 0).sum() for y in years if y >= 2023]))
        need = {}
        for t in (1.05, 1.10, 1.20):
            n_need = (L.Z90 * sd / (t - 1.0)) ** 2
            need[str(t)] = {"races": n_need, "years": n_need / max(n_bet_year, 1.0)}
        out[L.BT_NAMES[bt]] = {"rule": g.describe(u, f_all), "power_2025": pc, "per_race_mean": mu, "per_race_sd": sd, "bet_races_per_year": n_bet_year, "need": need}
        print(L.BT_NAMES[bt], {k: round(v["AB"], 3) for k, v in pc.items()}, round(mu, 3), round(sd, 2), need, flush=True)
    L.OUT.mkdir(parents=True, exist_ok=True)
    (L.OUT / "r500_power.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_freeze(a):
    """候補（馬連・EV 較正後 ≥ 1.0・レース的中率 20% 以上・払戻 500% 以上・5 点まで・絞り込みなし）の較正の折れ線を 2022〜2025 年で当てはめ、設定として凍結する。
    事後に見つけた候補なので採用ではなく、2026-10-22 以降の新しいレースで前向きに追跡するための仕様。"""
    from sklearn.isotonic import IsotonicRegression

    d = L.load_tickets()
    out = {"name": "tp62_umaren_h20_m500", "frozen_at": "2026-10-21", "fit_years": [2022, 2023, 2024, 2025],
           "rule": {"bet": "umaren", "p_source": "p_mix（V4 の確率 + 単勝の Benter 混合 + 割引 Harville）", "calibration": "p_mix → 的中 の単調回帰（下の knots）",
                    "ev_cal_min": 1.0, "race_hit_min": H_MAIN, "payoff_multiple_min": M_MAIN, "max_points": 5, "order": "ev_cal 降順", "stake": "等額",
                    "odds": "確定オッズ（直前オッズでの再検証が必要）", "race_filter": "なし"},
           "knots": {}}
    yr = d["year"].to_numpy()
    for bt in (2,):
        m = (d["bt"].to_numpy() == bt) & np.isin(yr, out["fit_years"])
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(d["p_mix"].to_numpy(np.float64)[m], (d["ret"].to_numpy()[m] > 0).astype(float))
        out["knots"][L.BT_NAMES[bt]] = {"x": [float(x) for x in ir.X_thresholds_], "y": [float(y) for y in ir.y_thresholds_]}
    L.OUT.mkdir(parents=True, exist_ok=True)
    f = L.OUT / "tp62_frozen_rule.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", f, {k: len(v["x"]) for k, v in out["knots"].items()})


def cmd_null(a):
    d = L.load_tickets(max_races=a.max_races)
    races = L.build_races(d)
    years = sorted(set(races.year.tolist()))
    grids = {ty: make_fold_grid(d, races, [y for y in years if 2022 <= y < ty], a.H, a.M) for ty in TEST_YEARS if ty in years}
    sim = L.NullSim(d, lam=a.lam)
    c = sim.calibrate(n=20, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    recs, t0 = [], time.time()
    for r in range(a.reps):
        ret = sim.draw_ret(rng)
        for g in grids.values():
            g.set_returns(ret)
        recs.append(run_fold_all(d, races, a.H, a.M, with_detail=False, grids=grids))
        if r % 5 == 0:
            print(f"rep {r} ({time.time() - t0:.0f}s)", flush=True)
        L.OUT.mkdir(parents=True, exist_ok=True)
        (L.OUT / (f"r500_null_seed{a.seed}.json" if (a.H, a.M) == (H_MAIN, M_MAIN) else f"r500_null_H{a.H:g}_M{a.M:g}_seed{a.seed}.json")).write_text(json.dumps({"c": c.tolist(), "reps": recs}, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("wf", "sens", "census", "null", "freeze", "power"):
        p = sp.add_parser(nm)
        p.add_argument("--max-races", type=int, default=0)
        if nm in ("power",):
            p.add_argument("--reps", type=int, default=300)
            p.add_argument("--seed", type=int, default=1)
        if nm == "null":
            p.add_argument("--reps", type=int, default=40)
            p.add_argument("--seed", type=int, default=1)
            p.add_argument("--lam", type=float, default=1.0)
            p.add_argument("--H", type=float, default=H_MAIN)
            p.add_argument("--M", type=float, default=M_MAIN)
    a = ap.parse_args()
    {"wf": cmd_wf, "sens": cmd_sens, "census": cmd_census, "null": cmd_null, "freeze": cmd_freeze, "power": cmd_power}[a.cmd](a)


if __name__ == "__main__":
    main()
