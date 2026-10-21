"""tp-62: 月次過半・年間 100% 超の馬券戦略探索（ホストで実行。重い処理は heavy_run.sh 経由）。

python -m src.scripts.data.tp62_run wf   [--max-races N]            # 実データのウォークフォワード（検証年 2023〜2026）
python -m src.scripts.data.tp62_run null --reps 40 --seed 1         # 帰無（市場が正しい）で同じ手順を繰り返す
成功基準（事前に固定）: A 年間回収率 > 100%、B 回収率 > 100% の月が過半（その年の月数の半分超）。購入が無い月は不成立。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.scripts.data import tp62_lib as L

TEST_YEARS = (2023, 2024, 2025, 2026)
MODES = ("A", "B", "C")


def make_grid(max_races: int = 0, overlay: bool = False):
    t0 = time.time()
    d = L.load_tickets(max_races=max_races, overlay=overlay)
    races = L.build_races(d)
    meta, pr, pw = L.build_rules(d, overlay=overlay)
    print(f"rules={len(meta)} pairs={len(pr):,} rows={len(d):,} races={len(races.race_id):,} ({time.time() - t0:.0f}s)", flush=True)
    grid = L.Grid(d, races, meta, pr, pw)
    print(f"grid built ({time.time() - t0:.0f}s) strategies={2 * len(meta) * grid.F:,}", flush=True)
    return d, grid


def run_folds(grid: L.Grid) -> dict:
    out = {}
    for ty in TEST_YEARS:
        if ty not in grid.years:
            continue
        train = [y for y in grid.years if y < ty and y >= 2022]
        res = {"train_years": train, "insample": L.insample_pass_count(grid, ty)}
        for mode in MODES:
            sel = L.select(grid, train, mode)
            if sel is None:
                res[mode] = None
                continue
            u, f, info = sel
            res[mode] = {"strategy": grid.describe(u, f), **info, "test": L.test_result(grid, u, f, ty)}
        out[str(ty)] = res
    return out


def cmd_wf(a) -> None:
    d, grid = make_grid(a.max_races, a.overlay)
    out = run_folds(grid)
    L.OUT.mkdir(parents=True, exist_ok=True)
    f = L.OUT / (("wf_small" if a.max_races else "wf") + ("_ov" if a.overlay else "") + ".json")
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for ty, res in out.items():
        print("==", ty, "訓練", res["train_years"], "事後(年内)で基準を満たした戦略数", res["insample"])
        for mode in MODES:
            r = res[mode]
            if not r:
                print(" ", mode, "選べる戦略なし")
                continue
            t = r["test"]
            print(f"  {mode} {r['strategy']}\n     訓練 ROI={r['train_roi']:.3f} 下限={r['train_zlow']:.3f} 購入R={r['train_bets_races']:.0f} → "
                  f"検証 ROI={t['roi']:.3f} 購入R={t['bet_races']:.0f} 勝ち月={t['win_months']}/{t['months']} A={t['A']} B={t['B']}")


def cmd_null(a) -> None:
    d, grid = make_grid(a.max_races, a.overlay)
    sim = L.NullSim(d, lam=a.lam)
    c = sim.calibrate(n=20, seed=a.seed)
    print("null calibration c_bt", np.round(c, 3), "raw sim mean ROI", np.round(sim.raw_mean, 3), flush=True)
    rng = np.random.default_rng(a.seed)
    recs = []
    t0 = time.time()
    for r in range(a.reps):
        grid.set_returns(sim.draw_ret(rng))
        out = run_folds(grid)
        recs.append(out)
        if r % 5 == 0:
            print(f"rep {r} ({time.time() - t0:.0f}s)", flush=True)
        L.OUT.mkdir(parents=True, exist_ok=True)
        (L.OUT / f"null{'_ov' if a.overlay else ''}_seed{a.seed}.json").write_text(json.dumps({"c": c.tolist(), "reps": recs}, ensure_ascii=False), encoding="utf-8")


def cmd_power(a) -> None:
    d, grid = make_grid(a.max_races, a.overlay)
    sim = L.NullSim(d)
    sim.calibrate(n=20, seed=a.seed)
    out = {}
    for ty in TEST_YEARS:
        train = [y for y in grid.years if y < ty and y >= 2022]
        for mode in ("A", "B"):
            sel = L.select(grid, train, mode)
            if sel is None:
                continue
            u, f, info = sel
            pc = L.power_curve(grid, sim, u, f, ty, reps=a.reps, seed=a.seed)
            out[f"{ty}|{mode}"] = {"strategy": grid.describe(u, f), "test": L.test_result(grid, u, f, ty), "power": pc}
            print(ty, mode, grid.describe(u, f), {k: round(v["AB"], 3) for k, v in pc.items()}, flush=True)
    L.OUT.mkdir(parents=True, exist_ok=True)
    (L.OUT / ("power_ov.json" if a.overlay else "power.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("wf")
    w.add_argument("--max-races", type=int, default=0)
    w.add_argument("--overlay", action="store_true")
    n = sp.add_parser("null")
    n.add_argument("--reps", type=int, default=40)
    n.add_argument("--seed", type=int, default=1)
    n.add_argument("--lam", type=float, default=1.0)
    n.add_argument("--max-races", type=int, default=0)
    n.add_argument("--overlay", action="store_true")
    p = sp.add_parser("power")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-races", type=int, default=0)
    p.add_argument("--overlay", action="store_true")
    a = ap.parse_args()
    {"wf": cmd_wf, "null": cmd_null, "power": cmd_power}[a.cmd](a)


if __name__ == "__main__":
    main()
