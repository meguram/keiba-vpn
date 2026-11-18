"""第 2 回の凍結の 2025 年（test）と 2026 年（事前登録の主の群）を合算した、評価の期間の値（判断・選び直しには使わない）。

python -m src.scripts.data.bet_r2_combine

- 元: bet_cycles.OUT_DIR/<tag>/final_test_races_<hash>.parquet と prereg_main_races_<hash>.parquet・prereg_main_b0_races_<hash>.parquet（どちらも T-60 の再現 5 通りの、版 × レースの記録）。
  2026 年の主の群の記録（docs/operations/retrain_trials/bet_2026_r2_main_<hash>.json）が done でなければ止まる（1 回の確かめの前に 2026 年を読まない）。
- 版を並べた記録のまま合算し、件数は版の数で割る（bet_cycles.metrics と同じ数え方）。区間はレースで束ねたブートストラップ（Poisson の重み、1,000 回）。
- 出力: docs/operations/retrain_trials/bet_2025_2026_r2_combined_<hash>.json（bet_cycles_report の「第 2 回の買い方の評価の期間」の合算の行）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "docs/operations/retrain_trials"


def stacked(path: Path) -> tuple[pd.DataFrame, int]:
    d = pd.read_parquet(path, columns=["race_id", "cost", "ret", "ver"])
    return d, int(d["ver"].nunique())


def metrics(d: pd.DataFrame, k: int, n_boot: int = 1000, seed: int = 0) -> dict:
    """版 × レースを並べたまま、bet_cycles.metrics と同じ数え方（件数は版の数で割り、上位を除く件数は版の数を掛ける）。
    区間は、同じレースの版に同じ重みを付けたブートストラップ。"""
    cost, ret = d["cost"].to_numpy(float), d["ret"].to_numpy(float)
    rc = d["race_id"].astype("category").cat.codes.to_numpy()
    nr = rc.max() + 1
    W = np.random.default_rng(seed).poisson(1.0, size=(n_boot, nr)).astype(np.float32)
    cr = np.bincount(rc, weights=cost, minlength=nr)
    rr = np.bincount(rc, weights=ret, minlength=nr)
    bs = (W @ rr) / np.maximum(W @ cr, 1e-12)
    bet = cost > 0
    top = np.sort(ret[bet])[::-1]
    return {"races": int(nr), "bet_races": int(round(bet.sum() / k)), "bet_rate": float(bet.sum() / len(cost)),
            "hit_rate": float((ret[bet] > 0).mean()) if bet.any() else None, "roi": float(ret.sum() / max(cost.sum(), 1e-12)),
            "roi_lo": float(np.quantile(bs, 0.025)), "roi_hi": float(np.quantile(bs, 0.975)),
            "roi_ex_top5": float((ret[bet].sum() - top[:5 * k].sum()) / max(cost[bet].sum(), 1e-12)),
            "avg_return": float(ret.sum() / max(bet.sum(), 1)), "median_return": float(np.median(ret[bet])) if bet.any() else None}


def paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """買い方 a と基準 b の回収率の差（同じレースの集合・同じ重み）。"""
    ids = pd.Index(sorted(set(a["race_id"]) | set(b["race_id"])))
    W = np.random.default_rng(seed).poisson(1.0, size=(n_boot, len(ids))).astype(np.float32)

    def per_race(d):
        g = d.groupby("race_id")[["cost", "ret"]].sum().reindex(ids, fill_value=0.0)
        return g["cost"].to_numpy(float), g["ret"].to_numpy(float)
    ca, ra = per_race(a)
    cb_, rb = per_race(b)
    d = (W @ ra) / np.maximum(W @ ca, 1e-12) - (W @ rb) / np.maximum(W @ cb_, 1e-12)
    return float(ra.sum() / ca.sum() - rb.sum() / cb_.sum()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def b0_2025(body: dict) -> pd.DataFrame:
    """2025 年（test）の基準 B0 の版 × レースの記録（凍結の入力で作り直す。test の記録と同じ期間・同じ再現）。"""
    from src.scripts.data import bet_cycles as bc

    per, _ = bc.prepare_periods({"periods": {"test": body["inputs"]["periods"]["test"]}}, ("test",), cuts=body["qcuts"])
    p = per["test"]
    base = json.loads((bc.CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"][0]
    r = bc.run_period(p, base)
    n0 = len(p.races) // max(p.k, 1)
    return pd.DataFrame({"race_id": [x.race_id for x in p.races], "ver": np.arange(len(p.races)) // n0, "cost": r["cost"], "ret": r["ret"]})


def main() -> None:
    from src.scripts.data import bet_cycles as bc

    body = json.loads((ROOT / "config/bet_cycles/frozen_20261117_r2.json").read_text(encoding="utf-8"))
    h = body["hash"]
    OUT = bc.OUT_DIR / body["tag"]
    mk = T / f"bet_2026_r2_main_{h}.json"
    if not mk.exists() or json.loads(mk.read_text(encoding="utf-8")).get("status") != "done":
        raise SystemExit("2026 年の主の群がまだ流れていません（事前登録の 1 回の確かめの後に合算する）")
    a, ka = stacked(OUT / f"final_test_races_{h}.parquet")
    b, kb = stacked(OUT / f"prereg_main_races_{h}.parquet")
    if ka != kb:
        raise SystemExit(f"版の数が違います（2025 年 {ka}・2026 年 {kb}）")
    ab = pd.concat([a, b], ignore_index=True)
    m = metrics(ab, ka)
    b0 = pd.concat([b0_2025(body), pd.read_parquet(OUT / f"prereg_main_b0_races_{h}.parquet")], ignore_index=True)
    m0 = metrics(b0, ka)
    res = {"hash": h, "parts": {"2025": int(a["race_id"].nunique()), "2026_main": int(b["race_id"].nunique())}, "metrics": m,
           "B0": {k: m0.get(k) for k in ("roi", "roi_lo", "roi_hi", "hit_rate", "bet_races")}, "diff_vs_B0": paired(ab, b0),
           "note": "評価の期間の合算。判断・選び直しには使わない。判定のオッズは T-60 の再現（5 通り）。精算は確定の払戻。"}
    dst = T / f"bet_2025_2026_r2_combined_{h}.json"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("書いた", dst.relative_to(ROOT), json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
