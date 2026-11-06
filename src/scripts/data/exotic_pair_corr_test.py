"""組合せ確率（MB60）の実際とのずれが、予想位置取り（脚質）の近さで説明できるかの検定。

説明できる = 馬同士の「一緒に来やすさ」に相関があり、レースシミュレーションを試す価値がある。
説明できない = 馬ごとの強さだけの分布で足りる。

方法: 各レースの全組合せ（馬連=ペア、3連複=3頭）について、排他的な結果（実際の組合せ）の条件付きロジット
    スコア = log P_MB60(組合せ) + β · f(組合せの予想位置取り)
を、年を除いた年で β を当てはめ（LOYO）、除いた年で対数尤度の改善（β=0 との差）を見る。
f は個別の効果（前に行く馬の数）と相互作用（前の馬同士・後ろの馬同士・位置の近さ）の両方を入れる。
予想位置取りは馬柱から作る pf_early（position_forecast。レース前に得られる量だけ）。

    python -m src.scripts.data.exotic_pair_corr_test run
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.scripts.data import exotic_method_eval as ev  # noqa: E402
from src.scripts.data.tp53_dataset import TABLES, load_pf  # noqa: E402

FRONT, BACK = 0.35, 0.65
OUT = ev.OUT


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_races_with_pf() -> list[dict]:
    from src.scripts.data import tp27_eval as t

    _, _, meta, _, _ = t.setup()
    meta = meta[["race_id", "horse_id", "finish_position"]].copy()
    o = pd.read_parquet(ev.OOF)
    for d in (o, meta):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    m = o.merge(meta, on=["race_id", "horse_id"], how="left")
    m["finish_position"] = pd.to_numeric(m["finish_position"], errors="coerce")
    years = sorted(m["year"].unique())
    hn, pf = [], []
    for y in years:
        f = pd.read_parquet(TABLES / str(int(y)) / "race_result_flat.parquet", columns=["race_id", "horse_id", "horse_number"])
        f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
        hn.append(f)
        h, _ = load_pf(int(y))
        h["race_id"] = h["race_id"].astype(str)
        pf.append(h[["race_id", "horse_number", "pf_early"]])
        _log(f"位置取り予想 {int(y)}: {len(h)} 行")
    hn = pd.concat(hn).drop_duplicates(["race_id", "horse_id"])
    pf = pd.concat(pf).drop_duplicates(["race_id", "horse_number"])
    m = m.merge(hn, on=["race_id", "horse_id"], how="left").merge(pf, on=["race_id", "horse_number"], how="left")
    races = []
    for rid, g in m.groupby("race_id", sort=True):
        n = len(g)
        if n < 4:
            continue
        rk = g["finish_position"].to_numpy(dtype=float)
        order = np.argsort(np.where(np.isnan(rk) | (rk <= 0), 1e9, rk), kind="stable")
        if not np.array_equal(rk[order[:3]], [1.0, 2.0, 3.0]) or np.sum(np.isin(rk, [1.0, 2.0, 3.0])) != 3:
            continue
        e = g["pf_early"].to_numpy(dtype=float)
        races.append({
            "race_id": rid, "year": int(g["year"].iloc[0]), "n": n,
            "win": g["p_win"].to_numpy(float), "top2": g["p_top2"].to_numpy(float), "top3": g["p_top3"].to_numpy(float),
            "abc": order[:3].copy(), "rank": np.where(np.isnan(rk) | (rk <= 0), 99.0, rk),
            "e": e, "known": float(np.isfinite(e).mean()),
        })
    _log(f"レース {len(races)}")
    return races


def pair_feats(e: np.ndarray, idx: np.ndarray) -> np.ndarray:
    a, b = e[idx[:, 0]], e[idx[:, 1]]
    fa, fb = (a <= FRONT).astype(float), (b <= FRONT).astype(float)
    ba, bb = (a >= BACK).astype(float), (b >= BACK).astype(float)
    return np.column_stack([fa + fb, ba + bb, fa * fb, ba * bb, np.abs(a - b)])


def triple_feats(e: np.ndarray, idx: np.ndarray) -> np.ndarray:
    v = np.column_stack([e[idx[:, 0]], e[idx[:, 1]], e[idx[:, 2]]])
    nf, nb = (v <= FRONT).sum(1).astype(float), (v >= BACK).sum(1).astype(float)
    return np.column_stack([nf, nb, nf ** 2, nb ** 2, v.max(1) - v.min(1)])


NAMES = {"umaren": ["前の馬の数", "後ろの馬の数", "前×前", "後×後", "位置の差"],
         "sanrenpuku": ["前の馬の数", "後ろの馬の数", "前の数²", "後ろの数²", "位置の幅"]}


def build(races: list[dict], method: str = "MB60") -> dict[str, list]:
    years = sorted({r["year"] for r in races})
    cals = ev.marginal_calibrators(races, years)
    gal = {y: ev.global_alpha([r for r in races if r["year"] != y]) for y in years}
    out: dict[str, list] = {"umaren": [], "sanrenpuku": []}
    t0 = time.time()
    for k, r in enumerate(races):
        n = r["n"]
        a, b, c = (int(x) for x in r["abc"])
        e = np.where(np.isfinite(r["e"]), r["e"], 0.5)
        d = ev.make_dist(method, r, gal[r["year"]], cals[r["year"]])
        p2, p3 = d._p2, d._p3
        pr = ev._combs(n, 2)
        um = p2[pr[:, 0], pr[:, 1]] + p2[pr[:, 1], pr[:, 0]]
        yu = (((pr[:, 0] == a) & (pr[:, 1] == b)) | ((pr[:, 0] == b) & (pr[:, 1] == a)))
        tr = ev._combs(n, 3)
        i, j, kk = tr[:, 0], tr[:, 1], tr[:, 2]
        tri = p3[i, j, kk] + p3[i, kk, j] + p3[j, i, kk] + p3[j, kk, i] + p3[kk, i, j] + p3[kk, j, i]
        ys = np.isin(i, [a, b, c]) & np.isin(j, [a, b, c]) & np.isin(kk, [a, b, c])
        out["umaren"].append((r["year"], np.log(np.clip(um, 1e-12, None)), pair_feats(e, pr), int(np.argmax(yu))))
        out["sanrenpuku"].append((r["year"], np.log(np.clip(tri, 1e-12, None)), triple_feats(e, tr), int(np.argmax(ys))))
        if (k + 1) % 1000 == 0:
            _log(f"{k + 1}/{len(races)}（{time.time() - t0:.0f}s）")
    return out


def _ll(beta: np.ndarray, rows: list, grad: bool = False):
    tot, g = 0.0, np.zeros_like(beta)
    for _, lp, F, t in rows:
        s = lp + F @ beta
        mx = s.max()
        w = np.exp(s - mx)
        z = w.sum()
        tot += s[t] - mx - np.log(z)
        if grad:
            g += F[t] - (w / z) @ F
    return (tot, g) if grad else tot


def _fit(rows: list, k: int, l2: float = 1.0) -> np.ndarray:
    from scipy.optimize import minimize

    def f(b):
        v, g = _ll(b, rows, True)
        return -(v) + l2 * 0.5 * (b @ b), -g + l2 * b

    return minimize(f, np.zeros(k), jac=True, method="L-BFGS-B").x


def per_race_gain(beta: np.ndarray, rows: list) -> np.ndarray:
    out = []
    for _, lp, F, t in rows:
        out.append(_ll(beta, [(0, lp, F, t)]) - _ll(np.zeros_like(beta), [(0, lp, F, t)]))
    return np.array(out)


def run() -> None:
    races = load_races_with_pf()
    known = np.mean([r["known"] for r in races])
    _log(f"予想位置取りが取れた馬の割合 {known:.3f}")
    data = build(races)
    years = sorted({r["year"] for r in races})
    rng = np.random.default_rng(0)
    res: dict = {"n_races": len(races), "known_share": float(known), "bets": {}}
    for bet, rows in data.items():
        k = len(NAMES[bet])
        gains, by_year, betas = [], {}, {}
        for y in years:
            tr = [x for x in rows if x[0] != y]
            te = [x for x in rows if x[0] == y]
            b = _fit(tr, k)
            g = per_race_gain(b, te)
            gains.append(g)
            by_year[str(y)] = float(g.mean())
            betas[str(y)] = b.round(4).tolist()
        g = np.concatenate(gains)
        bs = rng.integers(0, len(g), (1000, len(g)))
        m = g[bs].mean(1)
        ball = _fit(rows, k)
        res["bets"][bet] = {"mean_gain_per_race": float(g.mean()), "lo": float(np.quantile(m, .025)), "hi": float(np.quantile(m, .975)),
                            "by_year": by_year, "beta_by_heldout_year": betas, "beta_all_years": dict(zip(NAMES[bet], ball.round(4).tolist()))}
        _log(f"{bet}: 1 レースあたりの対数尤度の改善 {g.mean():+.5f} [{np.quantile(m, .025):+.5f}, {np.quantile(m, .975):+.5f}] 年別 {by_year}")
        _log(f"  β（全年）{dict(zip(NAMES[bet], ball.round(3).tolist()))}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pair_corr_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("完了")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        print(__doc__)
