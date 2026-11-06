"""tp-85: 血統から言える軸（ギアチェンジ・TS・Hペース耐性）から、言えない軸（TS持続力・Cr性能・基礎スピード）を逆算できるか。ホストで実行。

python -m src.scripts.data.tp85_axis_infer
同じ新しい馬（初めて軸が付いたのが 2025-03 以降）で、次の予測の当てはまりを 5 分割の交差検証で比べる（観測 R²、馬単位）:
  M0 血統だけで目標の軸を直接予測（A2s）  M1 実測のギア・TS・耐性から  M2 血統が予測したギア・TS・耐性から  M3 M0+M2
  M4 血統の予測値 8 観点（ギア・TS・耐性・出脚・先行力・位置取り・ダート比率・距離）から
学習用の血統モデルは検証馬を含めない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from src.scripts.data import tp85_ped_traits as t85

CORE = ["ax_gear", "ax_ts", "ax_endure"]
EXT = CORE + ["ax_debut", "ax_front", "style_q", "dirt_share", "dist_run"]
TARGETS = ["ax_sustain", "ax_cr", "basic", "ax_makuri"]
OUT = t85.ROOT / "data/local/modeling/tp85_axis_infer"


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pedigree_mu(name: str, t: pd.DataFrame, V: set, ped, nodes_fn, sex: pd.Series) -> pd.Series:
    """学習 = 検証馬でなく初出走が SPLIT_AX 以前の馬。検証馬『全員』の予測（自然な単位）を horse_id で返す（その観点の実測が無い馬も含む）。"""
    t = t[t["horse_id"].isin(ped)].reset_index(drop=True)
    tr = t[(~t["horse_id"].isin(V)) & (t["first_date"] <= t85.SPLIT_AX)].reset_index(drop=True)
    hv = sorted(V)
    te = pd.DataFrame({"horse_id": hv, "sex": sex.reindex(hv).fillna("牡").to_numpy(), "y": float(tr["y"].mean()) + np.linspace(-1, 1, len(hv)) * 1e-3 * float(tr["y"].std()), "v": 1.0, "first_date": t85.SPLIT_AX + pd.Timedelta(days=1)})
    o, st = t85.fit_variant("A2s", [1, 2], "shared", ped, nodes_fn, tr, te, sexmode="fix")
    _log(f"  血統→{name}: 学習 {len(tr):,} 頭 / 予測 {len(te):,} 頭")
    return pd.Series(st["m"] + st["s"] * st["te"]["mu"], index=te["horse_id"].to_numpy())


def cv_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray, k: int = 5, seed: int = 0) -> tuple[np.ndarray, float]:
    """5 分割の重み付き最小二乗。返り値 = 外の予測と、観測 R²（y の平均まわり）。"""
    n = len(y)
    fold = np.random.default_rng(seed).integers(0, k, n)
    pred = np.zeros(n)
    A = np.column_stack([np.ones(n), X]) if X.size else np.ones((n, 1))
    for f in range(k):
        tr, te = fold != f, fold == f
        sw = np.sqrt(w[tr])
        beta = np.linalg.lstsq(A[tr] * sw[:, None], y[tr] * sw, rcond=None)[0]
        pred[te] = A[te] @ beta
    return pred, 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def boot_ci(y: np.ndarray, pred: np.ndarray, n: int = 400) -> tuple[float, float]:
    rng = np.random.default_rng(1)
    r = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        r.append(1 - ((y[i] - pred[i]) ** 2).sum() / ((y[i] - y[i].mean()) ** 2).sum())
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def tails(y: np.ndarray, pred: np.ndarray, v: np.ndarray) -> dict:
    lo, hi = np.quantile(pred, [0.1, 0.9])
    b, t = pred <= lo, pred >= hi
    se = lambda m: float(np.sqrt(np.mean(v[m]) / m.sum() + np.var(y[m]) / m.sum()))
    return {"pred_diff": round(float(pred[t].mean() - pred[b].mean()), 3), "actual_diff": round(float(y[t].mean() - y[b].mean()), 3), "se": round(float(np.hypot(se(b), se(t))), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = t85.load_runs()
    T = t85.build_traits(d)
    ped = t85.load_ped(sorted(set().union(*[set(T[k]["horse_id"]) for k in EXT + TARGETS if k in T])))
    nodes_fn = t85.make_nodes(json.loads(t85.ORIGIN_PATH.read_text(encoding="utf-8")))
    base = T["ax_gear"]
    V = set(base[(base["first_date"] > t85.SPLIT_AX)]["horse_id"]) & set(ped)
    for k in CORE:
        V &= set(T[k]["horse_id"])
    _log(f"検証馬（3 軸とも実測あり・血統あり・初軸 2025-03 以降） {len(V):,} 頭")
    sex = d.dropna(subset=["sex"]).groupby("horse_id")["sex"].first()
    mu = {k: pedigree_mu(k, T[k], V, ped, nodes_fn, sex) for k in EXT}
    out = {"n_val": len(V), "targets": {}}
    corr_idx = sorted(V)
    meas = pd.DataFrame({k: T[k].drop_duplicates("horse_id").set_index("horse_id")["y"] for k in CORE}).reindex(corr_idx)
    pm = pd.DataFrame({k: mu[k] for k in CORE}).reindex(corr_idx)
    out["corr_measured_core"] = meas.corr().round(3).to_dict()
    out["corr_pedigree_core"] = pm.corr().round(3).to_dict()
    for tg in TARGETS:
        if tg not in T:
            continue
        mu_t = pedigree_mu(tg, T[tg], V, ped, nodes_fn, sex)
        tt = T[tg].drop_duplicates("horse_id").set_index("horse_id")
        h = [x for x in corr_idx if x in tt.index and x in mu_t.index]
        y, v = tt.loc[h, "y"].to_numpy(), tt.loc[h, "v"].to_numpy()
        tau0 = max(float(np.var(y) - v.mean()), 1e-9)
        w = 1 / (tau0 + v)
        ceil = tau0 / (tau0 + float(v.mean()))
        feats = {
            "M0 血統だけで直接(A2s)": mu_t.loc[h].to_numpy()[:, None],
            "M1 実測ギア・TS・耐性から": meas.loc[h].to_numpy(),
            "M2 血統予測のギア・TS・耐性から": np.column_stack([mu[k].loc[h].to_numpy() for k in CORE]),
            "M3 M0+M2": np.column_stack([mu_t.loc[h].to_numpy()] + [mu[k].loc[h].to_numpy() for k in CORE]),
            "M4 血統予測の8観点から": np.column_stack([mu[k].loc[h].to_numpy() for k in EXT]),
            "M5 M4+M0": np.column_stack([mu_t.loc[h].to_numpy()] + [mu[k].loc[h].to_numpy() for k in EXT]),
        }
        res = {"n": len(h), "ceiling_r2": round(ceil, 3), "models": {}}
        _log(f"{tg}: 検証 {len(h):,} 頭 上限R²={ceil:.2f}")
        for nm, X in feats.items():
            pred, r2 = cv_r2(X, y, w)
            lo, hi = boot_ci(y, pred)
            res["models"][nm] = {"r2_obs": round(r2, 4), "ci95": [round(lo, 4), round(hi, 4)], "tails": tails(y, pred, v)}
            _log(f"  {nm}: R²={r2:+.4f} (95%区間 {lo:+.4f}〜{hi:+.4f}) 端: 予測差{res['models'][nm]['tails']['pred_diff']:+.3f}→実測差{res['models'][nm]['tails']['actual_diff']:+.3f}±{res['models'][nm]['tails']['se']:.3f}")
        out["targets"][tg] = res
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
