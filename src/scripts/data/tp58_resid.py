"""tp-58 V5 の OOF 予測にまだ学べる構造が残っているか（残差ブースティング）。

V5 の OOF ロジットを init_score にした浅い木のブースティングを、検証年以外の 2 年で学習して検証年に当てる（LOYO）。
深さ 1（切り株＝列ごとの加法効果のみ）と深さ 3（3 列までの交互作用）の差が、交互作用で取れる分。
ラウンド数は学習 2 年の内側（1 年目で学習・2 年目で検証）で決め、検証年は見ない。

python -m src.scripts.data.tp58_resid [--top 400]
結果: $TP58_DIR/out/resid.json（ロジ損失の差 = 新 − V5、負が改善。レース単位のブートストラップ区間つき）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("TP58_DIR", ROOT / "data/local/modeling/shared_cache/tp58"))
CACHE, OUT = WORKDIR / "cache", WORKDIR / "out"
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP58_THREADS", 2))
YEARS = (2022, 2023, 2024)
GRID = (20, 40, 80, 160, 320)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fit(X, y, init, leaves, depth, rounds, seed=0):
    import lightgbm as lgb

    prm = dict(objective="binary", learning_rate=0.03, num_leaves=leaves, max_depth=depth, min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=20.0, num_threads=THREADS, verbose=-1, seed=seed)
    return lgb.train(prm, lgb.Dataset(X, y, init_score=init), num_boost_round=rounds)


def _boot(loss_b, loss_n, race, n=500, seed=0):
    d = pd.DataFrame({"r": race, "d": loss_n - loss_b}).groupby("r")["d"].sum()
    v = d.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    means = v[idx].sum(1) / len(loss_b)
    return float(v.sum() / len(loss_b)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--base", default=None, help="OOF の parquet（既定は V5 の LightGBM 単体）")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    imp = json.loads((CACHE / "importance_gain.json").read_text(encoding="utf-8"))
    cols = [c for c, _ in imp if not any(b in c.lower() for b in ("odds", "ninki", "popular"))][: a.top]
    F = pd.read_parquet(CACHE / "v5_frame_num.parquet", columns=KEY + ["finish_position", "year", "date"] + cols)
    F["race_id"], F["horse_id"] = F["race_id"].astype(str), F["horse_id"].astype(str)
    O = pd.read_parquet(a.base or OOF)
    O["race_id"], O["horse_id"] = O["race_id"].astype(str), O["horse_id"].astype(str)
    D = F[F["year"].isin(YEARS)].merge(O[KEY + ["p_win", "p_top2", "p_top3"]], on=KEY, how="inner").reset_index(drop=True)
    fp = pd.to_numeric(D["finish_position"], errors="coerce").to_numpy()
    X = D[cols].to_numpy(dtype="float32")
    res = {}
    for tgt, k in (("win", 1), ("top2", 2), ("top3", 3)):
        y = (fp <= k).astype(float)
        p0 = np.clip(D[f"p_{tgt}"].to_numpy(), 1e-6, 1 - 1e-6)
        init = np.log(p0 / (1 - p0))
        for name, leaves, depth in (("stump", 2, 1), ("depth3", 8, 3)):
            tot_b, tot_n, races, per_year = [], [], [], {}
            for h in YEARS:
                tr = [y_ for y_ in YEARS if y_ != h]
                m1, m2, mh = (D["year"] == tr[0]).to_numpy(), (D["year"] == tr[1]).to_numpy(), (D["year"] == h).to_numpy()
                # 内側: 1 年目で学習・2 年目で検証してラウンド数を選ぶ
                mdl = _fit(X[m1], y[m1], init[m1], leaves, depth, max(GRID))
                best, best_r = 1e9, GRID[0]
                for r in GRID:
                    z = init[m2] + mdl.predict(X[m2], num_iteration=r, raw_score=True)
                    l = _ll(y[m2], 1 / (1 + np.exp(-z))).mean()
                    if l < best:
                        best, best_r = l, r
                mtr = m1 | m2
                mdl = _fit(X[mtr], y[mtr], init[mtr], leaves, depth, best_r)
                z = init[mh] + mdl.predict(X[mh], raw_score=True)
                ln, lb = _ll(y[mh], 1 / (1 + np.exp(-z))), _ll(y[mh], p0[mh])
                tot_b.append(lb), tot_n.append(ln), races.append(D.loc[mh, "race_id"].to_numpy())
                per_year[str(h)] = {"rounds": best_r, "delta": float((ln - lb).mean())}
                _log(f"{tgt} {name} 検証{h}: rounds={best_r} Δ={(ln - lb).mean():.5f}")
            lb, ln, rc = np.concatenate(tot_b), np.concatenate(tot_n), np.concatenate(races)
            d, hi = _boot(lb, ln, rc)
            res[f"{tgt}_{name}"] = {"delta": d, "hi": hi, "per_year": per_year}
            _log(f"== {tgt} {name}: Δ={d:.5f} 上限={hi:.5f}")
    (OUT / f"resid{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
