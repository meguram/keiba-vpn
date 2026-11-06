"""tp-85: 観点ごとに説明量が最大の単位を選び、傾き k で補正した血統の期待値モデル（PedPrior）を作って保存する。ホストで実行。

python -m src.scripts.data.tp85_pedprior_build [--traits dirt_share,ax_ts] [--no-final]
入力: tp85_ped_traits の評価結果（_sex, _sex2 の results）。出力: data/local/modeling/tp85_ped_traits/model/<観点>.pkl と model/manifest.json。
選び方: 候補 A2 < A2s(性別の平均) < A2i(+牡牝ごとのずれ) の中で、検証の観測 R² が最大のもの。
        ただし単純な方に対する上乗せが MIN_GAIN 未満なら単純な方を採る。補正後の観測 R² が ADOPT_R2 未満の観点は採用しない。
検証 = 2023 年以降に初出走した馬（軸は 2025-02-28 以降）。最終モデルは 2025 年末までの全馬で学習し直す（2026 年は使わない）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src.pipeline.features.tp85_pedprior import MODEL_DIR, PedPrior
from src.scripts.data import tp85_ped_traits as t85

BASE = t85.ROOT / "data/local/modeling"
DIRS = [BASE / "tp85_ped_traits_sex" / "results"]
ORDER = ["A2", "A2s", "A2i"]          # 入力は性別・父・母父だけ（母母父以降は使わない）
MIN_GAIN = 0.003
ADOPT_R2 = 0.01
SPEC = {v[0]: v for v in t85.VARIANTS + t85.SEX_VARIANTS}


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def choose(trait: str) -> tuple[str, dict]:
    r2 = {}
    for d in DIRS:
        f = d / f"{trait}.json"
        if f.exists():
            for k, v in json.loads(f.read_text(encoding="utf-8")).get("variants", {}).items():
                r2[k] = v["r2_obs"]
    best = None
    for k in ORDER:
        if k in r2 and (best is None or r2[k] - r2[best] >= MIN_GAIN):
            best = k
    return best, r2


def metrics_cal(st: dict, k: float) -> dict:
    te = st["te"]
    r = te["yte"] - k * te["mu"]
    tot = float((te["yte"] ** 2).sum())
    cov = float((np.abs(r) <= t85.Z80 * np.sqrt(te["var_true"] + te["vte"])).mean())
    return {"r2_obs_cal": round(1 - float((r ** 2).sum()) / tot, 4), "cover80_cal": round(cov, 4)}


DERIVED = {"ax_sustain": ["ax_gear", "ax_ts", "ax_endure"]}      # 血統が予測した入力の観点 → 派生の観点（基礎スピード・Cr・まくりは血統経由でも増えず採用しない）


def build_derived(name: str, inputs: list[str], T: dict, ped: dict, train_pp: dict) -> dict:
    """検証馬（学習に使っていない新しい馬）で、入力の観点の血統予測 → 目標の実測 を重み付き最小二乗で当てはめる（5 分割 CV で採否と区間を決める）。"""
    t = T[name]
    t = t[t["horse_id"].isin(ped) & (t["first_date"] > t85.SPLIT_AX)].reset_index(drop=True)
    X = np.column_stack([train_pp[k].predict(ped, t[["horse_id", "sex"]])["mean"].to_numpy() for k in inputs])
    y, v = t["y"].to_numpy(), t["v"].to_numpy()
    tau_pop = max(float(np.var(y) - v.mean()), 1e-9)
    w = 1 / (tau_pop + v)
    A = np.column_stack([np.ones(len(y)), X])
    fold = np.random.default_rng(0).integers(0, 5, len(y))
    pred = np.zeros(len(y))
    for f in range(5):
        a, b = fold != f, fold == f
        sw = np.sqrt(w[a])
        pred[b] = A[b] @ np.linalg.lstsq(A[a] * sw[:, None], y[a] * sw, rcond=None)[0]
    r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    tau_res = max(float(np.mean((y - pred) ** 2 - v)), 1e-9)
    cover = float((np.abs(y - pred) <= t85.Z80 * np.sqrt(tau_res + v)).mean())
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    lo, hi = np.quantile(pred, [0.1, 0.9])
    tail = {"pred_diff": round(float(pred[pred >= hi].mean() - pred[pred <= lo].mean()), 4), "actual_diff": round(float(y[pred >= hi].mean() - y[pred <= lo].mean()), 4)}
    return {"trait": name, "kind": "derived", "inputs": inputs, "beta": [float(x) for x in beta], "tau_res": tau_res, "tau_pop": tau_pop, "r2_cv": round(r2, 4),
            "cover80_cv": round(cover, 4), "ceiling_r2": round(tau_pop / (tau_pop + float(v.mean())), 4), "n": len(y), "tails": tail, "adopted": bool(r2 >= ADOPT_R2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", default="")
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()
    d = t85.load_runs()
    T = t85.build_traits(d)
    want = [x for x in a.traits.split(",") if x] or [t for t in T if (DIRS[0] / f"{t}.json").exists()]
    ped = t85.load_ped(sorted(set().union(*[set(T[k]["horse_id"]) for k in want])))
    nodes_fn = t85.make_nodes(json.loads(t85.ORIGIN_PATH.read_text(encoding="utf-8")))
    manifest = {}
    train_pp: dict = {}
    for tr_name in want:
        vn, r2 = choose(tr_name)
        if vn is None:
            _log(f"{tr_name}: 評価結果なし")
            continue
        t = T[tr_name][T[tr_name]["horse_id"].isin(ped)].reset_index(drop=True)
        sp_ = t85.SPLIT_AX if tr_name.startswith("ax_") else t85.SPLIT
        tr, te = t[t["first_date"] <= sp_].reset_index(drop=True), t[t["first_date"] > sp_].reset_index(drop=True)
        _, pos, mode = SPEC[vn]
        sexmode = t85.SEXMODE.get(vn, "none")
        o, st = t85.fit_variant(vn, pos, mode, ped, nodes_fn, tr, te, sexmode=sexmode)
        k = o["calib"]["slope"]
        mc = metrics_cal(st, k)
        meta = {"trait": tr_name, "unit": vn, "positions": pos, "mode": mode, "sexmode": sexmode, "slope": k, "slope_se": o["calib"]["slope_se"],
                "r2_obs": o["r2_obs"], "ceiling_r2": o["ceiling_r2"], **mc, "cover80_raw": o["cover80"], "candidates_r2": r2,
                "n_train": len(tr), "n_test": len(te), "bottom10": o["calib"]["bottom10"], "top10": o["calib"]["top10"]}
        adopted = mc["r2_obs_cal"] >= ADOPT_R2
        meta["adopted"] = bool(adopted)
        _log(f"{tr_name}: 単位={vn} 傾き={k}±{o['calib']['slope_se']} R²(未補正)={o['r2_obs']:+.3f} → 補正後={mc['r2_obs_cal']:+.3f} 被覆(補正後)={mc['cover80_cal']:.3f} 採用={adopted}")
        if adopted:
            pp = PedPrior(meta, st)
            train_pp[tr_name] = pp
            sub = te.head(300)
            got = pp.predict(ped, sub[["horse_id", "sex"]])["mean"].to_numpy()
            exp = st["m"] + st["s"] * k * st["te"]["mu"][: len(sub)]
            assert np.allclose(got, exp, atol=1e-6 * max(1.0, float(np.abs(exp).max()))), f"{tr_name}: 推論が評価と一致しない"
            if not a.no_final:
                full = t
                o2, st2 = t85.fit_variant(vn, pos, mode, ped, nodes_fn, full, full.head(300), sexmode=sexmode)   # te は形式上（最終モデルの係数だけ使う）
                PedPrior({**meta, "final_n": len(full), "final_through": str(t85.LAST_DATE.date())}, st2).save()
        manifest[tr_name] = meta
    for dn, ins in DERIVED.items():
        if dn in T and all(i in train_pp for i in ins):
            de = build_derived(dn, ins, T, ped, train_pp)
            manifest[dn] = {**manifest.get(dn, {}), **de}
            _log(f"{dn}（{'・'.join(ins)} の血統予測から）: CV R²={de['r2_cv']:+.4f}（上限 {de['ceiling_r2']}） 被覆(CV)={de['cover80_cv']:.3f} 端 予測差{de['tails']['pred_diff']:+.3f}→実測差{de['tails']['actual_diff']:+.3f} 採用={de['adopted']}")
    if not a.no_final:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
