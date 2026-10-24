"""tp-63 後半: ハイパーパラメータの調整履歴の確認と、メンバー別・レース条件別の感度曲線。

python -m ... sweep [--tag base] [--years 2022,2023] [--targets win,top3] [--fast]    # 主要パラメータを 1 つずつ動かし、全体と条件別の LogLoss を木数ごとに記録
python -m ... show --tag base                                                          # 結果の要約（条件ごとの最適値、条件別に最適化したときの見込みの利得）

入力は V5 と同じ 1,142 列。学習は年の前進（検証年より前の全年）。選択は valid 2022・2023、確認は 2024（最終候補のみ）。test 2025 は使わない。市場オッズ・人気は使わない。
条件: 面（芝/ダ）、距離帯、馬場状態、クラス帯、頭数。重い処理は /tmp/heavy_run.sh 経由で起動する。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.scripts.data import tp63_members as mem
except ImportError:      # ホストのリポジトリは書けないので、新規ファイルは別ディレクトリに置く
    import tp63_members as mem  # type: ignore

OUT = mem.OUT / "tune"
THREADS = int(os.environ.get("TP63_THREADS", 3))
ITERS = [60, 100, 150, 200, 260, 330, 400, 500, 600]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sweep_configs(base: dict, which: list[str]) -> dict[str, dict]:
    cfgs = {"base": {}}
    if "leaves" in which:
        for v in (7, 63):
            cfgs[f"leaves={v}"] = {"num_leaves": v}
    if "min_data" in which:
        for v in (20, 400):
            cfgs[f"min_data={v}"] = {"min_data_in_leaf": v}
    if "l2" in which:
        for v in (10, 300):
            cfgs[f"l2={v}"] = {"lambda_l2": v}
    if "combo" in which:
        cfgs.update({
            "combo=l7_ff5": {"num_leaves": 7, "feature_fraction": 0.5},
            "combo=l7_ff5_md200": {"num_leaves": 7, "feature_fraction": 0.5, "min_data_in_leaf": 200},
            "combo=l4_ff5": {"num_leaves": 4, "feature_fraction": 0.5},
            "combo=l11_ff5": {"num_leaves": 11, "feature_fraction": 0.5},
            "combo=l7_ff35": {"num_leaves": 7, "feature_fraction": 0.35},
            "combo=l7_ff5_l2_30": {"num_leaves": 7, "feature_fraction": 0.5, "lambda_l2": 30},
        })
    if "ff" in which:
        for v in (0.2, 0.5):
            cfgs[f"ff={v}"] = {"feature_fraction": v}
    return cfgs


def cmd_sweep(a) -> None:
    import lightgbm as lgb

    from src.pipeline.models import win_models as wm

    df, X, meta, folds, lp = mem.load_frame()
    OUT.mkdir(parents=True, exist_ok=True)
    years = [int(y) for y in a.years.split(",")]
    targets = a.targets.split(",")
    p_base = dict(lp["lgb"])
    fast = 3.0 if a.fast else 1.0
    p_base["learning_rate"] = float(p_base["learning_rate"]) * fast
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    fin = meta["finish_position"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top3": (fin <= 3).astype(int), "top2": (fin <= 2).astype(int)}
    conds = mem.condition_labels(df)
    path = OUT / f"sweep_{a.tag}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfgs = sweep_configs(p_base, a.params.split(","))
    iters = [k for k in ITERS if k <= a.max_rounds]
    for f in [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]:
        y = int(meta["year"].to_numpy()[f.valid_index][0])
        tr, va = f.train_index, f.valid_index
        for tgt in targets:
            ds = lgb.Dataset(X.iloc[tr], label=Y[tgt][tr], categorical_feature=cats or "auto", free_raw_data=False, params={"feature_pre_filter": False})
            yv = Y[tgt][va]
            for cname, over in cfgs.items():
                key = f"{cname}|{y}|{tgt}"
                if key in res:
                    continue
                t0 = time.time()
                p = {**wm.BASE_PARAMS, **p_base, **over, "seed": 0, "num_threads": a.threads, "objective": "binary", "verbose": -1, "feature_pre_filter": False}
                b = lgb.train(p, ds, num_boost_round=max(iters))
                row = {"seconds": time.time() - t0}
                for k in iters:
                    pr = np.clip(b.predict(X.iloc[va], num_iteration=k), 1e-9, 1 - 1e-9)
                    ll = -(yv * np.log(pr) + (1 - yv) * np.log(1 - pr))
                    r = {"all": float(ll.mean())}
                    for cn, lab in conds.items():
                        lv = lab[va]
                        for c in np.unique(lv):
                            r[f"{cn}={c}"] = float(ll[lv == c].mean())
                    row[str(k)] = r
                res[key] = row
                path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
                best = min(iters, key=lambda k: row[str(k)]["all"])
                _log(f"{key}: best@{best} LL {row[str(best)]['all']:.5f} ({row['seconds']:.0f}s)")


def cmd_show(a) -> None:
    res = json.loads((OUT / f"sweep_{a.tag}.json").read_text(encoding="utf-8"))
    cfgs = sorted({k.split("|")[0] for k in res})
    years = sorted({k.split("|")[1] for k in res})
    tgt = a.target
    # 全体: 構成 × 年の最良木数での LogLoss
    print(f"target={tgt} years={years}")
    for c in cfgs:
        cells = []
        for y in years:
            r = res.get(f"{c}|{y}|{tgt}")
            if not r:
                continue
            ks = [k for k in r if k != "seconds"]
            kb = min(ks, key=lambda k: r[k]["all"])
            cells.append(f"{y}: {r[kb]['all']:.5f}@{kb}")
        print(f"{c:16s} " + "  ".join(cells))


def cmd_pick(a) -> None:
    """感度の掃引から、パラメータごとの最良値（基準比 −0.00003 以上の改善があるものだけ）を組み合わせた設定を tune/best_params.json に書く。
    学習率 3 倍の簡易設定で掃引しているので、木数は最良ラウンドの 3 倍（V5 と同じ学習率に戻す）。"""
    res = json.loads((OUT / f"sweep_{a.tag}.json").read_text(encoding="utf-8"))
    years = sorted({k.split("|")[1] for k in res})

    def best_ll(c):
        vals, its = [], []
        for y in years:
            r = res.get(f"{c}|{y}|win")
            if not r:
                return None, None
            ks = [x for x in r if x != "seconds"]
            kb = min(ks, key=lambda x: r[x]["all"])
            vals.append(r[kb]["all"])
            its.append(int(kb))
        return float(np.mean(vals)), float(np.mean(its))

    base, base_it = best_ll("base")
    groups = {"num_leaves": "leaves=", "min_data_in_leaf": "min_data=", "lambda_l2": "l2=", "feature_fraction": "ff="}
    chosen, table = {}, {}
    for key, pref in groups.items():
        cands = {c: best_ll(c) for c in {k.split("|")[0] for k in res} if c.startswith(pref)}
        cands = {c: v for c, v in cands.items() if v[0] is not None}
        if not cands:
            continue
        c, (ll, it) = min(cands.items(), key=lambda kv: kv[1][0])
        table[key] = {"best": c, "ll": ll, "delta_vs_base": ll - base, "all": {k: v[0] for k, v in cands.items()}}
        if ll - base <= -0.00003:
            v = c.split("=")[1]
            chosen[key] = int(v) if key in ("num_leaves", "min_data_in_leaf") else float(v)
    rounds = int(min(1200, round(3 * base_it / 50) * 50)) if base_it else 789
    out = {"lgb": chosen, "n_rounds": rounds, "base_ll": base, "base_best_iter_fast": base_it, "table": table}
    (OUT / "best_params.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"採用する設定: {chosen}、木数 {rounds}（基準 LL {base:.5f}）")


def cmd_pick2(a) -> None:
    """組み合わせの掃引（tag）で、全設定のうち平均 LogLoss が最良のものを tune/best_params2.json に書く（木数 = 最良ラウンド × 3。簡易設定は学習率 3 倍）。"""
    res = json.loads((OUT / f"sweep_{a.tag}.json").read_text(encoding="utf-8"))
    years = sorted({k.split("|")[1] for k in res})
    cfgs = {k.split("|")[0] for k in res}
    over = sweep_configs({}, ["leaves", "combo"])
    best = None
    for c in cfgs:
        vals, its = [], []
        for y in years:
            r = res.get(f"{c}|{y}|win")
            if not r:
                break
            ks = [x for x in r if x != "seconds"]
            kb = min(ks, key=lambda x: r[x]["all"])
            vals.append(r[kb]["all"])
            its.append(int(kb))
        else:
            if best is None or np.mean(vals) < best[0]:
                best = (float(np.mean(vals)), float(np.mean(its)), c)
    ll, it, c = best
    ov = over.get(c, {})
    out = {"lgb": ov, "n_rounds": int(min(1500, round(3 * it / 50) * 50)), "config": c, "ll": ll, "best_iter_fast": it,
           "all": {k: None for k in cfgs}}
    (OUT / "best_params2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"最良: {c} {ov} LL {ll:.5f} 木数(簡易)={it:.0f} → 木数 {out['n_rounds']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sweep")
    s.add_argument("--tag", default="base")
    s.add_argument("--years", default="2022,2023")
    s.add_argument("--targets", default="win")
    s.add_argument("--params", default="leaves,min_data,l2,ff")
    s.add_argument("--threads", type=int, default=THREADS)
    s.add_argument("--max-rounds", type=int, default=400)
    s.add_argument("--fast", action="store_true")
    p2 = sp.add_parser("pick2")
    p2.add_argument("--tag", default="combo")
    pk = sp.add_parser("pick")
    pk.add_argument("--tag", default="base")
    w = sp.add_parser("show")
    w.add_argument("--tag", default="base")
    w.add_argument("--target", default="win")
    a = ap.parse_args()
    {"sweep": cmd_sweep, "show": cmd_show, "pick": cmd_pick, "pick2": cmd_pick2}[a.cmd](a)


if __name__ == "__main__":
    main()
