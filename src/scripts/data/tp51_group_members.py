"""tp-51 段階 B: 特徴量グルーピング型のメンバー（グループ別の専門家 / 1 グループを抜く / ランダム部分空間）を OOF で作り、合成の効果を見る。

python -m src.scripts.data.tp51_group_members run KIND [--years 2022,2023] [--seeds 0] [--full-quality] [--limit N]
    KIND: full（全列、基準）/ experts（グループ + COND だけの LightGBM）/ logo（1 グループを抜く）/ bags（グループ単位のランダム部分空間）/ nnexp（NN の専門家）
python -m src.scripts.data.tp51_group_members combine [--years 2022,2023]      # 平均・スタッキング・階層スタッキングを作り、全列モデルと比べる
python -m src.scripts.data.tp51_group_members importance                        # 全列モデルの gain をグループ別に集計
python -m src.scripts.data.tp51_group_members corr                              # メンバー間の OOF 相関（多様性）

粗（既定）: valid 2022・2023、seed 1 本、学習率 3 倍・ラウンド 1/3 の『fast』設定（基準の全列モデルも同じ設定）。有望なものだけ --full-quality（V4 と同じ設定）で 2024 を足して確認。
test 2025・2026 年は使わない。市場オッズ・人気は使わない。重い処理は /tmp/heavy_run.sh 経由で起動する。
環境変数: TP51_THREADS（既定 3）, TP51_OUT。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TP40_BASE", "old")          # 基準 = V4 の元の tl_（V4 の 7 メンバーと同じ列）
from src.scripts.data import tp27_eval as t          # noqa: E402
from src.scripts.data import tp40_eval as e40        # noqa: E402,F401  （環境変数の既定値を設定する副作用のため）
try:
    from src.scripts.data import tp51_groups as G  # noqa: E402
except ImportError:      # ホストのリポジトリは書けないので、新規ファイルは別ディレクトリに置く
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tp51_groups as G  # type: ignore  # noqa: E402

V = Path(e40.V)
KEY = ["race_id", "horse_id"]
OUT = Path(os.environ.get("TP51_OUT") or "/home/jovyan/work/tp51_out")
OOF = OUT / "gm_oof"
THREADS = int(os.environ.get("TP51_THREADS", 3))
TARGETS = ("win", "top2", "top3")
CLUSTERS = {
    "PERF": ["HIST_RESULT", "HIST_SPEED", "AFFINITY", "EXP", "STATE"],
    "CONN": ["CONN_OWNER", "CONN_JKTR"],
    "FIELD": ["FIELD_MEM", "FIELD_DEMO", "FIELD_CONN", "FIELD_ENV", "FIELD_PACE"],
    "STATIC": ["PED", "TL", "STAGE_AUX"],
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_frame():
    df, base, meta, folds, lp = t.setup()
    X = pd.read_parquet(V / "tp31" / "v4_frame.parquet")
    X.index = df.index
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    grp = G.group_columns(list(X.columns))
    return df, X, meta, folds, lp, grp


def lgb_params(lp: dict, fast: bool) -> dict:
    p = json.loads(json.dumps({"lgb": lp["lgb"], "n_rounds": lp["n_rounds"]}))
    if fast:
        p["lgb"]["learning_rate"] = float(p["lgb"]["learning_rate"]) * 3
        p["n_rounds"] = max(60, int(p["n_rounds"] // 3))
    return p


def run_one(name: str, cols: list[str], X, meta, folds, lp, years: list[int], seeds: list[int], fast: bool, recipe: str = "lgb_binary", params: dict | None = None) -> None:
    from src.pipeline.models import zoo_runner

    p = OOF / f"{name}.parquet"
    if p.exists():
        return
    fs = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    t0 = time.time()
    prm = (params if (recipe == "lgb_binary" and params) else lgb_params(lp, fast)) if recipe == "lgb_binary" else dict(params or {})
    res, oof = zoo_runner.run_oof(name, recipe, X[cols], meta, fs, seeds, THREADS, params=prm)
    OOF.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(p, index=False)
    _log(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} cols={len(cols)} ({time.time() - t0:.0f}s)")


def cmd_run(a) -> None:
    df, X, meta, folds, lp, grp = load_frame()
    years = [int(y) for y in a.years.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    fast = not a.full_quality
    sfx = "" if fast else "_q"
    allc = list(X.columns)
    cond = grp["COND"]
    todo: list[tuple[str, list[str], str, dict | None]] = []
    seeds_by_name: dict[str, list[int]] = {}
    if a.kind == "full":
        todo.append((f"full{sfx}", allc, "lgb_binary", None))
    elif a.kind == "experts":
        todo.append((f"exp_COND{sfx}", cond, "lgb_binary", None))
        for g, cols in grp.items():
            if g != "COND":
                todo.append((f"exp_{g}{sfx}", cond + cols, "lgb_binary", None))
    elif a.kind == "logo":
        for g, cols in grp.items():
            if g != "COND":
                drop = set(cols)
                todo.append((f"logo_{g}{sfx}", [c for c in allc if c not in drop], "lgb_binary", None))
    elif a.kind == "bags":
        rng = np.random.default_rng(a.bag_seed)
        names = [g for g in grp if g != "COND"]
        for b in range(a.n_bags):
            k = max(2, len(names) // 2)
            pick = list(rng.choice(names, size=k, replace=False))
            cols = cond + [c for g in pick for c in grp[g]]
            todo.append((f"bag{a.bag_seed}_{b}{sfx}", cols, "lgb_binary", None))
            (OUT / "bag_defs.json").parent.mkdir(parents=True, exist_ok=True)
            d = json.loads((OUT / "bag_defs.json").read_text(encoding="utf-8")) if (OUT / "bag_defs.json").exists() else {}
            d[f"bag{a.bag_seed}_{b}{sfx}"] = pick
            (OUT / "bag_defs.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    elif a.kind == "fullseeds":
        # 対照: 全列モデルを別の seed で 2 本（full の seed 0 と合わせて 3 本の seed 平均が、グルーピング型の『多数の平均』の効果の対照になる）
        for sd in (1, 2):
            todo.append((f"full_s{sd}", allc, "lgb_binary", None))
            seeds_by_name[f"full_s{sd}"] = [sd]
    elif a.kind in ("cat", "lgbet", "xgbfull"):
        # V4 の全列（1,062 列）で別系統のメンバーを 3 年分（2022〜24）作り、V4 の 7 メンバーへの追加として評価する（V4 と同じ品質設定）
        years = [2022, 2023, 2024]
        if a.kind == "cat":
            todo.append(("cat_v4", allc, "cat_binary", {"iterations": 700, "depth": 6, "learning_rate": 0.06, "l2_leaf_reg": 5.0, "task_type": "GPU"}))
        elif a.kind == "lgbet":
            q = lgb_params(lp, False)
            q["lgb"]["extra_trees"] = True
            q["lgb"]["feature_fraction"] = 0.5
            todo.append(("lgbet_v4", allc, "lgb_binary", q))
        else:
            todo.append(("xgb_v4full", allc, "xgb_binary", {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "colsample_bytree": 0.5, "subsample": 0.8}))
        seeds = [0, 1, 2] if a.kind != "cat" else [0, 1]
        fast = False
    elif a.kind == "nnexp":
        for g in ("PED", "CONN_OWNER", "CONN_JKTR", "FIELD_MEM", "FIELD_DEMO", "FIELD_CONN", "FIELD_ENV", "FIELD_PACE", "TL"):
            if g in grp:
                todo.append((f"nnexp_{g}", cond + grp[g], "nn", {"hidden": (128, 64), "epochs": 14}))
    else:
        raise SystemExit(a.kind)
    if a.limit:
        todo = todo[: a.limit]
    for name, cols, recipe, prm in todo:
        run_one(name, cols, X, meta, folds, lp, years, seeds_by_name.get(name, seeds), fast, recipe, prm)
        if a.kind in ("cat", "lgbet", "xgbfull") and (OOF / f"{name}.parquet").exists():
            (OUT / "members").mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy(OOF / f"{name}.parquet", OUT / "members" / f"{name}.parquet")


def _load_oofs(names: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for n in names:
        p = OOF / f"{n}.parquet"
        if p.exists():
            o = pd.read_parquet(p)
            o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
            out[n] = o
    return out


def _merge(oofs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = None
    for n, o in oofs.items():
        o = o.rename(columns={f"p_{t_}": f"{n}__{t_}" for t_ in TARGETS})[KEY + ["year"] * (base is None) + [f"{n}__{t_}" for t_ in TARGETS]]
        base = o if base is None else base.merge(o, on=KEY, how="inner")
    return base


def cmd_combine(a) -> None:
    from sklearn.linear_model import LogisticRegression
    from src.pipeline.models import ensemble as en, feature_bakeoff as fb, win_models as wm

    names = sorted(p.stem for p in OOF.glob("*.parquet"))
    fastn = [n for n in names if not n.endswith("_q")]
    oofs = _load_oofs(fastn)
    if "full" not in oofs:
        raise SystemExit("full の OOF がありません")
    M = _merge(oofs)
    meta = M[KEY + ["year"]].copy()
    df, X, mt, folds, lp, grp = None, None, None, None, None, None
    fin = pd.read_parquet(V / "tp27/oof/lgb_extras_ped__v4.parquet")[["race_id", "horse_id"]].astype(str)       # 行の並びの確認用ではなく、finish は train_table から
    import pyarrow.parquet as pq
    ft = pq.read_table(V / "features/train_table.parquet", columns=KEY + ["finish_position"]).to_pandas()
    ft["race_id"], ft["horse_id"] = ft["race_id"].astype(str), ft["horse_id"].astype(str)
    M = M.merge(ft, on=KEY, how="left")
    race, years, finp = M["race_id"].to_numpy(), M["year"].to_numpy(), M["finish_position"].to_numpy()
    Y = {"win": (finp <= 1).astype(float), "top2": (finp <= 2).astype(float), "top3": (finp <= 3).astype(float)}
    TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}
    fin_map = M.set_index(KEY)["finish_position"]

    def logits(prefixes: list[str], t_: str) -> np.ndarray:
        return np.stack([en.logit(M[f"{n}__{t_}"].to_numpy()) for n in prefixes], axis=1)

    def mk(raw: dict) -> pd.DataFrame:
        o = M[KEY + ["year"]].copy()
        for t_ in TARGETS:
            o[f"p_{t_}"] = wm.normalize_to_total(race, np.clip(raw[t_], 1e-9, 1 - 1e-9), TOT[t_])
        return o

    def loyo(Z_by_t: dict, C=1.0) -> dict:
        raw = {}
        for t_ in TARGETS:
            Z = Z_by_t[t_]
            p = np.zeros(len(Z))
            for yr in np.unique(years):
                va = years == yr
                m = LogisticRegression(C=C, max_iter=1000).fit(Z[~va], Y[t_][~va])
                p[va] = m.predict_proba(Z[va])[:, 1]
            raw[t_] = p
        return raw

    base = mk({t_: M[f"full__{t_}"].to_numpy() for t_ in TARGETS})
    cands: dict[str, pd.DataFrame] = {}
    exp_names = [n for n in oofs if n.startswith("exp_") and n != "exp_COND"]
    logo_names = [n for n in oofs if n.startswith("logo_")]
    bag_names = [n for n in oofs if n.startswith("bag")]
    nn_names = [n for n in oofs if n.startswith("nnexp_")]
    for label, pool in (("avg_logo", logo_names), ("avg_bags", bag_names), ("avg_experts", exp_names), ("avg_nnexp", nn_names)):
        if len(pool) >= 2:
            cands[label] = mk({t_: en.sigmoid(logits(pool, t_).mean(1)) for t_ in TARGETS})
            cands[label + "+full"] = mk({t_: en.sigmoid(0.5 * logits(pool, t_).mean(1) + 0.5 * logits(["full"], t_)[:, 0]) for t_ in TARGETS})
    if exp_names:
        cands["stack_experts"] = mk(loyo({t_: logits(["full"] + exp_names, t_) for t_ in TARGETS}, C=0.3))
    if nn_names and exp_names:
        cands["stack_experts_nn"] = mk(loyo({t_: logits(["full"] + exp_names + nn_names, t_) for t_ in TARGETS}, C=0.3))
    # 階層スタッキング: クラスター内で平均 → クラスター間を LR
    cl = {}
    for c, gs in CLUSTERS.items():
        mem = [f"exp_{g}" for g in gs if f"exp_{g}" in oofs]
        if mem:
            cl[c] = mem
    if len(cl) >= 2:
        Zc = {t_: np.column_stack([logits(["full"], t_)[:, 0]] + [logits(m, t_).mean(1) for m in cl.values()]) for t_ in TARGETS}
        cands["hier_stack"] = mk(loyo(Zc, C=0.3))
    if bag_names and exp_names:
        cands["avg_all_groupy"] = mk({t_: en.sigmoid(logits(exp_names + bag_names + logo_names, t_).mean(1)) for t_ in TARGETS})
    ctrl = None
    if "full_s1" in oofs and "full_s2" in oofs:
        ctrl = mk({t_: en.sigmoid(logits(["full", "full_s1", "full_s2"], t_).mean(1)) for t_ in TARGETS})
        cands["avg_fullseeds3(対照)"] = ctrl
    res = {}
    for k, o in cands.items():
        c = fb.compare_oof(base, o, fin_map, k)
        res[k] = {kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in c.items()}
        _log(f"{k}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
        # 単体のメンバーとしても保存（V4 の 7 メンバーへの追加の評価で使う）
        mdir = OUT / "members_2y"
        mdir.mkdir(parents=True, exist_ok=True)
        o.to_parquet(mdir / f"gm_{k.replace('+', '_')}.parquet", index=False)
    if ctrl is not None:        # 対照（全列モデルの seed 3 本平均）との比較: グルーピング型の平均が『モデル数を増やした効果』以上か
        for k, o in cands.items():
            if k.startswith("avg_fullseeds3"):
                continue
            c = fb.compare_oof(ctrl, o, fin_map, k + " vs seed3")
            res[k + " | vs対照"] = {kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in c.items()}
            _log(f"{k} vs 対照(seed3): Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
    (OUT / "gm_combine.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_importance(a) -> None:
    import lightgbm as lgb

    df, X, meta, folds, lp, grp = load_frame()
    f = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) == 2023][0]
    tr = f.train_index
    y = (meta["finish_position"].to_numpy()[tr] <= 1).astype(int)
    prm = lgb_params(lp, True)
    p = dict(objective="binary", num_threads=THREADS, verbose=-1, seed=0, **prm["lgb"])
    b = lgb.train(p, lgb.Dataset(X.iloc[tr], y), num_boost_round=prm["n_rounds"])
    gain = pd.Series(b.feature_importance("gain"), index=X.columns)
    out = {g: float(gain[cols].sum() / gain.sum()) for g, cols in grp.items()}
    (OUT / "gm_importance.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("gain 比: " + json.dumps({k: round(v, 3) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}, ensure_ascii=False))


def cmd_corr(a) -> None:
    from src.pipeline.models import ensemble as en

    oofs = _load_oofs([p.stem for p in OOF.glob("*.parquet") if not p.stem.endswith("_q")])
    M = _merge(oofs)
    Z = pd.DataFrame({n: en.logit(M[f"{n}__win"].to_numpy()) for n in oofs})
    C = Z.corr()
    out = {"vs_full": {n: round(float(C.loc[n, "full"]), 4) for n in Z.columns if n != "full"}, "mean_offdiag": round(float((C.values.sum() - len(C)) / (len(C) * (len(C) - 1))), 4)}
    (OUT / "gm_corr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(json.dumps(out, ensure_ascii=False)[:1500])


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("kind")
    r.add_argument("--years", default="2022,2023")
    r.add_argument("--seeds", default="0")
    r.add_argument("--full-quality", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--n-bags", type=int, default=8)
    r.add_argument("--bag-seed", type=int, default=0)
    for n in ("combine", "importance", "corr"):
        q = sp.add_parser(n)
        q.add_argument("--years", default="2022,2023")
    a = ap.parse_args()
    {"run": cmd_run, "combine": cmd_combine, "importance": cmd_importance, "corr": cmd_corr}[a.cmd](a)


if __name__ == "__main__":
    main()
