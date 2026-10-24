"""tp-25e: tp-25c の補助ヘッド予測（mtl_C）を v3 のアンサンブルに組み込んだ増分の検証。

python -m src.scripts.data.tp25e_eval aux                       # 補助予測の表を作る（A: 2020 年は欠損 / B: 2020 年を時間ブロックの k-fold で補う）
python -m src.scripts.data.tp25e_eval member lgb_extras --aux A # v3 のメンバーに補助列を足して OOF（valid 2022-24）を作る
python -m src.scripts.data.tp25e_eval ens --name v4a --members lgb_extras_ped:A xgb_extras_ped:A ...   # アンサンブル（tp-23 と同じ手順）と v3 比
環境変数: SEEDS=0,1,2  THREADS=3  TP25E_OUT=出力先
ベース: v3 と同じ旧表の特徴量（tp05_data）+ TL_A + 補助列。test 2025 / 2026 年は使わない。市場オッズ・人気系は使わない。
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
WORK = Path(os.environ.get("TP_WORK") or ROOT / "data/local/modeling/training_pipeline")     # v4 の土台で回すときは TP_WORK=training_pipeline_v4
OUT = Path(os.environ.get("TP25E_OUT") or WORK / "tp25e")
C_OUT = Path(os.environ.get("TP25C_OUT") or WORK / "tp25c")
SEEDS = [int(s) for s in (os.environ.get("SEEDS") or "0,1,2").split(",")]
THREADS = int(os.environ.get("THREADS") or 3)
KEY = ["race_id", "horse_id"]
AUX_NAMES = ["t_pos_early", "t_pos_late", "t_l3f_dev", "t_time_dev", "t_rank_norm", "ten_rel_actual", "c1_rel_actual", "r_pace_bal", "r_rpci"]   # tp25c の AUX_C


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────────── 補助予測の表
def rel_cols(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    g = df.groupby("race_id", sort=False)
    out = {}
    for c in names:
        out[c + "_rk"] = g[c].rank(pct=True)
        out[c + "_gap"] = df[c] - g[c].transform("mean")
    return pd.DataFrame(out, index=df.index)


def kfold_2020(df: pd.DataFrame, base: list[str], A: np.ndarray, Y: dict, race: np.ndarray, k: int = 4, gap_days: int = 10) -> np.ndarray:
    """2020 年の行の補助予測を、2020 年を日付順の k ブロックに分けた『他のブロックで学習』で作る（前後 gap_days は学習から外す）。
    2020 年の行は学習側の古い行としてしか使わず、検証年（2022〜）の行には使わないので、リークは検証に影響しない。"""
    from src.pipeline.models.tp25c_mtl import mtl_fit_predict

    dates = pd.to_datetime(df["date"])
    idx = np.flatnonzero(dates.dt.year.to_numpy() == 2020)
    order = idx[np.argsort(dates.to_numpy()[idx], kind="stable")]
    blocks = np.array_split(order, k)
    out = np.full((len(df), A.shape[1]), np.nan, dtype="float32")
    X = df[base]
    for i, va in enumerate(blocks):
        lo, hi = dates.iloc[va].min() - pd.Timedelta(days=gap_days), dates.iloc[va].max() + pd.Timedelta(days=gap_days)
        tr = np.array([j for j in idx if not (lo <= dates.iloc[j] <= hi)])
        acc = np.zeros((len(va), A.shape[1]))
        for seed in SEEDS:
            o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[va], race[va], seed=seed, aux_w=0.5)
            acc += o["aux"] / len(SEEDS)
        out[va] = acc
        _log(f"kfold 2020 block {i + 1}/{k} done (train {len(tr)})")
    return out


def cmd_aux(_a):
    from src.scripts.data import tp25c_eval as c

    OUT.mkdir(parents=True, exist_ok=True)
    df, base, _sx = c.load_frame()
    oof = np.load(C_OUT / "oof_aux_mtl_C.npy")
    names = ["mtl_" + x for x in c.AUX_C]
    assert [x for x in c.AUX_C] == AUX_NAMES
    fin = df["finish_position"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    A = df[c.AUX_C].to_numpy(dtype="float64")
    race = df["race_id"].to_numpy()
    for tag in ("A", "B"):
        o = oof.copy()
        if tag == "B":
            cache = OUT / "oof_aux_2020_kfold.npy"
            if cache.exists():
                k20 = np.load(cache)
            else:
                k20 = kfold_2020(df, base, A, Y, race)
                np.save(cache, k20)
            m = df["year"].to_numpy() == 2020
            o[m] = k20[m]
        t = pd.DataFrame(o, columns=names, index=df.index)
        t = pd.concat([t, rel_cols(pd.concat([df[["race_id"]], t], axis=1), names)], axis=1)
        t.insert(0, "horse_id", df["horse_id"].to_numpy())
        t.insert(0, "race_id", df["race_id"].to_numpy())
        t.to_parquet(OUT / f"aux_{tag}.parquet", index=False)
        cov = {int(y): float(np.isfinite(o[df["year"].to_numpy() == y, 0]).mean()) for y in sorted(df["year"].unique())}
        _log(f"aux_{tag}: {t.shape} coverage_by_year={cov}")


# ───────────────────────────── v3 と同じ入力
def load_v3():
    import json as _j

    from src.pipeline.models import tp05_data, zoo

    D = tp05_data.load_dataset(WORK)
    df, meta, folds = D["df"], D["meta"], D["folds"]
    EP, EXTRAS, _PED = tp05_data.load_extras_ped(df, WORK)
    X1 = df[D["FEATS"]["base+S+C1"]]
    TL = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    TLm = df[KEY].merge(TL, on=KEY, how="left")
    TLm.index = df.index
    TLX = TLm[[c for c in TLm.columns if c not in KEY]].add_prefix("tl_")
    best = _j.loads((WORK / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    P = {"lgb": zoo.tuned_params("lgb", best["lgb"]["params"]), "xgb": zoo.tuned_params("xgb", best["xgb"]["params"]), "nn": zoo.tuned_params("nn", best["nn"]["params"])}
    X_ep = pd.concat([X1, EP], axis=1)
    X_ex = pd.concat([X1, EP[EXTRAS]], axis=1)
    members = {
        "lgb_extras_ped": ("lgb_binary", P["lgb"], X_ep), "xgb_extras_ped": ("xgb_binary", P["xgb"], X_ep), "nn_extras_ped": ("nn", P["nn"], X_ep),
        "lgb_extras": ("lgb_binary", P["lgb"], X_ex), "lgb_softmax": ("lgb_softmax", {}, X1), "linear": ("linear", {"epochs": 14}, X1),
        "nn_s": ("nn", {"hidden": (128, 64), "epochs": 14}, X1),
    }
    return df, meta, folds, members, TLX


def cmd_member(a):
    from src.pipeline.models import zoo_runner

    df, meta, folds, members, TLX = load_v3()
    recipe, params, X = members[a.name]
    X = pd.concat([X, TLX], axis=1)
    tag = a.aux
    if tag != "none":
        aux = pd.read_parquet(OUT / f"aux_{tag}.parquet")
        m = df[KEY].merge(aux, on=KEY, how="left")
        assert len(m) == len(df)
        cols = [c for c in m.columns if c not in KEY]
        if a.aux_cols == "raw":
            cols = [c for c in cols if not c.endswith(("_rk", "_gap"))]
        elif a.aux_cols == "rel":
            cols = [c for c in cols if c.endswith(("_rk", "_gap"))]
        A = m[cols]
        A.index = df.index
        X = pd.concat([X, A], axis=1)
    name = f"{a.name}__aux{tag}" + ("" if a.aux_cols == "all" else f"_{a.aux_cols}")
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    p = OUT / "oof" / f"{name}.parquet"
    if p.exists() and not a.force:
        _log(f"{name} 既存")
        return
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, recipe, X, meta, folds, SEEDS, THREADS, params=dict(params), out_dir=None)
    oof.to_parquet(p, index=False)
    res.assign(member=name).to_csv(OUT / "oof" / f"{name}.csv", index=False)
    _log(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} ({time.time() - t0:.0f}s) cols={X.shape[1]}")


def cmd_final_table(_a):
    """最終アンサンブルの学習行に付ける補助予測の表: 2020 年=時間ブロック k-fold、2021〜2024 年=expanding OOF（tp-25c）、2025 年=2024 年までで学習した OOF。"""
    aux = pd.read_parquet(OUT / "aux_B.parquet")
    a25 = pd.read_parquet(WORK / "final_aux/aux_oof_2025.parquet")
    k = aux.merge(a25[KEY].assign(_is25=1), on=KEY, how="left")
    keep = aux[k["_is25"].isna().to_numpy()]
    t = pd.concat([keep, a25[aux.columns]], ignore_index=True)
    t.to_parquet(WORK / "final_aux/aux_train.parquet", index=False)
    _log(f"final_aux/aux_train.parquet: {t.shape} 欠損率={float(t['mtl_t_pos_early'].isna().mean()):.4f}")


def cmd_mtl(a):
    """v3 の NN メンバーと同じ入力(+TL)で、補助タスクつき NN（mtl_C 相当）の OOF を作る（NN メンバーの置き換え/追加用）。"""
    import torch

    from src.pipeline.models import zoo
    from src.pipeline.models.tp25c_mtl import mtl_fit_predict

    torch.set_num_threads(THREADS)
    df, meta, folds, members, TLX = load_v3()
    _recipe, params, X = members[a.base]
    X = pd.concat([X, TLX], axis=1)
    ex = pd.read_parquet(C_OUT / "frame_extra.parquet")
    A = df[KEY].merge(ex[KEY + AUX_NAMES], on=KEY, how="left")[AUX_NAMES].to_numpy(dtype="float64")
    race, fin = meta["race_id"].to_numpy(), meta["finish_position"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    kw = {k: params[k] for k in ("hidden", "epochs", "lr", "wd", "dropout") if k in params}
    sums = {k: np.zeros(len(meta)) for k in ("win", "top2", "top3")}
    covered = np.zeros(len(meta), bool)
    t0 = time.time()
    for f in folds:
        tr, va = f.train_index, f.valid_index
        if len(tr) < 1000:
            continue
        for seed in SEEDS:
            o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[va], race[va], seed=seed, aux_w=a.aux_w, **kw)
            fp = zoo.finalize(race[va], {"win": __import__("src.pipeline.models.win_models", fromlist=["x"]).softmax_by_race(race[va], o["win_score"]), "top2": zoo.sigmoid(o["top2_logit"]), "top3": zoo.sigmoid(o["top3_logit"])})
            for k in sums:
                sums[k][va] += fp[k]
        covered[va] = True
        _log(f"mtl {a.base} fold {f.fold_id} done")
    idx = np.flatnonzero(covered)
    oof = pd.DataFrame({"race_id": race[idx], "horse_id": meta["horse_id"].to_numpy()[idx], **{f"p_{k}": sums[k][idx] / len(SEEDS) for k in sums}})
    name = f"mtl_{a.base}_w{a.aux_w}"
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUT / "oof" / f"{name}__auxmtl.parquet", index=False)
    from src.pipeline.models import win_models as wm

    yy = {t: (fin[idx] <= k).astype(float) for t, k in (("win", 1), ("top2", 2), ("top3", 3))}
    _log(f"{name}: " + " ".join(f"{t} {wm.binary_logloss(oof[f'p_{t}'].to_numpy(), yy[t]):.5f}" for t in yy) + f" ({time.time() - t0:.0f}s) kw={kw}")


# ───────────────────────────── アンサンブル
def _load_oof(spec: str) -> tuple[str, pd.DataFrame]:
    """spec: 'lgb_extras_ped:A' → tp25e の OOF / 'lgb_extras_ped:v3' → v3 の OOF（tl3）/ 'mtl_C:nn' → tp25c の mtl_C 実行結果。"""
    name, tag = spec.split(":")
    if tag == "v3":
        return name, pd.read_parquet(WORK / "oof/model_oof" / f"{name}_tl3.parquet")
    if tag == "nn":
        return name, mtl_oof(name)
    if tag == "mtl":
        return name, pd.read_parquet(OUT / "oof" / f"{name}__auxmtl.parquet")
    return f"{name}__{tag}", pd.read_parquet(OUT / "oof" / f"{name}__aux{tag}.parquet")


def mtl_oof(variant: str) -> pd.DataFrame:
    """tp25c の runs/*.npz（修正後の表の行順）から OOF（seed 平均）を作る。"""
    from src.pipeline.models.time_split import yearly_expanding_folds
    from src.scripts.data import tp25c_eval as c

    df, _b, _s = c.load_frame()
    folds = list(yearly_expanding_folds(df))
    r = c._assemble(df, "mtl_C" if variant.startswith("mtl_C") else variant, folds)
    idx, w, t2, t3, _ax = r
    return pd.DataFrame({"race_id": df["race_id"].to_numpy()[idx], "horse_id": df["horse_id"].to_numpy()[idx], "p_win": w, "p_top2": t2, "p_top3": t3})


def cmd_ens(a):
    from src.pipeline.models import ensemble as en
    from src.pipeline.models import feature_bakeoff as fb
    from src.pipeline.models import tp05_data

    D = tp05_data.load_dataset(WORK)
    meta = D["meta"]
    base = meta[KEY + ["finish_position", "year"]].copy()
    names = []
    for s in a.members:
        n, o = _load_oof(s)
        names.append(n)
        o = o[KEY + ["p_win", "p_top2", "p_top3"]].copy()
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        base = base.merge(o.rename(columns={f"p_{t}": f"{n}__{t}" for t in ("win", "top2", "top3")}), on=KEY, how="inner")
    race, years, fin = base["race_id"].to_numpy(), base["year"].to_numpy(), base["finish_position"].to_numpy()
    TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}
    Y = {"win": (fin == 1).astype(float), "top2": (fin <= 2).astype(float), "top3": (fin <= 3).astype(float)}
    METHOD = {"win": "weighted_logit", "top2": "greedy", "top3": "greedy"}
    ENS = base[KEY + ["year"]].copy()
    SPEC = {}
    for t in TOT:
        P = np.stack([base[f"{n}__{t}"].to_numpy() for n in names], axis=1)
        llv, pred = en.loyo_eval(METHOD[t], P, years, race, Y[t], TOT[t])
        ENS[f"p_{t}"] = pred
        st = en.fit_method(METHOD[t], P, en.logit(P), Y[t])
        SPEC[t] = {"method": METHOD[t], "weights": dict(zip(names, [round(float(x), 4) for x in st]))}
        _log(f"{a.name} {t} LOYO {llv:.5f} 重み {SPEC[t]['weights']}")
    (OUT / "ens").mkdir(parents=True, exist_ok=True)
    ENS.to_parquet(OUT / "ens" / f"{a.name}.parquet", index=False)
    fin_map = meta.set_index(KEY)["finish_position"]
    v3 = pd.read_parquet(WORK / "oof/model_oof/ensemble_v3.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
    v3 = v3.merge(ENS[KEY], on=KEY, how="inner")
    cmp_ = fb.compare_oof(v3, ENS[KEY + ["p_win", "p_top2", "p_top3"]], fin_map, f"{a.name} vs v3")
    cmp_["spec"] = SPEC
    (OUT / "ens" / f"{a.name}.json").write_text(json.dumps(cmp_, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in cmp_.items() if k != "spec"})


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("aux")
    m = sp.add_parser("member")
    m.add_argument("name")
    m.add_argument("--aux", default="A", choices=["A", "B", "none"])
    m.add_argument("--aux-cols", default="all", choices=["all", "raw", "rel"])
    m.add_argument("--force", action="store_true")
    sp.add_parser("final-table")
    mt = sp.add_parser("mtl")
    mt.add_argument("base", choices=["nn_extras_ped", "nn_s"])
    mt.add_argument("--aux-w", type=float, default=0.5)
    e = sp.add_parser("ens")
    e.add_argument("--name", required=True)
    e.add_argument("--members", nargs="+", required=True)
    a = ap.parse_args()
    {"aux": cmd_aux, "member": cmd_member, "mtl": cmd_mtl, "final-table": cmd_final_table, "ens": cmd_ens}[a.cmd](a)


if __name__ == "__main__":
    main()
