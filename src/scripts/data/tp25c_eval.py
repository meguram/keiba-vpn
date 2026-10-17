"""tp-25c マルチタスク NN の PoC（補助タスク = テン1F・位置取り・上がり・走破タイム・ペース）。

python -m src.scripts.data.tp25c_eval prepare                      # 補助の目的変数と SmartRC 入力を付けた表を作る（キャッシュ）
python -m src.scripts.data.tp25c_eval run --variants nn_base mtl_B  # fold(2022-24) × seed で学習（1 本ごとに保存。再開可）
python -m src.scripts.data.tp25c_eval summary                       # 勝率/連対/複勝 LogLoss と基準 nn_base との paired bootstrap
python -m src.scripts.data.tp25c_eval oof --variant mtl_B           # C3: 補助ヘッドの OOF 予測（2021-2024）を LightGBM に足す
環境変数: SEEDS=0,1,2  THREADS=4  TP25C_OUT=出力先
ベース: 修正後の学習表（training_pipeline_fixed）の tp-03 採用 175 列（V0）。valid 2022・2023・2024、purge 7 日 + embargo 3 日。test 2025 / 2026 年は使わない。
市場オッズ・人気系の列は使わない。
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
TP = ROOT / "data/local/modeling/training_pipeline"
FIX = Path(os.environ.get("TP_BASE_DIR") or ROOT / "data/local/modeling/training_pipeline_fixed")   # v4 の土台で回すときは TP_BASE_DIR=training_pipeline_v4
SX = TP / "smartrc_ten1f"
OUT = Path(os.environ.get("TP25C_OUT") or TP / "tp25c")
SEEDS = [int(s) for s in (os.environ.get("SEEDS") or "0,1,2").split(",")]
THREADS = int(os.environ.get("THREADS") or 4)

AUX_A = ["t_pos_early", "t_pos_late", "t_l3f_dev", "t_time_dev", "t_rank_norm"]
AUX_B = AUX_A + ["ten_rel_actual", "c1_rel_actual"]
AUX_C = AUX_B + ["r_pace_bal", "r_rpci"]
AUX_D = ["t_pos_early", "t_pos_late", "ten_rel_actual", "c1_rel_actual", "r_pace_bal", "r_rpci"]      # 位置取り・ペースだけ（上がり/タイム/着順を外す）

VARIANTS: dict[str, dict] = {
    "nn_base": dict(zoo=True, inputs="base"),
    "nn_sx": dict(zoo=True, inputs="sx"),
    "mtl_A": dict(aux=AUX_A, w=0.5, inputs="base"),
    "mtl_B": dict(aux=AUX_B, w=0.5, inputs="base"),
    "mtl_C": dict(aux=AUX_C, w=0.5, inputs="base"),
    "mtl_D": dict(aux=AUX_D, w=0.5, inputs="base"),
    "mtl_B_w02": dict(aux=AUX_B, w=0.2, inputs="base"),
    "mtl_B_w1": dict(aux=AUX_B, w=1.0, inputs="base"),
    "mtl_B_sx": dict(aux=AUX_B, w=0.5, inputs="sx"),
    "mtl_C_sx": dict(aux=AUX_C, w=0.5, inputs="sx"),
    "mtl_B_int": dict(aux=AUX_B, w=0.5, inputs="base", interact=True),
    "mtl_C_int": dict(aux=AUX_C, w=0.5, inputs="base", interact=True),
    "mtl_C_int_sx": dict(aux=AUX_C, w=0.5, inputs="sx", interact=True),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────────── データ
def base_columns() -> list[str]:
    a = json.loads((TP / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    return list(a["b0_features"]) + list(a["features"])


def load_frame():
    """修正後の学習表 + 補助の目的変数 + SmartRC の as-of 特徴。行は tp-21 と同じ（学習対象年・着順>0）。"""
    from src.pipeline.features import smartrc_ten1f as S
    from src.pipeline.models import ablation as ab
    from src.pipeline.models import stage_models as sm
    from src.pipeline.models.time_split import restrict_to_training_years

    cache = OUT / "frame_extra.parquet"
    df = ab.load_table(FIX / "features/train_table.parquet")
    df = restrict_to_training_years(df).reset_index(drop=True)
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    if cache.exists():
        ex = pd.read_parquet(cache)
        assert (ex["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (ex["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
    else:
        labels = pd.read_parquet(TP / "oof/stage_labels.parquet")
        empty = pd.DataFrame({"race_id": [], "pace_bal": [], "rpci": [], "pace_bal_cls": [], "rpci_cls": []})
        t = sm.build_targets(df, labels, empty)
        ex = pd.DataFrame({"race_id": df["race_id"], "horse_id": df["horse_id"]})
        for c in ("t_pos_early", "t_pos_late", "t_l3f_dev", "t_time_dev", "t_rank_norm"):
            ex[c] = t[c].to_numpy()
        lab = labels[~labels["is_hurdle"]].drop_duplicates("race_id").set_index("race_id")
        for src, dst in (("pace_bal", "r_pace_bal"), ("rpci", "r_rpci")):
            ex[dst] = df["race_id"].map(lab[src]).to_numpy()
        runners = pd.read_parquet(SX / "runners.parquet")
        career = pd.read_parquet(SX / "career.parquet")
        ent = df[["race_id", "horse_id", "date", "distance", "surface"]].copy()
        ent["surface"] = ent["surface"].astype(str)
        feats = S.build_features(ent, career, runners)
        assert (feats["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (feats["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
        for c in S.CORE + S.RELATIVE + S.RUNNER_RANKS:
            ex[c] = feats[c].to_numpy()
        tg = S.race_targets(df[["race_id", "horse_id", "date"]], career)
        for c in ("ten_rel_actual", "c1_rel_actual"):
            ex[c] = tg[c].to_numpy()
        OUT.mkdir(parents=True, exist_ok=True)
        ex.to_parquet(cache, index=False)
    for c in ex.columns:
        if c not in ("race_id", "horse_id"):
            df[c] = ex[c].to_numpy()
    sx_cols = [c for c in ex.columns if c.startswith("sx_")]
    return df, base_columns(), sx_cols


# ───────────────────────────── 学習
def _finalize(race, out):
    from src.pipeline.models import win_models as wm
    from src.pipeline.models import zoo

    o = {"win": wm.softmax_by_race(race, out["win_score"]), "top2": zoo.sigmoid(out["top2_logit"]), "top3": zoo.sigmoid(out["top3_logit"])}
    return zoo.finalize(race, o)


def cmd_prepare(_a):
    df, base, sx = load_frame()
    cov = {c: float(df[c].notna().mean()) for c in AUX_C + ["sx_ten_mean3"]}
    _log(f"frame {df.shape} base={len(base)} sx={len(sx)} coverage={cov}")
    (OUT / "prepare_meta.json").write_text(json.dumps({"rows": len(df), "base": len(base), "sx": len(sx), "coverage": cov}, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_run(a):
    import torch

    from src.pipeline.models import zoo
    from src.pipeline.models.tp25c_mtl import mtl_fit_predict
    from src.pipeline.models.time_split import yearly_expanding_folds

    torch.set_num_threads(THREADS)
    df, base, sx = load_frame()
    folds = list(yearly_expanding_folds(df))
    race = df["race_id"].to_numpy()
    fin = df["finish_position"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for name in a.variants:
        v = VARIANTS[name]
        cols = base + (sx if v["inputs"] == "sx" else [])
        X = df[cols]
        A = df[v["aux"]].to_numpy(dtype="float64") if v.get("aux") else None
        for f in folds:
            tr, va = f.train_index, f.valid_index
            for seed in SEEDS:
                p = OUT / "runs" / f"{name}__{f.fold_id}__s{seed}.npz"
                if p.exists():
                    continue
                t0 = time.time()
                ytr = {k: x[tr] for k, x in Y.items()}
                if v.get("zoo"):
                    o = zoo.fit_predict("nn", X.iloc[tr], ytr, race[tr], X.iloc[va], race[va], THREADS, seed=seed, params={"hidden": (128, 64), "epochs": 14})
                    fin_ = zoo.finalize(race[va], o)
                    np.savez(p, win=fin_["win"], top2=fin_["top2"], top3=fin_["top3"], va=va)
                else:
                    o = mtl_fit_predict(X.iloc[tr], ytr, None if A is None else A[tr], race[tr], X.iloc[va], race[va], seed=seed,
                                        aux_w=v.get("w", 0.5), interact=v.get("interact", False))
                    fin_ = _finalize(race[va], o)
                    np.savez(p, win=fin_["win"], top2=fin_["top2"], top3=fin_["top3"], va=va, aux=o.get("aux", np.zeros((len(va), 0))))
                _log(f"{name} {f.fold_id} seed{seed} done {time.time() - t0:.0f}s")


def _assemble(df, name, folds):
    """seed 平均した OOF（valid 3 年分）。戻り値: (行 index, win, top2, top3, aux)。"""
    idx, W, T2, T3, AX = [], [], [], [], []
    for f in folds:
        fs = [OUT / "runs" / f"{name}__{f.fold_id}__s{s}.npz" for s in SEEDS]
        if not all(p.exists() for p in fs):
            return None
        z = [np.load(p) for p in fs]
        idx.append(z[0]["va"])
        W.append(np.mean([x["win"] for x in z], axis=0))
        T2.append(np.mean([x["top2"] for x in z], axis=0))
        T3.append(np.mean([x["top3"] for x in z], axis=0))
        AX.append(np.mean([x["aux"] if "aux" in x.files else np.zeros((len(x["va"]), 0)) for x in z], axis=0))
    return np.concatenate(idx), np.concatenate(W), np.concatenate(T2), np.concatenate(T3), np.concatenate(AX)


def cmd_summary(_a):
    from src.pipeline.models import win_models as wm
    from src.pipeline.models.time_split import yearly_expanding_folds

    df, _b, _s = load_frame()
    folds = list(yearly_expanding_folds(df))
    race, fin = df["race_id"].to_numpy(), df["finish_position"].to_numpy()
    res, losses = {}, {}
    for name in VARIANTS:
        r = _assemble(df, name, folds)
        if r is None:
            continue
        idx, w, t2, t3, _ax = r
        yr = df["year"].to_numpy()[idx]
        d = {"n": int(len(idx))}
        for tname, p, y in (("win", w, fin[idx] == 1), ("top2", t2, fin[idx] <= 2), ("top3", t3, fin[idx] <= 3)):
            y = y.astype(float)
            d[tname] = wm.binary_logloss(p, y)
            d[f"{tname}_by_year"] = {int(Y): wm.binary_logloss(p[yr == Y], y[yr == Y]) for Y in sorted(set(yr))}
            losses[(name, tname)] = wm.per_race_loss(race[idx], p, y)
        res[name] = d
    if "nn_base" in res:
        for name in list(res):
            if name == "nn_base":
                continue
            for tname in ("win", "top2", "top3"):
                bd = wm.paired_bootstrap_diff(losses[("nn_base", tname)], losses[(name, tname)], n_boot=1000)
                res[name][f"vs_base_{tname}"] = {k: round(v, 6) if isinstance(v, float) else v for k, v in bd.items()}
            res[name]["adopt_vs_nn_base"] = bool(res[name]["vs_base_win"]["hi"] < 0 and res[name]["vs_base_win"]["diff"] <= -0.0002
                                                 and res[name]["vs_base_top2"]["diff"] <= 0.0002 and res[name]["vs_base_top3"]["diff"] <= 0.0002)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = []
    for n, d in res.items():
        rows.append([n, round(d["win"], 5), round(d["top2"], 5), round(d["top3"], 5)] + ([round(d["vs_base_win"]["diff"], 5), round(d["vs_base_win"]["hi"], 5), d["adopt_vs_nn_base"]] if "vs_base_win" in d else ["", "", ""]))
    print(pd.DataFrame(rows, columns=["variant", "win", "top2", "top3", "Δwin", "hi95", "adopt"]).to_string(index=False))


# ───────────────────────────── C3: 補助ヘッドの OOF 予測を LightGBM に足す
def _rel_cols(df, aux_names):
    g = df.groupby("race_id", sort=False)
    out = {}
    for c in aux_names:
        x = df[c]
        out[c + "_rk"] = g[c].rank(pct=True)
        out[c + "_gap"] = x - g[c].transform("mean")
    return pd.DataFrame(out, index=df.index)


def cmd_oof(a):
    import torch

    from src.pipeline.models import ablation as ab
    from src.pipeline.models.tp25c_mtl import mtl_fit_predict
    from src.pipeline.models.time_split import yearly_expanding_folds

    torch.set_num_threads(THREADS)
    df, base, sx = load_frame()
    v = VARIANTS[a.variant]
    cols = base + (sx if v["inputs"] == "sx" else [])
    X = df[cols]
    A = df[v["aux"]].to_numpy(dtype="float64")
    race, fin, dates, years = df["race_id"].to_numpy(), df["finish_position"].to_numpy(), df["date"], df["year"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    k = len(v["aux"])
    cache = OUT / f"oof_aux_{a.variant}.npy"
    if cache.exists():
        oof = np.load(cache)
    else:
        oof = np.full((len(df), k), np.nan, dtype="float32")
        gap = pd.Timedelta(days=10)
        for Yr in (2021, 2022, 2023, 2024):
            tr = np.flatnonzero((years >= 2020) & (dates < pd.Timestamp(f"{Yr}-01-01") - gap))
            va = np.flatnonzero(years == Yr)
            acc = np.zeros((len(va), k))
            for seed in SEEDS:
                o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[va], race[va], seed=seed, aux_w=v.get("w", 0.5), interact=v.get("interact", False))
                acc += o["aux"] / len(SEEDS)
            oof[va] = acc
            _log(f"oof aux {Yr} done")
        np.save(cache, oof)
    names = ["mtl_" + c for c in v["aux"]]
    if getattr(a, "oof_only", False):
        _log(f"oof only: {cache}")
        return
    for j, c in enumerate(names):
        df[c] = oof[:, j]
    rel = _rel_cols(df, names)
    for c in rel.columns:
        df[c] = rel[c].to_numpy()
    folds = list(yearly_expanding_folds(df))
    specs = [ab.Spec("LGB_V0", base), ab.Spec("LGB_V0+AUX_OOF", base + names), ab.Spec("LGB_V0+AUX_OOF+REL", base + names + list(rel.columns))]
    if a.with_sx:
        specs.append(ab.Spec("LGB_V0+SX", base + sx))
    if a.stage_only:
        st = pd.read_parquet(Path(os.environ.get("TP_WORK") or TP) / "oof/stage_features_expanding.parquet")[["race_id", "horse_id", "s_time_pred", "s_time_gap", "s_rank_pred"]]
        st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
        m = df[["race_id", "horse_id"]].merge(st, on=["race_id", "horse_id"], how="left")
        assert len(m) == len(df)
        for c in ("s_time_pred", "s_time_gap", "s_rank_pred"):
            df[c] = m[c].to_numpy()
        S3 = ["s_time_pred", "s_time_gap", "s_rank_pred"]
        specs = [ab.Spec("LGB_V0+S", base + S3), ab.Spec("LGB_V0+S+AUX_OOF+REL", base + S3 + names + list(rel.columns))]
    rows = []
    for sp in specs:
        rows += ab.evaluate(df, folds, sp, SEEDS[:2], THREADS, targets=("win", "top2", "top3"))
        _log(f"lgb {sp.name} done")
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"oof_lgb_{a.variant}{'_stage' if a.stage_only else ''}.csv", index=False)
    for t in ("win", "top2", "top3"):
        s = ab.summarize(r, t)
        print(t, "\n", s.round(5).to_string())


def cmd_lgbci(a):
    """LightGBM（V0 固定条件）で、補助 OOF 予測を足した効果をレース単位の paired bootstrap で測る。S = tp-04 の前段 3 列。"""
    import lightgbm as lgb

    from src.pipeline.models import ablation as ab
    from src.pipeline.models import win_models as wm
    from src.pipeline.models.time_split import yearly_expanding_folds

    df, base, _sx = load_frame()
    v = VARIANTS[a.variant]
    oof = np.load(OUT / f"oof_aux_{a.variant}.npy")
    names = ["mtl_" + c for c in v["aux"]]
    for j, c in enumerate(names):
        df[c] = oof[:, j]
    rel = _rel_cols(df, names)
    for c in rel.columns:
        df[c] = rel[c].to_numpy()
    st = pd.read_parquet(TP / "oof/stage_features_expanding.parquet")[["race_id", "horse_id", "s_time_pred", "s_time_gap", "s_rank_pred"]]
    st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
    m = df[["race_id", "horse_id"]].merge(st, on=["race_id", "horse_id"], how="left")
    S3 = ["s_time_pred", "s_time_gap", "s_rank_pred"]
    for c in S3:
        df[c] = m[c].to_numpy()
    AR = names + list(rel.columns)
    specs = {"V0": base, "V0+AUX": base + AR, "V0+S": base + S3, "V0+S+AUX": base + S3 + AR}
    folds = list(yearly_expanding_folds(df))
    race, fin = df["race_id"].to_numpy(), df["finish_position"].to_numpy()
    P = {k: {t: np.zeros(len(df)) for t in ("win", "top2", "top3")} for k in specs}
    seeds = SEEDS[:2]
    for k, cols in specs.items():
        cats = [c for c in cols if str(df[c].dtype) == "category"]
        for f in folds:
            tr, va = f.train_index, f.valid_index
            for t, y in (("win", fin == 1), ("top2", fin <= 2), ("top3", fin <= 3)):
                ds = lgb.Dataset(df.iloc[tr][cols], label=y[tr].astype(int), categorical_feature=cats or "auto", free_raw_data=False)
                for sd in seeds:
                    b = lgb.train(dict(ab.LGB_PARAMS, seed=sd, num_threads=THREADS), ds, num_boost_round=ab.N_ROUNDS)
                    P[k][t][va] += b.predict(df.iloc[va][cols]) / len(seeds)
        _log(f"lgb {k} done")
    va_all = np.concatenate([f.valid_index for f in folds])
    res = {}
    for k in specs:
        res[k] = {}
        for t, y in (("win", fin == 1), ("top2", fin <= 2), ("top3", fin <= 3)):
            res[k][t] = wm.binary_logloss(P[k][t][va_all], y[va_all].astype(float))
    def cmp(b, c):
        out = {}
        for t, y in (("win", fin == 1), ("top2", fin <= 2), ("top3", fin <= 3)):
            la = wm.per_race_loss(race[va_all], P[b][t][va_all], y[va_all].astype(float))
            lb = wm.per_race_loss(race[va_all], P[c][t][va_all], y[va_all].astype(float))
            out[t] = {k_: (round(x, 6) if isinstance(x, float) else x) for k_, x in wm.paired_bootstrap_diff(la, lb, n_boot=1000).items()}
        return out
    res["V0+AUX_vs_V0"] = cmp("V0", "V0+AUX")
    res["V0+S_vs_V0"] = cmp("V0", "V0+S")
    res["V0+S+AUX_vs_V0+S"] = cmp("V0+S", "V0+S+AUX")
    (OUT / f"lgbci_{a.variant}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    r = sp.add_parser("run")
    r.add_argument("--variants", nargs="+", required=True)
    sp.add_parser("summary")
    o = sp.add_parser("oof")
    o.add_argument("--variant", required=True)
    o.add_argument("--with-sx", action="store_true")
    o.add_argument("--stage-only", action="store_true")
    o.add_argument("--oof-only", action="store_true", help="補助予測の OOF だけ保存して終える（LightGBM の評価はしない）")
    c = sp.add_parser("lgbci")
    c.add_argument("--variant", required=True)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summary": cmd_summary, "oof": cmd_oof, "lgbci": cmd_lgbci}[a.cmd](a)


if __name__ == "__main__":
    main()
