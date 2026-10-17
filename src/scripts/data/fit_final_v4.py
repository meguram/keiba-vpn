"""アンサンブル v4 の最終モデルを学習して、推論用に保存する（tp-07）。

python -m src.scripts.data.fit_final_v4 --fit-end 2024     # 2020〜2024 で学習 → bundle_v4_2024 と test 2025 の予測（一度だけ）
python -m src.scripts.data.fit_final_v4 --fit-end 2025     # 2020〜2025 の全行で学習 → bundle_v4_2025（本番用。2026 年は使わない）
出力: data/local/modeling/training_pipeline/final_v4/bundle_v4_{fit_end}/、評価は final_eval_v4/

土台は training_pipeline_v4（先読み修正・血統キー・course_variant 反映後の学習表と S3）。特徴量の群は tp-27 の V4（馬主・補助列・G1 8 群・G3 7 群・PED_sire_cond）。
メンバー: v4_e2（lgb_extras_ped, xgb_extras_ped, lgb_extras, lgb_softmax, linear, mtl_nn_extras_ped, mtl_nn_s）。NN には追加列を重要度上位 100 列に絞って渡す。
アンサンブルの重みは tp27_eval ens（LOYO、valid 2022〜24）の結果（TP27_OUT/ens/<name>.json）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
for k, v in {"TP_BASE_DIR": V4, "TP_WORK": V4, "TP25C_OUT": V4 / "tp25c", "TP25E_OUT": V4 / "tp25e", "TP27_FAC_FIELD": V4 / "factory_field", "TP27_OUT": V4 / "tp27"}.items():
    os.environ.setdefault(k, str(v))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.models import ensemble_bundle as eb  # noqa: E402
from src.pipeline.models import feature_bakeoff as fb  # noqa: E402
from src.pipeline.models import zoo  # noqa: E402
from src.pipeline.models.compute import detect_compute  # noqa: E402
from src.scripts.data import tp25e_eval as e  # noqa: E402
from src.scripts.data import tp27_eval as t27  # noqa: E402

KEY = ["race_id", "horse_id"]
OUT = OLD / "final_v4"
EVAL = OLD / "final_eval_v4"
MEMBERS = [  # (名前, recipe, v3 のメンバー名, 追加列の与え方)
    ("lgb_extras_ped", "lgb_binary", "lgb_extras_ped", "full+aux"),
    ("xgb_extras_ped", "xgb_binary", "xgb_extras_ped", "full+aux"),
    ("lgb_extras", "lgb_binary", "lgb_extras", "full+aux"),
    ("lgb_softmax", "lgb_softmax", "lgb_softmax", "full+aux"),
    ("linear", "linear", "linear", "full+aux"),
    ("mtl_nn_extras_ped", "mtl_nn", "nn_extras_ped", "top100"),
    ("mtl_nn_s", "mtl_nn", "nn_s", "top100"),
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dedup(X: pd.DataFrame) -> pd.DataFrame:
    return X.loc[:, ~X.columns.duplicated()]


def build_matrices():
    """学習・予測に使う全行の特徴量。戻り値: df, meta, {メンバー名: (recipe, params, X)}, aux 目的変数 A（行 × 9）, 補助列 A_feat。"""
    df, meta, _folds, v3members, TLX = e.load_v3()
    raw = t27._raw_keys()
    aux = pd.read_parquet(V4 / "final_aux/aux_train.parquet")
    aux["race_id"], aux["horse_id"] = aux["race_id"].astype(str), aux["horse_id"].astype(str)
    m = df[KEY].merge(aux, on=KEY, how="left")
    A_feat = m[[c for c in m.columns if c not in KEY]]
    A_feat.index = df.index
    E_full = t27._extra_for_ensemble(df, "core", None)
    E100 = t27._extra_for_ensemble(df, "core", t27.NN_TOP)
    out = {}
    for name, recipe, v3name, how in MEMBERS:
        _r, params, X = v3members[v3name]
        if how == "full+aux":
            Xm = pd.concat([X, TLX, A_feat, E_full], axis=1)
        else:
            Xm = pd.concat([X, TLX, E100], axis=1)
        p = dict(params)
        if recipe == "mtl_nn":
            p = {k: p[k] for k in ("hidden", "epochs", "lr", "wd", "dropout") if k in p} | {"aux_w": 0.5}
        out[name] = (recipe, p, _dedup(Xm))
    ex = pd.read_parquet(V4 / "tp25c/frame_extra.parquet")
    ex["race_id"], ex["horse_id"] = ex["race_id"].astype(str), ex["horse_id"].astype(str)
    A = df[KEY].merge(ex[KEY + e.AUX_NAMES], on=KEY, how="left")[e.AUX_NAMES].to_numpy(dtype="float64")
    return df, meta, out, A, A_feat


def load_spec(names: list[str], ens_name: str) -> dict:
    j = json.loads((V4 / "tp27/ens" / f"{ens_name}.json").read_text(encoding="utf-8"))["spec"]
    suf = "__v4"
    targets = {}
    for t, s in j.items():
        w = s["weights"]
        targets[t] = {"method": s["method"], "weights": [float(w.get(n + suf, 0.0)) for n in names]}
    return {"targets": targets, "models": names, "source": f"tp27 ens {ens_name} (LOYO)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ens", default="v4_e2")
    a = ap.parse_args()
    t0 = time.time()
    df, meta, mats, A, _A_feat = build_matrices()
    names = [m[0] for m in MEMBERS]
    cols_union: list[str] = []
    for n in names:
        for c in mats[n][2].columns:
            if c not in cols_union:
                cols_union.append(c)
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    members = [eb.MemberSpec(n, mats[n][0], mats[n][1], list(mats[n][2].columns)) for n in names]
    spec = load_spec(names, a.ens)
    fin = meta["finish_position"].to_numpy()
    dates = pd.to_datetime(meta["date"])
    if a.fit_end == 2024:
        tr = np.flatnonzero(dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10))
    else:
        tr = np.flatnonzero((meta["year"].to_numpy() <= 2025) & np.isfinite(fin.astype(float)))
    race = meta["race_id"].to_numpy()
    y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    ytr = {k: v[tr] for k, v in y.items()}
    o = pd.read_parquet(V4 / "tp27/oof/lgb_softmax__v4.parquet").merge(meta[KEY + ["finish_position"]], on=KEY)
    gam = (zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 2), zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 3))
    bundle = eb.EnsembleBundle(members, spec, {"lgb_softmax": gam})
    threads = detect_compute(reserve_cores=16).threads
    _log(f"学習 {len(tr):,} 行（{dates.iloc[tr].min():%Y-%m-%d}〜{dates.iloc[tr].max():%Y-%m-%d}） / 列 {X_all.shape[1]} / seed {a.seeds} / threads {threads} / メンバー {names}")
    bundle.fit(X_all.iloc[tr], ytr, race[tr], threads, list(range(a.seeds)), aux=A[tr])
    _log(f"学習完了 {time.time() - t0:.0f}s")
    out = OUT / f"bundle_v4_{a.fit_end}"
    bundle.save(out)
    (out / "fit_info.json").write_text(json.dumps({"fit_end": a.fit_end, "train_rows": int(len(tr)), "seeds": a.seeds, "n_features": int(X_all.shape[1]), "members": names,
                                                  "ens": a.ens, "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"保存: {out}")
    if a.fit_end == 2024:
        va = np.flatnonzero((meta["year"].to_numpy() == 2025) & np.isfinite(fin.astype(float)))
        pred = bundle.predict(X_all.iloc[va], race[va])
        P = pd.DataFrame({"race_id": race[va], "horse_id": meta["horse_id"].to_numpy()[va], "finish_position": fin[va]}).join(pred.reset_index(drop=True))
        EVAL.mkdir(parents=True, exist_ok=True)
        P.to_parquet(EVAL / "test2025_predictions.parquet", index=False)
        v3 = pd.read_parquet(OLD / "final_eval_v3/test2025_predictions.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
        fin_map = P.set_index(KEY)["finish_position"]
        mm = P[KEY].merge(v3[KEY], on=KEY, how="inner")
        cmp_ = fb.compare_oof(v3.merge(mm, on=KEY), P[KEY + ["p_win", "p_top2", "p_top3"]].merge(mm, on=KEY), fin_map, "v4 vs v3 (test 2025)")
        ll = {}
        for t_, k in (("win", 1), ("top2", 2), ("top3", 3)):
            yy = (P["finish_position"].to_numpy() <= k).astype(float)
            p = P[f"p_{t_}"].to_numpy()
            ll[t_] = float(-np.mean(yy * np.log(np.clip(p, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-9, 1))))
        rep = {"rows": int(len(P)), "races": int(P["race_id"].nunique()), "logloss": ll, "vs_v3": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cmp_.items()},
               "reference": {"v1_win": 0.21349, "v2_win": 0.21240, "v3_win": 0.21039}}
        (EVAL / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        (EVAL / "TEST2025_EVALUATED.json").write_text(json.dumps({"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": "test 2025 を評価済み（tp-06d、v4）。結果を見てやり直さない。"}, ensure_ascii=False), encoding="utf-8")
        _log("test 2025: " + json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
