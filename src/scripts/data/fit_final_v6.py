"""アンサンブル v6（v5 + めぐ指数/netkeiba 指数の履歴 RB 16 列 + 前段 S3B）の最終モデルを学習して、推論用に保存する（fit_final_v5 と同じ組み立て）。

V6_STAGE=b V5_GROUPS=CY,BEST,RB python -m src.scripts.data.fit_final_v6 --fit-end 2024   # 2020〜2024 で学習 → bundle_v6_2024 と test 2025 の予測（一度だけ。選択には使わない）
V6_STAGE=b V5_GROUPS=CY,BEST,RB python -m src.scripts.data.fit_final_v6 --fit-end 2025   # 2020〜2025 の全行で学習 → bundle_v6_2025（2026 年は使わない）
出力: data/local/modeling/training_pipeline/final_v6{V6_TAG}/bundle_v6_{fit_end}/、評価は final_eval_v6{V6_TAG}/（final_v5 は触らない）
アンサンブルの重み: tp27_v6/ens/v6.json（v6_members の LOYO）。環境変数: V6_STAGE, V5_GROUPS, V5_FIT_THREADS（既定 6）
前段 S3B は expanding の OOF（学習行）。推論用の全期間版の前段モデルは別（serving_v6）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

if os.environ.get("V6_STAGE"):
    from src.scripts.data import v6_work

    v6_work.use(os.environ["V6_STAGE"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.scripts.data import fit_final_v4 as f4  # noqa: E402
from src.scripts.data import fit_final_v5 as f5  # noqa: E402
from src.scripts.data import v5_common  # noqa: E402
from src.scripts.data import v5_select_eval as sel  # noqa: E402
from src.scripts.data import v6_eval as ev  # noqa: E402

KEY = f4.KEY
TAG = os.environ.get("V6_TAG", "")
OUT = f4.OLD / f"final_v6{TAG}"
EVAL = f4.OLD / f"final_eval_v6{TAG}"
TP27_V6 = f4.V4 / f"tp27_v6{TAG}"
f5.TP27_V5 = TP27_V6                                   # load_spec_v5 が読む LOYO の場所を v6 に向ける
V6_GROUPS = {"PT", "GB", "CB", "OWNN", "RS59", *ev.EXTERNAL}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def patch() -> None:
    v5_common.patch()
    orig = sel.load_group

    def load_group(name: str, df: pd.DataFrame) -> pd.DataFrame:
        return ev.load_new(name, df, pd.DataFrame(index=df.index)) if name in V6_GROUPS else orig(name, df)

    sel.load_group = load_group


def evaluate_test2025(bundle, X_all, meta) -> dict:
    """test 2025 の 1 回評価（報告用。選択には使わない）。v5 の bundle_v5_2024 の同じ行（final_eval_v5）との対比と、初出走などの部分集合。"""
    from src.pipeline.models import feature_bakeoff as fb
    from src.scripts.data import tp40_eval as e40

    fin = meta["finish_position"].to_numpy()
    race = meta["race_id"].to_numpy()
    va = np.flatnonzero((meta["year"].to_numpy() == 2025) & np.isfinite(fin.astype(float)))
    pred = bundle.predict(X_all.iloc[va], race[va])
    P = pd.DataFrame({"race_id": race[va], "horse_id": meta["horse_id"].to_numpy()[va], "finish_position": fin[va]}).join(pred.reset_index(drop=True))
    EVAL.mkdir(parents=True, exist_ok=True)
    P.to_parquet(EVAL / "test2025_predictions.parquet", index=False)
    v5 = pd.read_parquet(f4.OLD / "final_eval_v5/test2025_predictions.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
    for d in (v5, P):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    fin_map = P.set_index(KEY)["finish_position"]
    mm = P[KEY].merge(v5[KEY], on=KEY, how="inner")
    a, b = v5.merge(mm, on=KEY), P[KEY + ["p_win", "p_top2", "p_top3"]].merge(mm, on=KEY)
    cmp_ = fb.compare_oof(a, b, fin_map, "v6 vs v5 (test 2025)")
    rm = pd.read_parquet(e40.OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    sub = e40._subset_cmp(a, b, fin_map, rm, "v6 vs v5 test2025")
    ll = {}
    for t_, k in (("win", 1), ("top2", 2), ("top3", 3)):
        yy = (P["finish_position"].to_numpy() <= k).astype(float)
        p = P[f"p_{t_}"].to_numpy()
        ll[t_] = float(-np.mean(yy * np.log(np.clip(p, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-9, 1))))
    rep = {"rows": int(len(P)), "races": int(P["race_id"].nunique()), "logloss": ll, "groups": v5_common.groups(), "stage": os.environ.get("V6_STAGE"),
           "vs_v5": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cmp_.items()}, "subsets_vs_v5": sub}
    (EVAL / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (EVAL / "TEST2025_EVALUATED.json").write_text(json.dumps({"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": "test 2025 を評価済み（v6）。結果を見てやり直さない（選択に使わない）。"}, ensure_ascii=False), encoding="utf-8")
    _log("test 2025: " + json.dumps(rep, ensure_ascii=False))
    return rep


def main() -> None:
    from src.pipeline.models import ensemble_bundle as eb
    from src.pipeline.models import zoo

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ens", default="v6")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    if a.fit_end == 2024 and (EVAL / "TEST2025_EVALUATED.json").exists() and not a.no_eval:
        raise SystemExit("test 2025 は評価済み（v6）。やり直さない")
    t0 = time.time()
    patch()
    df, meta, mats, A, _A_feat = f4.build_matrices()
    names = [m[0] for m in f4.MEMBERS]
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    members = [eb.MemberSpec(n, mats[n][0], mats[n][1], list(mats[n][2].columns)) for n in names]
    spec, ens_source = f5.load_spec_v5(names, a.ens)
    fin = meta["finish_position"].to_numpy()
    dates = pd.to_datetime(meta["date"])
    if a.fit_end == 2024:
        tr = np.flatnonzero(dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10))
    else:
        tr = np.flatnonzero((meta["year"].to_numpy() <= 2025) & np.isfinite(fin.astype(float)))
    race = meta["race_id"].to_numpy()
    y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    ytr = {k: v[tr] for k, v in y.items()}
    gp = TP27_V6 / "oof/lgb_softmax__v4.parquet"
    o = pd.read_parquet(gp).merge(meta[KEY + ["finish_position"]], on=KEY)
    gam = (zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 2), zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 3))
    bundle = eb.EnsembleBundle(members, spec, {"lgb_softmax": gam})
    threads = int(os.environ.get("V5_FIT_THREADS", "6"))
    _log(f"学習 {len(tr):,} 行 / 列 {X_all.shape[1]} / seed {a.seeds} / threads {threads} / 群 {v5_common.groups()} / 前段 {os.environ.get('V6_STAGE')} / 重み {ens_source}")
    bundle.fit(X_all.iloc[tr], ytr, race[tr], threads, list(range(a.seeds)), aux=A[tr])
    _log(f"学習完了 {time.time() - t0:.0f}s")
    out = OUT / f"bundle_v6_{a.fit_end}"
    bundle.save(out)
    (out / "fit_info.json").write_text(json.dumps({"fit_end": a.fit_end, "train_rows": int(len(tr)), "seeds": a.seeds, "n_features": int(X_all.shape[1]), "members": names, "groups": v5_common.groups(),
                                                  "stage": os.environ.get("V6_STAGE"), "ens": a.ens, "ens_source": ens_source, "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"保存: {out}")
    if a.fit_end == 2024 and not a.no_eval:
        evaluate_test2025(bundle, X_all, meta)


if __name__ == "__main__":
    main()
