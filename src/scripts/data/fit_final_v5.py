"""アンサンブル v5 の最終モデルを学習して、推論用に保存する（fit_final_v4 の組み立てに V5_GROUPS と再取得版 tl_ を差し込む）。

V5_GROUPS=CY python -m src.scripts.data.fit_final_v5 --fit-end 2024     # 2020〜2024 で学習 → bundle_v5_2024 と test 2025 の予測（一度だけ。選択には使わない）
V5_GROUPS=CY python -m src.scripts.data.fit_final_v5 --fit-end 2025     # 2020〜2025 の全行で学習 → bundle_v5_2025（2026 年は使わない）
出力: data/local/modeling/training_pipeline/final_v5/bundle_v5_{fit_end}/、評価は final_eval_v5/（既存の final_v4 は上書きしない）
アンサンブルの重み: tp27_v5/ens/<--ens>.json（v5_members の LOYO）。無ければ V4 の v4_e2 の重みを流用し、fit_info に ens_source を記録する。
環境変数: V5_GROUPS, V5_FIT_THREADS（既定 6）
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.scripts.data import fit_final_v4 as f4
from src.scripts.data import v5_common

KEY = f4.KEY
OUT = f4.OLD / "final_v5"
EVAL = f4.OLD / "final_eval_v5"
TP27_V5 = v5_common.V5_TP27


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_spec_v5(names: list[str], ens_name: str) -> tuple[dict, str]:
    p = TP27_V5 / "ens" / f"{ens_name}.json"
    src = f"tp27_v5 ens {ens_name} (LOYO)"
    if not p.exists():
        p = f4.V4 / "tp27/ens/v4_e2.json"
        src = "V4 の v4_e2 の重みを流用（v5 の LOYO 未実施）"
    j = json.loads(p.read_text(encoding="utf-8"))["spec"]
    targets = {t: {"method": s["method"], "weights": [float(s["weights"].get(n + "__v4", 0.0)) for n in names]} for t, s in j.items()}
    return {"targets": targets, "models": names, "source": src}, src


def evaluate_test2025(bundle, X_all, meta) -> dict:
    """test 2025 の 1 回評価（報告用。選択には使わない）。v4 の同じ行との対比も出す。"""
    from src.pipeline.models import feature_bakeoff as fb

    fin = meta["finish_position"].to_numpy()
    race = meta["race_id"].to_numpy()
    va = np.flatnonzero((meta["year"].to_numpy() == 2025) & np.isfinite(fin.astype(float)))
    pred = bundle.predict(X_all.iloc[va], race[va])
    P = pd.DataFrame({"race_id": race[va], "horse_id": meta["horse_id"].to_numpy()[va], "finish_position": fin[va]}).join(pred.reset_index(drop=True))
    EVAL.mkdir(parents=True, exist_ok=True)
    P.to_parquet(EVAL / "test2025_predictions.parquet", index=False)
    v4 = pd.read_parquet(f4.OLD / "final_eval_v4/test2025_predictions.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
    fin_map = P.set_index(KEY)["finish_position"]
    mm = P[KEY].merge(v4[KEY], on=KEY, how="inner")
    cmp_ = fb.compare_oof(v4.merge(mm, on=KEY), P[KEY + ["p_win", "p_top2", "p_top3"]].merge(mm, on=KEY), fin_map, "v5 vs v4 (test 2025)")
    ll = {}
    for t_, k in (("win", 1), ("top2", 2), ("top3", 3)):
        yy = (P["finish_position"].to_numpy() <= k).astype(float)
        p = P[f"p_{t_}"].to_numpy()
        ll[t_] = float(-np.mean(yy * np.log(np.clip(p, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-9, 1))))
    rep = {"rows": int(len(P)), "races": int(P["race_id"].nunique()), "logloss": ll, "groups": v5_common.groups(),
           "vs_v4": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cmp_.items()}, "reference": {"v1_win": 0.21349, "v2_win": 0.21240, "v3_win": 0.21039, "v4_win": 0.20740}}
    (EVAL / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (EVAL / "TEST2025_EVALUATED.json").write_text(json.dumps({"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": "test 2025 を評価済み（v5）。結果を見てやり直さない（選択に使わない）。"}, ensure_ascii=False), encoding="utf-8")
    _log("test 2025: " + json.dumps(rep, ensure_ascii=False))
    return rep


def main() -> None:
    from src.pipeline.models import ensemble_bundle as eb
    from src.pipeline.models import feature_bakeoff as fb
    from src.pipeline.models import zoo

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ens", default="v5")
    ap.add_argument("--no-eval", action="store_true", help="fit-end 2024 の test 2025 評価を後回しにする（重みが確定してから `eval` で 1 回だけ行う）")
    ap.add_argument("--eval-only", action="store_true", help="保存済みの bundle_v5_2024 を読み、tp27_v5 の LOYO 重みで spec を更新して test 2025 を 1 回だけ評価する")
    a = ap.parse_args()
    t0 = time.time()
    v5_common.patch()
    df, meta, mats, A, _A_feat = f4.build_matrices()
    names = [m[0] for m in f4.MEMBERS]
    if a.eval_only:
        X_all = pd.concat([mats[n][2] for n in names], axis=1)
        X_all = X_all.loc[:, ~X_all.columns.duplicated()]
        spec, ens_source = load_spec_v5(names, a.ens)
        bundle = eb.EnsembleBundle.load(OUT / f"bundle_v5_{a.fit_end}")
        bundle.spec = spec
        (OUT / f"bundle_v5_{a.fit_end}" / "bundle.json").write_text(json.dumps({**json.loads((OUT / f"bundle_v5_{a.fit_end}" / "bundle.json").read_text(encoding="utf-8")), "spec": spec}, ensure_ascii=False, indent=1), encoding="utf-8")
        _log(f"重みを更新（{ens_source}）")
        evaluate_test2025(bundle, X_all, meta)
        return
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    members = [eb.MemberSpec(n, mats[n][0], mats[n][1], list(mats[n][2].columns)) for n in names]
    spec, ens_source = load_spec_v5(names, a.ens)
    fin = meta["finish_position"].to_numpy()
    dates = pd.to_datetime(meta["date"])
    if a.fit_end == 2024:
        tr = np.flatnonzero(dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10))
    else:
        tr = np.flatnonzero((meta["year"].to_numpy() <= 2025) & np.isfinite(fin.astype(float)))
    race = meta["race_id"].to_numpy()
    y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    ytr = {k: v[tr] for k, v in y.items()}
    gp = TP27_V5 / "oof/lgb_softmax__v4.parquet"
    o = pd.read_parquet(gp if gp.exists() else f4.V4 / "tp27/oof/lgb_softmax__v4.parquet").merge(meta[KEY + ["finish_position"]], on=KEY)
    gam = (zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 2), zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 3))
    bundle = eb.EnsembleBundle(members, spec, {"lgb_softmax": gam})
    threads = int(os.environ.get("V5_FIT_THREADS", "6"))
    _log(f"学習 {len(tr):,} 行 / 列 {X_all.shape[1]} / seed {a.seeds} / threads {threads} / 群 {v5_common.groups()} / 重み {ens_source}")
    bundle.fit(X_all.iloc[tr], ytr, race[tr], threads, list(range(a.seeds)), aux=A[tr])
    _log(f"学習完了 {time.time() - t0:.0f}s")
    out = OUT / f"bundle_v5_{a.fit_end}"
    bundle.save(out)
    (out / "fit_info.json").write_text(json.dumps({"fit_end": a.fit_end, "train_rows": int(len(tr)), "seeds": a.seeds, "n_features": int(X_all.shape[1]), "members": names, "groups": v5_common.groups(),
                                                  "ens": a.ens, "ens_source": ens_source, "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"保存: {out}")
    if a.fit_end == 2024 and not a.no_eval:
        evaluate_test2025(bundle, X_all, meta)


if __name__ == "__main__":
    main()
