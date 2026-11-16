"""改善サイクルで選んだ版の最終の学習（束）。fit_final_v6 と同じ作り（build_matrices → EnsembleBundle.fit）に、サイクルの変更を当てる。

    REBUILD_BASE=<土台> CYCLES_OUT=<出力> CYCLES_SPEC='{...}' CYCLES_XGB_PARAMS='{...}' \\
        python -m src.scripts.data.model_cycles_final --fit-end 2024 --ens <ens の名前>   # 2016〜2024 で学習 → 2025 年 test を 1 回だけ予測
    ... --fit-end 2025                                                                     # 2016〜2025 の全行で学習（推論・stg 用の束）

- メンバー: --members（既定は重みのある 5 本: xgb_extras_ped・lgb_softmax・linear・mtl_nn_extras_ped・mtl_nn_s）。
- 入力: 各メンバーの行列に model_cycles.apply_spec（drop・drop_prefix・blank_before）を当てる。年は meta の year。
- 重み: 勝率は CYCLES_OUT/ens/<ens>.json（LOYO で当てた重み）。連対・複勝は、候補 v6e16 の束の重みを、使うメンバーに絞って合計 1 に直したもの
  （xgb の連対・複勝の OOF を作り直していないため。記録に書く）。
- 出力: CYCLES_OUT/final/bundle_v6_<fit_end>/（束）と、fit_end 2024 のときは CYCLES_OUT/final/test2025_predictions.parquet（1 回きり。作り直さない）。
- 本番の束・旧 base_2016 の置き場には書かない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERS5 = ["xgb_extras_ped", "lgb_softmax", "linear", "mtl_nn_extras_ped", "mtl_nn_s"]
CAND_BUNDLE = Path("/home/jovyan/work/retrain_out/base_2016/final_v6e16/bundle_v6_2025/bundle.json")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_spec(names: list[str], ens_json: Path) -> dict:
    ens = json.loads(ens_json.read_text(encoding="utf-8"))
    if set(ens["weights"]) != set(names):
        raise SystemExit(f"ens の重みのメンバー {sorted(ens['weights'])} と、束のメンバー {sorted(names)} が違う")
    win = [float(ens["weights"][n]) for n in names]
    cb = json.loads(CAND_BUNDLE.read_text(encoding="utf-8"))
    cnames = [m["name"] for m in cb["members"]]
    targets = {"win": {"method": "weighted_logit", "weights": win}}
    for t in ("top2", "top3"):
        s = cb["spec"]["targets"][t]
        w = np.array([s["weights"][cnames.index(n)] if n in cnames else 0.0 for n in names], dtype=float)
        if w.sum() <= 0:
            raise SystemExit(f"{t} の重みが使うメンバーに無い")
        targets[t] = {"method": s["method"], "weights": [round(float(x), 6) for x in w / w.sum()]}
    return {"targets": targets, "models": names, "source": f"win: {ens_json}（LOYO）、top2・top3: 候補 v6e16 の束の重みを使うメンバーに絞って正規化"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ens", required=True, help="CYCLES_OUT/ens/<名前>.json（勝率の重み）")
    ap.add_argument("--members", nargs="*", default=MEMBERS5)
    a = ap.parse_args()
    if not os.environ.get("REBUILD_BASE") or not os.environ.get("CYCLES_OUT"):
        raise SystemExit("REBUILD_BASE と CYCLES_OUT を付ける")
    out_root = Path(os.environ["CYCLES_OUT"]) / "final"
    forbidden = [Path("/home/jovyan/work/retrain_out/base_2016").resolve(), Path(os.environ["REBUILD_BASE"]).resolve(),
                 (Path(__file__).resolve().parents[3] / "data/local/modeling").resolve()]
    if any(str(out_root.resolve()).startswith(str(f)) for f in forbidden):
        raise SystemExit(f"{out_root} には書かない（本番・旧 base_2016・土台の中）")
    test_out = out_root / "test2025_predictions.parquet"
    if a.fit_end == 2024 and (test_out.exists() or (out_root / "TEST2025_EVALUATED.json").exists()):
        raise SystemExit(f"{out_root} の 2025 年 test は作成済み（1 回きり。作り直さない）")
    ens_info = json.loads((Path(os.environ["CYCLES_OUT"]) / "ens" / f"{a.ens}.json").read_text(encoding="utf-8"))
    if ens_info.get("spec_check") != {"spec": json.loads(os.environ.get("CYCLES_SPEC") or "{}"), "xgb_params": json.loads(os.environ.get("CYCLES_XGB_PARAMS") or "{}")}:
        raise SystemExit("ens の記録（spec_check）と、CYCLES_SPEC・CYCLES_XGB_PARAMS が違う（別の版の重みを使おうとしている）")
    spec = json.loads(os.environ.get("CYCLES_SPEC") or "{}")
    bad = set(spec) - {"drop", "drop_prefix", "blank_before"}
    if bad:
        raise SystemExit(f"対応しない変更: {sorted(bad)}")
    xgb_over = json.loads(os.environ.get("CYCLES_XGB_PARAMS") or "{}")

    from src.scripts.data import parent_final_2016 as pf

    pf.setup_env()
    from src.pipeline.models import ensemble_bundle as eb
    from src.pipeline.models import zoo
    from src.scripts.data import fit_final_v4 as f4
    from src.scripts.data import fit_final_v6 as f6
    from src.scripts.data import model_cycles as mc

    t0 = time.time()
    f6.patch()
    df, meta, mats, A, _ = f4.build_matrices()
    if not meta.index.equals(pd.RangeIndex(len(meta))):
        raise SystemExit("meta の index が 0 からの連番でない（行の対応を確かめられない）")
    year = meta["year"].astype(int).to_numpy()
    m_year = pd.DataFrame({"year": year, "race_class": ""})
    names = list(a.members)
    for n in names:
        recipe, params, X = mats[n]
        if len(X) != len(meta) or not X.index.equals(meta.index):
            raise SystemExit(f"{n} の行が meta と一致しない")
        X = mc.apply_spec(X.reset_index(drop=True), m_year, spec)
        if n == "xgb_extras_ped" and xgb_over:
            params = {**params, **xgb_over}
        mats[n] = (recipe, params, X)
        log(f"{n}: 列 {X.shape[1]} 設定 {params if n == 'xgb_extras_ped' else ''}")
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    members = [eb.MemberSpec(n, mats[n][0], mats[n][1], list(mats[n][2].columns)) for n in names]
    bspec = build_spec(names, Path(os.environ["CYCLES_OUT"]) / "ens" / f"{a.ens}.json")
    fin = meta["finish_position"].to_numpy()
    dates = pd.to_datetime(meta["date"])
    if a.fit_end == 2024:
        tr = np.flatnonzero((dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10)).to_numpy() & np.isfinite(fin.astype(float)))
    else:
        tr = np.flatnonzero((year <= 2025) & np.isfinite(fin.astype(float)))
    race = meta["race_id"].to_numpy()
    y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    gp = Path(os.environ["REBUILD_BASE"]) / "tp27_v6e/oof/lgb_softmax__v4.parquet"   # spec を当てて作り直した lgb_softmax の OOF（valid 2022〜2024 だけ。2025 年は入らない）
    o = pd.read_parquet(gp)
    if "year" in o.columns and (o["year"].astype(int) > 2024).any():
        raise SystemExit("ガンマの元の OOF に 2025 年以降の行がある")
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    mm = meta[["race_id", "horse_id", "finish_position"]].copy()
    mm["race_id"], mm["horse_id"] = mm["race_id"].astype(str), mm["horse_id"].astype(str)
    o = o.merge(mm, on=["race_id", "horse_id"])
    gam = (zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 2),
           zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 3))
    bundle = eb.EnsembleBundle(members, bspec, {"lgb_softmax": gam})
    threads = int(os.environ.get("V5_FIT_THREADS", "6"))
    log(f"学習 {len(tr):,} 行 / 列 {X_all.shape[1]} / seed {a.seeds} / メンバー {names} / 重み {bspec['targets']}")
    bundle.fit(X_all.iloc[tr], {k: v[tr] for k, v in y.items()}, race[tr], threads, list(range(a.seeds)), aux=A[tr])
    out = out_root / f"bundle_v6_{a.fit_end}"
    bundle.save(out)
    (out / "fit_info.json").write_text(json.dumps({"fit_end": a.fit_end, "train_rows": int(len(tr)), "seeds": a.seeds, "n_features": int(X_all.shape[1]),
                                                  "members": names, "spec": spec, "xgb_params": xgb_over, "ens": a.ens, "source": bspec["source"],
                                                  "base": os.environ["REBUILD_BASE"], "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"保存: {out}（{time.time() - t0:.0f}s）")
    if a.fit_end == 2024:
        va = np.flatnonzero((year == 2025) & np.isfinite(fin.astype(float)))
        pred = bundle.predict(X_all.iloc[va], race[va])
        P = pd.DataFrame({"race_id": race[va], "horse_id": meta["horse_id"].to_numpy()[va], "year": 2025, "finish_position": fin[va]}).join(pred.reset_index(drop=True))
        P.to_parquet(test_out, index=False)
        (out_root / "TEST2025_EVALUATED.json").write_text(json.dumps({"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                                     "note": "2025 年の test を 1 回だけ予測した。結果を見てやり直さない（選択に使わない）。"}, ensure_ascii=False), encoding="utf-8")
        log(f"2025 年 test: {len(P)} 行 → {test_out}")


if __name__ == "__main__":
    sys.exit(main())
