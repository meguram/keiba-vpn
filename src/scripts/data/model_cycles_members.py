"""改善サイクルで採った土台（入力の列の変更）を、v6e のほかのメンバー（lgb_softmax・linear・NN など）にも当てて、LOYO 用の OOF を作り直す。

    REBUILD_BASE=<土台> CYCLES_SPEC='{...}' python -m src.scripts.data.model_cycles_members member lgb_softmax
    REBUILD_BASE=<土台> CYCLES_SPEC='{...}' python -m src.scripts.data.model_cycles_members mtl nn_extras_ped

- 学習の手順は parent_2016（= v6_members → tp27_eval の member・mtl）と同じ。fold・seed・設定も同じ。
- 入力の最後の段（tp27_eval._drop_by_env）で、model_cycles.apply_spec と同じ変更（落とす・書き換え・blank_before）を当てる。
  apply_spec に渡す年は、load_v3 の df の行の年（race_id の先頭 4 桁）。
- 出力は REBUILD_BASE/tp27_v6e/oof/（重ね合わせの土台の中の自分の置き場。data-audit や旧 base_2016 の置き場には書かない）。
  CYCLES_MEMBER_OUT=<名前> で REBUILD_BASE/<名前>/oof/ に分け、CYCLES_XGB_PARAMS='{...}' で xgb_extras_ped の設定を上書きする。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd


def main() -> None:
    if not os.environ.get("REBUILD_BASE"):
        raise SystemExit("REBUILD_BASE（重ね合わせの土台）を付ける。付けないと旧 base_2016 に書く")
    spec = json.loads(os.environ.get("CYCLES_SPEC") or "{}")
    bad = set(spec) - {"drop", "drop_prefix", "blank_before"}
    if bad:
        raise SystemExit(f"このメンバーの作り直しが対応しない変更: {sorted(bad)}（xgb と入力がずれるので止める）")
    from src.scripts.data import parent_2016 as p

    p.prepare()
    for k, v in {"TP_BASE_DIR": p.BASE, "TP_WORK": p.BASE, "TP25C_OUT": p.BASE / "tp25c", "TP25E_OUT": p.BASE / "tp25e",
                 "TP27_FAC_FIELD": p.BASE / "factory/field", "V6_OUT": p.BASE / "v6", "V5_TP38_OUT": p.BASE / "tp38",
                 "TP27_OUT": p.BASE / "tp27_v6e"}.items():
        os.environ[k] = str(v)
    os.environ.setdefault("V5_GROUPS", "CY,BEST,RB")
    os.environ.pop("V6_STAGE", None)
    from src.scripts.data import rebuild_groups_2016 as rg

    rg.redirect()
    from src.scripts.data import model_cycles as mc
    from src.scripts.data import tp25e_eval as e
    from src.scripts.data import tp27_eval as t27
    from src.scripts.data import v5_common
    from src.scripts.data import v6_members as m

    out_dir = os.environ.get("CYCLES_MEMBER_OUT")            # 設定を変えた版の書き出し先（土台の中の名前。既定は土台の tp27_v6e）
    if out_dir and ("/" in out_dir or out_dir.startswith(".")):
        raise SystemExit("CYCLES_MEMBER_OUT は土台の中の名前だけ（例 tp27_v6e_lr3）")
    if os.environ.get("CYCLES_XGB_PARAMS") and not out_dir:
        raise SystemExit("CYCLES_XGB_PARAMS を付けるときは CYCLES_MEMBER_OUT も付ける（土台の版と混ざらないように）")
    m.OUT = t27.OUT = p.BASE / (out_dir or "tp27_v6e")
    os.environ["TP27_OUT"] = str(m.OUT)
    if out_dir:
        import shutil

        (m.OUT / "oof").mkdir(parents=True, exist_ok=True)
        src = p.BASE / "tp27_v6e/importance_added.json"
        if src.exists() and not (m.OUT / "importance_added.json").exists():
            shutil.copy2(src, m.OUT / "importance_added.json")
    xgb_over = json.loads(os.environ.get("CYCLES_XGB_PARAMS") or "{}")   # xgb_extras_ped の設定の上書き（例: 学習率を半分・木を倍）
    v5_common.V = p.BASE
    years: dict[str, np.ndarray] = {}
    orig_load, orig_drop = e.load_v3, t27._drop_by_env

    def load_v3():
        out = orig_load()
        if xgb_over:
            r_, prm, X_ = out[3]["xgb_extras_ped"]
            out[3]["xgb_extras_ped"] = (r_, {**prm, **xgb_over}, X_)
            print(f"[model_cycles_members] xgb_extras_ped の設定を上書き: {xgb_over}", flush=True)
        years["y"] = out[1]["year"].astype(int).to_numpy()          # meta の year（model_cycles の member と同じ出どころ）
        years["n"] = len(out[0])
        return out

    def drop_by_env(X: pd.DataFrame) -> pd.DataFrame:
        X = orig_drop(X)
        if len(X) != years["n"] or not X.index.equals(pd.RangeIndex(years["n"])):
            raise SystemExit("入力の行が load_v3 の df と一致しない（年を貼れない）")
        meta = pd.DataFrame({"year": years["y"], "race_class": ""}, index=X.index)
        before = X.shape[1]
        X = mc.apply_spec(X, meta, spec)
        print(f"[model_cycles_members] spec を当てた: 列 {before} → {X.shape[1]}", flush=True)
        return X

    e.load_v3, t27._drop_by_env = load_v3, drop_by_env
    m.main()


if __name__ == "__main__":
    sys.exit(main())
