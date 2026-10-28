"""tp-81 続き: 予測の精度をどこまで上げれば展開利が効くか（実際 ↔ 予測の混合ダイヤル）と、追走難度（tp54）を第3の軸にした版。

  python -m src.scripts.data.tp81_dial dial    # 位置 / ペース / 両方を q×実際 + (1-q)×予測 に混ぜて、補正の LogLoss 改善幅を q ごとに出す
  python -m src.scripts.data.tp81_dial trk     # 追走難度 EASE2（tp54 の年別 LOYO）を 3 軸目にした版（オラクル＝実際の追走負荷、予測＝EASE2）
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.scripts.data import tp81_tenkai as K

OUT = K.OUT


def _row(r: dict) -> dict:
    o = r["絶対"]["overall"]
    return {m: round(o.get(m, float("nan")), 5) for m in ("Δwin", "win_hi", "Δtop2", "Δtop3")} | {"by_year_Δwin": {y: round(v.get("Δwin", float("nan")), 5) for y, v in r["絶対"]["by_year"].items()}, "lam": r["lam"]}


def cmd_dial() -> None:
    D, Tb, X = K.load_all()
    pos, pos_ok, pace, pace_ok, qs = K.actual_axes(X, Tb)
    Op, Oc = K.onehot(pos, 4, pos_ok), K.onehot(pace, 3, pace_ok)
    Pp, _ = K.predicted_pos(X, D)
    Pc, _ = K.predicted_pace(X, Tb, qs)
    res = {}
    lams = (100.0, 1000.0)
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        mix = lambda a, b: q * a + (1 - q) * b
        cases = {
            f"位置のみ q={q}": [mix(Op, Pp), np.full((len(X), 3), 1 / 3)],
            f"位置(q)×予測ペース q={q}": [mix(Op, Pp), Pc],
            f"予測位置×ペース(q) q={q}": [Pp, mix(Oc, Pc)],
            f"位置(q)×ペース(q) q={q}": [mix(Op, Pp), mix(Oc, Pc)],
        }
        for name, axes in cases.items():
            r, _ = K.run_variant(X, axes, name, lams=lams, relative_opts=(False,))
            res[name] = _row(r)
            K._log(f"{name}: {res[name]}")
    (OUT / "dial.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def load_trk(X: pd.DataFrame):
    """EASE2（tp54 の年別 LOYO。y_hat=予測の追走負荷）と、実際の追走負荷（Y2 = 実測の上がり偏差 − ベースライン）。"""
    g = pd.read_parquet(K.ROOT / "data/local/modeling/tp54/groups/EASE2.parquet")
    g["race_id"], g["horse_id"] = g["race_id"].astype(str), g["horse_id"].astype(str)
    h = pd.read_parquet(K.ROOT / "data/local/modeling/tp54/horse_ease.parquet")
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    return g, h


def cmd_trk() -> None:
    D, Tb, X = K.load_all()
    g, h = load_trk(X)
    K._log(f"EASE2 {g.shape} {list(g.columns)}; horse_ease {h.shape}")
    pred_col = next(c for c in g.columns if c not in K.KEY and "drain" in c) if any("drain" in c for c in g.columns) else [c for c in g.columns if c not in K.KEY][0]
    y_col = next((c for c in ("y2", "Y2", "y_drain", "act_l3f_dev") if c in h.columns), None)
    K._log(f"予測列={pred_col} 実測列={y_col}")
    M = X[K.KEY].merge(g[K.KEY + [pred_col]], on=K.KEY, how="left").merge(h[K.KEY + ([y_col] if y_col else [])], on=K.KEY, how="left")
    pos, pos_ok, pace, pace_ok, qs = K.actual_axes(X, Tb)
    Op, Oc = K.onehot(pos, 4, pos_ok), K.onehot(pace, 3, pace_ok)
    Pp, _ = K.predicted_pos(X, D)
    Pc, _ = K.predicted_pace(X, Tb, qs)
    tp = M[pred_col].to_numpy(dtype="float64")
    cuts = np.nanquantile(tp[X["year"].to_numpy() <= 2022], [1 / 3, 2 / 3])
    Tp = K.onehot(np.digitize(tp, cuts), 3, ~np.isnan(tp))
    res = {"coverage_pred": float(np.mean(~np.isnan(tp)))}
    if y_col:
        ty = M[y_col].to_numpy(dtype="float64")
        Y = K.onehot(np.digitize(ty, np.nanquantile(ty, [1 / 3, 2 / 3])), 3, ~np.isnan(ty))
        res["corr_pred_actual_drain"] = float(pd.Series(tp).corr(pd.Series(ty), method="spearman"))
    lams = (100.0, 1000.0)
    U3 = np.full((len(X), 3), 1 / 3)
    cases = {
        "予測: 位置×ペース（再掲）": [Pp, Pc],
        "予測: 位置×ペース×追走難度": [Pp, Pc, Tp],
        "予測: 位置×追走難度": [Pp, Tp],
        "予測: 追走難度のみ": [np.full((len(X), 4), 0.25), Tp],
    }
    if y_col:
        cases |= {
            "オラクル: 位置×ペース×実際の追走負荷": [Op, Oc, Y],
            "オラクル: 位置×実際の追走負荷": [Op, Y],
            "オラクル: 実際の追走負荷のみ": [np.full((len(X), 4), 0.25), Y],
            "実際の位置×ペース＋予測の追走難度": [Op, Oc, Tp],
        }
    for name, axes in cases.items():
        r, _ = K.run_variant(X, axes, name, lams=lams, relative_opts=(False,))
        res[name] = _row(r)
        K._log(f"{name}: {res[name]}")
    (OUT / "trk.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def cmd_demand() -> None:
    """追走難度の『要求側』だけの軸（結果の上がり3F は使わない）。実際 = (1-実際の初角位置)×(-実ペースz)×(距離km-0.6)、予測 = EASE2_demand。"""
    D, Tb, X = K.load_all()
    g, h = load_trk(X)
    pos, pos_ok, pace, pace_ok, qs = K.actual_axes(X, Tb)
    za = X["race_id"].map(Tb.set_index("race_id")["z_act"]).to_numpy(dtype="float64")
    dk = X["distance"].to_numpy(dtype="float64") / 1000.0
    act = (1 - X["act_pos"].to_numpy(dtype="float64")) * (-za) * (dk - 0.6)
    M = X[K.KEY].merge(g[K.KEY + ["EASE2_demand", "EASE2_drain"]], on=K.KEY, how="left")
    pdem, pdr = M["EASE2_demand"].to_numpy(dtype="float64"), M["EASE2_drain"].to_numpy(dtype="float64")
    tert = lambda v: K.onehot(np.digitize(v, np.nanquantile(v, [1 / 3, 2 / 3])), 3, ~np.isnan(v))
    Op, Oc = K.onehot(pos, 4, pos_ok), K.onehot(pace, 3, pace_ok)
    Pp, _ = K.predicted_pos(X, D)
    Pc, _ = K.predicted_pace(X, Tb, qs)
    res = {"corr_actual_demand_vs_pred_demand": float(pd.Series(act).corr(pd.Series(pdem), method="spearman"))}
    cases = {
        "オラクル: 位置×ペース×実際の要求量": [Op, Oc, tert(act)],
        "オラクル: 実際の要求量のみ": [np.full((len(X), 4), 0.25), tert(act)],
        "予測: 位置×ペース×予測の要求量(EASE2_demand)": [Pp, Pc, tert(pdem)],
        "予測: 予測の要求量のみ": [np.full((len(X), 4), 0.25), tert(pdem)],
        "予測: 位置×ペース×EASE2_drain": [Pp, Pc, tert(pdr)],
    }
    for name, axes in cases.items():
        r, _ = K.run_variant(X, axes, name, lams=(100.0, 1000.0), relative_opts=(False,))
        res[name] = _row(r)
        K._log(f"{name}: {res[name]}")
    (OUT / "demand.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dial", "trk", "demand"])
    a = ap.parse_args()
    {"dial": cmd_dial, "trk": cmd_trk, "demand": cmd_demand}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
