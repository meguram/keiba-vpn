"""tp-61: 展開タイプの予測モデル（ペース・隊列の分類器）を最終学習し、API 用の設定（config/pace_type.json, config/pace_type_models.json）を書き出す。

python -m src.scripts.data.tp61_final blend   # ペースの確率に lap_forecast（tp-55）を混ぜる重みの選択（選択 2024・確認 2025）
python -m src.scripts.data.tp61_final export  # 学習 2020〜2025 の分類器 + 脚質別の表 + シナリオ係数 + コース表 → config/*.json（2026 の LogLoss を併記）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference import pace_type as pt
from src.scripts.data import tp61_pre as pre
from src.scripts.data import tp61_scen as sc

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp61"
CONF = D / "export"   # ホストでは config/ が root 所有のため、出力は作業ディレクトリに置いてから config/ へ配置する
PACE, SHAPE = pre.PACE, pre.SHAPE
FEATS = None   # 学習に使う列（RACE_FEATS + HANA）


def _feats(d: pd.DataFrame) -> list[str]:
    return [c for c in pre.RACE_FEATS if c in d.columns] + [c for c in pre.HANA if c in d.columns]


def cmd_blend() -> int:
    d = pd.read_parquet(D / "pre.parquet")
    d = d[d["pace_cls"].isin(PACE) & d["shape_cls"].isin(SHAPE)].merge(pre.load_tp55(), on="race_id", how="left")
    feats = _feats(d)
    tr = d[d["year"].isin([2020, 2021, 2022, 2023])]
    va = d[d["year"].isin([2024, 2025]) & d["p0"].notna()]
    m = pre._fit_lgb(tr[feats], tr["pace_cls"].map({c: i for i, c in enumerate(PACE)}).to_numpy(), 3)
    p_me = m.predict_proba(va[feats])
    p_lap = va[["p0", "p1", "p2"]].to_numpy()
    y = va["pace_cls"].map({c: i for i, c in enumerate(PACE)}).to_numpy()
    for yr in (2024, 2025):
        msk = (va["year"] == yr).to_numpy()
        print(yr, {w: round(pre._ll(y[msk], w * p_me[msk] + (1 - w) * p_lap[msk]), 5) for w in (0.0, 0.25, 0.5, 0.75, 1.0)})
        lg = np.exp(0.5 * np.log(np.clip(p_me[msk], 1e-9, 1)) + 0.5 * np.log(np.clip(p_lap[msk], 1e-9, 1)))
        print(yr, "geom-mean", round(pre._ll(y[msk], lg), 5))
    return 0


def style_tables(cells_h: pd.DataFrame) -> dict:
    """想定の脚質（forecast style）× (ペース, 隊列) の3着内率の比（その脚質の全体の率に対する倍率）とレース単位ブートストラップ区間。"""
    from src.scripts.data.tp61_analysis import _boot

    ref = cells_h.groupby("pf_style")[["is_top3", "is_win"]].mean()
    out = {}
    for (pc, sc_), g in cells_h.groupby(["pace_cls", "shape_cls"]):
        row = {}
        for st in ("front", "stalker", "mid", "closer"):
            gs = g[g["pf_style"] == st]
            per = gs.groupby("race_id").agg(top3=("is_top3", "sum"), win=("is_win", "sum"), n=("is_top3", "size"))
            if len(per) < 30:
                continue
            b3 = _boot(pd.DataFrame({"a": per["top3"], "b": ref.loc[st, "is_top3"] * per["n"]}), ["a", "b"], n=300)
            bw = _boot(pd.DataFrame({"a": per["win"], "b": ref.loc[st, "is_win"] * per["n"]}), ["a", "b"], n=300)
            row[st] = {"n_races": int(len(per)), "top3_lift": round(b3["mean"], 3), "top3_lo": round(b3["lo"], 3), "top3_hi": round(b3["hi"], 3),
                       "win_lift": round(bw["mean"], 3), "win_lo": round(bw["lo"], 3), "win_hi": round(bw["hi"], 3)}
        out[f"{pc}_{sc_}"] = row
    return out


def course_tables(d: pd.DataFrame) -> dict:
    """コース（場|面|距離）ごとの縮約平均（全期間）。サービス時の crs_* に使う。"""
    out: dict = {}
    cols = {"c4_gap_per_horse": "crs_gap", "bal": "crs_bal", "t3f_dev": "crs_t3", "c4_lead_group_frac": "crs_lead"}
    gm = {v: float(d[k].mean()) for k, v in cols.items()}
    d = d.copy()
    d["ck"] = d["venue"].astype(str) + "|" + d["surface"].astype(str) + "|" + d["distance"].astype(int).astype(str)
    for ck, g in d.groupby("ck"):
        n = int(g["c4_gap_per_horse"].notna().sum())
        row = {"n": n}
        for k, v in cols.items():
            row[v] = round(float((g[k].fillna(gm[v]).sum() + 20.0 * gm[v]) / (len(g) + 20.0)), 4)
        out[ck] = row
    return {"global": {k: round(v, 4) for k, v in gm.items()}, "by_course": out}


def geo_table() -> dict:
    from src.pipeline.features import tp55_lapfeat as lf

    geo = lf.load_course_geo(ROOT / "data/meta/course_master/jra_courses.json")
    vc = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
    out = {}
    for (code, surface, dist), g in geo.items():
        v = vc.get(code)
        if v:
            out[f"{v}|{surface}|{dist}"] = [None if np.isnan(g["geo_first_curve_m"]) else round(float(g["geo_first_curve_m"]), 1),
                                            None if np.isnan(g["geo_start_geom"]) else round(float(g["geo_start_geom"]), 2)]
    return out


def cmd_export() -> int:
    d = pd.read_parquet(D / "pre.parquet")
    d = d[d["pace_cls"].isin(PACE) & d["shape_cls"].isin(SHAPE)].copy()
    feats = _feats(d)
    meta = json.loads((D / "pre_meta.json").read_text(encoding="utf-8"))
    tr = d[d["year"].isin(range(2020, 2026))]
    te = d[d["year"] == 2026]
    models, rep = {}, {}
    for name, classes, col in (("pace", PACE, "pace_cls"), ("shape", SHAPE, "shape_cls")):
        y = tr[col].map({c: i for i, c in enumerate(classes)}).to_numpy()
        m = pre._fit_lgb(tr[feats], y, 3)
        models[name] = m.booster_.model_to_string()
        yt = te[col].map({c: i for i, c in enumerate(classes)}).to_numpy()
        p = m.predict_proba(te[feats])
        prior = np.bincount(y, minlength=3) / len(y)
        lv = pd.get_dummies(tr["pf_level"].fillna("unknown")).reindex(columns=["slow", "mid", "high", "unknown"], fill_value=0).astype(float)
        from sklearn.linear_model import LogisticRegression

        pb = LogisticRegression(max_iter=500).fit(lv, y).predict_proba(pd.get_dummies(te["pf_level"].fillna("unknown")).reindex(columns=["slow", "mid", "high", "unknown"], fill_value=0).astype(float))
        rep[name] = {"test2026_ll": round(pre._ll(yt, p), 5), "test2026_acc": round(float((p.argmax(1) == yt).mean()), 4), "test2026_ece": round(pre._ece(yt, p), 4),
                     "prior_ll": round(pre._ll(yt, np.tile(prior, (len(yt), 1))), 5), "pf_level_ll": round(pre._ll(yt, pb), 5), "pf_level_acc": round(float((pb.argmax(1) == yt).mean()), 4),
                     "n_train_races": int(len(tr)), "n_test_races": int(len(te))}
        print(name, rep[name])
    # 脚質別の表（想定の脚質ベース）
    hh = sc.load_horses()
    hh = hh.dropna(subset=["pf_style"])
    hh = hh[hh["year"] <= 2025]
    st = style_tables(hh)
    # シナリオ係数（2022〜2025 の馬、最終モデルの出力にシナリオごとの位置の効果を足す）
    h_all = sc.load_horses()
    th = sc.fit_params(h_all[h_all["year"].isin([2022, 2023, 2024, 2025])], "add")
    conf = {
        "version": pt.VERSION,
        "pace_classes": PACE, "shape_classes": SHAPE, "features": feats, "venue_code": meta["venue_code"], "grade_code": meta["grade_code"],
        "course": course_tables(d[d["year"] <= 2026]), "geo": geo_table(),
        "style_fit": st, "scenario_theta": {k: [round(float(x), 5) for x in v] for k, v in th.items()},
        "own_weight_with_lap": 0.35, "report": rep,
        "taxon": json.loads((D / "taxon_thresholds.json").read_text(encoding="utf-8")),
    }
    CONF.mkdir(parents=True, exist_ok=True)
    (CONF / "pace_type.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8")
    (CONF / "pace_type_models.json").write_text(json.dumps(models), encoding="utf-8")
    print("wrote", CONF)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["blend", "export"])
    a = ap.parse_args()
    return {"blend": cmd_blend, "export": cmd_export}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
