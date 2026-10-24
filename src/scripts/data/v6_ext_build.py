"""v6 の追加候補の群を作る（年ごとの as-of。行キーは race_id, horse_id）。

python -m src.scripts.data.v6_ext_build pt    # 展開タイプ系（tp-61）: 事前の展開確率・ハナ主張馬・脚質×展開の期待有利度
python -m src.scripts.data.v6_ext_build gb    # コース別の枠順バイアス（年ごとの as-of 係数）
python -m src.scripts.data.v6_ext_build cb    # クラス基準（年齢×クラスの 2 着馬の指数の突破ライン、年ごとの as-of）
出力: $V6_OUT（既定 data/local/modeling/training_pipeline_v4/v6）/groups/{PT,GB,CB}.parquet
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import v6_ext as vx
from src.pipeline.features import v6_groups as vg

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp61"
TABLES = ROOT / "data/page_reference/tables"
OUT = Path(os.environ.get("V6_OUT") or ROOT / "data/local/modeling/training_pipeline_v4/v6") / "groups"
KEY = ["race_id", "horse_id"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _horses() -> pd.DataFrame:
    h = pd.read_parquet(D / "horse.parquet", columns=["race_id", "date", "venue", "surface", "distance", "field_size", "horse_number", "horse_id", "year",
                                                        "early_norm", "pass_first", "is_top3"])
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    h["course"] = h["venue"].astype(str) + "|" + h["surface"].astype(str) + "|" + h["distance"].astype(float).astype(int).astype(str)
    return h


def build_pt() -> None:
    from src.scripts.data import tp61_hana as hana
    from src.scripts.data import tp61_pre as pre

    OUT.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(D / "pre.parquet")
    pr["race_id"] = pr["race_id"].astype(str)
    feats = [c for c in pre.RACE_FEATS if c in pr.columns] + [c for c in pre.HANA if c in pr.columns]
    lab = pr[pr["pace_cls"].isin(pre.PACE) & pr["shape_cls"].isin(pre.SHAPE)]
    parts = []
    for y in sorted(int(v) for v in pr["year"].unique()):
        tr = lab[lab["year"].isin(vx.asof_train_years(y))]
        te = pr[pr["year"] == y]
        o = pd.DataFrame({"race_id": te["race_id"].to_numpy()})
        for name, classes, col in (("pace", pre.PACE, "pace_cls"), ("shape", pre.SHAPE, "shape_cls")):
            m = pre._fit_lgb(tr[feats], tr[col].map({c: i for i, c in enumerate(classes)}).to_numpy(), 3)
            p = m.predict_proba(te[feats])
            for i, c in enumerate(classes):
                o[f"pt_p_{c}"] = p[:, i].astype("float32")
        parts.append(o)
        _log(f"{y}: 学習 {len(tr)}R → 予測 {len(te)}R")
    P = pd.concat(parts, ignore_index=True)
    P["pt_pace_ev"] = P["pt_p_high"] - P["pt_p_slow"]
    P["pt_shape_ev"] = P["pt_p_stretched"] - P["pt_p_compact"]
    pc = P[["pt_p_slow", "pt_p_mid", "pt_p_high"]].to_numpy(float)
    P["pt_pace_ent"] = -(pc * np.log(np.clip(pc, 1e-9, 1))).sum(axis=1)
    hn = ["hn_claim", "hn_claim_strong", "hn_l1", "hn_sum3", "hn_pairs", "hn_outer", "hn_inner", "hn_uncertain"]
    R = P.merge(pr[["race_id"] + hn].rename(columns={c: f"pt_{c}" for c in hn}), on="race_id", how="left")

    h = _horses()
    pf_h, _ = pre._pf_all()
    pf_h["race_id"] = pf_h["race_id"].astype(str)
    h = h.merge(pf_h.drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left")
    a = hana.horse_asof(h.assign(date=h["date"]))
    a["race_id"], a["horse_id"] = a["race_id"].astype(str), a["horse_id"].astype(str)
    h = h.merge(a[["race_id", "horse_id", "lead5", "n5"]], on=KEY, how="left")
    h = h.merge(pr[["race_id", "pace_cls", "shape_cls"]], on="race_id", how="left")
    idx = h["pf_style"].map({s: i for i, s in enumerate(vx.STYLES)}).fillna(-1).astype(int).to_numpy()
    h = h.merge(R, on="race_id", how="left")
    parts = []
    for y in sorted(int(v) for v in h["year"].unique()):
        L = vx.style_lift_table(h[h["year"].isin(vx.asof_train_years(y)) & h["pf_style"].notna() & h["pace_cls"].isin(pre.PACE)])
        m = (h["year"] == y).to_numpy()
        pp = h.loc[m, ["pt_p_slow", "pt_p_mid", "pt_p_high"]].to_numpy(float)
        ss = h.loc[m, ["pt_p_compact", "pt_p_normal", "pt_p_stretched"]].to_numpy(float)
        f = vx.expected_style_fit(pp, ss, idx[m], L)
        f.index = h.index[m]
        parts.append(f)
    F = pd.concat(parts).reindex(h.index)
    F["pt_style_score"] = ((F["pt_style_lift"] - 1.0) / 0.2).clip(-1, 1)
    F["pt_is_hana"] = ((h["n5"] >= 3) & (h["lead5"] >= 0.4)).astype(float).where(h["lead5"].notna())
    F["pt_hana_strong"] = ((h["n5"] >= 3) & (h["lead5"] >= 0.6)).astype(float).where(h["lead5"].notna())
    base = pd.concat([h[["race_id", "horse_id"]], F], axis=1)
    rel = vg.race_rel(base, ["pt_style_lift", "pt_style_sd"], prefix="pt_r_")
    cols = ["pt_p_slow", "pt_p_mid", "pt_p_high", "pt_p_compact", "pt_p_normal", "pt_p_stretched", "pt_pace_ev", "pt_shape_ev", "pt_pace_ent"] + [f"pt_{c}" for c in hn]
    out = pd.concat([h[["race_id", "horse_id"]], h[cols].astype("float32"), F.astype("float32"), rel], axis=1)
    out = out.drop_duplicates(KEY)
    out.to_parquet(OUT / "PT.parquet", index=False)
    _log(f"PT: {out.shape}, 非欠損 {out['pt_style_lift'].notna().mean():.3f}")


def build_gb() -> None:
    from src.scripts.data import tp61_hana as hana
    from src.scripts.data import tp61_pre as pre

    OUT.mkdir(parents=True, exist_ok=True)
    h = _horses()
    pf_h, _ = pre._pf_all()
    pf_h["race_id"] = pf_h["race_id"].astype(str)
    h = h.merge(pf_h.drop_duplicates(["race_id", "horse_number"])[["race_id", "horse_number", "pf_early"]], on=["race_id", "horse_number"], how="left")
    a = hana.horse_asof(h)
    a["race_id"], a["horse_id"] = a["race_id"].astype(str), a["horse_id"].astype(str)
    h = h.merge(a[["race_id", "horse_id", "lead5", "n5"]], on=KEY, how="left")
    hn = pd.read_parquet(D / "hana.parquet", columns=["race_id", "hn_firstcurve"])
    hn["race_id"] = hn["race_id"].astype(str)
    h = h.merge(hn, on="race_id", how="left")
    stats = vx.gate_year_stats(h)
    surface_of = h.drop_duplicates("course").set_index("course")["surface"].to_dict()
    parts = []
    for y in sorted(int(v) for v in h["year"].unique()):
        coeff = vx.gate_coeff_asof(stats, y, surface_of)
        m = h["year"] == y
        g = vx.gate_features(h[m], coeff)
        parts.append(g)
        _log(f"{y}: コース {sum(1 for k in coeff if not k.startswith('_'))} 件の係数, 面平均 { {k: round(v, 4) for k, v in coeff.items() if k.startswith('_')} }")
    G = pd.concat(parts).reindex(h.index)
    rel = vg.race_rel(pd.concat([h[["race_id"]], G], axis=1), ["gb_pos_adj"], prefix="gb_r_")
    out = pd.concat([h[KEY], G, rel], axis=1).drop_duplicates(KEY)
    out.to_parquet(OUT / "GB.parquet", index=False)
    _log(f"GB: {out.shape}, gb_k 非欠損 {out['gb_k'].notna().mean():.3f}")


def build_cb() -> None:
    from src.pipeline.inference import class_benchmark as cbm
    from src.scripts.data import class_benchmark_build as cbb

    OUT.mkdir(parents=True, exist_ok=True)
    years = [y for y in range(2020, 2027) if (TABLES / str(y) / "megu_index_flat.parquet").exists()]
    df = cbb.load(TABLES, years)
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df["idx"] = df["idx"].where(df["ok"])
    df["date"] = pd.to_datetime(df["date"])
    df["dist"] = pd.to_numeric(df["distance"], errors="coerce")
    line = pd.Series(np.nan, index=df.index)
    med = pd.Series(np.nan, index=df.index)
    for y in years:
        base = cbb.basis_rows(df[df["year"] < y], "second")
        if base.empty:
            continue
        doc = {"cells": cbb.build_cells(base)}
        m = df["year"] == y
        keys = df.loc[m, ["race_class", "grade", "race_name", "surface", "distance"]].astype(str).drop_duplicates()
        cache = {}
        for r in keys.itertuples(index=False):
            hit = cbm.lookup(r.race_class, r.grade, r.race_name, r.surface, r.distance, doc)
            cache[tuple(r)] = (hit["line"], hit["median"]) if hit else (np.nan, np.nan)
        kk = list(zip(*[df.loc[m, c].astype(str) for c in ("race_class", "grade", "race_name", "surface", "distance")]))
        line[m] = [cache[k][0] for k in kk]
        med[m] = [cache[k][1] for k in kk]
        _log(f"{y}: 基準セル {len(doc['cells'])}（{base['race_id'].nunique()} レース）、突破ライン付き {line[m].notna().mean():.3f}")
    runs = df[["horse_id", "date", "idx", "surf", "dist"]].copy()
    F = vx.prior_window_features(runs, line)
    F["cb_median"] = med
    F["cb_max5_over_median"] = F["cb_max5"] - F["cb_median"]
    out = pd.concat([df[KEY], F.drop(columns=["cb_prior_n"]).astype("float32")], axis=1).drop_duplicates(KEY)
    out.to_parquet(OUT / "CB.parquet", index=False)
    _log(f"CB: {out.shape}, cb_max5_gap 非欠損 {out['cb_max5_gap'].notna().mean():.3f}")


def build_own() -> None:
    """馬主名の表記の揺れ（（株）・株式会社 など）をそろえた馬主の as-of 成績（走数・勝ち/3着内の超過）と、正規化で増えた履歴の走数。"""
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("V6_WORK") or "/home/jovyan/work")
    runs = pd.read_parquet(work / "acq/runs_all.parquet")
    runs = runs[runs["is_jra"].astype(bool)].copy()
    runs["race_id"], runs["horse_id"] = runs["race_id"].astype(str), runs["horse_id"].astype(str)
    runs["date"] = pd.to_datetime(runs["date"])
    own = pd.read_parquet(work / "acq/owners_names.parquet", columns=["horse_id", "owner"])
    own["horse_id"] = own["horse_id"].astype(str)
    runs = runs.merge(own.drop_duplicates("horse_id"), on="horse_id", how="left")
    runs["owner"] = runs["owner"].fillna("").astype(str).str.strip()
    runs["owner_n"] = runs["owner"].map(vx.normalize_owner)
    raw = vx.owner_asof(runs.assign(okey=runs["owner"]), "okey").rename(columns={"n": "n_raw", "ex_win": "ex_win_raw", "ex_t3": "ex_t3_raw"})
    nrm = vx.owner_asof(runs.assign(okey=runs["owner_n"]), "okey")
    spell = runs[runs["owner"] != ""].groupby("owner_n")["owner"].nunique()
    out = nrm.merge(raw[["race_id", "horse_id", "n_raw"]], on=KEY, how="left")
    out["on_n"] = out["n"]
    out["on_gain_n"] = out["n"] - out["n_raw"]
    out["on_ex_win"], out["on_ex_t3"] = out["ex_win"], out["ex_t3"]
    out = out.merge(runs[KEY + ["owner_n"]].drop_duplicates(KEY), on=KEY, how="left")
    out["on_multi"] = out["owner_n"].map(spell).fillna(1).gt(1).astype(float)
    out = out[KEY + ["on_n", "on_gain_n", "on_ex_win", "on_ex_t3", "on_multi"]].drop_duplicates(KEY)
    out[["on_n", "on_gain_n", "on_ex_win", "on_ex_t3", "on_multi"]] = out[["on_n", "on_gain_n", "on_ex_win", "on_ex_t3", "on_multi"]].astype("float32")
    out.to_parquet(OUT / "OWNN.parquet", index=False)
    _log(f"OWNN: {out.shape}, 正規化で履歴が増えた行 {(out['on_gain_n'] > 0).mean():.4f}, 複数表記の馬主の行 {out['on_multi'].mean():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=["pt", "gb", "cb", "own"])
    a = ap.parse_args()
    {"pt": build_pt, "gb": build_gb, "cb": build_cb, "own": build_own}[a.group]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
