"""tp-77 追加4: 8 世代以内（父方・母方の両方）のクロスと、コース・馬場ごとの『来やすさ／来ない』の分析と特徴量。

python -m src.scripts.data.ptrain_cross scan    # 10 世代 JSON から、8 世代以内で同じ祖先が 2 回以上現れるクロスを縦長表に（成績は使わない）
python -m src.scripts.data.ptrain_cross build   # 条件別の来やすさを as-of で特徴量化（cross_rows.parquet）し、全期間の分析表（cross_report.json）を書く
成績 res=(頭数−着順)/(頭数−1)−0.5。条件別の来やすさ = 『そのクロスの条件セルでの平均 res』−『そのクロスの全体の平均 res』（疑似 40 走で縮約）。
セル: c1=面×距離帯、c2=面×馬場（良／稍重／重・不良）、c3=場×面。より前の開催日の結果だけを使う。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain")) / "cross"
PED = Path(os.environ.get("PTRAIN_PED10", ROOT / "data/local/horse_pedigree_10gen"))
TABLES = Path(os.environ.get("PTRAIN_TABLES", ROOT / "data/page_reference/tables"))
KEY = ["race_id", "horse_id"]
MAXG = 8
KAPPA = 40.0
MIN_SUPPORT = int(os.environ.get("PTRAIN_CROSS_SUPPORT", 60))     # そのクロスを持つ馬の数の下限
BANDS = [0, 1200, 1500, 1900, 2400, 99999]
CONDS = ("c1", "c2", "c3")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def crosses_of(ancestors: list[dict]) -> list[tuple]:
    """8 世代以内で 2 回以上現れる祖先ごとに (祖先ID, 名前, 出現数, 血量, 最小世代, 父方にも母方にもいる, 父方側の血量, 母方側の血量, 父方側の最小世代, 母方側の最小世代)。"""
    occ: dict[str, list] = {}
    for a in ancestors:
        pf, aid = a.get("path_fm") or "", a.get("horse_id")
        g = len(pf) if pf else a.get("generation")
        if not aid or not g or g > MAXG:
            continue
        o = occ.setdefault(aid, [a.get("name") or "", []])
        o[1].append((int(g), pf[:1]))
    out = []
    for aid, (nm, lst) in occ.items():
        if len(lst) >= 2:
            f = [g for g, sd in lst if sd == "F"]
            m = [g for g, sd in lst if sd == "M"]
            out.append((aid, nm, len(lst), float(sum(0.5**g for g, _ in lst)), min(g for g, _ in lst), bool(f) and bool(m),
                        float(sum(0.5**g for g in f)), float(sum(0.5**g for g in m)), min(f) if f else 99, min(m) if m else 99))
    return out


def _scan_prefix(args: tuple[str, list[str]]) -> pd.DataFrame:
    pref, hids = args
    rows = []
    for h in hids:
        p = PED / pref / f"{h}.json"
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for c in crosses_of(j.get("ancestors") or []):
            rows.append((h,) + c)
    return pd.DataFrame(rows, columns=["horse_id", "anc", "name", "n_occ", "blood", "gmin", "both_sides", "blood_f", "blood_m", "gmin_f", "gmin_m"])


def load_runs() -> pd.DataFrame:
    cols = ["race_id", "date", "horse_id", "venue", "surface", "distance", "field_size", "finish_position", "track_condition"]
    r = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in range(2019, 2026)], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"])].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"])
    for c in ("distance", "field_size", "finish_position"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["distance", "field_size", "finish_position"])
    r["res"] = (r["field_size"] - r["finish_position"]) / (r["field_size"] - 1).replace(0, np.nan) - 0.5
    r = r.dropna(subset=["res"])
    r["c1"] = r["surface"] + "|" + pd.cut(r["distance"], BANDS, labels=False).astype(str)
    r["c2"] = r["surface"] + "|" + r["track_condition"].map({"良": "良", "稍重": "稍重"}).fillna("重不良")
    r["c3"] = r["venue"].astype(str) + "|" + r["surface"]
    return r


def cmd_scan(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=["horse_id"]) for y in range(2019, 2026)])["horse_id"].astype(str)))
    by: dict[str, list[str]] = {}
    for h in ids:
        by.setdefault(h[:4], []).append(h)
    with ProcessPoolExecutor(int(os.environ.get("PTRAIN_CROSS_PROCS", 6))) as ex:
        parts = list(ex.map(_scan_prefix, list(by.items())))
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(OUT / "cross_long.parquet", index=False)
    _log(f"cross_long: {len(d)} 行、馬 {d['horse_id'].nunique()} / {len(ids)}、祖先 {d['anc'].nunique()}")


def _cum(L: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """keys 群ごとの、より前の開催日までの累積 n・res 合計（同日は含めない）。"""
    d = L.groupby(keys + ["date"], sort=True)["res"].agg(["size", "sum"]).reset_index()
    for c in ("size", "sum"):
        d[c] = d.groupby(keys)[c].cumsum() - d[c]
    return d


def asof_cross(L: pd.DataFrame) -> pd.DataFrame:
    """L: race_id, horse_id, date, anc, blood, res, c1..c3（走×クロス）。行ごと（race_id, horse_id）の集約特徴量。"""
    allc = _cum(L, ["anc"]).rename(columns={"size": "n_all", "sum": "s_all"})
    M = L.merge(allc, on=["anc", "date"], how="left")
    M["m_all"] = M["s_all"].fillna(0) / (M["n_all"].fillna(0) + KAPPA)
    feats = {"m_all": M["m_all"]}
    for c in CONDS:
        cc = _cum(L, ["anc", c]).rename(columns={"size": f"n_{c}", "sum": f"s_{c}"})
        M = M.merge(cc, on=["anc", c, "date"], how="left")
        M[f"lift_{c}"] = M[f"s_{c}"].fillna(0) / (M[f"n_{c}"].fillna(0) + KAPPA) - M["m_all"]
    M["w"] = M["blood"]
    g = M.groupby(KEY, sort=False)
    out = pd.DataFrame({"P77_X_n": g["anc"].size(), "P77_X_blood_max": g["blood"].max(), "P77_X_blood_sum": g["blood"].sum()})
    for name in ["m_all"] + [f"lift_{c}" for c in CONDS]:
        M["_wv"] = M[name] * M["w"]
        out[f"P77_X_{name}_max"] = M.groupby(KEY, sort=False)[name].max()
        out[f"P77_X_{name}_min"] = M.groupby(KEY, sort=False)[name].min()
        out[f"P77_X_{name}_wmean"] = M.groupby(KEY, sort=False)["_wv"].sum() / g["w"].sum()
    return out.reset_index().astype({c: "float32" for c in out.columns if c.startswith("P77_X_")})


BAL_EDGE = 0.25
GROUP_DIMS = ("g_bal", "g_lvl", "g_near")


def horse_groups(cl: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの、クロスの血量・位置（父方側／母方側／またがる）・近さのグループ。クロスのない馬は groups='なし'。"""
    g = cl.assign(span=np.where(cl["both_sides"], cl["blood"], 0.0), pat=np.where(~cl["both_sides"] & (cl["blood_f"] > 0), cl["blood"], 0.0),
                  mat=np.where(~cl["both_sides"] & (cl["blood_m"] > 0), cl["blood"], 0.0)).groupby("horse_id")
    h = pd.DataFrame({"P77_X_bF": g["blood_f"].sum(), "P77_X_bM": g["blood_m"].sum(), "P77_X_bspan": g["span"].sum(),
                      "P77_X_bpat": g["pat"].sum(), "P77_X_bmat": g["mat"].sum(), "P77_X_gmin": g["gmin"].min()})
    tot = h["P77_X_bF"] + h["P77_X_bM"]
    h["P77_X_balance"] = (h["P77_X_bM"] - h["P77_X_bF"]) / tot
    h["g_bal"] = np.where(h["P77_X_balance"] > BAL_EDGE, "母方優勢", np.where(h["P77_X_balance"] < -BAL_EDGE, "父方優勢", "同程度"))
    h["g_lvl"] = pd.cut(tot, [0, 0.03125, 0.0625, 0.125, 99], labels=["極薄", "薄", "中", "濃"], right=False).astype(str)
    h["g_near"] = pd.cut(h["P77_X_gmin"], [0, 4, 6, 99], labels=["近い(~4代)", "中(5~6代)", "遠い(7~8代)"]).astype(str)
    return h.reset_index()


def asof_group_lift(R: pd.DataFrame) -> pd.DataFrame:
    """R: race_id, horse_id, date, res, c1, c2 と g_* 列（全走）。グループ×セルの、より前の結果から『セルの全馬平均との差』（疑似40走で縮約）。"""
    out = pd.DataFrame(index=R.index)
    for c in ("c1", "c2"):
        base = _cum(R, [c]).rename(columns={"size": "nb", "sum": "sb"})
        B = R[[c, "date"]].merge(base, on=[c, "date"], how="left")
        mb = (B["sb"].fillna(0) / (B["nb"].fillna(0) + KAPPA)).to_numpy()
        for d in GROUP_DIMS:
            cu = _cum(R, [d, c]).rename(columns={"size": "n", "sum": "s"})
            M = R[[d, c, "date"]].merge(cu, on=[d, c, "date"], how="left")
            out[f"P77_X_{d}_lift_{c}"] = (M["s"].fillna(0) / (M["n"].fillna(0) + KAPPA)).to_numpy() - mb
    return out.astype("float32")


def group_report(R: pd.DataFrame) -> dict:
    """全期間（2020–2025）: グループ×セルの平均成績の、セル全馬平均との差。レース数と年ごとの符号。"""
    R = R[R["date"] >= "2020-01-01"].copy()
    R["year"] = R["date"].dt.year
    rep = {}
    for c in ("c1", "c2"):
        cm = R.groupby([c, "year"])["res"].mean().rename("cm").reset_index()
        for d in GROUP_DIMS:
            g = R.groupby([d, c]).agg(n_runs=("res", "size"), n_races=("race_id", "nunique"), mean=("res", "mean")).reset_index()
            cmm = R.groupby(c)["res"].mean().rename("cell_mean")
            g = g.merge(cmm, on=c)
            g["diff"] = g["mean"] - g["cell_mean"]
            y = R.groupby([d, c, "year"])["res"].mean().rename("ym").reset_index().merge(cm, on=[c, "year"])
            y["dd"] = y["ym"] - y["cm"]
            sg = y.groupby([d, c])["dd"].agg(pos=lambda s: int((s > 0).sum()), n_years="size").reset_index()
            rep[f"{d}|{c}"] = g.merge(sg, on=[d, c]).round(4).to_dict("records")
    return rep


def report(L: pd.DataFrame, names: dict[str, str]) -> dict:
    """全期間（2020–2025）の分析表。クロス×セルごとに、レース数・条件別の来やすさ・年ごとの符号の一致。"""
    L = L[L["date"] >= "2020-01-01"].copy()
    L["year"] = L["date"].dt.year
    base = L.groupby(["anc", "year"])["res"].mean().rename("by").reset_index()
    base_all = L.groupby("anc")["res"].mean().rename("all_mean")
    rep = {}
    for c in CONDS:
        g = L.groupby(["anc", c]).agg(n_runs=("res", "size"), n_races=("race_id", "nunique"), mean=("res", "mean")).reset_index()
        g = g[g["n_races"] >= 150].merge(base_all, on="anc")
        g["lift"] = (g["mean"] * g["n_runs"] / (g["n_runs"] + KAPPA)) - g["all_mean"]
        y = L.groupby(["anc", c, "year"])["res"].mean().rename("ym").reset_index().merge(base, on=["anc", "year"])
        y["d"] = y["ym"] - y["by"]
        sg = y.groupby(["anc", c])["d"].agg(n_years="size", pos=lambda s: int((s > 0).sum()), neg=lambda s: int((s < 0).sum())).reset_index()
        g = g.merge(sg, on=["anc", c])
        g["name"] = g["anc"].map(names)
        g["consistent"] = np.where(g["lift"] > 0, g["pos"], g["neg"])
        sel = g[(g["n_years"] >= 4)]
        rep[c] = {"up": sel.nlargest(25, "lift").round(4).to_dict("records"), "down": sel.nsmallest(25, "lift").round(4).to_dict("records"),
                  "share_consistent_ge5of6": float((sel["consistent"] >= 5).mean()) if len(sel) else None, "n_cells": int(len(sel))}
    return rep


def cmd_build(a) -> None:
    cl = pd.read_parquet(OUT / "cross_long.parquet")
    sup = cl.groupby("anc")["horse_id"].nunique()
    cl = cl[cl["anc"].isin(sup[sup >= MIN_SUPPORT].index)]
    r = load_runs()
    L = r[["race_id", "horse_id", "date", "res"] + list(CONDS)].merge(cl[["horse_id", "anc", "blood"]], on="horse_id", how="inner")
    _log(f"走×クロス {len(L)}（クロス祖先 {L['anc'].nunique()}）")
    feats = asof_cross(L)
    H = horse_groups(cl)
    RG = r[KEY + ["date", "res", "c1", "c2"]].merge(H[["horse_id"] + list(GROUP_DIMS)], on="horse_id", how="left")
    for d in GROUP_DIMS:
        RG[d] = RG[d].fillna("なし")
    gl = asof_group_lift(RG)
    allk = pd.concat([r[KEY].reset_index(drop=True), gl.reset_index(drop=True)], axis=1)
    allk = allk.merge(H[["horse_id", "P77_X_bF", "P77_X_bM", "P77_X_bspan", "P77_X_bpat", "P77_X_bmat", "P77_X_gmin", "P77_X_balance"]], on="horse_id", how="left")
    allk = allk.merge(feats, on=KEY, how="left")
    allk["P77_X_n"] = allk["P77_X_n"].fillna(0)
    for c in ("P77_X_bF", "P77_X_bM", "P77_X_bspan", "P77_X_bpat", "P77_X_bmat"):
        allk[c] = allk[c].fillna(0)
    allk.to_parquet(OUT.parent / "cross_rows.parquet", index=False)
    names = cl.drop_duplicates("anc").set_index("anc")["name"].to_dict()
    rep = report(L, names)
    rep["groups"] = group_report(RG)
    (OUT.parent / "cross_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"cross_rows: {allk.shape}、被覆 {float(allk['P77_X_m_all_max'].notna().mean()):.3f}、"
         f"年符号 5/6 以上のセル割合 {rep['c1']['share_consistent_ge5of6']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("scan").set_defaults(fn=cmd_scan)
    sp.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
