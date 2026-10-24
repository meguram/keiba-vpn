"""tp-72 補助分析（検定ではなく記述・機構・頑健性）。

python -m src.scripts.data.tp72_extra run [--out DIR]

A. クッション値・含水率の深掘り: 勝ち馬の脚質・枠の比率（出走馬の比率に対する比）、モデルの勝率の較正差を、クッション区分 × 脚質 × 枠で（年別の符号も）
B. 逃げはさみ・ブリンカーの機構: 実際の初角位置が普段（as-of の脚質）より前になるか（pos1_chg）を、同じ脚質の対照と比べる
C. 特徴量化の見込み: 性別 × 季節のずれ（logit のシフト）を発見期間で当てはめ、2025 の対数損失がどれだけ下がるか（学習は集計のみ）
D. 頑健性: 通過した型を V5 の OOF（LightGBM 単体、2022〜2024）の確率で再検定
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp72_eval import CONF, DESC, DISC, OUT, cluster_z, prepare


def A_cushion(d: pd.DataFrame) -> dict:
    out = {}
    win = d["win"] == 1
    for nm, col, surf in (("turf_cushion", "cush_bin", 1), ("dirt_moisture", "dmoist_bin", 0), ("turf_moisture", "tmoist_bin", 1)):
        s = d[(d["is_turf"] == surf) & d[col].notna() & d["style4"].isin(["nige", "senko", "sashi", "oikomi"]) & d["year"].between(2022, 2025)]
        rows = []
        for b in (0, 1, 2):
            sb = s[s[col] == b]
            if sb.empty:
                continue
            for st in ("nige", "senko", "sashi", "oikomi"):
                ss = sb[sb["style4"] == st]
                if len(ss) < 300:
                    continue
                e = (ss["y1"] - ss["p_win"])
                yrs = {int(y): float((g["y1"] - g["p_win"]).mean()) for y, g in ss.groupby("year")}
                rows.append({"bin": int(b), "style": st, "n": int(len(ss)), "win_rate": float(ss["y1"].mean()), "p_win": float(ss["p_win"].mean()), "resid_pt": float(e.mean() * 100),
                             "roi_win": float(ss["r1"].mean()), "roi_place": float(ss["r3"].mean()), "resid_by_year_pt": {k: v * 100 for k, v in yrs.items()}})
        out[nm] = rows
        # 勝ち馬の枠の比（内/外）
        dr = []
        for b in (0, 1, 2):
            sb = s[s[col] == b]
            for nm2, mk in (("inner", sb["inner"] == 1), ("outer", sb["outer"] == 1)):
                n = int(mk.sum())
                if n >= 300:
                    dr.append({"bin": int(b), "draw": nm2, "n": n, "win_rate": float(sb.loc[mk, "y1"].mean()), "p_win": float(sb.loc[mk, "p_win"].mean()), "resid_pt": float((sb.loc[mk, "y1"] - sb.loc[mk, "p_win"]).mean() * 100)})
        out[nm + "_draw"] = dr
    return out


def B_mechanism(d: pd.DataFrame) -> dict:
    out = {}
    base = d[d["pos1_chg"].notna() & d["style4"].isin(["nige", "senko", "sashi", "oikomi"]) & d["year"].between(2022, 2025)]
    for k in ("d1", "d2", "d3"):
        for t in ("pin", "p1"):
            sel = (base[f"{k}_{t}"] == 1) & (base[f"{k}_self_nl"] == 1)
            rows = []
            for st in ("senko", "sashi", "oikomi"):
                a = base[sel & (base["style4"] == st)]
                b = base[(~sel) & (base["style4"] == st) & (base[f"{k}_self_nl"] == 1)]
                if len(a) < 200:
                    continue
                diff = float(a["pos1_chg"].mean() - b["pos1_chg"].mean())
                se = float(np.sqrt(a["pos1_chg"].var() / len(a) + b["pos1_chg"].var() / len(b)))
                rows.append({"style": st, "n": int(len(a)), "pos1_chg_sel": float(a["pos1_chg"].mean()), "pos1_chg_ctl": float(b["pos1_chg"].mean()), "diff": diff, "z": diff / se if se > 0 else float("nan")})
            out[f"{k}_{t}"] = rows
    for bs in ("first", "cont", "off"):
        rows = []
        for st in ("nige", "senko", "sashi", "oikomi"):
            a = base[(base["blk_state"] == bs) & (base["style4"] == st)]
            b = base[(base["blk_state"] == "none") & (base["style4"] == st)]
            if len(a) < 200:
                continue
            diff = float(a["pos1_chg"].mean() - b["pos1_chg"].mean())
            se = float(np.sqrt(a["pos1_chg"].var() / len(a) + b["pos1_chg"].var() / len(b)))
            rows.append({"style": st, "n": int(len(a)), "pos1_chg_sel": float(a["pos1_chg"].mean()), "pos1_chg_ctl": float(b["pos1_chg"].mean()), "diff": diff, "z": diff / se if se > 0 else float("nan")})
        out[f"blinker_{bs}"] = rows
    return out


def _lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def C_feasibility(d: pd.DataFrame) -> dict:
    out = {}
    d = d[d["sex"].isin(["牡", "牝", "セ"])].copy()
    for keyname, key in (("sex×season", d["sex"] + "_" + d["season"]), ("sex×month", d["sex"] + "_" + d["month"].astype(str))):
        d["_k"] = key
        for tgt, (y, p) in {"win": ("y1", "p_win"), "top3": ("y3", "p_top3")}.items():
            tr = d[d["year"].isin(DISC)]
            g = tr.groupby("_k").apply(lambda x: (x[y] - x[p]).sum() / ((x[p] * (1 - x[p])).sum() + 20.0))
            shift = d["_k"].map(g).fillna(0.0).to_numpy()
            z0 = _lg(d[p].to_numpy())
            p2 = 1 / (1 + np.exp(-(z0 + shift)))
            res = {}
            for yr in (2025, 2026):
                m = (d["year"] == yr).to_numpy()
                res[str(yr)] = {"ll_before": _ll(d.loc[m, y].to_numpy(), d.loc[m, p].to_numpy()), "ll_after": _ll(d.loc[m, y].to_numpy(), p2[m])}
                res[str(yr)]["delta"] = res[str(yr)]["ll_after"] - res[str(yr)]["ll_before"]
            out[f"{keyname}|{tgt}"] = res
    return out


def D_robust(d: pd.DataFrame, names: dict[str, str]) -> dict:
    p = Path("data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet")
    if not p.exists():
        import os

        p = Path(os.environ.get("TP72_REPO", ".")) / p
    v5 = pd.read_parquet(p)[["race_id", "horse_id", "p_win", "p_top3"]].rename(columns={"p_win": "v5_win", "p_top3": "v5_top3"})
    v5["race_id"] = v5["race_id"].astype(str)
    v5["horse_id"] = v5["horse_id"].astype(str)
    d = d.merge(v5, on=["race_id", "horse_id"], how="left")
    d["e_v1"] = d["y1"] - d["v5_win"]
    d["e_v3"] = d["y3"] - d["v5_top3"]
    nrace = int(d["rc"].max()) + 1
    out = {}
    for nm, expr in names.items():
        mask = d.eval(expr, engine="python").fillna(False).to_numpy().astype(bool) & d["v5_win"].notna().to_numpy()
        row = {}
        for tag, sel in (("disc", np.isin(d["year"].to_numpy(), DISC)), ("y2022", d["year"].to_numpy() == 2022), ("y2023", d["year"].to_numpy() == 2023), ("y2024", d["year"].to_numpy() == 2024)):
            mm = mask & sel
            n = int(mm.sum())
            row[tag] = {"n": n}
            if n:
                z1, t1, _ = cluster_z(d, mm, "e_v1", nrace)
                z3, t3, _ = cluster_z(d, mm, "e_v3", nrace)
                row[tag].update({"z_v5_win": z1, "mean_v5_win": t1 / n, "z_v5_top3": z3, "mean_v5_top3": t3 / n})
        out[nm] = row
    return out


def run(out_dir: Path, cands_path: Path, passed: list[str]) -> dict:
    d = prepare()
    spec = {c["name"]: c["expr"] for c in json.loads(cands_path.read_text(encoding="utf-8"))["candidates"]}
    res = {"A_cushion": A_cushion(d), "B_mechanism": B_mechanism(d), "C_feasibility": C_feasibility(d), "D_robust": D_robust(d, {n: spec[n] for n in passed if n in spec})}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extra.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cands", default=str(Path(__file__).with_name("tp72_candidates.json")))
    ap.add_argument("--stage1", default=str(OUT / "stage1.json"))
    a = ap.parse_args()
    s1 = json.loads(Path(a.stage1).read_text(encoding="utf-8"))
    passed = list(dict.fromkeys(s1["passed_names"] + s1["top_names"][:10]))
    r = run(Path(a.out), Path(a.cands), passed)
    print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in r.items()})
