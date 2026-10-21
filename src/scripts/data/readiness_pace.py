"""導入可否の確認用: 追走難度 v2・ペース想定 v2・ラップ予想を、2026 年の保存済みの出力と実測で判定する（readiness-audit）。

  python -m src.scripts.data.readiness_pace --out /home/jovyan/work/readiness_out/pace.json

保存済みの出力（ホストの data/local/inference/ 配下の tracking_v2 / tracking_v2_eval / lap_forecast）と、2026 年のレース結果の
ラップ（race_result_flat の pace）を突き合わせる。基準は readiness_criteria.json の pace_difficulty_lap。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def actual_laps(tables: Path, year: int) -> pd.DataFrame:
    r = pd.read_parquet(tables / str(year) / "race_result_flat.parquet", columns=["race_id", "distance", "surface", "lap_times", "pace", "race_class"]).drop_duplicates("race_id")
    def parse(p):
        try:
            d = json.loads(p) if isinstance(p, str) else dict(p)
            return float(d.get("t3f") or d.get("first_half_3f")), float(d.get("l3f") or d.get("second_half_3f"))
        except Exception:  # noqa: BLE001
            return np.nan, np.nan
    pf = r["pace"].map(parse)
    r["a_first3f"] = [x[0] for x in pf]
    r["a_last3f"] = [x[1] for x in pf]
    r["race_id"] = r["race_id"].astype(str)
    return r


def lap_forecast_eval(inf: Path, act: pd.DataFrame, base: pd.DataFrame) -> dict:
    rows = []
    for f in glob.glob(str(inf / "lap_forecast" / "*" / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        rows.append({"race_id": str(d["race_id"]), "e_first": d.get("expected_first3f"), "e_last": d.get("expected_last3f"), "model": d.get("model"),
                     "lo": d.get("band80_lo"), "hi": d.get("band80_hi"), "exp": d.get("expected_laps")})
    p = pd.DataFrame(rows)
    m = p.merge(act, on="race_id", how="inner")
    out = {"stored": int(len(p)), "matched": int(len(m))}
    mm = m.dropna(subset=["e_first", "e_last", "a_first3f", "a_last3f"])
    b = base.groupby("distance").agg(b_first=("a_first3f", "mean"), b_last=("a_last3f", "mean")).reset_index()
    mm = mm.merge(b, on="distance", how="left")
    out["rows"] = int(len(mm))
    out["mae_first3f"] = float((mm["e_first"] - mm["a_first3f"]).abs().mean())
    out["mae_last3f"] = float((mm["e_last"] - mm["a_last3f"]).abs().mean())
    ok = mm.dropna(subset=["b_first", "b_last"])
    out["baseline_mae_first3f_by_distance_2025"] = float((ok["b_first"] - ok["a_first3f"]).abs().mean())
    out["baseline_mae_last3f_by_distance_2025"] = float((ok["b_last"] - ok["a_last3f"]).abs().mean())
    out["model_mae_first3f_same_rows"] = float((ok["e_first"] - ok["a_first3f"]).abs().mean())
    out["model_mae_last3f_same_rows"] = float((ok["e_last"] - ok["a_last3f"]).abs().mean())
    cov = []
    for _, r in m.iterrows():
        try:
            lt = r["lap_times"]
            lt = json.loads(lt) if isinstance(lt, str) else list(lt)
            lo, hi = r["lo"], r["hi"]
            if not lo or not hi or len(lt) != len(lo):
                continue
            for a_, l_, h_ in zip(lt, lo, hi):
                if l_ is not None and h_ is not None:
                    cov.append(1.0 if (l_ <= a_ <= h_) else 0.0)
        except Exception:  # noqa: BLE001
            continue
    out["band80_coverage_per_lap"] = float(np.mean(cov)) if cov else None
    out["model_share"] = p["model"].value_counts(normalize=True).round(3).to_dict()
    return out


def pace_eval(inf: Path, act: pd.DataFrame, races_all: set[str]) -> dict:
    rows, n_doc = [], 0
    for f in glob.glob(str(inf / "tracking_v2" / "*" / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        n_doc += 1
        pu = d.get("pace_unified") or {}
        pf = d.get("pace_forecast") or {}
        rows.append({"race_id": str(d["race_id"]), "n": d.get("n_horses"), "n_v2": d.get("n_with_v2"), "z": pu.get("z"), "label": pu.get("label"), "pf_level": pf.get("level")})
    p = pd.DataFrame(rows)
    m = p.merge(act, on="race_id", how="inner").dropna(subset=["z", "a_first3f"])
    out = {"docs": n_doc, "race_coverage_vs_results": n_doc / max(len(races_all), 1), "horse_coverage": float(p["n_v2"].sum() / max(p["n"].sum(), 1))}
    cs, ws = [], []
    for (sf, dist), g in m.groupby(["surface", "distance"]):
        if len(g) >= 20:
            c = g["z"].astype(float).corr(g["a_first3f"].astype(float), method="spearman")
            if pd.notna(c):
                cs.append(c)
                ws.append(len(g))
    out["spearman_z_vs_first3f_by_surface_distance"] = float(np.average(cs, weights=ws)) if cs else None
    out["groups"] = len(cs)
    # 面×距離ごとに前半 3F を標準化したうえでの全体の相関（設計時の値 0.45 に近い測り方）
    g = m.groupby(["surface", "distance"])["a_first3f"]
    sd = g.transform("std")
    m2 = m[(g.transform("size") >= 10) & (sd > 0)].copy()
    m2["zf"] = (m2["a_first3f"] - g.transform("mean")[m2.index]) / sd[m2.index]
    out["pearson_z_vs_standardized_first3f"] = float(m2["z"].astype(float).corr(m2["zf"]))
    out["races_used"] = int(len(m2))
    return out


def tracking_week_eval(inf: Path) -> dict:
    t = inf / "tracking_v2_eval" / "_trend.json"
    if not t.exists():
        return {"trend": None}
    d = json.loads(t.read_text(encoding="utf-8"))
    w = pd.DataFrame(d["weeks"])
    n = w["n_pairs"].to_numpy(float)
    out = {"weeks": int(len(w)), "pairs": int(n.sum()), "corr_drain_y_weighted": float(np.average(w["corr_drain_y"], weights=n)),
           "corr_min": float(w["corr_drain_y"].min()), "weeks_positive": int((w["corr_drain_y"] > 0).sum()),
           "decile_monotonic_weighted": float(np.average(w["decile_monotonic_spearman"], weights=n)),
           "top_minus_bottom_y_weighted": float(np.average(w["decile_top_minus_bottom_y"], weights=n)),
           "coverage_weighted": float(np.average(w["coverage"], weights=n))}
    out["last4"] = w.tail(4)[["week_start", "n_pairs", "corr_drain_y", "decile_monotonic_spearman"]].to_dict("records")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="data/page_reference/tables")
    ap.add_argument("--inference", default="data/local/inference")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tables, inf = Path(a.tables), Path(a.inference)
    act = actual_laps(tables, 2026)
    base = actual_laps(tables, 2025)
    races_all = set(act["race_id"][act["race_id"].str[4:6].isin([f"{i:02d}" for i in range(1, 11)])])
    out = {"lap_forecast": lap_forecast_eval(inf, act, base), "pace_v2": pace_eval(inf, act, races_all), "tracking_week": tracking_week_eval(inf)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "tracking_week"}, ensure_ascii=False, default=str)[:2500])


if __name__ == "__main__":
    main()
