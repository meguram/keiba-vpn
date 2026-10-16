"""3 つの独立収集（A = JRA 公式の本文と図、B = netkeiba・競馬ラボ・Wikipedia、C = JRA-VAN・Wikipedia・補助サイト）から、JRA コースの形状マスタを作る。

  python -m src.research.race.build_course_master --a /tmp/jra_course_master_raw.json --b /tmp/jra_course_master_B.json --c /tmp/jra_course_master_C.json --out data/meta/course_master

各値に「何の情報源が一致したか」を付ける。一致しない値・1 つの情報源だけの値は ``unverified`` に入れ、特徴量では使わないか確信度を下げて使う。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import pandas as pd

from src.research.race import course_master as cm

TOL_POS = 45.0        # 坂の位置（ゴール前 m）の許容
TOL_RISE = 0.35       # 坂の高低差（m）の許容


def _goal(row: dict, src: str) -> dict | None:
    g = row.get("goal_slope")
    if g is None:
        return None
    if isinstance(g, dict):
        ex = g.get("exists", g.get("present"))
        return {"exists": ex, "start": cm.to_num(g.get("start_before_goal_m")), "end": cm.to_num(g.get("end_before_goal_m")), "rise": cm.to_num(g.get("rise_m", g.get("height_m"))), "grad": cm.to_num(g.get("max_grad_pct", g.get("gradient_pct")))}
    return {"exists": cm.has_goal_slope(g), "start": None, "end": None, "rise": None, "grad": None}


def _agree(vals: dict[str, float | None], tol: float) -> tuple[float | None, list[str], str]:
    have = {k: v for k, v in vals.items() if v is not None}
    if not have:
        return None, [], "欠損"
    if len(have) == 1:
        k, v = next(iter(have.items()))
        return v, [k], "未確認"
    med = median(have.values())
    ok = [k for k, v in have.items() if abs(v - med) <= tol]
    if len(ok) == len(have):
        return median(have.values()), ok, "合意"
    if len(ok) >= 2:
        return median([have[k] for k in ok]), ok, "合意(外れ値あり)"
    return None, [], "不一致"


def build(a: list[dict], b: list[dict], c: list[dict], n_corner_obs: dict | None = None) -> tuple[list[dict], list[dict]]:
    srcs = {"A": {cm.row_key(r): r for r in a}, "B": {cm.row_key(r): r for r in b}, "C": {cm.row_key(r): r for r in c}}
    keys = sorted(set().union(*[set(d) for d in srcs.values()]))
    rows, issues = [], []
    for k in keys:
        by = {n: d.get(k) for n, d in srcs.items() if d.get(k) is not None}
        base = by.get("A") or by.get("C") or by.get("B")
        out = {"venue_code": k[0], "surface": k[1], "distance": k[2], "variant": k[3], "venue": base.get("venue"), "direction": base.get("direction"), "sources_present": sorted(by)}
        for f, tol in (("lap_length_m", 30.0), ("final_straight_m", 10.0), ("elevation_range_m", 0.4)):
            v, ok, st = _agree({n: cm.to_num(r.get(f)) for n, r in by.items()}, tol)
            out[f], out[f + "_agree"], out[f + "_status"] = v, ok, st
            if st in ("不一致", "未確認", "欠損"):
                issues.append({"key": k, "field": f, "status": st, "values": {n: cm.to_num(r.get(f)) for n, r in by.items()}})
        # 最初の曲線までの距離（ポケットの合流曲線を含む。B・C の定義）と、本線の最初のコーナーまで（A の定義）
        v, ok, st = _agree({n: cm.to_num(by[n].get("dist_start_to_first_corner_m")) for n in ("B", "C") if n in by}, 40.0)
        out["first_curve_m"], out["first_curve_agree"], out["first_curve_status"] = v, ok, st
        out["first_main_corner_m_A"] = cm.to_num(by["A"].get("dist_start_to_first_corner_m")) if "A" in by else None
        if out["first_main_corner_m_A"] is not None and v is not None and abs(out["first_main_corner_m_A"] - v) > 60:
            out["first_corner_definition_differs"] = True
        # 坂
        goals = {n: _goal(r, n) for n, r in by.items()}
        ex = [g["exists"] for g in goals.values() if g and g["exists"] is not None]
        out["goal_slope_votes"] = {n: (g or {}).get("exists") for n, g in goals.items()}
        if ex:
            out["goal_slope_exists"] = sum(bool(x) for x in ex) * 2 > len(ex)
            out["goal_slope_unanimous"] = len(set(map(bool, ex))) == 1
            if not out["goal_slope_unanimous"]:
                issues.append({"key": k, "field": "goal_slope_exists", "status": "不一致", "values": out["goal_slope_votes"]})
        for f in ("start", "end", "rise"):
            v, ok, st = _agree({n: (g or {}).get(f) for n, g in goals.items()}, TOL_POS if f != "rise" else TOL_RISE)
            out[f"goal_slope_{f}"], out[f"goal_slope_{f}_status"] = v, st
        # コーナー数（幾何学的な数。ポケットの合流曲線は数えない）は A、記録上の数（JRA の通過順位の記録点）は B・C と実データ
        out["n_corners_geom_A"] = cm.to_num(by["A"].get("n_corners")) if "A" in by else None
        out["n_corners_BC"] = {n: cm.to_num(by[n].get("n_corners")) for n in ("B", "C") if n in by}
        if n_corner_obs and (k[0], k[1], k[2]) in n_corner_obs:
            out["n_corners_record_mode"] = n_corner_obs[(k[0], k[1], k[2])]
        for f in ("start_position", "corner_radius_class", "turf_type"):
            out[f + "_A"] = by["A"].get(f) if "A" in by else None
        if "A" in by:
            for f in ("other_slopes", "corner_sequence", "start_geometry", "spiral_curve", "width_m"):
                out[f] = by["A"].get(f)
        out["confidence_A"] = by["A"].get("confidence") if "A" in by else None
        rows.append(out)
    return rows, issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--c", required=True)
    ap.add_argument("--fingerprint", default="/tmp/course_fingerprint.json")
    ap.add_argument("--out", type=Path, default=Path("data/meta/course_master"))
    a = ap.parse_args()
    fp = json.loads(Path(a.fingerprint).read_text(encoding="utf-8")) if Path(a.fingerprint).exists() else []
    obs = {(r["venue_code"], r["surface"], r["distance"]): r["n_corners_mode"] for r in fp}
    rows, issues = build(cm.load(a.a), cm.load(a.b), cm.load(a.c), obs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "jra_courses.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (a.out / "verification_issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    df = pd.DataFrame(rows)
    summ = {f: df[f + "_status"].value_counts().to_dict() for f in ("lap_length_m", "final_straight_m", "elevation_range_m", "first_curve", "goal_slope_start", "goal_slope_end", "goal_slope_rise") if (f + "_status") in df or (f.replace("first_curve", "first_curve") + "_status") in df}
    print(json.dumps({"rows": len(rows), "issues": len(issues), "status": summ, "goal_slope_unanimous": df["goal_slope_unanimous"].value_counts().to_dict()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
