"""走法が合う Best5 の検証用の表: 評価するレースごと・馬ごとに、実力・適性（コース特注ゾーン・馬場特注ゾーン）・走法の軸・結果を 1 行にまとめる。

目的（docs/todos/best5_model_plan.md の段階 1）: コースの走法適性が結果にどう寄与するか、実力・同じ適性の馬の頭数・ペースとの絡み合いを測る材料にする。本番は触らない。

時刻の安全:
  - コース特注ゾーン（地図）は CUT より前の走だけで作り直す（data/local/research/best5/course_maps_cut<日付>.json。無ければ作る）。
  - 馬場特注ゾーンは running_map_day の通常の作り（そのレースより前の結果だけ。今日の前のレース＋前日、直前ほど重い）。結果は手元のレース結果の表から読む（GCS は使わない）。
  - 実力は予想モデル v6e の、学習に使っていない年の予測（final_eval_v6e/test2025_predictions.parquet。2025 年）。

使い方:
    python -m src.scripts.data.best5_table --from 2025-07-01 --to 2025-12-31 [--limit-days 3]
出力: data/local/research/best5/rows/<日付>.parquet（日ごと。あるものは飛ばすので、止まっても続きから）
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/research/best5"
ROWS = OUT / "rows"
TABLES = ROOT / "data/page_reference/tables"
PRED2025 = ROOT / "data/local/modeling/training_pipeline/final_eval_v6e/test2025_predictions.parquet"
CUT = pd.Timestamp("2025-07-01")
log = logging.getLogger("best5_table")


def flat(years: range) -> pd.DataFrame:
    from src.research.columns.compare_series import jump_mask

    cols = ["race_id", "date", "venue", "surface", "distance", "track_condition", "field_size", "finish_position", "horse_id", "horse_number",
            "bracket_number", "horse_name", "odds", "popularity", "race_name", "race_class"]
    r = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"]) & ~jump_mask(r)].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"].astype(str).str[:10])
    return r


def cut_maps(cut: pd.Timestamp = CUT) -> dict:
    p = OUT / f"course_maps_cut{cut:%Y%m%d}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    from src.research.race import running_map as rm
    from src.research.race import running_profile as rp

    d = rm._data()
    mc = d["metrics"][d["metrics"]["date"] < cut]
    table = rp.profile_table(mc, d["sires"], cols=[col for _, col, *_ in rm.AXES])
    maps = rm._build_course_maps(mc, table, d["ref"]["quantiles"], d["ref"]["sd"])
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(maps, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return maps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="2025-07-01")
    ap.add_argument("--to", dest="d_to", default="2025-12-31")
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from src.research.race import running_map as rm
    from src.research.race import running_map_course as rmc
    from src.research.race.running_map_best5 import _smooth_lookup

    maps = cut_maps()
    rm.WET_AXIS = {}                                     # 道悪の地図は作らない（良馬場の地図だけで足りる。時間を半分にする）
    rm.course_map_for = lambda cond, wet=False: rmc.lookup(maps.get("wet", {}) if wet else maps, cond.get("venue"), cond.get("surface"), cond.get("distance"))
    rm.aptitude_for = lambda cond: None

    r = flat(range(2024, 2027))
    ev = r[(r["date"] >= a.d_from) & (r["date"] <= a.d_to)]
    pred = pd.read_parquet(PRED2025)
    pred["race_id"], pred["horse_id"] = pred["race_id"].astype(str), pred["horse_id"].astype(str)
    by_race = {rid: g for rid, g in r.groupby("race_id", sort=False)}
    cards: dict[str, dict] = {}

    def load_result(rid: str):
        if rid not in cards:
            g = by_race.get(rid)
            cards[rid] = None if g is None else {"surface": g["surface"].iloc[0], "track_condition": g["track_condition"].iloc[0],
                                                  "entries": [{"horse_id": h, "finish_position": f} for h, f in zip(g["horse_id"], pd.to_numeric(g["finish_position"], errors="coerce"))
                                                              if f == f and f >= 1]}
        return cards[rid]

    ROWS.mkdir(parents=True, exist_ok=True)
    days = sorted(ev["date"].dt.strftime("%Y-%m-%d").unique())
    if a.limit_days:
        days = days[: a.limit_days]
    t0 = time.time()
    for di, day in enumerate(days):
        path = ROWS / f"{day}.parquet"
        if path.exists():
            continue
        rows = []
        for rid, g in ev[ev["date"] == day].groupby("race_id", sort=True):
            g = g.sort_values("horse_number")
            if len(g) < 5:
                continue
            cond = {"date": day, "venue": g["venue"].iloc[0], "surface": g["surface"].iloc[0], "distance": int(g["distance"].iloc[0]),
                    "track_condition": rm._norm_condition(g["track_condition"].iloc[0])}
            entries = [{"horse_id": h, "horse_number": n, "bracket_number": b, "horse_name": nm}
                       for h, n, b, nm in zip(g["horse_id"], g["horse_number"], g["bracket_number"], g["horse_name"])]
            try:
                out = rm.build_running_map(entries, rid, cond, load_result=load_result)
            except Exception as exc:       # 1 レースの失敗は他を止めない
                log.warning("失敗 %s: %s", rid, exc)
                continue
            if out.get("status") != "ok" or not out.get("horses"):
                continue
            hs = out["horses"]
            xy = np.array([[h["x"], h["y"]] for h in hs], float)
            ext = out.get("extent")
            apt = out.get("aptitude") or {}
            dt = out.get("day_trend") or {}
            heat = (dt.get("heat") or {})
            cl = apt.get("lift")
            c_s, c_raw = _smooth_lookup(cl, ext, xy), _smooth_lookup(cl, ext, xy, 0)
            d_s, d_raw = _smooth_lookup(heat.get("lift"), ext, xy), _smooth_lookup(heat.get("lift"), ext, xy, 0)
            sig = heat.get("sig") or ""
            cm = rm.course_map_for(cond)
            res = g.set_index("horse_id")
            for i, h in enumerate(hs):
                x = res.loc[h["horse_id"]]
                rows.append({
                    "race_id": rid, "date": day, "venue": cond["venue"], "surface": cond["surface"], "distance": cond["distance"],
                    "going": cond["track_condition"], "n_field": len(g), "horse_id": h["horse_id"], "horse_number": h.get("horse_number"),
                    "x": h["x"], "y": h["y"], "c_s": c_s[i], "c_raw": c_raw[i], "d_s": d_s[i], "d_raw": d_raw[i],
                    "finish": pd.to_numeric(x["finish_position"], errors="coerce"), "odds": x["odds"], "popularity": x["popularity"],
                    "estimated": bool(h.get("estimated")), "n_lap_runs": h.get("n_lap_runs"),
                    "map_axes": ",".join(cm["axes"]) if cm else "", "map_level": (cm or {}).get("level"),
                    "day_status": dt.get("status"), "day_races_today": dt.get("n_races_today"), "day_races_prev": dt.get("n_races_prev"),
                    "day_runs": dt.get("n_runs"), "day_sig2": sig.count("2"), "day_sig1": sig.count("1"),
                    **{f"pct_{k}": v for k, v in (h.get("percentiles") or {}).items()},
                })
        if rows:
            df = pd.DataFrame(rows)
            df = df.merge(pred[["race_id", "horse_id", "p_win", "p_top3"]], on=["race_id", "horse_id"], how="left")
            df.to_parquet(path, index=False)
        log.info("%s 完了 (%d/%d 日) 行 %d 経過 %.0f 秒", day, di + 1, len(days), len(rows), time.time() - t0)


if __name__ == "__main__":
    main()
