"""tp-77 追加7: 風向・風速とコースの向きから、区間ごとの向かい風・横風の成分を作り、トラックバイアス・位置取り・ペースへの影響を調べる（調査・分析）。

python -m src.scripts.data.ptrain_wind headings         # コース別の走行方位（ゴールから逆に辿った周回路）を確認
python -m src.scripts.data.ptrain_wind build            # レースごとの風の成分 → wind_race.parquet、分析表 → wind_report.json
風データ: data/page_reference/tables/jma_weather_hourly_flat.parquet（python -m src.scraper.jma_weather export。気象庁の時別値。
wind_dir_deg=風が吹いてくる方位（北=0 時計回り）、その時刻の直前10分の平均風）。
走行方位: routes.json の pts_q（q=0 がゴール、8m ごとに後ろ＝走ってきた側へ並ぶ）。走る向きは q が減る向き。
向かい風成分 = 風速 × cos(風向 − 進行方位)（正=向かい風）、横風成分 = 風速 × sin(風向 − 進行方位)（正=進行方向の右から）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
ROUTES = ROOT / "data/meta/course_master/geo/routes.json"
WIND = Path(os.environ.get("PTRAIN_WIND", ROOT / "data/page_reference/tables/jma_weather_hourly_flat.parquet"))
TABLES = Path(os.environ.get("PTRAIN_TABLES", ROOT / "data/page_reference/tables"))
OOF = Path(os.environ.get("PTRAIN_OOF", ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_z:RB@e.parquet"))
PTS_STEP = 8.0
KEY = ["race_id", "horse_id"]


def bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """東=x、北=y の変位から進行方位（北=0、時計回り、度）。"""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def course_headings(routes: dict) -> dict[tuple[str, str, int], list[np.ndarray]]:
    """(場コード, 面, 距離) → 候補ごと（内外など）のレース全体の進行方位。s=0..D を 8m 刻みで返す（スタートからゴール）。"""
    out: dict = {}
    for vc, v in routes["venues"].items():
        for key, co in v["courses"].items():
            c = v["circuits"].get(co["circuit"])
            if not c or not c.get("pts_q"):
                continue
            surf, _, dist = key.split("|")
            pts = np.asarray(c["pts_q"], dtype=float)                 # i 番目 = ゴールの手前 q=8i
            L = float(c["L_run"]); D = int(dist)
            s = np.arange(0.0, D + 1e-6, PTS_STEP)
            q = (float(co["q_start"]) + float(co.get("laps_before", 0)) * 0 - s) % L          # 走るほど q は減る（周回は法線で巻き戻る）
            qi = q / PTS_STEP
            n = len(pts)
            i0 = np.floor(qi).astype(int) % n; i1 = (i0 + 1) % n
            # q が減る向きに走る: 進行方位 = (q の小さい側の点 − q の大きい側の点)
            h = bearing(pts[i0, 0] - pts[i1, 0], pts[i0, 1] - pts[i1, 1])
            out.setdefault((vc, surf, D), []).append(h)
    return out


def wind_components(speed: float, wind_from: float, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = np.radians(wind_from - heading)
    return speed * np.cos(rel), speed * np.sin(rel)


def race_wind_features(speed: float, wind_from: float, h: np.ndarray, distance: int) -> dict:
    """h: s=0..D（8m 刻み）の進行方位。区間: 前半3F(0〜600m)、中盤、後半3F(D−600〜D)、最後の直線(D−400〜D)。"""
    head, cross = wind_components(speed, wind_from, h)
    s = np.arange(len(h)) * PTS_STEP
    seg = {"first3f": s <= 600, "last3f": s >= distance - 600, "home": s >= distance - 400, "mid": (s > 600) & (s < distance - 600)}
    f = {f"wind_head_{k}": float(head[m].mean()) if m.any() else np.nan for k, m in seg.items()}
    f.update({"wind_head_all": float(head.mean()), "wind_cross_all": float(cross.mean()), "wind_cross_abs": float(np.abs(cross).mean()),
              "wind_cross_home": float(cross[seg["home"]].mean()) if seg["home"].any() else np.nan})
    return f


def cmd_headings(a) -> None:
    H = course_headings(json.loads(ROUTES.read_text(encoding="utf-8")))
    for vc, surf, D in [("05", "芝", 1600), ("05", "芝", 2400), ("06", "芝", 1600), ("09", "芝", 1600)]:
        for h in H.get((vc, surf, D), []):
            s = np.arange(len(h)) * PTS_STEP
            print(vc, surf, D, "スタート", round(float(h[0])), "前半3F", round(float(h[s <= 600].mean())), "最後の直線", round(float(h[s >= D - 400].mean())))


def cmd_build(a) -> None:
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    H = course_headings(routes)
    w = pd.read_parquet(WIND)
    w["observed_at"] = pd.to_datetime(w["observed_at"]).dt.tz_localize(None) if getattr(w["observed_at"].dt, "tz", None) is not None else pd.to_datetime(w["observed_at"])
    w["venue_code"] = w["venue_code"].astype(str).str.zfill(2)
    cols = ["race_id", "date", "venue_code", "surface", "distance", "start_time", "horse_id", "finish_position", "field_size", "passing_order"]
    r = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in range(2019, 2026)], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"])].copy()
    r["venue_code"] = r["venue_code"].astype(str).str.zfill(2)
    races = r.drop_duplicates("race_id")[["race_id", "date", "venue_code", "surface", "distance", "start_time"]].copy()
    ts = pd.to_datetime(races["date"].astype(str).str[:10] + " " + races["start_time"].astype(str), errors="coerce")
    races["obs_time"] = ts.dt.round("h")
    w1 = w.rename(columns={"observed_at": "obs_time"})[["venue_code", "obs_time", "wind_dir_deg", "wind_speed_ms"]].drop_duplicates(["venue_code", "obs_time"])
    m = races.merge(w1, on=["venue_code", "obs_time"], how="left")
    rows, miss_geo, miss_wind = [], 0, int(m["wind_speed_ms"].isna().sum())
    for x in m.dropna(subset=["wind_speed_ms"]).itertuples():
        hs = H.get((x.venue_code, x.surface, int(x.distance)))
        if not hs or len(hs) != 1 or np.isnan(x.wind_dir_deg):         # 内外が複数ある距離・静穏（風向なし）は除く
            miss_geo += 1
            continue
        d = {"race_id": x.race_id, "wind_speed": float(x.wind_speed_ms), "wind_from": float(x.wind_dir_deg)}
        d.update(race_wind_features(float(x.wind_speed_ms), float(x.wind_dir_deg), hs[0], int(x.distance)))
        rows.append(d)
    wr = pd.DataFrame(rows)
    wr.to_parquet(OUT / "wind_race.parquet", index=False)
    print(f"レース {len(races)}、風あり {len(wr)}（風なし {miss_wind}、形状・風向なし {miss_geo}）")
    if len(wr) < 300:
        return
    d = r.merge(wr, on="race_id", how="inner")
    d["fs"] = pd.to_numeric(d["field_size"], errors="coerce")
    d["fp"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d["p_first"] = d["passing_order"].astype(str).str.split("-").str[0].map(lambda v: float(v) if v.isdigit() else np.nan)
    d["q_first"] = (d["p_first"] - 1) / (d["fs"] - 1)
    d["front"] = d["q_first"] <= 0.25
    d["top3"] = (d["fp"] <= 3).astype(float)
    if OOF.exists():
        oof = pd.read_parquet(OOF)
        oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
        d = d.merge(oof[KEY + ["p_top3"]], on=KEY, how="left")
        d["r3"] = d["top3"] - d["p_top3"]
    rep = {}
    for feat in ("wind_head_home", "wind_head_first3f", "wind_cross_abs"):
        d["bin"] = pd.qcut(d[feat], 3, labels=["弱(向かい風小〜追い風)" if feat != "wind_cross_abs" else "弱", "中", "強(向かい風大)" if feat != "wind_cross_abs" else "強"], duplicates="drop")
        g = d.groupby(["venue_code", "bin"], observed=True).apply(lambda x: pd.Series({
            "n_races": x["race_id"].nunique(),
            "front_top3": x[x["front"]]["top3"].mean(), "rest_top3": x[~x["front"]]["top3"].mean(),
            "front_resid_top3": x[x["front"]]["r3"].mean() if "r3" in x else np.nan}))
        rep[feat] = g.round(4).reset_index().to_dict("records")
    (OUT / "wind_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("headings").set_defaults(fn=cmd_headings)
    sp.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
