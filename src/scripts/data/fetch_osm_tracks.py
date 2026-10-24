#!/usr/bin/env python3
"""OpenStreetMap から JRA 10 競馬場のコース面（芝・ダート・障害）とメインスタンドの形状を取得して保存する。

  python -m src.scripts.data.fetch_osm_tracks [--venues 05 06] [--out data/meta/course_master/geo/osm]

出力: <out>/<場コード>.json（緯度経度。features[].kind = track | stand、surface = turf | dirt | steeple | other）。
データは © OpenStreetMap contributors（ODbL）。帰属は data/meta/course_master/geo/README.md に記載。
Overpass API は公開サービスなので、場ごとに間隔を空け、失敗時は指数バックオフで再試行する。
重い依存（shapely）はこのスクリプトだけが使う。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data/meta/course_master/geo/osm"
UA = "keiba-vpn-research/1.0 (JRA course geometry; contact via repository)"

# 競馬場のおおよその中心（緯度, 経度）。検索範囲の指定にだけ使う。
CENTERS = {
    "01": (43.0788, 141.3304), "02": (41.7783, 140.7778), "03": (37.7612, 140.4778), "04": (37.9578, 139.1914), "05": (35.6623, 139.4857),
    "06": (35.7249, 139.9614), "07": (35.0679, 136.9907), "08": (34.9065, 135.7280), "09": (34.7800, 135.3636), "10": (33.8390, 130.8722),
}
# 面ごとのタグが無い場合の明示指定（OSM の取得 id）。中京は芝の帯が landuse=grass の閉じた way として描かれている。
EXTRA_IDS = {"07": [("way", 497422534)]}
SURFACE_MAP = {"grass": "turf", "turf": "turf", "dirt": "dirt", "sand": "dirt", "woodchips": "other", "ground": "dirt"}


def overpass(query: str, tries: int = 7) -> dict:
    for t in range(tries):
        r = subprocess.run(["curl", "-s", "-m", "90", "-A", UA, "-H", "Accept: application/json", "--data-urlencode", f"data={query}", "https://overpass-api.de/api/interpreter"],
                           capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            wait = 20 * (t + 1)
            print(f"  Overpass が混雑。{wait} 秒待って再試行 ({t + 1}/{tries})")
            time.sleep(wait)
    raise RuntimeError("Overpass から取得できませんでした")


def _ring(geom: list[dict]) -> list[list[float]]:
    return [[round(g["lon"], 6), round(g["lat"], 6)] for g in geom]


def element_polygons(e: dict) -> list[dict]:
    """way / multipolygon relation → [{'outer': [[lon, lat]...], 'holes': [...]}]（shapely で外周と穴をつなぎ直す）。"""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    if e["type"] == "way":
        pts = _ring(e.get("geometry") or [])
        return [{"outer": pts, "holes": []}] if len(pts) > 3 else []
    outers, inners = [], []
    for m in e.get("members", []):
        if m.get("geometry"):
            (outers if m.get("role") == "outer" else inners).append(LineString([(g["lon"], g["lat"]) for g in m["geometry"]]))
    if not outers:
        return []
    shells = list(polygonize(unary_union(outers)))
    holes = list(polygonize(unary_union(inners))) if inners else []
    out = []
    for s in shells:
        hs = [h for h in holes if s.contains(h.representative_point())]
        out.append({"outer": [[round(x, 6), round(y, 6)] for x, y in s.exterior.coords], "holes": [[[round(x, 6), round(y, 6)] for x, y in h.exterior.coords] for h in hs]})
    return out


def area_m2(poly: dict) -> float:
    pts = poly["outer"]
    lat0 = sum(p[1] for p in pts) / len(pts)
    k = math.cos(math.radians(lat0)) * 111320.0
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += (x1 * k) * (y2 * 110540.0) - (x2 * k) * (y1 * 110540.0)
    return abs(a) / 2


def fetch_venue(vc: str) -> dict:
    la, lo = CENTERS[vc]
    d = 0.02
    bb = f"({la - d},{lo - d},{la + d},{lo + d})"
    q = f'[out:json][timeout:90];(way["leisure"="track"]{bb};relation["leisure"="track"]{bb};);out geom;'
    feats = []
    seen = set()
    j = overpass(q)
    for e in j.get("elements", []):
        tags = e.get("tags", {})
        if tags.get("sport") in ("athletics", "running", "cycling", "bmx"):
            continue
        surf = SURFACE_MAP.get(tags.get("surface"))
        if surf is None and tags.get("sport") != "horse_racing":
            continue
        polys = element_polygons(e)
        polys = [p for p in polys if area_m2(p) > 3000]
        if polys:
            feats.append({"osm": f'{e["type"]}/{e["id"]}', "kind": "track", "surface": surf or "unknown", "tags": {k: tags[k] for k in ("surface", "name", "sport") if k in tags}, "polygons": polys})
            seen.add((e["type"], e["id"]))
    for typ, oid in EXTRA_IDS.get(vc, []):
        if (typ, oid) in seen:
            continue
        for e in overpass(f"[out:json][timeout:60];{typ}(id:{oid});out geom;").get("elements", []):
            polys = element_polygons(e)
            feats.append({"osm": f'{typ}/{oid}', "kind": "track", "surface": "turf", "tags": {k: e.get("tags", {}).get(k) for k in ("landuse", "natural") if e.get("tags", {}).get(k)}, "polygons": polys,
                          "note": "OSM では面の種別タグが無く、閉じた帯として描かれている。芝の帯とみなす"})
    time.sleep(3)
    # メインスタンド候補: 競馬場の中心付近の大きな建物
    d2 = 0.009
    bb2 = f"({la - d2},{lo - d2},{la + d2},{lo + d2})"
    qb = f'[out:json][timeout:90];(way["building"]{bb2};relation["building"]{bb2};);out geom;'
    cand = []
    for e in overpass(qb).get("elements", []):
        polys = element_polygons(e)
        polys = [p for p in polys if area_m2(p) > 2500]
        if polys:
            tags = e.get("tags", {})
            cand.append((sum(area_m2(p) for p in polys), {"osm": f'{e["type"]}/{e["id"]}', "kind": "stand", "surface": "n/a",
                                                            "tags": {k: tags[k] for k in ("name", "building:levels", "leisure", "sport", "operator") if k in tags}, "polygons": polys}))
    cand.sort(key=lambda x: -x[0])
    feats += [c[1] for c in cand[:8]]
    return {"venue_code": vc, "center": CENTERS[vc], "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "attribution": "© OpenStreetMap contributors (ODbL)", "features": feats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venues", nargs="*", default=sorted(CENTERS))
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for vc in a.venues:
        print("取得", vc)
        data = fetch_venue(vc)
        (out / f"{vc}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print("  features", [(f["kind"], f["surface"], f["osm"]) for f in data["features"]])
        time.sleep(8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
