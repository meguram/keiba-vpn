"""JRA コースの実形状（OpenStreetMap のコース面）から、周回路・引込み線・各コースの経路を作る。

流れ:
  1. data/meta/course_master/geo/osm/<場コード>.json（src.scripts.data.fetch_osm_tracks が保存した緯度経度の面ポリゴン）を、場ごとの平面座標（m）に直す。
  2. 面（芝・ダート）ごとに骨格化して中心線のグラフにする（重い依存: shapely・scikit-image・skan・networkx・scipy。生成時だけ使う）。
  3. グラフの閉路から、コースマスタの周長に合う周回路（内回り・外回り・ダート）を選び、突起（引込み線）を拾う。
  4. メインスタンドとマスタの「最後の直線の長さ」でゴール位置を決め、各コースのスタート位置を「ゴールから距離 D を逆に辿る」か「引込み線の先端」で決める。
  5. 結果を data/meta/course_master/geo/routes.json に保存する。アトラスと検証テストは、この保存結果（と osm/*.json）だけを使う。

座標: x = 東向き m、y = 北向き m（場の中心が原点）。周回路は反時計回り（正の面積）を正準形として保存し、走る向き（右回り = 時計回り）は別に持つ。
走行距離は「内ラチから RAIL_M 外側を走る線」の長さで測る（中心線より内側を走るぶん、コーナーで短くなる）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GEO_DIR = REPO_ROOT / "data/meta/course_master/geo"
OSM_DIR = GEO_DIR / "osm"
ROUTES = GEO_DIR / "routes.json"
MASTER = REPO_ROOT / "data/meta/course_master/jra_courses.json"

RAIL_M = 3.0          # 内ラチから走行線までの距離（周長の測り方）
CURVE_THR = 1 / 260   # 曲率がこの値を超える区間を「コーナー」とみなす（1/m）。最初のコーナーまでの距離でマスタと照合して調整
STEP = 4.0            # 中心線の再サンプル間隔（m）


# ------------------------------------------------------------------ 座標・読み込み

def projector(lat0: float, lon0: float):
    kx = math.cos(math.radians(lat0)) * 111320.0
    ky = 110540.0

    def f(lon: float, lat: float) -> tuple[float, float]:
        return (lon - lon0) * kx, (lat - lat0) * ky

    return f


def load_osm(vc: str, osm_dir: Path = OSM_DIR) -> dict:
    """保存済み OSM を平面座標に直して返す。{'origin': (lat0, lon0), 'tracks': [{'surface','osm','polys':[(outer, [holes])]}], 'stands': [...]}"""
    d = json.loads((osm_dir / f"{vc}.json").read_text(encoding="utf-8"))
    lat0, lon0 = d["center"]
    f = projector(lat0, lon0)

    def conv(ring):
        return np.array([f(x, y) for x, y in ring], dtype=float)

    out = {"origin": (lat0, lon0), "tracks": [], "stands": [], "attribution": d.get("attribution", "")}
    for ft in d["features"]:
        polys = [(conv(p["outer"]), [conv(h) for h in p["holes"]]) for p in ft["polygons"]]
        item = {"surface": ft["surface"], "osm": ft["osm"], "tags": ft.get("tags", {}), "polys": polys, "note": ft.get("note")}
        (out["tracks"] if ft["kind"] == "track" else out["stands"]).append(item)
    return out


def main_stand(osm: dict) -> np.ndarray | None:
    """メインスタンドの中心（平面座標）。名前に『スタンド』を含む建物のうち最大のもの。無ければ最大の建物。"""
    cand = []
    for s in osm["stands"]:
        name = (s["tags"].get("name") or "")
        area = sum(_area(o) for o, _ in s["polys"])
        c = np.mean(np.vstack([o for o, _ in s["polys"]]), axis=0)
        try:
            lv = float(s["tags"].get("building:levels") or 0)
        except ValueError:
            lv = 0.0
        cand.append((("スタンド" in name or "stand" in name.lower()), lv, area, c))
    if not cand:
        return None
    cand.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return cand[0][3]


EXCLUDE_NAMES = ("駅", "体育館", "センター", "卸売", "ホテル", "パドック", "厩舎", "駐車")


def stand_points(osm: dict, track_union=None, max_dist: float = 230.0, min_area: float = 2500.0) -> np.ndarray | None:
    """メインスタンド候補（コース面の近くにある大きな建物）の外周の点。ゴールの位置をスタンドの正面に寄せる手がかり。"""
    from shapely.geometry import Polygon

    pts = []
    for s in osm["stands"]:
        name = s["tags"].get("name") or ""
        if any(w in name for w in EXCLUDE_NAMES):
            continue
        for outer, _ in s["polys"]:
            poly = Polygon(outer).buffer(0)
            if poly.area < min_area:
                continue
            if track_union is not None and poly.distance(track_union) > max_dist:
                continue
            ring = np.array(poly.exterior.coords)[:-1]
            pts.append(resample(ring, 15.0, closed=True))
    return np.vstack(pts) if pts else None


def _area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


# ------------------------------------------------------------------ 折れ線の基本操作

def resample(pts: np.ndarray, step: float = STEP, closed: bool = False) -> np.ndarray:
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / step)), 3)
    t = np.linspace(0, s[-1], n + (0 if closed else 1), endpoint=not closed)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def smooth(pts: np.ndarray, sigma_m: float, closed: bool = False, step: float = STEP) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    p = resample(pts, step, closed)
    sg = sigma_m / step
    mode = "wrap" if closed else "nearest"
    q = np.column_stack([gaussian_filter1d(p[:, 0], sg, mode=mode), gaussian_filter1d(p[:, 1], sg, mode=mode)])
    if not closed:
        q[0], q[-1] = p[0], p[-1]
    return q


def signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def heading_curvature(p: np.ndarray, closed: bool, win_m: float = 60.0, step: float = STEP) -> np.ndarray:
    """各点の曲率（1/m。左に曲がる向きが正）。win_m の幅で平滑化した方位の変化から求める。"""
    from scipy.ndimage import gaussian_filter1d

    d = np.roll(p, -1, axis=0) - p if closed else np.vstack([np.diff(p, axis=0), np.diff(p, axis=0)[-1:]])
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    sg = win_m / step / 2
    if closed:
        total = float(ang[-1] - ang[0] + (ang[1] - ang[0]))          # 一周で増える方位（約 ±2π）
        ang = _wrap_filter(ang, sg)
        prev, nxt = np.roll(ang, 1), np.roll(ang, -1)
        prev[0] -= total
        nxt[-1] += total
        return (nxt - prev) / 2 / step
    ang = gaussian_filter1d(ang, sg, mode="nearest")
    return np.gradient(ang) / step


def _wrap_filter(ang: np.ndarray, sg: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    n = len(ang)
    total = ang[-1] - ang[0] + (ang[1] - ang[0])        # 一周で増える方位（約 ±2π）
    ext = np.concatenate([ang - total, ang, ang + total])
    return gaussian_filter1d(ext, sg, mode="nearest")[n:2 * n]


# ------------------------------------------------------------------ 骨格グラフ（重い依存）

def skeleton_graph(polys: list[tuple[np.ndarray, list[np.ndarray]]], res: float = 2.0, spur_m: float = 45.0) -> dict:
    """面ポリゴン（平面座標）の骨格を、節点と辺（中心線の折れ線 + 半幅）のグラフにする。短い突起は削り、次数 2 の節点は縮約する。
    戻り値: {'nodes': {id: [x, y]}, 'edges': [{'u','v','pts': [[x,y]...], 'hw': [...]}]}"""
    import networkx as nx
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from skan import Skeleton, summarize
    from skimage.draw import polygon as skpoly
    from skimage.morphology import skeletonize

    P = unary_union([Polygon(o, holes=hs).buffer(0) for o, hs in polys])
    minx, miny, maxx, maxy = P.bounds
    W, H = int((maxx - minx) / res) + 4, int((maxy - miny) / res) + 4
    img = np.zeros((H, W), bool)
    for g in ([P] if P.geom_type == "Polygon" else list(P.geoms)):
        xs, ys = g.exterior.xy
        rr, cc = skpoly((np.array(ys) - miny) / res + 1, (np.array(xs) - minx) / res + 1, img.shape)
        img[rr, cc] = True
        for h in g.interiors:
            xs, ys = h.xy
            rr, cc = skpoly((np.array(ys) - miny) / res + 1, (np.array(xs) - minx) / res + 1, img.shape)
            img[rr, cc] = False
    edt = distance_transform_edt(img) * res
    sk = skeletonize(img)
    S = Skeleton(sk, spacing=res)
    df = summarize(S, separator="_")
    G = nx.MultiGraph()
    for i, r in df.iterrows():
        path = S.path_coordinates(i)
        src_rc = np.array([r["image_coord_src_0"], r["image_coord_src_1"]])
        if np.hypot(*(np.asarray(path[0]) - src_rc)) > np.hypot(*(np.asarray(path[-1]) - src_rc)):
            path = path[::-1]                      # skan の経路の向きは保証されないので、src → dst にそろえる
        pts = np.array([(minx + (c - 1) * res, miny + (rw - 1) * res) for rw, c in path])
        hw = np.array([edt[int(rw), int(c)] for rw, c in path])
        G.add_edge(int(r["node_id_src"]), int(r["node_id_dst"]), key=i, pts=pts, hw=hw, src=int(r["node_id_src"]), dst=int(r["node_id_dst"]))
    return _clean(G, spur_m)


def _plen(p: np.ndarray) -> float:
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _clean(G, spur_m: float) -> dict:
    import networkx as nx

    H = nx.Graph()
    keep: set[int] = set()                 # 並行する辺を分けるために足した節点（縮約しない）
    mid_id = -1
    for _u, _v, k, d in G.edges(keys=True, data=True):
        u, v = d["src"], d["dst"]                     # pts は src → dst の向き（MultiGraph の列挙順とは無関係）
        ln = _plen(smooth(d["pts"], 12.0)) if len(d["pts"]) > 3 else _plen(d["pts"])
        if u == v and ln < 300:
            continue
        pts, hw = d["pts"], d["hw"]
        if u != v and H.has_edge(u, v):
            # 並行辺（同じ 2 点を結ぶ別の経路）は、中間に節点を挿入して別々の辺として残す
            m = len(pts) // 2
            w = mid_id
            mid_id -= 1
            keep.add(w)
            H.add_edge(u, w, len=_plen(pts[: m + 1]), pts=pts[: m + 1], hw=hw[: m + 1], ends=(u, w))
            H.add_edge(w, v, len=_plen(pts[m:]), pts=pts[m:], hw=hw[m:], ends=(w, v))
            continue
        H.add_edge(u, v, len=ln, pts=pts, hw=hw, ends=(u, v))
    changed = True
    while changed:
        changed = False
        # 次数 2 の節点を縮約（短い枝が連なった鎖を 1 本にまとめてから突起を判定する）
        ch = True
        while ch:
            ch = False
            for n in list(H.nodes):
                if H.degree(n) == 2 and not H.has_edge(n, n) and n not in keep:
                    a, b = list(H[n])
                    if a == b or H.has_edge(a, b):
                        continue
                    pa, ha = _orient(H[a][n], a, n)
                    pb, hb = _orient(H[n][b], n, b)
                    H.add_edge(a, b, len=H[a][n]["len"] + H[n][b]["len"], pts=np.vstack([pa, pb[1:]]), hw=np.concatenate([ha, hb[1:]]), ends=(a, b))
                    H.remove_node(n)
                    ch = True
                    break
        for n in list(H.nodes):
            if n in H and H.degree(n) == 1 and not H.has_edge(n, n):
                u = next(iter(H[n]))
                if H[n][u]["len"] < spur_m:
                    H.remove_node(n)
                    changed = True
    nodes = {}
    edges = []
    for u, v, d in H.edges(data=True):
        pts, hw = _orient(d, u, v)
        pts = smooth(pts, 12.0)
        hw = np.interp(np.linspace(0, 1, len(pts)), np.linspace(0, 1, len(hw)), hw)
        edges.append({"u": int(u), "v": int(v), "pts": np.round(pts, 1).tolist(), "hw": np.round(hw, 1).tolist()})
        nodes[int(u)] = np.round(pts[0], 1).tolist()
        nodes[int(v)] = np.round(pts[-1], 1).tolist()
    return {"nodes": nodes, "edges": edges}


def _orient(d: dict, a, b):
    """辺データ d（ends = (u, v)）を a → b の向きにそろえて (pts, hw) を返す。"""
    pts, hw = d["pts"], d["hw"]
    if d["ends"] != (a, b) and d["ends"] == (b, a):
        pts, hw = pts[::-1], hw[::-1]
    return np.asarray(pts, float), np.asarray(hw, float)


# ------------------------------------------------------------------ 周回路

class Circuit:
    """反時計回りに閉じた周回路の走行線（内ラチから RAIL_M の線）。距離は走行線の長さで測る。"""

    def __init__(self, run_pts: np.ndarray, center_len: float | None = None, method: str = "approx"):
        r = smooth(run_pts, 16.0, closed=True)
        if signed_area(r) < 0:
            r = r[::-1].copy()
        self.xy_run = r
        self.ds = np.hypot(*(np.roll(r, -1, axis=0) - r).T)
        self.L_run = float(self.ds.sum())
        self.L_center = center_len
        self.kappa = heading_curvature(r, True)
        self.method = method

    def oriented(self, clockwise: bool) -> dict:
        """走る向きにそろえた配列（時計回りなら反転）。cum[i] = 先頭からの走行距離。"""
        if clockwise:
            xy = self.xy_run[::-1]
            ds = np.roll(self.ds[::-1], 1)
            kap = self.kappa[::-1]        # 内側へ曲がる向きを正にそろえる
        else:
            xy, ds, kap = self.xy_run, self.ds, self.kappa
        cum = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
        return {"xy": xy, "ds": ds, "kappa": kap, "cum": cum, "L": self.L_run}


def _inner_normals(c: np.ndarray) -> np.ndarray:
    tang = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    return np.column_stack([-tang[:, 1], tang[:, 0]])           # 反時計回りの内側 = 左


def approx_run_line(pts: np.ndarray, hw: np.ndarray, rail: float = RAIL_M) -> np.ndarray:
    """骨格の半幅から概算した走行線（候補の比較用）。"""
    c = smooth(pts, 24.0, closed=True)
    h = np.interp(np.linspace(0, 1, len(c), endpoint=False), np.linspace(0, 1, len(hw)), hw)
    if signed_area(c) < 0:
        c, h = c[::-1].copy(), h[::-1].copy()
    d = np.minimum(np.clip(h - rail, 0.0, None), 18.0)
    return c + _inner_normals(c) * d[:, None]


def hole_run_line(center: np.ndarray, P, rail: float = RAIL_M) -> np.ndarray | None:
    """中心線の内側にある面の穴（= 内ラチの輪）を 1 本見つけられたら、それを rail だけ外へ広げた線を返す。穴の輪が周回路全体に沿っていることを確かめる。"""
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon

    c = smooth(center, 24.0, closed=True)
    if signed_area(c) < 0:
        c = c[::-1].copy()
    tree_c = cKDTree(c)
    best = None
    for g in ([P] if P.geom_type == "Polygon" else list(P.geoms)):
        for h in g.interiors:
            hp = Polygon(h)
            if hp.area < 5e3:
                continue
            ring = resample(np.array(h.coords)[:-1], STEP, closed=True)
            d, _ = tree_c.query(ring)
            cover = float((d < 40).mean())
            L = float(np.hypot(*(np.roll(ring, -1, axis=0) - ring).T).sum())
            d2, _ = cKDTree(ring).query(c)
            cover_c = float((d2 < 40).mean())
            if cover > 0.85 and cover_c > 0.72:
                if best is None or L > best[0]:
                    best = (L, hp)
    if best is None:
        return None
    ex = best[1].buffer(rail, join_style=1).exterior
    return resample(np.array(ex.coords)[:-1], STEP, closed=True)


def ray_run_line(center: np.ndarray, hw: np.ndarray, boundary, rail: float = RAIL_M, max_dist: float = 70.0) -> np.ndarray:
    """中心線から内側へ光線を飛ばし、面の境界（内ラチ）に当たった距離を中央値フィルタでならしてから、rail だけ戻した線。"""
    from scipy.ndimage import median_filter
    from shapely.geometry import LineString

    c = smooth(center, 24.0, closed=True)
    h = np.interp(np.linspace(0, 1, len(c), endpoint=False), np.linspace(0, 1, len(hw)), hw)
    if signed_area(c) < 0:
        c, h = c[::-1].copy(), h[::-1].copy()
    n_in = _inner_normals(c)
    r = np.empty(len(c))
    for i, (p, n) in enumerate(zip(c, n_in)):
        x = LineString([tuple(p), tuple(p + n * max_dist)]).intersection(boundary)
        if x.is_empty:
            r[i] = h[i]
        else:
            r[i] = min(float(np.hypot(*(np.array(g.coords[0]) - p))) for g in getattr(x, "geoms", [x]))
    r = median_filter(r, size=31, mode="wrap")
    return c + n_in * np.maximum(r - rail, 0.0)[:, None]


def _turning(p: np.ndarray) -> float:
    d = np.roll(p, -1, axis=0) - p
    a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(a[-1] - a[0] + (a[1] - a[0]))


def graph_nx(g: dict):
    import networkx as nx

    H = nx.Graph()
    for e in g["edges"]:
        H.add_edge(e["u"], e["v"], pts=np.array(e["pts"]), hw=np.array(e["hw"]), len=_plen(np.array(e["pts"])), ends=(e["u"], e["v"]))
    return H


def cycle_polyline(H, cyc: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pts, hw = [], []
    n = len(cyc)
    if n == 1:
        d = H[cyc[0]][cyc[0]]
        return d["pts"], d["hw"]
    for i in range(n):
        a, b = cyc[i], cyc[(i + 1) % n]
        p, h = _orient(H[a][b], a, b)
        pts.append(p if i == 0 else p[1:])
        hw.append(h if i == 0 else h[1:])
    return np.vstack(pts), np.concatenate(hw)


def candidate_circuits(g: dict, lo: float, hi: float, max_nodes: int = 60) -> list[dict]:
    """グラフの単純閉路のうち、中心線の長さが [lo, hi] で、自己交差せず、回転がほぼ一周のもの。"""
    import networkx as nx
    from shapely.geometry import LineString

    H = graph_nx(g)
    out = []
    cycles = [[u] for u, v in H.edges if u == v]
    cycles += [c for c in nx.simple_cycles(H, length_bound=max_nodes)]
    for c in cycles:
        pts, hw = cycle_polyline(H, c)
        if len(pts) < 20:
            continue
        L = _plen(np.vstack([pts, pts[:1]]))
        if not lo <= L <= hi:
            continue
        sm = smooth(pts, 24.0, closed=True)
        if not LineString(np.vstack([sm, sm[:1]])).is_simple:
            continue
        turn = math.degrees(abs(_turning(sm)))
        if abs(turn - 360) > 45:
            continue
        out.append({"nodes": [int(x) for x in c], "pts": pts, "hw": hw, "turn": turn})
    return out


def pick_circuit(g: dict, lap_official: float, P=None, tol: float = 0.12, exclude: list[list[int]] | None = None) -> tuple[Circuit | None, dict]:
    """マスタの周長（走行線）に最も合う周回路を選ぶ。同程度に合う候補は、最も滑らか（総回転が小さい）なものを優先する。
    P（面の shapely ポリゴン）があれば、内ラチの輪または光線で走行線を決め直す。"""
    cands = candidate_circuits(g, lap_official * (1 - tol), lap_official * (1 + tol + 0.1))
    ex = [set(x) for x in (exclude or [])]
    cands = [c for c in cands if set(c["nodes"]) not in ex]
    scored = []
    for c in cands:
        ci = Circuit(approx_run_line(c["pts"], c["hw"]), method="approx")
        err = ci.L_run / lap_official - 1
        d = np.roll(ci.xy_run, -1, axis=0) - ci.xy_run
        a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        scored.append((abs(err), float(np.abs(np.diff(a)).sum()), err, c, ci))
    if not scored:
        return None, {"error": "候補なし"}
    best_err = min(s_[0] for s_ in scored)
    near = sorted([s_ for s_ in scored if s_[0] <= best_err + 0.03], key=lambda s_: (s_[1], len(s_[3]["nodes"])))
    _, _, err, c, ci = near[0]
    method = "approx"
    if P is not None:
        center = c["pts"]
        run = hole_run_line(center, P)
        method = "hole"
        if run is None:
            run = ray_run_line(center, np.interp(np.linspace(0, 1, len(center)), np.linspace(0, 1, len(c["hw"])), c["hw"]), P.boundary)
            method = "ray"
        ci = Circuit(run, center_len=_plen(np.vstack([center, center[:1]])), method=method)
        err = ci.L_run / lap_official - 1
    return ci, {"nodes": c["nodes"], "err": err, "method": method, "n_candidates": len(scored), "n_near": len(near), "L_run": ci.L_run}


# ------------------------------------------------------------------ 面の分類・ゴール・コーナー

def surface_union(osm: dict, surface: str):
    """surface = 'turf'（平地の芝。ダートの内側にある芝 = 障害コースは除く）または 'dirt'。shapely のポリゴン。"""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def feat_poly(t):
        return unary_union([Polygon(a, holes=h).buffer(0) for a, h in t["polys"]])

    dirt = [t for t in osm["tracks"] if t["surface"] == "dirt"]
    dirt_fill = unary_union([Polygon(a).buffer(0) for t in dirt for a, _ in t["polys"]]) if dirt else None
    keep = []
    main_fill = None
    if surface == "dirt" and dirt:
        big = max(dirt, key=lambda t: sum(_area(a) for a, _ in t["polys"]))
        main_fill = unary_union([Polygon(a).buffer(0) for a, _ in big["polys"]])
    for t in osm["tracks"]:
        if t["surface"] != surface:
            continue
        fp = feat_poly(t)
        if main_fill is not None and t is not big and main_fill.contains(fp.representative_point()):
            continue                          # 本線のダートの輪の内側にある別のダート（調教用など）は本線から除く
        if surface == "turf" and dirt_fill is not None and not dirt_fill.is_empty:
            if dirt_fill.contains(fp.representative_point()):
                continue                      # ダートの輪の内側にある芝 = 障害コース
        keep.append(fp)
    if not keep:
        if surface == "dirt" and osm.get("derive_dirt_from_turf"):
            return None
        return None
    return unary_union(keep).buffer(1.5, join_style=2).buffer(-1.5, join_style=2)          # 面を 1 本のスリットで切った閉じた帯は輪につなぐ


def curve_groups(kappa: np.ndarray, ds: np.ndarray, thr: float = CURVE_THR, min_turn_deg: float = 25.0, merge_gap_m: float = 70.0) -> list[tuple[int, int, float]]:
    """曲率の絶対値がしきい値を超える連続区間（i0, i1, 回転角[度]）。周回は巻き戻して扱う。"""
    n = len(kappa)
    on = np.abs(kappa) > thr
    if on.all() or not on.any():
        return []
    start = int(np.flatnonzero(~on)[0])
    idx = (np.arange(n) + start) % n
    segs, i = [], 0
    while i < n:
        if on[idx[i]]:
            j = i
            while j + 1 < n and on[idx[j + 1]]:
                j += 1
            segs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for sgm in segs:
        if merged and float(ds[idx[merged[-1][1] + 1:sgm[0]]].sum()) < merge_gap_m:
            merged[-1][1] = sgm[1]
        else:
            merged.append(sgm)
    out = []
    for a, b in merged:
        sl = idx[a:b + 1]
        turn = float(np.degrees(np.sum(kappa[sl] * ds[sl])))
        if abs(turn) >= min_turn_deg:
            out.append((int(sl[0]), int(sl[-1]), turn))
    return out


def locate_goal(O: dict, stand_xy: np.ndarray | None, final_straight: float, thr: float = CURVE_THR) -> dict:
    """走る向きにそろえた周回路 O（xy, ds, kappa, cum, L）でゴールの位置（添字）を決める。
    ゴールは「コーナー（内側へ曲がる区間）の出口から final_straight だけ進んだ点」。出口の候補（直線の長さが final_straight 以上のもの）のうち、
    その点がメインスタンドに最も近いものを選ぶ。スタンドが無ければ候補の先頭（周回の大きな直線）を使う。"""
    n = len(O["xy"])
    cum, L = O["cum"], O["L"]
    groups = [g for g in curve_groups(O["kappa"], O["ds"], thr=thr) if g[2] > 0]
    groups.sort(key=lambda t: cum[t[0]])
    cands = []
    for k, (i0, i1, turn) in enumerate(groups):
        nxt = groups[(k + 1) % len(groups)]
        gap = float((cum[nxt[0]] - cum[i1]) % L) if len(groups) > 1 else L
        if gap < final_straight * 0.85:
            continue
        target = (cum[i1] + final_straight) % L
        ig = int(np.argmin(np.abs(((cum - target + L / 2) % L) - L / 2)))
        cands.append({"idx": ig, "i_exit": i1, "gap": gap})
    if not cands:
        return {"idx": None}
    if stand_xy is not None:
        sp = np.atleast_2d(stand_xy)
        for c in cands:
            c["d_stand"] = float(np.min(np.hypot(*(sp - O["xy"][c["idx"]]).T)))
        best = min(cands, key=lambda c: c["d_stand"])
    else:
        best = max(cands, key=lambda c: c["gap"])
        best["d_stand"] = float("nan")
    return {"idx": best["idx"], "i_exit": best["i_exit"], "stand_offset_m": best["d_stand"], "straight_back_m": float(O["ds"][best["i_exit"]:best["idx"]].sum()) if best["idx"] >= best["i_exit"] else float("nan"),
            "n_candidates": len(cands)}


def q_arrays(O: dict, i_goal: int) -> dict:
    """ゴールを q = 0 とし、後ろ（走ってきた側）へ q を取った配列。q[i] = ゴールまでの残り走行距離（0〜L）。"""
    q = (O["cum"][i_goal] - O["cum"]) % O["L"]
    return {"q": q}


def points_at_q(O: dict, i_goal: int, qs: np.ndarray) -> np.ndarray:
    q = (O["cum"][i_goal] - O["cum"]) % O["L"]
    order = np.argsort(q)
    qq = q[order]
    x = np.interp(qs, qq, O["xy"][order, 0])
    y = np.interp(qs, qq, O["xy"][order, 1])
    return np.column_stack([x, y])


# ------------------------------------------------------------------ 引込み線・コース

def find_spurs(g: dict, run_xy: np.ndarray, min_len: float = 45.0, max_junction_dist: float = 45.0) -> list[dict]:
    """骨格グラフの行き止まりの枝のうち、接続点が周回路の走行線から max_junction_dist 以内にあるもの（斜めに合流する引込み線の候補）。
    tip = 先端の節点、pts = 先端 → 接続点の中心線。"""
    from scipy.spatial import cKDTree

    tree = cKDTree(run_xy)
    deg: dict[int, int] = {}
    for e in g["edges"]:
        deg[e["u"]] = deg.get(e["u"], 0) + 1
        deg[e["v"]] = deg.get(e["v"], 0) + 1
    out = []
    for e in g["edges"]:
        pts = np.array(e["pts"])
        for tip, junc, p in ((e["u"], e["v"], pts), (e["v"], e["u"], pts[::-1])):
            if deg.get(tip, 0) == 1 and tip != junc and _plen(p) >= min_len:
                d, _ = tree.query(p[-1])
                if d <= max_junction_dist:
                    out.append({"tip": tip, "junction": junc, "pts": p, "len": _plen(p), "hw": float(np.mean(e["hw"])), "junction_dist": float(d)})
    return out


def first_curve_distance(route: np.ndarray, thr: float = CURVE_THR, min_turn_deg: float = 12.0) -> float | None:
    """経路（スタート → ゴール）に沿って、最初のコーナーに入るまでの距離。曲率がしきい値を超えて続く最初の点。"""
    r = resample(route, STEP)
    if len(r) < 12:
        return None
    kap = np.abs(heading_curvature(r, False, win_m=60.0))
    on = kap > thr
    ds = np.hypot(*np.diff(r, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(ds)])
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            turn = float(np.degrees(np.sum(kap[i:j + 1]) * STEP))
            if turn >= min_turn_deg:
                return float(cum[i])
            i = j + 1
        else:
            i += 1
    return None


# ------------------------------------------------------------------ 場ごとの組み立て

Q_STEP = 8.0


def label_corners(O: dict, i_goal: int, thr: float = CURVE_THR) -> dict:
    """走る向きに曲がる区間（コーナー）を、ゴールから後ろへ 4C・3C・2C・1C と名づける。終端のカーブ（回転 110° 以上）は 2 つに割る。
    名づけられない形（カーブが 2 つ + 小さな曲がりだけ、にならないもの）は labelled=False。"""
    groups = [g for g in curve_groups(O["kappa"], O["ds"], thr=thr) if g[2] > 0]
    L = O["L"]
    cum = O["cum"]
    qg = cum[i_goal]

    def qof(i):
        return float((qg - cum[i]) % L)

    items = []
    for i0, i1, turn in groups:
        q_far, q_near = qof(i0), qof(i1)
        if q_far < q_near:
            q_far += L
        items.append({"q_far": q_far, "q_near": q_near, "turn": turn, "i0": i0, "i1": i1})
    items.sort(key=lambda t: t["q_near"])
    big = [t for t in items if t["turn"] >= 110]
    names = ["4C", "3C", "2C", "1C"]
    corners = []
    labelled = len(big) == 2
    if labelled:
        k = 0
        for t in items:
            if t["turn"] >= 110:
                # 回転角の半分になる位置で 2 分割
                idx = np.arange(t["i0"], t["i0"] + int(round((t["q_far"] - t["q_near"]) / STEP)) + 1) % len(O["kappa"])
                cumturn = np.cumsum(O["kappa"][idx] * O["ds"][idx])
                half = idx[int(np.searchsorted(cumturn, cumturn[-1] / 2))]
                qh = qof(int(half))
                corners.append({"name": names[k], "q_near": t["q_near"], "q_far": qh})
                corners.append({"name": names[k + 1], "q_near": qh, "q_far": t["q_far"]})
                k += 2
        corners.sort(key=lambda c: c["q_near"])
    return {"groups": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items() if k in ("q_far", "q_near", "turn")} for t in items],
            "corners": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in c.items()} for c in corners], "labelled": labelled}


def _nearest_idx(xy: np.ndarray, p) -> int:
    return int(np.argmin(np.hypot(*(xy - np.asarray(p)).T)))


def build_circuit_entry(ci: Circuit, direction: str, stand: np.ndarray | None, final_straight: float | None, g: dict, nodes: list[int], lap_official: float, info: dict, thr: float = CURVE_THR) -> dict:
    cw = direction == "右"
    O = ci.oriented(cw)
    goal = locate_goal(O, stand, float(final_straight or 400.0), thr)
    ent = {"direction": direction, "lap_official": lap_official, "L_run": round(ci.L_run, 1), "err": round(ci.L_run / lap_official - 1, 4), "method": info.get("method"), "nodes": nodes,
           "goal": {k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else v) for k, v in goal.items() if k in ("stand_offset_m", "n_candidates")}}
    if goal["idx"] is None:
        ent["pts_q"] = None
        return ent
    qs = np.arange(0.0, ci.L_run, Q_STEP)
    pts = points_at_q(O, goal["idx"], qs)
    ent["pts_q"] = np.round(pts, 1).tolist()
    ent["goal_xy"] = np.round(O["xy"][goal["idx"]], 1).tolist()
    ent["stand_pts"] = None if stand is None else np.round(stand[:: max(1, len(stand) // 60)], 1).tolist()
    ent["corners"] = label_corners(O, goal["idx"], thr)
    # コーナーの出口（直線の始まり）: 引込み線はここから後ろへ直線で延びるとみなす
    exits = []
    n = len(O["xy"])
    for i0, i1, turn in sorted([g_ for g_ in curve_groups(O["kappa"], O["ds"], thr=thr) if g_[2] > 0], key=lambda t: O["cum"][t[1]]):
        a, b = O["xy"][(i1 - 3) % n], O["xy"][(i1 + 3) % n]
        t = (b - a) / max(float(np.hypot(*(b - a))), 1e-9)
        exits.append({"q": round(float((O["cum"][goal["idx"]] - O["cum"][i1]) % O["L"]), 1), "xy": np.round(O["xy"][i1], 1).tolist(), "tangent": np.round(t, 4).tolist(),
                      "q_entry": round(float((O["cum"][goal["idx"]] - O["cum"][i0]) % O["L"]), 1)})
    ent["exits"] = exits
    # 引込み線
    spurs = []
    for sp in find_spurs(g, O["xy"]):
        j = _nearest_idx(O["xy"], sp["pts"][-1])
        qj = float((O["cum"][goal["idx"]] - O["cum"][j]) % O["L"])
        spurs.append({"tip": int(sp["tip"]), "junction": int(sp["junction"]), "len": round(sp["len"], 1), "q_junction": round(qj, 1), "tip_xy": np.round(sp["pts"][0], 1).tolist(),
                      "pts": np.round(resample(sp["pts"], Q_STEP), 1).tolist()})
    ent["spurs"] = spurs
    return ent


def hole_circuit(P, lap_official: float, rail: float = RAIL_M, close_m: float = 18.0, tol: float = 0.06) -> tuple[Circuit | None, dict]:
    """単純な輪（ダートや、内外の別が無い芝）: 面の穴（= 内ラチの輪）を直接使う。穴に細く突き出た部分（引込み線の根元など）は閉じる処理でならす。
    周長がマスタに近いもの（tol 以内）があれば採用する。"""
    from shapely.geometry import Polygon

    best = None
    for g in ([P] if P.geom_type == "Polygon" else list(P.geoms)):
        for h in g.interiors:
            hp = Polygon(h)
            if hp.area < 3e4:
                continue
            for cm in (0.0, close_m):
                q = hp.buffer(cm, join_style=1).buffer(-cm, join_style=1) if cm else hp
                if q.is_empty or q.geom_type != "Polygon":
                    continue
                ex = q.buffer(rail, join_style=1).exterior
                run = resample(np.array(ex.coords)[:-1], STEP, closed=True)
                ci = Circuit(run, method="hole-direct")
                err = ci.L_run / lap_official - 1
                if best is None or abs(err) < abs(best[0]):
                    best = (err, ci, cm)
    if best is None or abs(best[0]) > tol:
        return None, {"error": "穴の輪が周長に合わない" if best else "穴なし"}
    return best[1], {"nodes": [], "err": best[0], "method": "hole-direct", "closing_m": best[2], "L_run": best[1].L_run}


def derive_dirt_from_turf(osm: dict, width_m: float = 25.0):
    """OSM にダートの面が無い場（中京）の推定: 芝の内ラチの内側に幅 width_m の帯があるものとして導出する。"""
    from shapely.geometry import Polygon

    P = surface_union(osm, "turf")
    if P is None:
        return None
    g = max(([P] if P.geom_type == "Polygon" else list(P.geoms)), key=lambda x: x.area)
    if not g.interiors:
        return None
    hole = max((Polygon(h) for h in g.interiors), key=lambda x: x.area)
    return hole.difference(hole.buffer(-width_m))


def _polys_of(P) -> list:
    return [(np.array(g.exterior.coords)[:-1], [np.array(h.coords)[:-1] for h in g.interiors]) for g in ([P] if P.geom_type == "Polygon" else list(P.geoms))]


def _variant_key(surface: str, variant: str) -> str:
    return f"{surface}|{variant}"


def pick_venue(vc: str, rows: list[dict], osm_dir: Path = OSM_DIR) -> dict:
    """重い段階（骨格化と周回路の選択）。結果は assemble_venue に渡す。"""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    osm = load_osm(vc, osm_dir)
    all_tracks = [Polygon(a, holes=h).buffer(0) for t in osm["tracks"] for a, h in t["polys"]]
    stand = stand_points(osm, unary_union(all_tracks)) if all_tracks else None
    pk = {"vc": vc, "osm": osm, "stand": stand, "issues": [], "picks": {}, "P": {}, "Pall": unary_union(all_tracks) if all_tracks else None}
    for surf_jp, surf in (("芝", "turf"), ("ダート", "dirt")):
        srows = [r for r in rows if r["surface"] == surf_jp]
        if not srows:
            continue
        P = surface_union(osm, surf)
        derived = False
        if (P is None or P.is_empty) and surf == "dirt":
            P = derive_dirt_from_turf(osm)
            derived = P is not None
            if derived:
                pk["issues"].append("ダート: OSM に面が無いため、芝の内ラチの内側に幅 25m の帯があるとして推定")
        if P is None or P.is_empty:
            pk["issues"].append(f"{surf_jp}: OSM に面が無い")
            continue
        pk["P"][surf_jp] = P
        g = skeleton_graph(_polys_of(P))
        variants = sorted({r["variant"] for r in srows if r["variant"] != "直"}, key=lambda v: -float(np.median([r["lap_length_m"] for r in srows if r["variant"] == v and r["lap_length_m"]])))
        picked: list[list[int]] = []
        for var in variants:
            vrows = [r for r in srows if r["variant"] == var]
            lap = float(np.median([r["lap_length_m"] for r in vrows if r["lap_length_m"]]))
            ci, info = (None, {})
            if len(variants) == 1:
                ci, info = hole_circuit(P, lap)
            if ci is None:
                ci, info = pick_circuit(g, lap, P, exclude=picked)
            key = _variant_key(surf_jp, var)
            if ci is None:
                pk["issues"].append(f"{key}: 周回路の候補なし")
                continue
            picked.append(info["nodes"])
            pk["picks"][key] = {"ci": ci, "info": info, "g": g, "derived": derived, "lap": lap, "vrows": vrows, "surface": surf_jp}
    return pk


def layers_of(pk: dict, step: float = 6.0) -> list[dict]:
    """アトラスの背景用に、面を種別ごとにまとめた輪郭（平面座標 m。外周と穴）にする。kind = turf | dirt | dirt_est | steeple | other | stand。"""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    osm = pk["osm"]
    layers = []

    def add(kind, geom):
        if geom is None or geom.is_empty:
            return
        polys = []
        for outer, holes in _polys_of(geom):
            polys.append({"outer": np.round(resample(outer, step, closed=True), 1).tolist(), "holes": [np.round(resample(h, step, closed=True), 1).tolist() for h in holes if len(h) > 5]})
        layers.append({"kind": kind, "polys": polys})

    dirt_est = "ダート" in pk["P"] and not any(t["surface"] == "dirt" for t in osm["tracks"])
    add("turf", pk["P"].get("芝"))
    add("dirt_est" if dirt_est else "dirt", pk["P"].get("ダート"))
    flat = pk["P"].get("芝")
    steeple, other = [], []
    for t in osm["tracks"]:
        fp = unary_union([Polygon(a, holes=h).buffer(0) for a, h in t["polys"]])
        if t["surface"] == "turf" and flat is not None and flat.buffer(2).contains(fp.representative_point()):
            continue
        if t["surface"] == "turf":
            steeple.append(fp)
        elif t["surface"] == "dirt":
            dp = pk["P"].get("ダート")
            if dp is not None and not dp.buffer(2).contains(fp.representative_point()):
                other.append(fp)
        else:
            other.append(fp)
    add("steeple", unary_union(steeple) if steeple else None)
    add("other", unary_union(other) if other else None)
    tracks_u = unary_union([Polygon(a, holes=h).buffer(0) for t in osm["tracks"] for a, h in t["polys"]]) if osm["tracks"] else None
    stands = []
    for s_ in osm["stands"]:
        name = s_["tags"].get("name") or ""
        if any(w in name for w in EXCLUDE_NAMES):
            continue
        for outer, _ in s_["polys"]:
            poly = Polygon(outer).buffer(0)
            if poly.area >= 2500 and (tracks_u is None or poly.distance(tracks_u) <= 230.0):
                stands.append(poly)
    add("stand", unary_union(stands) if stands else None)
    return layers


def assemble_venue(pk: dict, rows: list[dict], thr: float = CURVE_THR) -> dict:
    out = {"venue_code": pk["vc"], "origin": pk["osm"]["origin"], "circuits": {}, "courses": {}, "issues": list(pk["issues"]), "layers": layers_of(pk)}
    for key, p in pk["picks"].items():
        vrows = p["vrows"]
        fs = float(np.median([r["final_straight_m"] for r in vrows if r.get("final_straight_m")]))
        ent = build_circuit_entry(p["ci"], vrows[0]["direction"], pk["stand"], fs, p["g"], p["info"]["nodes"], p["lap"], p["info"], thr)
        ent["derived"] = p["derived"]
        out["circuits"][key] = ent
    out["courses"] = build_courses(out, rows, dict(pk["P"], all=pk["Pall"]), thr)
    return out


def build_venue(vc: str, rows: list[dict], osm_dir: Path = OSM_DIR, thr: float = CURVE_THR) -> dict:
    return assemble_venue(pick_venue(vc, rows, osm_dir), rows, thr)


# ------------------------------------------------------------------ コースごとの経路

def circuit_key_for(row: dict) -> str:
    var = row["variant"]
    if var == "直":
        var = "外"          # 直線コース（新潟）は外回りの最後の直線へつながる
    return _variant_key(row["surface"], var if var in ("内", "外") else "")


def route_points(circ: dict, q_start: float, stub: list | None = None, spur: dict | None = None) -> np.ndarray:
    """スタート → ゴールの経路（平面座標）。circ['pts_q'] は q = 0, Q_STEP, … のゴールから遡った点。stub は引込み線（先端 → 接続点の折れ線）。"""
    P = np.array(circ["pts_q"])
    n = min(int(q_start // Q_STEP) + 1, len(P))
    body = P[:n][::-1]
    if stub is not None:
        return np.vstack([np.array(stub), body])
    if spur is not None:
        return np.vstack([np.array(spur["pts"]), body])
    return body


STUB_MIN, STUB_MAX = 15.0, 500.0


def find_stub(circ: dict, D: float, start_geometry: str, fs: float, P, fc_master: float | list | None = None, thr: float = CURVE_THR) -> tuple[dict | None, list[dict]]:
    """引込み線: コーナーの出口（直線の始まり）から、直線の延長上を後ろへ (D − q_出口 − k×周長) だけ延びた先端。
    先端までの線が OSM のコース面（芝・ダート・その他すべての和）の中にあり、最初のコーナーまでの距離がマスタと合えば、実在する引込み線とみなす。"""
    from shapely.geometry import LineString, Point

    L = float(circ["L_run"])
    Pq = np.array(circ["pts_q"])
    after = Pq[::-1][: int(900 // Q_STEP)]
    cands = []
    for ex in circ.get("exits", []):
        t = np.array(ex["tangent"])
        xy = np.array(ex["xy"])
        for k in range(0, int(D // L) + 1):
            s_ = D - ex["q"] - k * L
            if not STUB_MIN <= s_ <= STUB_MAX:
                continue
            tip = xy - t * s_
            line = LineString([tuple(tip), tuple(xy)])
            inside = float(line.intersection(P.buffer(10)).length / max(line.length, 1e-9)) if P is not None else float("nan")
            body = Pq[: int(ex["q"] // Q_STEP) + 1][::-1]
            fc = first_curve_distance(np.vstack([tip, xy, body, after]), thr)
            cands.append({"q_exit": ex["q"], "len": s_, "k": k, "tip": tip, "xy": xy, "inside": inside, "fc": fc})
    targets = [x for x in (fc_master if isinstance(fc_master, list) else [fc_master]) if x]

    def fc_ok(c):
        return not targets or c["fc"] is None or min(abs(c["fc"] - x) for x in targets) <= 80
    ok = [c for c in cands if c["inside"] >= 0.8 and fc_ok(c)]
    if not ok and targets:
        # 斜めに合流する引込み線は直線の延長から外れる。本線の最初のコーナーまでの距離が合い、半分以上が面に収まるなら採る
        ok = [c for c in cands if c["inside"] >= 0.5 and c["fc"] is not None and min(abs(c["fc"] - x) for x in targets) <= 40]
    if not ok:
        # 面にほぼ完全に収まる（形が OSM に実在する）が、最初のコーナーまでの距離だけマスタとずれる場合は、形を採って『推定』とする
        ok = [dict(c, fc_mismatch=True) for c in cands if c["inside"] >= 0.95]
    if not ok:
        return None, cands
    if start_geometry == "home_stretch_pocket":
        ok.sort(key=lambda c: (abs(c["q_exit"] - fs), -c["inside"]))
    else:
        ok.sort(key=lambda c: (abs(c["q_exit"] - fs) < 150, -c["inside"], c["len"]))
    return ok[0], cands


def find_spur_route(circ: dict, D: float, fcm: float | list | None, thr: float = CURVE_THR) -> dict | None:
    """斜めに合流する引込み線: 骨格の行き止まりの枝（先端 → 接続点）+ 接続点から先の周回路 の長さが距離 D に合い、
    最初のコーナーまでの距離がマスタと合うもの。"""
    L = float(circ["L_run"])
    Pq = np.array(circ["pts_q"])
    after = Pq[::-1][: int(900 // Q_STEP)]
    best = None
    targets = [x for x in (fcm if isinstance(fcm, list) else [fcm]) if x]
    for sp in circ.get("spurs", []):
        for k in range(0, int(D // L) + 1):
            tot = sp["len"] + sp["q_junction"] + k * L
            err = tot / D - 1
            if abs(err) > 0.06:
                continue
            body = Pq[: int(sp["q_junction"] // Q_STEP) + 1][::-1]
            fc = first_curve_distance(np.vstack([np.array(sp["pts"]), body, after]), thr)
            fc_err = min(abs(fc - x) for x in targets) if (fc is not None and targets) else 0.0
            mismatch = bool(targets and fc is not None and fc_err > 80)
            if mismatch and abs(err) > 0.02:
                continue
            cand = {"sp": sp, "k": k, "err": err, "fc": fc, "fc_err": fc_err, "fc_mismatch": mismatch}
            if best is None or (mismatch, fc_err, abs(err)) < (best["fc_mismatch"], best["fc_err"], abs(best["err"])):
                best = cand
    return best


def build_courses(venue: dict, rows: list[dict], Ps: dict | None = None, thr: float = CURVE_THR) -> dict:
    out = {}
    for r in rows:
        key = f'{r["surface"]}|{r["variant"]}|{int(r["distance"])}'
        ck = circuit_key_for(r)
        circ = venue["circuits"].get(ck)
        entry = {"circuit": ck, "flags": []}
        if circ is None or circ.get("pts_q") is None:
            entry["flags"].append("周回路を得られなかった")
            out[key] = entry
            continue
        D, L = float(r["distance"]), float(circ["L_run"])
        sg = r.get("start_geometry") or "main_track"
        fs = float(r.get("final_straight_m") or 400)
        Pall = (Ps or {}).get("all")
        route = None
        if r["variant"] == "直":
            best = None
            for sp in circ["spurs"]:
                err = (sp["len"] + sp["q_junction"]) / D - 1
                if best is None or abs(err) < abs(best[0]):
                    best = (err, sp)
            if best is not None and abs(best[0]) <= 0.08:
                entry.update({"start": "straight", "spur": best[1]["tip"], "route_err": round(best[0], 4), "q_start": round(best[1]["q_junction"], 1)})
                route = route_points(circ, best[1]["q_junction"], spur=best[1])
                entry["flags"].append("直線コースは骨格の枝から推定")
        elif sg != "main_track":
            fct = [r.get("first_curve_m"), r.get("first_main_corner_m_A")]
            st, allc = find_stub(circ, D, sg, fs, Pall, fct, thr)
            if st is not None:
                entry.update({"start": "chute", "q_start": round(st["q_exit"], 1), "stub_len": round(st["len"], 1), "laps_before": st["k"], "stub_inside": round(st["inside"], 2)})
                if st.get("fc_mismatch"):
                    entry["flags"].append(f"引込み線の形は OSM の面に収まるが、最初のコーナーまでの距離が図 {st['fc'] or 0:.0f}m / マスタ {r.get('first_curve_m')}m とずれる")
                stub = np.vstack([st["tip"], st["xy"]])
                entry["stub"] = np.round(stub, 1).tolist()
                route = route_points(circ, st["q_exit"], stub=stub)
            else:
                sr = find_spur_route(circ, D, fct, thr)
                if sr is not None:
                    sp = sr["sp"]
                    entry.update({"start": "chute_diag", "spur": sp["tip"], "q_start": round(sp["q_junction"], 1), "route_err": round(sr["err"], 4), "laps_before": sr["k"], "stub_len": round(sp["len"], 1)})
                    route = route_points(circ, sp["q_junction"], spur=sp)
                    if sr.get("fc_mismatch"):
                        entry["flags"].append(f"斜めに合流する引込み線の経路長は距離と合うが、最初のコーナーまでの距離が図 {sr['fc'] or 0:.0f}m / マスタ {r.get('first_curve_m')}m とずれる")
                else:
                    entry["flags"].append("引込み線の先端が OSM の面に収まらず、本線上の点で代用" + (f"（候補 {len(allc)} 件）" if allc else "（候補なし）"))
        if route is None:
            k = int(D // L)
            qm = D - k * L
            if qm < 20:
                qm = L - 1
            fcm = r.get("first_curve_m")
            if sg != "main_track" and fcm and circ.get("exits"):
                # 引込み線の形が OSM に無い: 最初のコーナーまでの距離がマスタと合う本線上の点で近似する
                q_c = (D - float(fcm)) % L
                ent = min(circ["exits"], key=lambda e: abs(((e["q_entry"] - q_c + L / 2) % L) - L / 2))
                if abs(((ent["q_entry"] - q_c + L / 2) % L) - L / 2) < 90:
                    qv = ent["q_entry"] + float(fcm)
                    if qv < L:
                        entry.update({"start": "main_approx", "q_start": round(qv, 1), "laps_before": k, "approx_chute_len": round(D - k * L - qv, 1)})
                        route = route_points(circ, qv)
                        entry["flags"].append(f"引込み線の形状が OSM に無く、最初のコーナーまでの距離が合う本線上の点で近似（実際はこの手前に約 {D - k * L - qv:.0f}m の引込み線）")
            if route is None:
                entry.update({"start": "main", "q_start": round(qm, 1), "laps_before": k})
                route = route_points(circ, qm)
        entry["start_xy"] = np.round(route[0], 1).tolist()
        Pq = np.array(circ["pts_q"])
        after_goal = Pq[::-1][: int(900 // Q_STEP)]                  # ゴールを過ぎたあとも走り続ける道（最初のコーナーまでの距離の照合用）
        fc = first_curve_distance(np.vstack([route, after_goal]), thr)
        entry["first_curve_calc"] = None if fc is None else round(fc, 0)
        entry["first_curve_master"] = r.get("first_curve_m")
        out[key] = entry
    return out


def build_all(master_path: Path = MASTER, osm_dir: Path = OSM_DIR, venues: list[str] | None = None, thr: float = CURVE_THR) -> dict:
    master = json.loads(Path(master_path).read_text(encoding="utf-8"))
    result = {"attribution": "コース形状: © OpenStreetMap contributors (ODbL) から骨格化して導出", "rail_m": RAIL_M, "curve_thr": thr, "venues": {}}
    for vc in sorted({r["venue_code"] for r in master}):
        if venues and vc not in venues:
            continue
        rows = [r for r in master if r["venue_code"] == vc and r.get("in_use") is not False]
        result["venues"][vc] = build_venue(vc, rows, osm_dir, thr)
    return result


NOTES = ROUTES.parent / "verification_notes.json"

STATIC_NOTES = [
    "形状は OpenStreetMap の面ポリゴン。走行線は『内ラチから 3m』の線（面の穴 = 内ラチの輪を直接使う。内回り・外回りが重なる場所は骨格化した中心線から内側へ光線を飛ばして内ラチに当てる）で、周長はマスタと 3% 以内で合う（多くは 1% 以内）。",
    "ゴール = コーナー出口 + マスタの『最後の直線の長さ』の点のうち、メインスタンドに最も近いもの。スタンド（建物）は OSM の大きな建物のうち、コース面から 230m 以内のもの。",
    "曲率のしきい値 1/260（m⁻¹）は、本線上のスタートのコースで『最初のコーナーまでの距離』が図とマスタで最も合う値に調整した（中央値の誤差 約 1m、平均絶対誤差 約 20m）。OSM の走行線のコーナー入口は、公式のコーナー開始点より数十 m 早く検出される傾向がある（東京の芝1800・2000 など、長い直線が続く場で 90〜120m ずれる）。",
    "引込み線 = コーナー出口の直線を後ろへ延長した先端（長さ = 距離 − 出口から ゴールまでの距離）。先端までの線が OSM の面に 80% 以上収まり、最初のコーナーまでの距離（マスタの first_curve_m または first_main_corner_m_A）が 80m 以内で合えば『実形状』。",
    "斜めに合流する引込み線（札幌・中京・福島など）は、骨格の行き止まりの枝（先端 → 接続点）+ 接続点から先の周回路の長さが距離に合うものを採る。合わない・距離が照合できないものは『推定』にしている。",
    "中京: OSM にダートの面が無いため、芝の内ラチの内側に幅 25m の帯があるとして推定（周長はマスタと +0.8%）。",
    "新潟の直線 1000m は、外回りの最後の直線へつながる枝の骨格から推定（『推定』表示）。",
    "障害コースは OSM の別の面（ダートの輪の内側にある芝）として背景に薄く描くだけで、経路には使わない。",
    "複数周のコース（約 1.5 周など）は最後の周だけを描く。阪神の 3200m 内回り（1 周目は外回り）の初回の周は描いていない。",
]


def verification_summary(result: dict, master: list[dict]) -> dict:
    """生成結果の検証用の要約（周長の誤差・最初のコーナーまでの距離の誤差・推定にとどまるコース）。"""
    laps, fc_err, flagged, kinds = [], [], [], {}
    for vc, v in result["venues"].items():
        for key, c in v["circuits"].items():
            laps.append({"venue": vc, "circuit": key, "L_run": c["L_run"], "lap_official": c["lap_official"], "err": c["err"], "method": c["method"], "derived": c.get("derived", False),
                         "goal_to_stand_m": c["goal"].get("stand_offset_m")})
        for key, c in v["courses"].items():
            kinds[c.get("start")] = kinds.get(c.get("start"), 0) + 1
            if c["flags"]:
                flagged.append({"venue": vc, "course": key, "start": c.get("start"), "first_curve_calc": c.get("first_curve_calc"), "first_curve_master": c.get("first_curve_master"), "reasons": c["flags"]})
            elif c.get("first_curve_calc") is not None and c.get("first_curve_master") is not None:
                fc_err.append(c["first_curve_calc"] - c["first_curve_master"])
    e = np.array(fc_err) if fc_err else np.zeros(1)
    return {"circuits": laps, "n_circuits": len(laps), "max_abs_lap_err": max(abs(x["err"]) for x in laps), "courses_total": sum(kinds.values()), "start_kinds": kinds,
            "first_curve_check": {"n": len(fc_err), "median_err_m": float(np.median(e)), "mean_abs_err_m": float(np.abs(e).mean()), "max_abs_err_m": float(np.abs(e).max())},
            "estimated_courses": flagged, "curve_thr": result["curve_thr"]}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--venues", nargs="*")
    ap.add_argument("--out", default=str(ROUTES))
    a = ap.parse_args()
    res = build_all(venues=a.venues)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("wrote", a.out)
    if not a.venues:
        master = [r for r in json.loads(MASTER.read_text(encoding="utf-8")) if r.get("in_use") is not False]
        NOTES.write_text(json.dumps({"notes": STATIC_NOTES, "summary": verification_summary(res, master)}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", NOTES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
