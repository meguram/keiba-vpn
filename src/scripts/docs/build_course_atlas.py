#!/usr/bin/env python3
"""JRA コースの概形を、競馬場 → コース（芝/ダート・内回り/外回り）→ 距離 の順に切り替えて見られる HTML（JavaScript なし。CSS のラジオ + :has）にする。

  python -m src.scripts.docs.build_course_atlas            # docs/html/modeling/chapters/course_geometry_atlas.html を作る
  python -m src.scripts.docs.build_docs_book               # ドキュメント総覧（docs/html/book.html）に統合（特徴量と血統の部）

実形状: 背景は OpenStreetMap の芝・ダート・障害コースの面（芝は内回り・外回りの帯、引込み線を含む）、経路・ゴール・コーナー・スタートは
src.research.race.track_geometry が骨格化して作った周回路（data/meta/course_master/geo/routes.json）から描く。スタート位置は「ゴールから距離 D を遡った点」または
「コーナー出口から直線の延長上に延びる引込み線の先端」で、OSM の面に収まり最初のコーナーまでの距離がマスタと合うものだけを実形状とし、それ以外は『推定』と表示する。
坂の位置・高低差は、確定したコースマスタ（data/meta/course_master/jra_courses.json）と JRA の高低断面図から数値化した周回プロファイル（profiles_A.json）の値。
形状は © OpenStreetMap contributors（ODbL）。誤りを探すための図で、縮尺は地図どおりだが、位置は数十 m の誤差を含む。
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import numpy as np

from src.research.race.track_geometry import Q_STEP, circuit_key_for, route_points

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER = REPO_ROOT / "data/meta/course_master/jra_courses.json"
ROUTES = REPO_ROOT / "data/meta/course_master/geo/routes.json"
PROFILES = REPO_ROOT / "data/meta/course_master/profiles_A.json"
OUT = REPO_ROOT / "docs/html/modeling/chapters/course_geometry_atlas.html"
ROMAJI = {"01": "sapporo", "02": "hakodate", "03": "fukushima", "04": "niigata", "05": "tokyo", "06": "nakayama", "07": "chukyo", "08": "kyoto", "09": "hanshin", "10": "kokura"}
VENUE_NAME = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def fnum(x, digits=1, unit="") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return (f"{x:.{digits}f}" if digits else f"{x:.0f}") + unit


MAP_W, MAP_H, PAD = 640.0, 360.0, 14.0
KIND_VAR = ("turf", "dirt", "dirt_est", "steeple", "other", "stand")


class Frame:
    """競馬場ごとの座標変換: 周回路の長軸が横になるよう回転し、スタンドが下になるよう向きをそろえ、地図の枠に収める（y は SVG の下向き）。"""

    def __init__(self, venue: dict):
        pts = [np.array(c["pts_q"]) for c in venue["circuits"].values() if c.get("pts_q")]
        cloud = np.vstack(pts) if pts else np.zeros((2, 2))
        ctr = cloud.mean(axis=0)
        w, v = np.linalg.eigh(np.cov((cloud - ctr).T))
        ax = v[:, int(np.argmax(w))]
        ang = math.atan2(ax[1], ax[0])
        ang = round(math.degrees(ang) / 5.0) * 5.0
        self.th = -math.radians(ang)
        self.ctr = ctr
        self.flip = False
        c0 = next((c for c in venue["circuits"].values() if c.get("stand_pts") and c.get("goal_xy")), None)
        if c0 is not None:
            # ゴールにいちばん近いスタンドの点がゴールより上（画面の上側）に来るなら 180° 回して、スタンドを下にそろえる
            st = np.array(c0["stand_pts"])
            goal = np.array(c0["goal_xy"])
            near = st[int(np.argmin(np.hypot(*(st - goal).T)))]
            self.flip = self._rot(near[None, :])[0, 1] > self._rot(goal[None, :])[0, 1]
        allp = [self._rot(np.array(poly["outer"])) for layer in venue["layers"] if layer["kind"] != "stand" for poly in layer["polys"]]
        allp = np.vstack(allp) if allp else self._rot(cloud)
        self.min = allp.min(axis=0)
        mx = allp.max(axis=0)
        self.scale = min((MAP_W - 2 * PAD) / max(mx[0] - self.min[0], 1.0), (MAP_H - 2 * PAD) / max(mx[1] - self.min[1], 1.0))
        self.off = np.array([(MAP_W - self.scale * (mx[0] - self.min[0])) / 2, (MAP_H - self.scale * (mx[1] - self.min[1])) / 2])

    def _rot(self, p: np.ndarray) -> np.ndarray:
        c, s_ = math.cos(self.th), math.sin(self.th)
        q = (np.asarray(p, float) - self.ctr) @ np.array([[c, -s_], [s_, c]]).T
        return -q if getattr(self, "flip", False) else q

    def xy(self, p) -> np.ndarray:
        q = self._rot(np.atleast_2d(p))
        X = self.off[0] + (q[:, 0] - self.min[0]) * self.scale
        Y = MAP_H - self.off[1] - (q[:, 1] - self.min[1]) * self.scale
        return np.column_stack([X, Y])

    def pts(self, p) -> str:
        return " ".join("%.1f,%.1f" % (x, y) for x, y in self.xy(p))

    def path(self, ring) -> str:
        q = self.xy(np.array(ring))
        return "M" + " L".join("%.1f %.1f" % (x, y) for x, y in q) + " Z"


def slope_class(direction: str) -> str:
    return "up" if "上" in str(direction) else "down"


def layer_defs(vc: str, venue: dict, fr: Frame) -> str:
    """競馬場の背景（コース面）を <g id> にまとめる。コースごとの SVG は <use> で呼ぶ。"""
    parts = [f'<g id="bg-{vc}">']
    for layer in venue["layers"]:
        d = " ".join(fr.path(poly["outer"]) + "".join(" " + fr.path(h) for h in poly["holes"]) for poly in layer["polys"])
        k = layer["kind"] if layer["kind"] in KIND_VAR else "other"
        parts.append(f'<path d="{d}" style="fill:var(--f-{k});stroke:var(--s-{k});stroke-width:.6;{"stroke-dasharray:4 3;" if k == "dirt_est" else ""}" fill-rule="evenodd"/>')
    parts.append("</g>")
    return "".join(parts)


def corner_label_points(circ: dict) -> list[tuple[str, int]]:
    P = circ["pts_q"]
    out = []
    for c in circ.get("corners", {}).get("corners", []):
        q = (c["q_near"] + c["q_far"]) / 2
        out.append((c["name"], min(int(q // Q_STEP), len(P) - 1)))
    return out


def course_route(circ: dict, course: dict) -> np.ndarray:
    q = float(course.get("q_start") or 0.0)
    if course.get("stub"):
        return route_points(circ, q, stub=course["stub"])
    if course.get("start") in ("straight", "chute_diag"):
        sp = next(s for s in circ["spurs"] if s["tip"] == course["spur"])
        return route_points(circ, q, spur=sp)
    return route_points(circ, q)


def course_confidence(course: dict) -> tuple[str, str]:
    """('実形状'|'推定', 説明)。"""
    fl = course.get("flags") or []
    st = course.get("start")
    fcc, fcm = course.get("first_curve_calc"), course.get("first_curve_master")
    if st == "chute" and not fl:
        return "実形状", "引込み線が OSM の面に収まり、最初のコーナーまでの距離がマスタと合う"
    if st == "pocket":
        return "推定", ("ポケットの位置は独立した記述（JRA-VAN ほか）と OSM の面のふくらみで裏づけがあるが、先端から本線への曲線は補間。" + "；".join(fl))
    if st == "chute_diag" and not fl:
        return "実形状", "斜めに合流する引込み線（OSM の面の骨格）。経路長が距離と合い、最初のコーナーまでの距離がマスタと合う"
    if st == "main" and not fl:
        if fcc is not None and fcm is not None and abs(fcc - fcm) > 60:
            return "推定", f"本線上のスタートだが、最初のコーナーまでの距離が図 {fcc:.0f}m / マスタ {fcm:.0f}m とずれる（コーナー位置の検出誤差か、マスタのスタート位置の定義の違い）"
        return "実形状", "本線上のスタート（ゴールから距離を遡った点）"
    return "推定", "；".join(fl) or "—"


def track_svg(vc: str, fr: Frame, circ: dict, venue_courses: list[tuple[dict, dict]], k_ids: list[str], prof: dict | None, r0: dict, surface_cls: str) -> str:
    """1 つのコース（芝/ダート・内外）の SVG。背景は <use>、距離ごとの経路は <g class="rt-k"> に入れ、CSS で 1 つだけ見せる。"""
    out = ['<svg class="atl-svg" viewBox="0 0 640 640" xmlns="http://www.w3.org/2000/svg" role="img"><defs><marker id="atl-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9aa7c7"/></marker></defs>']
    out.append(f'<svg x="0" y="0" width="640" height="{MAP_H:.0f}" viewBox="0 0 640 {MAP_H:.0f}" class="map"><rect x="0" y="0" width="640" height="{MAP_H:.0f}" class="mapbg"/><use href="#bg-{vc}" class="bg {surface_cls}"/>')
    P = np.array(circ["pts_q"])
    out.append(f'<polyline points="{fr.pts(np.vstack([P, P[:1]]))}" class="circ"/>')
    lap = float(circ["L_run"])
    # ゴール
    g0, g1 = fr.xy(P[0])[0], fr.xy(P[1])[0]
    d = g0 - g1
    d = d / (np.hypot(*d) or 1.0)
    nx_, ny_ = -d[1], d[0]
    out.append(f'<line x1="{g0[0] - nx_ * 11:.1f}" y1="{g0[1] - ny_ * 11:.1f}" x2="{g0[0] + nx_ * 11:.1f}" y2="{g0[1] + ny_ * 11:.1f}" class="goal"/><text x="{g0[0] + nx_ * 20:.1f}" y="{g0[1] + ny_ * 20 + 4:.1f}" class="lab goal-t" text-anchor="middle">ゴール</text>')
    out.append(f'<line x1="{g0[0] + d[0] * 26:.1f}" y1="{g0[1] + d[1] * 26 - 0:.1f}" x2="{g0[0] + d[0] * 46:.1f}" y2="{g0[1] + d[1] * 46:.1f}" class="dir" marker-end="url(#atl-arrow)"/>')
    # コーナー名
    ctr = fr.xy(P).mean(axis=0)
    for name, idx in corner_label_points(circ):
        x, y = fr.xy(P[idx])[0]
        v = np.array([x, y]) - ctr
        v = v / (np.hypot(*v) or 1.0)
        out.append(f'<text x="{x + v[0] * 14:.1f}" y="{y + v[1] * 14 + 4:.1f}" class="lab cor" text-anchor="middle">{name}</text>')
    # 坂
    gs = r0.get("goal_slope_exists")
    def seg(a, b):
        ia, ib = sorted((int(a // Q_STEP), int(b // Q_STEP)))
        return fr.pts(P[max(ia, 0): min(ib, len(P) - 1) + 1])
    if gs and r0.get("goal_slope_start") is not None:
        out.append(f'<polyline points="{seg(float(r0["goal_slope_start"]), float(r0.get("goal_slope_end") or 0.0))}" class="slope up goal"/>')
    for sl in (r0.get("other_slopes") or []):
        try:
            s_, e_ = float(sl["start_before_goal_m"]), float(sl["end_before_goal_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(sl.get("rise_m") or 0) < 0.8 or (gs and abs(e_ - float(r0.get("goal_slope_end") or 0)) < 40 and s_ < 700):
            continue
        out.append(f'<polyline points="{seg(s_, e_)}" class="slope {slope_class(sl.get("direction"))}"/>')
    # 距離ごとの経路
    for (row, course), kid in zip(venue_courses, k_ids):
        D = float(row["distance"])
        route = course_route(circ, course)
        conf, why = course_confidence(course)
        out.append(f'<g class="rt {kid}">')
        out.append(f'<polyline points="{fr.pts(route)}" class="route{" est" if conf == "推定" else ""}"/>')
        if course.get("stub"):
            out.append(f'<polyline points="{fr.pts(np.array(course["stub"]))}" class="stub"/>')
        x0, y0 = fr.xy(route[0])[0]
        v = np.array([x0, y0]) - ctr
        v = v / (np.hypot(*v) or 1.0)
        lx, ly = x0 + v[0] * 18, y0 + v[1] * 18
        anchor = "start" if v[0] > 0.35 else ("end" if v[0] < -0.35 else "middle")
        if anchor == "end" and lx < 110:
            anchor, lx = "start", x0 + 12
        elif anchor == "start" and lx > 440:
            anchor, lx, ly = "end", x0 - 10, y0 - 12
        elif anchor == "middle":
            lx = min(max(lx, 70), 570)
        ly = min(max(ly, 16), MAP_H - 40)
        out.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="5" class="start"/><text x="{lx:.1f}" y="{ly + 4:.1f}" class="lab start-t" text-anchor="{anchor}">スタート {int(D)}m{"（推定）" if conf == "推定" else ""}</text>')
        note = []
        if D > lap:
            note.append(f"約 {D / lap:.2f} 周（図は最後の周）")
        if course.get("start") == "chute":
            note.append(f"引込み線 約 {course['stub_len']:.0f}m（コーナー出口から直線の延長上）")
        if course.get("start") == "chute_diag":
            note.append(f"引込み線 約 {course['stub_len']:.0f}m（斜めに合流）")
        if course.get("start") == "pocket":
            note.append(f"ポケット 約 {course['stub_len']:.0f}m（本線の外側から斜めに合流。曲線は補間）")
        if course.get("start") == "main_approx":
            note.append("引込み線の形は OSM に無く、本線上で近似")
        for i_, t_ in enumerate(note):
            out.append(f'<text x="14" y="{MAP_H - 14 - 14 * (len(note) - 1 - i_):.0f}" class="lab warn">{esc(t_)}</text>')
        out.append("</g>")
    out.append(f'<text x="626" y="{MAP_H - 6:.0f}" class="lab ax" text-anchor="end">形状 © OpenStreetMap contributors</text></svg>')
    # 標高プロファイル
    elev = (prof or {}).get("elevation_from_figure_m")
    if elev:
        pts = sorted(((float(k), float(v)) for k, v in elev.items()), reverse=True)
        lo, hi = min(v for _, v in pts), max(v for _, v in pts)
        x0c, x1c, y0c, y1c = 60.0, 600.0, 590.0, 480.0
        qmax = pts[0][0]
        def px(q): return x0c + (x1c - x0c) * (1 - q / qmax)
        def py(v): return y0c - (y0c - y1c) * ((v - lo) / max(hi - lo, 0.5))
        out.append(f'<rect x="{x0c}" y="{y1c - 14}" width="{x1c - x0c}" height="{y0c - y1c + 28}" class="chart"/>')
        for c in circ.get("corners", {}).get("corners", []):
            xa, xb = px(min(c["q_far"], qmax)), px(min(c["q_near"], qmax))
            out.append(f'<rect x="{min(xa, xb):.1f}" y="{y1c - 14}" width="{abs(xb - xa):.1f}" height="{y0c - y1c + 28}" class="cband"/><text x="{(xa + xb) / 2:.1f}" y="{y1c - 18}" class="lab cor" text-anchor="middle">{c["name"]}</text>')
        for (row, course), kid in zip(venue_courses, k_ids):
            q0 = float(course.get("q_start") or 0.0)
            out.append(f'<rect x="{px(min(q0, qmax)):.1f}" y="{y1c - 14}" width="{px(0) - px(min(q0, qmax)):.1f}" height="{y0c - y1c + 28}" class="runband rt {kid}"/>')
        out.append('<polyline points="' + " ".join(f"{px(q):.1f},{py(v):.1f}" for q, v in pts) + '" class="elev"/>')
        out.append(f'<text x="{x0c}" y="{y0c + 26}" class="lab ax" text-anchor="start">← 遠い（ゴールまで {int(qmax)}m）</text><text x="{x1c}" y="{y0c + 26}" class="lab ax" text-anchor="end">ゴール →</text>')
        out.append(f'<text x="{x0c - 6}" y="{y1c + 4}" class="lab ax" text-anchor="end">{hi:.1f}m</text><text x="{x0c - 6}" y="{y0c + 4}" class="lab ax" text-anchor="end">{lo:.1f}m</text>')
        out.append(f'<text x="{(x0c + x1c) / 2:.0f}" y="{y1c - 32}" class="lab ttl" text-anchor="middle">高低断面図（JRA の図を数値化した相対の高さ。誤差 高さ ±0.15m・位置 ±10〜15m）。青の帯 = 選んだ距離で走る区間</text>')
    out.append("</svg>")
    return "".join(out)


def badge(status: str | None, extra: str = "") -> str:
    s = status or "—"
    cls = {"合意": "ok", "合意(外れ値あり)": "ok2", "未確認": "warn", "不一致": "bad", "欠損": "na"}.get(s, "na")
    return f'<span class="bg {cls}">{esc(s)}{esc(extra)}</span>'


def info_table(r: dict, course: dict | None = None, circ: dict | None = None) -> str:
    agree = lambda k: f"（{'・'.join(r.get(k) or [])}）" if r.get(k) else ""
    rows = [
        ("距離", f'{int(r["distance"])}m　{esc(r.get("direction") or "")}回り　{esc(r.get("variant") or "")}'),
        ("発走位置", esc(r.get("start_position_A") or "—")),
        ("最初の曲線まで", f'{fnum(r.get("first_curve_m"), 0)}m {badge(r.get("first_curve_status") if r.get("first_curve_m") is not None else "欠損")}　<span class="mut">（本線の最初のコーナーまで {fnum(r.get("first_main_corner_m_A"), 0)}m）</span>' + (f'<br><span class="mut">{esc(r.get("first_curve_note"))}</span>' if r.get("first_curve_note") else "")),
        ("通過コーナー", f'形状 {esc(" ".join(r.get("corner_sequence") or []) or "—")}（{fnum(r.get("n_corners_geom_A"), 0)}回）／ 記録上の通過順位の点 {fnum(r.get("n_corner_records") or r.get("n_corner_records_mode") or r.get("n_corners_record_mode"), 0)}'),
        ("周長", f'{fnum(r.get("lap_length_m"))}m {badge(r.get("lap_length_m_status"), agree("lap_length_m_agree"))}'),
        ("最後の直線", f'{fnum(r.get("final_straight_m"))}m {badge(r.get("final_straight_m_status"), agree("final_straight_m_agree"))}'),
        ("コース全体の高低差", f'{fnum(r.get("elevation_range_m"))}m {badge(r.get("elevation_range_m_status"), agree("elevation_range_m_agree"))}'),
        ("ゴール前の坂", (f'あり：残り {fnum(r.get("goal_slope_start"), 0)}〜{fnum(r.get("goal_slope_end"), 0)}m、高低差 {fnum(r.get("goal_slope_rise"))}m ' if r.get("goal_slope_exists") else "なし ") + f'<span class="mut">（3 者の判断 {esc(" ".join(f"{k}:{ {True: "あり", False: "なし", None: "不明"}.get(v, "不明")}" for k, v in (r.get("goal_slope_votes") or {}).items()))}）</span>'),
        ("ほかの坂", esc("　".join(f'{s.get("direction")}：{s.get("zone_from")}→{s.get("zone_to")}（残り{fnum(s.get("start_before_goal_m"), 0)}〜{fnum(s.get("end_before_goal_m"), 0)}m）{fnum(s.get("rise_m"))}m' for s in (r.get("other_slopes") or []) if abs(s.get("rise_m") or 0) >= 0.8) or "—")),
        ("周回の注記", esc(r.get("n_laps_note") or ("最後の周は " + r["last_lap_variant"] + "回り" if r.get("last_lap_variant") else ""))) if (r.get("n_laps_note") or r.get("last_lap_variant")) else ("", ""),
        ("コーナーの形", esc(f'{r.get("corner_radius_class_A") or "—"}　{"スパイラルカーブ" if r.get("spiral_curve") else ""}')),
        ("幅員・芝", esc(f'{r.get("width_m") or "—"}　/ {r.get("turf_type_A") or ""}')),
        ("信頼度（JRA 図からの導出）", esc(r.get("confidence_A") or "—")),
    ]
    if course is not None:
        conf, why = course_confidence(course)
        fcc, fcm = course.get("first_curve_calc"), course.get("first_curve_master")
        extra = []
        if circ is not None:
            extra.append(f'周回路の長さ 図 {circ["L_run"]:.0f}m / マスタ {circ["lap_official"]:.0f}m（{circ["err"] * 100:+.1f}%）')
        if fcc is not None and fcm is not None:
            extra.append(f"最初のコーナーまで 図 {fcc:.0f}m / マスタ {fcm:.0f}m")
        if course.get("start") == "pocket":
            extra.append(f'ポケット 約{course["stub_len"]:.0f}m（本線の外側のポケット。経路長の誤差 {course["route_err"] * 100:+.1f}%）')
        if course.get("override"):
            extra.append("発走位置を独立資料で修正: " + str(course["override"]))
        if course.get("start") == "chute_diag":
            extra.append(f'引込み線 約{course["stub_len"]:.0f}m（本線に斜めに合流。経路長の誤差 {course["route_err"] * 100:+.1f}%）')
        if course.get("start") == "chute":
            extra.append(f'引込み線 約{course["stub_len"]:.0f}m（コーナー出口の直線の延長。OSM の面に {course["stub_inside"] * 100:.0f}% 収まる）')
        rows.append(("図の根拠", f'<span class="bg {"ok" if conf == "実形状" else "warn"}">{conf}</span> {esc(why)}' + ("<br><span class=\"mut\">" + esc("　".join(extra)) + "</span>" if extra else "")))
    if r.get("corrections"):
        rows.append(("第 2 ラウンドの訂正", "<br>".join(f'<span class="bg fix">訂正</span> {esc(c["evidence"])}' for c in r["corrections"])))
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows if k)
    return f'<table class="atl-info">{body}</table>'


def build_html() -> str:
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    venues: dict[str, dict] = {}
    for r in master:
        if r.get("in_use") is False:
            continue
        t = (r["surface"], r["variant"] if r["variant"] in ("内", "外", "直") else "")
        venues.setdefault(r["venue_code"], {}).setdefault(t, []).append(r)
    css = _css()
    nv = len(venues)
    frames = {vc: Frame(routes["venues"][vc]) for vc in venues}
    defs = '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>' + "".join(layer_defs(vc, routes["venues"][vc], frames[vc]) for vc in sorted(venues)) + "</defs></svg>"
    parts = [f'<div class="atl" data-n="{nv}">{defs}']
    parts.append('<div class="atl-bar"><span class="atl-cap">競馬場</span>')
    for i, vc in enumerate(sorted(venues)):
        parts.append(f'<label class="atl-chip"><input type="radio" name="atl_v" class="v{i}"{" checked" if i == 0 else ""}>{VENUE_NAME[vc]}</label>')
    parts.append("</div>")
    css_rules = []
    for i, vc in enumerate(sorted(venues)):
        tracks = venues[vc]
        venue = routes["venues"][vc]
        fr = frames[vc]
        order = sorted(tracks, key=lambda t: ({"芝": 0, "ダート": 1}.get(t[0], 2), {"": 0, "内": 1, "外": 2, "直": 3}.get(t[1], 9)))
        css_rules.append(f".atl:has(input.v{i}:checked) .vp{i}{{display:block}}")
        parts.append(f'<div class="vp vp{i}"><h2 class="atl-h">{VENUE_NAME[vc]}競馬場</h2>')
        if venue.get("issues"):
            parts.append('<div class="atl-note">' + "<br>".join(esc(x) for x in venue["issues"]) + "</div>")
        parts.append('<div class="atl-bar"><span class="atl-cap">コース</span>')
        for j, t in enumerate(order):
            label = f'{t[0]}{("・" + t[1] + "回り") if t[1] in ("内", "外") else ("（直線）" if t[1] == "直" else "")}'
            parts.append(f'<label class="atl-chip"><input type="radio" name="atl_t{i}" class="t{i}_{j}"{" checked" if j == 0 else ""}>{esc(label)}</label>')
        parts.append("</div>")
        for j, t in enumerate(order):
            rows = sorted(tracks[t], key=lambda r: r["distance"])
            css_rules.append(f".vp{i}:has(input.t{i}_{j}:checked) .tp{i}_{j}{{display:block}}")
            prof = profiles.get(f"{ROMAJI[vc]}|{'turf' if t[0] == '芝' else 'dirt'}|{t[1] if t[1] in ('内', '外') else ''}")
            r0 = rows[0]
            parts.append(f'<div class="tp tp{i}_{j}">')
            circ = venue["circuits"].get(circuit_key_for(r0))
            if circ is None or not circ.get("pts_q"):
                parts.append('<div class="atl-note">このコースの形状は OSM から導出できませんでした。情報だけ表示します。</div>')
                parts.append("".join(info_table(r) for r in rows[:1]))
                parts.append("</div>")
                continue
            kids = [f"d{i}_{j}_{k}" for k in range(len(rows))]
            courses = [venue["courses"][f'{rr["surface"]}|{rr["variant"]}|{int(rr["distance"])}'] for rr in rows]
            parts.append('<div class="atl-bar"><span class="atl-cap">距離</span>')
            for k, rr in enumerate(rows):
                css_rules.append(f".tp{i}_{j}:has(input.d{i}_{j}_{k}:checked) .rt.{kids[k]}{{display:inline}}")
                css_rules.append(f".tp{i}_{j}:has(input.d{i}_{j}_{k}:checked) .di.{kids[k]}{{display:block}}")
                conf, _ = course_confidence(courses[k])
                parts.append(f'<label class="atl-chip sm{" est" if conf == "推定" else ""}"><input type="radio" name="atl_d{i}_{j}" class="d{i}_{j}_{k}"{" checked" if k == 0 else ""}>{int(rr["distance"])}m{"＊" if conf == "推定" else ""}</label>')
            parts.append("</div>")
            surf_cls = "s-turf" if t[0] == "芝" else "s-dirt"
            parts.append(f'<div class="atl-grid"><div>{track_svg(vc, fr, circ, list(zip(rows, courses)), kids, prof, r0, surf_cls)}</div><div>')
            for k, rr in enumerate(rows):
                parts.append(f'<div class="di {kids[k]}">{info_table(rr, courses[k], circ)}</div>')
            parts.append("</div></div></div>")
        parts.append("</div>")
    parts.append("</div>")
    body = "\n".join(parts)
    legend = ('<div class="atl-legend"><span class="lg route"></span>選んだ距離の走行区間（点線 = 推定）　<span class="lg up"></span>上り坂　<span class="lg down"></span>下り坂　<span class="lg goal"></span>ゴール前の坂　'
              '<span class="lg start"></span>スタート位置　<span class="lg turf"></span>芝の面　<span class="lg dirt"></span>ダートの面　<span class="lg steeple"></span>障害・その他　<span class="lg stand"></span>スタンド</div>')
    intro = ("<p class=\"muted\">JRA 全 10 競馬場のコースの<b>実形状</b>の図。背景は OpenStreetMap のコース面（芝は内回り・外回りの帯や引込み線を含む）で、選んだコースの周回路・ゴール・コーナー・スタートを重ねています。"
             "ゴールはメインスタンドの正面とマスタの『最後の直線の長さ』で、スタートは「ゴールから距離を遡った点」か「コーナー出口の直線の延長上の引込み線の先端」で決め、"
             "引込み線が OSM の面に収まり、最初のコーナーまでの距離がマスタと合うものだけを<b>実形状</b>、そうでないものを<b>推定</b>（距離ボタンの＊）と表示します。"
             "坂の位置・高低差はコースマスタと JRA の高低断面図の数値化です。位置は数十 m の誤差を含みます。<b>誤りを探すための図</b>です。気づいたら、競馬場・コース・距離・項目を教えてください。</p>"
             "<p class=\"muted\">上の『競馬場』『コース』『距離』を選ぶと、図と右の表が切り替わります（JavaScript は使いません。ブラウザが :has に対応している必要があります）。"
             "検証の記録は <a href=\"course_master_verification.html\">JRA コース形状マスタの検証記録</a> を参照してください。"
             "形状データ: © OpenStreetMap contributors（ODbL）。</p>")
    return (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>JRA コース概形アトラス — keiba-vpn docs</title><style>{css}\n' + "\n".join(css_rules) + f'</style></head><body><main><h1>JRA コース概形アトラス</h1>{intro}{legend}{body}</main></body></html>')


def _css() -> str:
    base = (REPO_ROOT / "docs/html/modeling/chapters/training_pipeline_decisions.html").read_text(encoding="utf-8")
    m = re.search(r"<style>(.*?)</style>", base, re.S)
    atl = """
.atl{margin-top:14px}
.atl-bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}
.atl-cap{font-size:12px;color:#9aa7c7;margin-right:6px}
.atl-chip{position:relative;cursor:pointer;border:1px solid #3a4a7a;border-radius:16px;padding:4px 12px;font-size:13px;background:#101a33;color:#cfd8f5;user-select:none}
.atl-chip.sm{padding:2px 9px;font-size:12px}
.atl-chip.est{border-style:dashed}
.atl-chip input{position:absolute;opacity:0;pointer-events:none}
.atl-chip:has(input:checked){background:#3b6fe0;border-color:#7aa9ff;color:#fff}
.vp,.tp,.di{display:none}
.rt{display:none}
.atl-grid{display:grid;grid-template-columns:minmax(360px,1.05fr) minmax(300px,1fr);gap:14px;align-items:start}
@media (max-width:900px){.atl-grid{grid-template-columns:1fr}}
.atl-svg{width:100%;height:auto;background:#0d152b;border:1px solid #243358;border-radius:8px}
.atl-svg .mapbg{fill:#0d152b}
.atl-svg .bg{--f-turf:#26324d;--s-turf:#3a4a73;--f-dirt:#26324d;--s-dirt:#3a4a73;--f-dirt_est:#26324d;--s-dirt_est:#3a4a73;--f-steeple:#1b2742;--s-steeple:#2d3a5c;--f-other:#1b2742;--s-other:#2d3a5c;--f-stand:#4a5070;--s-stand:#6a7196}
.atl-svg .bg.s-turf{--f-turf:#1d5a3c;--s-turf:#2f8a5c}
.atl-svg .bg.s-dirt{--f-dirt:#6a4f2c;--s-dirt:#9a7a48;--f-dirt_est:#6a4f2c;--s-dirt_est:#9a7a48}
.atl-svg .circ{fill:none;stroke:#cfd8f5;stroke-width:.8;stroke-dasharray:3 3;opacity:.7}
.atl-svg .goal{stroke:#ff5a5a;stroke-width:3}
.atl-svg .route{fill:none;stroke:#8c9bff;stroke-width:4.5;stroke-linecap:round;stroke-linejoin:round;opacity:.9}
.atl-svg .route.est{stroke-dasharray:7 5}
.atl-svg .stub{fill:none;stroke:#ffb54a;stroke-width:4.5;stroke-linecap:round}
.atl-svg .start{fill:#ffd24a;stroke:#000;stroke-width:1.5}
.atl-svg .slope{fill:none;stroke-width:3;stroke-linecap:butt}
.atl-svg .slope.up{stroke:#ff9a3c}.atl-svg .slope.down{stroke:#2fe39a}.atl-svg .slope.goal{stroke:#ff6a00;stroke-width:4}
.atl-svg .dir{stroke:#9aa7c7;stroke-width:2}
.atl-svg .lab{fill:#cfd8f5;font-size:11px;font-family:system-ui,sans-serif;paint-order:stroke;stroke:#0d152b;stroke-width:3px;stroke-linejoin:round}
.atl-svg .cor{fill:#9fb4ff;font-size:12px;font-weight:700}
.atl-svg .start-t{fill:#ffd24a;font-weight:700}
.atl-svg .goal-t{fill:#ff8a8a}
.atl-svg .mid{fill:#9aa7c7}.atl-svg .st{fill:#cfd8f5}.atl-svg .warn{fill:#ffb54a}
.atl-svg .chart{fill:#0b1226;stroke:#243358}
.atl-svg .cband{fill:#243358;opacity:.55}
.atl-svg .runband{fill:#3b6fe0;opacity:.28}
.atl-svg .elev{fill:none;stroke:#7bd88f;stroke-width:2}
.atl-svg .ax{fill:#8a96b8;font-size:10px}.atl-svg .ttl{fill:#8a96b8;font-size:10px}
.atl-info{width:100%;font-size:12.5px}
.atl-info th{width:9.5em;text-align:left;vertical-align:top;color:#9aa7c7;font-weight:600;padding:5px 8px;border-bottom:1px solid #243358}
.atl-info td{padding:5px 8px;border-bottom:1px solid #243358}
.atl-info .mut{color:#8a96b8}
.bg{display:inline-block;border-radius:9px;padding:0 7px;font-size:11px;margin-left:4px}
.bg.ok{background:#1f6f4a;color:#d7ffe9}.bg.ok2{background:#5f7a1f;color:#f1ffd0}.bg.warn{background:#8a6a1a;color:#fff0c4}.bg.bad{background:#8a2a2a;color:#ffd8d8}.bg.na{background:#3a4468;color:#cfd8f5}.bg.fix{background:#6a3aa8;color:#efe0ff}
.atl-legend{font-size:12px;color:#cfd8f5;margin:6px 0}
.lg{display:inline-block;width:22px;height:8px;border-radius:4px;vertical-align:middle;margin:0 3px 0 10px}
.lg.turf{background:#1d5a3c}.lg.dirt{background:#6a4f2c}.lg.steeple{background:#26324d}.lg.stand{background:#3a3f55}
.lg.route{background:#8c9bff}.lg.up{background:#ff9a3c}.lg.down{background:#2fe39a}.lg.goal{background:#ff6a00}.lg.start{background:#ffd24a;width:10px;height:10px;border-radius:50%}
.atl-h{margin:14px 0 4px}
.atl-note{background:#16213f;border:1px solid #243358;border-radius:8px;padding:8px 12px;font-size:13px}
"""
    return (m.group(1) if m else "") + atl


def main() -> int:
    OUT.write_text(build_html(), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
