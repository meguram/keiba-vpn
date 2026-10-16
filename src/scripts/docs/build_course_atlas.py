#!/usr/bin/env python3
"""JRA コースの概形を、競馬場 → コース（芝/ダート・内回り/外回り）→ 距離 の順に切り替えて見られる HTML（JavaScript なし。CSS のラジオ + :has）にする。

  python -m src.scripts.docs.build_course_atlas            # docs/html/modeling/chapters/course_geometry_atlas.html を作る
  python -m src.scripts.docs.build_modeling_book           # モデリング総覧（book.html）に統合（第III部 特徴量）

模式図: コースはスタジアム型に単純化し、コーナー区間・直線の比率と、スタート位置・ゴール前の坂・ほかの坂の位置を、確定したコースマスタ
（data/meta/course_master/jra_courses.json）と、JRA の高低断面図から数値化した周回プロファイル（profiles_A.json）の値で描く。誤りを探すための図で、縮尺は厳密ではない。
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER = REPO_ROOT / "data/meta/course_master/jra_courses.json"
PROFILES = REPO_ROOT / "data/meta/course_master/profiles_A.json"
OUT = REPO_ROOT / "docs/html/modeling/chapters/course_geometry_atlas.html"
ROMAJI = {"01": "sapporo", "02": "hakodate", "03": "fukushima", "04": "niigata", "05": "tokyo", "06": "nakayama", "07": "chukyo", "08": "kyoto", "09": "hanshin", "10": "kokura"}
VENUE_NAME = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}

# スタジアム型の描画領域（SVG 座標）
CXL, CXR, YT, YB, R = 150.0, 490.0, 95.0, 235.0, 70.0
W = CXR - CXL


class Layout:
    """周回上の位置 q（ゴールまでの残り距離 m、0〜周長 L）を、スタジアム型の SVG 座標に写す。
    b = [b1…b6]: 最後の直線の始まり、4C/3C の境、3C の終わり、向正面の終わり、2C/1C の境、1C の終わり（いずれもゴールまでの残り距離）。"""

    def __init__(self, lap: float, bounds: list[float], clockwise: bool):
        self.L = float(lap)
        self.b1, self.b2, self.b3, self.b4, self.b5, self.b6 = bounds
        self.cw = clockwise
        self.home = self.b1 + (self.L - self.b6)          # ホームストレッチ全体（ゴール前 + ゴール後）

    def _pt(self, px: float, py: float) -> tuple[float, float]:
        x = (CXL + CXR) / 2 + px
        y = (YT + YB) / 2 - py
        if self.cw:
            x = CXL + CXR - x
        return round(x, 1), round(y, 1)

    def xy(self, q: float) -> tuple[float, float]:
        q = max(0.0, min(self.L, q))
        half = (CXR - CXL) / 2
        if q <= self.b1:                                                   # ゴール前の直線（左 → 右へ）
            x = -half + W * (self.b1 - q) / self.home
            return self._pt(x, -R)
        if q >= self.b6:                                                   # ゴール後の直線
            x = -half + W * (self.b1 + (self.L - q)) / self.home
            return self._pt(x, -R)
        if q >= self.b4:                                                   # 右の弧（1C → 2C）
            s = (self.b6 - q) / (self.b6 - self.b4)
            a = -math.pi / 2 + math.pi * s
            return self._pt(half + R * math.cos(a), R * math.sin(a))
        if q >= self.b3:                                                   # 向正面（右 → 左）
            s = (self.b4 - q) / (self.b4 - self.b3)
            return self._pt(half - W * s, R)
        a = math.pi / 2 + math.pi * (self.b3 - q) / (self.b3 - self.b1)    # 左の弧（3C → 4C）
        return self._pt(-half + R * math.cos(a), R * math.sin(a))

    def goal_x(self) -> float:
        return self.xy(0.0)[0]

    def polyline(self, q_from: float, q_to: float, step: float = 12.0) -> str:
        """q_from（遠い）から q_to（近い）へ向かう折れ線の points 属性。"""
        n = max(2, int(abs(q_from - q_to) / step) + 1)
        return " ".join("%.1f,%.1f" % self.xy(q_from + (q_to - q_from) * i / (n - 1)) for i in range(n))

    def outline_path(self) -> str:
        l, r = CXL, CXR
        if not self.cw:
            return f"M {l} {YB} L {r} {YB} A {R} {R} 0 0 0 {r} {YT} L {l} {YT} A {R} {R} 0 0 0 {l} {YB} Z"
        return f"M {r} {YB} L {l} {YB} A {R} {R} 0 0 1 {l} {YT} L {r} {YT} A {R} {R} 0 0 1 {r} {YB} Z"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def fnum(x, digits=1, unit="") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return (f"{x:.{digits}f}" if digits else f"{x:.0f}") + unit


def bounds_from_profile(p: dict | None, lap: float, straight: float) -> list[float]:
    if p and p.get("boundaries_before_goal_m"):
        b = p["boundaries_before_goal_m"]
        v = [b.get("straight_end(4C_start)"), b.get("4C_end(3C_start)"), b.get("3C_end(back_start)"), b.get("back_end(2C_start)"), b.get("2C_end(1C_start)"), b.get("1C_end(straight_after_goal_start)")]
        if all(x is not None for x in v) and v == sorted(v):
            return [float(x) for x in v]
    # 無ければ、直線 S とコーナー 2 つの弧が等しいと仮定した近似
    arc = max((lap - 2 * (straight + 60)) / 2, 200.0)
    b1 = straight
    return [b1, b1 + arc / 2, b1 + arc, b1 + arc + (lap - 2 * arc - b1 - 60), b1 + arc + (lap - 2 * arc - b1 - 60) + arc / 2, lap - 60.0]


def slope_class(direction: str) -> str:
    return "up" if "上" in str(direction) else "down"


def track_svg(layout: Layout, rows: list[dict], profile: dict | None, k_ids: list[str]) -> str:
    """1 つのコース（芝/ダート・内外）の SVG。距離ごとの経路は <g class="rt-k"> に入れ、CSS で 1 つだけ見せる。"""
    r0 = rows[0]
    lap = layout.L
    gs = r0.get("goal_slope_exists")
    out = ['<svg class="atl-svg" viewBox="0 0 640 500" xmlns="http://www.w3.org/2000/svg" role="img"><defs><marker id="atl-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9aa7c7"/></marker></defs>']
    out.append(f'<path d="{layout.outline_path()}" class="trk"/>')
    inner = f'M {CXL} {YB - 22} L {CXR} {YB - 22} A {R - 22} {R - 22} 0 0 {1 if layout.cw else 0} {CXR} {YT + 22} L {CXL} {YT + 22} A {R - 22} {R - 22} 0 0 {1 if layout.cw else 0} {CXL} {YB - 22} Z'
    out.append(f'<path d="{inner}" class="infield"/>')
    # ゴール線・コーナー名・矢印
    gx, gy = layout.xy(0.0)
    out.append(f'<line x1="{gx}" y1="{YB - 14}" x2="{gx}" y2="{YB + 14}" class="goal"/><text x="{gx}" y="{YB + 28}" class="lab goal-t" text-anchor="middle">ゴール</text>')
    for name, qa, qb in (("1C", layout.b5, layout.b6), ("2C", layout.b4, layout.b5), ("3C", layout.b2, layout.b3), ("4C", layout.b1, layout.b2)):
        x, y = layout.xy((qa + qb) / 2)
        cx = (CXL + CXR) / 2
        dx = 16 if x > cx else -16
        out.append(f'<text x="{x + dx:.1f}" y="{y + (4 if True else 0):.1f}" class="lab cor" text-anchor="middle">{name}</text>')
    ax, ay = layout.xy(layout.b1 * 0.75)
    bx, by = layout.xy(layout.b1 * 0.75 - 40)
    out.append(f'<line x1="{ax}" y1="{ay + 22}" x2="{bx}" y2="{by + 22}" class="dir" marker-end="url(#atl-arrow)"/>')
    out.append(f'<text x="{(CXL + CXR) / 2:.0f}" y="{(YT + YB) / 2 + 4:.0f}" class="lab mid" text-anchor="middle">{esc(r0.get("direction") or "")}回り　周長 {fnum(lap, 0)}m</text>')
    sx, sy = layout.xy(layout.b1 * 0.5)
    out.append(f'<text x="{(CXL + CXR) / 2:.0f}" y="{YB + 52}" class="lab st" text-anchor="middle">最後の直線 {fnum(r0.get("final_straight_m"))}m（{"ゴール前の坂あり" if gs else "坂なし"}）</text>')
    # 距離ごとの経路
    for rr, kid in zip(rows, k_ids):
        D = float(rr["distance"])
        q0 = D % lap if D > lap else D
        laps = D / lap
        out.append(f'<g class="rt {kid}">')
        out.append(f'<polyline points="{layout.polyline(q0, 0.0, 8)}" class="route"/>')
        x0, y0 = layout.xy(q0)
        cxm, cym = (CXL + CXR) / 2, (YT + YB) / 2
        vx, vy = x0 - cxm, y0 - cym
        nrm = math.hypot(vx, vy) or 1.0
        lx, ly = x0 + vx / nrm * 20, y0 + vy / nrm * 20
        anchor = "start" if vx > 40 else ("end" if vx < -40 else "middle")
        out.append(f'<circle cx="{x0}" cy="{y0}" r="6" class="start"/><text x="{lx:.1f}" y="{ly + 4:.1f}" class="lab start-t" text-anchor="{anchor}">スタート {int(D)}m</text>')
        if D > lap:
            out.append(f'<text x="{(CXL + CXR) / 2:.0f}" y="{(YT + YB) / 2 + 22:.0f}" class="lab warn" text-anchor="middle">約 {laps:.2f} 周（図は最後の周）</text>')
        out.append("</g>")
    # 坂（ゴール前と、その他）
    if gs and r0.get("goal_slope_start") is not None:
        s, e = float(r0["goal_slope_start"]), float(r0.get("goal_slope_end") or 0.0)
        out.append(f'<polyline points="{layout.polyline(s, e, 6)}" class="slope up goal"/>')
    for sl in (r0.get("other_slopes") or []):
        try:
            s, e = float(sl["start_before_goal_m"]), float(sl["end_before_goal_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(sl.get("rise_m") or 0) < 0.8 or (gs and abs(e - float(r0.get("goal_slope_end") or 0)) < 40 and s < 700):
            continue
        out.append(f'<polyline points="{layout.polyline(s, e, 10)}" class="slope {slope_class(sl.get("direction"))}"/>')
    # 標高プロファイル
    elev = (profile or {}).get("elevation_from_figure_m")
    if elev:
        pts = sorted(((float(k), float(v)) for k, v in elev.items()), reverse=True)   # q が大きい（遠い）ほうから
        lo, hi = min(v for _, v in pts), max(v for _, v in pts)
        x0c, x1c, y0c, y1c = 60.0, 600.0, 462.0, 358.0
        qmax = pts[0][0]
        def px(q): return x0c + (x1c - x0c) * (1 - q / qmax)
        def py(v): return y0c - (y0c - y1c) * ((v - lo) / max(hi - lo, 0.5))
        out.append(f'<rect x="{x0c}" y="{y1c - 14}" width="{x1c - x0c}" height="{y0c - y1c + 28}" class="chart"/>')
        for name, qa, qb in (("1C", layout.b5, layout.b6), ("2C", layout.b4, layout.b5), ("3C", layout.b2, layout.b3), ("4C", layout.b1, layout.b2)):
            xa, xb = px(min(qa, qmax)), px(min(qb, qmax))
            out.append(f'<rect x="{min(xa, xb):.1f}" y="{y1c - 14}" width="{abs(xb - xa):.1f}" height="{y0c - y1c + 28}" class="cband"/><text x="{(xa + xb) / 2:.1f}" y="{y1c - 18}" class="lab cor" text-anchor="middle">{name}</text>')
        for rr, kid in zip(rows, k_ids):
            D = float(rr["distance"]); q0 = D % lap if D > lap else D
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


def info_table(r: dict) -> str:
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
    if r.get("corrections"):
        rows.append(("第 2 ラウンドの訂正", "<br>".join(f'<span class="bg fix">訂正</span> {esc(c["evidence"])}' for c in r["corrections"])))
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows if k)
    return f'<table class="atl-info">{body}</table>'


def build_html() -> str:
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    venues: dict[str, dict] = {}
    for r in master:
        if r.get("in_use") is False:
            continue
        t = (r["surface"], r["variant"] if r["variant"] in ("内", "外", "直") else "")
        venues.setdefault(r["venue_code"], {}).setdefault(t, []).append(r)
    css = _css()
    nv = len(venues)
    parts = [f'<div class="atl" data-n="{nv}">']
    # 競馬場の切替
    parts.append('<div class="atl-bar"><span class="atl-cap">競馬場</span>')
    for i, vc in enumerate(sorted(venues)):
        parts.append(f'<label class="atl-chip"><input type="radio" name="atl_v" class="v{i}"{" checked" if i == 0 else ""}>{VENUE_NAME[vc]}</label>')
    parts.append("</div>")
    css_rules = []
    for i, vc in enumerate(sorted(venues)):
        tracks = venues[vc]
        order = sorted(tracks, key=lambda t: ({"芝": 0, "ダート": 1}.get(t[0], 2), {"": 0, "内": 1, "外": 2, "直": 3}.get(t[1], 9)))
        css_rules.append(f".atl:has(input.v{i}:checked) .vp{i}{{display:block}}")
        parts.append(f'<div class="vp vp{i}"><h2 class="atl-h">{VENUE_NAME[vc]}競馬場</h2><div class="atl-bar"><span class="atl-cap">コース</span>')
        for j, t in enumerate(order):
            label = f'{t[0]}{("・" + t[1] + "回り") if t[1] in ("内", "外") else ("（直線）" if t[1] == "直" else "")}'
            parts.append(f'<label class="atl-chip"><input type="radio" name="atl_t{i}" class="t{i}_{j}"{" checked" if j == 0 else ""}>{esc(label)}</label>')
        parts.append("</div>")
        for j, t in enumerate(order):
            rows = sorted(tracks[t], key=lambda r: r["distance"])
            css_rules.append(f".vp{i}:has(input.t{i}_{j}:checked) .tp{i}_{j}{{display:block}}")
            prof = profiles.get(f"{ROMAJI[vc]}|{'turf' if t[0] == '芝' else 'dirt'}|{t[1] if t[1] in ('内', '外') else ''}")
            r0 = rows[0]
            lap = float(r0.get("lap_length_m") or (prof or {}).get("lap_length_m") or 1800.0)
            parts.append(f'<div class="tp tp{i}_{j}">')
            if t[1] == "直":
                parts.append(f'<div class="atl-note">直線コース（{int(rows[0]["distance"])}m）。スタジアム型の図は描かず、情報だけ表示します。</div>')
                parts.append("".join(info_table(r) for r in rows[:1]))
                parts.append("</div>")
                continue
            layout = Layout(lap, bounds_from_profile(prof, lap, float(r0.get("final_straight_m") or 400)), clockwise=(r0.get("direction") == "右"))
            kids = [f"d{i}_{j}_{k}" for k in range(len(rows))]
            parts.append('<div class="atl-bar"><span class="atl-cap">距離</span>')
            for k, rr in enumerate(rows):
                css_rules.append(f".tp{i}_{j}:has(input.d{i}_{j}_{k}:checked) .rt.{kids[k]}{{display:inline}}")
                css_rules.append(f".tp{i}_{j}:has(input.d{i}_{j}_{k}:checked) .di.{kids[k]}{{display:block}}")
                parts.append(f'<label class="atl-chip sm"><input type="radio" name="atl_d{i}_{j}" class="d{i}_{j}_{k}"{" checked" if k == 0 else ""}>{int(rr["distance"])}m</label>')
            parts.append("</div>")
            parts.append(f'<div class="atl-grid"><div>{track_svg(layout, rows, prof, kids)}</div><div>')
            for k, rr in enumerate(rows):
                parts.append(f'<div class="di {kids[k]}">{info_table(rr)}</div>')
            parts.append("</div></div></div>")
        parts.append("</div>")
    parts.append("</div>")
    body = "\n".join(parts)
    legend = ('<div class="atl-legend"><span class="lg route"></span>選んだ距離の走行区間　<span class="lg up"></span>上り坂　<span class="lg down"></span>下り坂　<span class="lg goal"></span>ゴール前の坂　'
              '<span class="lg start"></span>スタート位置</div>')
    intro = ("<p class=\"muted\">JRA 全 10 競馬場のコースの模式図。コースはスタジアム型に単純化し、コーナー区間と直線の比率、スタート位置、坂の位置・高低差は確定したコースマスタ（3 つの独立収集 + 一次資料での再確認）と、"
             "JRA の高低断面図から数値化した周回プロファイルの値で描いています。縮尺は厳密ではありません。<b>誤りを探すための図</b>です。気づいたら、競馬場・コース・距離・項目を教えてください。</p>"
             "<p class=\"muted\">上の『競馬場』『コース』『距離』を選ぶと、図と右の表が切り替わります（JavaScript は使いません。ブラウザが :has に対応している必要があります）。"
             "検証の記録は <a href=\"course_master_verification.html\">JRA コース形状マスタの検証記録</a> を参照してください。</p>")
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
.atl-chip input{position:absolute;opacity:0;pointer-events:none}
.atl-chip:has(input:checked){background:#3b6fe0;border-color:#7aa9ff;color:#fff}
.vp,.tp,.di{display:none}
.rt{display:none}
.atl-grid{display:grid;grid-template-columns:minmax(360px,1.05fr) minmax(300px,1fr);gap:14px;align-items:start}
@media (max-width:900px){.atl-grid{grid-template-columns:1fr}}
.atl-svg{width:100%;height:auto;background:#0d152b;border:1px solid #243358;border-radius:8px}
.atl-svg .trk{fill:#1c2f4d;stroke:#566a9a;stroke-width:2}
.atl-svg .infield{fill:#122a24;stroke:#2c5a49;stroke-width:1}
.atl-svg .goal{stroke:#ff5a5a;stroke-width:3}
.atl-svg .route{fill:none;stroke:#8c9bff;stroke-width:13;stroke-linecap:round;stroke-linejoin:round;opacity:.55}
.atl-svg .start{fill:#ffd24a;stroke:#000;stroke-width:1.5}
.atl-svg .slope{fill:none;stroke-width:5;stroke-linecap:butt}
.atl-svg .slope.up{stroke:#ff9a3c}.atl-svg .slope.down{stroke:#2fe39a}.atl-svg .slope.goal{stroke:#ff6a00;stroke-width:6}
.atl-svg .dir{stroke:#9aa7c7;stroke-width:2}
.atl-svg .lab{fill:#cfd8f5;font-size:11px;font-family:system-ui,sans-serif}
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
