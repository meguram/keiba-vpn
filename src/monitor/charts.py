"""監視ポータル用の小さな SVG グラフ（サーバー側で組み立てる。JS 不要）。ダーク面前提。

色は dataviz の検証済みパレットのダーク列（系列1=青 #3987e5 / 系列2=橙 #d95926）。状態色（緑・黄・赤）は系列に使わない。
ホバーは各マークの ``<title>``（ブラウザ標準のツールチップ）。同じ数値は画面側の表（``<details>``）でも出す。
"""

from __future__ import annotations

from html import escape
from typing import Sequence

from markupsafe import Markup

C1, C2 = "#3987e5", "#d95926"
GRID, MUTED = "#30363d", "#8b949e"


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    for p in range(0, 10):
        for step in (1, 2, 5):
            if step * 10 ** p >= v:
                return float(step * 10 ** p)
    return float(v)


def combo(labels: Sequence[str], bars: Sequence[float | None], line: Sequence[float | None] | None = None, *,
          bar_name: str = "", line_name: str = "", w: int = 720, h: int = 190, tick_every: int = 15) -> Markup:
    """棒（系列1）と折れ線（系列2）を 1 本の軸に重ねる。None は欠測（描かない）。"""
    pad_l, pad_r, pad_t, pad_b = 40, 8, 8, 22
    n = max(len(labels), 1)
    vals = [v for v in list(bars) + list(line or []) if v is not None]
    top = _nice_max(max(vals) if vals else 0)
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    slot = iw / n
    bw = max(1.0, min(slot - 2, 14))
    y = lambda v: pad_t + ih - (v / top) * ih  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{escape(bar_name)}の推移" style="max-width:{w}px;display:block">']
    for frac in (0, 0.5, 1):
        gy = y(top * frac)
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" fill="{MUTED}" font-size="10" text-anchor="end">{top * frac:,.0f}</text>')
    for i, v in enumerate(bars):
        if v is None:
            continue
        x = pad_l + i * slot + (slot - bw) / 2
        bh = max(0.0, ih * v / top)
        tip = f"{labels[i]}  {bar_name} {v:,.0f}" + (f" / {line_name} {line[i]:,.0f}" if line and line[i] is not None else "")
        out.append(f'<rect x="{x:.1f}" y="{pad_t + ih - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{C1}"><title>{escape(tip)}</title></rect>')
    if line:
        seg: list[str] = []
        for i, v in enumerate(line):
            if v is None:
                if len(seg) > 1:
                    out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{C2}" stroke-width="2" stroke-linejoin="round"/>')
                seg = []
                continue
            seg.append(f"{pad_l + i * slot + slot / 2:.1f},{y(v):.1f}")
        if len(seg) > 1:
            out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{C2}" stroke-width="2" stroke-linejoin="round"/>')
        elif len(seg) == 1:
            cx, cy = seg[0].split(",")
            out.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{C2}"/>')
    for i, lab in enumerate(labels):
        if i % tick_every == 0 or i == n - 1:
            out.append(f'<text x="{pad_l + i * slot + slot / 2:.1f}" y="{h - 6}" fill="{MUTED}" font-size="10" text-anchor="middle">{escape(lab)}</text>')
    out.append("</svg>")
    return Markup("".join(out))


def hbars(rows: Sequence[tuple[str, float]], *, label_w: int = 260, w: int = 640, row_h: int = 22, color: str = C1) -> Markup:
    """横棒（上位ページなど）。値は棒の右に直接出す。"""
    if not rows:
        return Markup("")
    top = max(v for _, v in rows) or 1
    iw = w - label_w - 60
    h = row_h * len(rows)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;display:block">']
    for i, (name, v) in enumerate(rows):
        yy = i * row_h
        shown = name if len(name) <= 38 else name[:37] + "…"
        out.append(f'<text x="0" y="{yy + 15}" fill="#c9d1d9" font-size="12"><title>{escape(name)}</title>{escape(shown)}</text>')
        out.append(f'<rect x="{label_w}" y="{yy + 4}" width="{max(2.0, iw * v / top):.1f}" height="{row_h - 9}" rx="3" fill="{color}"><title>{escape(name)}  {v:,.0f}</title></rect>')
        out.append(f'<text x="{label_w + max(2.0, iw * v / top) + 6:.1f}" y="{yy + 15}" fill="{MUTED}" font-size="11">{v:,.0f}</text>')
    out.append("</svg>")
    return Markup("".join(out))
