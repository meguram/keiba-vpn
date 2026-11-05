"use client";

import { useMemo } from "react";
import { radialLayout, formatTreemapValue, type TreemapMetric, type TreemapNode } from "@/lib/columns";

const HIGH_RGB = "88,166,255";
const LOW_RGB = "240,136,62";
const NEUTRAL = "rgba(139,148,158,0.16)";

function overlay(t: number): string {
  return `rgba(${t >= 0 ? HIGH_RGB : LOW_RGB},${(Math.abs(t) * 0.78).toFixed(3)})`;
}

function fit(text: string, max: number): string {
  const cs = [...text];
  return cs.length <= max ? text : cs.slice(0, max - 1).join("") + "…";
}

type Props = {
  view: TreemapNode;
  mi: number;
  metric: TreemapMetric;
  selId: string | null;
  /** view から選んだ節までの節の id（経路の強調に使う） */
  pathIds: Set<string>;
  narrow: boolean;
  title: string;
  tOf: (v: number | null | undefined) => number;
  onSelect: (id: string) => void;
  onDrill: (id: string) => void;
};

/** 円形の系統樹。中心が表示中の系統、外へ向かって父系が枝分かれする。選んだ節から中心までの経路を太線で示す */
export function ColumnRadial({ view, mi, metric, selId, pathIds, narrow, title, tOf, onSelect, onDrill }: Props) {
  const pad = narrow ? 72 : 130;
  const W = narrow ? 340 : 760;
  const R = W / 2 - pad;
  const limit = narrow ? 30 : 64;
  const { items, sectors, rings } = useMemo(() => radialLayout(view, limit, R), [view, limit, R]);
  const byId = new Map(items.map((it) => [it.node.id, it]));
  const cx = W / 2, cy = W / 2;
  const P = (r: number, a: number) => `${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`;
  const fs = narrow ? 9 : 11;
  const maxChars = narrow ? 7 : 11;
  const halo = { paintOrder: "stroke" as const, stroke: "var(--surface, #0d1117)", strokeWidth: 3, strokeLinejoin: "round" as const };

  return (
    <svg viewBox={`0 0 ${W} ${W}`} width="100%" style={{ display: "block", maxWidth: W, margin: "0 auto", touchAction: "manipulation" }}
      role="img" aria-label={`${title}（${view.name}）`}>
      {/* 直下の系統ごとの扇形（どの系統の下かを背景で示す） */}
      {sectors.map((s, i) => {
        const big = s.a1 - s.a0 > Math.PI ? 1 : 0;
        const d = `M${cx},${cy} L${P(R + 4, s.a0)} A${R + 4},${R + 4} 0 ${big} 1 ${P(R + 4, s.a1)} Z`;
        return <path key={s.node.id} d={d} fill={i % 2 ? "rgba(139,148,158,0.13)" : "rgba(139,148,158,0.02)"} stroke="none" style={{ pointerEvents: "none" }} />;
      })}
      {rings.map((r, i) => <circle key={i} cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2 3" style={{ pointerEvents: "none" }} />)}
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2 3" style={{ pointerEvents: "none" }} />

      {/* 枝: 親の半径で弧を描いて子の角度へ振り、子まで放射状に伸ばす */}
      {items.map((it) => {
        const p = it.parentId ? byId.get(it.parentId) : null;
        if (!p) return null;
        const on = pathIds.has(it.node.id);
        const arc = p.r > 0.5 && Math.abs(it.angle - p.angle) > 1e-4 ? ` A${p.r.toFixed(1)},${p.r.toFixed(1)} 0 0 ${it.angle > p.angle ? 1 : 0} ${P(p.r, it.angle)}` : "";
        const d = `M${P(p.r, p.angle)}${arc} L${P(it.r, it.angle)}`;
        return <path key={it.node.id} d={d} fill="none" stroke={on ? "var(--text)" : "var(--text-dim)"} strokeOpacity={on ? 1 : 0.45}
          strokeWidth={on ? 2.2 : 1} style={{ pointerEvents: "none" }} />;
      })}

      {/* 節 */}
      {items.map((it) => {
        const n = it.node;
        const t = tOf(n.metrics[mi]?.[0]);
        const isView = it.depth === 0;
        const rad = isView ? 9 : it.folded ? 9 : it.outer ? 5.5 : 5;
        const on = selId === n.id || (selId === null && isView);
        const inPath = pathIds.has(n.id);
        const cos = Math.cos(it.angle);
        const x = cx + it.r * cos, y = cy + it.r * Math.sin(it.angle);
        const deg = (it.angle * 180) / Math.PI;
        const clickable = !!n.children?.length;
        const tip = `${n.name}（${n.value.toLocaleString()}頭）: ${metric.label} ${formatTreemapValue(n.metrics[mi]?.[0], metric.format)}${clickable && !isView ? "（タップでこの系統へ）" : ""}`;
        const square = n.kind === "line";
        return (
          <g key={n.id} style={{ cursor: "pointer" }} onMouseEnter={() => onSelect(n.id)}
            onClick={() => (clickable && !isView ? onDrill(n.id) : onSelect(n.id))}>
            <title>{tip}</title>
            {square
              ? <>
                <rect x={x - rad} y={y - rad} width={rad * 2} height={rad * 2} rx={2} fill={NEUTRAL} />
                <rect x={x - rad} y={y - rad} width={rad * 2} height={rad * 2} rx={2} fill={overlay(t)} stroke={on ? "var(--text)" : inPath ? "var(--text)" : "var(--text-dim)"} strokeWidth={on ? 2 : 1} />
              </>
              : <>
                <circle cx={x} cy={y} r={rad} fill={NEUTRAL} />
                <circle cx={x} cy={y} r={rad} fill={overlay(t)} stroke={on ? "var(--text)" : inPath ? "var(--text)" : "var(--text-dim)"} strokeWidth={on ? 2 : 1} />
              </>}
            {it.folded && !isView && (
              <text x={x} y={y + 3} fontSize={8} textAnchor="middle" fill="var(--text)" style={{ pointerEvents: "none" }}>{it.sires > 99 ? "99+" : it.sires}</text>
            )}
            {/* 外周の名前: 放射状に外へ。左半分は 180° 回して読める向きにする */}
            {it.outer && !isView && (
              <text transform={`translate(${(cx + (it.r + rad + 3) * cos).toFixed(1)},${(cy + (it.r + rad + 3) * Math.sin(it.angle)).toFixed(1)}) rotate(${cos < 0 ? deg + 180 : deg})`}
                textAnchor={cos < 0 ? "end" : "start"} dominantBaseline="central" fontSize={fs} fill="var(--text)" fontWeight={inPath ? 700 : 400}
                style={{ pointerEvents: "none" }}>{fit(n.name, maxChars)}</text>
            )}
            {/* 内側の節（系統・父系の祖先）の名前は、経路上か、浅い段のものだけ */}
            {!it.outer && (isView || inPath || !narrow || it.depth <= 1) && (
              <text x={x} y={isView ? y + rad + fs + 2 : y - rad - 3} textAnchor="middle" fontSize={fs} fontWeight={isView || inPath ? 700 : 600}
                fill="var(--text)" style={{ pointerEvents: "none", ...halo }}>{fit(n.name, narrow ? 9 : 14)}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
