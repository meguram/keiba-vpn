"use client";

import { useMemo, useState } from "react";
import { useNarrow } from "@/components/RaceStrip";
import { formatTreemapValue, orderedLayout, type ColumnTreemapBlock, type Rect, type TreemapMetric, type TreemapNode } from "@/lib/columns";

/** 発散型の両端（中央は灰色）。暗い背景の上に半透明で重ねるので、文字は var(--text) のまま読める */
const HIGH_RGB = "88,166,255";
const LOW_RGB = "240,136,62";
const NEUTRAL = "rgba(139,148,158,0.16)";
const HEADER = 16;

type Item = { node: TreemapNode; rect: Rect; depth: number };

function tOf(v: number | null | undefined, m: TreemapMetric): number {
  if (typeof v !== "number" || !Number.isFinite(v) || !(m.scale > 0)) return 0;
  return Math.max(-1, Math.min(1, (v - m.center) / m.scale));
}

function overlay(t: number): string {
  const a = Math.abs(t) * 0.78;
  return `rgba(${t >= 0 ? HIGH_RGB : LOW_RGB},${a.toFixed(3)})`;
}

/** 文字の幅のおおよそ（全角 1em・半角 0.58em）で、w に収まるよう切る */
function fit(text: string, w: number, size: number): string {
  const width = (s: string) => [...s].reduce((a, c) => a + (c.charCodeAt(0) < 0x2000 ? 0.58 : 1) * size, 0);
  if (width(text) <= w) return text;
  const cs = [...text];
  while (cs.length > 1 && width(cs.join("") + "…") > w) cs.pop();
  return cs.length >= 3 ? cs.join("") + "…" : "";
}

function layout(view: TreemapNode, W: number, H: number, maxDepth: number): Item[] {
  const items: Item[] = [];
  const rec = (node: TreemapNode, rect: Rect, depth: number) => {
    items.push({ node, rect, depth });
    const kids = node.children ?? [];
    if (!kids.length) return;
    let inner = rect;
    if (depth > 0) {
      const framed = depth <= maxDepth;
      const pad = framed ? 2 : 1;
      const head = framed && rect.h >= HEADER * 2.2 && rect.w >= 44 ? HEADER : 0;
      inner = { x: rect.x + pad, y: rect.y + head + pad, w: Math.max(0, rect.w - pad * 2), h: Math.max(0, rect.h - head - pad * 2) };
    }
    const rs = orderedLayout(kids.map((k) => k.value), inner);
    kids.forEach((k, i) => rec(k, rs[i], depth + 1));
  };
  rec(view, { x: 0, y: 0, w: W, h: H }, 0);
  return items;
}

export function ColumnTreemap({ block }: { block: ColumnTreemapBlock }) {
  const narrow = useNarrow();
  const W = narrow ? 320 : 720, H = narrow ? 440 : 460;
  const maxDepth = block.maxDepth ?? 3;
  const { byId, parent } = useMemo(() => {
    const byId = new Map<string, TreemapNode>();
    const parent = new Map<string, string>();
    const walk = (n: TreemapNode) => {
      byId.set(n.id, n);
      (n.children ?? []).forEach((c) => { parent.set(c.id, n.id); walk(c); });
    };
    walk(block.root);
    return { byId, parent };
  }, [block.root]);
  const startId = block.start && byId.has(block.start) ? block.start : block.root.id;
  const [viewId, setViewId] = useState(startId);
  const [mi, setMi] = useState(block.defaultMetric ?? 0);
  const [selId, setSelId] = useState<string | null>(null);
  const view = byId.get(viewId) ?? block.root;
  const metric = block.metrics[mi] ?? block.metrics[0];
  const items = useMemo(() => layout(view, W, H, maxDepth), [view, W, H, maxDepth]);

  const ancestors = (id: string): TreemapNode[] => {
    const out: TreemapNode[] = [];
    let cur: string | undefined = id;
    while (cur) { const n = byId.get(cur); if (n) out.unshift(n); cur = parent.get(cur); }
    return out;
  };
  const crumbs = ancestors(view.id);
  const sel = (selId && byId.get(selId)) || view;
  const selPath = ancestors(sel.id);
  // 選んだ節から表示中の節までの間にある、潜れる系統（表示中の節の子から、選んだ節の親まで。選んだ節が系統ならそれも）
  const vi = selPath.findIndex((n) => n.id === view.id);
  const drillTargets = vi >= 0 ? selPath.slice(vi + 1).filter((n) => n.children?.length) : [];

  const drill = (id: string) => { setViewId(id); setSelId(null); };

  const kindText = (n: TreemapNode) =>
    n.kind === "line" ? "系統" : n.kind === "branch" ? "父系の祖先でまとめた群" : n.kind === "pool" ? `${n.sires ?? ""}頭をまとめた群` : n.children?.length ? "群" : "種牡馬";
  const mv = (n: TreemapNode, i: number) => n.metrics[i]?.[0] ?? null;

  return (
    <figure className="card" style={{ padding: "12px 12px 8px" }}>
      <figcaption className="font-bold" style={{ fontSize: 15, marginBottom: 8 }}>{block.title}</figcaption>

      <div className="flex flex-wrap items-center gap-2" style={{ marginBottom: 6 }}>
        <label style={{ fontSize: 12, color: "var(--text-dim)" }} htmlFor={`tm-${block.root.id}`}>色</label>
        <select id={`tm-${block.root.id}`} value={mi} onChange={(e) => setMi(Number(e.target.value))}
          style={{ fontSize: 13, padding: "3px 6px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface2)", color: "var(--text)", maxWidth: "100%" }}>
          {block.metrics.map((m, i) => <option key={m.key} value={i}>{m.label}</option>)}
        </select>
      </div>

      {/* 凡例（発散型） */}
      <div style={{ marginBottom: 6 }}>
        <div style={{ height: 10, borderRadius: 3, border: "1px solid var(--border)",
          background: `linear-gradient(90deg, rgba(${LOW_RGB},0.78), ${NEUTRAL} 50%, rgba(${HIGH_RGB},0.78))` }} />
        <div className="flex justify-between gap-2" style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 2 }}>
          <span>← {metric.low}（{formatTreemapValue(metric.center - metric.scale, metric.format)}）</span>
          <span>{formatTreemapValue(metric.center, metric.format)}</span>
          <span style={{ textAlign: "right" }}>{metric.high}（{formatTreemapValue(metric.center + metric.scale, metric.format)}）→</span>
        </div>
        {metric.desc && <p style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 2 }}>{metric.kind ? `${metric.kind}: ` : ""}{metric.desc}</p>}
      </div>

      {/* パンくず */}
      <nav aria-label="系統の階層" className="flex flex-wrap items-center" style={{ fontSize: 12, gap: 4, marginBottom: 4 }}>
        {crumbs.map((n, i) => (
          <span key={n.id} className="flex items-center" style={{ gap: 4 }}>
            {i > 0 && <span style={{ color: "var(--text-dim)" }}>›</span>}
            {i < crumbs.length - 1 ? (
              <button type="button" onClick={() => drill(n.id)} style={{ color: "var(--accent)", background: "none", border: 0, padding: 0, cursor: "pointer", fontSize: 12 }}>{n.name}</button>
            ) : <span className="font-bold">{n.name}</span>}
          </span>
        ))}
      </nav>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", maxWidth: W, touchAction: "manipulation" }}
        role="img" aria-label={`${block.title}（${view.name}）`}>
        {items.map(({ node, rect, depth }) => {
          if (depth === 0 || rect.w < 0.5 || rect.h < 0.5) return null;
          const leaf = !node.children?.length;
          const t = tOf(mv(node, mi), metric);
          const tip = `${node.name}（${node.value.toLocaleString()}走）: ${metric.label} ${formatTreemapValue(mv(node, mi), metric.format)}`;
          if (leaf) {
            const size = narrow ? 10 : 11;
            const label = rect.w >= 30 && rect.h >= 13 ? fit(node.name, rect.w - 6, size) : "";
            const showVal = label && rect.h >= 30 && rect.w >= 40;
            const on = selId === node.id;
            return (
              <g key={node.id} style={{ cursor: "pointer" }}
                onMouseEnter={() => setSelId(node.id)}
                onClick={() => setSelId(node.id)}>
                <title>{tip}</title>
                <rect x={rect.x + 0.5} y={rect.y + 0.5} width={Math.max(0, rect.w - 1)} height={Math.max(0, rect.h - 1)} rx={2} fill={NEUTRAL} />
                <rect x={rect.x + 0.5} y={rect.y + 0.5} width={Math.max(0, rect.w - 1)} height={Math.max(0, rect.h - 1)} rx={2} fill={overlay(t)}
                  stroke={on ? "var(--text)" : "none"} strokeWidth={on ? 1.5 : 0} />
                {label && <text x={rect.x + 3} y={rect.y + size + 2} fontSize={size} fill="var(--text)" style={{ pointerEvents: "none" }}>{label}</text>}
                {showVal && <text x={rect.x + 3} y={rect.y + size * 2 + 4} fontSize={size - 1} fill="var(--text)" opacity={0.75} style={{ pointerEvents: "none" }}>
                  {fit(formatTreemapValue(mv(node, mi), metric.format), rect.w - 6, size - 1)}</text>}
              </g>
            );
          }
          if (depth > maxDepth) {
            return <rect key={node.id} x={rect.x} y={rect.y} width={rect.w} height={rect.h} fill="none" stroke="var(--border)" strokeWidth={0.6} style={{ pointerEvents: "none" }} />;
          }
          const head = rect.h >= HEADER * 2.2 && rect.w >= 44;
          const size = depth === 1 ? 12 : 11;
          return (
            <g key={node.id}>
              <rect x={rect.x} y={rect.y} width={rect.w} height={rect.h} rx={3} fill="none" stroke="var(--text-dim)" strokeOpacity={depth === 1 ? 0.9 : 0.5}
                strokeWidth={depth === 1 ? 1.4 : 0.9} style={{ pointerEvents: "none" }} />
              {head && (
                <g style={{ cursor: "pointer" }} onClick={() => drill(node.id)} onMouseEnter={() => setSelId(node.id)}>
                  <title>{`${tip}（タップでこの系統へ）`}</title>
                  <rect x={rect.x + 0.5} y={rect.y + 0.5} width={Math.max(0, rect.w - 1)} height={HEADER} fill="var(--surface2)" />
                  <rect x={rect.x + 0.5} y={rect.y + 0.5} width={5} height={HEADER} fill={overlay(tOf(mv(node, mi), metric))} />
                  <text x={rect.x + 8} y={rect.y + 12} fontSize={size} fontWeight={depth === 1 ? 700 : 600} fill="var(--text)" style={{ pointerEvents: "none" }}>
                    {fit(`${node.name} ›`, rect.w - 12, size)}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>

      {/* 選んだ節の値（ホバー・タップ） */}
      <div style={{ marginTop: 8, padding: "8px 10px", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }} aria-live="polite">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="font-bold" style={{ fontSize: 14 }}>{sel.name}</span>
          <span style={{ color: "var(--text-dim)" }}>{kindText(sel)}・産駒 {sel.value.toLocaleString()} 走</span>
        </div>
        {selPath.length > 1 && (
          <p style={{ color: "var(--text-dim)", marginTop: 2 }}>{selPath.slice(0, -1).map((n) => n.name).join(" › ")}</p>
        )}
        <table style={{ width: "100%", marginTop: 6, borderCollapse: "collapse", fontVariantNumeric: "tabular-nums" }}>
          <tbody>
            {block.metrics.map((m, i) => {
              const [v, lo, hi] = sel.metrics[i] ?? [];
              return (
                <tr key={m.key} style={{ borderTop: "1px solid var(--border)", fontWeight: i === mi ? 700 : 400 }}>
                  <td style={{ padding: "3px 4px", whiteSpace: "nowrap" }}>
                    <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, marginRight: 4, background: overlay(tOf(v, m)), border: "1px solid var(--border)" }} />
                    {m.label}
                  </td>
                  <td style={{ padding: "3px 4px", textAlign: "right" }}>{formatTreemapValue(v, m.format)}</td>
                  <td style={{ padding: "3px 4px", textAlign: "right", color: "var(--text-dim)" }}>
                    {typeof lo === "number" && typeof hi === "number" ? `${formatTreemapValue(lo, m.format)}〜${formatTreemapValue(hi, m.format)}` : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {drillTargets.length > 0 && (
          <div className="flex flex-wrap items-center gap-1" style={{ marginTop: 6 }}>
            <span style={{ color: "var(--text-dim)" }}>この系統へ:</span>
            {drillTargets.map((n) => (
              <button key={n.id} type="button" onClick={() => drill(n.id)}
                style={{ fontSize: 12, padding: "2px 8px", borderRadius: 999, border: "1px solid var(--border)", background: "var(--surface2)", color: "var(--text)", cursor: "pointer" }}>
                {n.name}
              </button>
            ))}
          </div>
        )}
      </div>
      {block.caption && <p style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 6 }}>{block.caption}</p>}
    </figure>
  );
}
