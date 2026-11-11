"use client";

import { useState } from "react";
import { useNarrow } from "@/components/RaceStrip";
import type { ColumnWordcloudBlock, WordcloudItem } from "@/lib/columns";

const GREY = [148, 163, 184];
const GOOD = [79, 139, 255];
const BAD = [251, 146, 60];

/** log2(倍率) を ±0.6 で飽和させた発散色。1 より大きいと青、小さいと橙 */
function colorOf(lift: number): string {
  const x = Math.max(-1, Math.min(1, Math.log2(Math.max(lift, 1e-6)) / 0.6));
  const to = x > 0 ? GOOD : BAD;
  const c = GREY.map((g, i) => Math.round(g + (to[i] - g) * Math.abs(x)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const LEGEND = Array.from({ length: 21 }, (_, i) => colorOf(2 ** ((i / 10 - 1) * 0.6))).join(",");
const MODES = [{ key: "lift" as const, label: "そのままの倍率" }, { key: "adj" as const, label: "人気をそろえた倍率" }];

export function ColumnWordcloud({ block }: { block: ColumnWordcloudBlock }) {
  const narrow = useNarrow();
  const [mode, setMode] = useState<"lift" | "adj">("lift");
  const [sel, setSel] = useState<WordcloudItem | null>(null);
  const d = narrow ? block.narrow : block.wide;
  const dim: React.CSSProperties = { color: "var(--text-dim)", fontSize: 12 };
  return (
    <figure className="card" style={{ padding: "12px 12px 10px" }}>
      <p className="font-bold text-sm" style={{ marginBottom: 8 }}>{block.title}</p>
      <div role="tablist" style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
        {MODES.map((m) => (
          <button key={m.key} role="tab" aria-selected={mode === m.key} onClick={() => setMode(m.key)}
            style={{ padding: "4px 12px", borderRadius: 999, fontSize: 13, cursor: "pointer",
              border: "1px solid " + (mode === m.key ? "var(--accent)" : "var(--border)"),
              background: mode === m.key ? "var(--accent)" : "transparent", color: mode === m.key ? "#fff" : "var(--text)" }}>
            {m.label}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${d.w} ${d.h}`} role="img" aria-label={block.title}
        style={{ width: "100%", height: "auto", background: "#0b1224", border: "1px solid var(--border)", borderRadius: 8 }}
        onClick={() => setSel(null)}>
        {d.items.map((it) => (
          <text key={it.label} x={it.x} y={it.y} fontSize={it.size} fill={colorOf(it[mode])} textAnchor="middle"
            dominantBaseline="central" fontWeight={it.size > 34 ? 700 : 500} style={{ cursor: "pointer" }}
            onClick={(e) => { e.stopPropagation(); setSel(it); }}>
            <title>{`${it.label}\n出た数 ${it.n.toLocaleString("en-US")} / 3着以内 ${(it.rate * 100).toFixed(1)}%（全体 ${(block.base * 100).toFixed(1)}%）\n倍率 ${it.lift.toFixed(2)} / 人気をそろえた倍率 ${it.adj.toFixed(2)}`}</title>
            {it.label}
          </text>
        ))}
      </svg>
      <p className="text-sm" style={{ ...dim, minHeight: 22, marginTop: 6 }}>
        {sel
          ? `「${sel.label}」 出た数 ${sel.n.toLocaleString("en-US")} ・ 3着以内 ${(sel.rate * 100).toFixed(1)}%（全体 ${(block.base * 100).toFixed(1)}%） ・ 倍率 ${sel.lift.toFixed(2)} ・ 人気をそろえた倍率 ${sel.adj.toFixed(2)}`
          : "語をタップすると数値が出ます。"}
      </p>
      <div style={{ maxWidth: 420, margin: "6px 0 2px" }}>
        <div style={{ ...dim, display: "flex", justifyContent: "space-between" }}><span>凡走寄り（橙）</span><span>好走寄り（青）</span></div>
        <div style={{ height: 10, borderRadius: 5, background: `linear-gradient(90deg,${LEGEND})` }} />
        <div style={{ ...dim, display: "flex", justifyContent: "space-between", fontSize: 11 }}><span>×0.66</span><span>×1.0</span><span>×1.52</span></div>
      </div>
      {block.caption && <figcaption style={{ ...dim, marginTop: 6 }}>{block.caption}</figcaption>}
    </figure>
  );
}
