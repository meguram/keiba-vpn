"use client";

import { useState } from "react";
import { useNarrow } from "@/components/RaceStrip";
import { chartIsEmpty, chartMax, type ColumnChartBlock } from "@/lib/columns";

const COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff"];

/** 折れ線グラフ（SVG）。系列はボタンで切り替え・重ね描き（最大 4 つ）。baseline は常に点線 */
export function ColumnChart({ block }: { block: ColumnChartBlock }) {
  const narrow = useNarrow();
  // 狭い画面では描画領域を小さく取り（文字が縮まない）、横軸の文字を斜めにする
  const W = narrow ? 360 : 640, H = narrow ? 300 : 300, L = 42, R = narrow ? 22 : 34, T = 16, B = narrow ? 74 : 44;
  const choices = block.series.map((s, i) => ({ s, i })).filter((x) => !x.s.baseline);
  const base = block.series.filter((s) => s.baseline);
  const [picked, setPicked] = useState<number[]>(choices.length ? [choices[0].i] : []);
  const empty = chartIsEmpty(block.series);
  const max = chartMax(block.series);
  const n = block.xs.length;
  const x = (k: number) => L + (n <= 1 ? 0 : (k * (W - L - R)) / (n - 1));
  const y = (v: number) => T + (1 - v / max) * (H - T - B);
  const step = max <= 0.2 ? 0.05 : 0.1;
  const ticks = Array.from({ length: Math.round(max / step) + 1 }, (_, k) => k * step);
  const toggle = (i: number) =>
    setPicked((cur) => (cur.includes(i) ? (cur.length > 1 ? cur.filter((c) => c !== i) : cur) : cur.length >= COLORS.length ? cur : [...cur, i]));

  const path = (vals: (number | null)[]) => {
    let d = "", pen = false;
    vals.forEach((v, k) => {
      if (typeof v !== "number") { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)} `;
      pen = true;
    });
    return d.trim();
  };

  return (
    <figure className="card" style={{ padding: "12px 12px 8px" }}>
      <figcaption className="flex flex-wrap items-center justify-between gap-2" style={{ marginBottom: 8 }}>
        <span className="font-bold" style={{ fontSize: 15 }}>{block.title}</span>
        {block.dummy && <span style={{ fontSize: 11, color: "#d29922", border: "1px solid #d2992280", borderRadius: 999, padding: "1px 8px" }}>ダミー表示（実データではありません）</span>}
      </figcaption>

      {choices.length > 1 && (
        <div className="flex flex-wrap gap-2" style={{ marginBottom: 8 }} role="group" aria-label="表示する系列">
          {choices.map(({ s, i }) => {
            const on = picked.includes(i);
            const c = COLORS[Math.max(0, picked.indexOf(i)) % COLORS.length];
            return (
              <button key={i} type="button" onClick={() => toggle(i)} aria-pressed={on}
                style={{ fontSize: 12, padding: "3px 10px", borderRadius: 999, cursor: "pointer",
                  border: `1px solid ${on ? c : "var(--border)"}`, background: on ? `${c}26` : "transparent", color: on ? c : "var(--text-dim)" }}>
                {s.name}
              </button>
            );
          })}
        </div>
      )}

      <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${block.title}（横軸 ${block.xLabel}、縦軸 ${block.yLabel}）`} style={{ display: "block" }}>
          {ticks.map((t, k) => (
            <g key={k}>
              <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
              <text x={L - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-dim)">{Math.round(t * 100)}%</text>
            </g>
          ))}
          {block.xs.map((lab, k) => (
            narrow
              ? <text key={k} x={x(k)} y={H - B + 14} textAnchor="end" fontSize={11} fill="var(--text-dim)" transform={`rotate(-40 ${x(k)} ${H - B + 14})`}>{lab}</text>
              : <text key={k} x={x(k)} y={H - B + 16} textAnchor="middle" fontSize={11} fill="var(--text-dim)">{lab}</text>
          ))}
          <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--text-dim)">{block.xLabel}</text>
          <text x={12} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} fill="var(--text-dim)" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>{block.yLabel}</text>

          {base.map((s, k) => <path key={`b${k}`} d={path(s.values)} fill="none" stroke="#8b949e" strokeWidth={2} strokeDasharray="5 4" />)}
          {picked.map((i, k) => {
            const s = block.series[i];
            const c = COLORS[k % COLORS.length];
            return (
              <g key={i}>
                <path d={path(s.values)} fill="none" stroke={c} strokeWidth={2.5} strokeLinejoin="round" />
                {s.values.map((v, j) => typeof v === "number" && <circle key={j} cx={x(j)} cy={y(v)} r={3.5} fill={c}><title>{`${s.name} ${block.xs[j]}: ${(v * 100).toFixed(1)}%`}</title></circle>)}
              </g>
            );
          })}
        </svg>
        {empty && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
            <span style={{ fontSize: 13, color: "var(--text-dim)", background: "var(--surface)", padding: "4px 14px", borderRadius: 999, border: "1px solid var(--border)" }}>集計中（公開までお待ちください）</span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-4" style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
        {base.map((s, k) => <span key={k}><span style={{ display: "inline-block", width: 18, borderTop: "2px dashed #8b949e", verticalAlign: "middle", marginRight: 4 }} />{s.name}</span>)}
        {block.caption && <span>{block.caption}</span>}
      </div>
    </figure>
  );
}
