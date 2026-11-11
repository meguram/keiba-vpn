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
const VIEWS = [{ key: "cloud" as const, label: "ワードクラウド" }, { key: "list" as const, label: "一覧（並べ替え）" }];
const SORTS = [
  { key: "rate-desc" as const, label: "好走率が高い順" },
  { key: "rate-asc" as const, label: "好走率が低い順" },
  { key: "n-desc" as const, label: "出た数が多い順" },
  { key: "win-desc" as const, label: "単勝回収率が高い順" },
  { key: "place-desc" as const, label: "複勝回収率が高い順" },
];
type SortKey = (typeof SORTS)[number]["key"];
const pct0 = (v: number) => `${Math.round(v * 100)}%`;
const MODES = [{ key: "lift" as const, label: "そのままの倍率" }, { key: "adj" as const, label: "人気をそろえた倍率" }];

export function ColumnWordcloud({ block }: { block: ColumnWordcloudBlock }) {
  const narrow = useNarrow();
  const [mode, setMode] = useState<"lift" | "adj">("lift");
  const [sel, setSel] = useState<WordcloudItem | null>(null);
  const [view, setView] = useState<"cloud" | "list">("cloud");
  const [sort, setSort] = useState<SortKey>("rate-desc");
  const d = narrow ? block.narrow : block.wide;
  const dim: React.CSSProperties = { color: "var(--text-dim)", fontSize: 12 };
  const pill = (on: boolean): React.CSSProperties => ({ padding: "4px 12px", borderRadius: 999, fontSize: 13, cursor: "pointer",
    border: "1px solid " + (on ? "var(--accent)" : "var(--border)"), background: on ? "var(--accent)" : "transparent", color: on ? "#fff" : "var(--text)" });
  const rows = [...block.wide.items].sort((a, b) => (sort === "rate-desc" ? b.rate - a.rate : sort === "rate-asc" ? a.rate - b.rate : sort === "win-desc" ? b.win - a.win : sort === "place-desc" ? b.place - a.place : b.n - a.n));
  const maxRate = Math.max(...rows.map((r) => r.rate), block.base);
  return (
    <figure className="card" style={{ padding: "12px 12px 10px" }}>
      <p className="font-bold text-sm" style={{ marginBottom: 8 }}>{block.title}</p>
      <div role="tablist" style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
        {VIEWS.map((v) => (
          <button key={v.key} role="tab" aria-selected={view === v.key} onClick={() => setView(v.key)} style={pill(view === v.key)}>{v.label}</button>
        ))}
        <span style={{ width: 1, background: "var(--border)", margin: "0 4px" }} />
        {MODES.map((m) => (
          <button key={m.key} role="tab" aria-selected={mode === m.key} onClick={() => setMode(m.key)} style={pill(mode === m.key)}>{m.label}</button>
        ))}
      </div>
      {view === "list" && (
        <>
          <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
            {SORTS.map((o) => (
              <button key={o.key} aria-pressed={sort === o.key} onClick={() => setSort(o.key)} style={pill(sort === o.key)}>{o.label}</button>
            ))}
          </div>
          <div style={{ maxHeight: 460, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 8, padding: "6px 8px" }}>
            {(() => {
              const grid: React.CSSProperties = { display: "grid", gridTemplateColumns: narrow ? "18px minmax(64px, 1fr) 0.6fr 40px 34px 34px 38px" : "26px minmax(90px, 1.1fr) 3fr 52px 52px 52px 64px", gap: narrow ? 5 : 8, alignItems: "center", fontSize: narrow ? 12 : 13, padding: "2px 0" };
              return (
                <div style={{ ...grid, ...dim, fontSize: 11, position: "sticky", top: -6, background: "var(--bg)" }}>
                  <span /><span>語・句</span><span /><span style={{ textAlign: "right" }}>3着内</span><span style={{ textAlign: "right" }}>単勝</span><span style={{ textAlign: "right" }}>複勝</span><span style={{ textAlign: "right" }}>数</span>
                </div>
              );
            })()}
            {rows.map((r, i) => (
              <div key={r.label} style={{ display: "grid", gridTemplateColumns: narrow ? "18px minmax(64px, 1fr) 0.6fr 40px 34px 34px 38px" : "26px minmax(90px, 1.1fr) 3fr 52px 52px 52px 64px", gap: narrow ? 5 : 8, alignItems: "center", fontSize: narrow ? 12 : 13, padding: "2px 0" }}>
                <span style={dim}>{i + 1}</span>
                <span style={{ color: colorOf(r[mode]), fontWeight: 600 }}>{r.label}</span>
                <span style={{ position: "relative", height: 10, background: "rgba(139,148,158,0.15)", borderRadius: 5 }}>
                  <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${(r.rate / maxRate) * 100}%`, background: colorOf(r[mode]), borderRadius: 5 }} />
                  <span style={{ position: "absolute", top: -3, bottom: -3, left: `${(block.base / maxRate) * 100}%`, borderLeft: "1px dashed var(--text-dim)" }} />
                </span>
                <span className="font-mono" style={{ textAlign: "right" }}>{(r.rate * 100).toFixed(1)}%</span>
                <span className="font-mono" style={{ textAlign: "right", fontWeight: r.win >= 1 ? 700 : 400 }}>{pct0(r.win)}</span>
                <span className="font-mono" style={{ textAlign: "right", fontWeight: r.place >= 1 ? 700 : 400 }}>{pct0(r.place)}</span>
                <span className="font-mono" style={{ textAlign: "right", ...dim }}>{r.n.toLocaleString("en-US")}</span>
              </div>
            ))}
          </div>
          <p style={{ ...dim, marginTop: 6 }}>棒 = その語を含む馬の3着以内率、点線 = 全体（{(block.base * 100).toFixed(1)}%）、色は上のボタンで選んだ倍率。単勝・複勝 = その語の馬を毎回100円買い続けた回収率（太字 = 100%超。出た数が少ないとぶれる）、右端は出た数。</p>
        </>
      )}
      {view === "cloud" && (<>
      <svg viewBox={`0 0 ${d.w} ${d.h}`} role="img" aria-label={block.title}
        style={{ width: "100%", height: "auto", background: "#0b1224", border: "1px solid var(--border)", borderRadius: 8 }}
        onClick={() => setSel(null)}>
        {d.items.map((it) => (
          <text key={it.label} x={it.x} y={it.y} fontSize={it.size} fill={colorOf(it[mode])} textAnchor="middle"
            dominantBaseline="central" fontWeight={it.size > 34 ? 700 : 500} style={{ cursor: "pointer" }}
            onClick={(e) => { e.stopPropagation(); setSel(it); }}>
            <title>{`${it.label}\n出た数 ${it.n.toLocaleString("en-US")} / 3着以内 ${(it.rate * 100).toFixed(1)}%（全体 ${(block.base * 100).toFixed(1)}%）\n倍率 ${it.lift.toFixed(2)} / 人気をそろえた倍率 ${it.adj.toFixed(2)}\n単勝回収率 ${pct0(it.win)} / 複勝回収率 ${pct0(it.place)}`}</title>
            {it.label}
          </text>
        ))}
      </svg>
      <p className="text-sm" style={{ ...dim, minHeight: 22, marginTop: 6 }}>
        {sel
          ? `「${sel.label}」 出た数 ${sel.n.toLocaleString("en-US")} ・ 3着以内 ${(sel.rate * 100).toFixed(1)}%（全体 ${(block.base * 100).toFixed(1)}%） ・ 倍率 ${sel.lift.toFixed(2)} ・ 人気をそろえた倍率 ${sel.adj.toFixed(2)} ・ 単勝回収率 ${pct0(sel.win)} ・ 複勝回収率 ${pct0(sel.place)}`
          : "語をタップすると数値が出ます。"}
      </p>
      </>)}
      <div style={{ maxWidth: 420, margin: "6px 0 2px" }}>
        <div style={{ ...dim, display: "flex", justifyContent: "space-between" }}><span>凡走寄り（橙）</span><span>好走寄り（青）</span></div>
        <div style={{ height: 10, borderRadius: 5, background: `linear-gradient(90deg,${LEGEND})` }} />
        <div style={{ ...dim, display: "flex", justifyContent: "space-between", fontSize: 11 }}><span>×0.66</span><span>×1.0</span><span>×1.52</span></div>
      </div>
      {block.caption && <figcaption style={{ ...dim, marginTop: 6 }}>{block.caption}</figcaption>}
    </figure>
  );
}
