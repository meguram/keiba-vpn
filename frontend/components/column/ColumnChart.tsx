"use client";

import { useState, useMemo } from "react";
import { useNarrow } from "@/components/RaceStrip";
import { axis2Max, chartIsEmpty, chartMax, chartStep, chartVariant, formatP, formatValue, SIGNIFICANT_P, type ColumnChartBlock, type JockeyDistanceChartBlock } from "@/lib/columns";

const COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff"];
/** 右の縦軸の系列の色（左の系列の色と重ならないもの） */
const Y2_COLOR = "#f778ba";

/** 騎手別・距離帯別勝率グラフ（プルダウン＋ソート切り替え） */
export function JockeyDistanceChart({ block }: { block: JockeyDistanceChartBlock }) {
  const narrow = useNarrow();
  const W = narrow ? 360 : 640, H = narrow ? 300 : 300, L = 42, R = narrow ? 22 : 34, T = 16, B = narrow ? 74 : 54;
  const [sortMode, setSortMode] = useState<"rank" | "reading">("rank");
  const [selectedName, setSelectedName] = useState(block.jockeys[0]?.name ?? "");

  const sorted = useMemo(
    () => [...block.jockeys].sort((a, b) =>
      sortMode === "rank" ? a.rank - b.rank : a.reading.localeCompare(b.reading, "ja")
    ),
    [block.jockeys, sortMode]
  );

  const jockey = block.jockeys.find((j) => j.name === selectedName) ?? block.jockeys[0];
  const allVals = [...(jockey?.values ?? []), ...(block.baseline ?? [])].filter((v): v is number => typeof v === "number");
  const max = allVals.length ? Math.round(Math.max(0.1, Math.ceil(Math.max(...allVals) / 0.05 + 1e-9) * 0.05) * 100) / 100 : 0.4;
  const n = block.xs.length;
  const x = (k: number) => L + (n <= 1 ? 0 : (k * (W - L - R)) / (n - 1));
  const y = (v: number) => T + (1 - v / max) * (H - T - B);
  const step = max <= 0.2 ? 0.05 : 0.1;
  const ticks = Array.from({ length: Math.round(max / step) + 1 }, (_, k) => k * step);

  const linePath = (vals: (number | null)[]) => {
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
      <figcaption className="font-bold" style={{ fontSize: 15, marginBottom: 10 }}>{block.title}</figcaption>

      {/* ソート切り替え + プルダウン */}
      <div className="flex flex-wrap gap-2 items-center" style={{ marginBottom: 8 }}>
        <div className="flex" style={{ borderRadius: 6, overflow: "hidden", border: "1px solid var(--border)", flexShrink: 0 }}>
          {(["rank", "reading"] as const).map((m) => (
            <button key={m} type="button" onClick={() => setSortMode(m)}
              style={{ fontSize: 12, padding: "4px 10px", cursor: "pointer", border: "none",
                background: sortMode === m ? "#58a6ff26" : "transparent",
                color: sortMode === m ? "#58a6ff" : "var(--text-dim)",
                borderRight: m === "rank" ? "1px solid var(--border)" : "none" }}>
              {m === "rank" ? "乗鞍数順" : "五十音順"}
            </button>
          ))}
        </div>
        <select value={selectedName} onChange={(e) => setSelectedName(e.target.value)}
          style={{ fontSize: 13, padding: "4px 8px", borderRadius: 6, border: "1px solid var(--border)",
            background: "var(--surface)", color: "var(--text)", flex: 1, minWidth: 120, maxWidth: 220 }}>
          {sorted.map((j) => (
            <option key={j.name} value={j.name}>{j.name}</option>
          ))}
        </select>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label={`${jockey?.name}の距離帯別勝率`} style={{ display: "block" }}>
        {ticks.map((t, k) => (
          <g key={k}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
            <text x={L - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-dim)">{Math.round(t * 100)}%</text>
          </g>
        ))}
        {block.xs.map((lab, k) => (
          narrow
            ? <text key={k} x={x(k)} y={H - B + 14} textAnchor="end" fontSize={10} fill="var(--text-dim)" transform={`rotate(-40 ${x(k)} ${H - B + 14})`}>{lab}</text>
            : <text key={k} x={x(k)} y={H - B + 16} textAnchor="middle" fontSize={11} fill="var(--text-dim)">{lab}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--text-dim)">距離</text>
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} fill="var(--text-dim)" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>勝率</text>

        {/* baseline */}
        {block.baseline && (
          <path d={linePath(block.baseline)} fill="none" stroke="#8b949e" strokeWidth={2} strokeDasharray="5 4" />
        )}
        {/* selected jockey */}
        {jockey && (
          <g>
            <path d={linePath(jockey.values)} fill="none" stroke="#58a6ff" strokeWidth={2.5} strokeLinejoin="round" />
            {jockey.values.map((v, k) =>
              typeof v === "number" && (
                <circle key={k} cx={x(k)} cy={y(v)} r={3.5} fill="#58a6ff">
                  <title>{`${jockey.name} ${block.xs[k]}: ${(v * 100).toFixed(1)}%`}</title>
                </circle>
              )
            )}
          </g>
        )}
      </svg>

      <div className="flex flex-wrap gap-4" style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
        <span><span style={{ display: "inline-block", width: 18, borderTop: "2px dashed #8b949e", verticalAlign: "middle", marginRight: 4 }} />全体平均</span>
        <span><span style={{ display: "inline-block", width: 18, borderTop: "2.5px solid #58a6ff", verticalAlign: "middle", marginRight: 4 }} />{jockey?.name}</span>
        {block.caption && <span>{block.caption}</span>}
      </div>
    </figure>
  );
}

/** 折れ線グラフ（SVG）。系列はボタンで切り替え・重ね描き（最大 4 つ）。baseline は常に点線。
    compare があれば「その条件」と「それ以外」の 2 系列を最初から出し、95% 信頼区間の縦線と、差が有意な点の印（＊）を描く */
export function ColumnChart({ block: whole }: { block: ColumnChartBlock }) {
  const narrow = useNarrow();
  const [filterSel, setFilterSel] = useState<string[]>(() => (whole.filters ?? []).map((f) => f.options[0] ?? ""));
  const block = chartVariant(whole, filterSel);
  // 狭い画面では描画領域を小さく取り（文字が縮まない）、横軸の文字を斜めにする（斜めの先頭の文字が左で切れないよう左を広めに）
  const y2 = block.y2;
  const W = narrow ? 360 : 640, H = narrow ? 300 : 300, L = narrow ? 58 : y2 ? 52 : 42, R = y2 ? (narrow ? 44 : 60) : narrow ? 22 : 34, T = 16, B = narrow ? 74 : 44;
  const choices = block.series.map((s, i) => ({ s, i })).filter((x) => !x.s.baseline);
  const base = block.series.filter((s) => s.baseline);
  const cmp = block.compare;
  const pairIdx = cmp ? [cmp.a, cmp.b].map((nm) => block.series.findIndex((s) => s.name === nm)).filter((i) => i >= 0) : [];
  const [picked, setPicked] = useState<number[]>(pairIdx.length === 2 ? pairIdx : choices.length ? [choices[0].i] : []);
  const showSig = pairIdx.length === 2 && pairIdx.every((i) => picked.includes(i));
  const fmt = block.format ?? "pct";
  const pct = (v: number) => formatValue(v, fmt);
  const empty = chartIsEmpty(block.series);
  const max = chartMax(block.series, fmt);
  const n = block.xs.length;
  const x = (k: number) => L + (n <= 1 ? 0 : (k * (W - L - R)) / (n - 1));
  const y = (v: number) => T + (1 - v / max) * (H - T - B);
  const step = chartStep(max, fmt);
  const ticks = Array.from({ length: Math.round(max / step) + 1 }, (_, k) => k * step);
  const fmt2 = y2?.format ?? "num";
  const max2 = y2 ? axis2Max(y2.series, ticks.length - 1, fmt2) : 1;
  const yR = (v: number) => T + (1 - v / max2) * (H - T - B);
  const toggle = (i: number) =>
    setPicked((cur) => (cur.includes(i) ? (cur.length > 1 ? cur.filter((c) => c !== i) : cur) : cur.length >= COLORS.length ? cur : [...cur, i]));

  const path = (vals: (number | null)[], yy: (v: number) => number = y) => {
    let d = "", pen = false;
    vals.forEach((v, k) => {
      if (typeof v !== "number") { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(k).toFixed(1)},${yy(v).toFixed(1)} `;
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

      {whole.filters?.length ? (
        <div className="flex flex-wrap gap-2" style={{ marginBottom: 8 }}>
          {whole.filters.map((f, fi) => (
            <label key={f.label} className="flex items-center gap-1" style={{ fontSize: 12, color: "var(--text-dim)" }}>
              {f.label}
              <select value={filterSel[fi]} onChange={(e) => setFilterSel((cur) => cur.map((v, i) => (i === fi ? e.target.value : v)))}
                style={{ fontSize: 14, minHeight: 34, padding: "0 8px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)" }}>
                {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </label>
          ))}
        </div>
      ) : null}

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
              <text x={L - 6} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-dim)">{fmt === "num" ? Math.round(t) : `${Math.round(t * 100)}%`}</text>
              {y2 && <text x={W - R + 6} y={y(t) + 4} textAnchor="start" fontSize={11} fill={Y2_COLOR}>{fmt2 === "num" ? +(k * max2 / (ticks.length - 1)).toFixed(2) : `${Math.round(k * max2 / (ticks.length - 1) * 100)}%`}</text>}
            </g>
          ))}
          {block.xs.map((lab, k) => (
            narrow
              ? <text key={k} x={x(k)} y={H - B + 14} textAnchor="end" fontSize={11} fill="var(--text-dim)" transform={`rotate(-40 ${x(k)} ${H - B + 14})`}>{lab}</text>
              : <text key={k} x={x(k)} y={H - B + 16} textAnchor="middle" fontSize={11} fill="var(--text-dim)">{lab}</text>
          ))}
          <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--text-dim)">{block.xLabel}</text>
          <text x={12} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} fill="var(--text-dim)" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>{block.yLabel}{y2 ? "（左）" : ""}</text>
          {y2 && <text x={W - 8} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} fill={Y2_COLOR} transform={`rotate(90 ${W - 8} ${(T + H - B) / 2})`}>{y2.label}（右）</text>}

          {base.map((s, k) => <path key={`b${k}`} d={path(s.values)} fill="none" stroke="#8b949e" strokeWidth={2} strokeDasharray="5 4" />)}
          {picked.map((i, k) => {
            const s = block.series[i];
            const c = COLORS[k % COLORS.length];
            const dx = s.ci ? (k - (picked.length - 1) / 2) * 5 : 0;      // 信頼区間の縦線が重ならないよう少しずらす
            return (
              <g key={i}>
                <path d={path(s.values)} fill="none" stroke={c} strokeWidth={2.5} strokeLinejoin="round" />
                {s.ci?.map((ci, j) => ci && (
                  <g key={`ci${j}`} stroke={c} strokeWidth={1.4} opacity={0.75}>
                    <line x1={x(j) + dx} x2={x(j) + dx} y1={y(Math.min(ci[1], max))} y2={y(Math.max(0, ci[0]))} />
                    <line x1={x(j) + dx - 3} x2={x(j) + dx + 3} y1={y(Math.min(ci[1], max))} y2={y(Math.min(ci[1], max))} />
                    <line x1={x(j) + dx - 3} x2={x(j) + dx + 3} y1={y(Math.max(0, ci[0]))} y2={y(Math.max(0, ci[0]))} />
                  </g>
                ))}
                {s.values.map((v, j) => typeof v === "number" && (
                  <circle key={j} cx={x(j)} cy={y(v)} r={3.5} fill={c}>
                    <title>{`${s.name} ${block.xs[j]}: ${pct(v)}${s.n?.[j] != null ? `（${s.n[j]}走）` : ""}${s.ci?.[j] ? ` 95%区間 ${pct(s.ci[j]![0])}〜${pct(s.ci[j]![1])}` : ""}`}</title>
                  </circle>
                ))}
              </g>
            );
          })}
          {y2?.series.map((s, k) => (
            <g key={`y2${k}`}>
              <path d={path(s.values, yR)} fill="none" stroke={Y2_COLOR} strokeWidth={2.5} strokeDasharray="8 3" strokeLinejoin="round" />
              {s.ci?.map((ci, j) => ci && (
                <g key={`ci${j}`} stroke={Y2_COLOR} strokeWidth={1.4} opacity={0.75}>
                  <line x1={x(j) + 7} x2={x(j) + 7} y1={yR(Math.min(ci[1], max2))} y2={yR(Math.max(0, ci[0]))} />
                  <line x1={x(j) + 4} x2={x(j) + 10} y1={yR(Math.min(ci[1], max2))} y2={yR(Math.min(ci[1], max2))} />
                  <line x1={x(j) + 4} x2={x(j) + 10} y1={yR(Math.max(0, ci[0]))} y2={yR(Math.max(0, ci[0]))} />
                </g>
              ))}
              {s.values.map((v, j) => typeof v === "number" && (
                <rect key={j} x={x(j) - 4} y={yR(v) - 4} width={8} height={8} fill={Y2_COLOR}>
                  <title>{`${s.name} ${block.xs[j]}: ${formatValue(v, fmt2)}${s.n?.[j] != null ? `（${s.n[j]}走）` : ""}${s.ci?.[j] ? ` 95%区間 ${formatValue(s.ci[j]![0], fmt2)}〜${formatValue(s.ci[j]![1], fmt2)}` : ""}`}</title>
                </rect>
              ))}
            </g>
          ))}
          {showSig && cmp!.p.map((p, j) => {
            if (p == null || p >= SIGNIFICANT_P) return null;
            const top = Math.max(...pairIdx.map((i) => block.series[i].ci?.[j]?.[1] ?? block.series[i].values[j] ?? 0));
            return <text key={`sig${j}`} x={x(j)} y={Math.max(T + 10, y(Math.min(top, max)) - 6)} textAnchor="middle" fontSize={14} fontWeight={700} fill="var(--text)"><title>{`差の p 値 ${formatP(p)}`}</title>＊</text>;
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
        {y2?.series.map((s, k) => <span key={`l2${k}`} style={{ color: Y2_COLOR }}><span style={{ display: "inline-block", width: 18, borderTop: `2.5px dashed ${Y2_COLOR}`, verticalAlign: "middle", marginRight: 4 }} />■ {s.name}（右軸）</span>)}
        {showSig && <span>縦線は95%信頼区間。＊は2つの差が統計的に有意（p&lt;0.05）</span>}
        {block.caption && <span>{block.caption}</span>}
      </div>

      {pairIdx.length === 2 ? <CompareTable block={block} a={pairIdx[0]} b={pairIdx[1]} /> : (block.series.some((s) => s.n) || y2?.series.some((s) => s.n)) && <SeriesTable block={block} />}
    </figure>
  );
}

/** 比較グラフの数値（各点の率・標本数・差・p 値）。触れないと数値が出ないスマホでも確かめられるように */
function CompareTable({ block, a, b }: { block: ColumnChartBlock; a: number; b: number }) {
  const sa = block.series[a], sb = block.series[b];
  const fmt = block.format ?? "pct";
  const pct = (v: number | null | undefined) => formatValue(v, fmt);
  const TD = { padding: "5px 4px", borderTop: "1px solid var(--border)", textAlign: "right" as const, fontVariantNumeric: "tabular-nums" as const };
  const N = { display: "block", fontSize: 10, color: "var(--text-dim)" };
  return (
    <details style={{ marginTop: 8, fontSize: 12 }}>
      <summary style={{ cursor: "pointer", color: "var(--text-dim)", minHeight: 32, display: "flex", alignItems: "center" }}>数値と検定（標本数・差・p値）</summary>
      <div>
        <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 4 }}>
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th style={{ ...TD, textAlign: "left", borderTop: "none" }}>{block.xLabel}</th>
              <th style={{ ...TD, borderTop: "none" }}>{sa.name}</th>
              <th style={{ ...TD, borderTop: "none" }}>{sb.name}</th>
              <th style={{ ...TD, borderTop: "none" }}>差</th>
              <th style={{ ...TD, borderTop: "none" }}>p値</th>
            </tr>
          </thead>
          <tbody>
            {block.xs.map((lab, j) => {
              const va = sa.values[j], vb = sb.values[j], p = block.compare?.p[j];
              const sig = p != null && p < SIGNIFICANT_P;
              return (
                <tr key={j}>
                  <td style={{ ...TD, textAlign: "left", whiteSpace: "nowrap" }}>{lab.replace(/\n/g, " ")}</td>
                  <td style={TD} data-sort={va ?? ""}>{pct(va)}{sa.n?.[j] != null && <small style={N}>{sa.n[j]!.toLocaleString()}走</small>}</td>
                  <td style={TD} data-sort={vb ?? ""}>{pct(vb)}{sb.n?.[j] != null && <small style={N}>{sb.n[j]!.toLocaleString()}走</small>}</td>
                  <td style={TD} data-sort={typeof va === "number" && typeof vb === "number" ? va - vb : ""}>{typeof va === "number" && typeof vb === "number" ? `${va >= vb ? "+" : ""}${fmt === "num" ? (va - vb).toFixed(2) : `${((va - vb) * 100).toFixed(1)}pt`}` : "–"}</td>
                  <td style={{ ...TD, whiteSpace: "nowrap", fontWeight: sig ? 700 : undefined, color: sig ? "var(--text)" : "var(--text-dim)" }} data-sort={p ?? ""}>{formatP(p)}{sig ? " ＊" : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/** 比較の無いグラフの数値（各点の値・標本数・95% 信頼区間） */
function SeriesTable({ block }: { block: ColumnChartBlock }) {
  const rows = [
    ...block.series.filter((s) => s.n).map((s) => ({ s, fmt: block.format ?? "pct" })),
    ...(block.y2?.series ?? []).filter((s) => s.n).map((s) => ({ s, fmt: block.y2!.format ?? "num" })),
  ];
  const TD = { padding: "5px 4px", borderTop: "1px solid var(--border)", textAlign: "right" as const, fontVariantNumeric: "tabular-nums" as const };
  return (
    <details style={{ marginTop: 8, fontSize: 12 }}>
      <summary style={{ cursor: "pointer", color: "var(--text-dim)", minHeight: 32, display: "flex", alignItems: "center" }}>数値（標本数・95%信頼区間）</summary>
      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 4 }}>
        <thead>
          <tr style={{ color: "var(--text-dim)" }}>
            <th style={{ ...TD, textAlign: "left", borderTop: "none" }}>{block.xLabel}</th>
            {rows.length > 1 && <th style={{ ...TD, textAlign: "left", borderTop: "none" }}>系列</th>}
            <th style={{ ...TD, borderTop: "none" }}>{block.y2 ? "値" : block.yLabel}</th>
            <th style={{ ...TD, borderTop: "none" }}>標本数</th>
            <th style={{ ...TD, borderTop: "none" }}>95%信頼区間</th>
          </tr>
        </thead>
        <tbody>
          {block.xs.flatMap((lab, j) => rows.map(({ s, fmt }) => (
            <tr key={`${j}-${s.name}`}>
              <td style={{ ...TD, textAlign: "left", whiteSpace: "nowrap" }}>{lab.replace(/\n/g, " ")}</td>
              {rows.length > 1 && <td style={{ ...TD, textAlign: "left" }}>{s.name}</td>}
              <td style={TD} data-sort={s.values[j] ?? ""}>{formatValue(s.values[j], fmt)}</td>
              <td style={TD} data-sort={s.n?.[j] ?? ""}>{s.n?.[j] != null ? s.n[j]!.toLocaleString() : "–"}</td>
              <td style={{ ...TD, whiteSpace: "nowrap" }}>{s.ci?.[j] ? `${formatValue(s.ci[j]![0], fmt)}〜${formatValue(s.ci[j]![1], fmt)}` : "–"}</td>
            </tr>
          )))}
        </tbody>
      </table>
    </details>
  );
}
