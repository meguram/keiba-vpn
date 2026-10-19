"use client";

import { useMemo, useState } from "react";
import type { HorseGrowth, GrowthPoint } from "@/lib/entryCard";

/* 馬の成長曲線（過去走ごとのめぐ指数の推移）。外部ライブラリなしの SVG。
   横軸=レース日（時間の目盛り）、縦軸=めぐ指数（時期ごとの水準差をそろえた値）。丸の中の数字は着順。 */

const W = 360, H = 190, PL = 34, PR = 40, PT = 14, PB = 28;

const dayNum = (ymd: string) => Date.UTC(Number(ymd.slice(0, 4)), Number(ymd.slice(4, 6)) - 1, Number(ymd.slice(6, 8))) / 86400000;
const fmtDate = (ymd: string) => `${ymd.slice(0, 4)}/${Number(ymd.slice(4, 6))}/${Number(ymd.slice(6, 8))}`;
const fin = (f: GrowthPoint["finish"]) => { const n = Number(f); return Number.isFinite(n) && n > 0 ? n : null; };

function tone(f: number | null) {
  if (f == null) return { bg: "#475569", fg: "#e2e8f0", label: "着順不明" };
  if (f <= 3) return { bg: "#fbbf24", fg: "#1f2937", label: "3着以内" };
  if (f <= 5) return { bg: "#60a5fa", fg: "#0b1220", label: "4〜5着" };
  return { bg: "#64748b", fg: "#f1f5f9", label: "6着以下" };
}

const TREND_STYLE: Record<string, { arrow: string; color: string }> = {
  "上昇中": { arrow: "↗", color: "#4ade80" },
  "横ばい": { arrow: "→", color: "#cbd5e1" },
  "下降中": { arrow: "↘", color: "#fb923c" },
};

function ageText(m: number | null | undefined) {
  if (m == null) return null;
  const y = Math.floor(m / 12), mo = m % 12;
  return `${y}歳${mo}か月`;
}

export function GrowthCurve({ g }: { g: HorseGrowth | null | undefined }) {
  const pts = useMemo<GrowthPoint[]>(() => (g && g.status === "ok" ? g.points : []), [g]);
  const [sel, setSel] = useState<number | null>(null);
  const cur = sel ?? pts.length - 1;

  const geo = useMemo(() => {
    if (pts.length < 2 || !g || g.status !== "ok") return null;
    const xs = pts.map((p) => dayNum(p.date));
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const vals = pts.map((p) => p.value).concat([g.axis.ref]);
    const lo = Math.floor((Math.min(...vals) - 4) / 5) * 5, hi = Math.ceil((Math.max(...vals) + 4) / 5) * 5;
    const X = (d: number) => PL + ((d - x0) / Math.max(1, x1 - x0)) * (W - PL - PR - 8) + 4;
    const Y = (v: number) => PT + (1 - (v - lo) / Math.max(1, hi - lo)) * (H - PT - PB);
    const ticks: { x: number; t: string }[] = [];
    const y0 = Number(pts[0].date.slice(0, 4)), y1 = Number(pts[pts.length - 1].date.slice(0, 4));
    for (let y = y0; y <= y1; y++) {
      const d = dayNum(`${y}0101`);
      if (d >= x0 - 1 && d <= x1 + 1) ticks.push({ x: X(d), t: `${y}` });
    }
    const yt: number[] = [];
    const step = hi - lo > 40 ? 10 : 5;
    for (let v = lo; v <= hi; v += step) yt.push(v);
    return { X, Y, xs, ticks, yt, lo, hi };
  }, [pts, g]);

  if (!g || g.status !== "ok") {
    return <div className="gc-empty">成長曲線はまだ出せません<small>{g?.reason ?? "出走履歴がありません"}</small></div>;
  }
  if (!geo) {
    const p = pts[0];
    return <div className="gc-empty">走数が少なく曲線は出せません<small>{g.axis.name}のある走は {pts.length} 走のみ（{fmtDate(p.date)} {p.race_name ?? ""}・{p.value.toFixed(1)}）</small></div>;
  }

  const tr = g.trend ? TREND_STYLE[g.trend.label] : null;
  const p = pts[cur];
  const line = pts.map((q, i) => `${i ? "L" : "M"}${geo.X(geo.xs[i]).toFixed(1)},${geo.Y(q.value).toFixed(1)}`).join(" ");
  const bestY = geo.Y(g.best.value);

  return (
    <div className="gc">
      <div className="gc-head">
        {g.trend && tr && (
          <span className="gc-trend" style={{ color: tr.color, borderColor: tr.color }} title={g.trend.basis}>
            <b>{tr.arrow} {g.trend.label}</b><small>{g.trend.delta > 0 ? "+" : ""}{g.trend.delta}</small>
          </span>
        )}
        <span className="gc-best">自己ベスト <b>{g.best.value.toFixed(1)}</b><small>{fmtDate(g.best.date)}</small></span>
        <span className="gc-n">{g.n}走</span>
      </div>
      <svg className="gc-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`成長曲線。${g.axis.name}の推移 ${pts.map((q) => `${fmtDate(q.date)} ${q.value.toFixed(1)}`).join("、")}`}>
        {geo.yt.map((v) => (
          <g key={v}>
            <line x1={PL} x2={W - PR} y1={geo.Y(v)} y2={geo.Y(v)} stroke="var(--border)" strokeWidth={0.6} />
            <text x={PL - 5} y={geo.Y(v) + 3.5} fontSize={10} textAnchor="end" fill="var(--text-dim)">{v}</text>
          </g>
        ))}
        {geo.ticks.map((t) => (
          <g key={t.t}>
            <line x1={t.x} x2={t.x} y1={PT} y2={H - PB} stroke="var(--border)" strokeWidth={0.8} strokeDasharray="2 3" />
            <text x={t.x + 3} y={H - 9} fontSize={10} fill="var(--text-dim)">{t.t}</text>
          </g>
        ))}
        <line x1={PL} x2={W - PR} y1={geo.Y(g.axis.ref)} y2={geo.Y(g.axis.ref)} stroke="#94a3b8" strokeWidth={1} strokeDasharray="5 4" />
        <text x={W - PR + 4} y={geo.Y(g.axis.ref) - 1} fontSize={9} fill="#94a3b8">全体の</text>
        <text x={W - PR + 4} y={geo.Y(g.axis.ref) + 9} fontSize={9} fill="#94a3b8">水準 {g.axis.ref}</text>
        <line x1={PL} x2={W - PR} y1={bestY} y2={bestY} stroke="#fbbf24" strokeWidth={0.8} strokeDasharray="1 3" opacity={0.8} />
        <path d={line} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {pts.map((q, i) => {
          const f = fin(q.finish), c = tone(f), on = i === cur;
          const cx = geo.X(geo.xs[i]), cy = geo.Y(q.value);
          return (
            <g key={q.race_id + i} role="button" tabIndex={0} aria-label={`${fmtDate(q.date)} ${q.race_name ?? ""} ${f ?? "?"}着 ${g.axis.name} ${q.value.toFixed(1)}`}
              onClick={() => setSel(i)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSel(i); } }} style={{ cursor: "pointer", outline: "none" }}>
              <circle cx={cx} cy={cy} r={16} fill="transparent" />
              <circle cx={cx} cy={cy} r={on ? 10 : 8} fill={c.bg} stroke={on ? "#fff" : "var(--surface)"} strokeWidth={on ? 2 : 1.2} />
              <text x={cx} y={cy + 3.4} fontSize={on ? 10.5 : 9.5} fontWeight={800} textAnchor="middle" fill={c.fg}>{f ?? "?"}</text>
            </g>
          );
        })}
      </svg>
      <div className="gc-sel" aria-live="polite">
        <b>{fmtDate(p.date)}</b>
        <span>{[p.venue, p.race_name].filter(Boolean).join(" ")}</span>
        <span>{[p.surface, p.distance ? `${p.distance}m` : null, p.field ? `${p.field}頭` : null].filter(Boolean).join(" ")}</span>
        <span>{fin(p.finish) ?? "?"}着</span>
        <strong>{g.axis.name} {p.value.toFixed(1)}</strong>
        <small>元の値 {p.raw.toFixed(1)}{ageText(p.age_months) ? ` ・ ${ageText(p.age_months)}` : ""}</small>
      </div>
      <p className="gc-note">縦軸: {g.axis.name}（{g.axis.note}）。丸の中の数字は着順で、<i style={{ background: "#fbbf24" }} />3着以内 <i style={{ background: "#60a5fa" }} />4〜5着 <i style={{ background: "#64748b" }} />6着以下。点をタップすると詳細。{g.excluded > 0 ? ` 基準外で除いた走 ${g.excluded}。` : ""}</p>
    </div>
  );
}

export const GROWTH_CSS = `
.gc{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:10px 10px 8px}
.gc-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.gc-trend{display:inline-flex;align-items:baseline;gap:6px;border:1px solid;border-radius:99px;padding:2px 10px;font-size:13px}
.gc-trend small{font-size:11px;opacity:.85}
.gc-best{font-size:12px;color:var(--text-dim);display:inline-flex;align-items:baseline;gap:5px}
.gc-best b{color:#fbbf24;font-size:14px}
.gc-n{margin-left:auto;font-size:12px;color:var(--text-dim)}
.gc-svg{display:block;width:100%;height:auto;max-height:260px}
.gc-svg g[role=button]:focus-visible circle:nth-child(2){stroke:#fff;stroke-width:2.5}
.gc-sel{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;margin-top:6px;padding:8px 10px;border-radius:10px;background:var(--surface);border:1px solid var(--border);font-size:13px}
.gc-sel strong{color:#f0f6fc}
.gc-sel small{color:var(--text-dim);font-size:11.5px}
.gc-note{margin:6px 2px 0;font-size:11.5px;line-height:1.6;color:var(--text-dim)}
.gc-note i{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 3px 0 6px;vertical-align:-1px}
.gc-empty{display:flex;flex-direction:column;gap:3px;padding:14px 12px;border:1px dashed var(--border);border-radius:12px;color:var(--text-dim);font-size:13px}
.gc-empty small{font-size:11.5px}
`;
