"use client";

import { useEffect, useRef, useState } from "react";
import {
  buildBand, buildPath, cumulativeMeters, niceTicks, scenarioColor, tickIndices, yRange,
} from "@/lib/lapForecast";
import type { LapForecast as LapForecastData } from "@/lib/lapForecast";

/* ラップ予想（予測 API の lap_forecast）。要約・折れ線グラフ・シナリオ表。
   表示幅を測って SVG の寸法にそのまま使う（文字を実寸 11px 以上に保ち、390px でも横スクロールを出さない）。 */

const sec1 = (v?: number | null) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(1) : "—");
const pct = (v?: number | null) => (typeof v === "number" && Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—");

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{ flex: "1 1 110px", minWidth: 0, padding: "8px 10px", borderRadius: 8, background: "var(--surface2)", border: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800, color: "#f0f6fc", fontVariantNumeric: "tabular-nums" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{sub}</div>}
    </div>
  );
}

function LegendLine({ color, width = 2, dash, label }: { color: string; width?: number; dash?: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, whiteSpace: "nowrap" }}>
      <svg width="22" height="10" aria-hidden><line x1="1" y1="5" x2="21" y2="5" stroke={color} strokeWidth={width} strokeDasharray={dash} strokeLinecap="round" /></svg>
      {label}
    </span>
  );
}

function Chart({ lf }: { lf: LapForecastData }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(340);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const apply = () => setW(Math.max(260, Math.floor(el.clientWidth)));
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const n = lf.n_laps;
  const scenarios = lf.scenarios ?? [];
  const series: (number | null | undefined)[][] = [lf.expected_laps, lf.band80_lo ?? [], lf.band80_hi ?? [], lf.par ?? [], ...scenarios.map((s) => s.laps)];
  const range = yRange(series);
  if (!range) return null;

  const h = Math.min(300, Math.max(210, Math.round(w * 0.5)));
  const m = { l: 38, r: 12, t: 10, b: 26 };
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const x = (i: number) => m.l + (n <= 1 ? iw / 2 : (i * iw) / (n - 1));
  const y = (v: number) => m.t + ih - ((v - range.min) / (range.max - range.min)) * ih;
  const yt = niceTicks(range.min, range.max, 5);
  const cum = cumulativeMeters(lf.lap_meters ?? []);
  const xt = tickIndices(n, w < 420 ? 5 : 8);
  const band = buildBand(lf.band80_lo ?? [], lf.band80_hi ?? [], x, y);
  const label = `ラップ予想のグラフ。期待ラップ ${lf.expected_laps.map((v) => sec1(v)).join("、")} 秒`;

  return (
    <div ref={ref} style={{ width: "100%", maxWidth: 820 }}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label} style={{ display: "block", maxWidth: "100%" }}>
        {yt.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke="#334155" strokeWidth="1" opacity="0.6" />
            <text x={m.l - 5} y={y(t) + 4} textAnchor="end" fontSize="11" fill="#94a3b8">{t.toFixed(1)}</text>
          </g>
        ))}
        {band && <path d={band} fill="#e5e7eb" opacity="0.16" />}
        {lf.par && <path d={buildPath(lf.par, x, y)} fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="4 3" strokeLinecap="round" />}
        {scenarios.map((s, k) => (
          <path key={s.name + k} d={buildPath(s.laps, x, y)} fill="none" stroke={scenarioColor(s.name, k)} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.9" />
        ))}
        <path d={buildPath(lf.expected_laps, x, y)} fill="none" stroke="#f0f6fc" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        {lf.expected_laps.map((v, i) => (typeof v === "number" && Number.isFinite(v) ? <circle key={i} cx={x(i)} cy={y(v)} r="2.5" fill="#f0f6fc" /> : null))}
        {xt.map((i) => (
          <text key={i} x={x(i)} y={h - m.b + 16} textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"} fontSize="11" fill="#94a3b8">
            {cum[i] != null ? `${cum[i]}m` : `${i + 1}`}
          </text>
        ))}
      </svg>
    </div>
  );
}

export function LapForecast({ lf }: { lf: LapForecastData }) {
  const scenarios = lf.scenarios ?? [];
  const noTen = lf.model === "A_noten";
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <Tile label="起こりやすい傾向" value={lf.pace_label ?? "—"} sub={lf.pace_prob != null ? `確率 ${pct(lf.pace_prob)}` : undefined} />
        <Tile label="期待の前半 3F" value={`${sec1(lf.expected_first3f)}秒`} />
        <Tile label="期待の上がり 3F" value={`${sec1(lf.expected_last3f)}秒`} />
      </div>

      <Chart lf={lf} />

      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px", fontSize: 12, color: "var(--text-dim)" }}>
        <LegendLine color="#f0f6fc" width={3} label="期待ラップ（太線）" />
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5, whiteSpace: "nowrap" }}>
          <span aria-hidden style={{ width: 22, height: 10, background: "rgba(229,231,235,0.25)", borderRadius: 2, display: "inline-block" }} />80%の範囲
        </span>
        {scenarios.map((s, k) => <LegendLine key={s.name + k} color={scenarioColor(s.name, k)} label={`${s.name} ${pct(s.prob)}`} />)}
        {lf.par && lf.par.some((v) => typeof v === "number") && <LegendLine color="#94a3b8" dash="4 3" label="コースの基準（破線）" />}
      </div>

      {scenarios.length > 0 && (
        <div role="table" aria-label="ラップ予想のシナリオ" style={{ display: "grid", gap: 3, fontSize: 12 }}>
          <div role="row" style={{ display: "grid", gridTemplateColumns: "minmax(60px,1.2fr) 1fr 1fr 1fr", gap: 8, padding: "0 8px", color: "var(--text-dim)", fontSize: 11 }}>
            <span role="columnheader">傾向</span><span role="columnheader">確率</span><span role="columnheader">前半 3F</span><span role="columnheader">上がり 3F</span>
          </div>
          {scenarios.map((s, k) => {
            const main = s.name === lf.pace_label;
            return (
              <div key={s.name + k} role="row" style={{ display: "grid", gridTemplateColumns: "minmax(60px,1.2fr) 1fr 1fr 1fr", gap: 8, padding: "5px 8px", borderRadius: 6, background: "var(--surface2)", border: `1px solid ${main ? scenarioColor(s.name, k) : "var(--border)"}`, fontVariantNumeric: "tabular-nums" }}>
                <span role="cell" style={{ display: "inline-flex", alignItems: "center", gap: 6, fontWeight: 700 }}>
                  <span aria-hidden style={{ width: 9, height: 9, borderRadius: 999, background: scenarioColor(s.name, k) }} />{s.name}
                </span>
                <span role="cell">{pct(s.prob)}</span>
                <span role="cell">{sec1(s.first3f)}秒</span>
                <span role="cell">{sec1(s.last3f)}秒</span>
              </div>
            );
          })}
        </div>
      )}

      <p style={{ margin: 0, fontSize: 12, color: "var(--text-dim)", lineHeight: 1.6 }}>
        横軸はスタートからの距離（区間の終わり）、縦軸は区間ごとのラップ（秒）です。
        過去の同条件レースのラップ分布から作った目安です（AI予測ではありません）。帯は80%の範囲で、再計算のたびに最大0.15秒ほど動くことがあります。
        {noTen && " この予想はテン1F（最初の1F）を使わないモデルで作っています。"}
        {" "}最初の区間は200mに満たないことがあります。
      </p>
    </div>
  );
}
