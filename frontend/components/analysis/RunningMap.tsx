"use client";

import { useState } from "react";
import { bracketChipStyle } from "@/lib/bracket";
import type { AnalysisEntry, RunningMapAxis, RunningMapFigure, RunningMapHorse } from "@/lib/raceAnalysis";

/** グループの色（A〜E） */
const GROUP_COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff", "#f778ba"];
const groupColor = (g: string) => GROUP_COLORS[(g.charCodeAt(0) - 65) % GROUP_COLORS.length] ?? "#8b949e";

const S = 360, PAD = 34;
// 広い画面では左に図、右に詳細とグループを並べる（横の余白を使い、スクロールなしで一面に収める）
const CSS = `.rm-wrap{display:grid;gap:10px;margin-top:8px}
@media (min-width:900px){.rm-wrap{grid-template-columns:minmax(0,520px) minmax(0,1fr);align-items:start}}`;

/** 矢印: TS と GC はほぼ同じ向きなので「末脚・加速」として 1 本にまとめる */
function arrows(axes: RunningMapAxis[]): { label: string; v: [number, number]; r2: number }[] {
  const ts = axes.find((a) => a.key === "ts"), gc = axes.find((a) => a.key === "gc");
  const out: { label: string; v: [number, number]; r2: number }[] = [];
  if (ts?.arrow && gc?.arrow) out.push({ label: "末脚・加速", v: [ts.arrow[0] + gc.arrow[0], ts.arrow[1] + gc.arrow[1]], r2: Math.max(ts.r2 ?? 0, gc.r2 ?? 0) });
  for (const a of axes) if (a.arrow && a.key !== "ts" && a.key !== "gc") out.push({ label: a.label, v: a.arrow, r2: a.r2 ?? 0 });
  return out;
}

function label(h: RunningMapHorse): string {
  return h.horse_number ? String(h.horse_number) : (h.horse_name ?? "?").slice(0, 2);
}

export function RunningMap({ figure, entries }: { figure: RunningMapFigure; entries: AnalysisEntry[] }) {
  const [sel, setSel] = useState<string | null>(null);
  if (figure.status !== "ok") {
    return <p style={{ fontSize: 13, color: "var(--text-dim)", marginTop: 8 }}>走法データのある馬が少なく、図を作れません。</p>;
  }
  const hs = figure.horses;
  const arr = arrows(figure.axes);
  const ext = Math.max(0.5, ...hs.flatMap((h) => [Math.abs(h.x), Math.abs(h.y)]));
  const scale = (S / 2 - PAD) / ext;
  const amax = Math.max(1e-9, ...arr.map((a) => Math.hypot(a.v[0], a.v[1])));
  const aScale = (ext * 0.85) / amax;
  const px = (x: number) => S / 2 + x * scale;
  const py = (y: number) => S / 2 - y * scale;
  const bracketOf = new Map(entries.map((e) => [e.key, e.bracket_number]));
  const picked = hs.find((h) => (h.key ?? h.horse_id) === sel) ?? null;
  const explained = figure.fidelity?.explained;

  return (
    <div>
      <style>{CSS}</style>
      <div className="rm-wrap">
      <svg viewBox={`0 0 ${S} ${S}`} width="100%" style={{ display: "block", maxWidth: 560, margin: "0 auto", touchAction: "manipulation" }} role="img" aria-label="走法マップ（近い馬ほど走り方が似ている）">
        <line x1={PAD / 2} x2={S - PAD / 2} y1={S / 2} y2={S / 2} stroke="var(--border)" />
        <line y1={PAD / 2} y2={S - PAD / 2} x1={S / 2} x2={S / 2} stroke="var(--border)" />
        <defs>
          <marker id="rm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#8b949e" />
          </marker>
        </defs>
        {arr.map((a) => {
          const tx = px(a.v[0] * aScale), ty = py(a.v[1] * aScale);
          const dx = a.v[0], dy = a.v[1];
          const anchor = Math.abs(dx) < Math.abs(dy) * 0.4 ? "middle" : dx > 0 ? "start" : "end";
          // 見出しが図の外にはみ出さないよう、文字幅の目安（1 文字 10.5）で内側へ寄せる
          const w = a.label.length * 10.5;
          let lx = tx + (anchor === "start" ? 4 : anchor === "end" ? -4 : 0);
          if (anchor === "end") lx = Math.max(lx, 2 + w);
          else if (anchor === "start") lx = Math.min(lx, S - 2 - w);
          else lx = Math.min(Math.max(lx, 2 + w / 2), S - 2 - w / 2);
          return (
            <g key={a.label} opacity={a.r2 >= 0.5 ? 0.9 : 0.4}>
              <line x1={S / 2} y1={S / 2} x2={tx} y2={ty} stroke="#8b949e" strokeWidth={1.4} markerEnd="url(#rm-arrow)" />
              <text x={lx} y={ty + (dy > 0 ? -5 : 13)} textAnchor={anchor} fontSize={10.5} fill="var(--text-dim)">{a.label}</text>
            </g>
          );
        })}
        {hs.map((h) => {
          const id = h.key ?? h.horse_id ?? "";
          const c = groupColor(h.group);
          const on = id === sel;
          return (
            <g key={id} onClick={() => setSel(on ? null : id)} style={{ cursor: "pointer" }}>
              <circle cx={px(h.x)} cy={py(h.y)} r={on ? 14 : 11.5} fill={h.estimated ? "var(--surface)" : c} stroke={on ? "#fff" : h.estimated ? c : "rgba(0,0,0,.35)"}
                strokeWidth={on ? 2.5 : h.estimated ? 1.8 : 1} strokeDasharray={h.estimated ? "3 2" : undefined} />
              <text x={px(h.x)} y={py(h.y) + 4} textAnchor="middle" fontSize={h.horse_number ? 11.5 : 9} fontWeight={800} fill={h.estimated ? c : "#0d1117"} pointerEvents="none">{label(h)}</text>
              <title>{`${h.horse_name ?? ""}（${h.group}）`}</title>
            </g>
          );
        })}
      </svg>

      <div>
      {picked ? <HorseDetail h={picked} axes={figure.axes} bracket={bracketOf.get(picked.key ?? "") ?? null} onClose={() => setSel(null)} />
        : <p style={{ fontSize: 12, color: "var(--text-dim)", margin: "4px 0 0" }}>馬（図の点・グループの名前）を押すと、走り方の内訳が見られます</p>}

      <div style={{ display: "grid", gap: 6, marginTop: 10 }}>
        {figure.groups.map((g) => (
          <div key={g.id} style={{ fontSize: 12, borderLeft: `4px solid ${groupColor(g.id)}`, paddingLeft: 8 }}>
            <div style={{ fontWeight: 700, color: groupColor(g.id) }}>{g.id}：{g.label}</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 6px", marginTop: 3 }}>
              {hs.filter((h) => h.group === g.id).map((h) => {
                const id = h.key ?? h.horse_id ?? "";
                return (
                  <button key={id} type="button" onClick={() => setSel(sel === id ? null : id)}
                    style={{ display: "inline-flex", alignItems: "center", gap: 4, minHeight: 30, padding: "0 8px 0 3px", borderRadius: 999, cursor: "pointer", fontSize: 12,
                      border: `1px solid ${sel === id ? groupColor(g.id) : "var(--border)"}`, background: sel === id ? `${groupColor(g.id)}26` : "transparent", color: "var(--text)" }}>
                    {h.horse_number ? <span style={{ ...bracketChipStyle(bracketOf.get(h.key ?? "") ?? null, 20) }}>{h.horse_number}</span> : null}
                    {h.horse_name}{h.estimated ? <small style={{ color: "var(--text-dim)" }}>（推定）</small> : null}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      </div>
      </div>

      {figure.unmapped.length > 0 && (
        <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 8 }}>
          図に載せていない馬: {figure.unmapped.map((u) => `${u.horse_name ?? "?"}（${u.reason}）`).join("、")}
        </p>
      )}
      <p style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 6 }}>
        点線の白抜きは、過去の出走が無く父の産駒の平均から推定した馬。
        {typeof explained === "number" ? ` この図は、6 方面での馬どうしの違いの約 ${Math.round(explained * 100)}% を表しています。` : ""}
        {figure.rq_source ? "" : " このコース条件のレース質の記録が無いため、「レース質との相性」は全馬同じ扱いです。"}
      </p>
    </div>
  );
}

function HorseDetail({ h, axes, bracket, onClose }: { h: RunningMapHorse; axes: RunningMapAxis[]; bracket: number | null; onClose: () => void }) {
  const c = groupColor(h.group);
  return (
    <div style={{ marginTop: 8, padding: "10px 12px", borderRadius: 10, border: `1px solid ${c}`, background: `${c}14` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {h.horse_number ? <span style={bracketChipStyle(bracket, 22)}>{h.horse_number}</span> : null}
        <b style={{ fontSize: 14, color: "#fff", flex: "1 1 auto", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{h.horse_name}</b>
        <span style={{ fontSize: 12, color: c, fontWeight: 700 }}>グループ {h.group}</span>
        <button type="button" onClick={onClose} aria-label="閉じる" style={{ minWidth: 32, minHeight: 32, border: "none", background: "transparent", color: "var(--text-dim)", fontSize: 18, cursor: "pointer" }}>×</button>
      </div>
      <div style={{ fontSize: 11, color: "var(--text-dim)", margin: "2px 0 6px" }}>
        {h.estimated ? "過去の出走が無いため、父の産駒の平均から推定" : `平地 ${h.n_runs} 走（直近 10 走まで使用）・前走 ${h.last_run_date ?? "–"}${h.own_weight < 0.5 ? "・出走が少ないため父の産駒の平均に寄せています" : ""}`}
      </div>
      <div style={{ display: "grid", gap: 5 }}>
        {axes.map((a) => {
          const p = h.percentiles[a.key];
          return (
            <div key={a.key} style={{ display: "grid", gridTemplateColumns: "8em 1fr 2.6em", alignItems: "center", gap: 6, fontSize: 11.5 }}>
              <span style={{ color: "var(--text-dim)" }}>{a.label}</span>
              <span style={{ position: "relative", height: 8, borderRadius: 4, background: "var(--border)" }} title={`${a.low} ← → ${a.high}`}>
                {typeof p === "number" && <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${p}%`, borderRadius: 4, background: c }} />}
                <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--text-dim)", opacity: 0.6 }} />
              </span>
              <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{typeof p === "number" ? Math.round(p) : "–"}</span>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-dim)", marginTop: 4 }}>数字は全馬の中での位置（100 に近いほど「末脚が速い」「前に行く」など、各方面の強い側）。真ん中の線が平均。</div>
    </div>
  );
}
