"use client";

import { useState } from "react";
import { bracketChipStyle } from "@/lib/bracket";
import type { AnalysisEntry, RunningMapAxis, RunningMapExtra, RunningMapFigure, RunningMapHorse } from "@/lib/raceAnalysis";

/** グループの色（A〜E） */
const GROUP_COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff", "#f778ba"];
const groupColor = (g: string) => GROUP_COLORS[(g.charCodeAt(0) - 65) % GROUP_COLORS.length] ?? "#8b949e";

const W = 360, PAD = 30;
/** コース適性の色（高いほど濃い）。グループの色と紛れないよう金色 */
const APT_RGB = "227,179,65";
// 広い画面では左に図、右に詳細とグループを並べる（横の余白を使い、スクロールなしで一面に収める）
const CSS = `.rm-wrap{display:grid;gap:10px;margin-top:8px}
@media (min-width:900px){.rm-wrap{grid-template-columns:minmax(0,520px) minmax(0,1fr);align-items:start}}`;

function label(h: RunningMapHorse): string {
  return h.horse_number ? String(h.horse_number) : (h.horse_name ?? "?").slice(0, 2);
}

export function RunningMap({ figure, entries }: { figure: RunningMapFigure; entries: AnalysisEntry[] }) {
  const [sel, setSel] = useState<string | null>(null);
  if (figure.status !== "ok") {
    return <p style={{ fontSize: 13, color: "var(--text-dim)", marginTop: 8 }}>走法データのある馬が少なく、図を作れません。</p>;
  }
  const hs = figure.horses;
  const ex = figure.extent;
  const xs = [...(ex ? ex.x : [-2, 2]), ...hs.map((h) => h.x)], ys = [...(ex ? ex.y : [-3, 3]), ...hs.map((h) => h.y)];
  const x0 = Math.min(...xs) - 0.25, x1 = Math.max(...xs) + 0.25, y0 = Math.min(...ys) - 0.25, y1 = Math.max(...ys) + 0.25;
  const scale = (W - 2 * PAD) / (x1 - x0);
  const H = (y1 - y0) * scale + 2 * PAD;
  const px = (x: number) => PAD + (x - x0) * scale;
  const py = (y: number) => H - PAD - (y - y0) * scale;
  const apt = figure.aptitude;
  const cells = [] as { k: number; x: number; y: number; o: number }[];
  if (apt && ex) {
    const n = ex.n, dx = (ex.x[1] - ex.x[0]) / (n - 1), dy = (ex.y[1] - ex.y[0]) / (n - 1);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const k = i * n + j, sg = apt.sig[k], lf = apt.lift[k];
      if (sg === "0" || !(lf > 1)) continue;
      const o = Math.min(1, (lf - 1) / 0.5) * (sg === "2" ? 0.62 : 0.2);
      cells.push({ k, x: ex.x[0] + i * dx, y: ex.y[0] + j * dy, o });
    }
  }
  const cw = ex ? ((ex.x[1] - ex.x[0]) / (ex.n - 1)) * scale : 0, ch = ex ? ((ex.y[1] - ex.y[0]) / (ex.n - 1)) * scale : 0;
  const bracketOf = new Map(entries.map((e) => [e.key, e.bracket_number]));
  const picked = hs.find((h) => (h.key ?? h.horse_id) === sel) ?? null;
  // 選んだ馬と、8 方面で本当に近い馬（地図の近さは 2 次元に落とした分だけずれるので、線で結んで補う）
  const near = new Map((picked?.similar ?? []).map((s, i) => [s.horse_id, i]));
  const byHorseId = new Map(hs.map((h) => [h.horse_id, h]));
  const explained = figure.fidelity?.explained;

  return (
    <div>
      <style>{CSS}</style>
      <div className="rm-wrap">
      <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", maxWidth: 520, maxHeight: "82vh", margin: "0 auto", touchAction: "manipulation" }} role="img" aria-label="走法マップ（近い馬ほど走り方が似ている。背景の濃い色はこのコースで好走しやすいゾーン）">
        {cells.map((c) => (
          <rect key={c.k} x={px(c.x) - cw / 2} y={py(c.y) - ch / 2} width={cw + 0.5} height={ch + 0.5} fill={`rgb(${APT_RGB})`} fillOpacity={c.o} shapeRendering="crispEdges" />
        ))}
        <line x1={PAD / 2} x2={W - PAD / 2} y1={py(0)} y2={py(0)} stroke="var(--border)" />
        <line y1={PAD / 2} y2={H - PAD / 2} x1={px(0)} x2={px(0)} stroke="var(--border)" />
        {picked && (picked.similar ?? []).map((s, i) => {
          const o = byHorseId.get(s.horse_id);
          if (!o) return null;
          return <line key={`sim-${s.horse_id}`} x1={px(picked.x)} y1={py(picked.y)} x2={px(o.x)} y2={py(o.y)} stroke="#fff" strokeOpacity={0.85 - i * 0.2} strokeWidth={2.2 - i * 0.5} strokeDasharray="5 3" />;
        })}
        {hs.map((h) => {
          const id = h.key ?? h.horse_id ?? "";
          const c = groupColor(h.group);
          const on = id === sel;
          const dim = picked != null && !on && !near.has(h.horse_id);
          return (
            <g key={id} onClick={() => setSel(on ? null : id)} style={{ cursor: "pointer" }} opacity={dim ? 0.3 : 1}>
              <circle cx={px(h.x)} cy={py(h.y)} r={on ? 14 : 11.5} fill={h.estimated ? "var(--surface)" : c} stroke={on ? "#fff" : h.estimated ? c : "rgba(0,0,0,.35)"}
                strokeWidth={on ? 2.5 : h.estimated ? 1.8 : 1} strokeDasharray={h.estimated ? "3 2" : undefined} />
              <text x={px(h.x)} y={py(h.y) + 4} textAnchor="middle" fontSize={h.horse_number ? 11.5 : 9} fontWeight={800} fill={h.estimated ? c : "#0d1117"} pointerEvents="none">{label(h)}</text>
              <title>{`${h.horse_name ?? ""}（${h.group}）`}</title>
            </g>
          );
        })}
      </svg>
      {apt ? <AptLegend apt={apt} /> : <p style={{ fontSize: 11, color: "var(--text-dim)", margin: "4px 0 0" }}>このコースは出走数が少なく、好走しやすさの色は付けていません。</p>}
      </div>

      <div>
      {picked ? <HorseDetail h={picked} axes={figure.axes} extraAxes={figure.extra_axes ?? []} bracket={bracketOf.get(picked.key ?? "") ?? null} onClose={() => setSel(null)}
          onPick={(hid) => { const o = byHorseId.get(hid); if (o) setSel(o.key ?? o.horse_id ?? null); }} bracketOf={(hid) => bracketOf.get(byHorseId.get(hid)?.key ?? "") ?? null} />
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
        点線の白抜きは、実測ラップのある過去の出走が無く父の産駒の平均から推定した馬。
        {typeof explained === "number" ? ` この図は、8 方面での馬どうしの違いの約 ${Math.round(explained * 100)}% を表しています。` : ""}
        近い位置にいる馬ほど走り方が似ています（2 次元に落とした分、近さはおおよそ）。馬を押すと、8 方面で本当に近い馬を線で結びます。座標はどのレースでも共通です。
      </p>
    </div>
  );
}

function AptLegend({ apt }: { apt: NonNullable<RunningMapFigure["aptitude"]> }) {
  return (
    <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-dim)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span>{apt.course}：好走しやすさ</span>
        <span>低</span>
        <span aria-hidden style={{ width: 90, height: 8, borderRadius: 4, background: `linear-gradient(90deg, rgba(${APT_RGB},0.05), rgba(${APT_RGB},0.65))`, border: "1px solid var(--border)" }} />
        <span>高</span>
      </div>
      <div style={{ marginTop: 2 }}>
        色が濃いほど、2020 年以降の同じコース（{apt.n_races.toLocaleString()} レース・{apt.n_runs.toLocaleString()} 頭）で、そのゾーンの走り方の馬が頭数から見込まれる以上に 3 着以内に入っています。
        濃い色は偶然では説明しにくいゾーン（多重検定を補正）、薄い色は傾向のみ。色の無い所は平均以下か、データ不足です。測った期間のレースでは結果を含む値です。
      </div>
    </div>
  );
}

function extraText(e: RunningMapExtra | undefined, key: string): string {
  if (!e || e.value == null) {
    if (key === "slope") return `不明（中山・阪神の実測が ${e?.runs ?? 0} 走）`;
    return `不明（道悪の走 ${e?.n_power ?? 0} 走・通常の走 ${e?.n_normal ?? 0} 走。各 3 走以上で判定）`;
  }
  const v = Math.round(e.value);
  return key === "slope" ? `${v}（実測 ${e.runs ?? 0} 走）` : `${v}${e.low ? "（走数が少なく低信頼）" : ""}`;
}

function HorseDetail({ h, axes, extraAxes, bracket, onClose, onPick, bracketOf }: {
  h: RunningMapHorse; axes: RunningMapAxis[]; extraAxes: NonNullable<RunningMapFigure["extra_axes"]>; bracket: number | null; onClose: () => void;
  onPick: (horseId: string | null) => void; bracketOf: (horseId: string | null) => number | null;
}) {
  const c = groupColor(h.group);
  const fromPrior = new Set(h.from_prior ?? []);
  return (
    <div style={{ marginTop: 8, padding: "10px 12px", borderRadius: 10, border: `1px solid ${c}`, background: `${c}14` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {h.horse_number ? <span style={bracketChipStyle(bracket, 22)}>{h.horse_number}</span> : null}
        <b style={{ fontSize: 14, color: "#fff", flex: "1 1 auto", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{h.horse_name}</b>
        <span style={{ fontSize: 12, color: c, fontWeight: 700 }}>グループ {h.group}</span>
        <button type="button" onClick={onClose} aria-label="閉じる" style={{ minWidth: 32, minHeight: 32, border: "none", background: "transparent", color: "var(--text-dim)", fontSize: 18, cursor: "pointer" }}>×</button>
      </div>
      <div style={{ fontSize: 11, color: "var(--text-dim)", margin: "2px 0 6px" }}>
        {h.estimated ? "実測ラップのある過去の出走が無いため、父の産駒の平均から推定" : `平地 ${h.n_runs} 走のうち実測ラップ ${h.n_lap_runs ?? 0} 走（直近 10 走まで使用）・前走 ${h.last_run_date ?? "–"}${h.own_weight < 0.5 ? "・実測が少ないため父の産駒の平均に寄せています" : ""}`}
      </div>
      {(h.similar ?? []).length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 6px", margin: "0 0 8px", fontSize: 12 }}>
          <span style={{ color: "var(--text-dim)" }}>走法が近い馬</span>
          {(h.similar ?? []).map((s) => (
            <button key={s.horse_id ?? s.horse_name ?? ""} type="button" onClick={() => onPick(s.horse_id)}
              style={{ display: "inline-flex", alignItems: "center", gap: 4, minHeight: 30, padding: "0 8px 0 3px", borderRadius: 999, cursor: "pointer", fontSize: 12,
                border: "1px solid var(--border)", background: "transparent", color: "var(--text)" }}>
              {s.horse_number ? <span style={{ ...bracketChipStyle(bracketOf(s.horse_id), 20) }}>{s.horse_number}</span> : null}
              {s.horse_name}
            </button>
          ))}
        </div>
      )}
      <div style={{ display: "grid", gap: 5 }}>
        {axes.map((a) => {
          const p = h.percentiles[a.key];
          return (
            <div key={a.key} style={{ display: "grid", gridTemplateColumns: "8.5em 1fr 3em", alignItems: "center", gap: 6, fontSize: 11.5 }}>
              <span style={{ color: "var(--text-dim)" }} title={`実測の走 ${h.axis_runs?.[a.key] ?? 0} 走`}>{a.label}</span>
              <span style={{ position: "relative", height: 8, borderRadius: 4, background: "var(--border)" }} title={`${a.low} ← → ${a.high}`}>
                {typeof p === "number" && <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${p}%`, borderRadius: 4, background: c, opacity: fromPrior.has(a.key) ? 0.35 : 1 }} />}
                <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--text-dim)", opacity: 0.6 }} />
              </span>
              <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{typeof p === "number" ? Math.round(p) : "–"}{fromPrior.has(a.key) ? "*" : ""}</span>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-dim)", marginTop: 4 }}>数字は全馬の中での位置（100 に近いほど「末脚が速い」「前に行く」など、各方面の強い側）。真ん中の線が平均。* は実測ラップの走が無く、父の産駒の平均から推定（薄い色）。</div>
      {extraAxes.length > 0 && (
        <div style={{ display: "grid", gap: 3, marginTop: 8, paddingTop: 6, borderTop: "1px solid var(--border)", fontSize: 11.5 }}>
          {extraAxes.map((x) => (
            <div key={x.key} style={{ display: "grid", gridTemplateColumns: "8.5em 1fr", gap: 6 }}>
              <span style={{ color: "var(--text-dim)" }} title={`高い: ${x.high} / 低い: ${x.low}`}>{x.label}</span>
              <span>{extraText(h.extras?.[x.key], x.key)}</span>
            </div>
          ))}
          <div style={{ fontSize: 10.5, color: "var(--text-dim)" }}>地図の位置には使わない補助の軸。50 が平均で、高いほど「{extraAxes.map((x) => x.high).join("」「")}」。不明は判定できる走が足りない場合です。</div>
        </div>
      )}
    </div>
  );
}
