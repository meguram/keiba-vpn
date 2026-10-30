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

/** 軸の値で塗るときの色（低い＝青 ⇆ 高い＝橙、平均＝白、不明＝灰） */
const AXIS_LOW = [88, 166, 255], AXIS_MID = [230, 237, 243], AXIS_HIGH = [240, 136, 62];
const AXIS_NA = "#6e7681";
function axisColor(p: number | null | undefined): string {
  if (typeof p !== "number") return AXIS_NA;
  const t = Math.max(-1, Math.min(1, (p - 50) / 50));
  const to = t < 0 ? AXIS_LOW : AXIS_HIGH, a = Math.abs(t);
  return `rgb(${AXIS_MID.map((m, i) => Math.round(m + (to[i] - m) * a)).join(",")})`;
}
/** ⇆ の線: これより短い（地図の位置でほとんど表せない）軸は見出しを付けず、地図の下に書く */
const GUIDE_MIN = 0.3;

function label(h: RunningMapHorse): string {
  return h.horse_number ? String(h.horse_number) : (h.horse_name ?? "?").slice(0, 2);
}

export function RunningMap({ figure, entries }: { figure: RunningMapFigure; entries: AnalysisEntry[] }) {
  const [sel, setSel] = useState<string | null>(null);
  // 丸の色: "group" か軸のキー。最初は、地図の位置で最も表せない軸（位置×色で全軸を見られるように）
  const [colorKey, setColorKey] = useState<string>(() => {
    const worst = [...figure.axes].sort((a, b) => (a.r2 ?? 0) - (b.r2 ?? 0))[0];
    return worst?.key ?? "group";
  });
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
  const colorAxis = figure.axes.find((a) => a.key === colorKey) ?? null;
  const fillOf = (h: RunningMapHorse) => (colorAxis ? axisColor(h.percentiles[colorAxis.key]) : groupColor(h.group));

  // 8 本の ⇆: 中心を通る線。長さは「地図の位置でその軸をどれだけ表せるか」（r2 の平方根）に比例
  const R = Math.min(-x0, x1, -y0, y1) * 0.9;
  const guides = figure.axes.filter((a) => a.arrow).map((a) => {
    const n = Math.hypot(a.arrow![0], a.arrow![1]) || 1;
    const L = R * Math.sqrt(Math.max(0, a.r2 ?? 0));
    return { a, ux: a.arrow![0] / n, uy: a.arrow![1] / n, L };
  });
  const hidden = guides.filter((g) => g.L < R * GUIDE_MIN);
  // 見出しは図の外周に置く（馬の丸と重ならない）。向きがほぼ同じ軸は「末脚の速さ・持続力」のように 1 つにまとめる。
  // 色分けしている軸だけは両端（高い＝橙・低い＝青）に言葉を出す。位置でほとんど表せない軸は図の上には出さない（下の注記と凡例）
  const ringLabels = (() => {
    const items: { ang: number; text: string; tone: "dim" | "high" | "low" }[] = [];
    for (const g of guides) {
      if (g.L < R * GUIDE_MIN) continue;
      const ang = Math.atan2(g.uy, g.ux);
      if (g.a.key === colorKey) {
        items.push({ ang, text: g.a.high, tone: "high" }, { ang: ang + Math.PI, text: g.a.low, tone: "low" });
      } else items.push({ ang, text: g.a.label, tone: "dim" });
    }
    const norm = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
    items.sort((a, b) => norm(a.ang) - norm(b.ang));
    const merged: typeof items = [];
    for (const it of items) {
      const last = merged[merged.length - 1];
      if (last && last.tone === "dim" && it.tone === "dim" && Math.abs(norm(it.ang - last.ang)) < (16 * Math.PI) / 180) {
        last.text = `${last.text}・${it.text.replace(/^末脚の/, "")}`;
        last.ang = (last.ang + it.ang) / 2;
      } else merged.push({ ...it });
    }
    return merged.map((m) => {
      const cx = px(0), cy = py(0);
      const rx = Math.min(cx, W - cx) - 6, ry = Math.min(cy, H - cy) - 8;   // 図の枠に内接する楕円の上
      const x = cx + Math.cos(m.ang) * rx, y = cy - Math.sin(m.ang) * ry;
      const anchor: "middle" | "end" | "start" = Math.abs(Math.cos(m.ang)) < 0.35 ? "middle" : Math.cos(m.ang) > 0 ? "end" : "start";
      return { ...m, x, y: y + 3, anchor };
    });
  })();
  // 重なった見出しは、色分けしている軸（橙・青）を動かさず、他を図の中心から離れる向きへ 12px ずつずらす
  const textW = (t: string, fs: number) => t.length * fs;
  const spanX = (l: (typeof ringLabels)[number], fs: number) => {
    const w = textW(l.text, fs);
    return l.anchor === "start" ? [l.x, l.x + w] : l.anchor === "end" ? [l.x - w, l.x] : [l.x - w / 2, l.x + w / 2];
  };
  for (const l of [...ringLabels].sort((a, b) => (a.tone === "dim" ? 1 : 0) - (b.tone === "dim" ? 1 : 0))) {
    if (l.tone !== "dim") continue;
    for (let k = 0; k < 4; k++) {
      const [a0, a1] = spanX(l, 9.5);
      const hit = ringLabels.some((o) => o !== l && Math.abs(o.y - l.y) < 12 && (() => { const [b0, b1] = spanX(o, o.tone === "dim" ? 9.5 : 10.5); return a0 < b1 && b0 < a1; })());
      if (!hit) break;
      const out = l.y + (l.y < py(0) ? -12 : 12);
      l.y = out >= 10 && out <= H - 4 ? out : l.y + (l.y < py(0) ? 12 : -12);   // 図の端に当たったら内側へ
    }
  }

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
        {guides.map((g) => {
          const on = g.a.key === colorKey;
          const hx = px(g.ux * g.L), hy = py(g.uy * g.L), lx_ = px(-g.ux * g.L), ly_ = py(-g.uy * g.L);
          return (
            <g key={`ax-${g.a.key}`} onClick={() => setColorKey(on ? "group" : g.a.key)} style={{ cursor: "pointer" }}>
              <title>{`${g.a.label}　低い: ${g.a.low} ⇆ 高い: ${g.a.high}（押すとこの軸で丸を色分け）`}</title>
              <line x1={lx_} y1={ly_} x2={hx} y2={hy} stroke={on ? "#e6edf3" : "#8b949e"} strokeOpacity={on ? 0.9 : 0.45} strokeWidth={on ? 1.8 : 1} />
              <line x1={lx_} y1={ly_} x2={hx} y2={hy} stroke="transparent" strokeWidth={12} />
            </g>
          );
        })}
        {ringLabels.map((l, i) => (
          <text key={`rl-${i}`} x={l.x} y={l.y} textAnchor={l.anchor} fontSize={l.tone === "dim" ? 9.5 : 10.5} fontWeight={l.tone === "dim" ? 400 : 700}
            fill={l.tone === "high" ? "rgb(240,136,62)" : l.tone === "low" ? "rgb(88,166,255)" : "var(--text-dim)"} pointerEvents="none">{l.text}</text>
        ))}
        {picked && (picked.similar ?? []).map((s, i) => {
          const o = byHorseId.get(s.horse_id);
          if (!o) return null;
          return <line key={`sim-${s.horse_id}`} x1={px(picked.x)} y1={py(picked.y)} x2={px(o.x)} y2={py(o.y)} stroke="#fff" strokeOpacity={0.85 - i * 0.2} strokeWidth={2.2 - i * 0.5} strokeDasharray="5 3" />;
        })}
        {hs.map((h) => {
          const id = h.key ?? h.horse_id ?? "";
          const c = fillOf(h);
          const on = id === sel;
          const dim = picked != null && !on && !near.has(h.horse_id);
          return (
            <g key={id} onClick={() => setSel(on ? null : id)} style={{ cursor: "pointer" }} opacity={dim ? 0.3 : 1}>
              <circle cx={px(h.x)} cy={py(h.y)} r={on ? 14 : 11.5} fill={h.estimated ? "var(--surface)" : c} stroke={on ? "#fff" : h.estimated ? c : "rgba(0,0,0,.35)"}
                strokeWidth={on ? 2.5 : h.estimated ? 1.8 : 1} strokeDasharray={h.estimated ? "3 2" : undefined} />
              <text x={px(h.x)} y={py(h.y) + 4} textAnchor="middle" fontSize={h.horse_number ? 11.5 : 9} fontWeight={800} fill={h.estimated ? c : "#0d1117"} pointerEvents="none">{label(h)}</text>
              <title>{`${h.horse_name ?? ""}（グループ ${h.group}${colorAxis ? `・${colorAxis.label} ${typeof h.percentiles[colorAxis.key] === "number" ? Math.round(h.percentiles[colorAxis.key] as number) : "不明"}` : ""}）`}</title>
            </g>
          );
        })}
      </svg>
      <ColorControl axes={figure.axes} colorKey={colorKey} onChange={setColorKey} />
      {hidden.length > 0 && (
        <p style={{ fontSize: 11, color: "var(--text-dim)", margin: "2px 0 0" }}>
          地図の位置ではほとんど表せない軸: {hidden.map((g) => g.a.label).join("・")}（丸の色で見られます）
        </p>
      )}
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
        近い位置にいる馬ほど走り方が似ています（2 次元に落とした分、近さはおおよそ）。馬を押すと、8 方面で本当に近い馬を線で結びます。
        灰色の線は 8 方面それぞれの向き（長いほど地図の位置でよく表せる軸）で、押すとその軸で丸を色分けします。座標はどのレースでも共通です。
      </p>
    </div>
  );
}

function ColorControl({ axes, colorKey, onChange }: { axes: RunningMapAxis[]; colorKey: string; onChange: (k: string) => void }) {
  const a = axes.find((x) => x.key === colorKey) ?? null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 8px", marginTop: 6, fontSize: 11.5 }}>
      <label style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--text-dim)" }}>
        丸の色
        <select value={colorKey} onChange={(e) => onChange(e.target.value)}
          style={{ minHeight: 30, fontSize: 12, background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 6, padding: "0 6px" }}>
          <option value="group">グループ</option>
          {axes.map((x) => (
            <option key={x.key} value={x.key}>{x.label}{typeof x.r2 === "number" ? `（位置で表せる ${Math.round(x.r2 * 100)}%）` : ""}</option>
          ))}
        </select>
      </label>
      {a && (
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--text-dim)" }}>
          <span style={{ color: "rgb(88,166,255)" }}>{a.low}</span>
          <span aria-hidden style={{ width: 70, height: 8, borderRadius: 4, border: "1px solid var(--border)",
            background: `linear-gradient(90deg, ${axisColor(0)}, ${axisColor(50)}, ${axisColor(100)})` }} />
          <span style={{ color: "rgb(240,136,62)" }}>{a.high}</span>
          <span><span aria-hidden style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4, background: AXIS_NA, marginRight: 3 }} />不明</span>
        </span>
      )}
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
