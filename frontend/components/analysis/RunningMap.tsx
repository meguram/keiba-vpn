"use client";

import { useState } from "react";
import { bracketChipStyle } from "@/lib/bracket";
import type { AnalysisEntry, RunningMapAxis, RunningMapBest5, RunningMapBest5Mode, RunningMapDayTrend, RunningMapExtra, RunningMapFigure, RunningMapHorse } from "@/lib/raceAnalysis";

/** グループの色（A〜E） */
const GROUP_COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff", "#f778ba"];
const groupColor = (g: string) => GROUP_COLORS[(g.charCodeAt(0) - 65) % GROUP_COLORS.length] ?? "#8b949e";

/** コース適性の色（高いほど濃い）。グループの色と紛れないよう金色 */
const APT_RGB = "227,179,65";
/** その日の馬場傾向の色（コース特注ゾーンの金色と見分ける） */
const DAY_RGB = "63,185,80";
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
/** 地図に使った 6 つの軸の印（数値バー・色の選択で使う） */
const TOP_RGB = "227,179,65";
/** ⇆ の線: これより短い（地図の位置でほとんど表せない）軸は図に出さない（丸の色で見る）。コース別の地図には軸の線を引かない */
const GUIDE_MIN = 0.3;

/** 補助の軸（偏差値 50±10）を、丸の色に使う 0〜100 の位置へ（正規分布の累積） */
function extraPct(v: number | null | undefined): number | null {
  if (typeof v !== "number") return null;
  const z = (v - 50) / 10, t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp((-z * z) / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return Math.round((z > 0 ? 1 - d : d) * 1000) / 10;
}

/** 出走馬の中での順位（0〜100。同じ値は平均の順位）。全馬で似た値の軸でも色の差が出るように、丸の色はこれで塗る */
function rankInField(vals: (number | null | undefined)[]): (number | null)[] {
  const ok = vals.map((v, i) => [v, i] as const).filter((t): t is readonly [number, number] => typeof t[0] === "number").sort((a, b) => a[0] - b[0]);
  const out: (number | null)[] = vals.map(() => null);
  if (ok.length === 1) out[ok[0][1]] = 50;
  for (let i = 0; i < ok.length && ok.length > 1;) {
    let j = i;
    while (j + 1 < ok.length && ok[j + 1][0] === ok[i][0]) j++;
    const r = ((i + j) / 2 / (ok.length - 1)) * 100;
    for (let k = i; k <= j; k++) out[ok[k][1]] = r;
    i = j + 1;
  }
  return out;
}

/** 図の上の札（馬番＋馬名 4 文字）。全角は 1 文字 = 1em、数字は約 0.62em として幅を見積もる */
const FS = 10.5, AXFS = 11, R_PX = 210, GAP = 2;
const textWidth = (t: string, fs: number) => [...t].reduce((w, c) => w + (/[\x20-\x7e]/.test(c) ? 0.62 : 1) * fs, 0);
const shortLabel = (h: RunningMapHorse) => `${h.horse_number ?? ""}${h.horse_number ? " " : ""}${[...(h.horse_name ?? "?")].slice(0, 4).join("")}`;
const fullLabel = (h: RunningMapHorse) => `${h.horse_number ? `${h.horse_number}:` : ""}${h.horse_name ?? "?"}${placeOf(h) ? `（${h.finish_position}着）` : ""}`;
/** 結果が出ているレースの 5 着以内の印（色は群・方面の塗りと別に、縁と角の丸札で出す）。1 着=金、2 着=銀、3 着=銅、4〜5 着=青 */
const PLACE_STYLE: Record<number, { color: string; label: string }> = {
  1: { color: "#f5c518", label: "1着" }, 2: { color: "#d0d7de", label: "2着" }, 3: { color: "#e0803a", label: "3着" }, 4: { color: "#3aa6e0", label: "4〜5着" },
};
const placeOf = (h: RunningMapHorse): number | null => (h.finish_position != null && h.finish_position >= 1 && h.finish_position <= 5 ? Math.min(h.finish_position, 4) : null);
type Box = { x: number; y: number; w: number; h: number };
const overlap = (a: Box, b: Box) => Math.min((a.w + b.w) / 2 + GAP - Math.abs(a.x - b.x), (a.h + b.h) / 2 + GAP - Math.abs(a.y - b.y));

/** 札の重なりをほどく: 重なった 2 枚を浅い向きへ押し離し、少しずつ元の位置へ戻す（軸名の札は動かさない障害物）。決定的 */
function relax(boxes: Box[], anchors: { x: number; y: number }[], fixed: Box[]): void {
  for (let it = 0; it < 400; it++) {
    let moved = false;
    for (let i = 0; i < boxes.length; i++) {
      const a = boxes[i];
      for (let j = i + 1; j < boxes.length; j++) {
        const b = boxes[j];
        if (overlap(a, b) <= 0) continue;
        const ox = (a.w + b.w) / 2 + GAP - Math.abs(a.x - b.x), oy = (a.h + b.h) / 2 + GAP - Math.abs(a.y - b.y);
        if (oy <= ox) { const s = (a.y < b.y || (a.y === b.y && i < j)) ? -1 : 1; a.y += (s * oy) / 2; b.y -= (s * oy) / 2; }
        else { const s = (a.x < b.x || (a.x === b.x && i < j)) ? -1 : 1; a.x += (s * ox) / 2; b.x -= (s * ox) / 2; }
        moved = true;
      }
      for (const f of fixed) {
        if (overlap(a, f) <= 0) continue;
        const ox = (a.w + f.w) / 2 + GAP - Math.abs(a.x - f.x), oy = (a.h + f.h) / 2 + GAP - Math.abs(a.y - f.y);
        if (oy <= ox) a.y += (a.y < f.y ? -1 : 1) * oy; else a.x += (a.x < f.x ? -1 : 1) * ox;
        moved = true;
      }
      a.x += (anchors[i].x - a.x) * 0.03;
      a.y += (anchors[i].y - a.y) * 0.03;
    }
    if (!moved && it > 20) break;
  }
}

function RunningMapView({ figure, entries }: { figure: RunningMapFigure; entries: AnalysisEntry[] }) {
  const [sel, setSel] = useState<string | null>(null);
  // 丸の色: "group" か軸のキー。コース別の地図では、寄与率がいちばん高い軸
  const [colorKey, setColorKey] = useState<string>(() => {
    const cm = figure.course_map;
    if (cm) return cm.axes[0] ?? "group";
    const worst = [...figure.axes].sort((a, b) => (a.r2 ?? 0) - (b.r2 ?? 0))[0];
    return worst?.key ?? "group";
  });
  // 背景: コース特注ゾーン（金色の塗り）と、馬場特注ゾーン（今日・前日の結果。緑）を重ねて出す。どちらも切り替えられる
  const day = figure.day_trend ?? null;
  const [showCourse, setShowCourse] = useState(true);
  const [showDay, setShowDay] = useState(true);
  const [b5mode, setB5mode] = useState<RunningMapBest5Mode | null>(null);
  if (figure.status !== "ok") {
    return <p style={{ fontSize: 13, color: "var(--text-dim)", marginTop: 8 }}>走法データのある馬が少なく、図を作れません。</p>;
  }
  const hs = figure.horses;
  const ex = figure.extent;
  // 縮尺: 原点から最も遠い馬が半径 R_PX に来るように。軸の先端はその外側に置き、軸名が馬の札と重ならないようにする
  const Rd = Math.max(0.5, ...hs.map((h) => Math.hypot(h.x, h.y))) * 1.08;
  const s = R_PX / Rd;
  const px = (x: number) => x * s;
  const py = (y: number) => -y * s;
  const apt = figure.aptitude;
  const zone = (heat: { lift: number[]; sig: string } | null | undefined, sigOnly: number, faint: number) => {
    const out = [] as { k: number; x: number; y: number; o: number; sig: boolean }[];
    if (!heat || !ex) return out;
    const n = ex.n, dx = (ex.x[1] - ex.x[0]) / (n - 1), dy = (ex.y[1] - ex.y[0]) / (n - 1);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const k = i * n + j, sg = heat.sig[k], lf = heat.lift[k];
      if (sg === "0" || !(lf > 1)) continue;
      const o = Math.min(1, (lf - 1) / 0.5) * (sg === "2" ? sigOnly : faint);
      out.push({ k, x: ex.x[0] + i * dx, y: ex.y[0] + j * dy, o, sig: sg === "2" });
    }
    return out;
  };
  const cells = showCourse ? zone(apt, 0.62, 0.2) : [];
  // 馬場特注ゾーン: 偶然とは言いにくい（補正後に有意な）格子は濃い緑＋枠、好走寄りだが有意でない格子はごく薄く
  const dayCells = showDay ? zone(day?.heat, 0.55, 0.12) : [];
  const cw = ex ? ((ex.x[1] - ex.x[0]) / (ex.n - 1)) * s : 0, ch = ex ? ((ex.y[1] - ex.y[0]) / (ex.n - 1)) * s : 0;
  const bracketOf = new Map(entries.map((e) => [e.key, e.bracket_number]));
  const picked = hs.find((h) => (h.key ?? h.horse_id) === sel) ?? null;
  const near = new Map((picked?.similar ?? []).map((sm, i) => [sm.horse_id, i]));
  const byHorseId = new Map(hs.map((h) => [h.horse_id, h]));
  // 丸の色に使える軸: 地図の方面＋補助の軸（急坂・道悪）。色は出走馬の中での順位で塗る
  const colorAxes: RunningMapAxis[] = [...figure.axes, ...(figure.extra_axes ?? []).map((x) => ({ ...x, weight: 0 }))];
  const isExtra = (k: string) => (figure.extra_axes ?? []).some((x) => x.key === k);
  const pctOf = (h: RunningMapHorse, k: string) => (isExtra(k) ? extraPct(h.extras?.[k]?.value) : h.percentiles[k]);
  const colorAxis = colorAxes.find((a) => a.key === colorKey) ?? null;
  // 色の選択肢: コース別の地図では、地図に使った 6 つの軸（今日の馬場の傾向から選んだ軸は、そのとき足す）
  const topKeys = figure.course_map?.axes ?? [];
  const choices = figure.course_map
    ? [...topKeys, ...(colorAxis && !topKeys.includes(colorAxis.key) ? [colorAxis.key] : [])].map((k) => colorAxes.find((a) => a.key === k)).filter((a): a is RunningMapAxis => !!a)
    : colorAxes;
  const ranks = colorAxis ? rankInField(hs.map((h) => pctOf(h, colorAxis.key))) : [];
  const fillOf = (i: number) => (colorAxis ? axisColor(ranks[i]) : groupColor(hs[i].group));

  // 軸: 原点を通る線を、馬のいる範囲の外（半径 Rd）まで引き、高い側の先端に軸名を置く
  const guides = figure.axes.filter((a) => a.arrow && !figure.course_map && (a.r2 ?? 0) >= GUIDE_MIN * GUIDE_MIN).map((a) => {
    const n = Math.hypot(a.arrow![0], a.arrow![1]) || 1;
    return { a, ux: a.arrow![0] / n, uy: a.arrow![1] / n };
  });
  const axisLabels = guides.map((g) => {
    const on = g.a.key === colorKey;
    const fs = AXFS, w = textWidth(g.a.label, fs) + 4, h = fs + 4;
    let r = R_PX + 8;
    const place = () => {
      const tx = g.ux * r, ty = -g.uy * r;
      // 先端から外向きに札を置く（右向きなら札の左端が先端、上向きなら札の下端が先端）
      return { x: tx + (g.ux * w) / 2, y: ty - (g.uy * h) / 2, w, h };
    };
    return { g, on, fs, get box() { return place(); }, push() { r += 12; } };
  });
  const axBoxes: Box[] = [];
  for (const l of axisLabels) {
    for (let k = 0; k < 6 && axBoxes.some((b) => overlap(b, l.box) > 0); k++) l.push();
    axBoxes.push(l.box);
  }
  // 馬の札: 本来の位置から始めて、重なりをほどく。動いた札は細い線で本来の位置（小さな点）とつなぐ
  const anchors = hs.map((h) => ({ x: px(h.x), y: py(h.y) }));
  const boxes: Box[] = hs.map((h, i) => ({ x: anchors[i].x, y: anchors[i].y, w: textWidth(shortLabel(h), FS) + 8, h: FS + 6 }));
  relax(boxes, anchors, axBoxes);
  // 表示範囲: 馬の札・軸名・軸の線をすべて含む最小の四角（＋余白）
  const all = [...boxes, ...axBoxes, { x: 0, y: 0, w: 2 * R_PX * 0.2, h: 2 * R_PX * 0.2 }];
  const minX = Math.min(...all.map((b) => b.x - b.w / 2)) - 6, maxX = Math.max(...all.map((b) => b.x + b.w / 2)) + 6;
  const minY = Math.min(...all.map((b) => b.y - b.h / 2)) - 6, maxY = Math.max(...all.map((b) => b.y + b.h / 2)) + 6;
  const order = hs.map((_, i) => i).sort((a, b) => (hs[a] === picked ? 1 : 0) - (hs[b] === picked ? 1 : 0) || (near.has(hs[a].horse_id) ? 1 : 0) - (near.has(hs[b].horse_id) ? 1 : 0));

  return (
    <div>
      <style>{CSS}</style>
      <div className="rm-wrap">
      <div>
      {figure.course_map && (
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 11, color: "var(--text-dim)", margin: "2px 4px 2px" }}>
          <span>近い馬ほど走法が近い（縦・横の軸に意味はありません）</span>
          <span style={{ color: `rgb(${APT_RGB})`, whiteSpace: "nowrap" }}>↑ このコースで好走しやすい走法</span>
        </div>
      )}
      <svg viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`} width="100%" style={{ display: "block", maxWidth: 560, maxHeight: "82vh", margin: "0 auto", touchAction: "manipulation" }} role="img" aria-label="走法マップ">
        {cells.map((c) => (
          <rect key={c.k} x={px(c.x) - cw / 2} y={py(c.y) - ch / 2} width={cw + 0.5} height={ch + 0.5} fill={`rgb(${APT_RGB})`} fillOpacity={c.o} shapeRendering="crispEdges" />
        ))}
        {dayCells.map((c) => (
          <rect key={`d${c.k}`} x={px(c.x) - cw / 2 + 0.5} y={py(c.y) - ch / 2 + 0.5} width={cw - 0.5} height={ch - 0.5} fill={`rgb(${DAY_RGB})`} fillOpacity={c.o}
            stroke={c.sig ? `rgb(${DAY_RGB})` : "none"} strokeOpacity={0.9} strokeWidth={0.8} shapeRendering="crispEdges" />
        ))}
        {axisLabels.map((l) => {
          const b = l.box, r = Math.hypot(b.x, b.y);
          const ex_ = l.g.ux * (R_PX + 4), ey_ = -l.g.uy * (R_PX + 4);
          return (
            <g key={`ax-${l.g.a.key}`} onClick={() => setColorKey(l.on ? "group" : l.g.a.key)} style={{ cursor: "pointer" }}>
              <title>{`${l.g.a.label}　低い: ${l.g.a.low} ⇆ 高い: ${l.g.a.high}`}</title>
              <line x1={-ex_} y1={-ey_} x2={ex_} y2={ey_} stroke={l.on ? "#e6edf3" : "#8b949e"} strokeOpacity={l.on ? 0.9 : 0.5} strokeWidth={l.on ? 1.8 : 1} />
              <line x1={-ex_} y1={-ey_} x2={ex_} y2={ey_} stroke="transparent" strokeWidth={12} />
              {r > R_PX + 14 && <line x1={ex_} y1={ey_} x2={b.x - (l.g.ux * b.w) / 2} y2={b.y + (l.g.uy * b.h) / 2} stroke="#8b949e" strokeOpacity={0.5} strokeDasharray="2 2" />}
              <text x={b.x} y={b.y + l.fs * 0.36} textAnchor="middle" fontSize={l.fs} fontWeight={l.on ? 800 : 600} fill={l.on ? "#e6edf3" : "var(--text-dim)"}>{l.g.a.label}</text>
            </g>
          );
        })}
        {picked && (picked.similar ?? []).map((sm, i) => {
          const j = hs.findIndex((h) => h.horse_id === sm.horse_id), a = hs.indexOf(picked);
          if (j < 0) return null;
          return <line key={`sim-${sm.horse_id}`} x1={boxes[a].x} y1={boxes[a].y} x2={boxes[j].x} y2={boxes[j].y} stroke="#fff" strokeOpacity={0.85 - i * 0.2} strokeWidth={2.2 - i * 0.5} strokeDasharray="5 3" />;
        })}
        {/* 札を本来の位置から動かした馬は、本来の位置（小さな点）と細い線でつなぐ。札より先に描き、札の下に隠す */}
        {hs.map((h, i) => {
          const b = boxes[i], a = anchors[i];
          if (Math.hypot(b.x - a.x, b.y - a.y) <= 6) return null;
          const c = fillOf(i), dim = picked != null && h !== picked && !near.has(h.horse_id);
          return (
            <g key={`ld-${h.key ?? h.horse_id ?? i}`} opacity={dim ? 0.3 : 1} pointerEvents="none">
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={c} strokeOpacity={0.8} strokeWidth={1} />
              <circle cx={a.x} cy={a.y} r={2.6} fill={c} />
            </g>
          );
        })}
        {order.map((i) => {
          const h = hs[i];
          const id = h.key ?? h.horse_id ?? "";
          const c = fillOf(i);
          const on = h === picked;
          const dim = picked != null && !on && !near.has(h.horse_id);
          const b = boxes[i];
          const text = on ? fullLabel(h) : shortLabel(h);
          const w = on ? textWidth(text, FS + 1) + 10 : b.w, hh = on ? b.h + 2 : b.h;
          const pl = placeOf(h);
          return (
            <g key={id} onClick={() => setSel(on ? null : id)} style={{ cursor: "pointer" }} opacity={dim ? 0.3 : 1}>
              {pl != null && <rect x={b.x - w / 2 - 3} y={b.y - hh / 2 - 3} width={w + 6} height={hh + 6} rx={(hh + 6) / 2} fill="none" stroke={PLACE_STYLE[pl].color} strokeWidth={3} />}
              <rect x={b.x - w / 2} y={b.y - hh / 2} width={w} height={hh} rx={hh / 2} fill={h.estimated ? "var(--surface)" : c}
                stroke={on ? "#fff" : h.estimated ? c : "rgba(0,0,0,.35)"} strokeWidth={on ? 2 : h.estimated ? 1.5 : 1} strokeDasharray={h.estimated ? "3 2" : undefined} />
              <text x={b.x} y={b.y + (on ? FS + 1 : FS) * 0.36} textAnchor="middle" fontSize={on ? FS + 1 : FS} fontWeight={700} fill={h.estimated ? c : "#0d1117"}>{text}</text>
              {pl != null && (
                <g pointerEvents="none">
                  <circle cx={b.x - w / 2 + 2} cy={b.y - hh / 2 - 1} r={7.5} fill={PLACE_STYLE[pl].color} stroke="#0d1117" strokeWidth={1} />
                  <text x={b.x - w / 2 + 2} y={b.y - hh / 2 - 1 + 3.4} textAnchor="middle" fontSize={10} fontWeight={800} fill="#0d1117">{h.finish_position}</text>
                </g>
              )}
              <title>{fullLabel(h)}</title>
            </g>
          );
        })}
      </svg>
      {hs.some((h) => placeOf(h) != null) && <PlaceLegend />}
      <ColorControl axes={choices} colorKey={colorKey} onChange={setColorKey} topKeys={topKeys} />
      <BgSwitch showCourse={showCourse} onCourse={setShowCourse} showDay={showDay} onDay={setShowDay} day={day} hasCourse={!!apt} />
      {showCourse && apt && <HeatLegend rgb={APT_RGB} text={`コース特注ゾーン（${apt.course}の好走しやすさ）`} />}
      {showDay && day?.heat && <HeatLegend rgb={DAY_RGB} text={`馬場特注ゾーン（${day.surface ?? ""}・枠つきは偶然とは言いにくい）`} />}
      </div>

      <div>
      {figure.best5 && <Best5Panel b5={figure.best5} mode={b5mode ?? figure.best5.default} onMode={setB5mode} hs={hs} onPick={(h) => setSel(h.key ?? h.horse_id ?? null)} />}
      {figure.course_map && <CourseAxes cm={figure.course_map} />}
      {day && <DayTrend day={day} colorKey={colorKey} onColor={(k) => setColorKey(colorKey === k ? "group" : k)} />}
      {picked && <HorseDetail h={picked} axes={figure.axes} topKeys={topKeys} extraAxes={figure.extra_axes ?? []} bracket={bracketOf.get(picked.key ?? "") ?? null} onClose={() => setSel(null)}
          onPick={(hid) => { const o = byHorseId.get(hid); if (o) setSel(o.key ?? o.horse_id ?? null); }} bracketOf={(hid) => bracketOf.get(byHorseId.get(hid)?.key ?? "") ?? null} />}
      </div>
      </div>

      {figure.unmapped.length > 0 && (
        <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 8 }}>
          図に無い馬: {figure.unmapped.map((u) => `${u.horse_number ? `${u.horse_number}:` : ""}${u.horse_name ?? "?"}`).join("、")}
        </p>
      )}
    </div>
  );
}

function PlaceLegend() {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10, marginTop: 6, fontSize: 11, color: "var(--text-dim)" }}>
      <span>結果（縁と丸札）</span>
      {Object.values(PLACE_STYLE).map((p) => (
        <span key={p.label} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span aria-hidden style={{ width: 12, height: 12, borderRadius: 999, background: p.color, border: "1px solid var(--border)" }} />{p.label}
        </span>
      ))}
    </div>
  );
}

function HeatLegend({ rgb, text }: { rgb: string; text: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6, fontSize: 11, color: "var(--text-dim)" }}>
      <span>{text}</span>
      <span>低</span>
      <span aria-hidden style={{ width: 90, height: 8, borderRadius: 4, background: `linear-gradient(90deg, rgba(${rgb},0.05), rgba(${rgb},0.65))`, border: "1px solid var(--border)" }} />
      <span>高</span>
    </div>
  );
}

function ColorControl({ axes, colorKey, onChange, topKeys }: { axes: RunningMapAxis[]; colorKey: string; onChange: (k: string) => void; topKeys: string[] }) {
  const a = axes.find((x) => x.key === colorKey) ?? null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 8px", marginTop: 6, fontSize: 11.5 }}>
      <label style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--text-dim)" }}>
        丸の色
        <select value={colorKey} onChange={(e) => onChange(e.target.value)}
          style={{ minHeight: 30, fontSize: 12, background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 6, padding: "0 6px" }}>
          <option value="group">グループ</option>
          {axes.map((x) => {
            const r = topKeys.indexOf(x.key);
            return <option key={x.key} value={x.key}>{r >= 0 ? `寄与率${r + 1}位: ` : ""}{x.label}</option>;
          })}
        </select>
      </label>
      {a && (
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5, color: "var(--text-dim)" }}>
          <span>出走馬の中で</span>
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

const B5_LABEL: Record<RunningMapBest5Mode, string> = { both: "両方", course: "コース", day: "馬場" };
const B5_SRC: Record<string, string> = { course: "コース特注ゾーン", day: "馬場特注ゾーン" };

/** Best5: 地図の上で、このコース・今日の馬場に走法が合う順の 5 頭。結果が出ていれば実際の着順を並べる */
function Best5Panel({ b5, mode, onMode, hs, onPick }: { b5: RunningMapBest5; mode: RunningMapBest5Mode; onMode: (m: RunningMapBest5Mode) => void; hs: RunningMapHorse[]; onPick: (h: RunningMapHorse) => void }) {
  const cur = b5.modes[mode] ?? b5.modes[b5.default];
  if (!cur) return null;
  const byNo = new Map(hs.map((h) => [h.horse_number, h]));
  const rows = cur.picks.map((p) => ({ p, h: byNo.get(p.horse_number) ?? null }));
  const done = hs.some((h) => h.finish_position != null);
  const hit = rows.filter((r) => r.h?.finish_position != null && r.h.finish_position <= 3).length;
  const top3 = hs.filter((h) => h.finish_position != null && h.finish_position <= 3).sort((a, b) => (a.finish_position ?? 0) - (b.finish_position ?? 0));
  const inPicks = new Set(cur.picks.map((p) => p.horse_number));
  const modes = (["both", "course", "day"] as RunningMapBest5Mode[]).filter((m) => b5.modes[m]);
  return (
    <div style={{ fontSize: 12, margin: "0 0 8px", padding: "6px 8px", borderRadius: 8, border: "1px solid var(--border)" }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6 }}>
        <b>走法が合う Best5</b>
        {modes.map((m) => (
          <button key={m} type="button" onClick={() => onMode(m)} aria-pressed={m === mode}
            style={{ minHeight: 28, padding: "0 10px", borderRadius: 999, cursor: "pointer", fontSize: 12, color: "var(--text)",
              border: `1px solid ${m === mode ? "var(--text)" : "var(--border)"}`, background: m === mode ? "rgba(139,148,158,0.2)" : "transparent" }}>{B5_LABEL[m]}</button>
        ))}
      </div>
      {cur.fallback && <div style={{ color: "var(--text-dim)", marginTop: 2 }}>{mode === "day" ? "馬場特注ゾーンがまだ無いので" : "片方のゾーンしか無いので"}{B5_SRC[cur.fallback]}で選んでいます。</div>}
      <ol style={{ listStyle: "none", margin: "4px 0 0", padding: 0 }}>
        {rows.map(({ p, h }) => {
          const fp = h?.finish_position ?? null;
          return (
            <li key={p.rank} style={{ display: "flex", gap: 8, alignItems: "baseline", padding: "2px 0", cursor: h ? "pointer" : "default" }} onClick={() => h && onPick(h)}>
              <span style={{ width: 18, color: "var(--text-dim)" }}>{p.rank}</span>
              <span style={{ flex: 1 }}>{p.horse_number ?? "?"} {h?.horse_name ?? ""}</span>
              {done && <span style={{ fontWeight: fp != null && fp <= 3 ? 800 : 400, color: fp != null && fp <= 3 ? "#3fb950" : "var(--text-dim)" }}>{fp != null ? `${fp}着` : "着外・中止"}</span>}
            </li>
          );
        })}
      </ol>
      {done && (
        <div style={{ color: "var(--text-dim)", marginTop: 3 }}>
          5 頭中 <b style={{ color: "var(--text)" }}>{hit}</b> 頭が 3 着以内。実際の 1〜3 着: {top3.map((h) => `${h.finish_position}着 ${h.horse_number ?? "?"}${h.horse_name ?? ""}${inPicks.has(h.horse_number) ? "（Best5）" : ""}`).join(" / ")}
        </div>
      )}
      <div style={{ color: "var(--text-dim)", marginTop: 3 }}>
        地図の背景の色（コース特注ゾーン・馬場特注ゾーン）が濃い所にいる馬の順です。人気は考えていません。
        2025〜2026 年の約 1,500 レースで検証すると、5 頭中の 3 着以内は平均約 1.6 頭（でたらめな 5 頭は約 1.1 頭、人気上位 5 頭は約 2.2 頭）。
        人気で説明できる分を除いた上乗せはごくわずかで、コース・馬場・両方の差は誤差の範囲です。予想の目安の 1 つとして見てください。
      </div>
    </div>
  );
}

function CourseAxes({ cm }: { cm: NonNullable<RunningMapFigure["course_map"]> }) {
  return (
    <div style={{ fontSize: 12, margin: "4px 0 8px" }}>
      <div style={{ color: "var(--text-dim)" }}>
        {cm.label}で好走への寄与率（太字の {cm.axes.length} 方面で地図を作成）{cm.level === "venue_group" ? "（このコースは出走が少ないため、同じ競馬場・芝ダートの距離帯でまとめて測定）" : cm.level === "group" ? "（出走が少ないため、全競馬場の芝ダ・距離帯でまとめて測定）" : ""}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 10px", marginTop: 3 }}>
        {cm.importance.slice(0, Math.max(5, cm.axes.length + 1)).map((i, n) => (
          <span key={i.key} style={{ fontWeight: n < cm.axes.length ? 700 : 400, color: n < cm.axes.length ? "var(--text)" : "var(--text-dim)" }}>
            {n < cm.axes.length ? `${n + 1}. ` : ""}{i.label} {Math.round(i.share * 100)}%{i.sign < 0 ? "（低いほど好走）" : ""}
          </span>
        ))}
      </div>
    </div>
  );
}

function BgSwitch({ showCourse, onCourse, showDay, onDay, day, hasCourse }: {
  showCourse: boolean; onCourse: (v: boolean) => void; showDay: boolean; onDay: (v: boolean) => void; day: RunningMapDayTrend | null; hasCourse: boolean;
}) {
  const btn = (on: boolean, set: (v: boolean) => void, text: string, rgb: string, disabled = false) => (
    <button type="button" onClick={() => set(!on)} aria-pressed={on} disabled={disabled}
      style={{ minHeight: 30, padding: "0 10px", borderRadius: 999, cursor: disabled ? "default" : "pointer", fontSize: 12, color: disabled ? "var(--text-dim)" : "var(--text)",
        border: `1px solid ${on && !disabled ? `rgb(${rgb})` : "var(--border)"}`, background: on && !disabled ? `rgba(${rgb},0.18)` : "transparent" }}>{on && !disabled ? "✓ " : ""}{text}</button>
  );
  const dayText = !day ? "馬場特注ゾーン（前のレースなし）"
    : day.heat ? `馬場特注ゾーン${day.status === "trend" ? "（傾向あり）" : ""}` : "馬場特注ゾーン（まだ判定できません）";
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6, marginTop: 6, fontSize: 11.5, color: "var(--text-dim)" }}>
      背景
      {hasCourse && btn(showCourse, onCourse, "コース特注ゾーン", APT_RGB)}
      {btn(showDay, onDay, dayText, DAY_RGB, !day?.heat)}
    </div>
  );
}

function DayTrend({ day, colorKey, onColor }: { day: RunningMapDayTrend; colorKey: string; onColor: (k: string) => void }) {
  const span = day.n_races_prev != null ? `今日 ${day.n_races_today ?? 0} レース＋前日 ${day.n_races_prev} レース` : `前の ${day.n_races} レース`;
  const head = `今日の${day.surface ?? ""}馬場（このレースより前の ${span}・${day.n_runs} 頭。直前のレースほど重く数える${day.track_today ? `・今日は${day.track_today}` : ""}）`;
  return (
    <div style={{ fontSize: 12, margin: "0 0 8px", padding: "6px 8px", borderRadius: 8, border: `1px solid rgba(${DAY_RGB},0.5)`, background: `rgba(${DAY_RGB},0.06)` }}>
      <div style={{ color: "var(--text-dim)" }}>{head}</div>
      {day.status === "few" ? <div style={{ marginTop: 2 }}>前のレースが少なく、まだ傾向は判定できません。</div>
        : day.trend.length === 0 ? <div style={{ marginTop: 2 }}>はっきりした傾向は見られません。</div>
        : (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 6px", marginTop: 4 }}>
            {day.trend.map((t) => (
              <button key={t.key} type="button" onClick={() => onColor(t.key)} aria-pressed={colorKey === t.key}
                title="押すとこの軸で丸を色分け"
                style={{ minHeight: 30, padding: "0 10px", borderRadius: 999, cursor: "pointer", fontSize: 12, color: "var(--text)",
                  border: `1px solid ${colorKey === t.key ? `rgb(${DAY_RGB})` : "var(--border)"}`, background: colorKey === t.key ? `rgba(${DAY_RGB},0.2)` : "transparent" }}>
                {t.text}馬が好走
              </button>
            ))}
          </div>
        )}
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

function HorseDetail({ h, axes, topKeys, extraAxes, bracket, onClose, onPick, bracketOf }: {
  h: RunningMapHorse; axes: RunningMapAxis[]; topKeys: string[]; extraAxes: NonNullable<RunningMapFigure["extra_axes"]>; bracket: number | null; onClose: () => void;
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
        {h.estimated ? "推定（父の産駒の平均）" : `実測ラップ ${h.n_lap_runs ?? 0} 走・前走 ${h.last_run_date ?? "–"}`}
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
        {/* 地図に使った寄与率の上位の軸を先に並べ、金色の枠・順位の印で強調する */}
        {[...axes].sort((x, y) => (topKeys.indexOf(x.key) + 1 || 99) - (topKeys.indexOf(y.key) + 1 || 99)).map((a) => {
          const p = h.percentiles[a.key];
          // 全馬の中での位置（0〜100）を 10 点満点・0.5 点刻みに
          const sc = typeof p === "number" ? Math.round(p / 5) / 2 : null;
          const rk = topKeys.indexOf(a.key);
          const top = rk >= 0;
          return (
            <div key={a.key} style={{ display: "grid", gridTemplateColumns: "8.5em 1fr 3em", alignItems: "center", gap: 6, fontSize: 11.5,
              ...(top ? { padding: "3px 6px", margin: "0 -6px", borderRadius: 6, background: `rgba(${TOP_RGB},0.10)`, border: `1px solid rgba(${TOP_RGB},0.55)` } : {}) }}>
              <span style={{ color: top ? "var(--text)" : "var(--text-dim)", fontWeight: top ? 700 : 400, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                title={`${top ? `このコースの寄与率 ${rk + 1} 位・` : ""}実測の走 ${h.axis_runs?.[a.key] ?? 0} 走`}>
                {top && <span style={{ display: "inline-block", minWidth: "1.3em", marginRight: 3, padding: "0 3px", borderRadius: 4, fontSize: 10, fontWeight: 800,
                  color: "#0d1117", background: `rgb(${TOP_RGB})`, textAlign: "center" }}>{rk + 1}</span>}
                {a.label}
              </span>
              <span style={{ position: "relative", height: top ? 10 : 8, borderRadius: 5, background: "var(--border)" }} title={`${a.low} ← → ${a.high}`}>
                {sc != null && <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${sc * 10}%`, borderRadius: 5, background: top ? `rgb(${TOP_RGB})` : c, opacity: fromPrior.has(a.key) ? 0.35 : 1 }} />}
                <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--text-dim)", opacity: 0.6 }} />
              </span>
              <span style={{ textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: top ? 700 : 400 }}>{sc != null ? sc.toFixed(1) : "–"}{fromPrior.has(a.key) ? "*" : ""}</span>
            </div>
          );
        })}
      </div>
      {extraAxes.length > 0 && (
        <div style={{ display: "grid", gap: 3, marginTop: 8, paddingTop: 6, borderTop: "1px solid var(--border)", fontSize: 11.5 }}>
          {extraAxes.map((x) => (
            <div key={x.key} style={{ display: "grid", gridTemplateColumns: "8.5em 1fr", gap: 6 }}>
              <span style={{ color: "var(--text-dim)" }} title={`高い: ${x.high} / 低い: ${x.low}`}>{x.label}</span>
              <span>{extraText(h.extras?.[x.key], x.key)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** 良馬場の地図（既定）と、道悪（稍重以上）の地図を切り替える。道悪の地図は道悪適性を軸に含み、稍重以上のレースだけで好走ゾーンを測っている */
export function RunningMap({ figure, entries }: { figure: RunningMapFigure; entries: AnalysisEntry[] }) {
  const wet = figure.wet ?? null;
  const [mode, setMode] = useState<"good" | "wet">("good");
  const shown = mode === "wet" && wet ? wet : figure;
  return (
    <div>
      {wet && (
        <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
          {(["good", "wet"] as const).map((m) => (
            <button key={m} type="button" onClick={() => setMode(m)} aria-pressed={mode === m}
              style={{ fontSize: 12, padding: "3px 10px", borderRadius: 6, cursor: "pointer", border: "1px solid var(--border)",
                background: mode === m ? "var(--accent, #58a6ff)" : "transparent", color: mode === m ? "#0d1117" : "var(--text)" }}>
              {m === "good" ? "良馬場の地図" : "道悪の地図"}
            </button>
          ))}
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
            {mode === "good" ? "良馬場のレースだけで測った地図（基礎スピードを軸に含む）" : "稍重以上のレースだけで測った地図（基礎スピードの代わりに道悪適性を軸に含む）"}
          </span>
        </div>
      )}
      <RunningMapView key={mode} figure={shown} entries={entries} />
    </div>
  );
}
