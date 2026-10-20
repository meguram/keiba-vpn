"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { bracketColor } from "@/lib/bracket";
import { CONF_LABEL, ForecastHorse, GATE_TONE, GateBias, HorseTd, PACE_TONE, PaceFit, PaceForecast, STYLE_LANES, easeTone, fmtDelta, gateWord, lanesOf, packTrack, packTrackGate, paceFit } from "@/lib/positionForecast";
import { useNarrow } from "@/components/RaceStrip";

/* 展開予想（位置取り）。過去走の通過順位から作る予想で、実モデルの予測ではない（画面にも明記する）。
   PC: 横=前後の位置取り、縦=上が内ラチ・下が外の隊列図（同じ位置なら内枠が上＝ラチ寄り）。
   スマホ: 脚質ごとの行に、内枠→外枠の順で馬を並べる縦レイアウト。 */

const CHIP = 30;
const SHORT = 5;

const FIT_TONE: Record<PaceFit["label"], { fg: string; bg: string; bd: string }> = {
  向く: { fg: "#86efac", bg: "rgba(34,197,94,0.12)", bd: "rgba(34,197,94,0.5)" },
  ふつう: { fg: "#cbd5e1", bg: "transparent", bd: "rgba(148,163,184,0.35)" },
  不向き: { fg: "#fca5a5", bg: "rgba(239,68,68,0.12)", bd: "rgba(239,68,68,0.5)" },
};

function EaseDot({ pct, size = 11 }: { pct: number | null | undefined; size?: number }) {
  if (pct == null) return null;
  return <span aria-hidden style={{ width: size, height: size, borderRadius: 999, background: easeTone(pct).color, border: "2px solid #0d1117", boxSizing: "content-box", flex: "none" }} />;
}

function Chip({ h, on, onClick, big, ease }: { h: ForecastHorse; on: boolean; onClick: () => void; big?: boolean; ease?: number | null }) {
  const b = bracketColor(h.bracket);
  const none = !h.f;
  const et = ease != null ? easeTone(ease) : null;
  return (
    <span style={{ position: "relative", display: "inline-flex" }}>
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      aria-label={`${h.no}番 ${h.name}${h.f ? `、${h.f.style}、想定${h.f.early_pos_est ?? "—"}番手` : "、過去走の通過順位なし"}${et ? `、追走${et.word}` : ""}`}
      title={`${h.no} ${h.name}`}
      style={{
        width: big ? "auto" : CHIP, minWidth: CHIP, height: CHIP, borderRadius: 999, padding: big ? "0 10px 0 0" : 0, display: "inline-flex", alignItems: "center", gap: 6,
        background: none ? "transparent" : b.bg, color: none ? "var(--text-dim)" : b.color,
        border: on ? "2px solid #fff" : none ? "1px dashed var(--border)" : "1px solid rgba(255,255,255,0.35)",
        boxShadow: on ? "0 0 0 3px rgba(96,165,250,0.55)" : "none", fontSize: 13, fontWeight: 800, cursor: "pointer", justifyContent: "center",
      }}
    >
      <span style={{ minWidth: CHIP - 2, textAlign: "center" }}>{h.no}</span>
      {big && <span style={{ fontSize: 12, fontWeight: 700, color: none ? "var(--text-dim)" : b.color, whiteSpace: "nowrap" }}>{h.name.slice(0, SHORT)}{h.name.length > SHORT ? "…" : ""}</span>}
    </button>
    {et && <span style={{ position: "absolute", top: -3, right: -3, pointerEvents: "none", display: "inline-flex" }}><EaseDot pct={ease} /></span>}
    </span>
  );
}

function Detail({ h, gate, td, fit }: { h: ForecastHorse; gate: GateBias | null | undefined; td?: HorseTd | null; fit?: PaceFit | null }) {
  const f = h.f;
  const et = td?.ease_pct != null ? easeTone(td.ease_pct) : null;
  const shifted = !!gate?.applied && f != null && f.gate_shift != null && Math.abs(f.gate_shift) >= 0.005;
  const slots = shifted && gate?.field_last_number ? Math.abs(f!.gate_shift!) * (gate.field_last_number - 1) : 0;
  return (
    <div role="status" style={{ marginTop: 10, padding: "10px 12px", borderRadius: 10, background: "var(--surface2)", border: "1px solid var(--border)", fontSize: 13, display: "flex", flexWrap: "wrap", gap: "6px 16px", alignItems: "baseline" }}>
      <strong style={{ fontSize: 15, color: "#f0f6fc" }}>{h.no} {h.name}</strong>
      {f ? (
        <>
          <span>脚質 <b>{f.style}</b></span>
          <span>想定 <b>{f.early_pos_est ?? "—"}番手</b>{gate?.applied && <span style={{ color: "var(--text-dim)", fontSize: 11 }}>（枠順補正込み）</span>}</span>
          {f.gate_rel != null && <span style={{ color: "var(--text-dim)" }}>枠 <b style={{ color: "var(--text)" }}>{gateWord(f.gate_rel)}</b>（馬番 {h.no}）</span>}
          {shifted && <span style={{ color: "var(--text-dim)" }}>補正 <b style={{ color: "var(--text)" }}>{f.gate_shift! < 0 ? "前へ" : "後ろへ"} {slots.toFixed(1)}番手分</b></span>}
          <span style={{ color: "var(--text-dim)" }}>過去 {f.runs_used} 走の最初のコーナー平均 <b style={{ color: "var(--text)" }}>{f.avg_first_corner}番手</b></span>
          <span style={{ color: "var(--text-dim)" }}>直近の通過 <b style={{ color: "var(--text)" }}>{f.last_passing}</b></span>
          <span style={{ color: "var(--text-dim)" }}>確からしさ <b style={{ color: "var(--text)" }}>{CONF_LABEL[f.confidence]}</b></span>
          {td && (
            <>
              {et && <span style={{ color: "var(--text-dim)" }}>追走難度 <b style={{ color: et.color }}>{et.word}</b>（{td.ease_pct!.toFixed(0)}％）</span>}
              {td.flow_pattern && <span style={{ color: "var(--text-dim)" }}>流れ <b style={{ color: "var(--text)" }}>{td.flow_pattern}</b></span>}
              {td.stamina_label && <span style={{ color: td.stamina_concern ? "#fbbf24" : "var(--text-dim)" }}>消耗 <b>{td.stamina_label}</b></span>}
              {td.last3f_sec != null && <span style={{ color: "var(--text-dim)" }}>想定上がり <b style={{ color: "var(--text)" }}>{td.last3f_sec.toFixed(1)}</b>{td.last3f_delta != null && `（${fmtDelta(td.last3f_delta)}）`}</span>}
              {td.style && td.style !== f.style && <span style={{ color: "var(--text-dim)", fontSize: 11 }}>追走側の脚質は「{td.style}」（別算出のため位置取りの予想とずれることがあります）</span>}
            </>
          )}
          {fit && (
            <span style={{ flexBasis: "100%", color: "var(--text-dim)" }}>
              展開 <b style={{ color: FIT_TONE[fit.label].fg }}>{fit.label}</b>{fit.reasons.length > 0 && `（${fit.reasons.join("・")}）`}
            </span>
          )}
        </>
      ) : (
        <span style={{ color: "var(--text-dim)" }}>過去走の通過順位がないため予想できません（初出走・馬柱なし）。</span>
      )}
    </div>
  );
}

/** 馬別一覧（展開向き／不向き・追走難度・消耗・想定上がり）。追走難度が未取得の間は、脚質×ペースの判定だけを出す。 */
function HorseList({ horses, pace, td, narrow, sel, onPick, detail }: { horses: ForecastHorse[]; pace: PaceForecast | null | undefined; td?: Map<number, HorseTd> | null; narrow: boolean; sel: number | null; onPick: (no: number) => void; detail: (h: ForecastHorse) => React.ReactNode }) {
  const cols = "30px minmax(96px,1.6fr) 52px 62px 124px 64px 104px";
  return (
    <div role="list" aria-label="馬別の展開" style={{ marginTop: 12, display: "grid", gap: 4 }}>
      {!narrow && (
        <div aria-hidden style={{ display: "grid", gridTemplateColumns: cols, gap: 8, padding: "0 8px", fontSize: 11, color: "var(--text-dim)" }}>
          <span>馬番</span><span>馬名</span><span>脚質</span><span>展開</span><span>追走難度</span><span>消耗</span><span>想定上がり</span>
        </div>
      )}
      {horses.map((h) => {
        const x = td?.get(h.no);
        const fit = h.f ? paceFit(h.f.style_code, pace?.level, x?.ease_pct) : null;
        const ft = fit ? FIT_TONE[fit.label] : null;
        const et = x?.ease_pct != null ? easeTone(x.ease_pct) : null;
        const b = bracketColor(h.bracket);
        const num = <span style={{ width: 26, height: 26, borderRadius: 999, background: b.bg, color: b.color, border: `1px solid ${b.border}`, boxSizing: "border-box", display: "inline-flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 13, flex: "none" }}>{h.no}</span>;
        const fitPill = fit && ft ? <span title={fit.reasons.join("・")} style={{ padding: "1px 8px", borderRadius: 999, border: `1px solid ${ft.bd}`, background: ft.bg, color: ft.fg, fontWeight: 700, fontSize: 12, whiteSpace: "nowrap" }}>{fit.label}</span> : <span style={{ color: "var(--text-dim)" }}>—</span>;
        const easeCell = et ? <span style={{ display: "inline-flex", alignItems: "center", gap: 5, whiteSpace: "nowrap" }}><span aria-hidden style={{ width: 9, height: 9, borderRadius: 999, background: et.color }} /><b style={{ color: "var(--text)" }}>{et.word}</b><span style={{ color: "var(--text-dim)" }}>{x!.ease_pct!.toFixed(0)}</span></span> : <span style={{ color: "var(--text-dim)" }}>—</span>;
        const stam = x?.stamina_label ? <span style={{ color: x.stamina_concern ? "#fbbf24" : "var(--text-dim)", whiteSpace: "nowrap" }}>{x.stamina_label}</span> : <span style={{ color: "var(--text-dim)" }}>—</span>;
        const l3 = x?.last3f_sec != null ? <span style={{ whiteSpace: "nowrap", color: "var(--text)" }}>{x.last3f_sec.toFixed(1)}<span style={{ color: "var(--text-dim)" }}>{x.last3f_delta != null ? `（${fmtDelta(x.last3f_delta)}）` : ""}</span></span> : <span style={{ color: "var(--text-dim)" }}>—</span>;
        const style = <span style={{ color: h.f ? "var(--text)" : "var(--text-dim)" }}>{h.f?.style ?? "データなし"}</span>;
        const common = { borderRadius: 8, background: sel === h.no ? "rgba(96,165,250,0.12)" : "var(--surface2)", border: `1px solid ${sel === h.no ? "rgba(96,165,250,0.6)" : "var(--border)"}`, fontSize: 12, textAlign: "left" as const, color: "var(--text)", font: "inherit", cursor: "pointer", width: "100%", boxSizing: "border-box" as const };
        const row = narrow ? (
          <button type="button" onClick={() => onPick(h.no)} aria-pressed={sel === h.no} style={{ ...common, fontSize: 12, padding: "7px 9px", display: "grid", gap: 4 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
              {num}
              <strong style={{ flex: "1 1 auto", minWidth: 0, fontSize: 13, color: "#f0f6fc", overflowWrap: "anywhere" }}>{h.name}</strong>
              {fitPill}
            </span>
            <span style={{ display: "flex", flexWrap: "wrap", gap: "2px 12px", color: "var(--text-dim)" }}>
              <span>脚質 {style}</span>
              <span>追走 {easeCell}</span>
              <span>消耗 {stam}</span>
              <span>上がり {l3}</span>
            </span>
            {fit && fit.reasons.length > 0 && <span style={{ color: "var(--text-dim)", fontSize: 11 }}>展開: {fit.reasons.join("・")}</span>}
          </button>
        ) : (
          <button type="button" onClick={() => onPick(h.no)} aria-pressed={sel === h.no} style={{ ...common, padding: "6px 8px", display: "grid", gridTemplateColumns: cols, gap: 8, alignItems: "center" }}>
            {num}
            <strong style={{ minWidth: 0, fontSize: 13, color: "#f0f6fc", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{h.name}</strong>
            {style}{fitPill}{easeCell}{stam}{l3}
          </button>
        );
        return <div key={h.no} role="listitem">{row}{sel === h.no && detail(h)}</div>;
      })}
    </div>
  );
}

export function PositionForecast({ horses, pace, gate, td, tdNote, embedded }: {
  horses: ForecastHorse[];
  pace: PaceForecast | null | undefined;
  gate?: GateBias | null;
  /** 追走難度（馬番 → 情報）。無い間は位置取りの予想だけを出す */
  td?: Map<number, HorseTd> | null;
  /** 追走難度の取得状況や注記（取得中・失敗・再読み込みなど） */
  tdNote?: React.ReactNode;
  /** 折りたたみの中に置くとき: 外枠・見出しを付けない */
  embedded?: boolean;
}) {
  const narrow = useNarrow();
  const [sel, setSel] = useState<number | null>(null);
  const [selFrom, setSelFrom] = useState<"chart" | "list">("chart");
  const [w, setW] = useState(0);
  const box = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, [narrow]);

  const known = horses.filter((h) => h.f);
  const hasGate = known.some((h) => h.f!.gate_rel != null);
  const placedOld = useMemo(() => packTrack(horses, w || 600, CHIP), [horses, w]);
  const rows = placedOld.reduce((m, p) => Math.max(m, p.row + 1), 1);
  const gated = useMemo(() => packTrackGate(horses, w || 600, CHIP, 3, 150), [horses, w]);
  const lanes = useMemo(() => lanesOf(horses, hasGate), [horses, hasGate]);
  const unknown = horses.filter((h) => !h.f);
  const cur = sel != null ? horses.find((h) => h.no === sel) ?? null : null;
  const tone = PACE_TONE[pace?.level ?? "unknown"];
  const gtone = gate ? GATE_TONE[gate.level] : null;
  const TOP = 24, BOTTOM = 20;
  const toggle = (no: number) => { setSelFrom("chart"); setSel((s) => (s === no ? null : no)); };
  const toggleList = (no: number) => { setSelFrom("list"); setSel((s) => (s === no ? null : no)); };
  const detailOf = (h: ForecastHorse) => <Detail h={h} gate={gate} td={td?.get(h.no)} fit={h.f ? paceFit(h.f.style_code, pace?.level, td?.get(h.no)?.ease_pct ?? null) : null} />;
  const hasTd = !!td && td.size > 0;
  const easeOf = (no: number) => td?.get(no)?.ease_pct ?? null;

  if (!horses.length) return null;
  return (
    <section aria-label="展開予想（位置取り）" style={embedded ? undefined : { marginBottom: 16, padding: "12px 14px", borderRadius: 12, background: "var(--surface)", border: "1px solid var(--border)" }}>
      <header style={{ display: "flex", flexWrap: "wrap", gap: "6px 10px", alignItems: "center", marginBottom: 8 }}>
        {!embedded && <h3 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: "#f0f6fc" }}>展開予想（位置取り）</h3>}
        <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 999, border: "1px solid var(--border)", color: "var(--text-dim)" }}>過去走からの予想（AI予測ではありません）</span>
        {gate && gtone && gate.level !== "na" && (
          <span style={{ marginLeft: "auto", padding: "3px 12px", borderRadius: 999, background: gtone.bg, border: `1px ${gate.level === "low" ? "dashed" : "solid"} ${gtone.bd}`, color: gtone.fg, fontWeight: 800, fontSize: 13 }}>
            {gtone.label}
          </span>
        )}
        {pace && (
          <span style={{ marginLeft: gate && gtone && gate.level !== "na" ? 0 : "auto", display: "inline-flex", alignItems: "center", gap: 6, padding: "3px 12px", borderRadius: 999, background: tone.bg, border: `1px solid ${tone.bd}`, color: tone.fg, fontWeight: 800, fontSize: 13 }}>
            <span aria-hidden>{tone.icon}</span>ペース {pace.label}
          </span>
        )}
      </header>
      {pace && (
        <p style={{ margin: "0 0 10px", fontSize: 12.5, color: "var(--text-dim)", lineHeight: 1.6 }}>
          {pace.note}
          <span style={{ marginLeft: 8, whiteSpace: "nowrap" }}>逃げ {pace.front_runner_count}頭・先行 {pace.stalker_count}頭（{pace.known_horses}/{pace.field_size}頭で判定）</span>
        </p>
      )}

      {gate && gate.level !== "na" && (
        <p style={{ margin: "0 0 10px", fontSize: 12.5, color: "var(--text-dim)", lineHeight: 1.6 }}>{gate.note}</p>
      )}

      {narrow ? (
        <div ref={box} style={{ display: "grid", gap: 8 }}>
          {lanes.map((l, i) => (
            <div key={l.code} style={{ display: "grid", gridTemplateColumns: "52px 1fr", gap: 8, alignItems: "start", padding: "8px 0", borderTop: i ? "1px solid var(--border)" : "none" }}>
              <div style={{ fontSize: 13, fontWeight: 800, color: "var(--text)" }}>{l.label}<div style={{ fontSize: 11, fontWeight: 500, color: "var(--text-dim)" }}>{l.horses.length}頭</div></div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {l.horses.length ? l.horses.map((h) => <Chip key={h.no} h={h} big ease={easeOf(h.no)} on={sel === h.no} onClick={() => toggle(h.no)} />) : <span style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: `${CHIP}px` }}>該当なし</span>}
              </div>
            </div>
          ))}
          <div style={{ fontSize: 11, color: "var(--text-dim)" }}>
            上の行ほど前、下の行ほど後ろの位置取り。{hasGate ? "各行は左が内枠→右が外枠の順（左ほどラチ寄り）。" : "各行の左ほど先頭に近い想定。"}
          </div>
        </div>
      ) : (
        <div>
          <div ref={box} style={{ position: "relative", height: hasGate ? TOP + gated.height + BOTTOM + 6 : 24 + rows * (CHIP + 4) + 8, borderRadius: 8, overflow: "hidden", background: "rgba(15,23,42,0.5)", border: "1px solid var(--border)" }}>
            {STYLE_LANES.map((l, i) => (
              <div key={l.code} style={{ position: "absolute", top: 0, bottom: 0, left: `${l.from * 100}%`, width: `${(l.to - l.from) * 100}%`, background: i % 2 ? "rgba(148,163,184,0.06)" : "transparent", borderLeft: i ? "1px dashed rgba(148,163,184,0.25)" : "none" }}>
                <span style={{ position: "absolute", top: 4, left: 8, fontSize: 11, fontWeight: 700, color: "var(--text-dim)" }}>{l.label}</span>
              </div>
            ))}
            {hasGate && (
              <>
                <div aria-hidden style={{ position: "absolute", left: 0, right: 0, top: TOP - 3, borderTop: "2px solid #e5e7eb" }} />
                <span style={{ position: "absolute", right: 8, top: 4, fontSize: 11, fontWeight: 800, color: "#e5e7eb" }}>━ 内ラチ</span>
                <div aria-hidden style={{ position: "absolute", left: 0, right: 0, bottom: BOTTOM - 2, borderTop: "1px dashed rgba(148,163,184,0.45)" }} />
                <span style={{ position: "absolute", right: 8, bottom: 3, fontSize: 11, fontWeight: 700, color: "var(--text-dim)" }}>外 ▼</span>
              </>
            )}
            {hasGate
              ? gated.placed.map(({ h, x, y }) => (
                  <div key={h.no} style={{ position: "absolute", top: TOP + y, left: x - CHIP / 2 }}>
                    <Chip h={h} ease={easeOf(h.no)} on={sel === h.no} onClick={() => toggle(h.no)} />
                  </div>
                ))
              : placedOld.map(({ h, x, row }) => (
                  <div key={h.no} style={{ position: "absolute", top: 24 + row * (CHIP + 4), left: x - CHIP / 2 }}>
                    <Chip h={h} ease={easeOf(h.no)} on={sel === h.no} onClick={() => toggle(h.no)} />
                  </div>
                ))}
            {!known.length && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", fontSize: 12, color: "var(--text-dim)" }}>過去走の通過順位が取れた馬がいません</div>}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-dim)", marginTop: 4, gap: 8 }}>
            <span>◀ 先頭（前）</span>
            <span style={{ textAlign: "center" }}>{hasGate ? "丸の中は馬番・枠色。横は前後の位置取り、縦は上がラチ寄り（内枠）・下が外枠" : "丸の中は馬番・枠色。近い馬は上下に重ねています"}</span>
            <span>後方（後ろ） ▶</span>
          </div>
        </div>
      )}

      {unknown.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6, marginTop: 10 }}>
          <span style={{ fontSize: 12, color: "var(--text-dim)" }}>データなし（—）</span>
          {unknown.map((h) => <Chip key={h.no} h={h} big={narrow} ease={easeOf(h.no)} on={sel === h.no} onClick={() => toggle(h.no)} />)}
        </div>
      )}
      {hasTd && (
        <div aria-label="追走難度の凡例" style={{ display: "flex", flexWrap: "wrap", gap: "2px 12px", marginTop: 8, fontSize: 11, color: "var(--text-dim)" }}>
          <span>丸の右上の点＝追走難度:</span>
          {[80, 65, 50, 35, 20].map((v) => { const e = easeTone(v); return <span key={v} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><span aria-hidden style={{ width: 9, height: 9, borderRadius: 999, background: e.color }} />{e.word}</span>; })}
        </div>
      )}
      {tdNote}
      {cur && !(embedded && selFrom === "list") && detailOf(cur)}
      {embedded && <HorseList horses={horses} pace={pace} td={td} narrow={narrow} sel={sel} onPick={toggleList} detail={detailOf} />}
    </section>
  );
}
