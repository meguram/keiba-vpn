"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { BRACKET } from "@/lib/entryCardUi";
import { CONF_LABEL, ForecastHorse, PACE_TONE, PaceForecast, STYLE_LANES, lanesOf, packTrack } from "@/lib/positionForecast";
import { useNarrow } from "@/components/RaceStrip";

/* 展開予想（位置取り）。過去走の通過順位から作る予想で、実モデルの予測ではない（画面にも明記する）。
   PC: 先頭→後方の隊列図（近い馬は上下に積む）。スマホ: 脚質ごとの行に馬を並べる縦レイアウト。 */

const CHIP = 30;
const SHORT = 5;

function Chip({ h, on, onClick, big }: { h: ForecastHorse; on: boolean; onClick: () => void; big?: boolean }) {
  const b = h.bracket && BRACKET[h.bracket] ? BRACKET[h.bracket] : { bg: "#243049", color: "#e5e7eb" };
  const none = !h.f;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      aria-label={`${h.no}番 ${h.name}${h.f ? `、${h.f.style}、想定${h.f.early_pos_est ?? "—"}番手` : "、過去走の通過順位なし"}`}
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
  );
}

function Detail({ h }: { h: ForecastHorse }) {
  const f = h.f;
  return (
    <div role="status" style={{ marginTop: 10, padding: "10px 12px", borderRadius: 10, background: "var(--surface2)", border: "1px solid var(--border)", fontSize: 13, display: "flex", flexWrap: "wrap", gap: "6px 16px", alignItems: "baseline" }}>
      <strong style={{ fontSize: 15, color: "#f0f6fc" }}>{h.no} {h.name}</strong>
      {f ? (
        <>
          <span>脚質 <b>{f.style}</b></span>
          <span>想定 <b>{f.early_pos_est ?? "—"}番手</b></span>
          <span style={{ color: "var(--text-dim)" }}>過去 {f.runs_used} 走の最初のコーナー平均 <b style={{ color: "var(--text)" }}>{f.avg_first_corner}番手</b></span>
          <span style={{ color: "var(--text-dim)" }}>直近の通過 <b style={{ color: "var(--text)" }}>{f.last_passing}</b></span>
          <span style={{ color: "var(--text-dim)" }}>確からしさ <b style={{ color: "var(--text)" }}>{CONF_LABEL[f.confidence]}</b></span>
        </>
      ) : (
        <span style={{ color: "var(--text-dim)" }}>過去走の通過順位がないため予想できません（初出走・馬柱なし）。</span>
      )}
    </div>
  );
}

export function PositionForecast({ horses, pace }: { horses: ForecastHorse[]; pace: PaceForecast | null | undefined }) {
  const narrow = useNarrow();
  const [sel, setSel] = useState<number | null>(null);
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
  const placed = useMemo(() => packTrack(horses, w || 600, CHIP), [horses, w]);
  const rows = placed.reduce((m, p) => Math.max(m, p.row + 1), 1);
  const lanes = useMemo(() => lanesOf(horses), [horses]);
  const unknown = horses.filter((h) => !h.f);
  const cur = sel != null ? horses.find((h) => h.no === sel) ?? null : null;
  const tone = PACE_TONE[pace?.level ?? "unknown"];
  const toggle = (no: number) => setSel((s) => (s === no ? null : no));

  if (!horses.length) return null;
  return (
    <section aria-label="展開予想（位置取り）" style={{ marginBottom: 16, padding: "12px 14px", borderRadius: 12, background: "var(--surface)", border: "1px solid var(--border)" }}>
      <header style={{ display: "flex", flexWrap: "wrap", gap: "6px 10px", alignItems: "center", marginBottom: 8 }}>
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: "#f0f6fc" }}>展開予想（位置取り）</h3>
        <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 999, border: "1px solid var(--border)", color: "var(--text-dim)" }}>過去走からの予想（AI予測ではありません）</span>
        {pace && (
          <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 6, padding: "3px 12px", borderRadius: 999, background: tone.bg, border: `1px solid ${tone.bd}`, color: tone.fg, fontWeight: 800, fontSize: 13 }}>
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

      {narrow ? (
        <div ref={box} style={{ display: "grid", gap: 8 }}>
          {lanes.map((l, i) => (
            <div key={l.code} style={{ display: "grid", gridTemplateColumns: "52px 1fr", gap: 8, alignItems: "start", padding: "8px 0", borderTop: i ? "1px solid var(--border)" : "none" }}>
              <div style={{ fontSize: 13, fontWeight: 800, color: "var(--text)" }}>{l.label}<div style={{ fontSize: 11, fontWeight: 500, color: "var(--text-dim)" }}>{l.horses.length}頭</div></div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {l.horses.length ? l.horses.map((h) => <Chip key={h.no} h={h} big on={sel === h.no} onClick={() => toggle(h.no)} />) : <span style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: `${CHIP}px` }}>該当なし</span>}
              </div>
            </div>
          ))}
          <div style={{ fontSize: 11, color: "var(--text-dim)" }}>上ほど前、下ほど後ろの位置取り。各行の左ほど先頭に近い想定。</div>
        </div>
      ) : (
        <div>
          <div ref={box} style={{ position: "relative", height: 24 + rows * (CHIP + 4) + 8, borderRadius: 8, overflow: "hidden", background: "rgba(15,23,42,0.5)", border: "1px solid var(--border)" }}>
            {STYLE_LANES.map((l, i) => (
              <div key={l.code} style={{ position: "absolute", top: 0, bottom: 0, left: `${l.from * 100}%`, width: `${(l.to - l.from) * 100}%`, background: i % 2 ? "rgba(148,163,184,0.06)" : "transparent", borderLeft: i ? "1px dashed rgba(148,163,184,0.25)" : "none" }}>
                <span style={{ position: "absolute", top: 4, left: 8, fontSize: 11, fontWeight: 700, color: "var(--text-dim)" }}>{l.label}</span>
              </div>
            ))}
            {placed.map(({ h, x, row }) => (
              <div key={h.no} style={{ position: "absolute", top: 24 + row * (CHIP + 4), left: x - CHIP / 2 }}>
                <Chip h={h} on={sel === h.no} onClick={() => toggle(h.no)} />
              </div>
            ))}
            {!known.length && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", fontSize: 12, color: "var(--text-dim)" }}>過去走の通過順位が取れた馬がいません</div>}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
            <span>◀ 先頭（前）</span><span>丸の中は馬番・枠色。近い馬は上下に重ねています</span><span>後方（後ろ） ▶</span>
          </div>
        </div>
      )}

      {unknown.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6, marginTop: 10 }}>
          <span style={{ fontSize: 12, color: "var(--text-dim)" }}>データなし（—）</span>
          {unknown.map((h) => <Chip key={h.no} h={h} big={narrow} on={sel === h.no} onClick={() => toggle(h.no)} />)}
        </div>
      )}
      {cur && <Detail h={cur} />}
    </section>
  );
}
