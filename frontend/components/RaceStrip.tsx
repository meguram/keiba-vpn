"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { LEVEL_STYLE, RaceValue, ValueLevel, valueTitle } from "@/lib/raceValue";
import { venuesOf } from "@/lib/weekDates";

export type StripRace = { race_id: string; race_name?: string; venue?: string; round?: number | string };

const BTN: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 12px", borderRadius: 6, fontSize: 12,
  border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", textDecoration: "none", whiteSpace: "nowrap", cursor: "pointer",
};

/** 幅 640px 未満（スマホ）か。スマホでは 1 行の横スクロールにして、画面をふさがない・指で押しやすい大きさにする */
export function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 639px)");
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return narrow;
}

/**
 * 開催日の中でレースを移動するバー。レース詳細（RaceNav）と同じ作り:
 * 開催場タブ（★ = 特に価値の高いレースの数）、R チップ（推奨の期待回収率で色分け）、前後の R ボタン。
 */
export function RaceStrip({
  races, venue, onVenue, values, selectedRaceId, onSelectRace,
}: {
  races: StripRace[];
  venue: string;
  onVenue: (v: string) => void;
  values: Map<string, RaceValue>;
  selectedRaceId: string;
  onSelectRace: (raceId: string) => void;
}) {
  const narrow = useNarrow();
  const chipsRef = useRef<HTMLDivElement | null>(null);
  const venues = useMemo(() => venuesOf(races), [races]);
  const sameVenue = useMemo(
    () => races.filter((r) => r.venue === venue).sort((a, b) => Number(a.round ?? 0) - Number(b.round ?? 0)),
    [races, venue],
  );
  useEffect(() => {
    if (!narrow) return;
    const el = chipsRef.current?.querySelector<HTMLElement>('[aria-current="true"]');
    el?.scrollIntoView({ inline: "center", block: "nearest" });
  }, [narrow, selectedRaceId, venue]);
  if (sameVenue.length === 0) return null;

  const idx = sameVenue.findIndex((r) => r.race_id === selectedRaceId);
  const prev = idx > 0 ? sameVenue[idx - 1] : undefined;
  const next = idx >= 0 ? sameVenue[idx + 1] : sameVenue[0];
  const roundOf = (r: StripRace) => Number(r.round ?? Number(r.race_id.slice(-2)));
  const hotIn = (v: string) => races.filter((r) => r.venue === v && values.get(r.race_id)?.level === "hot").length;
  const sel = idx >= 0 ? sameVenue[idx] : undefined;

  return (
    <nav
      aria-label="レース移動"
      style={{
        position: "sticky", top: 44, zIndex: 20, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12,
        padding: narrow ? "6px 8px" : "12px 16px", marginBottom: 14, display: "flex", flexDirection: "column", gap: narrow ? 6 : 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: narrow ? "nowrap" : "wrap", overflowX: narrow ? "auto" : undefined }}>
        {venues.map((v) => {
          const on = v === venue;
          const hot = hotIn(v);
          return (
            <button
              key={v}
              onClick={() => onVenue(v)}
              aria-current={on ? "page" : undefined}
              style={{
                ...BTN, fontSize: 13, fontWeight: on ? 700 : 500, minHeight: narrow ? 40 : undefined, flex: "none",
                border: `1px solid ${on ? "var(--accent)" : "var(--border)"}`,
                background: on ? "rgba(59,130,246,0.15)" : "var(--surface)", color: on ? "var(--accent)" : "var(--text-dim)",
              }}
            >
              {v}
              {hot > 0 && <span style={{ marginLeft: 6, fontSize: 11, fontWeight: 700, color: "#fb923c" }} title="特に価値の高いレース">★{hot}</span>}
            </button>
          );
        })}
        {sel && (
          <Link href={`/race/${sel.race_id}`} style={{ ...BTN, marginLeft: "auto", flex: "none", minHeight: narrow ? 40 : undefined, color: "var(--accent)" }}>
            {narrow ? `${roundOf(sel)}R 詳細` : `${venue} ${roundOf(sel)}R の詳細へ`}
          </Link>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: narrow ? "nowrap" : "wrap" }}>
        {!narrow && (prev ? (
          <button onClick={() => onSelectRace(prev.race_id)} style={BTN} title={prev.race_name}>← {roundOf(prev)}R</button>
        ) : <span style={{ ...BTN, opacity: 0.35, cursor: "default" }}>← 前のR</span>)}

        <div
          ref={chipsRef}
          style={narrow
            ? { display: "flex", gap: 6, flexWrap: "nowrap", flex: 1, overflowX: "auto", padding: "4px 2px", scrollSnapType: "x proximity", WebkitOverflowScrolling: "touch" }
            : { display: "flex", gap: 4, flexWrap: "wrap", flex: 1, justifyContent: "center" }}
        >
          {sameVenue.map((r) => {
            const on = r.race_id === selectedRaceId;
            const v = values.get(r.race_id);
            const lv: ValueLevel = v?.level ?? "none";
            const st = LEVEL_STYLE[lv];
            return (
              <button
                key={r.race_id}
                onClick={() => onSelectRace(r.race_id)}
                aria-current={on ? "true" : undefined}
                title={`${roundOf(r)}R ${r.race_name ?? ""}\n${valueTitle(v)}`}
                data-level={lv}
                style={{
                  minWidth: narrow ? 48 : 46, minHeight: narrow ? 46 : undefined, flex: "none", scrollSnapAlign: "center",
                  display: "inline-flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 1,
                  padding: "5px 6px", borderRadius: 8, lineHeight: 1.15, cursor: "pointer",
                  border: `1px solid ${st.border}`, background: st.bg, color: st.color,
                  boxShadow: [on ? "0 0 0 2px var(--bg), 0 0 0 4px #e6edf3" : "", st.glow ?? ""].filter(Boolean).join(", ") || undefined,
                  opacity: lv === "pass" || lv === "none" ? (on ? 1 : 0.9) : 1,
                  transform: lv === "hot" ? "translateY(-1px)" : undefined,
                }}
              >
                <span style={{ fontSize: 14, fontWeight: on || lv === "hot" ? 800 : 600 }}>{lv === "hot" ? `★${roundOf(r)}` : roundOf(r)}</span>
                <span style={{ fontSize: 10, fontWeight: 600, opacity: 0.95 }}>
                  {v?.expected_roi != null && lv !== "pass" ? `${(v.expected_roi * 100).toFixed(0)}%` : lv === "pass" ? "見送" : "–"}
                </span>
              </button>
            );
          })}
        </div>

        {!narrow && (next ? (
          <button onClick={() => onSelectRace(next.race_id)} style={BTN} title={next.race_name}>{roundOf(next)}R →</button>
        ) : <span style={{ ...BTN, opacity: 0.35, cursor: "default" }}>次のR →</span>)}
      </div>

      {values.size > 0 && !narrow && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", fontSize: 11, color: "var(--text-dim)" }}>
          <span>推奨の期待回収率:</span>
          {(["hot", "good", "ok", "pass"] as ValueLevel[]).map((k) => (
            <span key={k} title={LEVEL_STYLE[k].hint} style={{ padding: "1px 8px", borderRadius: 6, border: `1px solid ${LEVEL_STYLE[k].border}`, background: LEVEL_STYLE[k].bg, color: LEVEL_STYLE[k].color }}>
              {LEVEL_STYLE[k].label}
            </span>
          ))}
          <span>（確率はモデルの出力で未検証。目安として参照）</span>
        </div>
      )}
    </nav>
  );
}
