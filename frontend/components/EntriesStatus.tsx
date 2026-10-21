"use client";

import { changesSummary, fmtUpdated, isProvisional, phaseMeta, summarizeDay } from "@/lib/entriesStatus";
import type { EntriesStatus } from "@/lib/entriesStatus";

/** レースの出馬表の段階バッジ（特別登録／出走馬決定／出馬表確定／馬体重発表済み）と最終更新、確定前の注記。 */
export function EntriesStatusBadge({ status, compact = false }: { status?: EntriesStatus | null; compact?: boolean }) {
  if (!status || !status.phase) return null;
  const m = phaseMeta(status.phase);
  const upd = fmtUpdated(status.updated_at);
  const chg = changesSummary(status.recent_changes);
  const provisional = isProvisional(status.phase) && status.phase !== "none";
  return (
    <span role="status" aria-label={`出馬表の状態: ${m.label}`} style={{ display: "inline-flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      <span
        title={status.note ?? ""}
        style={{ fontSize: 11, fontWeight: 700, padding: "2px 9px", borderRadius: 6, whiteSpace: "nowrap", color: m.color, background: m.bg, border: `1px solid ${m.border}` }}
      >
        {m.label}
      </span>
      {!compact && upd && <span style={{ fontSize: 11, color: "var(--text-dim)" }}>更新 {upd}</span>}
      {!compact && chg && <span style={{ fontSize: 11, color: "#f59e0b" }}>{chg}</span>}
      {!compact && provisional && <span style={{ fontSize: 11, color: "var(--text-dim)" }}>頭数・騎手・枠順は確定前です</span>}
    </span>
  );
}

/** その日の全レースの出馬表の進み具合（段階ごとの件数）。 */
export function EntriesDaySummary({ races }: { races: { entries_phase?: string | null }[] }) {
  const s = summarizeDay(races);
  if (!s.total || !s.withPhase) return null;
  const order = ["registered", "entries", "final", "weighed", "none"];
  return (
    <div role="status" aria-label="この日の出馬表の状態" style={{ margin: "0 0 10px", display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", fontSize: 12, color: "var(--text-dim)" }}>
      <span>出馬表:</span>
      {order.filter((k) => (s.counts[k] ?? 0) > 0).map((k) => {
        const m = phaseMeta(k);
        return (
          <span key={k} style={{ color: m.color, background: m.bg, border: `1px solid ${m.border}`, borderRadius: 6, padding: "1px 8px", fontWeight: 600 }}>
            {m.label} {s.counts[k]}
          </span>
        );
      })}
    </div>
  );
}
