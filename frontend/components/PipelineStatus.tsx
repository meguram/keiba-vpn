"use client";

import { fmtUpdated } from "@/lib/entriesStatus";

export type PipelineStage = { key: string; label: string; state: "done" | "waiting" | "error"; detail?: string; at?: string | null };
export type PipelineStatusData = {
  stages: PipelineStage[];
  complete: boolean;
  done: number;
  total: number;
  next?: string | null;
  completed_at?: string | null;
  race_over?: boolean;
};

/** 開催日の予測パイプライン（出馬表→馬体重→AI予測→オッズ→推奨買い目）の進み具合。全工程が済むと『完了』を出す。 */
export function PipelineStatus({ status, compact = false }: { status?: PipelineStatusData | null; compact?: boolean }) {
  if (!status?.stages?.length) return null;
  const nextStage = status.stages.find((s) => s.key === status.next);
  const doneAt = fmtUpdated(status.completed_at);
  const tone = status.complete
    ? { color: "#22c55e", bg: "rgba(34,197,94,0.08)", border: "rgba(34,197,94,0.4)" }
    : { color: "#f59e0b", bg: "rgba(245,158,11,0.07)", border: "rgba(245,158,11,0.35)" };
  return (
    <section role="status" aria-label="予測パイプラインの進み具合" style={{ margin: "0 0 10px", padding: compact ? "8px 10px" : "12px 16px", borderRadius: 10, border: `1px solid ${tone.border}`, background: tone.bg }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap", marginBottom: compact ? 6 : 8 }}>
        <strong style={{ fontSize: compact ? 12 : 13, color: tone.color }}>
          {status.complete ? "✔ 予測から推奨買い目まで完了" : `処理中 ${status.done}/${status.total}`}
        </strong>
        {status.complete && doneAt && <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{doneAt} に予測を作成</span>}
        {!status.complete && nextStage && (
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>
            次: {nextStage.label}{nextStage.detail ? `（${nextStage.detail}）` : ""}
          </span>
        )}
      </div>
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", gap: compact ? 4 : 6, flexWrap: "wrap" }}>
        {status.stages.map((s, i) => {
          const ok = s.state === "done";
          const c = ok ? "#22c55e" : s.state === "error" ? "#ef4444" : "#94a3b8";
          return (
            <li
              key={s.key}
              title={[s.detail, s.at ? fmtUpdated(s.at) : ""].filter(Boolean).join(" / ")}
              aria-label={`${s.label}: ${ok ? "完了" : s.state === "error" ? "失敗" : "待ち"}${s.detail ? `（${s.detail}）` : ""}`}
              style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: compact ? 11 : 12, padding: "2px 9px", borderRadius: 999, color: c, border: `1px solid ${c}`, background: ok ? "rgba(34,197,94,0.1)" : "transparent", whiteSpace: "nowrap" }}
            >
              <span aria-hidden>{ok ? "✔" : s.state === "error" ? "✖" : `${i + 1}`}</span>
              {s.label}
              {!compact && ok && s.detail && <span style={{ color: "var(--text-dim)", fontSize: 11 }}>{s.detail}</span>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
