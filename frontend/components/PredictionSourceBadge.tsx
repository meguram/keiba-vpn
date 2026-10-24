/* GCS の確定した予測ではなく、ホストのローカルに残した予測（shadow / バックフィル）を表示しているときの注記。 */
const LABEL: Record<string, string> = { backfill: "算出済み（参考）", shadow: "検証用（参考）" };

export function PredictionSourceBadge({ source, note }: { source?: string; note?: string }) {
  if (!source || !LABEL[source]) return null;
  return (
    <div
      role="note"
      title={note}
      style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", gap: 8, margin: "0 0 8px", fontSize: 12, color: "var(--text-dim)" }}
    >
      <span
        style={{ fontWeight: 700, color: "#fbbf24", border: "1px solid rgba(251,191,36,0.45)", background: "rgba(251,191,36,0.08)", borderRadius: 999, padding: "1px 9px", whiteSpace: "nowrap" }}
      >
        {LABEL[source]}
      </span>
      {note && <span style={{ lineHeight: 1.5 }}>{note}</span>}
    </div>
  );
}
