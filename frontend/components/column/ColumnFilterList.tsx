"use client";

import { useMemo, useState } from "react";
import { ColumnCard, type ColumnSummary } from "@/components/column/ColumnCard";
import { VERDICTS, type ColumnVerdict } from "@/lib/columns";

/**
 * コラム一覧の絞り込み。結論ごとに「予想の材料になるか」で分ける。
 * 「傾向なし」= 人気（オッズ）にすでに織り込まれていて、知っていても得にならない通説。外れではなく『買いの根拠にならない』という成果として並べる。
 * ボタンの名前はカードのバッジ（VERDICTS）と同じにそろえ、意味は説明文で補う。
 */
const ORDER: ColumnVerdict[] = ["supported", "partial", "not_supported", "inconclusive", "pending"];
const HINT: Record<ColumnVerdict, string> = {
  supported: "人気をそろえても差が残る通説。予想の材料になる",
  partial: "特定の条件でだけ差が残る通説。その条件のときだけ材料になる",
  not_supported: "人気（オッズ）にすでに織り込まれていて、買いの根拠にならない通説",
  inconclusive: "数が足りない・言い切れない通説",
  pending: "集計の準備中",
};

export function ColumnFilterList({ columns, empty }: { columns: ColumnSummary[]; empty: string }) {
  const [filter, setFilter] = useState<"all" | ColumnVerdict>("all");
  const counts = useMemo(() => {
    const c: Partial<Record<ColumnVerdict, number>> = {};
    for (const x of columns) c[x.verdict] = (c[x.verdict] ?? 0) + 1;
    return c;
  }, [columns]);
  const kinds = ORDER.filter((v) => counts[v]);
  const shown = filter === "all" ? columns : columns.filter((x) => x.verdict === filter);

  if (!columns.length) return <div className="card text-sm" style={{ color: "var(--text-dim)" }}>{empty}</div>;
  const btn = (on: boolean): React.CSSProperties => ({
    padding: "8px 14px", minHeight: 44, borderRadius: 999, fontSize: 13, cursor: "pointer",
    border: `1px solid ${on ? "var(--accent)" : "var(--border)"}`,
    background: on ? "color-mix(in srgb, var(--accent) 14%, transparent)" : "transparent",
    color: on ? "var(--accent)" : "var(--text)",
  });
  return (
    <div className="space-y-3">
      {kinds.length >= 2 && (
        <>
          <div role="group" aria-label="結論で絞り込む" className="flex flex-wrap gap-2">
            <button type="button" onClick={() => setFilter("all")} aria-pressed={filter === "all"} style={btn(filter === "all")}>
              すべて（{columns.length}）
            </button>
            {kinds.map((v) => (
              <button key={v} type="button" onClick={() => setFilter(v)} aria-pressed={filter === v} aria-describedby="column-filter-hint" style={btn(filter === v)}>
                <span aria-hidden style={{ color: VERDICTS[v].color }}>● </span>{VERDICTS[v].label}（{counts[v]}）
              </button>
            ))}
          </div>
          <div id="column-filter-hint" aria-live="polite" style={{ color: "var(--text-dim)", fontSize: 12 }}>
            {filter === "all" ? `${columns.length} 件を表示中` : `${shown.length} 件を表示中。${HINT[filter]}。`}
          </div>
        </>
      )}
      {shown.map((c) => <ColumnCard key={c.slug} column={c} />)}
    </div>
  );
}
