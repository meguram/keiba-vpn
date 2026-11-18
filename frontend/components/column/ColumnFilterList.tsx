"use client";

import { useMemo, useState } from "react";
import { ColumnRow, type ColumnSummary } from "@/components/column/ColumnCard";
import { VERDICTS, categoryLabel, type ColumnVerdict } from "@/lib/columns";

/**
 * コラムの一覧と絞り込み（キーワード・結論・並び順）。カテゴリは URL（/column/category/...）で選ぶ。
 * 結論の「傾向なし」は、人気（オッズ）に織り込み済みで買いの根拠にならない通説。外れではなく成果として並べる。
 */
const ORDER: ColumnVerdict[] = ["supported", "partial", "not_supported", "inconclusive", "pending"];

const date = (c: ColumnSummary) => c.updated ?? c.published;
/** 検索のために、全角・半角と英字の大文字・小文字をそろえる */
const fold = (s: string) => s.normalize("NFKC").toLowerCase();

export function ColumnFilterList({ columns, empty }: { columns: ColumnSummary[]; empty: string }) {
  const [q, setQ] = useState("");
  const [verdict, setVerdict] = useState<"" | ColumnVerdict>("");
  const [order, setOrder] = useState<"new" | "old">("new");
  const counts = useMemo(() => {
    const c: Partial<Record<ColumnVerdict, number>> = {};
    for (const x of columns) c[x.verdict] = (c[x.verdict] ?? 0) + 1;
    return c;
  }, [columns]);
  const shown = useMemo(() => {
    const words = fold(q).trim().split(/\s+/).filter(Boolean);
    const hit = columns.filter((c) => {
      if (verdict && c.verdict !== verdict) return false;
      const text = fold(`${c.title}${c.claim}${c.summary}${c.categories.map(categoryLabel).join("")}`);
      return words.every((w) => text.includes(w));
    });
    return hit.sort((a, b) => (order === "new" ? date(b).localeCompare(date(a)) : date(a).localeCompare(date(b))));
  }, [columns, q, verdict, order]);

  if (!columns.length) return <p className="col-small">{empty}</p>;
  return (
    <div>
      <div className="col-tools" role="search">
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="キーワードで探す（例: 初ダート・騎手）" aria-label="キーワードで探す" />
        <select value={verdict} onChange={(e) => setVerdict(e.target.value as "" | ColumnVerdict)} aria-label="結論で絞り込む">
          <option value="">結論: すべて（{columns.length}）</option>
          {ORDER.filter((v) => counts[v]).map((v) => <option key={v} value={v}>{VERDICTS[v].label}（{counts[v]}）</option>)}
        </select>
        <select value={order} onChange={(e) => setOrder(e.target.value as "new" | "old")} aria-label="並び順">
          <option value="new">新しい順</option>
          <option value="old">古い順</option>
        </select>
      </div>
      <p className="col-small" style={{ margin: "16px 0 8px" }}>
        {shown.length} 本{verdict === "not_supported" && "（傾向なし = 人気に織り込み済みで、買いの根拠にならない通説）"}
      </p>
      <div className="col-list">
        {shown.length ? shown.map((c) => <ColumnRow key={c.slug} column={c} />) : <p className="col-small" style={{ padding: 16 }}>当てはまるコラムはありません。</p>}
      </div>
    </div>
  );
}
