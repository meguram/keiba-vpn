import Link from "next/link";
import { VERDICTS, categoryLabel, type Column } from "@/lib/columns";

const dim: React.CSSProperties = { color: "var(--text-dim)", fontSize: 12 };

/** 一覧に出す分だけ（本文 body を含めない。クライアントの一覧に渡しても重くならない） */
export type ColumnSummary = Pick<Column, "slug" | "title" | "claim" | "verdict" | "summary" | "categories" | "published" | "updated" | "draft">;

export function summarize(c: Column): ColumnSummary {
  const { slug, title, claim, verdict, summary, categories, published, updated, draft } = c;
  return { slug, title, claim, verdict, summary, categories, published, updated, draft };
}

export function VerdictBadge({ verdict }: { verdict: Column["verdict"] }) {
  const v = VERDICTS[verdict];
  return (
    <span style={{ fontSize: 12, fontWeight: 700, padding: "2px 10px", borderRadius: 999, whiteSpace: "nowrap",
      color: v.color, border: `1px solid ${v.color}80` }}>
      {v.label}
    </span>
  );
}

export function CategoryChip({ path }: { path: string }) {
  return (
    <Link href={`/column/category/${path}`} style={{ fontSize: 11, color: "var(--text-dim)", textDecoration: "none",
      padding: "1px 8px", borderRadius: 6, border: "1px solid var(--border)", whiteSpace: "nowrap" }}>
      {categoryLabel(path)}
    </Link>
  );
}

export function ColumnCard({ column }: { column: ColumnSummary }) {
  return (
    <Link href={`/column/${column.slug}`} className="card block" style={{ textDecoration: "none", color: "inherit" }}>
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1" style={{ minWidth: 0 }}>
          <p style={dim}>通説：「{column.claim}」</p>
          <h2 className="font-bold" style={{ fontSize: 16 }}>
            {column.draft && <span style={{ color: "#d29922" }}>[下書き] </span>}
            {column.title}
          </h2>
          <p className="text-sm" style={{ color: "var(--text-dim)" }}>{column.summary}</p>
        </div>
        <VerdictBadge verdict={column.verdict} />
      </div>
      <div className="flex flex-wrap items-center gap-2" style={{ marginTop: 8 }}>
        {column.categories.map((p) => <span key={p} style={dim}>#{categoryLabel(p)}</span>)}
        <span style={{ ...dim, marginLeft: "auto" }}>{column.updated ?? column.published}</span>
      </div>
    </Link>
  );
}
