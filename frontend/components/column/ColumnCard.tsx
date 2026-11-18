import Link from "next/link";
import { VERDICTS, categoryTrail, type Column } from "@/lib/columns";

/** 一覧に出す分だけ（本文 body を含めない。クライアントの一覧に渡しても重くならない） */
export type ColumnSummary = Pick<Column, "slug" | "title" | "claim" | "verdict" | "summary" | "categories" | "published" | "updated" | "draft">;

export function summarize(c: Column): ColumnSummary {
  const { slug, title, claim, verdict, summary, categories, published, updated, draft } = c;
  return { slug, title, claim, verdict, summary, categories, published, updated, draft };
}

/** 結論の種別。色の点の横に必ず名前を出す（色だけに頼らない） */
export function VerdictMark({ verdict }: { verdict: Column["verdict"] }) {
  const v = VERDICTS[verdict];
  return <span className="col-verdict"><i aria-hidden style={{ background: v.color }} />{v.label}</span>;
}

/** カテゴリのパスの末端の名前（「騎手 › 乗り替わり」なら「乗り替わり」） */
function leafLabel(path: string): string {
  const t = categoryTrail(path);
  return t?.[t.length - 1]?.label ?? path;
}

/** 一覧の 1 行: 結論・カテゴリ・日付 → 題 → 要約（2 行まで） */
export function ColumnRow({ column }: { column: ColumnSummary }) {
  return (
    <Link href={`/column/${column.slug}`} className="col-row">
      <div className="col-row-meta">
        <VerdictMark verdict={column.verdict} />
        <span>{[...new Set(column.categories.map(leafLabel))].join("・")}</span>
        <span style={{ marginLeft: "auto" }}>{column.updated ?? column.published}</span>
      </div>
      <h2 className="col-row-title">
        {column.draft && <span style={{ color: "var(--warn)" }}>[下書き] </span>}
        {column.title}
      </h2>
      <p className="col-row-sum">{column.summary}</p>
    </Link>
  );
}
