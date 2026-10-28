import Link from "next/link";
import { notFound } from "next/navigation";
import { PageShell } from "@/components/PageShell";
import { CategoryChip, ColumnBody, ColumnLayout, VerdictBadge } from "@/components/column/ColumnParts";
import { COLUMNS, findColumn } from "@/content/columns";
import { categoryTrail } from "@/lib/columns";

export const dynamicParams = false;

export function generateStaticParams() {
  return COLUMNS.map((c) => ({ slug: c.slug }));
}

export default async function ColumnArticlePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const col = findColumn(slug);
  if (!col) notFound();

  return (
    <PageShell title={col.title} note={false}>
      <p className="text-sm" style={{ color: "var(--text-dim)" }}>
        <Link href="/column" style={{ color: "var(--accent)" }}>めぐコラム</Link>
        {(categoryTrail(col.categories[0]) ?? []).map((c, i) => {
          const p = col.categories[0].split("/").slice(0, i + 1).join("/");
          return <span key={p}>{" › "}<Link href={`/column/category/${p}`} style={{ color: "var(--accent)" }}>{c.label}</Link></span>;
        })}
      </p>
      <ColumnLayout columns={COLUMNS} active={col.categories[0]}>
        <article className="card space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <p style={{ fontSize: 15 }}>検証した通説：<strong>「{col.claim}」</strong></p>
            <VerdictBadge verdict={col.verdict} />
          </div>
          <div className="flex flex-wrap gap-2">{col.categories.map((p) => <CategoryChip key={p} path={p} />)}</div>
          <dl className="grid gap-x-4 gap-y-1 text-sm" style={{ gridTemplateColumns: "max-content 1fr", color: "var(--text-dim)" }}>
            <dt>対象期間</dt><dd>{col.dataset.period}</dd>
            {col.dataset.conditions && <><dt>対象条件</dt><dd>{col.dataset.conditions}</dd></>}
            {col.dataset.sample && <><dt>件数</dt><dd>{col.dataset.sample}</dd></>}
            <dt>公開日</dt><dd>{col.published}{col.updated && `（更新 ${col.updated}）`}</dd>
          </dl>
          <hr style={{ borderColor: "var(--border)" }} />
          <ColumnBody blocks={col.body} />
        </article>
      </ColumnLayout>
    </PageShell>
  );
}
