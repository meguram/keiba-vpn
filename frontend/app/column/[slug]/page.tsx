import "@/styles/column.css";
import { notFound } from "next/navigation";
import { PageShell } from "@/components/PageShell";
import { ColumnArticle, ColumnTrail } from "@/components/column/ColumnParts";
import { COLUMNS, GLOSSARY, GUIDES, findColumn } from "@/content/columns";

export const dynamicParams = false;

export function generateStaticParams() {
  return COLUMNS.map((c) => ({ slug: c.slug }));
}

export default async function ColumnArticlePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const col = findColumn(slug);
  if (!col) notFound();
  const ds = col.dataset;
  const meta = [ds.period, ds.sample, `更新 ${col.updated ?? col.published}`].filter(Boolean).join(" · ");

  return (
    <div className="col-article">
      <PageShell title={col.title} description={meta} note={false}>
        <div className="col"><ColumnTrail path={col.categories[0]} /></div>
        <ColumnArticle col={col} guide={GUIDES[col.slug]} glossary={GLOSSARY} />
      </PageShell>
    </div>
  );
}
