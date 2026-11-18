import "@/styles/column.css";
import { notFound } from "next/navigation";
import { PageShell } from "@/components/PageShell";
import { ColumnBrowse, ColumnTrail } from "@/components/column/ColumnParts";
import { ColumnFilterList } from "@/components/column/ColumnFilterList";
import { summarize } from "@/components/column/ColumnCard";
import { COLUMNS } from "@/content/columns";
import { allCategoryPaths, categoryTrail, columnsIn } from "@/lib/columns";

export const dynamicParams = false;

export function generateStaticParams() {
  return allCategoryPaths().map((p) => ({ path: p.split("/") }));
}

export default async function ColumnCategoryPage({ params }: { params: Promise<{ path: string[] }> }) {
  const { path: segs } = await params;
  const path = segs.join("/");
  const trail = categoryTrail(path);
  if (!trail) notFound();
  const here = trail[trail.length - 1];

  return (
    <PageShell title={here.label} description={here.desc} note={false}>
      <div className="col"><ColumnTrail path={path} linkLast={false} /></div>
      <ColumnBrowse columns={COLUMNS} active={path}>
        <ColumnFilterList columns={columnsIn(COLUMNS, path).map(summarize)} empty="このカテゴリのコラムはまだありません。" />
      </ColumnBrowse>
    </PageShell>
  );
}
