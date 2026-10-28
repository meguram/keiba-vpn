import Link from "next/link";
import { notFound } from "next/navigation";
import { PageShell } from "@/components/PageShell";
import { ColumnLayout, ColumnList, SubCategories } from "@/components/column/ColumnParts";
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
      <p className="text-sm" style={{ color: "var(--text-dim)" }}>
        <Link href="/column" style={{ color: "var(--accent)" }}>めぐコラム</Link>
        {trail.map((c, i) => {
          const p = segs.slice(0, i + 1).join("/");
          return (
            <span key={p}>
              {" › "}
              {i < trail.length - 1 ? <Link href={`/column/category/${p}`} style={{ color: "var(--accent)" }}>{c.label}</Link> : c.label}
            </span>
          );
        })}
      </p>
      <ColumnLayout columns={COLUMNS} active={path}>
        <SubCategories columns={COLUMNS} path={path} node={here} />
        <ColumnList columns={columnsIn(COLUMNS, path)} empty="このカテゴリのコラムはまだありません。" />
      </ColumnLayout>
    </PageShell>
  );
}
