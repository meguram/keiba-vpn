import "@/styles/column.css";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { PageShell } from "@/components/PageShell";
import { ColumnArticle } from "@/components/column/ColumnParts";
import { GLOSSARY, GUIDES, findColumn } from "@/content/columns";
import { PUBLIC_COLUMN_SLUGS, isPublicColumn, publicColumn } from "@/lib/learn";

// 許可リストの記事だけを事前生成する（ほかの slug は 404）。本文はサーバで描く
export const dynamicParams = false;

export function generateStaticParams() {
  return PUBLIC_COLUMN_SLUGS.filter((s) => !!findColumn(s)).map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const col = isPublicColumn(slug) ? findColumn(slug) : undefined;
  return col ? { title: `${col.title} | めぐコラム`, description: col.summary } : {};
}

export default async function LearnColumnPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const found = isPublicColumn(slug) ? findColumn(slug) : undefined;
  if (!found) notFound();
  const col = publicColumn(found);
  const ds = col.dataset;
  const meta = [ds.period, ds.sample, `更新 ${col.updated ?? col.published}`].filter(Boolean).join(" · ");
  return (
    <div className="col-article">
      <PageShell title={col.title} description={meta} note={false}>
        <div className="col"><Link href="/learn" className="col-link">はじめての方へ</Link> › めぐコラム</div>
        <ColumnArticle col={col} guide={GUIDES[col.slug]} glossary={GLOSSARY} linkCategories={false} />
        <p className="col" style={{ color: "var(--text-dim)", fontSize: 12 }}>
          過去の成績の集計です。将来の結果を保証するものではありません。
        </p>
      </PageShell>
    </div>
  );
}
