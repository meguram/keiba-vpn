import Link from "next/link";
import { COLUMNS } from "@/content/columns";
import { PUBLIC_COLUMN_SLUGS } from "@/lib/learn";

/** トップの下に出す、匿名でも読める文章のページ（/learn）への入口。サーバで描くので HTML に入る */
export function LearnEntry() {
  const picks = PUBLIC_COLUMN_SLUGS.slice(0, 5)
    .map((s) => COLUMNS.find((c) => c.slug === s))
    .filter((c): c is NonNullable<typeof c> => !!c);
  return (
    <section aria-label="はじめての方へ" className="card space-y-2" style={{ marginTop: 24 }}>
      <h2 className="text-lg font-bold">はじめての方へ</h2>
      <p>めぐ競馬は、JRA の中央競馬のレースを、過去の成績のデータから AI で分析するサイトです。</p>
      <p>
        <Link href="/learn/guide" className="col-link">使い方</Link>・
        <Link href="/learn/glossary" className="col-link">用語集</Link>・
        <Link href="/learn" className="col-link">めぐコラム（抜粋）</Link>
      </p>
      <ul className="list-disc pl-5 text-sm">
        {picks.map((c) => (
          <li key={c.slug}><Link href={`/learn/column/${c.slug}`} className="col-link">{c.title}</Link></li>
        ))}
      </ul>
    </section>
  );
}
