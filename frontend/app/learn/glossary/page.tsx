import type { Metadata } from "next";
import Link from "next/link";
import { PageShell } from "@/components/PageShell";
import { GLOSSARY } from "@/content/columns";
import { SITE_TERMS } from "@/lib/learn";

export const metadata: Metadata = {
  title: "用語集 | めぐ競馬",
  description: "勝率・複勝率・人気・オッズから見込まれる率など、めぐ競馬とめぐコラムに出てくる言葉の意味。",
};

function Terms({ items }: { items: { term: string; desc: string }[] }) {
  return (
    <dl className="space-y-3 text-sm">
      {items.map((t) => (
        <div key={t.term}>
          <dt className="font-semibold">{t.term}</dt>
          <dd style={{ color: "var(--text-dim)" }}>{t.desc}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function GlossaryPage() {
  const data = Object.entries(GLOSSARY).map(([term, desc]) => ({ term, desc }));
  return (
    <PageShell title="用語集" description="めぐ競馬とめぐコラムに出てくる言葉の意味です。" note={false}>
      <p><Link href="/learn" className="col-link">はじめての方へ</Link> › 用語集</p>
      <section className="card space-y-3">
        <h2 className="text-lg font-bold">サイトの言葉</h2>
        <Terms items={SITE_TERMS} />
      </section>
      <section className="card space-y-3">
        <h2 className="text-lg font-bold">データの読み方</h2>
        <Terms items={data} />
      </section>
    </PageShell>
  );
}
