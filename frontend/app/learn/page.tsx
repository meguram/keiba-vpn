import "@/styles/column.css";
import type { Metadata } from "next";
import Link from "next/link";
import { PageShell } from "@/components/PageShell";
import { ColumnRow, summarize } from "@/components/column/ColumnCard";
import { COLUMNS } from "@/content/columns";
import { PUBLIC_COLUMN_SLUGS } from "@/lib/learn";

export const metadata: Metadata = {
  title: "はじめての方へ | めぐ競馬",
  description: "めぐ競馬の使い方・用語集と、競馬の通説をデータで確かめためぐコラム。",
};

// 匿名でも読める入口。本文はサーバで描く（静的に生成。API を呼ばない）
export default function LearnPage() {
  const cols = PUBLIC_COLUMN_SLUGS.map((s) => COLUMNS.find((c) => c.slug === s)).filter((c): c is NonNullable<typeof c> => !!c);
  return (
    <PageShell title="はじめての方へ" description="めぐ競馬の使い方と、競馬のデータの読み方をまとめました。" note={false}>
      <section className="card space-y-2">
        <h2 className="text-lg font-bold">めぐ競馬について</h2>
        <p>めぐ競馬は、JRA の中央競馬のレースを、過去の成績のデータから AI で分析するサイトです。</p>
        <p>レースの一覧・出走馬・AI の印（◎○▲△）は誰でも見られます。AI の予想の数字（勝率・複勝率など）は、有料会員の方に向けて出しています。</p>
        <p>予想は参考情報です。的中や結果を保証するものではありません。</p>
        <ul className="list-disc pl-5">
          <li><Link href="/learn/guide" className="col-link">使い方</Link>: 画面の見方と、数字の読み方。</li>
          <li><Link href="/learn/glossary" className="col-link">用語集</Link>: 勝率・複勝率・人気など、よく出る言葉の意味。</li>
        </ul>
      </section>
      <section className="col space-y-2">
        <h2 className="text-lg font-bold">めぐコラム（抜粋）</h2>
        <p>「休み明けは割引」「大外は不利」といった競馬の通説を、過去およそ 10 年の JRA の平地のレースの成績で確かめた記事です（期間は記事ごとに書いています）。</p>
        <p>人気（オッズ）から見込まれる成績と比べて、その通説に上積みがあるかを見ています。</p>
        <div className="col-list">
          {cols.map((c) => (
            <ColumnRow key={c.slug} column={summarize(c)} href={`/learn/column/${c.slug}`} />
          ))}
        </div>
      </section>
    </PageShell>
  );
}
