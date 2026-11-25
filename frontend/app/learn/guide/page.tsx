import type { Metadata } from "next";
import Link from "next/link";
import { PageShell } from "@/components/PageShell";

export const metadata: Metadata = {
  title: "使い方 | めぐ競馬",
  description: "めぐ競馬の画面の見方と、AI の予想・オッズの数字の読み方。",
};

// 匿名でも読める使い方。言い方は docs/operations/wording_glossary.md に合わせる（買うことを促さない）
export default function GuidePage() {
  return (
    <PageShell title="使い方" description="画面の見方と、数字の読み方です。" note={false}>
      <p><Link href="/learn" className="col-link">はじめての方へ</Link> › 使い方</p>

      <section className="card space-y-2">
        <h2 className="text-lg font-bold">画面の見方</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>めぐAI: 今週のレースの一覧です。日付と競馬場を選ぶと、その日のレースが並びます。</li>
          <li>レースの画面: 出走馬・枠・騎手などの出馬表と、結果の確定の後の着順を見られます。</li>
          <li>AI の印: AI の予想をもとに付けた ◎○▲△ の目安です。誰でも見られます。</li>
          <li>AI の予想の数字: 各馬が 1 着・3 着以内に入る見込みを、AI が計算した数字です。有料会員の方に向けて出しています。</li>
          <li>めぐ分析: 出馬表の統計のデータです。有料会員の方に向けて出しています。</li>
        </ul>
      </section>

      <section className="card space-y-2">
        <h2 className="text-lg font-bold">数字の読み方</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>勝率・複勝率: 1 着になる割合と、3 着以内に入る割合です。</li>
          <li>人気とオッズ: オッズが低い馬ほど、多くの人が勝つと見ています。オッズ 5 倍なら、およそ 20% の勝つ見込みです。</li>
          <li>AI の予想とオッズの比べ方: AI の見込みがオッズから見込まれる率より高い馬は、AI が人気より高く見ている馬です。外れることも多くあります。</li>
          <li>件数: 数字の元になったレースや馬の数です。件数が少ない数字は、たまたまの偏りで大きくずれます。</li>
        </ul>
        <p>くわしくは <Link href="/learn/glossary" className="col-link">用語集</Link> を見てください。</p>
      </section>

      <section className="card space-y-2">
        <h2 className="text-lg font-bold">ご注意</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>馬券の購入は 20 歳以上の方に限られます。</li>
          <li>当サイトの予想・サンプル買い目は参考情報で、的中や結果を保証するものではありません。</li>
          <li>回収率などの成績は過去の集計で、将来の結果を保証するものではありません。</li>
        </ul>
      </section>
    </PageShell>
  );
}
