import Link from "next/link";
import { PageShell } from "@/components/PageShell";

const NAV_CARDS = [
  {
    href: "/bloodline-cluster",
    title: "血統クラスター検索",
    desc: "L2 クラスタ分類・適性プロファイル（AN-08）",
    color: "var(--purple)",
  },
  {
    href: "/bloodline-vector",
    title: "血統ベクトル空間",
    desc: "PCA/UMAP/t-SNE 2D マッピング（AN-09）",
    color: "var(--cyan)",
  },
  {
    href: "/pedigree-map",
    title: "サイアー系図",
    desc: "D3.js フォースグラフによる 3 世代系図（AN-10）",
    color: "var(--teal)",
  },
  {
    href: "/pedigree-race-stats",
    title: "種牡馬成績分析",
    desc: "コース・距離・クラス別多軸フィルタリング（AN-01）",
    color: "var(--accent)",
  },
  {
    href: "/note-aptitude-race",
    title: "血統適性マップ",
    desc: "コース・距離・馬場別適性スコアマップ（AN-11）",
    color: "var(--warn)",
  },
];

export default function BloodlinePage() {
  return (
    <PageShell title="血統分析" description="血統データを軸とした各種分析ツール一覧">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {NAV_CARDS.map((c) => (
          <Link
            key={c.href}
            href={c.href}
            className="card block hover:opacity-90"
            style={{ borderColor: c.color }}
          >
            <h2 className="mb-1 font-semibold" style={{ color: c.color }}>{c.title}</h2>
            <p className="text-sm" style={{ color: "var(--text-dim)" }}>{c.desc}</p>
          </Link>
        ))}
      </div>
    </PageShell>
  );
}
