import { PageShell } from "@/components/PageShell";
import { MOCK_SIRE_TREE, SireNode } from "@/lib/mock";

function SireList({ node, depth = 0 }: { node: SireNode; depth?: number }) {
  return (
    <li>
      <span
        className="inline-block rounded px-2 py-0.5 text-sm"
        style={{
          marginLeft: depth * 20,
          background: depth === 0 ? "rgba(59,130,246,0.18)" : depth === 1 ? "rgba(167,139,250,0.12)" : "var(--surface2)",
          color: depth === 0 ? "var(--accent)" : depth === 1 ? "var(--purple)" : "var(--text)",
        }}
      >
        {node.name}
      </span>
      {node.children && node.children.length > 0 && (
        <ul className="mt-1 space-y-1">
          {node.children.map((child) => (
            <SireList key={child.id} node={child} depth={depth + 1} />
          ))}
        </ul>
      )}
    </li>
  );
}

export default function PedigreeMapPage() {
  return (
    <PageShell title="血統マップ" description="D3.js サイアー系図フォースグラフ（AN-10）">
      <div className="card space-y-4">
        <div
          className="flex items-center justify-center rounded-lg"
          style={{ background: "var(--surface2)", height: 200, color: "var(--text-dim)", fontSize: 13 }}
        >
          D3.js サイアー系図フォースグラフ — フル実装時に描画
        </div>
        <h3 className="font-semibold text-sm" style={{ color: "var(--text-dim)" }}>系譜ツリー（静的表示）</h3>
        <ul className="space-y-1">
          <SireList node={MOCK_SIRE_TREE} />
        </ul>
      </div>
    </PageShell>
  );
}
