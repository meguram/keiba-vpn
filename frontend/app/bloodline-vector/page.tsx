import { PageShell } from "@/components/PageShell";
import { MOCK_BLOODLINE_CLUSTERS } from "@/lib/mock";

export default function BloodlineVectorPage() {
  return (
    <PageShell title="血統ベクトル空間" description="PCA/UMAP/t-SNE Canvas 2D マップ（AN-09）">
      <div className="card space-y-4">
        <div
          className="flex items-center justify-center rounded-lg"
          style={{ background: "var(--surface2)", height: 220, color: "var(--text-dim)", fontSize: 13 }}
        >
          血統ベクトル空間 UMAP/PCA 2D マップ — D3.js フル実装時に描画
        </div>
        <p style={{ color: "var(--text-dim)", fontSize: 13 }}>
          各馬の血統を高次元ベクトルに変換し、主成分分析（PCA）または UMAP で 2D 空間に射影。
          近傍クラスターが類似した適性プロファイルを持つ。
        </p>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">クラスター ID</th>
              <th className="py-2 px-3 text-left">ラベル</th>
              <th className="py-2 px-3 text-right">頭数</th>
              <th className="py-2 px-3 text-right">芝適性</th>
              <th className="py-2 px-3 text-right">ダ適性</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_BLOODLINE_CLUSTERS.map((c) => (
              <tr key={c.cluster_id} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3 font-mono text-xs" style={{ color: "var(--accent)" }}>{c.cluster_id}</td>
                <td className="py-2 px-3">{c.label}</td>
                <td className="py-2 px-3 text-right">{c.horse_count}</td>
                <td className="py-2 px-3 text-right">{Math.round(c.turf_affinity * 100)}%</td>
                <td className="py-2 px-3 text-right">{Math.round(c.dirt_affinity * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
