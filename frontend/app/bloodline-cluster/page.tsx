import { PageShell } from "@/components/PageShell";
import { MOCK_BLOODLINE_CLUSTERS } from "@/lib/mock";

export default function BloodlineClusterPage() {
  return (
    <PageShell title="血統クラスター検索" description="L2 クラスタ分類・適性プロファイル（AN-08）">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {MOCK_BLOODLINE_CLUSTERS.map((c) => (
          <div key={c.cluster_id} className="card space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono" style={{ color: "var(--accent)" }}>{c.cluster_id}</span>
              <span className="text-xs" style={{ color: "var(--text-dim)" }}>{c.horse_count} 頭</span>
            </div>
            <h3 className="font-semibold">{c.label}</h3>
            <div className="flex gap-2 text-xs">
              <span className="rounded px-2 py-0.5" style={{ background: "rgba(167,139,250,0.15)", color: "var(--purple)" }}>
                芝 {Math.round(c.turf_affinity * 100)}%
              </span>
              <span className="rounded px-2 py-0.5" style={{ background: "rgba(245,158,11,0.12)", color: "var(--warn)" }}>
                ダ {Math.round(c.dirt_affinity * 100)}%
              </span>
            </div>
            <dl className="space-y-1 text-sm">
              <div className="flex justify-between">
                <dt style={{ color: "var(--text-dim)" }}>距離</dt>
                <dd>{c.distance_range} m</dd>
              </div>
              <div className="flex justify-between">
                <dt style={{ color: "var(--text-dim)" }}>脚質傾向</dt>
                <dd>{c.running_style}</dd>
              </div>
              <div className="flex justify-between">
                <dt style={{ color: "var(--text-dim)" }}>代表コース</dt>
                <dd className="text-right text-xs">{c.best_courses.join(" / ")}</dd>
              </div>
              <div className="flex justify-between">
                <dt style={{ color: "var(--text-dim)" }}>主要種牡馬</dt>
                <dd className="text-right text-xs">{c.key_sires.join(", ")}</dd>
              </div>
            </dl>
          </div>
        ))}
      </div>
    </PageShell>
  );
}
