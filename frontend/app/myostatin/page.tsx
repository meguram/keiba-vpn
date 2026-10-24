import { PageShell } from "@/components/PageShell";
import { MOCK_MYOSTATIN } from "@/lib/mock";

export default function MyostatinPage() {
  return (
    <PageShell title="Myostatin 遺伝子" description="MSTN 型別距離適性（AN-12）">
      <div className="card space-y-4">
        <p className="text-sm" style={{ color: "var(--text-dim)" }}>
          Myostatin（MSTN）遺伝子の多型は筋繊維タイプの比率に影響し、馬の適性距離と相関する。
          CC 型は速筋優位、TT 型は遅筋優位でスタミナ型、CT はバランス型。
        </p>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">遺伝子型</th>
              <th className="py-2 px-3 text-right">頭数</th>
              <th className="py-2 px-3 text-right">割合 (%)</th>
              <th className="py-2 px-3 text-right">最適距離</th>
              <th className="py-2 px-3 text-right">スタミナ指数</th>
              <th className="py-2 px-3 text-right">推定 VO₂max</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_MYOSTATIN.map((g) => (
              <tr key={g.genotype} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3 font-medium">{g.genotype}</td>
                <td className="py-2 px-3 text-right">{g.count}</td>
                <td className="py-2 px-3 text-right">{g.pct.toFixed(1)}</td>
                <td className="py-2 px-3 text-right">{g.best_distance} m</td>
                <td className="py-2 px-3 text-right font-mono">{g.stamina_index.toFixed(2)}</td>
                <td className="py-2 px-3 text-right font-mono">{g.vo2max_est.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="flex gap-4 text-xs" style={{ color: "var(--text-dim)" }}>
          <span>n = {MOCK_MYOSTATIN.reduce((s, r) => s + r.count, 0)} 頭（モックデータ）</span>
        </div>
      </div>
    </PageShell>
  );
}
