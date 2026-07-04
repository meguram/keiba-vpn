import { PageShell } from "@/components/PageShell";
import { MOCK_TRACKING_DIFFICULTY } from "@/lib/mock";

const rows = [...MOCK_TRACKING_DIFFICULTY].sort((a, b) => b.ease_score - a.ease_score);

export default function TrackingDifficultyPage() {
  return (
    <PageShell title="位置追跡難易度" description="ease スコア・ペース予想・序盤ラダー（AN-05）">
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">馬名</th>
              <th className="py-2 px-3 text-right">ease スコア</th>
              <th className="py-2 px-3 text-center">ポジション</th>
              <th className="py-2 px-3 text-right">ペース感応度</th>
              <th className="py-2 px-3 text-right">平均先頭差（馬身）</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.horse_id} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3 font-medium">{r.horse_name}</td>
                <td className="py-2 px-3 text-right">
                  <span
                    className="inline-block rounded px-2 py-0.5 text-xs font-bold"
                    style={{
                      background: r.ease_score >= 80 ? "rgba(34,197,94,0.18)" : r.ease_score >= 60 ? "rgba(245,158,11,0.18)" : "rgba(239,68,68,0.18)",
                      color: r.ease_score >= 80 ? "var(--ok)" : r.ease_score >= 60 ? "var(--warn)" : "var(--err)",
                    }}
                  >
                    {r.ease_score}
                  </span>
                </td>
                <td className="py-2 px-3 text-center">{r.position_label}</td>
                <td className="py-2 px-3 text-right">{r.pace_sensitivity.toFixed(2)}</td>
                <td className="py-2 px-3 text-right">{r.leader_gap_avg.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
