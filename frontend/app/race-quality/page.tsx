import { PageShell } from "@/components/PageShell";
import { MOCK_RACES } from "@/lib/mock";

const PACE_MAP: Record<string, string> = {
  GI: "ハイ", GIII: "ミドル", "未勝利": "スロー",
};
const STYLE_MAP: Record<string, string> = {
  GI: "先行 42% / 差し 38% / 逃げ 12% / 追い込み 8%",
  GIII: "先行 48% / 差し 31% / 逃げ 14% / 追い込み 7%",
  "未勝利": "先行 52% / 差し 28% / 逃げ 16% / 追い込み 4%",
};

export default function RaceQualityPage() {
  return (
    <PageShell title="レース品質分析" description="コース別・条件別統計ダッシュボード（AN-02）">
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">レース名</th>
              <th className="py-2 px-3 text-center">場所</th>
              <th className="py-2 px-3 text-center">馬場</th>
              <th className="py-2 px-3 text-right">距離 (m)</th>
              <th className="py-2 px-3 text-right">頭数</th>
              <th className="py-2 px-3 text-center">グレード</th>
              <th className="py-2 px-3 text-center">ペース傾向</th>
              <th className="py-2 px-3 text-left">脚質分布</th>
            </tr>
          </thead>
          <tbody>
            {MOCK_RACES.map((r) => (
              <tr key={r.race_id} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3 font-medium">{r.race_name}</td>
                <td className="py-2 px-3 text-center">{r.venue}</td>
                <td className="py-2 px-3 text-center">{r.surface}</td>
                <td className="py-2 px-3 text-right">{r.distance}</td>
                <td className="py-2 px-3 text-right">{r.field_size}</td>
                <td className="py-2 px-3 text-center">
                  <span
                    className="rounded px-2 py-0.5 text-xs font-semibold"
                    style={{
                      background: r.grade === "GI" ? "rgba(245,158,11,0.2)" : r.grade === "GIII" ? "rgba(59,130,246,0.2)" : "rgba(107,125,149,0.2)",
                      color: r.grade === "GI" ? "var(--warn)" : r.grade === "GIII" ? "var(--accent)" : "var(--text-dim)",
                    }}
                  >
                    {r.grade}
                  </span>
                </td>
                <td className="py-2 px-3 text-center">{PACE_MAP[r.grade] ?? "ミドル"}</td>
                <td className="py-2 px-3 text-xs" style={{ color: "var(--text-dim)" }}>{STYLE_MAP[r.grade] ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
