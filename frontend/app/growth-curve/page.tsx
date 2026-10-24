"use client";

import { useState } from "react";
import { PageShell } from "@/components/PageShell";
import { MOCK_GROWTH_CURVE } from "@/lib/mock";

export default function GrowthCurvePage() {
  const [selectedId, setSelectedId] = useState(MOCK_GROWTH_CURVE[0].horse_id);
  const horse = MOCK_GROWTH_CURVE.find((h) => h.horse_id === selectedId)!;

  return (
    <PageShell title="成長曲線" description="馬体重×タイム指数・レース間隔×タイム指数（AN-06）">
      <div className="card space-y-4">
        <div className="flex items-center gap-3">
          <label htmlFor="horse-select" className="text-sm" style={{ color: "var(--text-dim)" }}>馬選択</label>
          <select
            id="horse-select"
            className="rounded border bg-transparent px-3 py-1.5 text-sm"
            style={{ borderColor: "var(--border)", background: "var(--surface2)", color: "var(--text)" }}
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {MOCK_GROWTH_CURVE.map((h) => (
              <option key={h.horse_id} value={h.horse_id}>{h.horse_name}</option>
            ))}
          </select>
        </div>
        <div
          className="flex items-center justify-center rounded-lg"
          style={{ background: "var(--surface2)", height: 180, color: "var(--text-dim)", fontSize: 13 }}
        >
          Chart.js 折れ線グラフ（馬体重 × タイム指数）— フル実装時に描画
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">レース番号</th>
              <th className="py-2 px-3 text-right">タイム指数</th>
              <th className="py-2 px-3 text-right">馬体重 (kg)</th>
            </tr>
          </thead>
          <tbody>
            {horse.data.map((d) => (
              <tr key={d.race_no} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3">第 {d.race_no} 戦</td>
                <td className="py-2 px-3 text-right font-mono">{d.time_index}</td>
                <td className="py-2 px-3 text-right font-mono">{d.body_weight}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
