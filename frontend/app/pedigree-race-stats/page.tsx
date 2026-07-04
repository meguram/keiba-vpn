"use client";

import { useState } from "react";
import { PageShell } from "@/components/PageShell";
import { MOCK_SIRES } from "@/lib/mock";

export default function PedigreeRaceStatsPage() {
  const [query, setQuery] = useState("");

  const rows = query
    ? MOCK_SIRES.filter((s) => s.sire_name.includes(query))
    : MOCK_SIRES;

  return (
    <PageShell title="種牡馬成績分析" description="多軸フィルタリング（AN-01）">
      <div className="card space-y-4">
        <div className="flex flex-wrap gap-3">
          <input
            type="search"
            placeholder="種牡馬名で検索…"
            className="rounded border bg-transparent px-3 py-1.5 text-sm"
            style={{ borderColor: "var(--border)", color: "var(--text)", minWidth: 200 }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select
            className="rounded border bg-transparent px-3 py-1.5 text-sm"
            style={{ borderColor: "var(--border)", background: "var(--surface2)", color: "var(--text)" }}
            disabled
          >
            <option>コース（モック中は無効）</option>
          </select>
          <select
            className="rounded border bg-transparent px-3 py-1.5 text-sm"
            style={{ borderColor: "var(--border)", background: "var(--surface2)", color: "var(--text)" }}
            disabled
          >
            <option>クラス（モック中は無効）</option>
          </select>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">種牡馬</th>
              <th className="py-2 px-3 text-right">勝率</th>
              <th className="py-2 px-3 text-right">連対率</th>
              <th className="py-2 px-3 text-right">芝出走率</th>
              <th className="py-2 px-3 text-right">最良距離</th>
              <th className="py-2 px-3 text-right">最良馬場</th>
              <th className="py-2 px-3 text-right">サンプル数</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.sire_id} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3 font-medium">{s.sire_name}</td>
                <td className="py-2 px-3 text-right">{(s.win_rate * 100).toFixed(1)}%</td>
                <td className="py-2 px-3 text-right">{(s.place_rate * 100).toFixed(1)}%</td>
                <td className="py-2 px-3 text-right">{(s.turf_rate * 100).toFixed(1)}%</td>
                <td className="py-2 px-3 text-right">{s.best_distance} m</td>
                <td className="py-2 px-3 text-right">{s.best_going}</td>
                <td className="py-2 px-3 text-right" style={{ color: "var(--text-dim)" }}>{s.sample_n.toLocaleString()}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="py-4 text-center" style={{ color: "var(--text-dim)" }}>該当なし</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
