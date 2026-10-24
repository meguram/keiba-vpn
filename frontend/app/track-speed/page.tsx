"use client";

import { useState } from "react";
import { PageShell } from "@/components/PageShell";
import { MOCK_TRACK_SPEED } from "@/lib/mock";

const VENUES = ["全て", "東京", "阪神", "中京"];

export default function TrackSpeedPage() {
  const [venue, setVenue] = useState("全て");
  const rows = venue === "全て" ? MOCK_TRACK_SPEED : MOCK_TRACK_SPEED.filter((r) => r.venue === venue);

  return (
    <PageShell title="馬場速度指数" description="TSI（Track Speed Index）— 馬場コンディション別集計（AN-07）">
      <div className="card space-y-4">
        <div className="flex flex-wrap gap-2">
          {VENUES.map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setVenue(v)}
              className="rounded-full px-3 py-1 text-sm"
              style={{
                background: venue === v ? "var(--accent)" : "var(--surface2)",
                color: venue === v ? "white" : "var(--text-dim)",
                border: "1px solid",
                borderColor: venue === v ? "var(--accent)" : "var(--border)",
              }}
            >
              {v}
            </button>
          ))}
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--text-dim)" }}>
              <th className="py-2 px-3 text-left">日付</th>
              <th className="py-2 px-3 text-center">場所</th>
              <th className="py-2 px-3 text-center">馬場</th>
              <th className="py-2 px-3 text-center">馬場状態</th>
              <th className="py-2 px-3 text-right">TSI</th>
              <th className="py-2 px-3 text-right">含水率 (%)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.date}-${r.venue}-${r.surface}`} style={{ borderTop: "1px solid var(--border)" }}>
                <td className="py-2 px-3 font-mono text-xs">{r.date}</td>
                <td className="py-2 px-3 text-center">{r.venue}</td>
                <td className="py-2 px-3 text-center">{r.surface}</td>
                <td className="py-2 px-3 text-center">{r.going}</td>
                <td
                  className="py-2 px-3 text-right font-semibold"
                  style={{ color: r.tsi >= 100 ? "var(--ok)" : r.tsi >= 97 ? "var(--warn)" : "var(--err)" }}
                >
                  {r.tsi.toFixed(1)}
                </td>
                <td className="py-2 px-3 text-right text-xs" style={{ color: "var(--text-dim)" }}>
                  {r.moisture_pct.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
