"use client";

import { useState } from "react";
import { fetchApi, PredictionsResponse } from "@/lib/api";
import { PageShell } from "@/components/PageShell";

export default function RaceDetailPage({ params }: { params: { id: string } }) {
  const [tab, setTab] = useState<"shutuba" | "result" | "ai" | "horse">("shutuba");
  const [race, setRace] = useState<Record<string, unknown> | null>(null);
  const [pred, setPred] = useState<PredictionsResponse | null>(null);

  async function load() {
    const [r, p] = await Promise.all([
      fetchApi<Record<string, unknown>>(`/api/v1/races/${params.id}`),
      fetchApi<PredictionsResponse>(`/api/v1/races/${params.id}/predictions`).catch(() => null),
    ]);
    setRace(r);
    setPred(p);
  }

  return (
    <PageShell title={`レース ${params.id}`}>
      <button type="button" className="btn mb-4" onClick={load}>読込</button>
      <div className="mb-4 flex gap-2">
        {(["shutuba", "result", "ai", "horse"] as const).map((t) => (
          <button key={t} type="button" className="rounded border px-3 py-1 text-sm" style={{ borderColor: tab === t ? "var(--accent)" : "var(--border)" }} onClick={() => setTab(t)}>
            {t === "shutuba" ? "出馬表" : t === "result" ? "結果" : t === "ai" ? "AI予測" : "出走馬詳細"}
          </button>
        ))}
      </div>
      {tab === "shutuba" && race && (
        <pre className="card overflow-auto text-xs">{JSON.stringify((race.entries as unknown[]) || [], null, 2)}</pre>
      )}
      {tab === "ai" && pred && (
        <table className="w-full text-sm">
          <thead><tr><th>馬番</th><th>勝率</th><th>単ROI</th><th>複ROI</th><th>VB</th></tr></thead>
          <tbody>
            {pred.horses.map((h) => (
              <tr key={h.horse_id} style={{ background: h.is_value_bet ? "rgba(34,197,94,0.15)" : undefined }}>
                <td>{h.post_no}</td>
                <td>{h.win_prob?.toFixed(3)}</td>
                <td>{h.expected_win_roi}</td>
                <td>{h.expected_show_roi}</td>
                <td>{h.is_value_bet ? "✓" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </PageShell>
  );
}
