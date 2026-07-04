"use client";

import { useState } from "react";
import { PageShell } from "@/components/PageShell";

export default function BettingPage() {
  const [raceId, setRaceId] = useState("");
  const [result, setResult] = useState<string>("");

  async function optimize() {
    const res = await fetch("/api/v1/betting/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ race_id: raceId, bankroll: 100000 }),
    });
    setResult(JSON.stringify(await res.json(), null, 2));
  }

  return (
    <PageShell title="馬券最適化" description="Kelly 基準ポートフォリオ（ログイン必須・AN-13）">
      <div className="card space-y-3">
        <input className="w-full rounded border bg-transparent p-2" style={{ borderColor: "var(--border)" }} placeholder="race_id" value={raceId} onChange={(e) => setRaceId(e.target.value)} />
        <button type="button" className="btn" onClick={optimize}>最適化</button>
        {result && <pre className="overflow-auto text-xs">{result}</pre>}
      </div>
    </PageShell>
  );
}
