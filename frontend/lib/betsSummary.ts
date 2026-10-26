"use client";

import { useEffect, useState } from "react";
import { USE_MOCK } from "@/lib/mock";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";

export type RaceBet = { total_bet: number; total_return: number; hit: boolean; roi: number | null; estimated: boolean };
export type RatedCounts = { settled: number; rated: number; skipped: number; unrated: number };
export type BetsBlock = { counts?: RatedCounts; invested: number; returned: number; roi: number | null; races_bought: number; races_hit: number; estimated_races: number; status: string };
export type BetsSummary = { date: string; label: string; week: BetsBlock & { from: string; to: string; days: string[] }; day: BetsBlock; races: Record<string, RaceBet>; note: string };

const cache = new Map<string, { at: number; v: BetsSummary }>();

/** 開催日（YYYYMMDD）の仮想購入の結果（各レースの的中・その日とその週の回収率）。有料会員のみ。取得できなければ null。 */
export function useBetsSummary(date: string): BetsSummary | null {
  const { isMember, loading } = useAuthStatus();
  const [v, setV] = useState<BetsSummary | null>(null);
  useEffect(() => {
    let alive = true;
    setV(null);
    if (!date || USE_MOCK || loading || !isMember) return;
    const hit = cache.get(date);
    if (hit && Date.now() - hit.at < 30_000) { setV(hit.v); return; }
    fetch(`/api/prediction-results/bets-summary?date=${date}`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d: BetsSummary | null) => { if (alive && d) { cache.set(date, { at: Date.now(), v: d }); setV(d); } })
      .catch(() => { if (alive) setV(null); });
    return () => { alive = false; };
  }, [date, loading, isMember]);
  return v;
}

export const yen = (n: number) => `¥${Math.round(n).toLocaleString("ja-JP")}`;
export const roiText = (r: number | null | undefined) => (r == null ? "—" : `${(r * 100).toFixed(0)}%`);
