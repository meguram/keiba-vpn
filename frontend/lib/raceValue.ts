"use client";

import { useEffect, useState } from "react";
import { USE_MOCK, MOCK_WEEKLY_RACES, getMockBetRecommendation } from "@/lib/mock";

/** レースの「価値」= 推奨買い目の期待回収率（バックエンド delegates.race_value_summary と同じ区分） */
export type ValueLevel = "hot" | "good" | "ok" | "pass" | "none";

export type RaceValue = {
  race_id: string;
  level: ValueLevel;
  expected_roi: number | null;
  max_ev?: number | null;
  n_bets?: number;
  total_bet?: number;
  rank?: number | null;
  top_bet?: { bet_type: string; label: string; ev: number } | null;
  reason?: string;
};

export const LEVEL_STYLE: Record<ValueLevel, { label: string; hint: string; color: string; bg: string; border: string; glow?: string }> = {
  hot: { label: "★ 特に高い", hint: "期待回収率 150% 以上かつその日の上位 20%", color: "#fff", bg: "linear-gradient(135deg,#f97316,#ef4444)", border: "#fb923c", glow: "0 0 10px rgba(249,115,22,.65)" },
  good: { label: "高い", hint: "期待回収率 125% 以上", color: "#bbf7d0", bg: "rgba(34,197,94,0.22)", border: "#22c55e" },
  ok: { label: "プラス期待", hint: "期待回収率 105% 以上", color: "#bfdbfe", bg: "rgba(96,165,250,0.14)", border: "#60a5fa" },
  pass: { label: "見送り水準", hint: "期待回収率 105% 未満、または買い目なし", color: "#6b7d95", bg: "transparent", border: "#243049" },
  none: { label: "未計算", hint: "予測またはオッズが無い", color: "#566178", bg: "transparent", border: "#243049" },
};

const BET_JA: Record<string, string> = { tansho: "単勝", fukusho: "複勝", umaren: "馬連", wide: "ワイド", umatan: "馬単", sanrenpuku: "3連複" };

export function valueTitle(v: RaceValue | undefined): string {
  if (!v) return "";
  const s = LEVEL_STYLE[v.level];
  if (v.level === "none") return `${s.label}${v.reason ? `（${v.reason}）` : ""}`;
  const roi = v.expected_roi != null ? `推奨の期待回収率 ${(v.expected_roi * 100).toFixed(0)}%` : "買い目なし";
  const top = v.top_bet ? ` ／ 最有力 ${BET_JA[v.top_bet.bet_type] ?? v.top_bet.bet_type} ${v.top_bet.label}（期待値 ${v.top_bet.ev.toFixed(2)}）` : "";
  const rank = v.rank ? ` ／ その日 ${v.rank} 位` : "";
  return `${s.label}｜${roi}${top}${rank}`;
}

function levelOf(roi: number | null, n: number): ValueLevel {
  if (!n || roi == null) return "pass";
  return roi >= 1.5 ? "hot" : roi >= 1.25 ? "good" : roi >= 1.05 ? "ok" : "pass";
}

function mockValues(): RaceValue[] {
  const rows: RaceValue[] = MOCK_WEEKLY_RACES.map((r) => {
    const rec = getMockBetRecommendation(r.race_id, 100000, "auto", "latest", 2000);
    const roi = rec.total_bet > 0 ? rec.expected_return / rec.total_bet : null;
    const top = [...rec.candidates].sort((a, b) => b.ev - a.ev)[0];
    return {
      race_id: r.race_id, level: levelOf(roi, rec.candidates.length), expected_roi: roi, n_bets: rec.candidates.length, total_bet: rec.total_bet,
      top_bet: top ? { bet_type: top.bet_type, label: top.pair_label, ev: top.ev } : null,
    };
  });
  const ranked = rows.filter((r) => r.expected_roi != null).sort((a, b) => (b.expected_roi ?? 0) - (a.expected_roi ?? 0));
  const cutoff = Math.max(1, Math.ceil(ranked.length * 0.2));
  ranked.forEach((r, i) => { r.rank = i + 1; if (r.level === "hot" && i + 1 > cutoff) r.level = "good"; });
  return rows;
}

const cache = new Map<string, { at: number; map: Map<string, RaceValue> }>();
const TTL_MS = 30_000;

async function fetchValues(date: string): Promise<Map<string, RaceValue>> {
  const hit = cache.get(date);
  if (hit && Date.now() - hit.at < TTL_MS) return hit.map;
  let rows: RaceValue[] = [];
  if (USE_MOCK) {
    rows = mockValues();
  } else {
    // ログイン必須。未ログイン（401）や失敗のときは強調表示なし
    const res = await fetch(`/api/v1/betting/race-values?date=${date}`, { credentials: "include" });
    if (res.ok) rows = (await res.json()).races ?? [];
  }
  const map = new Map(rows.map((r) => [r.race_id, r]));
  cache.set(date, { at: Date.now(), map });
  return map;
}

/** 開催日（YYYYMMDD）の各レースの価値。取得できるまでは空。 */
export function useRaceValues(date: string): Map<string, RaceValue> {
  const [map, setMap] = useState<Map<string, RaceValue>>(new Map());
  useEffect(() => {
    let alive = true;
    if (!date && !USE_MOCK) return;
    fetchValues(date).then((m) => alive && setMap(m)).catch(() => alive && setMap(new Map()));
    return () => { alive = false; };
  }, [date]);
  return map;
}
