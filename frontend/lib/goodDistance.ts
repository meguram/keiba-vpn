import type { RunningMapExtra } from "@/lib/raceAnalysis";

/** 距離（m）の表示。10m 単位に丸め、3 桁区切り（例 1,600m） */
export const fmtM = (v: number) => `${(Math.round(v / 10) * 10).toLocaleString()}m`;

/** 好走距離レンジの文言。3 走未満・走が無いときは「不明」 */
export function goodDistanceText(e: RunningMapExtra): string {
  const n = e.runs ?? 0;
  if (e.short_m == null || e.long_m == null) return `不明（過去の走 ${n} 走。3 走以上で判定）`;
  return e.short_m === e.long_m ? `${fmtM(e.short_m)}（過去の走 ${n} 走）` : `${fmtM(e.short_m)}〜${fmtM(e.long_m)}（過去の走 ${n} 走）`;
}
