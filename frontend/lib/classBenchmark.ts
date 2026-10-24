import type { ClassBenchmark, RecentRun } from "@/lib/expectedMegu";

/* クラスの基準（突破ライン）の表示まわりの純粋関数。基準は『そのクラスの2着馬の指数』の分布から作った目安。 */

const BAND_RANGE: Record<string, string> = { 短: "〜1300m", 中短: "1301〜1600m", 中: "1601〜2000m", 長: "2001m〜" };

/** 『2歳未勝利・芝』『3歳以上1勝クラス・ダ・1301〜1600m』 */
export function benchmarkTitle(b: ClassBenchmark): string {
  const parts = [b.label];
  if (b.surface) parts.push(b.surface);
  if (b.band) parts.push(BAND_RANGE[b.band] ?? b.band);
  return parts.join("・");
}

export const clearsLine = (value: number | null | undefined, b?: ClassBenchmark | null): boolean => value != null && !!b && value >= b.line;

export type PassState = { kind: "above" | "below" | "none"; best: number | null; gap: number | null };

/** 渡された走（呼び出し側で同条件の走に絞る）の最高指数が突破ライン以上か。走が無い・基準が無いときは none。 */
export function passState(runs: RecentRun[] | undefined, b?: ClassBenchmark | null): PassState {
  const vals = (runs ?? []).slice(0, 5).map((r) => r.value).filter((v) => Number.isFinite(v));
  if (!b || !vals.length) return { kind: "none", best: vals.length ? Math.max(...vals) : null, gap: null };
  const best = Math.max(...vals);
  return { kind: best >= b.line ? "above" : "below", best, gap: best - b.line };
}

export function benchmarkSummary(b: ClassBenchmark): string {
  return `2着馬の中央値 ${b.median.toFixed(0)} ／ 突破ライン ${b.line.toFixed(0)}（${b.period ?? ""}・n=${b.n}）`;
}
