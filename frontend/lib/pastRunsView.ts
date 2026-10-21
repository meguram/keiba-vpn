import type { RecentRun } from "@/lib/expectedMegu";
import { md } from "@/lib/expectedMegu";

/* 馬柱の 1 行表示（指数の推移の棒グラフ＋前走の短文）の純粋関数。 */

export const BAR_LO = 20;
export const BAR_HI = 80;
const BAR_MIN_PCT = 8;

/** 棒の高さ（%）。同じレース内で共通のスケール（指数 20〜80 を 0〜100%）。範囲外は丸め、最低でも見える高さを残す。 */
export function barHeightPct(value: number, lo = BAR_LO, hi = BAR_HI): number {
  if (!Number.isFinite(value) || hi <= lo) return BAR_MIN_PCT;
  const p = ((value - lo) / (hi - lo)) * 100;
  return Math.round(Math.max(BAR_MIN_PCT, Math.min(100, p)));
}

export type BarTone = "clear" | "below" | "plain";

/** 棒の色の区分。突破ライン以上=clear（緑）、未満=below（灰）、基準が無い=plain。 */
export function barTone(value: number, line?: number | null): BarTone {
  if (line == null || !Number.isFinite(line)) return "plain";
  return value >= line ? "clear" : "below";
}

export type FinishMark = "gold" | "blue" | "none";

/** 着順の小さな記号。1着=金、2〜3着=青、ほかは無印。 */
export function finishMark(finish?: number | null): FinishMark {
  return finish === 1 ? "gold" : finish === 2 || finish === 3 ? "blue" : "none";
}

/** 棒グラフに並べる走（古い→新しい）。API の recent は新しい順。 */
export function chronological(recent: RecentRun[] | undefined, max = 5): RecentRun[] {
  return [...(recent ?? [])].slice(0, max).reverse();
}

/** その馬の最高の走の位置（chronological の添字）。走が 2 未満なら -1（強調しない）。同値は新しい方。 */
export function bestIndex(runs: RecentRun[]): number {
  if (runs.length < 2) return -1;
  return runs.reduce((bi, r, i) => (r.value >= runs[bi].value ? i : bi), 0);
}

export type PrevSummary = { text: string; grade: string | null };

/** 前走の短文: 『前走 7/11 2歳未勝利 1着・8人気』。グレードは G1〜G3 のときだけ別に返す。 */
export function prevSummary(r: RecentRun | undefined): PrevSummary | null {
  if (!r) return null;
  const head = [`前走 ${md(r.date)}`, r.race_name].filter(Boolean).join(" ");
  const tail = [r.finish ? `${r.finish}着` : "", r.pop ? `${r.pop}人気` : ""].filter(Boolean).join("・");
  const grade = r.grade && /^G[123]$/.test(r.grade) ? r.grade : null;
  return { text: tail ? `${head} ${tail}` : head, grade };
}

/** 棒 1 本の読み上げ・ツールチップ用の説明。 */
export function barLabel(r: RecentRun): string {
  return [md(r.date), r.race_name, r.finish ? `${r.finish}着` : "", `指数${r.value.toFixed(0)}`, r.low ? "低信頼" : ""].filter(Boolean).join(" ");
}
