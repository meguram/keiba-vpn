import type { MeguMark, RecentRun } from "@/lib/expectedMegu";
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

export type BarTone = "clear" | "below" | "plain" | "off";

export type TodayCourse = { surface?: string | null; distance?: number | string | null };

const surfaceKind = (s?: string | null): "芝" | "ダ" | null => (s ? (s.startsWith("ダ") ? "ダ" : s.startsWith("芝") ? "芝" : null) : null);

/** 比べる距離の幅（m）。今回が 2400m 以上なら ±400m、2200m 以下（2300m を含む）なら ±200m。 */
export const distanceWindow = (distance: number): number => (distance >= 2400 ? 400 : 200);

/** その走が、今回のレースと比べる対象か（同じ芝/ダートで、距離の差が幅以内）。今回の条件が不明なら true（絞らない）。走の条件が不明なら false。 */
export function isComparable(r: RecentRun, today?: TodayCourse | null): boolean {
  const ts = surfaceKind(today?.surface ?? null);
  const td = Number(today?.distance);
  if (!ts || !Number.isFinite(td) || td <= 0) return true;
  const rs = surfaceKind(r.surface ?? null);
  if (!rs || r.distance == null || !Number.isFinite(r.distance)) return false;
  return rs === ts && Math.abs(r.distance - td) <= distanceWindow(td);
}

/** 『芝1600』のようなコース表記。 */
export const courseText = (r: RecentRun): string => (r.surface ? `${r.surface.slice(0, 1)}${r.distance ?? ""}` : r.distance ? `${r.distance}` : "");

/** 棒の色の区分。突破ライン以上=clear（緑）、未満=below（灰）、基準が無い=plain。 */
export function barTone(value: number, line?: number | null, comparable = true): BarTone {
  if (!comparable) return "off";
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
export function bestIndex(runs: RecentRun[], pick: (r: RecentRun) => boolean = () => true): number {
  const ids = runs.map((_, i) => i).filter((i) => pick(runs[i]));
  if (ids.length < 2) return -1;
  return ids.reduce((bi, i) => (runs[i].value >= runs[bi].value ? i : bi), ids[0]);
}

export type PrevSummary = { text: string; grade: string | null };

/** 前走の短文: 『前走 7/11 2歳未勝利 1着』。グレードは G1〜G3 のときだけ別に返す。 */
export function prevSummary(r: RecentRun | undefined): PrevSummary | null {
  if (!r) return null;
  const head = [`前走 ${md(r.date)}`, r.cls ?? r.race_name].filter(Boolean).join(" ");
  const tail = r.finish ? `${r.finish}着` : "";
  const grade = r.grade && /^G[123]$/.test(r.grade) ? r.grade : null;
  return { text: tail ? `${head} ${tail}` : head, grade };
}

/** 棒 1 本の読み上げ・ツールチップ用の説明。 */
export function barLabel(r: RecentRun): string {
  return [md(r.date), r.cls ?? r.race_name, r.finish ? `${r.finish}着` : "", `指数${r.value.toFixed(0)}`, r.low ? "低信頼" : ""].filter(Boolean).join(" ");
}

export type MeguSortKey = "num" | "mark" | "mean" | "best" | "clear";

/** めぐ印の並べ替え値（大きいほど上）。印あり: ◎=2、○=1（同じ印の中は集約値の高い順）。印なしでも集約値があれば 0 台、判定不能は null。 */
export function markSortValue(m?: MeguMark | null): number | null {
  if (!m || m.value == null) return null;
  const tier = m.key === "top" ? 2 : m.key === "good" ? 1 : 0;
  return tier * 1000 + m.value;
}

/** 印の説明（ツールチップ・詳細用の 1〜2 行）。 */
export function markExplain(m?: MeguMark | null): string {
  if (!m) return "めぐ印: 判定なし";
  if (m.key) return `${m.label ?? ""} ${m.reason ?? ""}`.trim();
  return m.unavailable ?? m.reason ?? "印なし";
}

/** めぐ指数テーブルの並べ替え値。mean=直近の平均、best=直近の最高、clear=今回と比べられる走の最高 − 突破ライン。値が出せないときは null（並べ替えでは末尾）。 */
export function meguSortValue(recent: RecentRun[] | undefined, key: Exclude<MeguSortKey, "num" | "mark">, line?: number | null, today?: TodayCourse | null, max = 5): number | null {
  const runs = (recent ?? []).slice(0, max);
  if (!runs.length) return null;
  if (key === "mean") return runs.reduce((s, r) => s + r.value, 0) / runs.length;
  if (key === "best") return Math.max(...runs.map((r) => r.value));
  const cmp = runs.filter((r) => isComparable(r, today));
  if (!cmp.length || line == null || !Number.isFinite(line)) return null;
  return Math.max(...cmp.map((r) => r.value)) - line;
}
