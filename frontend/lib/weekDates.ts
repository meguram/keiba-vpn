/**
 * 週間予測ページの開催日選択で使う、日付まわりの純粋関数。
 * 日付は YYYYMMDD の文字列。『今日』は呼び出し側が JST で作って渡す（テストできるように）。
 */

/** 1 回の開催（土日と、続く月曜の祝日開催）。key はその開催の土曜（YYYYMMDD） */
export type RaceWeek = { key: string; dates: string[] };   // dates は昇順
export type WeekRel = "先週" | "今週" | "来週" | "";
export type PickedWeek = { week: RaceWeek; role: "prev" | "next"; rel: WeekRel };

export const WEEKDAY_JA = ["日", "月", "火", "水", "木", "金", "土"] as const;
export const COLOR_SAT = "#60a5fa";
export const COLOR_SUN = "#f87171";

function utc(ymd: string): Date {
  return new Date(Date.UTC(Number(ymd.slice(0, 4)), Number(ymd.slice(4, 6)) - 1, Number(ymd.slice(6, 8))));
}

export function toYmd(d: Date): string {
  return `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, "0")}${String(d.getUTCDate()).padStart(2, "0")}`;
}

/** 0=日 … 6=土 */
export function weekdayOf(ymd: string): number {
  return utc(ymd).getUTCDay();
}

export function addDays(ymd: string, n: number): string {
  const d = utc(ymd);
  d.setUTCDate(d.getUTCDate() + n);
  return toYmd(d);
}

/** その日を含むカレンダー週の月曜 */
export function mondayOf(ymd: string): string {
  return addDays(ymd, -((weekdayOf(ymd) + 6) % 7));
}

/**
 * その日が属する開催の土曜。日・月・火は直前の土曜に寄せる（月曜の祝日開催は前の土日と同じ開催）。
 * 水〜金に開催がある（まれな特別開催）場合は次の土曜に寄せる。
 */
export function meetingKeyOf(ymd: string): string {
  const w = weekdayOf(ymd);
  if (w === 6) return ymd;
  if (w === 0) return addDays(ymd, -1);
  if (w === 1) return addDays(ymd, -2);
  if (w === 2) return addDays(ymd, -3);
  return addDays(ymd, 6 - w);
}

/** 今日（JST）の YYYYMMDD。ブラウザのタイムゾーンに依らない */
export function todayJst(now: Date = new Date()): string {
  return toYmd(new Date(now.getTime() + 9 * 3600 * 1000));
}

export function groupWeeks(dates: string[]): RaceWeek[] {
  const map = new Map<string, string[]>();
  for (const d of [...new Set(dates)].sort()) {
    const m = meetingKeyOf(d);
    map.set(m, [...(map.get(m) ?? []), d]);
  }
  return [...map.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([key, ds]) => ({ key, dates: ds }));
}

/** 今日のカレンダー週に対して、その開催が先週/今週/来週か。開催中（今日が開催の最初の日〜最後の日）は今週 */
function relOf(week: RaceWeek, today: string): WeekRel {
  if (today >= week.dates[0] && today <= week.dates[week.dates.length - 1]) return "今週";
  const diff = Math.round((utc(mondayOf(week.key)).getTime() - utc(mondayOf(today)).getTime()) / (7 * 86400000));
  return diff === 0 ? "今週" : diff === 1 ? "来週" : diff === -1 ? "先週" : "";
}

/**
 * 『前回の開催週』と『次回の開催週』の最大 2 週を返す（昇順＝前回, 次回）。
 * 次回 = 今日以降に開催日がある最初の週（今日が開催日ならその週）。今日以降の開催が無ければ最後の週。前回 = その直前の開催週。
 * 開催週が 1 つしか無ければ 1 つだけ返す。
 */
export function pickTwoWeeks(dates: string[], today: string): PickedWeek[] {
  const weeks = groupWeeks(dates);
  if (weeks.length === 0) return [];
  let idx = weeks.findIndex((w) => w.dates[w.dates.length - 1] >= today);
  if (idx < 0) idx = weeks.length - 1;
  const out: PickedWeek[] = [];
  if (idx > 0) out.push({ week: weeks[idx - 1], role: "prev", rel: relOf(weeks[idx - 1], today) });
  out.push({ week: weeks[idx], role: "next", rel: relOf(weeks[idx], today) });
  return out;
}

/** 過去の開催だけを見るページ用: 今日以前の最新 2 開催（昇順＝その前, 直近）。今日が開催日ならその開催も含む（開催中・終了後を見られる） */
export function pickPastWeeks(dates: string[], today: string): PickedWeek[] {
  return groupWeeks(dates.filter((d) => d <= today)).slice(-2).map((week) => ({ week, role: "prev" as const, rel: relOf(week, today) }));
}

/** 開催の最初の開催日。開催中（今日が最初の日以降）なら、今日以降で最初の開催日（無ければ最初の日） */
export function firstDateOf(week: RaceWeek, today: string): string {
  return (today >= week.dates[0] ? week.dates.find((d) => d >= today) : undefined) ?? week.dates[0];
}

export type DayKind = { weekday: number; label: string; kind: "sat" | "sun" | "holiday" | "weekday"; color: string | undefined; holiday: boolean; isToday: boolean };

/** 土=青、日=赤、月〜金に開催がある日は祝日開催としてほぼ赤＋『祝』 */
export function dayKind(ymd: string, today: string): DayKind {
  const w = weekdayOf(ymd);
  const kind = w === 6 ? "sat" : w === 0 ? "sun" : "holiday";
  return {
    weekday: w, label: WEEKDAY_JA[w], kind,
    color: kind === "sat" ? COLOR_SAT : COLOR_SUN, holiday: kind === "holiday", isToday: ymd === today,
  };
}

export function formatMD(ymd: string): string {
  return `${Number(ymd.slice(4, 6))}/${Number(ymd.slice(6, 8))}`;
}

export function formatMDW(ymd: string): string {
  return `${formatMD(ymd)}(${WEEKDAY_JA[weekdayOf(ymd)]})`;
}

/** 『10/10(土)〜12(月)』。月をまたぐときは『9/26(土)〜10/4(日)』。1 日だけなら 1 日分 */
export function formatRange(dates: string[]): string {
  if (dates.length === 0) return "";
  const a = dates[0];
  const b = dates[dates.length - 1];
  if (a === b) return formatMDW(a);
  return `${formatMDW(a)}〜${a.slice(4, 6) === b.slice(4, 6) ? `${Number(b.slice(6, 8))}(${WEEKDAY_JA[weekdayOf(b)]})` : formatMDW(b)}`;
}

const GRADES = ["G1", "G2", "G3", "JG1", "JG2", "JG3", "J.G1", "J.G2", "J.G3"];

export type GradeRace = { race_id: string; grade: string; name: string; fullName: string; venue: string; round: number };

/** 『第12回サウジアラビアRC』→『サウジアラビアRC』。末尾の『(G3)』も除く */
export function shortRaceName(name: string): string {
  return name.replace(/^第\s*\d+回\s*/, "").replace(/[（(]\s*J?\.?G\d\s*[）)]\s*$/, "").trim();
}

/** 重賞（G1〜G3、障害の J.G も）だけを、開催場コード・R の順で返す */
export function pickGradeRaces(
  races: { race_id: string; race_name?: string; grade?: string; venue?: string; round?: number | string }[],
): GradeRace[] {
  return races
    .filter((r) => r.grade && GRADES.includes(r.grade))
    .map((r) => ({
      race_id: r.race_id, grade: String(r.grade), name: shortRaceName(r.race_name ?? r.race_id), fullName: r.race_name ?? r.race_id,
      venue: r.venue ?? "", round: Number(r.round ?? Number(r.race_id.slice(-2))),
    }))
    .sort((a, b) => a.race_id.slice(4, 6).localeCompare(b.race_id.slice(4, 6)) || a.round - b.round);
}

/** 開催場名を JRA のコード順（race_id の 5〜6 桁目）に並べて重複を除く */
export function venuesOf(races: { race_id: string; venue?: string }[]): string[] {
  const code = new Map<string, string>();
  for (const r of races) if (r.venue && !code.has(r.venue)) code.set(r.venue, r.race_id.slice(4, 6));
  return [...code.keys()].sort((a, b) => (code.get(a) ?? "").localeCompare(code.get(b) ?? ""));
}
