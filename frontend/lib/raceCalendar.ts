/** 過去の開催日カレンダー（/races）の日付まわりの純粋関数。日付は YYYYMMDD、月は YYYYMM。 */

export type CalDay = { date: string; n: number; venues: string[]; grades: { race_id: string; grade: string; name: string; venue: string; round: number | string }[] };

export function ymOf(ymd: string): string {
  return ymd.slice(0, 6);
}

export function shiftMonth(ym: string, n: number): string {
  const t = Number(ym.slice(0, 4)) * 12 + (Number(ym.slice(4, 6)) - 1) + n;
  return `${Math.floor(t / 12)}${String((t % 12) + 1).padStart(2, "0")}`;
}

/** [min, max]（ともに YYYYMM）の範囲に収める */
export function clampMonth(ym: string, min: string, max: string): string {
  return ym < min ? min : ym > max ? max : ym;
}

/** その月のカレンダー格子。週ごとに 7 要素（日曜始まり）。月外は null */
export function monthGrid(ym: string): (string | null)[][] {
  const y = Number(ym.slice(0, 4));
  const m = Number(ym.slice(4, 6));
  const first = new Date(Date.UTC(y, m - 1, 1)).getUTCDay();
  const days = new Date(Date.UTC(y, m, 0)).getUTCDate();
  const cells: (string | null)[] = [...Array(first).fill(null), ...Array.from({ length: days }, (_, i) => `${ym}${String(i + 1).padStart(2, "0")}`)];
  while (cells.length % 7) cells.push(null);
  return Array.from({ length: cells.length / 7 }, (_, w) => cells.slice(w * 7, w * 7 + 7));
}

/** 今日以前で最新の日付。無ければ空文字 */
export function latestOnOrBefore(dates: string[], today: string): string {
  return dates.filter((d) => d <= today).sort().pop() ?? "";
}

/** 年の選択肢（新しい順）。minYm〜maxYm（YYYYMM） */
export function yearsBetween(minYm: string, maxYm: string): number[] {
  const out: number[] = [];
  for (let y = Number(maxYm.slice(0, 4)); y >= Number(minYm.slice(0, 4)); y--) out.push(y);
  return out;
}

/** その年で選べる月（minYm〜maxYm の範囲内） */
export function monthsOfYear(year: number, minYm: string, maxYm: string): number[] {
  const out: number[] = [];
  for (let m = 1; m <= 12; m++) {
    const ym = `${year}${String(m).padStart(2, "0")}`;
    if (ym >= minYm && ym <= maxYm) out.push(m);
  }
  return out;
}

/** URL の ?date= を解釈する。未来なら直近の過去の日付に切り替え、切り替えたかを返す */
export function resolveDate(fromUrl: string, latest: string, today: string): { date: string; future: boolean } {
  if (!/^\d{8}$/.test(fromUrl)) return { date: latest, future: false };
  if (fromUrl > today) return { date: latest, future: true };
  return { date: fromUrl, future: false };
}
