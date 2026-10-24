/** 表の列ソート用の値の解釈と比較（DOM に依存しない部分）。画面側は components/TableSorter.tsx。 */

const NULLS = new Set(["", "—", "–", "-", "－", "ー", "−", "N/A", "n/a", "🔒", "未", "なし"]);
export const MARK_RANK: Record<string, number> = { "◎": 0, "○": 1, "▲": 2, "★": 3, "△": 4, "消": 5 };

export type Cell = { n: number | null; s: string; empty: boolean; numericLike: boolean };

const norm = (t: string) => t.normalize("NFKC").replace(/\s+/g, " ").trim();

/** セルの文字を、並べ替え用の数値（取れれば）と文字列にする。時間 1:56.1 は秒、範囲 1.9〜2.8 は先頭の値、%・倍・円・¥・番手・カンマは無視、印は ◎○▲★△消 の順 */
export function parseCell(raw: string, sortAttr?: string | null): Cell {
  const t = norm(sortAttr ?? raw);
  if (NULLS.has(t)) return { n: null, s: "", empty: true, numericLike: false };
  if (t in MARK_RANK) return { n: MARK_RANK[t], s: t, empty: false, numericLike: true };
  const time = t.match(/^(\d+):(\d{1,2}(?:\.\d+)?)$/);
  if (time) return { n: Number(time[1]) * 60 + Number(time[2]), s: t, empty: false, numericLike: true };
  const u = t.replace(/,/g, "");
  // 数値らしい形: 前後に短い記号・単位だけ（例: 77.7% / 1.6倍 / ¥1000 / 3番手 / +0.9 / 1.9~2.8）
  const m = u.match(/^[^\d+\-.]{0,2}([+-]?\d+(?:\.\d+)?)(?:\s*[~〜\-–]\s*[+-]?\d+(?:\.\d+)?)?[^\d]{0,4}$/);
  if (m) return { n: Number(m[1]), s: t, empty: false, numericLike: true };
  return { n: null, s: t, empty: false, numericLike: false };
}

/** 列の取れた値のうち 8 割以上が数値なら数値列として並べる */
export function isNumericColumn(cells: Cell[]): boolean {
  const filled = cells.filter((c) => !c.empty);
  return filled.length > 0 && filled.filter((c) => c.numericLike).length / filled.length >= 0.8;
}

/** 比較。値なしは昇順でも降順でも最後。dir=1 昇順 / -1 降順 */
export function compareCells(a: Cell, b: Cell, numeric: boolean, dir: 1 | -1): number {
  if (a.empty && b.empty) return 0;
  if (a.empty) return 1;
  if (b.empty) return -1;
  const c = numeric ? (a.n ?? Infinity) - (b.n ?? Infinity) : a.s.localeCompare(b.s, "ja");
  return Number.isNaN(c) ? 0 : c * dir;
}
