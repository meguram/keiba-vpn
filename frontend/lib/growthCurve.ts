/* 成長曲線の馬体重まわりの純粋関数（描画から切り出してテストする）。 */

export type WeightPoint = { date: string; race_id: string; weight: number; weight_change?: number | null; finish?: number | string | null };

/** 右軸（馬体重 kg）の範囲。min/max に余白を付け、目盛りの刻みに丸める。 */
export function weightAxis(weights: { weight: number }[]): { lo: number; hi: number; ticks: number[] } | null {
  const ws = weights.map((w) => w.weight).filter((v) => Number.isFinite(v) && v > 0);
  if (!ws.length) return null;
  const mn = Math.min(...ws), mx = Math.max(...ws);
  const span = Math.max(mx - mn, 10);
  const step = span > 60 ? 20 : span > 24 ? 10 : 5;
  const pad = Math.max(step / 2, span * 0.15);
  const lo = Math.floor((mn - pad) / step) * step, hi = Math.ceil((mx + pad) / step) * step;
  const ticks: number[] = [];
  for (let v = lo; v <= hi; v += step) ticks.push(v);
  return { lo, hi, ticks };
}

/** 折れ線を切る場所を考慮した区間分割。breakDates（体重の無い走の日付 YYYYMMDD）が 2 点の間にあれば、そこで線を切る。 */
export function weightSegments<T extends { date: string }>(weights: T[], breakDates: string[]): T[][] {
  const segs: T[][] = [];
  let cur: T[] = [];
  weights.forEach((w, i) => {
    if (i > 0) {
      const prev = weights[i - 1].date;
      if (breakDates.some((d) => d > prev && d < w.date)) { segs.push(cur); cur = []; }
    }
    cur.push(w);
  });
  if (cur.length) segs.push(cur);
  return segs;
}

/** 前走比の表示。0 は ±0、符号は + / −。読めなければ空文字。 */
export function fmtWeightChange(c: number | null | undefined): string {
  if (c == null || !Number.isFinite(c)) return "";
  const r = Math.round(c);
  return r === 0 ? "±0" : `${r > 0 ? "+" : "−"}${Math.abs(r)}`;
}

/** 点列を SVG の path にする（1 点だけの区間は描かない＝丸だけ出す）。 */
export function pathOf(pts: { x: number; y: number }[]): string {
  return pts.length < 2 ? "" : pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

/** 馬体重の棒の色の区分（前走比: 増=up、減=down、変わらず・不明=flat）。 */
export function weightBarKind(c: number | null | undefined): "up" | "down" | "flat" {
  if (c == null || !Number.isFinite(c) || Math.round(c) === 0) return "flat";
  return c > 0 ? "up" : "down";
}

/** 棒の幅（px）。隣の棒との最小間隔の 62%、3〜12px に収める（1本だけなら 10px）。 */
export function barWidth(xs: number[]): number {
  const a = [...xs].sort((p, q) => p - q);
  let gap = Infinity;
  for (let i = 1; i < a.length; i++) gap = Math.min(gap, a[i] - a[i - 1]);
  return Math.max(3, Math.min(12, Number.isFinite(gap) ? gap * 0.62 : 10));
}
