/** めぐ分析の「過去データの条件（通常）」タブの型と小さな関数。値は src/research/race_analysis/course_rules.py が作る（全レース版の凍結の第 3 版）。 */

export type CRBase = { place: number | null; roi_t: number | null; roi_f: number | null };

export type CRStat = {
  n: number;
  /** [1 着, 2 着, 3 着, 着外（4 着以下）] */
  rec: number[];
  place: number | null;
  roi_t: number | null;
  roi_f: number | null;
  /** 払戻の解けた走と、単・複の払戻の和（年を合わせるため） */
  npay?: number;
  tan?: number;
  fuku?: number;
  /** 条件に当たった馬と同じ人気の馬の値（比べる相手） */
  base_same_pop?: CRBase | null;
};

export type CRPeriods = { train: CRStat | null; "2025": CRStat | null; "2026": CRStat | null };

export type CRRule = { text: string; hyp?: string | null; edge?: number | null; p_null?: number | null; stats: CRPeriods };

export type CRFigure = {
  unit: string;
  unit_label?: string;
  layer: string | null;
  n_buy: number;
  n_kill: number;
  n_buy_hit?: number;
  rules: { buy: CRRule[]; kill: CRRule[] };
  unit_stats: CRPeriods;
  periods: { train: number[]; "2025": number[]; "2026": number[] } | null;
  data_until: string | null;
  cycle: number | null;
  track: string | null;
  track_ok: boolean;
  includes_after?: boolean;
};

/** これ未満の件数の値は強調しない（1 回の高配当で大きく動くため） */
export const MIN_N = 30;

/** 2 つの年を合わせる。払戻率は払戻の解けた走で割り直し、比べる相手は件数で重みづけする。 */
export function combine(a: CRStat | null, b: CRStat | null): CRStat | null {
  const xs = [a, b].filter((x): x is CRStat => !!x);
  if (!xs.length) return null;
  const n = xs.reduce((s, x) => s + x.n, 0);
  const rec4 = [0, 1, 2, 3].map((i) => xs.reduce((s, x) => s + (x.rec?.[i] ?? 0), 0));
  const npay = xs.reduce((s, x) => s + (x.npay ?? 0), 0);
  const tan = xs.reduce((s, x) => s + (x.tan ?? 0), 0);
  const fuku = xs.reduce((s, x) => s + (x.fuku ?? 0), 0);
  const wb = (k: keyof CRBase) => {
    const v = xs.filter((x) => x.base_same_pop?.[k] != null && x.n > 0);
    const w = v.reduce((s, x) => s + x.n, 0);
    return w ? v.reduce((s, x) => s + x.n * (x.base_same_pop![k] as number), 0) / w : null;
  };
  return {
    n, rec: rec4, place: n ? (rec4[0] + rec4[1] + rec4[2]) / n : null,
    roi_t: npay ? tan / npay : null, roi_f: npay ? fuku / npay : null, npay, tan, fuku,
    base_same_pop: n ? { place: wb("place"), roi_t: wb("roi_t"), roi_f: wb("roi_f") } : null,
  };
}
