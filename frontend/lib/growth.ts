/* 成長度（成長指数）の表示まわりの純粋関数。API: /api/race/{id}/predictions の predictions[].growth と growth_meta。
   指数の単位は指数ポイント（1点＝0.1秒）。『期待より』= その馬の直近の水準から統計的に期待される上がり幅との差。 */

export type GrowthStage = 0 | 1 | 2 | 3 | 4;

export type Growth = {
  index: number;
  stage: GrowthStage;
  label: string;
  contributions?: Record<string, number>;
  reasons?: string[];
  provisional?: boolean;
  notes?: string[];                 // 評価に含められなかった要素の注記（例: 追い切りの記録なし）
  n_missing?: number;
  coverage?: Record<string, boolean>;
};

export type GrowthLevelRow = { level: number; label?: string; n?: number | null; share?: number | null; share_resid_up?: number | null; mean_resid?: number | null; share_up?: number | null; mean_y_now?: number | null };
export type GrowthMeta = { version?: string; unit?: string; thresholds?: number[]; level_table?: GrowthLevelRow[]; note?: string };

export type GrowthView = { sym: string; word: string; color: string; bg: string; bd: string };

/** 5 段階。色だけに頼らず、記号と言葉を必ず添える。 */
export const GROWTH_VIEW: GrowthView[] = [
  { sym: "▼", word: "下降", color: "#f87171", bg: "rgba(248,113,113,0.14)", bd: "rgba(248,113,113,0.55)" },
  { sym: "▽", word: "やや停滞", color: "#fb923c", bg: "rgba(251,146,60,0.13)", bd: "rgba(251,146,60,0.5)" },
  { sym: "→", word: "横ばい", color: "#94a3b8", bg: "rgba(148,163,184,0.12)", bd: "rgba(148,163,184,0.45)" },
  { sym: "▲", word: "成長", color: "#4ade80", bg: "rgba(74,222,128,0.14)", bd: "rgba(74,222,128,0.55)" },
  { sym: "▲▲", word: "大きく成長", color: "#22c55e", bg: "rgba(34,197,94,0.22)", bd: "rgba(34,197,94,0.8)" },
];

export function growthView(stage: number | null | undefined): GrowthView {
  const s = Math.max(0, Math.min(4, Math.round(Number(stage ?? 2))));
  return GROWTH_VIEW[s];
}

export const GROWTH_GROUP_JA: Record<string, string> = { oikiri: "追い切り・厩舎コメント", stage: "成長期（月齢）", weight: "馬体重", ped: "血統" };

/** 要素ごとの寄与（指数ポイント）を、大きい順に。ctrl（統制による調整）は出さない。 */
export function contributionRows(g: Growth): { key: string; label: string; points: number }[] {
  const c = g.contributions ?? {};
  return Object.keys(GROWTH_GROUP_JA)
    .filter((k) => k in c)
    .map((k) => ({ key: k, label: GROWTH_GROUP_JA[k], points: c[k] }))
    .sort((a, b) => Math.abs(b.points) - Math.abs(a.points));
}

const sign = (v: number, nd: number) => (Math.round(v * 10 ** nd) / 10 ** nd === 0 ? "±0" : `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(nd)}`);

export const fmtPoints = (v: number): string => sign(v, 1);

/** 『期待より +1.7点（= 0.17秒速い）』 */
export function indexLine(g: Growth): string {
  const sec = Math.abs(g.index) / 10;
  const dir = g.index > 0 ? "速い" : g.index < 0 ? "遅い" : "同じ";
  return `期待より ${fmtPoints(g.index)}点（${g.index === 0 ? "同じ" : `≒ ${sec.toFixed(2)}秒${dir}`}）`;
}

export const pct = (v: number | null | undefined, nd = 0): string => (v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(nd)}%`);

/** 段階ごとの実績の 1 行（評価期間の実測）。無ければ null。 */
export function levelRow(meta: GrowthMeta | null | undefined, stage: number): GrowthLevelRow | null {
  return (meta?.level_table ?? []).find((r) => r.level === stage) ?? null;
}

/** 『この段階の馬は、期待より指数が上がった割合 44%（下降 36% → 大きく成長 68%）』 */
export function levelSummary(meta: GrowthMeta | null | undefined, stage: number): string | null {
  const row = levelRow(meta, stage);
  if (!row || row.share_resid_up == null) return null;
  const lo = levelRow(meta, 0)?.share_resid_up;
  const hi = levelRow(meta, 4)?.share_resid_up;
  const range = lo != null && hi != null ? `（全段階: 下降 ${pct(lo)} → 大きく成長 ${pct(hi)}）` : "";
  return `この段階の馬は、期待より指数が上がった割合 ${pct(row.share_resid_up)}${range}`;
}

export const GROWTH_NOTE = "過去の傾向から作った目安で、AI の予測ではありません。1走ごとのばらつきが大きいため、段階は目安です。";
