/**
 * めぐ分析の型と表示の純粋関数。API は GET /api/v1/races/{race_id}/analysis（src/research/race_analysis）。
 * 分析はモジュール単位で、モジュールが列（columns）と群（groups）を自己記述する。画面は、その定義から表とカードを組み立てる。
 */

export type CellFormat = "record" | "pct" | "int" | "text";

/** セル背景の強調の目盛り。mid で無色、hi 以上で最も濃い緑、lo 以下で最も濃い赤 */
export type HeatScale = { lo: number; mid: number; hi: number };

export type AnalysisColumn = { key: string; label: string; fmt: CellFormat; group?: string | null; title?: string | null; scale?: HeatScale | null; palette?: Record<string, string> | null };
export type AnalysisGroup = { key: string; label: string; sub?: string | null; n_key?: string | null; min_n: number };

/** 分析に使ったデータの出典（提供元の機関・集計した期間・件数） */
export type AnalysisSource = {
  provider: string;
  dataset: string;
  period_from?: string | null;
  period_to?: string | null;
  races: number;
  rows: number;
};

export type AnalysisModuleMeta = {
  id: string;
  label: string;
  description: string;
  status: "ok" | "not_applicable" | "error";
  columns?: AnalysisColumn[];
  groups?: AnalysisGroup[];
  notes?: string[];
  data_until?: string | null;
  source?: AnalysisSource | null;
  /** 分析の軸（父など）を表の列として出すときの見出し。値は各馬の cells の _key */
  key_header?: string | null;
};

export type AnalysisEntry = {
  key: string;
  horse_number: number | null;           // 馬番が未確定（特別登録・出走馬決定）の段階は null
  bracket_number?: number | null;
  horse_name?: string | null;
  horse_id?: string | null;
  sex_age?: string | null;
  jockey_name?: string | null;
  jockey_id?: string | null;
  jockey_weight?: number | null;
  trainer_name?: string | null;
  analysis?: Record<string, Record<string, unknown>>;
};

export type AnalysisRace = {
  race_id: string;
  race_name?: string | null;
  date?: string | null;
  venue?: string | null;
  round?: number | string | null;
  surface?: string | null;
  distance?: number | null;
  grade?: string | null;
  start_time?: string | null;
};

export type RaceAnalysis = {
  status: "ok" | "no_entries" | "pending";
  /** 分析を事前計算した時刻（ISO）。表示のたびには計算しない */
  computed_at?: string | null;
  message?: string;
  race: AnalysisRace;
  entries: AnalysisEntry[];
  provisional_entries: boolean;
  modules: AnalysisModuleMeta[];
};

const EMPTY = "–";

export function formatCell(v: unknown, fmt: CellFormat): string {
  if (v == null) return EMPTY;
  switch (fmt) {
    case "record":
      return Array.isArray(v) && v.length > 0 ? v.join("-") : EMPTY;
    case "pct":
      return typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : EMPTY;
    case "int":
      return typeof v === "number" && Number.isFinite(v) ? String(Math.round(v)) : EMPTY;
    default:
      return String(v) || EMPTY;
  }
}

/** 並べ替え用の値。成績（record）は 1 着の数で並べる。それ以外は表示文字から読まれる。 */
export function sortValue(v: unknown, fmt: CellFormat): string | undefined {
  if (fmt === "record" && Array.isArray(v) && v.length > 0) return String(v[0]);
  return undefined;
}

export type ColumnBlock = { group: AnalysisGroup | null; columns: AnalysisColumn[] };

/** 列を、群ごとのまとまりに並べ直す（群に属さない列は 1 つのまとまり）。群の順は定義順、群内の列の順は columns の順。 */
export function columnBlocks(columns: AnalysisColumn[], groups: AnalysisGroup[]): ColumnBlock[] {
  const blocks: ColumnBlock[] = [];
  const ungrouped = columns.filter((c) => !c.group || !groups.some((g) => g.key === c.group));
  for (const g of groups) {
    const cols = columns.filter((c) => c.group === g.key);
    if (cols.length) blocks.push({ group: g, columns: cols });
  }
  if (ungrouped.length) blocks.push({ group: null, columns: ungrouped });
  return blocks;
}

/** 標本数が足りない群か（薄く表示する）。標本数の列が無い群は常に false。 */
export function isThin(group: AnalysisGroup | null, cells: Record<string, unknown> | undefined): boolean {
  if (!group || !group.n_key || !group.min_n) return false;
  const n = cells?.[group.n_key];
  return typeof n === "number" ? n < group.min_n : false;
}

/**
 * 未来の開催日で、出馬表がまだ取得できていないレースか（選べなくする）。
 * 出馬表の段階（entries_phase）が無い／none のレースは、JRA の公示前で取得できていない。
 * 当日以前のレースは、出馬表か結果があるので常に選べる。date / today は YYYYMMDD。
 */
export function entriesUnavailable(race: { entries_phase?: string | null }, date: string, today: string): boolean {
  if (!date || date <= today) return false;
  return !race.entries_phase || race.entries_phase === "none";
}

const ymd = (d: string) => d.replace(/-/g, "/");

/** 出典の表示用の文。例: データの出典: netkeiba.com（レース結果） ／ 集計期間: 2020/01/05〜2026/10/25 ／ 対象: 2,880レース・39,639出走 */
export function formatSource(src: AnalysisSource): string {
  const parts = [`データの出典: ${src.provider}（${src.dataset}）`];
  if (src.period_from && src.period_to) parts.push(`集計期間: ${ymd(src.period_from)}〜${ymd(src.period_to)}`);
  if (src.races > 0) parts.push(`対象: ${src.races.toLocaleString("en-US")}レース・${src.rows.toLocaleString("en-US")}出走`);
  return parts.join(" ／ ");
}

/** 値に応じたセルの背景色（平均より高い=緑、低い=赤、差が大きいほど濃い）。目盛りが無い・値が無い・平均とほぼ同じなら色なし。 */
export function heatColor(v: unknown, scale: HeatScale | null | undefined): string | undefined {
  if (typeof v !== "number" || !Number.isFinite(v) || !scale) return undefined;
  const { lo, mid, hi } = scale;
  const t = v >= mid ? (hi > mid ? Math.min(1, (v - mid) / (hi - mid)) : 0) : (mid > lo ? -Math.min(1, (mid - v) / (mid - lo)) : 0);
  if (Math.abs(t) < 0.04) return undefined;
  const alpha = (0.08 + 0.42 * Math.abs(t)).toFixed(2);
  return t > 0 ? `rgba(34,197,94,${alpha})` : `rgba(248,81,73,${alpha})`;
}

/** 事前計算の時刻の表示用（例: 10/28 17:30 時点）。壊れた値は null。 */
export function formatComputedAt(iso: string | null | undefined): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso ?? "");
  return m ? `${Number(m[2])}/${Number(m[3])} ${m[4]}:${m[5]} 時点` : null;
}
