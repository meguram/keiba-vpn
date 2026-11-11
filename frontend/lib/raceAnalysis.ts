/**
 * めぐ分析の型と表示の純粋関数。API は GET /api/v1/races/{race_id}/analysis（src/research/race_analysis）。
 * 分析はモジュール単位で、モジュールが列（columns）と群（groups）を自己記述する。画面は、その定義から表とカードを組み立てる。
 */

import type { TPFigure } from "@/lib/trainerPattern";

export type CellFormat = "record" | "pct" | "int" | "text" | "runs" | "range";

/** 位置取りの幅: lo〜hi は直近の走の最前〜最後方、mean は加重平均、pts は [位置(0〜1), 濃さ(0〜1。新しい走ほど大きい)] */
export type RangeCell = { lo: number; hi: number; mean: number; pts: [number, number][] };

/** 過去の走 1 つ分: [初角の通過順位, 頭数, 外れ値として除いたか(1/0)] */
export type PastRun = [number, number, number];

/** セル背景の強調の目盛り。mid で無色、hi 以上で最も濃い緑、lo 以下で最も濃い赤 */
export type HeatScale = { lo: number; mid: number; hi: number };

export type AnalysisColumn = { key: string; label: string; fmt: CellFormat; group?: string | null; title?: string | null; scale?: HeatScale | null; palette?: Record<string, string> | null };
export type AnalysisGroup = { key: string; label: string; sub?: string | null; n_key?: string | null; min_n: number };

/** 分析に使ったデータの期間・件数（provider/dataset は API が返すが画面には出さない） */
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
  /** 見せ方。省略時は表。"map" は figure の走法マップ */
  view?: "table" | "map" | "trainer_pattern";
  figure?: RunningMapFigure | TPFigure | null;
};

/** 走法マップの方面（矢印）。arrow は 2 次元での向きと長さ、r2 はその方面がこの図でどれだけ表せているか（0〜1） */
export type RunningMapAxis = { key: string; label: string; high: string; low: string; weight: number; arrow?: [number, number]; r2?: number | null };

/** 地図の位置には入れない補助の軸（急坂適性・道悪の軸）。value は 50 が平均の偏差値で、判定できないときは null（不明） */
export type RunningMapExtra = { value: number | null; runs?: number; n_power?: number; n_normal?: number; low?: boolean };

export type RunningMapHorse = {
  key: string | null;
  horse_id: string | null;
  horse_number: number | null;
  horse_name: string | null;
  x: number;
  y: number;
  group: string;
  /** 方面ごとに、全馬の中で「高い」側から見た位置（0〜100） */
  percentiles: Record<string, number | null>;
  n_runs: number;
  /** 実測ラップのある過去の出走数（方面ごとの最大） */
  n_lap_runs?: number;
  /** 方面ごとの、実測ラップのある自分の走数 */
  axis_runs?: Record<string, number>;
  /** 自分の実測の走が無く、父の産駒の平均から推定した方面 */
  from_prior?: string[];
  /** 実測ラップのある過去の出走が無く、全方面を父の産駒の平均から推定した */
  estimated: boolean;
  /** 自分の走の重みの割合（0〜1。小さいほど父の平均に寄っている） */
  own_weight: number;
  /** 補助の軸（キーは extra_axes のもの） */
  extras?: Record<string, RunningMapExtra>;
  /** 8 方面の元の空間で近い馬（同じレースの中、近い順）。地図は 2 次元に落とした分だけ近さがずれるので、選んだ馬ではこれを強調する */
  similar?: { horse_id: string | null; horse_number: number | null; horse_name: string | null; distance: number }[];
  last_run_date: string | null;
  /** このレースの着順（結果が出ている馬だけ。未確定・中止は null） */
  finish_position?: number | null;
  /** 血統から見た傾向（性別・父・母父だけから。モデルの無い環境では無い） */
  ped?: RunningMapPed | null;
};

/** 観点ごとに [期待値, 80%区間の下, 上] */
export type RunningMapPed = { sex: string; sire: string | null; damsire: string | null; has_damsire: boolean; v: Record<string, [number, number, number]>;
  /** 10 点満点の観点の [点数, 80%区間の下, 上]（0.5 点刻み） */
  s?: Record<string, [number, number, number]> };

export type RunningMapPedTrait = {
  key: string; label: string; hint: string;
  /** pct=割合 / m=メートル / q=0〜1 / n=数値 */
  fmt: "pct" | "m" | "q" | "n";
  /** 全体の平均 */
  base: number | null;
  /** 上位・下位 1 割の馬の平均が全体の平均から離れる幅（目盛り） */
  ref: number | null;
  /** 新しい馬での当てはまり（観測 R²。血統でどれだけ言えるか） */
  r2: number | null;
  /** 10 点満点で見せる観点 */
  score?: boolean;
  /** 距離の観点: 芝・ダートそれぞれの平均距離（期待値は平均との差なので足して実際の距離にする） */
  abs?: Record<string, number>;
};

export type RunningMapFigure = {
  status: "ok" | "insufficient";
  axes: RunningMapAxis[];
  /** このレースで出す補助の軸（急坂適性は中山・阪神、道悪の軸は芝=芝道悪適性・ダート=ダートタフ適正） */
  extra_axes?: { key: string; label: string; high: string; low: string }[];
  horses: RunningMapHorse[];
  groups: { id: string; label: string; traits: { axis: string; diff: number; text: string }[]; horse_numbers: (number | null)[] }[];
  unmapped: { key: string | null; horse_name: string | null; horse_number: number | null; reason: string }[];
  fidelity: { explained: number | null; distance_spearman: number | null } | null;
  /** 好走しやすさの格子の範囲（全レース共通の座標。n×n、x が外側のループ） */
  extent: { n: number; x: [number, number]; y: [number, number] } | null;
  /** このコースの好走しやすさ。lift は期待に対する倍率、sig は 0=期待が小さく対象外 / 1=有意でない / 2=有意（FDR 5%）。添字は ix*n+iy */
  aptitude: { course: string; n_runs: number; n_races: number; lift: number[]; sig: string; score?: boolean } | null;
  /** コースで好走に効く上位 3 方面だけで作った地図（無ければ全レース共通の 8 方面の地図）。level=group は出走が少なく芝ダ×距離帯でまとめたもの */
  course_map?: RunningMapCourse | null;
  /** その日（同じ競馬場・同じ芝ダート、このレースより前）の馬場傾向。前のレースが無ければ null */
  day_trend?: RunningMapDayTrend | null;
  /** 適性が合う 5 頭（コース特注ゾーン＋馬場特注ゾーンの色から。1 パターン） */
  best5?: RunningMapBest5 | null;
  /** 血統から見た傾向の観点（各馬の ped.v のキー） */
  ped_traits?: RunningMapPedTrait[] | null;
  /** 道悪（稍重以上）切替の地図。同じ形（day_trend は無い）。道悪の標本が足りず作れなければ無い */
  wet?: RunningMapFigure | null;
};

export type RunningMapBest5 = {
  w_day: number;
  /** どのゾーンで選んだか（片方しか無いとき） */
  source: "both" | "course" | "day";
  picks: { rank: number; horse_number: number | null; horse_id: string | null; score: number }[];
};

export type RunningMapCourse = {
  axes: string[];
  level: "course" | "venue_group" | "group";
  label: string;
  /** 方面ごとの重要度（share は 0〜1 の割合、sign は高いほど好走なら 1・低いほど好走なら -1）。重要な順 */
  importance: { key: string; label: string; share: number; sign: number }[];
  since?: string | null;
};

export type RunningMapDayTrend = {
  status: "trend" | "none" | "few";
  surface: string | null;
  n_races: number;
  n_runs: number;
  races: string[];
  track_conditions: string[];
  /** 今日のこのレースより前のレース数・同じ開催の前日のレース数（前日は軽く数える）・今日の馬場状態 */
  n_races_today?: number;
  n_races_prev?: number;
  track_today?: string | null;
  /** 軸ごとの傾き（得点−期待を軸の値へ回帰）と t・p。trend は多重検定の補正後に傾向ありと判定されたもの */
  axes: { key: string; label: string; high: string; low: string; kind: "map" | "extra"; slope: number | null; t: number | null; p: number | null; n: number; trend: boolean }[];
  trend: { key: string; label: string; text: string; t: number | null }[];
  /** この地図の格子（extent と同じ並び）での、その日の好走しやすさ */
  heat: { lift: number[]; sig: string } | null;
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
    case "range": {
      const r = v as RangeCell;
      return typeof r?.lo === "number" ? `${Math.round(r.lo * 100)}〜${Math.round(r.hi * 100)}%` : EMPTY;
    }
    case "runs":
      return Array.isArray(v) && v.length > 0 ? (v as PastRun[]).map(([p, x]) => `${p}/${x}`).join(" ") : EMPTY;
    default:
      return String(v) || EMPTY;
  }
}

/** 位置取りチップの背景色。前（逃げ・先行）ほど赤、後ろほど青。外れ値は色を付けない（null）。 */
export function runColor(pos: number, field: number): string {
  const r = field > 1 ? Math.min(1, Math.max(0, (pos - 1) / (field - 1))) : 0;
  return `hsl(${Math.round(8 + r * 207)} 62% 34%)`;
}

/** 並べ替え用の値。成績（record）は 1 着の数で並べる。色分けのある文字の列（脚質など）は、色分けの定義の順
 *  （脚質ならハナ → 先行(前) → 先行(後ろ) → 差し → 追込）で並べ、定義に無い値（不明など）は向きによらず最後。それ以外は表示文字から読まれる。 */
export function sortValue(v: unknown, fmt: CellFormat, palette?: Record<string, string> | null): string | undefined {
  if (fmt === "record" && Array.isArray(v) && v.length > 0) return String(v[0]);
  if (fmt === "range") return typeof (v as RangeCell | null)?.mean === "number" ? String((v as RangeCell).mean) : "";
  if (palette && typeof v === "string") {
    const i = Object.keys(palette).indexOf(v);
    return i >= 0 ? String(i + 1) : "";
  }
  return undefined;
}

/** 狭い画面の見出し用に短くした列名（『連対率』→『連対』。2 文字の『勝率』はそのまま） */
export function shortLabel(label: string): string {
  return label.length >= 3 && label.endsWith("率") ? label.slice(0, -1) : label;
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

/** "2019-12-22" -> "2019年12月22日"。形式が違えばそのまま返す */
const jpDate = (d: string) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(d);
  return m ? `${m[1]}年${Number(m[2])}月${Number(m[3])}日` : d;
};

/** 集計に使ったデータの期間・件数の表示用。例: 集計期間: 2019年12月22日〜2026年10月24日 ／ 対象: 23,640レース・323,461出走 */
export function formatPeriod(src: AnalysisSource): string | null {
  const parts: string[] = [];
  if (src.period_from && src.period_to) parts.push(`集計期間: ${jpDate(src.period_from)}〜${jpDate(src.period_to)}`);
  if (src.races > 0) parts.push(`対象: ${src.races.toLocaleString("en-US")}レース・${src.rows.toLocaleString("en-US")}出走`);
  return parts.length ? parts.join(" ／ ") : null;
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
