/** 予測成績（予測結果の集計）API の型と整形。サーバ: src/api/prediction_results_api.py */

export type Counts = { n: number; f1: number; f2: number; f3: number; f4p: number };
export type MarkKey = "honmei" | "pair" | "star" | "kesi";
export type BetByType = { invested: number; returned: number; n: number; hit: number; label?: string; roi?: number | null };

export type Kpi = {
  honmei: { n: number; win: number | null; top2: number | null; top3: number | null };
  top1: { n: number; win: number | null; top3: number | null };
  marks: Record<MarkKey, Counts & { rates: { f1: number | null; f2: number | null; f3: number | null; f4p: number | null } }>;
  roi: number | null;
  invested: number;
  returned: number;
  races_bought: number;
  races_hit: number;
  estimated_races: number;
  by_type: Record<string, BetByType>;
  winner_logloss: number | null;
  /** 市場補正した確率のレースだけ。同じレースの元のモデルの確率と補正後の勝ち馬の対数損失 */
  winner_logloss_model?: { n: number; model: number; calibrated: number } | null;
};

export type RacePick = {
  horse_number: number;
  horse_name: string;
  finish: number | null;
  win_prob: number;
  win_odds?: number | null;
  ev_win?: number | null;
  mark?: string | null;
};

export type RaceRecord = {
  race_id: string;
  venue: string;
  round: number;
  race_name: string;
  grade?: string | null;
  included: boolean;
  reason?: string;
  result_status?: "final" | "provisional";
  payoff_official?: boolean;
  prediction_source?: string;
  top1?: RacePick;
  marks?: Record<MarkKey, RacePick[]>;
  result_top3?: { horse_number: number; finish: number }[];
  bets?: { total_bet: number; total_return: number; estimated?: boolean; status?: string; level?: number | null; lv?: { level?: number | null } };
  /** 画面に出している推奨馬券（target）の仮想購入。level は画面のレース推奨度（0=見送り〜4）。未記録のレースには無い */
  bets_t?: { status: string; level?: number; total_bet?: number; total_return?: number; estimated?: boolean };
};

/** 推奨度（レース期待値）が『標準』以上のレースだけの集計。会員・開発者の応答にだけ付く。 */
export type RatedBlock = {
  min_level: number;
  label: string;
  counts: { settled: number; rated: number; skipped: number; unrated: number };
  kpi: Kpi;
  summary: { marks: Record<MarkKey, Counts> };
  legend: { level: number; label: string; short: string }[];
};

export type Highlight = { kind: "ana_hit" | "big_return" | "big_miss"; race_id: string; value: number; text: string };

export type SeriesPoint = { date: string; status: "final" | "provisional"; roi: number | null; honmei_win: number | null; races: number; roi_rated?: number | null; honmei_win_rated?: number | null };

export type PredictionResults = {
  scope: "day" | "week" | "month" | "season";
  date: string;
  from: string;
  to: string;
  label: string;
  days: string[];
  statuses: Record<string, "final" | "provisional">;
  status: "final" | "provisional";
  kpi: Kpi;
  summary: { races: number; races_settled: number; races_no_prediction: number; races_no_result: number; marks: Record<MarkKey, Counts> };
  rated?: RatedBlock;
  series: SeriesPoint[];
  highlights: Highlight[];
  notes: string[];
  races?: RaceRecord[];
  prediction_sources?: Record<string, number>;
  models?: string[];
  built_at?: string;
};

export type NewsItem = {
  kind: "day" | "week" | "hit";
  title: string;
  subtitle: string;
  subtitle_public: string;
  badge: "暫定" | "確定";
  date: string;
  link: string;
  roi: number | null;
};

export const SCOPES: { id: PredictionResults["scope"]; label: string }[] = [
  { id: "day", label: "日" },
  { id: "week", label: "週" },
  { id: "month", label: "月" },
  { id: "season", label: "シーズン" },
];

export const MARK_META: Record<MarkKey, { sym: string; label: string; color: string }> = {
  honmei: { sym: "◎", label: "本命（1着優位）", color: "#ef4444" },
  pair: { sym: "○", label: "対抗（連対候補）", color: "#3b82f6" },
  star: { sym: "★", label: "中穴妙味（10〜30倍）", color: "#a78bfa" },
  kesi: { sym: "消", label: "消し（3着内に来る確率が低い）", color: "#cbd5e1" },
};

export const FINISH_META: { key: "f1" | "f2" | "f3" | "f4p"; label: string; color: string }[] = [
  { key: "f1", label: "1着", color: "#f59e0b" },
  { key: "f2", label: "2着", color: "#94a3b8" },
  { key: "f3", label: "3着", color: "#b45309" },
  { key: "f4p", label: "4着以下", color: "#334155" },
];

/** レース記録の推奨度（0=見送り〜4=最注目）。画面の推奨馬券（target）の仮想購入のもの。未記録は null。 */
export const raceLevel = (r: RaceRecord): number | null => (r.bets_t?.status === "ok" ? r.bets_t.level ?? null : null);

export const pct = (x: number | null | undefined, digits = 0): string => (x == null ? "—" : `${(x * 100).toFixed(digits)}%`);
export const yen = (x: number): string => `${Math.round(x).toLocaleString()}円`;

/** YYYYMMDD → "10/18(日)" */
export function dayLabel(d: string): string {
  const dt = new Date(Number(d.slice(0, 4)), Number(d.slice(4, 6)) - 1, Number(d.slice(6, 8)));
  return `${dt.getMonth() + 1}/${dt.getDate()}(${"日月火水木金土"[dt.getDay()]})`;
}

export function shiftDate(d: string, scope: PredictionResults["scope"], dir: -1 | 1): string {
  const dt = new Date(Number(d.slice(0, 4)), Number(d.slice(4, 6)) - 1, Number(d.slice(6, 8)));
  if (scope === "day") dt.setDate(dt.getDate() + dir);
  else if (scope === "week") dt.setDate(dt.getDate() + 7 * dir);
  else if (scope === "month") dt.setMonth(dt.getMonth() + dir);
  else dt.setFullYear(dt.getFullYear() + dir);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${dt.getFullYear()}${p(dt.getMonth() + 1)}${p(dt.getDate())}`;
}

export function periodLabel(r: Pick<PredictionResults, "scope" | "date" | "from" | "to">): string {
  if (r.scope === "day") return dayLabel(r.date);
  if (r.scope === "week") return `${dayLabel(r.from)}〜${dayLabel(r.to)}`;
  if (r.scope === "month") return `${r.date.slice(0, 4)}年${Number(r.date.slice(4, 6))}月`;
  return `${r.date.slice(0, 4)}年シーズン`;
}

/** 期間の識別子（日=YYYYMMDD、週=その週の月曜、月=YYYYMM、シーズン=YYYY）。『次へ』が最新を超えないかの判定用。 */
export function periodKey(scope: PredictionResults["scope"], d: string): string {
  if (scope === "day") return d;
  if (scope === "month") return d.slice(0, 6);
  if (scope === "season") return d.slice(0, 4);
  const dt = new Date(Number(d.slice(0, 4)), Number(d.slice(4, 6)) - 1, Number(d.slice(6, 8)));
  dt.setDate(dt.getDate() - ((dt.getDay() + 6) % 7));
  const p = (n: number) => String(n).padStart(2, "0");
  return `${dt.getFullYear()}${p(dt.getMonth() + 1)}${p(dt.getDate())}`;
}
