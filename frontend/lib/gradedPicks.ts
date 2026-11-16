/** めぐ分析の「データで予想」タブ（重賞だけ）の型と表示の小さな関数。値は src/research/race_analysis/graded_picks.py が作る。
 *  出すのは、2018〜2024 年のデータだけで決めて凍結した B の規則と、比べる基準の前日版の ◎ だけ（A は出さない）。 */

export type GPHorse = {
  hn: number;
  name: string | null;
  /** 確定の単勝オッズ（結果の表示だけ。選び方には使わない） */
  odds: number | null;
  pop: number | null;
  fin: number;
  /** 100 円あたりの払戻（円）。払戻の解けない回は null */
  tan: number | null;
  fuku: number | null;
};

export type GPRace = {
  race_id: string;
  date: string;
  venue: string;
  field: number;
  pay_ok: boolean;
  winners: GPHorse[];
  base: GPHorse | null;
  B: { honmei: GPHorse | null; kill: GPHorse[] | null };
};

export type GPRuleTerm = { text: string; w: number; small: boolean; key: string };
export type GPRuleText = { filter: string | null; terms: GPRuleTerm[]; tie: string };
export type GPRule = { rule: GPRuleText; label: string; default: boolean; dayof: boolean; k?: number };

export type GPHonmeiStats = { races: number; rec: number[]; win: number; place: number; roi_t: number | null; roi_f: number | null; pay_races: number };
export type GPKillStats = { races: number; killed: number; rec: number[]; top3_rate: number | null };
export type GPPeriodStats = { B_honmei: GPHonmeiStats | null; B_kill: GPKillStats | null; base_honmei: GPHonmeiStats | null };

export type GPFigure = {
  rkey: string;
  grade: string;
  course: string;
  rules: { honmei: GPRule; kill: GPRule | null };
  /** このレース: result = 結果つきの記録 / pending = 出馬表の確定後に表示 / dayof = 当日の値が要る規則 / default = 規則が無い重賞 */
  this: { status: "result" | "pending" | "dayof" | "default" | "picked"; race: GPRace | null; period: Period; picked?: GPPicked | null };
  past: GPRace[];
  stats: Partial<Record<Period, GPPeriodStats>>;
  periods?: { dev: number[]; eval: number[] } | null;
  track_from?: string | null;
  frozen_at?: string | null;
  data_until?: string | null;
};

export type Period = "dev" | "eval" | "forward";

/** 前日の計算の該当馬（出馬表の確定後。結果はまだ無い）。skipped は計算しなかった理由（dayof = 当日の値を使う規則、no_rule = 規則なし） */
export type GPPickHorse = { hn: number; name: string | null };
export type GPPicked = {
  race_id: string;
  date: string;
  data_until: string;
  computed_at: string;
  field: number;
  base: GPPickHorse | null;
  B: { honmei: GPPickHorse | null; kill: GPPickHorse[] | null };
  skipped?: Partial<Record<"honmei" | "kill", "dayof" | "no_rule" | "default">>;
};

export const SKIP_LABEL: Record<string, string> = {
  dayof: "当日に決まる値（馬体重・馬場など）を使う規則なので、前日には出しません（結果の後に記録します）",
  no_rule: "規則がありません",
  default: "2018〜2024 年にこの重賞の回が無く、規則を決められていないため出しません",
};

export const PERIOD_LABEL: Record<Period, string> = {
  dev: "2018〜2024 年（規則を選んだ期間・当てはめ）",
  eval: "2025 年〜凍結まで（決めた後に当てはめた）",
  forward: "凍結の後（前向きの記録）",
};

export function pct(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(1)}%`;
}

export function rec(r: number[] | null | undefined): string {
  return r && r.length === 4 ? r.join("-") : "—";
}

export function odds(h: GPHorse | null | undefined): string {
  return h?.odds == null ? "—" : h.odds.toFixed(1);
}

export function yen(v: number | null | undefined): string {
  return v == null ? "—" : `${Math.round(v).toLocaleString("ja-JP")}`;
}

export function fin(h: GPHorse | null | undefined): string {
  return h == null || !Number.isFinite(h.fin) ? "—" : String(Math.round(h.fin));
}

/** 消した馬のうち 3 着以内に来た頭数 */
export function killedTop3(kill: GPHorse[] | null | undefined): number {
  return (kill ?? []).filter((h) => h.fin <= 3).length;
}

/** 画面に出す期間の順（新しい順）。値の無い期間は出さない */
export function periodsOf(stats: GPFigure["stats"]): Period[] {
  return (["forward", "eval", "dev"] as Period[]).filter((p) => stats[p]);
}
