/** 想定めぐ指数（過去のめぐ指数からの想定。AI の予測ではない）。API: /api/race/{id}/predictions の expected_megu / expected_megu_meta */

export type ExpectedMeguAdjust = { key: string; label: string; delta: number };

export type ExpectedMegu = {
  value: number;
  range: number | null;
  n_runs: number;
  latest_run_date: string;              // YYYY-MM-DD
  confidence: "normal" | "reference" | "stale";
  low_conf_runs?: number;
  gap_days?: number | null;
  base_value?: number;                   // 補正前（過去走の加重平均）
  change?: "none" | "some" | "large";    // 今回の条件が過去走とどれだけ違うか
  reasons?: string[];
  adjust?: ExpectedMeguAdjust[];
  rank?: number;
  gap_sec?: number;                      // 最上位との差（秒。1 点 = 0.1 秒）
  latest_unreflected?: boolean;
  latest_past_date?: string;
  /** 実測のめぐ指数の直近走（新しい順。最大 5 走。low=低信頼） */
  recent?: { date: string; value: number; low?: boolean }[];
};

export type ExpectedMeguMeta = {
  name?: string; method?: string; data_through?: string | null; n_horses?: number; n_known?: number;
  model?: { version?: string; trained_on?: string; validated_on?: string; eval?: Record<string, number> } | null;
};

export const md = (ymd?: string | null): string => {
  const m = (ymd ?? "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
  return m ? `${Number(m[2])}/${Number(m[3])}` : "-";
};

/** 出馬表に出す直近走の指数（古い→新しいの順）。想定値は画面に出さない。 */
export const recentRuns = (e?: ExpectedMegu | null, max = 5) => [...(e?.recent ?? [])].slice(0, max).reverse();

/** 直近走の平均（並べ替え用）。 */
export const recentMean = (e?: ExpectedMegu | null): number | null => {
  const r = e?.recent ?? [];
  return r.length ? r.reduce((s, x) => s + x.value, 0) / r.length : null;
};

export const fmtValue = (e?: ExpectedMegu | null): string => (e ? e.value.toFixed(1) : "—");

export const fmtGap = (e?: ExpectedMegu | null): string =>
  e == null || e.gap_sec == null ? "" : e.gap_sec === 0 ? "最上位" : `+${e.gap_sec.toFixed(2)}秒`;

export const CONF_LABEL: Record<string, string> = { normal: "", reference: "参考（走数が少ない）", stale: "参考（間隔が空いている）" };

export const CHANGE_LABEL: Record<string, string> = { none: "", some: "条件変化あり", large: "条件が大きく変わる" };

/** 一覧のツールチップ用の 1 行 */
export function tipOf(e?: ExpectedMegu | null): string {
  if (!e) return "過去のめぐ指数がありません";
  const parts = [`過去${e.n_runs}走から ${e.value.toFixed(1)}${e.range != null ? ` ±${e.range.toFixed(1)}` : ""}`, `最新の反映走 ${md(e.latest_run_date)}`];
  if (e.reasons?.length) parts.push(e.reasons.join("・"));
  if (CONF_LABEL[e.confidence]) parts.push(CONF_LABEL[e.confidence]);
  if (e.latest_unreflected) parts.push(`直近の走(${md(e.latest_past_date)})は未反映`);
  return parts.join(" / ");
}
