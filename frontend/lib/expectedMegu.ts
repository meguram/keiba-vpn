/** 想定めぐ指数（過去のめぐ指数からの想定。AI の予測ではない）。API: /api/race/{id}/predictions の expected_megu / expected_megu_meta */

export type ExpectedMeguAdjust = { key: string; label: string; delta: number };

/** 過去の 1 走（馬柱の 1 セル）。race_id 以降のレース情報は、実績の表が読めたときだけ付く */
export type RecentRun = {
  date: string; value: number; low?: boolean;
  race_id?: string; cls?: string; race_name?: string; round?: number; venue?: string; surface?: string; distance?: number; going?: string; grade?: string;
  field?: number; finish?: number; pop?: number; jockey?: string; jw?: number; l3f?: number; passing?: string; bw?: number; bw_diff?: number; margin?: string;
};

/** このクラスの基準（突破ライン）。API: /api/race/{id}/predictions の class_benchmark。基準が無いレースはキー自体が無い */
export type ClassBenchmark = {
  label: string;                         // 『2歳未勝利』『3歳以上1勝クラス』
  key: string;
  level: "tier" | "cell" | "surface" | "distance";
  surface?: string | null;               // 面で分けた基準のときだけ（芝/ダ）
  band?: string | null;                  // 距離帯で分けた基準のときだけ（短/中短/中/長）
  n: number;                             // 基準にした2着馬の数
  median: number;                        // 2着馬の指数の中央値（そのクラスの標準的な水準）
  line: number;                          // 突破ライン = 2着馬の指数の下位25%
  p75: number;
  period?: string;                       // 『2023-2025』
  fallback?: boolean;                    // 年齢まで一致する基準が無く、クラスのみの基準を使った
};

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
  recent?: RecentRun[];
};

/** めぐ印（好走に値する馬）。API: predictions[].megu_mark。key が null の馬は印なし（unavailable か reason に理由） */
export type MeguMark = {
  version?: number;
  key: "top" | "good" | null;            // top=めぐ◎（レース内1位かつ突破ライン以上）、good=めぐ○（2〜3位かつ突破ライン以上）
  label: string | null;
  value: number | null;                  // v3b の集約値（直近5走を新しい走ほど重い加重 Huber で）
  margin: number | null;                 // 突破ラインとの差
  line: number | null;
  rank: number | null;                   // レース内の順位（集約値の高い順）
  z?: number | null;
  n_runs: number;
  reason?: string | null;
  unavailable?: string | null;
};

export type ExpectedMeguMeta = {
  name?: string; method?: string; data_through?: string | null; n_horses?: number; n_known?: number;
  model?: { version?: string; trained_on?: string; validated_on?: string; eval?: Record<string, number> } | null;
};

export const md = (ymd?: string | null): string => {
  const m = (ymd ?? "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
  return m ? `${Number(m[2])}/${Number(m[3])}` : "-";
};


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
