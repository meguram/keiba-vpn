/* 展開タイプ（pace_type）と展開別の有利さ（scenario_forecast）の型と純粋関数。
   API: GET /api/race/{id}/predictions の pace_type / scenario_forecast（無いレースは従来の paceFit にフォールバック）。
   シナリオ別の有利さは過去の傾向から作った説明用の目安で、勝率・期待値の予測には使っていない。 */

export type PaceKey = "high" | "mid" | "slow";
export type ShapeKey = "compact" | "normal" | "stretched";
export type StyleKey = "nige" | "senko" | "sashi" | "oikomi";

export type StyleFitItem = { label: string; score: number; expected_lift?: number; top_scenario_lift?: number; top_scenario_score?: number };

export type PaceType = {
  version?: string;
  key: string;
  label: string;
  note?: string;
  probs: Record<string, number>;
  pace_probs: Record<PaceKey, number>;
  shape_probs: Record<ShapeKey, number>;
  scenarios?: { key: string; label: string; prob: number; pace: PaceKey; shape: ShapeKey }[];
  style_fit: Partial<Record<StyleKey, StyleFitItem>>;
  confidence?: string;
  basis?: { pace_level?: string; strong_front_runners?: number; front_runners_claim?: number; course_stretched_tendency?: number; lap_forecast_used?: boolean };
};

export type ScenarioHorse = {
  horse_number: number;
  by_scenario: Record<string, { win?: number; top2?: number; top3?: number; fit_score: number }>;
  sensitivity?: number;
  sensitivity_label?: string;
  best_scenario?: string;
  best_scenario_label?: string;
  label?: string;
};

export type ScenarioForecast = {
  version?: string;
  scenarios: { key: string; label: string; prob: number }[];
  horses: ScenarioHorse[];
};

export const PACE_ORDER: PaceKey[] = ["high", "mid", "slow"];
export const SHAPE_ORDER: ShapeKey[] = ["compact", "normal", "stretched"];
export const STYLE_ORDER: StyleKey[] = ["nige", "senko", "sashi", "oikomi"];
export const PACE_JA: Record<PaceKey, string> = { high: "ハイ", mid: "ミドル", slow: "スロー" };
export const SHAPE_JA: Record<ShapeKey, string> = { compact: "凝縮", normal: "中間", stretched: "縦長" };
export const CONF_JA: Record<string, string> = { high: "高", mid: "中", low: "低" };

/** 有利/不利の境目。記号は ±0.15、語は ±0.15 と ±0.4。 */
export const FIT_EDGE = 0.15;
export const FIT_STRONG = 0.4;

export type FitTone = "plus" | "even" | "minus";

export const fitTone = (score: number | null | undefined): FitTone | null =>
  score == null || !Number.isFinite(score) ? null : score >= FIT_EDGE ? "plus" : score <= -FIT_EDGE ? "minus" : "even";

/** 小さな記号。有利=▲、互角=－、不利=▽。 */
export const fitMark = (score: number | null | undefined): string => {
  const t = fitTone(score);
  return t === "plus" ? "▲" : t === "minus" ? "▽" : t === "even" ? "－" : "—";
};

/** 脚質の有利度の語（有利/やや有利/互角/やや不利/不利）。 */
export function fitWord(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return "—";
  if (score >= FIT_STRONG) return "有利";
  if (score >= FIT_EDGE) return "やや有利";
  if (score <= -FIT_STRONG) return "不利";
  if (score <= -FIT_EDGE) return "やや不利";
  return "互角";
}

/** 馬の総合の語（シナリオ確率で重みづけした期待の fit_score から）。 */
export const overallWord = (score: number | null | undefined): "向く" | "ふつう" | "不向き" | null => {
  const t = fitTone(score);
  return t === "plus" ? "向く" : t === "minus" ? "不向き" : t === "even" ? "ふつう" : null;
};

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/** 確率の組を合計 1 にそろえる。合計が 0 以下なら一様。 */
export function normalize(p: Partial<Record<string, number>>, keys: string[]): Record<string, number> {
  const vals = keys.map((k) => (Number.isFinite(p[k] as number) && (p[k] as number) > 0 ? (p[k] as number) : 0));
  const s = sum(vals);
  return Object.fromEntries(keys.map((k, i) => [k, s > 0 ? vals[i] / s : 1 / keys.length]));
}

/** あるペース（high/mid/slow）での fit_score = そのペースの 3 隊列を、隊列確率で重みづけした平均。足りない隊列は除いて再正規化。 */
export function paceFitScore(h: ScenarioHorse, pace: PaceKey, shapeProbs: Partial<Record<ShapeKey, number>>): number | null {
  const items = SHAPE_ORDER.map((s) => ({ s, f: h.by_scenario?.[`${pace}_${s}`]?.fit_score })).filter((x) => Number.isFinite(x.f as number));
  if (!items.length) return null;
  const w = normalize(shapeProbs, items.map((x) => x.s));
  return sum(items.map((x) => w[x.s] * (x.f as number)));
}

/** 総合の期待 fit_score = 9 シナリオの確率で重みづけした平均。 */
export function overallFitScore(h: ScenarioHorse, scenarios: { key: string; prob: number }[]): number | null {
  const items = scenarios.map((s) => ({ p: s.prob, f: h.by_scenario?.[s.key]?.fit_score })).filter((x) => Number.isFinite(x.f as number) && Number.isFinite(x.p) && x.p > 0);
  const tp = sum(items.map((x) => x.p));
  if (!items.length || tp <= 0) return null;
  return sum(items.map((x) => x.p * (x.f as number))) / tp;
}

export type HorseFit = {
  paces: Record<PaceKey, number | null>;
  overall: number | null;
  word: "向く" | "ふつう" | "不向き" | null;
  sensitivity: string | null;
  bestLabel: string | null;
  grid: { key: string; label: string; pace: PaceKey; shape: ShapeKey; prob: number; score: number | null }[];
};

/** 馬番 → 展開別の有利さ。pace_type が無いときは scenario_forecast だけの確率を使う。 */
export function buildFitMap(sf: ScenarioForecast | null | undefined, pt: PaceType | null | undefined): Map<number, HorseFit> | null {
  if (!sf || !sf.horses?.length) return null;
  const shape = pt?.shape_probs ?? { compact: 1, normal: 1, stretched: 1 };
  const scen = (pt?.scenarios?.length ? pt.scenarios.map((s) => ({ key: s.key, prob: s.prob })) : sf.scenarios.map((s) => ({ key: s.key, prob: s.prob })));
  const m = new Map<number, HorseFit>();
  for (const h of sf.horses) {
    if (h.horse_number == null) continue;
    const overall = overallFitScore(h, scen);
    const grid = PACE_ORDER.flatMap((pace) => SHAPE_ORDER.map((sh) => {
      const key = `${pace}_${sh}`;
      return { key, label: `${PACE_JA[pace]}・${SHAPE_JA[sh]}`, pace, shape: sh, prob: scen.find((s) => s.key === key)?.prob ?? 0, score: h.by_scenario?.[key]?.fit_score ?? null };
    }));
    m.set(h.horse_number, {
      paces: { high: paceFitScore(h, "high", shape), mid: paceFitScore(h, "mid", shape), slow: paceFitScore(h, "slow", shape) },
      overall, word: overallWord(overall), sensitivity: h.sensitivity_label ?? null, bestLabel: h.best_scenario_label ?? null, grid,
    });
  }
  return m;
}

/** 積み上げバー用の区間。確率は合計 1 にそろえ、表示は 0〜100 のパーセント（整数）。 */
export function segments(probs: Partial<Record<string, number>>, order: string[]): { key: string; pct: number }[] {
  const n = normalize(probs, order);
  return order.map((k) => ({ key: k, pct: Math.round(n[k] * 100) }));
}

/** 中央ゼロの左右バーの幅（%）。score −1〜+1 を 0〜50% に。範囲外は丸める。 */
export const diverging = (score: number): { side: "plus" | "minus"; widthPct: number } => ({
  side: score >= 0 ? "plus" : "minus",
  widthPct: Math.round(Math.min(1, Math.abs(Number.isFinite(score) ? score : 0)) * 50),
});

/** 根拠の短い札。ハナ主張が 2 頭以上のときは注意書きも返す。 */
export function basisTags(b: PaceType["basis"] | undefined): { tags: string[]; warn: string | null } {
  const tags: string[] = [];
  let warn: string | null = null;
  if (b?.strong_front_runners != null) tags.push(`強い逃げ馬 ${b.strong_front_runners}頭`);
  if (b?.front_runners_claim != null) tags.push(`ハナ主張 ${b.front_runners_claim}頭`);
  if ((b?.front_runners_claim ?? 0) >= 2) warn = "ハナを主張する馬が複数いるため、ハイペースになりやすい";
  if (b?.lap_forecast_used) tags.push("ラップ予想を反映");
  return { tags, warn };
}

/** 実測の要点（説明用）。 */
export const PACE_FACTS: string[] = [
  "スロー・縦長: 逃げの3着内率は平均の1.34倍、追込は0.71倍（前が残りやすい）",
  "ハイ・凝縮: 逃げは0.64倍、追込は1.44倍（差しが決まりやすい）",
  "ミドルは隊列で向きが逆: 縦長は逃げ1.15倍、凝縮は追込1.10倍",
  "ハイでも縦長なら脚質による差は小さい",
];

export const SCENARIO_NOTE = "シナリオ別の有利さは過去の傾向から作った説明用の目安で、勝率・期待値の予測には使っていません。";
