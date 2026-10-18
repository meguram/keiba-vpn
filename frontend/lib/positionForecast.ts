/* 展開予想（位置取り）の型と、画面の組み立てに使う純粋関数。
   元データは過去走の通過順位（src/pipeline/inference/position_forecast.py）。実モデルの予測ではない。 */

export type StyleCode = "front" | "stalker" | "mid" | "closer";

export type PositionForecast = {
  style: string;                   // 枠順補正後の想定の脚質（補正が無ければ過去走そのまま）
  style_code: StyleCode;
  style_raw_code?: StyleCode;      // 補正前（過去走そのまま）の脚質。ペースの数え方はこちら
  early_pos_norm: number;          // 0=先頭 〜 1=最後方（枠順補正後）
  early_pos_raw?: number;          // 枠順補正前
  gate_rel?: number | null;        // 0=最内 〜 1=最外
  gate_shift?: number;             // 補正量（正=後ろへ）
  early_pos_est?: number;          // 想定の位置取り（今回の出走馬の中で何番手か）
  avg_first_corner: number;        // 過去走の最初のコーナーの平均順位
  runs_used: number;
  confidence: "high" | "mid" | "low";
  last_passing: string;
};

export type PaceForecast = {
  level: "high" | "mid" | "slow" | "unknown";
  label: string;
  note: string;
  front_runner_count: number;
  stalker_count: number;
  early_pressure: number;
  field_size: number;
  known_horses: number;
};

export type GateBias = {
  level: "inner" | "outer" | "flat" | "low" | "na";
  cls: string;
  note: string;
  n_races: number;
  k: number;
  applied: boolean;
  key?: string | null;
  field_last_number?: number;
};

export type ForecastHorse = { no: number; name: string; bracket: number | null; f: PositionForecast | null };

export const GATE_TONE: Record<GateBias["level"], { fg: string; bg: string; bd: string; label: string }> = {
  inner: { fg: "#86efac", bg: "rgba(34,197,94,0.12)", bd: "rgba(34,197,94,0.5)", label: "内枠有利" },
  outer: { fg: "#fdba74", bg: "rgba(249,115,22,0.12)", bd: "rgba(249,115,22,0.5)", label: "外枠有利" },
  flat: { fg: "#cbd5e1", bg: "rgba(148,163,184,0.10)", bd: "rgba(148,163,184,0.4)", label: "枠の差なし" },
  low: { fg: "#cbd5e1", bg: "transparent", bd: "rgba(148,163,184,0.4)", label: "傾向は参考程度" },
  na: { fg: "#94a3b8", bg: "transparent", bd: "rgba(148,163,184,0.3)", label: "枠順補正なし" },
};

/** 馬の枠の位置の言い方（0=最内〜1=最外）。 */
export function gateWord(rel: number | null | undefined): string {
  if (rel == null) return "—";
  return rel <= 0.25 ? "内寄り" : rel >= 0.75 ? "外寄り" : "中ほど";
}

export const STYLE_LANES: { code: StyleCode; label: string; from: number; to: number }[] = [
  { code: "front", label: "逃げ", from: 0, to: 0.15 },
  { code: "stalker", label: "先行", from: 0.15, to: 0.35 },
  { code: "mid", label: "差し", from: 0.35, to: 0.65 },
  { code: "closer", label: "追込", from: 0.65, to: 1 },
];

export const PACE_TONE: Record<PaceForecast["level"], { fg: string; bg: string; bd: string; icon: string }> = {
  high: { fg: "#fca5a5", bg: "rgba(239,68,68,0.12)", bd: "rgba(239,68,68,0.5)", icon: "▲" },
  mid: { fg: "#fcd34d", bg: "rgba(251,191,36,0.10)", bd: "rgba(251,191,36,0.45)", icon: "■" },
  slow: { fg: "#93c5fd", bg: "rgba(96,165,250,0.10)", bd: "rgba(96,165,250,0.45)", icon: "▼" },
  unknown: { fg: "#94a3b8", bg: "rgba(148,163,184,0.08)", bd: "rgba(148,163,184,0.3)", icon: "?" },
};

export const CONF_LABEL: Record<PositionForecast["confidence"], string> = { high: "高", mid: "中", low: "低（1走のみ）" };

/** 馬ごとに、トラック図（横軸 0=先頭〜1=後方）での置き場所を決める。近い馬は上下に積む。 */
export function packTrack(horses: ForecastHorse[], width: number, chip = 30, gap = 3): { h: ForecastHorse; x: number; row: number }[] {
  const known = horses.filter((h) => h.f).sort((a, b) => (a.f!.early_pos_norm - b.f!.early_pos_norm) || (a.no - b.no));
  const ends: number[] = [];
  const usable = Math.max(width - chip, 1);
  return known.map((h) => {
    const x = chip / 2 + h.f!.early_pos_norm * usable;
    let row = ends.findIndex((e) => x - chip / 2 >= e + gap);
    if (row < 0) row = ends.length;
    ends[row] = x + chip / 2;
    return { h, x, row };
  });
}

/** 脚質レーンごとの馬。スマホの縦レイアウト用。枠順が分かるなら内枠→外枠、無ければ先頭に近い順。 */
export function lanesOf(horses: ForecastHorse[], byGate = false): { code: StyleCode; label: string; horses: ForecastHorse[] }[] {
  const g = (h: ForecastHorse) => h.f?.gate_rel ?? 0.5;
  return STYLE_LANES.map((l) => ({
    code: l.code, label: l.label,
    horses: horses.filter((h) => h.f?.style_code === l.code).sort((a, b) =>
      (byGate ? (g(a) - g(b)) || (a.no - b.no) : 0) || (a.f!.early_pos_norm - b.f!.early_pos_norm) || (a.no - b.no)),
  }));
}

/** 内外の軸つきのトラック図。横=前後の位置取り、縦=上が内ラチ・下が外。内枠から順に置き、重なるときは外側（下）へずらす。 */
export function packTrackGate(horses: ForecastHorse[], width: number, chip = 30, gap = 3, depth = 150): { placed: { h: ForecastHorse; x: number; y: number }[]; height: number } {
  const known = horses.filter((h) => h.f).sort((a, b) =>
    ((a.f!.gate_rel ?? 0.5) - (b.f!.gate_rel ?? 0.5)) || (a.f!.early_pos_norm - b.f!.early_pos_norm) || (a.no - b.no));
  const usable = Math.max(width - chip, 1);
  const placed: { h: ForecastHorse; x: number; y: number }[] = [];
  for (const h of known) {
    const x = chip / 2 + h.f!.early_pos_norm * usable;
    let y = (h.f!.gate_rel ?? 0.5) * depth;
    // 浮動小数の誤差で『ちょうど接している』丸を衝突と誤判定して無限ループしないよう、判定に余裕（EPS）を持たせ、反復にも上限を置く
    const EPS = 1e-6;
    const clash = (yy: number) => placed.find((q) => Math.abs(q.x - x) < chip + gap - EPS && Math.abs(q.y - yy) < chip + gap - EPS);
    for (let c = clash(y), guard = 0; c && guard < 200; c = clash(y), guard++) y = c.y + chip + gap;
    placed.push({ h, x, y });
  }
  const height = placed.reduce((m, p) => Math.max(m, p.y + chip), depth * 0 + chip);
  return { placed, height };
}
