/* 展開予想（位置取り）の型と、画面の組み立てに使う純粋関数。
   元データは過去走の通過順位（src/pipeline/inference/position_forecast.py）。実モデルの予測ではない。 */

export type StyleCode = "front" | "stalker" | "mid" | "closer";

export type PositionForecast = {
  style: string;
  style_code: StyleCode;
  early_pos_norm: number;          // 0=先頭 〜 1=最後方
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

export type ForecastHorse = { no: number; name: string; bracket: number | null; f: PositionForecast | null };

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

/** 脚質レーンごとの馬（先頭に近い順）。スマホの縦レイアウト用。 */
export function lanesOf(horses: ForecastHorse[]): { code: StyleCode; label: string; horses: ForecastHorse[] }[] {
  return STYLE_LANES.map((l) => ({
    code: l.code, label: l.label,
    horses: horses.filter((h) => h.f?.style_code === l.code).sort((a, b) => (a.f!.early_pos_norm - b.f!.early_pos_norm) || (a.no - b.no)),
  }));
}
