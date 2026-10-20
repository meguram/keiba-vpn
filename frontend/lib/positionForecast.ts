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

/* ── 追走難度（/api/race/{id}/tracking-difficulty）を展開予想に重ねるための型と純粋関数。
   位置取りの予想（過去走の通過順位）とは別の算出なので、脚質の表現がずれることがある。 */

export type HorseTd = {
  ease_pct?: number;               // 大きいほど前に付いていきやすい
  ease_label?: string;
  style?: string;                  // 追走難度側の脚質
  flow_pattern?: string;
  stamina_label?: string;
  stamina_concern?: boolean;
  last3f_sec?: number;
  last3f_delta?: number;           // 同条件の基準との差（−が速い）
  v2?: boolean;                    // 追走難度 v2（予測 API の tracking_v2）の値で上書きしている
  drain_sec?: number;              // v2: 通過ラップのあとの終いの落ち（秒、大きいほど大変）
  demand_sec?: number;             // v2: 要求（キーが無いことがある）
  capacity_sec?: number;           // v2: 供給（キーが無いことがある）
};

/** 予測 API が各馬に付ける追走難度 v2（KEIBA_TRACKING_V2=1 のときだけ。キーが無い馬・レースは従来どおり） */
export type TrackingV2 = { ease_pct: number; label?: string; drain_sec?: number; demand_sec?: number; capacity_sec?: number };

/** 追走のしやすさの色（ラベルの境目は API の `_ease_label` と同じ: 75/60/45/30）。色だけに頼らず、文字でも出す。 */
export function easeTone(pct: number | null | undefined): { color: string; word: string } {
  if (pct == null) return { color: "transparent", word: "—" };
  if (pct >= 75) return { color: "#22c55e", word: "非常に楽" };
  if (pct >= 60) return { color: "#4ade80", word: "楽" };
  if (pct >= 45) return { color: "#94a3b8", word: "普通" };
  if (pct >= 30) return { color: "#f59e0b", word: "やや困難" };
  return { color: "#ef4444", word: "困難" };
}

export type PaceFit = { score: number; label: "向く" | "ふつう" | "不向き"; reasons: string[] };

/** 展開向き／不向き。脚質×想定ペース（前に行く馬が多い＝ハイなら差し・追込、少ない＝スローなら逃げ・先行）に、
    追走のしやすさ（60 以上 +1、30 未満 −1）を足した合計が +1 以上で「向く」、−1 以下で「不向き」。 */
export function paceFit(style: StyleCode | undefined, paceLevel: PaceForecast["level"] | undefined, easePct: number | null | undefined): PaceFit | null {
  if (!style) return null;
  const reasons: string[] = [];
  let score = 0;
  const front = style === "front" || style === "stalker";
  if (paceLevel === "high") {
    score += front ? -1 : 1;
    reasons.push(front ? "ハイペースで前に行く側は苦しい" : "ハイペースで後ろから差せる");
  } else if (paceLevel === "slow") {
    score += front ? 1 : -1;
    reasons.push(front ? "スローで前が残りやすい" : "スローで後ろは届きにくい");
  }
  if (easePct != null) {
    if (easePct >= 60) { score += 1; reasons.push("追走が楽"); }
    else if (easePct < 30) { score -= 1; reasons.push("追走が困難"); }
  }
  return { score, label: score >= 1 ? "向く" : score <= -1 ? "不向き" : "ふつう", reasons };
}

/** 予測 API の馬ごとの追走難度情報を作る（position_flow と entries を馬番で合わせる）。 */
export function buildTdMap(
  flows: { horse_number?: number; style?: string; flow_pattern?: string; stamina_concern?: boolean; stamina?: { stamina_label?: string }; expected_last_3f?: { seconds?: number; delta_sec?: number } }[] | undefined,
  entries: { horse_number?: number; tracking_difficulty?: { ease_pct?: number; label?: string } }[] | undefined,
): Map<number, HorseTd> {
  const m = new Map<number, HorseTd>();
  const get = (no: number) => m.get(no) ?? (m.set(no, {}), m.get(no)!);
  for (const e of entries ?? []) {
    if (e.horse_number == null) continue;
    Object.assign(get(e.horse_number), { ease_pct: e.tracking_difficulty?.ease_pct, ease_label: e.tracking_difficulty?.label });
  }
  for (const f of flows ?? []) {
    if (f.horse_number == null) continue;
    Object.assign(get(f.horse_number), {
      style: f.style, flow_pattern: f.flow_pattern, stamina_label: f.stamina?.stamina_label, stamina_concern: f.stamina_concern,
      last3f_sec: f.expected_last_3f?.seconds, last3f_delta: f.expected_last_3f?.delta_sec,
    });
  }
  return m;
}

/** 基準との差の表示。−0.0 にならないよう丸めてから符号を付ける。 */
export function fmtDelta(d: number): string {
  const r = Math.round(d * 10) / 10;
  return r === 0 ? "±0.0" : `${r > 0 ? "+" : ""}${r.toFixed(1)}`;
}
