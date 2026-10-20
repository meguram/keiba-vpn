/* ラップ予想（予測 API の最上位 lap_forecast）の型と、グラフ用の純粋関数。単位はすべて秒。 */

export type LapScenario = {
  name: string;
  prob: number;
  laps: (number | null)[];
  sd?: (number | null)[];
  first3f?: number | null;
  last3f?: number | null;
};

export type LapForecast = {
  version?: string;
  race_id?: string;
  date?: string;
  distance?: number;
  n_laps: number;
  lap_meters: number[];
  model?: string;
  pace_label?: string;
  pace_prob?: number | null;
  scenarios?: LapScenario[];
  expected_laps: (number | null)[];
  band80_lo?: (number | null)[];
  band80_hi?: (number | null)[];
  expected_first3f?: number | null;
  expected_last3f?: number | null;
  par?: (number | null)[];
};

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** 描画できる最低限の中身があるか（区間数と期待ラップが揃い、1 つ以上の数値がある） */
export function isUsableLapForecast(lf: LapForecast | null | undefined): lf is LapForecast {
  if (!lf || !isNum(lf.n_laps) || lf.n_laps < 1) return false;
  if (!Array.isArray(lf.expected_laps) || lf.expected_laps.length < 1) return false;
  return lf.expected_laps.some(isNum);
}

/** 値が入っている区間だけをつなぎ、null の所で線を切る SVG パス（孤立した 1 点は点として見えるよう同じ座標へ線を引く） */
export function buildPath(values: (number | null | undefined)[], x: (i: number) => number, y: (v: number) => number): string {
  const parts: string[] = [];
  let run: [number, number][] = [];
  const flush = () => {
    if (run.length === 1) parts.push(`M${run[0][0].toFixed(1)} ${run[0][1].toFixed(1)}L${run[0][0].toFixed(1)} ${run[0][1].toFixed(1)}`);
    else if (run.length > 1) parts.push(run.map(([px, py], k) => `${k === 0 ? "M" : "L"}${px.toFixed(1)} ${py.toFixed(1)}`).join(""));
    run = [];
  };
  values.forEach((v, i) => {
    if (isNum(v)) run.push([x(i), y(v)]);
    else flush();
  });
  flush();
  return parts.join("");
}

/** 下限・上限の両方が入っている連続区間ごとの塗り（2 点以上の区間だけ）。 */
export function buildBand(
  lo: (number | null | undefined)[], hi: (number | null | undefined)[], x: (i: number) => number, y: (v: number) => number,
): string {
  const n = Math.min(lo.length, hi.length);
  const parts: string[] = [];
  let run: number[] = [];
  const flush = () => {
    if (run.length >= 2) {
      const up = run.map((i) => `${x(i).toFixed(1)} ${y(hi[i] as number).toFixed(1)}`);
      const down = [...run].reverse().map((i) => `${x(i).toFixed(1)} ${y(lo[i] as number).toFixed(1)}`);
      parts.push(`M${up.join("L")}L${down.join("L")}Z`);
    }
    run = [];
  };
  for (let i = 0; i < n; i++) {
    if (isNum(lo[i]) && isNum(hi[i])) run.push(i);
    else flush();
  }
  flush();
  return parts.join("");
}

/** 縦軸の範囲。系列にある数値だけから最小・最大を取り、上下に余白を付ける。数値が無ければ null。 */
export function yRange(series: (number | null | undefined)[][], padRatio = 0.12): { min: number; max: number } | null {
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s) if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  const span = Math.max(hi - lo, 0.6);
  const mid = (hi + lo) / 2;
  return { min: mid - span / 2 - span * padRatio, max: mid + span / 2 + span * padRatio };
}

/** 軸の目盛り（0.1・0.2・0.5・1・2 秒刻みのうち、本数が maxTicks 以内で最も細かいもの） */
export function niceTicks(min: number, max: number, maxTicks = 5): number[] {
  for (const step of [0.1, 0.2, 0.5, 1, 2, 5]) {
    const first = Math.ceil(min / step - 1e-9) * step;
    const ticks: number[] = [];
    for (let t = first; t <= max + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100);
    if (ticks.length <= maxTicks && ticks.length >= 2) return ticks;
  }
  return [Math.round(min * 10) / 10, Math.round(max * 10) / 10];
}

/** 区間ごとの「スタートからの累積距離（m）」（区間の終わり）。最初の区間は 200m に満たないことがある。 */
export function cumulativeMeters(lapMeters: number[]): number[] {
  let acc = 0;
  return lapMeters.map((m) => (acc += isNum(m) ? m : 0));
}

/** 横軸ラベルを出す区間の番号。最後の区間は必ず出し、本数が多ければ間引く。 */
export function tickIndices(n: number, maxTicks: number): number[] {
  if (n <= 0) return [];
  if (n <= maxTicks) return Array.from({ length: n }, (_, i) => i);
  const step = Math.ceil((n - 1) / (maxTicks - 1));
  const out: number[] = [];
  for (let i = n - 1; i >= 0; i -= step) out.unshift(i);
  return out;
}

const sec1 = (v: number | null | undefined) => (isNum(v) ? v.toFixed(1) : "—");

/** 折りたたみの見出しに出す 1 行 */
export function lapSummaryText(lf: LapForecast): string {
  const parts: string[] = [];
  if (isNum(lf.expected_first3f)) parts.push(`前半3F ${sec1(lf.expected_first3f)}秒`);
  if (isNum(lf.expected_last3f)) parts.push(`上がり3F ${sec1(lf.expected_last3f)}秒`);
  if (lf.pace_label) parts.push(`傾向 ${lf.pace_label}${isNum(lf.pace_prob) ? ` ${Math.round(lf.pace_prob * 100)}%` : ""}`);
  return parts.join(" ・ ");
}

/** シナリオの線の色（ハイ=赤・ミドル=黄・スロー=青）。名前に含まれる語で決め、無ければ順番で割り当てる。 */
export function scenarioColor(name: string, index: number): string {
  if (name.includes("ハイ")) return "#f87171";
  if (name.includes("ミドル")) return "#fbbf24";
  if (name.includes("スロー")) return "#60a5fa";
  return ["#f87171", "#fbbf24", "#60a5fa"][index % 3];
}
