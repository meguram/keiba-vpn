/** レース結果の答え合わせ（展開予想 vs 実際）。純粋な計算だけ。 */

export type PaceLevel = "high" | "mid" | "slow";

export const PACE_JA: Record<PaceLevel, string> = { high: "ハイペース", mid: "ミドルペース", slow: "スローペース" };

/** 実際のペース: 前半3F − 後半3F。−1.0 秒以下（前が速い）= ハイ、+1.0 秒以上（前が遅い）= スロー、その間 = ミドル（目安） */
export function actualPace(first3f: number | null | undefined, second3f: number | null | undefined): { level: PaceLevel; diff: number } | null {
  if (typeof first3f !== "number" || typeof second3f !== "number" || !(first3f > 0) || !(second3f > 0)) return null;
  const diff = Math.round((first3f - second3f) * 10) / 10;
  return { level: diff <= -1.0 ? "high" : diff >= 1.0 ? "slow" : "mid", diff };
}

const RANK: Record<PaceLevel, number> = { slow: 0, mid: 1, high: 2 };

/** 予想と実際のペース: 同じ=的中、1段違い=惜しい、2段違い=外れ */
export function paceVerdict(pred: PaceLevel, actual: PaceLevel): "hit" | "near" | "miss" {
  const d = Math.abs(RANK[pred] - RANK[actual]);
  return d === 0 ? "hit" : d === 1 ? "near" : "miss";
}

/** 通過順位の文字列（"9-9-14-17"）から最初のコーナーの順位。読めなければ null */
export function firstCornerPos(passing: unknown): number | null {
  const m = String(passing ?? "").match(/\d+/);
  const n = m ? Number(m[0]) : NaN;
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** 位置取りの答え合わせ: 想定の番手と実際の最初のコーナー順位の差（正=想定より後ろ）。±2 番手以内=近い、±4 以内=やや違う、それ以上=外れ */
export function positionVerdict(est: number | null | undefined, actual: number | null | undefined): { diff: number; verdict: "hit" | "near" | "miss" } | null {
  if (typeof est !== "number" || typeof actual !== "number") return null;
  const diff = actual - est;
  const a = Math.abs(diff);
  return { diff, verdict: a <= 2 ? "hit" : a <= 4 ? "near" : "miss" };
}

export type PositionSummary = { n: number; hit: number; avgAbsDiff: number | null; frontPred: number | null; frontActual: number | null };

/** 馬ごとの結果から全体の集計。frontPred/frontActual は「想定が1番手の馬」「最初のコーナーが1番手の馬」の馬番 */
export function summarizePositions(rows: { no: number; est: number | null | undefined; actual: number | null }[]): PositionSummary {
  let n = 0, hit = 0, sum = 0;
  for (const r of rows) {
    const v = positionVerdict(r.est, r.actual);
    if (!v) continue;
    n += 1; sum += Math.abs(v.diff); hit += v.verdict === "hit" ? 1 : 0;
  }
  const frontPred = rows.filter((r) => r.est === 1).map((r) => r.no)[0] ?? null;
  const frontActual = rows.filter((r) => r.actual === 1).map((r) => r.no)[0] ?? null;
  return { n, hit, avgAbsDiff: n ? Math.round((sum / n) * 10) / 10 : null, frontPred, frontActual };
}
