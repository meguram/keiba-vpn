/** レースレベルの欄の札（components/RaceResult.tsx の RaceLevelNote）。段はサーバが付けた言葉（src/pipeline/race_time/levels.py の
 *  TRACK_WORDS・MEMBER_WORDS・pace_label。1000m あたりの秒で決めた段）から決める。秒の値から前で計算し直さない（境は 1000m あたり）。
 *  テストはこのファイルだけを変換して読むので、ほかの lib を import しない。 */

/** 段: −2（速い・高い・ハイ）〜 +2（時計が掛かる・低い・スロー）。null は表に無い言葉 */
export type LevelStep = -2 | -1 | 0 | 1 | 2;
export type LevelBadge = { text: string; step: LevelStep | null; color: string; strong: boolean };

const TRACK_STEP: Record<string, LevelStep> = { "速い": -2, "やや速い": -1, "標準": 0, "やや時計が掛かる": 1, "時計が掛かる": 2 };
const MEMBER_STEP: Record<string, LevelStep> = { "高い": -2, "やや高い": -1, "標準": 0, "やや低い": 1, "低い": 2 };
const PACE_STEP: Record<string, LevelStep> = { "ハイ": -2, "平均": 0, "スロー": 2 };

/** 札の色（速い・高い・ハイの側は暖色、遅い・低い・スローの側は寒色、標準は灰。±2 は濃い色）。明暗の両方で読める固定の色 */
export const STEP_COLOR: Record<string, string> = {
  "-2": "#ef4444", "-1": "#f59e0b", "0": "#94a3b8", "1": "#38bdf8", "2": "#3b82f6",
};
const NEUTRAL = "#94a3b8";

export type LevelKind = "track" | "member" | "pace";
const TABLE: Record<LevelKind, Record<string, LevelStep>> = { track: TRACK_STEP, member: MEMBER_STEP, pace: PACE_STEP };

/** 言葉から札。無ければ null。表に無い言葉はそのまま灰の札 */
export function levelBadge(word: string | null | undefined, kind: LevelKind): LevelBadge | null {
  if (!word) return null;
  const step = TABLE[kind][word];
  if (step === undefined) return { text: word, step: null, color: NEUTRAL, strong: false };
  return { text: kind === "pace" ? `${word}ペース` : word, step, color: STEP_COLOR[String(step)], strong: Math.abs(step) === 2 && kind !== "pace" };
}

/** 秒の値（負なら速い）を「−0.8 秒」の形に。無ければ「—」 */
export function secText(v: number | null | undefined): string {
  if (v == null) return "—";
  const r = Math.round(v * 10) / 10;   // 丸めてから符号（−0.04 を「−0.0」にしない）
  return `${r > 0 ? "+" : r < 0 ? "−" : "±"}${Math.abs(r).toFixed(1)} 秒`;
}

/** ペースの補正の分（補正後 − 補正前）。どちらかが無ければ null */
export function paceAdjust(sec: number | null | undefined, raw: number | null | undefined): number | null {
  return sec == null || raw == null ? null : Math.round((sec - raw) * 10) / 10;
}
