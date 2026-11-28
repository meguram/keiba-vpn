/** トラックバイアスの欄の表示の判定（components/RaceResult.tsx の TrackBiasNote・結果の表の位置の列）。値はホストで事前に計算した物を読むだけ。 */

export type BiasTone = "front" | "back" | "inner" | "outer" | "neutral";
export type BiasBadge = { text: string; tone: BiasTone; strong: boolean };

/** 札の色（前・内は暖色、差し・外は寒色、中立は灰）。CSS の変数ではなく固定の色（明暗の両方で読める濃さ） */
export const TONE_COLOR: Record<BiasTone, string> = {
  front: "#f87171", back: "#60a5fa", inner: "#fb923c", outer: "#22d3ee", neutral: "#94a3b8",
};

/** 前後の札。名札の文字（例「強い前残り」「差し有利」「前後は中立」）から。無ければ null */
export function frontBadge(label: string | null | undefined): BiasBadge | null {
  if (!label) return null;
  const strong = label.startsWith("強い");
  if (label.includes("前残り")) return { text: label, tone: "front", strong };
  if (label.includes("差し")) return { text: label, tone: "back", strong };
  if (label.includes("中立")) return { text: "前後：中立", tone: "neutral", strong: false };
  return null;
}

/** 内外の札。名札の文字（例「内有利」「強い外有利」「内外は中立」）から。無ければ null */
export function innerBadge(label: string | null | undefined): BiasBadge | null {
  if (!label) return null;
  const strong = label.startsWith("強い");
  if (label.includes("内有利")) return { text: label, tone: "inner", strong };
  if (label.includes("外有利")) return { text: label, tone: "outer", strong };
  if (label.includes("中立")) return { text: "内外：中立", tone: "neutral", strong: false };
  return null;
}

/** 1 行目の大きな文字（例「福島芝」）。名札の言葉は札（labelBadges）で出し、同じ言葉を 2 回並べない。面は「芝」「ダート」 */
export function headline(venue: string | null | undefined, surf: string | null | undefined): string {
  const s = surf === "ダ" ? "ダート" : (surf ?? "");
  return `${venue ?? ""}${s}`;
}

/** 1 行目の札。前後・内外のうち中立でない方だけを出し、両方中立なら「中立」の札 1 つ。判定なし（どちらも札にならない）なら名札の文字をそのまま灰の札に */
export function labelBadges(front: string | null | undefined, inner: string | null | undefined, label?: string | null): BiasBadge[] {
  const bs = [innerBadge(inner), frontBadge(front)].filter((b): b is BiasBadge => b != null);
  const strong = bs.filter((b) => b.tone !== "neutral");
  if (strong.length) return strong;
  if (bs.length) return [{ text: "中立", tone: "neutral", strong: false }];
  return label ? [{ text: label, tone: "neutral", strong: false }] : [];
}

/** 通過順（例「6-6-4-2」）の最初と最後の番手。数字が無ければ null */
export function corners(passing: string | null | undefined): { early: number; corner4: number } | null {
  const xs = String(passing ?? "").split("-").map((v) => v.trim()).filter((v) => /^\d+$/.test(v)).map(Number);
  return xs.length ? { early: xs[0], corner4: xs[xs.length - 1] } : null;
}

/** 結果の表の位置の列（例「序盤 6/15 → 4 角 2/15」）。序盤は通過順の最初（名札の判定に使う位置）、「4 角」は通過順の最後の角
 *  （角が 2 つのコースでも「4 角」と書く。サーバの corner4 と同じ約束）。通過順が無ければ「—」 */
export function positionText(passing: string | null | undefined, field: number | null | undefined): string {
  const c = corners(passing);
  if (!c || !field) return "—";
  return `序盤 ${c.early}/${field} → 4 角 ${c.corner4}/${field}`;
}

/** 狭い画面の位置の列（例「6→2/15」） */
export function positionShort(passing: string | null | undefined, field: number | null | undefined): string {
  const c = corners(passing);
  return !c || !field ? "—" : `${c.early}→${c.corner4}/${field}`;
}

/** 位置の列の並べ替えの値（序盤の番手。無ければ最後） */
export function earlySortKey(passing: string | null | undefined): number {
  return corners(passing)?.early ?? 999;
}
