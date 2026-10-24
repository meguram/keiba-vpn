import type { CSSProperties } from "react";

/* 枠番の色と枠割り（netkeiba の出馬表に合わせる）。枠色の定義はここだけ。 */

export type BracketColor = { bg: string; color: string; border: string };

/** netkeiba: 1白 2黒 3赤 4青 5黄 6緑 7橙 8桃。border は暗い背景でも枠の輪郭が見えるよう全色に付ける */
export const BRACKET: Record<number, BracketColor> = {
  1: { bg: "#ffffff", color: "#111827", border: "#94a3b8" },
  2: { bg: "#111111", color: "#ffffff", border: "#6b7280" },
  3: { bg: "#e60012", color: "#ffffff", border: "#e60012" },
  4: { bg: "#0068b7", color: "#ffffff", border: "#0068b7" },
  5: { bg: "#ffd400", color: "#111827", border: "#ffd400" },
  6: { bg: "#00913a", color: "#ffffff", border: "#00913a" },
  7: { bg: "#f08300", color: "#111827", border: "#f08300" },
  8: { bg: "#f5a3c6", color: "#111827", border: "#f5a3c6" },
};

export const BRACKET_FALLBACK: BracketColor = { bg: "#243049", color: "#e5e7eb", border: "#334155" };

export function bracketColor(bn: number | null | undefined): BracketColor {
  return bn != null && BRACKET[bn] ? BRACKET[bn] : BRACKET_FALLBACK;
}

/** 枠色を背景にした丸角チップ用の style（枠・馬番の数字を表示する要素に当てる） */
export function bracketChipStyle(bn: number | null | undefined, size = 22): CSSProperties {
  const c = bracketColor(bn);
  return {
    background: c.bg, color: c.color, border: `1px solid ${c.border}`, minWidth: size, height: size, borderRadius: 5,
    display: "inline-flex", alignItems: "center", justifyContent: "center", fontWeight: 800,
    fontSize: size <= 20 ? 11 : 12, fontVariantNumeric: "tabular-nums", lineHeight: 1, padding: "0 4px", boxSizing: "border-box" as const,
  };
}

/** 枠割り（JRA）。8頭以下は 1 頭 1 枠。9 頭目以降は後ろの枠から 2 頭、17 頭目以降は後ろの枠から 3 頭。 */
export function bracketCounts(fieldSize: number): number[] {
  const n = Math.max(0, Math.floor(fieldSize));
  if (n <= 8) return Array.from({ length: n }, () => 1);
  const counts: number[] = Array.from({ length: 8 }, () => (n >= 16 ? 2 : 1));
  if (n < 16) {
    for (let i = 0; i < n - 8; i++) counts[7 - i] = 2;
  } else {
    for (let i = 0; i < n - 16; i++) counts[7 - i] = 3;
  }
  return counts;
}

/** 馬番 → 枠番。API の bracket_number があればそれを優先し、無ければ頭数から枠割りで求める */
export function bracketOf(horseNumber: number, fieldSize: number, apiBracket?: number | null): number | null {
  if (apiBracket != null && apiBracket >= 1 && apiBracket <= 8) return apiBracket;
  if (!horseNumber || horseNumber < 1 || !fieldSize || horseNumber > fieldSize) return null;
  let rest = horseNumber;
  const counts = bracketCounts(fieldSize);
  for (let i = 0; i < counts.length; i++) {
    if (rest <= counts[i]) return i + 1;
    rest -= counts[i];
  }
  return null;
}
