/** めぐAI印の表示メタと並び順（◎ → ○ → ▲ → ★ → △ → 消）。内部名は保存データ互換のまま。 */
export const MARK_META: Record<string, { sym: string; color: string }> = {
  honmei:   { sym: "◎", color: "#ef4444" },
  pair:     { sym: "○", color: "#3b82f6" },
  anchor:   { sym: "△", color: "#22c55e" },
  show_val: { sym: "▲", color: "#f59e0b" },
  star:     { sym: "★", color: "#a78bfa" },
  kesi:     { sym: "消", color: "#cbd5e1" },
  none:     { sym: "-", color: "var(--text-dim)" },
};

export const MARK_ORDER: Record<string, number> = { honmei: 1, pair: 2, show_val: 3, star: 4, anchor: 5, kesi: 6 };
