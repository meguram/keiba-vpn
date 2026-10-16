/* 馬・騎手・調教師パネルの見せ方（色・判定・形）。画面の部品（EntryCardPanel）から切り出した純粋関数。
   「良い/悪い」は色だけでなく、必ず文字（◎/△/▽ と語）を併記するための判定もここに置く。 */
import type { CardStat } from "@/lib/entryCard";

/** 枠の色（JRA: 1白 2黒 3赤 4青 5黄 6緑 7橙 8桃） */
export const BRACKET: Record<number, { bg: string; color: string }> = {
  1: { bg: "#f8fafc", color: "#0f172a" }, 2: { bg: "#111827", color: "#fff" }, 3: { bg: "#ef4444", color: "#fff" }, 4: { bg: "#3b82f6", color: "#fff" },
  5: { bg: "#facc15", color: "#0f172a" }, 6: { bg: "#22c55e", color: "#0f172a" }, 7: { bg: "#fb923c", color: "#0f172a" }, 8: { bg: "#f9a8d4", color: "#0f172a" },
};

/** 全体の目安（JRA 平均）。複勝率 = 3/14 ≒ 21%、勝率 = 1/14 ≒ 7% */
export const BASE_TOP3 = 21;
export const BASE_WIN = 7;

export type Tone = "good" | "ok" | "warn" | "muted";
export const TONE: Record<Tone, { fg: string; bg: string; bd: string }> = {
  good: { fg: "#4ade80", bg: "rgba(74,222,128,0.10)", bd: "rgba(74,222,128,0.45)" },
  ok: { fg: "#93c5fd", bg: "rgba(96,165,250,0.09)", bd: "rgba(96,165,250,0.38)" },
  warn: { fg: "#fb923c", bg: "rgba(251,146,60,0.11)", bd: "rgba(251,146,60,0.5)" },
  muted: { fg: "#94a3b8", bg: "rgba(148,163,184,0.08)", bd: "rgba(148,163,184,0.3)" },
};

export type Tag = { text: string; tone: Tone };

/** 少数（母数が小さい）のしきい値。馬の同条件は 5 走未満、騎手・調教師の条件別は 30 走未満 */
export const FEW_HORSE = 5;
export const FEW_PERSON = 30;

/** 複勝率の評価（馬の同条件・騎手/調教師の条件別）。母数が少ないときは評価せず『少数』 */
export function placeTag(rate: number | null | undefined, starts: number | null | undefined, few: number): Tag {
  if (rate == null || !starts) return { text: "データなし", tone: "muted" };
  if (starts < few) return { text: "少数", tone: "muted" };
  if (rate >= BASE_TOP3 + 15) return { text: "◎得意", tone: "good" };
  if (rate >= BASE_TOP3 - 5) return { text: "△標準", tone: "ok" };
  return { text: "▽苦手", tone: "warn" };
}

/** 騎手・調教師の調子: 直近 90 日の複勝率を通算と比べる（30 走以上のときだけ判定） */
export function formTag(recent: CardStat | undefined, career: CardStat | null | undefined): Tag & { delta: number | null } {
  if (!recent || recent.top3_rate == null || (recent.starts ?? 0) < 30) return { text: "少数", tone: "muted", delta: null };
  const base = career?.top3_rate ?? BASE_TOP3;
  const delta = recent.top3_rate - base;
  if (delta >= 3) return { text: "◎好調", tone: "good", delta };
  if (delta <= -3) return { text: "▽不調", tone: "warn", delta };
  return { text: "△標準", tone: "ok", delta };
}

/** 前走からの間隔の評価 */
export function layoffTag(days: number | null | undefined): Tag {
  if (days == null) return { text: "不明", tone: "muted" };
  if (days < 14) return { text: "連戦", tone: "ok" };
  if (days <= 70) return { text: "間隔標準", tone: "ok" };
  if (days <= 120) return { text: "やや間隔あり", tone: "warn" };
  return { text: "休み明け", tone: "warn" };
}

/** 距離の増減 → 『短縮/延長/同距離』と矢印 */
export function distanceChange(d: number | null | undefined): { text: string; arrow: string } {
  if (d == null) return { text: "距離の変化 不明", arrow: "" };
  if (d === 0) return { text: "同距離", arrow: "＝" };
  return d < 0 ? { text: `距離 ${d}m 短縮`, arrow: "↓" } : { text: `距離 +${d}m 延長`, arrow: "↑" };
}

/** 追い切り評価 S/A/B/C の色と語 */
export function trainingGrade(e: string | null | undefined): { letter: string; tone: Tone; word: string; bg: string; fg: string } {
  const g = (e ?? "").trim().toUpperCase().slice(0, 1);
  if (g === "S") return { letter: "S", tone: "good", word: "◎絶好", bg: "#fbbf24", fg: "#1f2937" };
  if (g === "A") return { letter: "A", tone: "good", word: "◎良好", bg: "#4ade80", fg: "#052e16" };
  if (g === "B") return { letter: "B", tone: "ok", word: "△標準", bg: "#60a5fa", fg: "#0b1b3a" };
  if (g === "C") return { letter: "C", tone: "warn", word: "▽平凡", bg: "#94a3b8", fg: "#0f172a" };
  return { letter: "-", tone: "muted", word: "評価なし", bg: "transparent", fg: "#94a3b8" };
}

/** 着順バッジ: 1着=金、2着=銀、3着=銅、4-5着=灰青、6着以下=薄灰、中止・取消などは別表示 */
export function finishBadge(f: number | string | null | undefined): { text: string; bg: string; fg: string; bd: string; kind: "medal" | "mid" | "low" | "odd"; label: string } {
  const n = Number(f);
  if (f != null && f !== "" && Number.isFinite(n)) {
    if (n === 1) return { text: "1", bg: "#fbbf24", fg: "#1f2937", bd: "#fde68a", kind: "medal", label: "1着" };
    if (n === 2) return { text: "2", bg: "#cbd5e1", fg: "#1f2937", bd: "#f1f5f9", kind: "medal", label: "2着" };
    if (n === 3) return { text: "3", bg: "#d97706", fg: "#fff", bd: "#fbbf24", kind: "medal", label: "3着" };
    if (n <= 5) return { text: String(n), bg: "rgba(96,165,250,0.18)", fg: "#bfdbfe", bd: "rgba(96,165,250,0.45)", kind: "mid", label: `${n}着` };
    return { text: String(n), bg: "rgba(148,163,184,0.12)", fg: "#94a3b8", bd: "rgba(148,163,184,0.3)", kind: "low", label: `${n}着` };
  }
  const t = String(f ?? "").trim() || "-";
  const word = t.includes("中") ? "競走中止" : t.includes("取") ? "出走取消" : t.includes("除") ? "競走除外" : t.includes("失") ? "失格" : t.includes("降") ? "降着" : t;
  return { text: t.slice(0, 2), bg: "transparent", fg: "#fb923c", bd: "rgba(251,146,60,0.6)", kind: "odd", label: word };
}

/** 着順の推移バーの高さ（%）。1着が最も高い */
export function finishBarHeight(f: number | string | null | undefined): number {
  const n = Number(f);
  if (!Number.isFinite(n)) return 10;
  if (n <= 5) return [100, 84, 70, 56, 46][n - 1] ?? 46;
  return Math.max(14, 40 - (n - 5) * 3);
}

/** 人気と着順の差: 人気より上位でゴール = 好走（▲）、人気より下位 = 凡走（▼） */
export function popGap(pop: number | null | undefined, fin: number | string | null | undefined): { arrow: string; text: string; tone: Tone } | null {
  const f = Number(fin);
  if (!pop || !Number.isFinite(f)) return null;
  const d = pop - f;
  if (d >= 2) return { arrow: "▲", text: `人気より${d}着上`, tone: "good" };
  if (d <= -2) return { arrow: "▼", text: `人気より${-d}着下`, tone: "warn" };
  return { arrow: "＝", text: "人気どおり", tone: "muted" };
}

/** 追い切りのラップ文字列「-57.9(16.2)41.7(15.4)26.3(13.6)12.7(12.7)」→ [{累計, 1F}]（4F→1F の順） */
export function parseLap(s: string | null | undefined): { cum: number; lap: number }[] {
  const out: { cum: number; lap: number }[] = [];
  for (const m of (s ?? "").matchAll(/(-?\d+(?:\.\d+)?)\((\d+(?:\.\d+)?)\)/g)) out.push({ cum: Math.abs(Number(m[1])), lap: Number(m[2]) });
  return out;
}

/** 『第N回』『(GI)』などを落とした短いレース名 */
export function shortRaceName(n: string | null | undefined): string {
  return (n ?? "").replace(/^第\d+回/, "").trim();
}

/** 通過順「1-1-2-2」→ 数のリスト */
export function passingList(p: string | null | undefined): number[] {
  return (p ?? "").split("-").map((x) => Number(x)).filter((x) => Number.isFinite(x) && x > 0);
}

/** 『東京優駿 GI』『プリンシパ L』→ レース名とクラス（G1/G2/G3/L/OP）。末尾にクラスが付かなければ cls は空 */
export function splitRaceName(n: string | null | undefined, grade?: string | null): { name: string; cls: string } {
  const raw = (n ?? "").trim();
  const m = raw.match(/^(.*?)\s*[(（]?\s*(GIII|GII|GI|G[123]|JpnI{1,3}|L|OP)\s*[)）]?$/);
  const map: Record<string, string> = { GI: "G1", GII: "G2", GIII: "G3" };
  if (m && m[1]) return { name: m[1].trim(), cls: map[m[2]] ?? m[2] };
  return { name: raw, cls: (grade ?? "").replace(".", "") };
}

export const CLASS_STYLE: Record<string, { bg: string; fg: string }> = {
  G1: { bg: "#fbbf24", fg: "#1f2937" }, G2: { bg: "#f87171", fg: "#1f2937" }, G3: { bg: "#4ade80", fg: "#052e16" },
  L: { bg: "rgba(167,139,250,0.22)", fg: "#ddd6fe" }, OP: { bg: "rgba(167,139,250,0.22)", fg: "#ddd6fe" },
};

/** 芝=緑 / ダート=茶 / 障害=灰青 のコース色 */
export function surfaceStyle(s: string | null | undefined): { text: string; bg: string; fg: string } {
  const c = (s ?? "").slice(0, 1);
  if (c === "芝") return { text: "芝", bg: "#14532d", fg: "#bbf7d0" };
  if (c === "ダ") return { text: "ダ", bg: "#78350f", fg: "#fde68a" };
  if (c === "障") return { text: "障", bg: "#334155", fg: "#e2e8f0" };
  return { text: c, bg: "#334155", fg: "#e2e8f0" };
}

/** 脚質のめやす: 最初の通過順位 / 頭数 */
export function styleOf(passing: string | null | undefined, field: number | null | undefined): string {
  const p = passingList(passing);
  if (!p.length || !field) return "";
  if (p[0] === 1) return "逃げ";
  const r = p[0] / field;
  return r <= 0.3 ? "先行" : r <= 0.65 ? "差し" : "追込";
}
