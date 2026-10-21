/* 出馬表の公示の段階（特別登録／出走馬決定／出馬表確定／馬体重発表済み）の表示まわりの純粋関数。
   段階は収集側（src/scraper/entries_sync.py）が判定し、API の entries_status / レース一覧の entries_phase に載る。 */

export type EntriesPhase = "none" | "registered" | "entries" | "final" | "weighed";

export type EntriesStatus = {
  phase?: EntriesPhase | string | null;
  label?: string | null;
  note?: string | null;
  updated_at?: string | null;
  checked_at?: string | null;
  n_entries?: number;
  n_numbered?: number;
  n_jockey?: number;
  n_weight?: number;
  race_date?: string | null;
  is_future?: boolean;
  recent_changes?: { type: string; horse_name?: string; from?: string; to?: string }[];
};

export type PhaseMeta = { label: string; color: string; bg: string; border: string; provisional: boolean };

const META: Record<string, PhaseMeta> = {
  none:       { label: "出馬表待ち",       color: "#94a3b8", bg: "rgba(148,163,184,0.12)", border: "rgba(148,163,184,0.35)", provisional: true },
  registered: { label: "特別登録（暫定）", color: "#f59e0b", bg: "rgba(245,158,11,0.12)",  border: "rgba(245,158,11,0.4)",  provisional: true },
  entries:    { label: "出走馬決定",       color: "#60a5fa", bg: "rgba(96,165,250,0.12)",  border: "rgba(96,165,250,0.4)",  provisional: true },
  final:      { label: "出馬表確定",       color: "#22c55e", bg: "rgba(34,197,94,0.12)",   border: "rgba(34,197,94,0.4)",   provisional: false },
  weighed:    { label: "馬体重発表済み",   color: "#2dd4bf", bg: "rgba(45,212,191,0.12)",  border: "rgba(45,212,191,0.4)",  provisional: false },
};

export function phaseMeta(phase?: string | null): PhaseMeta {
  return META[phase ?? "none"] ?? META.none;
}

/** 確定前（頭数・騎手・枠順が変わりうる）の段階か。 */
export function isProvisional(phase?: string | null): boolean {
  return phaseMeta(phase).provisional;
}

const JST_OFFSET_MIN = 9 * 60;

/** ISO（+09:00 付き）の更新時刻を「10/23 16:05」の形にする。読めなければ空文字。 */
export function fmtUpdated(iso?: string | null): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  const j = new Date(t + JST_OFFSET_MIN * 60_000);   // UTC の getter で JST の値を読む
  const p = (n: number) => String(n).padStart(2, "0");
  return `${j.getUTCMonth() + 1}/${j.getUTCDate()} ${p(j.getUTCHours())}:${p(j.getUTCMinutes())}`;
}

/** 直近の差分の短い要約（取消・騎手変更の件数）。無ければ空文字。 */
export function changesSummary(changes?: EntriesStatus["recent_changes"]): string {
  if (!changes?.length) return "";
  const removed = changes.filter((c) => c.type === "removed").length;
  const jockey = changes.filter((c) => c.type === "jockey_change").length;
  const parts = [removed ? `取消 ${removed}頭` : "", jockey ? `騎手変更 ${jockey}件` : ""].filter(Boolean);
  return parts.join("・");
}

type RaceLike = { entries_phase?: string | null };

/** その日のレース一覧から、段階ごとの件数と、一番手前（遅れている）段階を返す。 */
export function summarizeDay(races: RaceLike[]): { counts: Record<string, number>; total: number; withPhase: number; slowest: string | null } {
  const order = ["none", "registered", "entries", "final", "weighed"];
  const counts: Record<string, number> = {};
  let withPhase = 0;
  for (const r of races) {
    const k = r.entries_phase ?? "none";
    counts[k] = (counts[k] ?? 0) + 1;
    if (r.entries_phase) withPhase += 1;
  }
  const slowest = order.find((k) => (counts[k] ?? 0) > 0) ?? null;
  return { counts, total: races.length, withPhase, slowest };
}
