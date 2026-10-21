/* メンテナンス時間帯（毎日 JST 03:00–04:00）の表示まわりの純粋関数。判定の正は API（/api/maintenance/status）で、
   ブラウザの時計は使わない（受信時刻からの経過で残りを数える）。 */

export type MaintenanceStatus = {
  enabled: boolean;
  active: boolean;
  phase: "normal" | "notice" | "active";
  now: string;
  timezone: string;
  message: string;
  source: string | null;
  window_start: string | null;
  window_end: string | null;
  starts_in_sec: number | null;
  remaining_sec: number | null;
  ends_at: string | null;
  ends_at_label: string | null;
  retry_after: number;
};

export type Stamped = { status: MaintenanceStatus; receivedAt: number };

export const STATUS_URL = "/api/maintenance/status";
export const BLOCK_EVENT = "keiba:maintenance";
export const MAINTENANCE_PATH = "/maintenance";

export const NORMAL_STATUS: MaintenanceStatus = {
  enabled: false, active: false, phase: "normal", now: "", timezone: "Asia/Tokyo", message: "", source: null,
  window_start: null, window_end: null, starts_in_sec: null, remaining_sec: null, ends_at: null, ends_at_label: null, retry_after: 0,
};

/** API の応答（または 503 の本文）から状態を作る。形が違うときは通常扱い。 */
export function parseStatus(raw: unknown): MaintenanceStatus {
  if (!raw || typeof raw !== "object") return NORMAL_STATUS;
  const r = raw as Record<string, unknown>;
  // 503 の本文（{maintenance:true, message, ends_at, remaining_sec}）も受ける
  if (r.maintenance === true && r.phase === undefined) {
    const rem = typeof r.remaining_sec === "number" ? r.remaining_sec : null;
    return {
      ...NORMAL_STATUS, enabled: true, active: true, phase: "active",
      message: typeof r.message === "string" ? r.message : "",
      ends_at: typeof r.ends_at === "string" ? r.ends_at : null,
      ends_at_label: typeof r.ends_at_label === "string" ? r.ends_at_label : null,
      remaining_sec: rem, retry_after: rem ?? 60,
    };
  }
  const phase = r.phase === "active" || r.phase === "notice" ? r.phase : "normal";
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const str = (v: unknown) => (typeof v === "string" ? v : null);
  return {
    enabled: r.enabled === true,
    active: r.active === true,
    phase,
    now: str(r.now) ?? "",
    timezone: str(r.timezone) ?? "Asia/Tokyo",
    message: str(r.message) ?? "",
    source: str(r.source),
    window_start: str(r.window_start),
    window_end: str(r.window_end),
    starts_in_sec: num(r.starts_in_sec),
    remaining_sec: num(r.remaining_sec),
    ends_at: str(r.ends_at),
    ends_at_label: str(r.ends_at_label),
    retry_after: num(r.retry_after) ?? 0,
  };
}

/** 受信時刻 receivedAt（ms）から nowMs までの経過を引いた残り秒。サーバが返した秒数が基準なので、端末の時計のずれに影響されない。 */
export function secondsLeft(field: number | null | undefined, receivedAt: number, nowMs: number): number | null {
  if (field == null) return null;
  return Math.max(0, Math.ceil(field - (nowMs - receivedAt) / 1000));
}

/** 現在の表示フェーズ。受信後に境界を跨いだ分（事前告知→開始、終了）をローカルの経過で先取りする。 */
export function livePhase(s: Stamped, nowMs: number): "normal" | "notice" | "active" {
  const { status, receivedAt } = s;
  if (status.phase === "active") return secondsLeft(status.remaining_sec, receivedAt, nowMs) === 0 ? "normal" : "active";
  if (status.phase === "notice") return secondsLeft(status.starts_in_sec, receivedAt, nowMs) === 0 ? "active" : "notice";
  return "normal";
}

/** 残り秒を「HH:MM:SS」（1 時間未満は「MM:SS」）に整形。 */
export function formatCountdown(sec: number | null): string {
  if (sec == null) return "--:--";
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  const p = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${p(h)}:${p(m)}:${p(x)}` : `${p(m)}:${p(x)}`;
}

/** 「あと約 N 分」（切り上げ。1 分未満は「まもなく」）。 */
export function approxMinutes(sec: number | null): string {
  if (sec == null) return "";
  if (sec <= 60) return "まもなく";
  return `あと約${Math.ceil(sec / 60)}分`;
}

/** ISO 文字列（JST オフセット付き）から JST の「HH:MM」。タイムゾーンの変換はしない（API が JST で返す）。 */
export function hhmm(iso: string | null | undefined): string {
  const m = /T(\d{2}):(\d{2})/.exec(iso ?? "");
  return m ? `${m[1]}:${m[2]}` : "";
}

/** 画面に出す「開始〜終了」（時刻は JST。「日本時間」の注記は呼び出し側で 1 回だけ付ける）。 */
export function windowLabel(s: MaintenanceStatus): string {
  const a = hhmm(s.window_start), b = s.ends_at_label || hhmm(s.window_end);
  return a && b ? `${a}〜${b}` : "";
}

/** fetch の応答がメンテナンスの 503 か（X-Keiba-Maintenance ヘッダー）。 */
export function isMaintenanceResponse(res: { status: number; headers: { get(name: string): string | null } }): boolean {
  return res.status === 503 && res.headers.get("x-keiba-maintenance") === "1";
}

/** プレビュー（dev 限定）。?maintenance=notice|active|ended[&remaining=秒] から疑似状態を作る。 */
export function previewStatus(kind: string | null, remaining?: string | null): MaintenanceStatus | null {
  if (kind !== "notice" && kind !== "active" && kind !== "ended") return null;
  const rem = Math.max(1, Math.min(7200, Number(remaining) || (kind === "active" ? 2400 : 900)));
  const base = { ...NORMAL_STATUS, enabled: true, timezone: "Asia/Tokyo", window_start: "2026-10-22T03:00:00+09:00", window_end: "2026-10-22T04:00:00+09:00", ends_at: "2026-10-22T04:00:00+09:00", ends_at_label: "04:00" };
  if (kind === "active") return { ...base, active: true, phase: "active", message: "ただいまメンテナンス中です。終了までしばらくお待ちください。", source: "daily", remaining_sec: rem, retry_after: rem };
  if (kind === "notice") return { ...base, phase: "notice", message: "まもなくメンテナンスを行います。この間はサービスをご利用いただけません。", source: "daily", starts_in_sec: rem };
  return { ...base, phase: "normal" };
}

/** 復帰先（?next=）の検証。同一オリジンの相対パスだけ許す（オープンリダイレクト防止）。 */
export function safeNext(v: string | null | undefined): string {
  if (!v || !v.startsWith("/") || v.startsWith("//") || v.includes("\\") || v.startsWith(MAINTENANCE_PATH)) return "/";
  return v;
}
