/** GET の JSON を取るときの共通処理: 同じ URL の同時リクエストは 1 本にまとめ、短時間は結果を使い回す。
 *  ページ間の移動（クライアント遷移）でもメモリは残るので、一覧→詳細で同じ API を取り直さない。 */
type Entry<T = unknown> = { at: number; promise: Promise<T> };

const cache = new Map<string, Entry>();

/** AI 予測 API の画面側の保持時間。一覧での先読みと、詳細ページ・レース切替で同じ値を使い、先読みした分をそのまま使えるようにする。 */
export const PRED_TTL_MS = 45_000;

/** persist: 取得できたら sessionStorage にも残す（次に開いたとき、readStored で前回の内容を先に出して裏で更新するため） */
export type JsonOpts = { ttlMs?: number; credentials?: RequestCredentials; force?: boolean; persist?: boolean };

/* ── 前回の表示を先に出すための保存（このタブの間だけ。ログイン区分が変わったら全部捨てる） ── */
const STORE_PREFIX = "jc:";
const TIER_KEY = "jc:tier";

function ss(): Storage | null {
  try { return typeof window !== "undefined" ? window.sessionStorage : null; } catch { return null; }
}

export function clearStored(): void {
  const st = ss();
  if (!st) return;
  for (const k of Object.keys(st)) if (k.startsWith(STORE_PREFIX)) st.removeItem(k);
}

/** ログイン区分（匿名/会員/開発者）を記録する。前と違えば、前の区分で保存した内容は捨てる（会員向けの項目が残らないように）。 */
export function setStoredTier(tier: string): void {
  const st = ss();
  if (!st) return;
  if (st.getItem(TIER_KEY) !== tier) { clearStored(); st.setItem(TIER_KEY, tier); }
}

export function readStored<T>(url: string, maxAgeMs = 30 * 60_000): T | null {
  const st = ss();
  if (!st || !st.getItem(TIER_KEY)) return null;
  try {
    const raw = st.getItem(`${STORE_PREFIX}d:${url}`);
    if (!raw) return null;
    const e = JSON.parse(raw) as { at: number; data: T };
    return Date.now() - e.at < maxAgeMs ? e.data : null;
  } catch { return null; }
}

function writeStored(url: string, data: unknown): void {
  const st = ss();
  if (!st || !st.getItem(TIER_KEY)) return;       // 区分が分からないうちは残さない
  try {
    st.setItem(`${STORE_PREFIX}d:${url}`, JSON.stringify({ at: Date.now(), data }));
  } catch {
    clearStored();                                  // 容量超過: 全部捨てて、区分だけ戻す
    try { st.setItem(TIER_KEY, "?"); } catch { /* 何もしない */ }
  }
}

/** 失敗（HTTP エラー・通信エラー）は覚えない。404 など「結果が無い」ことを返したいときは okOnly=false で呼ぶ側が扱う。 */
export function getJson<T>(url: string, opts: JsonOpts = {}): Promise<T> {
  const ttl = opts.ttlMs ?? 15_000;
  const hit = cache.get(url) as Entry<T> | undefined;
  if (hit && !opts.force && Date.now() - hit.at < ttl) return hit.promise;
  const promise = fetch(url, { credentials: opts.credentials }).then(async (r) => {
    if (!r.ok) {
      const err = new Error(`${url}: ${r.status}`) as Error & { status?: number };
      err.status = r.status;
      throw err;
    }
    const data = (await r.json()) as T;
    if (opts.persist) writeStored(url, data);
    return data;
  });
  const entry: Entry<T> = { at: Date.now(), promise };
  cache.set(url, entry as Entry);
  promise.catch(() => { if (cache.get(url) === (entry as Entry)) cache.delete(url); });
  return promise;
}

/** 操作の前に温めておく（マウスが乗った／触れ始めた時点）。失敗しても何もしない。GET だけに使う。 */
export function prefetchJson(url: string, opts: JsonOpts = {}): void {
  if (typeof window === "undefined") return;
  getJson(url, opts).catch(() => { /* 先読みの失敗は無視 */ });
}

/** 取得済み（または取得中）で新鮮なら true。待たずに表示できるかの判定用。 */
export function hasFreshJson(url: string, ttlMs = 15_000): boolean {
  const hit = cache.get(url);
  return !!hit && Date.now() - hit.at < ttlMs;
}

export function invalidateJson(prefix: string): void {
  for (const k of [...cache.keys()]) if (k.startsWith(prefix)) cache.delete(k);
}

export type AuthStatusJson = { logged_in?: boolean; is_admin?: boolean; is_member?: boolean; user_id?: string | null; user_name?: string | null };

/** ログイン状態。1 ページで Nav・フック・各画面が同時に呼んでも 1 回だけ取りに行く。 */
export function fetchAuthStatus(): Promise<AuthStatusJson> {
  return getJson<AuthStatusJson>("/api/v1/auth/status", { ttlMs: 20_000, credentials: "include" }).then((a) => {
    setStoredTier(a.logged_in ? (a.is_admin ? "admin" : a.is_member ? "member" : "user") : "anon");
    return a;
  }).catch((e: Error & { status?: number }) => {
    if (e.status !== undefined && e.status < 500) return { logged_in: false, is_admin: false, is_member: false };
    throw e;
  });
}

/** 開催日（YYYYMMDD）のレース一覧。サーバ側が重い API なので、画面間で共有する。 */
export function fetchRaceList(date: string, ttlMs = 60_000): Promise<unknown> {
  return getJson<unknown>(`/api/race-list/${date}`, { ttlMs });
}
