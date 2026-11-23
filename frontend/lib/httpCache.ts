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
  if (AUTH_URL.startsWith(prefix)) dropStoredAuth();
}

export type AuthStatusJson = { logged_in?: boolean; is_admin?: boolean; is_member?: boolean; user_id?: string | null; user_name?: string | null };

const AUTH_URL = "/api/v1/auth/status";
/** メモリの保持。ページ間の移動（Nav は画面を移るたびに聞く）で取り直さない。ログイン・ログアウトでは捨てる。 */
export const AUTH_TTL_MS = 3 * 60_000;
/** 「未ログイン」の結果だけ、このタブの sessionStorage に残す（ページを開き直すたびに Cloud Run へ聞かない）。
 *  会員・ログイン済みの結果は残さない（決済から戻ったとき・セッションが切れたときに古い区分を出さないため。新しいタブは必ず聞き直す）。 */
const AUTH_STORE_KEY = `${STORE_PREFIX}auth`;
const ANON_STORE_MS = 5 * 60_000;
const ANON: AuthStatusJson = { logged_in: false, is_admin: false, is_member: false };

function dropStoredAuth(): void {
  try { ss()?.removeItem(AUTH_STORE_KEY); } catch { /* 何もしない */ }
}

/** 「未ログイン」を残した時刻からの経過が 5 分未満か。残す時刻はネットワークから取れた応答のときだけ書く（使い回しで延ばさない）。 */
function storedAnon(): boolean {
  const st = ss();
  if (!st) return false;
  try {
    const raw = st.getItem(AUTH_STORE_KEY);
    if (!raw) return false;
    const at = Number(raw);
    return Number.isFinite(at) && Date.now() - at >= 0 && Date.now() - at < ANON_STORE_MS;
  } catch { return false; }
}

function tierOf(a: AuthStatusJson): string {
  return a.logged_in ? (a.is_admin ? "admin" : a.is_member ? "member" : "user") : "anon";
}

/** ログイン状態。1 ページで Nav・フック・各画面が同時に呼んでも 1 回だけ取りに行く。 */
export function fetchAuthStatus(): Promise<AuthStatusJson> {
  const network = !hasFreshJson(AUTH_URL, AUTH_TTL_MS);
  if (network && storedAnon()) return Promise.resolve(ANON);   // メモリには入れない（残した時刻の 5 分で切れるように）
  return getJson<AuthStatusJson>(AUTH_URL, { ttlMs: AUTH_TTL_MS, credentials: "include" }).then((a) => {
    setStoredTier(tierOf(a));
    if (network) {
      if (!a.logged_in) { try { ss()?.setItem(AUTH_STORE_KEY, String(Date.now())); } catch { /* 何もしない */ } }
      else dropStoredAuth();
    }
    return a;
  }).catch((e: Error & { status?: number }) => {
    if (e.status !== undefined && e.status < 500) return ANON;
    throw e;
  });
}

/** このタブで前に分かったログイン区分（会員 API を「認証の確認と並行して先に取るか」の目安だけに使う。呼ばない判断は fetchAuthStatus の結果で行う）。 */
export function likelyMember(): boolean {
  const st = ss();
  try {
    const t = st?.getItem(TIER_KEY);
    return t === "member" || t === "admin";
  } catch { return false; }
}

/** ログインの応答に入っている認証状態（GET /api/v1/auth/status と同じ形）で、状態のキャッシュを埋める。ログイン直後の画面が状態を取り直す往復を省く。 */
export function primeAuthStatus(a: AuthStatusJson): void {
  cache.clear();                    // ログインで区分が変わるので、匿名のときに取った応答はメモリからも捨てる
  dropStoredAuth();
  setStoredTier(tierOf(a));
  cache.set(AUTH_URL, { at: Date.now(), promise: Promise.resolve(a) });
}

/** 開催日（YYYYMMDD）のレース一覧。サーバ側が重い API なので、画面間で共有する。 */
export function fetchRaceList(date: string, ttlMs = 60_000): Promise<unknown> {
  return getJson<unknown>(`/api/race-list/${date}`, { ttlMs });
}
