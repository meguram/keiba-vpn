/** GET の JSON を取るときの共通処理: 同じ URL の同時リクエストは 1 本にまとめ、短時間は結果を使い回す。
 *  ページ間の移動（クライアント遷移）でもメモリは残るので、一覧→詳細で同じ API を取り直さない。 */
type Entry<T = unknown> = { at: number; promise: Promise<T>; persisted?: boolean; req?: ReqAt };

const cache = new Map<string, Entry>();

/** AI 予測 API の画面側の保持時間。一覧での先読みと、詳細ページ・レース切替で同じ値を使い、先読みした分をそのまま使えるようにする。 */
export const PRED_TTL_MS = 45_000;

/** persist: 取得できたら sessionStorage にも残す（次に開いたとき、readStored で前回の内容を先に出して裏で更新するため） */
export type JsonOpts = { ttlMs?: number; credentials?: RequestCredentials; force?: boolean; persist?: boolean };

/* ── 前回の表示を先に出すための保存（このタブの間だけ。ログイン区分が変わったら全部捨てる） ── */
const STORE_PREFIX = "jc:";
const TIER_KEY = "jc:tier";
/** 1 件の上限（文字数。UTF-16 で約 800KB）。レース本体は匿名で最大 約 270KB（バイト。2026-11-18 に 4 レースを実測）なので収まり、まれな巨大な応答だけ外す */
const STORE_MAX_CHARS = 400_000;

function ss(): Storage | null {
  try { return typeof window !== "undefined" ? window.sessionStorage : null; } catch { return null; }
}

function removeAllStored(st: Storage): void {
  for (const k of Object.keys(st)) if (k.startsWith(STORE_PREFIX)) st.removeItem(k);
}

/** ログアウトなどで全部捨てる。取得中の応答も、返ってきたときに残さない（authEpoch が変わるため） */
export function clearStored(): void {
  authEpoch++;
  const st = ss();
  if (st) removeAllStored(st);
}

/** ログイン区分（匿名/会員/開発者）を記録する。前と違えば、前の区分で保存した内容は捨てる（会員向けの項目が残らないように）。 */
export function setStoredTier(tier: string): void {
  const st = ss();
  if (!st) return;
  const prev = st.getItem(TIER_KEY);
  if (prev !== tier) {
    if (prev !== null) authEpoch++;            // 区分が変わった（前の区分で出した要求の応答は残さない）。最初の確認（記録なし→区分）は変わり目にしない
    removeAllStored(st);
    st.setItem(TIER_KEY, tier);
  }
  tierConfirmed = true;
}

/** 区分の変わり目（ログイン・ログアウト・区分の変化）で増える。要求を出したときと返ったときで違えば、その応答は残さない */
let authEpoch = 0;

/** このページ（JS の実行）の中で、認証の状態を取って区分を確かめたか。再読み込みの直後は、前の区分の記録が今の認証と合うか分からないので、
 *  確かめるまで readStored は何も返さない（他のタブでのログアウト・セッション切れの後に、会員向けの内容を先に出さないため）。 */
let tierConfirmed = false;

function storedTier(): string | null {
  return ss()?.getItem(TIER_KEY) ?? null;
}

export function readStored<T>(url: string, maxAgeMs = 30 * 60_000): T | null {
  const st = ss();
  if (!st || !st.getItem(TIER_KEY) || !tierConfirmed) return null;
  try {
    const raw = st.getItem(`${STORE_PREFIX}d:${url}`);
    if (!raw) return null;
    const e = JSON.parse(raw) as { at: number; data: T };
    return Date.now() - e.at < maxAgeMs ? e.data : null;
  } catch { return null; }
}

/** 古いものから消す（あふれたとき）。全部は消さない（ほかの画面の前回の内容を残すため） */
function evictOldest(st: Storage, keep: number): void {
  const items: { k: string; at: number }[] = [];
  for (const k of Object.keys(st)) {
    if (!k.startsWith(`${STORE_PREFIX}d:`)) continue;
    const m = /^\{"at":(\d+)/.exec((st.getItem(k) ?? "").slice(0, 32));   // 先頭の at だけ読む（全件を JSON.parse しない）
    const at = m ? Number(m[1]) : 0;                                         // 読めないものは先に消す
    items.push({ k, at });
  }
  items.sort((a, b) => a.at - b.at);
  for (const { k } of items.slice(0, Math.max(0, items.length - keep))) st.removeItem(k);
}

/** at: 要求を出したときの区分と変わり目の番号。返るまでに区分が変わったら（ログイン・ログアウト）、前の区分の応答は残さない */
type ReqAt = { tier: string | null; epoch: number };
function reqAt(): ReqAt {
  return { tier: storedTier(), epoch: authEpoch };
}

function writeStored(url: string, data: unknown, at: ReqAt): void {
  const st = ss();
  if (!st || !st.getItem(TIER_KEY) || !tierConfirmed) return;   // 区分が分からない・このページで確かめていないうちは残さない
  if (at.epoch !== authEpoch) return;
  if (at.tier !== null && st.getItem(TIER_KEY) !== at.tier) return;
  const raw = JSON.stringify({ at: Date.now(), data });
  if (raw.length > STORE_MAX_CHARS) return;        // 1 件が大きすぎるものは残さない
  const key = `${STORE_PREFIX}d:${url}`;
  try {
    st.setItem(key, raw);
  } catch {
    try {                                          // 容量超過: 古い半分を消して 1 回だけ入れ直す
      const n = Object.keys(st).filter((k) => k.startsWith(`${STORE_PREFIX}d:`)).length;
      evictOldest(st, Math.floor(n / 2));
      st.setItem(key, raw);
    } catch { /* 入らなければ残さない */ }
  }
}

/** 失敗（HTTP エラー・通信エラー）は覚えない。404 など「結果が無い」ことを返したいときは okOnly=false で呼ぶ側が扱う。 */
export function getJson<T>(url: string, opts: JsonOpts = {}): Promise<T> {
  const ttl = opts.ttlMs ?? 15_000;
  const hit = cache.get(url) as Entry<T> | undefined;
  if (hit && !opts.force && Date.now() - hit.at < ttl) {
    // 先読み（persist なし）に相乗りしたときも、persist を求められたら結果を残す（1 回だけ）
    if (opts.persist && !hit.persisted) {
      hit.persisted = true;
      const at = hit.req ?? reqAt();               // 区分は、その要求を出したときの値で比べる（今の値ではない）
      hit.promise.then((data) => writeStored(url, data, at)).catch(() => { /* 失敗は残さない */ });
    }
    return hit.promise;
  }
  const at = reqAt();
  const promise = fetch(url, { credentials: opts.credentials }).then(async (r) => {
    if (!r.ok) {
      const err = new Error(`${url}: ${r.status}`) as Error & { status?: number };
      err.status = r.status;
      throw err;
    }
    const data = (await r.json()) as T;
    if (opts.persist) writeStored(url, data, at);
    return data;
  });
  const entry: Entry<T> = { at: Date.now(), promise, persisted: !!opts.persist, req: at };
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

/** ログインの応答に入っている認証状態（GET /api/v1/auth/status と同じ形）で、状態のキャッシュを埋める。ログイン直後の画面が状態を取り直す往復を省く。 */
export function primeAuthStatus(a: AuthStatusJson): void {
  const url = "/api/v1/auth/status";
  cache.clear();                    // ログインで区分が変わるので、匿名のときに取った応答はメモリからも捨てる
  authEpoch++;                      // 取得中の匿名の応答が後から返っても、会員の区分で残さない

  setStoredTier(a.logged_in ? (a.is_admin ? "admin" : a.is_member ? "member" : "user") : "anon");
  cache.set(url, { at: Date.now(), promise: Promise.resolve(a) });
}

/** 開催日（YYYYMMDD）のレース一覧。サーバ側が重い API なので、画面間で共有する。
 *  persist: このタブの sessionStorage にも残す（/races が次に開いたとき前回の一覧を先に出すため。latency_plan_20261118.md の L1）。
 *  同じ URL の取得は最初の呼び出しの persist で決まるので、先読みと本読みで揃える。 */
export function fetchRaceList(date: string, ttlMs = 60_000, persist = false): Promise<unknown> {
  return getJson<unknown>(`/api/race-list/${date}`, { ttlMs, persist });
}

export function raceListUrl(date: string): string {
  return `/api/race-list/${date}`;
}
