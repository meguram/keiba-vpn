/** GET の JSON を取るときの共通処理: 同じ URL の同時リクエストは 1 本にまとめ、短時間は結果を使い回す。
 *  ページ間の移動（クライアント遷移）でもメモリは残るので、一覧→詳細で同じ API を取り直さない。 */
type Entry<T = unknown> = { at: number; promise: Promise<T> };

const cache = new Map<string, Entry>();

export type JsonOpts = { ttlMs?: number; credentials?: RequestCredentials; force?: boolean };

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
    return (await r.json()) as T;
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

export function invalidateJson(prefix: string): void {
  for (const k of [...cache.keys()]) if (k.startsWith(prefix)) cache.delete(k);
}

export type AuthStatusJson = { logged_in?: boolean; is_admin?: boolean; is_member?: boolean };

/** ログイン状態。1 ページで Nav・フック・各画面が同時に呼んでも 1 回だけ取りに行く。 */
export function fetchAuthStatus(): Promise<AuthStatusJson> {
  return getJson<AuthStatusJson>("/api/v1/auth/status", { ttlMs: 20_000, credentials: "include" }).catch((e: Error & { status?: number }) => {
    if (e.status !== undefined && e.status < 500) return { logged_in: false, is_admin: false, is_member: false };
    throw e;
  });
}

/** 開催日（YYYYMMDD）のレース一覧。サーバ側が重い API なので、画面間で共有する。 */
export function fetchRaceList(date: string, ttlMs = 60_000): Promise<unknown> {
  return getJson<unknown>(`/api/race-list/${date}`, { ttlMs });
}
