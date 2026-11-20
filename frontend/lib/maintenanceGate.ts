// middleware の保守の確認（/api/maintenance/status）を、画面の要求に待たせないための入れ物。
// docs/operations/latency_plan_20261118.md の L2a。
//
// - 期限内の値はそのまま返す（今までどおり 15 秒保持）。
// - 期限切れでも「保守ではない」値で、期限から STALE_GRACE_MS 以内なら、その値ですぐ答え、取り直しは裏で 1 本だけ走らせる（待たない）。
//   それより古い値（夜に要求が途切れたプロセスなど）は待つ。保守の開始の見落としを「期限 + 猶予」の内に収めるため。
// - 期限切れの値が「保守中」のとき・まだ値が無いとき（プロセスの起動直後）は、今までどおり取り直しを待つ。
//   保守の終わりを見落として保守画面を出し続けないため。
// - 取得に失敗したとき（429・5xx・打ち切り・オブジェクトでない応答）は、前の値を FAIL_TTL_MS だけ使い続ける（終わりの時刻が近ければそこまで）。
//   上限: 保守中で終わりの時刻が分かる値は、その時刻まで延ばし、過ぎたら「保守ではない」に倒す（出続けない）。
//   それ以外（保守ではない値・終わりの時刻が分からない保守中の値）は、最後に成功した取得から FAIL_KEEP_MAX_MS まで。
// - 失敗が続く間の「保守ではない」値は、失敗のたびに数え直すので、猶予を超えても待たずに答え続ける（以前の「失敗なら保守ではない」と同じ扱い。
//   Cloud Run が冷えて確認に失敗し続ける間、画面を待たせないため）。
//   以前は失敗で即「保守ではない」にしていたので、cache-warm と同じ流量制限の枠で 429 になると保守中でも通っていた。
// - 取得そのものは FETCH_CAP_MS で打ち切る（呼び出し側の打ち切りに頼らず、取り直しが残り続けないように）。
// 取り直しの回数は今と同じ（期限 1 回につき 1 本まで）なので、Cloud Run への要求は増えない。

export type MaintenanceStatus = { active?: boolean; remaining_sec?: number | null; retry_after?: number };
export type MaintenanceState = { at: number; ttl: number; active: boolean; retry: number; okAt: number; endsAt: number | null };

export const NORMAL_TTL_MS = 15_000;
export const FAIL_TTL_MS = 5_000;
export const STALE_GRACE_MS = 45_000;
export const FAIL_KEEP_MAX_MS = 60_000;
export const FETCH_CAP_MS = 3_000;

export function stateFromStatus(j: MaintenanceStatus, at: number): MaintenanceState {
  if (j === null || typeof j !== "object") throw new Error("bad status");   // 失敗の道（前の値を延ばす）へ
  const active = j.active === true;
  const rem = typeof j.remaining_sec === "number" ? j.remaining_sec : 15;
  return {
    at,
    ttl: active ? Math.max(1000, Math.min(NORMAL_TTL_MS, rem * 1000)) : NORMAL_TTL_MS,
    active,
    retry: j.retry_after ?? 60,
    okAt: at,
    endsAt: active && typeof j.remaining_sec === "number" ? at + j.remaining_sec * 1000 : null,
  };
}

/** 失敗のときの値。前の値を延ばせるのは上限の内だけ。 */
export function stateOnFailure(prev: MaintenanceState | null, at: number): MaintenanceState {
  const off: MaintenanceState = { at, ttl: FAIL_TTL_MS, active: false, retry: 0, okAt: prev?.okAt ?? 0, endsAt: null };
  if (!prev) return off;
  if (prev.active && prev.endsAt !== null) {
    if (at >= prev.endsAt) return off;
    return { ...prev, at, ttl: Math.max(1000, Math.min(FAIL_TTL_MS, prev.endsAt - at)) };
  }
  if (at - prev.okAt > FAIL_KEEP_MAX_MS) return off;
  return { ...prev, at, ttl: FAIL_TTL_MS };
}

export type GateRead = { state: MaintenanceState | null; pending: Promise<MaintenanceState> | null; mustWait: boolean };

function capped<T>(p: Promise<T>, ms: number): Promise<T> {
  let t: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => { t = setTimeout(() => reject(new Error("timeout")), ms); });
  return Promise.race([p, timeout]).finally(() => clearTimeout(t));
}

export function createMaintenanceGate(fetchStatus: () => Promise<MaintenanceStatus>, now: () => number = Date.now, capMs: number = FETCH_CAP_MS) {
  let cache: MaintenanceState | null = null;
  let inflight: Promise<MaintenanceState> | null = null;

  function refresh(): Promise<MaintenanceState> {
    if (inflight) return inflight;
    const started = now();
    let p: Promise<MaintenanceStatus>;
    try { p = fetchStatus(); } catch (e) { p = Promise.reject(e); }
    inflight = capped(p, capMs)
      .then((j) => stateFromStatus(j, started))
      .catch(() => stateOnFailure(cache, started))
      .then((st) => {
        cache = st;
        inflight = null;
        return st;
      });
    return inflight;
  }

  /** 今の値と、待つべきか。mustWait=false のときは state をすぐ使い、pending は裏で終わらせる（waitUntil に渡す）。 */
  function read(): GateRead {
    const t = now();
    if (cache && t - cache.at < cache.ttl) return { state: cache, pending: null, mustWait: false };
    const pending = refresh();
    if (cache && !cache.active && t - cache.at < cache.ttl + STALE_GRACE_MS) return { state: cache, pending, mustWait: false };
    return { state: cache, pending, mustWait: true };
  }

  return { read, refresh };
}
