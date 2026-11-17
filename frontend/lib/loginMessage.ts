/** ログインの試行が多すぎて止められたとき（429）の文言。retry_after（秒）があれば分に直して添える。 */
export function lockedMessage(retryAfterSec: number | null | undefined): string {
  const base = "試行が多すぎます。しばらくしてからお試しください";
  if (!retryAfterSec || retryAfterSec <= 0) return base;
  return `${base}（約${Math.max(1, Math.ceil(retryAfterSec / 60))}分）`;
}
