/** ログインの試行が多すぎて止められたとき（429）の文言。retry_after（秒）があれば分に直して添える。 */
export function lockedMessage(retryAfterSec: number | null | undefined): string {
  const base = "試行が多すぎます。しばらくしてからお試しください";
  if (!retryAfterSec || retryAfterSec <= 0) return base;
  return `${base}（約${Math.max(1, Math.ceil(retryAfterSec / 60))}分）`;
}

/** 共通ログインの先月のパスワードを打ったとき（401・reason=password_rotated）の文言。パスワードは毎月 1 日に変わる。 */
export const ROTATED_MESSAGE = "パスワードは毎月 1 日に新しくなります。今月のパスワードでログインしてください";
