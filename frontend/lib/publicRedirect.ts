/** 同じオリジンの中の転送先。request.nextUrl はプロキシの後ろだと内側のホスト名（localhost:3000）になり、
 *  匿名で /column を開くと https://localhost:3000/ に飛ばされていた（2026-11-25）。nginx が上書きして渡す Host・X-Forwarded-Proto から作る。
 *  - 相対の Location は Next.js の middleware の後処理（new NextURL）で例外になるので、絶対 URL にする。
 *  - X-Forwarded-Host は nginx の location / が上書きしない（利用者が偽装できる）ので見ない。
 *  - Host も許可したホスト（megu-keiba.com とその下・Cloud Run の run.app）だけ採る。それ以外・不正な値は元のオリジン。 */
const ALLOWED_HOST = /^(?:[a-z0-9-]+\.)*(?:megu-keiba\.com|run\.app)$/;

export function publicRedirectUrl(headers: Headers, fallbackOrigin: string, pathWithQuery: string): URL {
  const first = (v: string | null) => (v ?? "").split(",")[0].trim().toLowerCase();
  const host = first(headers.get("host"));
  const proto = first(headers.get("x-forwarded-proto"));
  if (ALLOWED_HOST.test(host) && (proto === "https" || proto === "http")) {
    try {
      return new URL(pathWithQuery, `${proto}://${host}`);
    } catch {
      /* 元のオリジンに戻す */
    }
  }
  return new URL(pathWithQuery, fallbackOrigin);
}
