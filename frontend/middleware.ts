import { NextRequest, NextResponse } from "next/server";
import { pathEnabled } from "@/lib/features";

// 開発者限定ページ。src/api/auth.py の DEV_ONLY_PAGES と同じ一覧（変更時は両方を揃える）。
const DEV_ONLY_PAGES = [
  "/monitor",
  "/data-viewer",
  "/queue-status",
  "/server-logs",
  "/scrape-upcoming",
  "/betting",
  "/track-speed/dev",
];

const DEV_COOKIE = "keiba_dev_session";

// dev の自動ログイン（NEXT_PUBLIC_DEV_AUTO_LOGIN=1）。この PC からの直アクセス（Host が localhost 系）のときだけ。
// tcpexposer など公開ドメイン経由は Host が公開ドメインになるので対象外。バックエンドも同じ条件で判定する（src/api/auth.py）。
const DEV_AUTO_LOGIN = process.env.NEXT_PUBLIC_DEV_AUTO_LOGIN === "1";
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

function isLocalDirectRequest(request: NextRequest): boolean {
  if (!DEV_AUTO_LOGIN) return false;
  const hostname = (h: string | null) => (h ?? "").split(",")[0].trim().toLowerCase().replace(/:\d+$/, "");
  const forwarded = request.headers.get("x-forwarded-host");
  if (forwarded && !LOCAL_HOSTS.has(hostname(forwarded))) return false;
  if (!LOCAL_HOSTS.has(hostname(request.headers.get("host")))) return false;
  const xff = request.headers.get("x-forwarded-for");
  // IPv4 射影（::ffff:127.0.0.1）も含めてループバックだけを許可
  const isLoopback = (x: string) => {
    const s = x.trim().replace(/^\[|\]$/g, "").replace(/^::ffff:/i, "");
    return s === "::1" || /^127\./.test(s);
  };
  if (xff && !xff.split(",").every(isLoopback)) return false;
  return true;
}

function isDevOnly(pathname: string): boolean {
  return DEV_ONLY_PAGES.some((p) => pathname === p || pathname.startsWith(p + "/"));
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 無効化中の機能（lib/features.ts）への直リンクはホームへ戻す
  if (!pathEnabled(pathname)) {
    const url = request.nextUrl.clone();
    url.pathname = "/";
    url.search = "";
    return NextResponse.redirect(url);
  }

  if (!isDevOnly(pathname)) return NextResponse.next();

  if (isLocalDirectRequest(request)) return NextResponse.next();

  // Edge ではクッキーの有無だけを見る。署名検証と実データの認可は API 側（auth.py）が行う。
  if (request.cookies.get(DEV_COOKIE)) return NextResponse.next();

  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = "";
  url.searchParams.set("next", pathname);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
