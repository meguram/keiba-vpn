import { NextRequest, NextResponse } from "next/server";

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

function isDevOnly(pathname: string): boolean {
  return DEV_ONLY_PAGES.some((p) => pathname === p || pathname.startsWith(p + "/"));
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (!isDevOnly(pathname)) return NextResponse.next();

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
